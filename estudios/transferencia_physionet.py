"""Transferencia desde PhysioNet (EEGMMIDB): cuantos ensayos de calibracion de MI ahorra un decoder
pre-entrenado con otras personas (4 de octubre).

Datos: EEG Motor Movement/Imagery Database (Schalk et al. 2004; PhysioNet), via mne.datasets.eegbci.
Corridas 4, 8 y 12 (puno izquierdo o derecho IMAGINADO): "mano derecha imaginada" (T2) contra
"reposo" (T0), con los 8 canales del Unicorn (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8) remuestreados de 160
a 250 Hz y la ventana de 2 s del lazo (config.VENTANA_MI, banda de config.BANDA_MI). La corrida 1
(un minuto de reposo con los ojos abiertos) hace de "revision de senal": EEG sin etiquetas que el
orquestador ya tiene antes de calibrar.

Para cada persona (dejando-una-fuera): se pre-entrena con todas las demas y se calibra con los
primeros n ensayos de sus corridas 4 y 8; se prueba SIEMPRE en su corrida 12 (otra corrida, minutos
despues). Tres decoders, todos con covarianzas -> recentrado riemanniano -> espacio tangente ->
regresion logistica, que es hardware.DecoderIM:

  cero          el de hoy: solo los n ensayos de la persona
  transferido   el pre-entrenado, sin tocar; la persona solo aporta su centro (recentrado sin
                etiquetas: con n = 0, el de su minuto de reposo; con n > 0, el de sus n ensayos)
  afinado       pre-entrenado + los n ensayos de la persona, que pesan PESO_PROPIO veces mas

El recentrado es lo que alinea a las personas: cada una queda centrada en la identidad (Zanini et
al. 2018), y por eso un clasificador de otras personas se puede aplicar tal cual.

Son personas reales, pero NO son el piloto, ni el Unicorn (electrodos de gel, 64 canales, otro
amplificador), ni la tarea exacta (ahi el reposo es entre ensayos): dice si la transferencia ayuda
en esos datos, no cuanto ayudara el dia de la demo.

Uso: python estudios/transferencia_physionet.py [personas]      descarga (si falta) y mide
     python estudios/transferencia_physionet.py modelo          guarda modelos/decoder_preentrenado.pkl
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
import numpy as np

import config
import hardware as hw

CARPETA = config.RESULTADOS / 'physionet'
CACHE = CARPETA / 'epocas'
FIGURA = config.RAIZ / 'docs' / 'figuras' / 'transferencia_physionet.png'
CORRIDAS_MI, CORRIDA_REPOSO, CORRIDA_PRUEBA = (4, 8, 12), 1, 12
FS = config.FLUJOS['EEG'][2]
INICIO_S = 1.0                    # la ventana de 2 s empieza 1 s despues de la senal (pasada la transicion)
ENES = (0, 4, 8, 12, 16, 20, 24, 28)
PESO_PROPIO = 20.0                # cada ensayo propio pesa como 20 de otras personas
EXCLUIDAS = (88, 92, 100)         # grabadas a 128 Hz o con tiempos distintos (conocido en EEGMMIDB)


def cargar(persona):
    """Ventanas de MI ya filtradas de una persona: X (n, 8, 500), y (1 = mano derecha imaginada,
    0 = reposo), corrida de cada ventana, y las ventanas sin etiqueta de su minuto de reposo."""
    ruta = CACHE / f'S{persona:03d}.pkl'
    if ruta.exists():
        return pickle.loads(ruta.read_bytes())
    import mne
    from mne.datasets import eegbci
    archivos = eegbci.load_data(persona, [CORRIDA_REPOSO, *CORRIDAS_MI], path=str(CARPETA), update_path=False, verbose='ERROR')
    n = int(config.VENTANA_MI * FS)

    def crudo(archivo):
        raw = mne.io.read_raw_edf(archivo, preload=True, verbose='ERROR')
        eegbci.standardize(raw)
        raw.pick(config.CANALES_EEG).reorder_channels(config.CANALES_EEG)
        return raw.resample(FS, verbose='ERROR')

    def ventana(x, i0):
        """Como BackendReal._ventana_mi: se filtra un tramo 1 s mas largo y se queda con los ultimos 2 s."""
        tramo = x[:, max(0, i0 - FS):i0 + n]
        return hw.filtrar(tramo, config.BANDA_MI, FS)[:, -n:] if tramo.shape[1] >= n + FS // 2 else None

    X, y, corrida = [], [], []
    for c, archivo in zip(CORRIDAS_MI, archivos[1:]):
        raw = crudo(archivo)
        x = raw.get_data() * 1e6                                   # microvoltios, como el contrato
        for a in raw.annotations:
            if a['description'] in ('T0', 'T2'):
                v = ventana(x, int((a['onset'] + INICIO_S) * FS))
                if v is not None and v.shape[1] == n:
                    X.append(v); y.append(int(a['description'] == 'T2')); corrida.append(c)
    x = crudo(archivos[0]).get_data() * 1e6
    reposo = [v for v in (ventana(x, i) for i in range(FS, x.shape[1] - n, n)) if v is not None]
    d = {'X': np.array(X), 'y': np.array(y), 'corrida': np.array(corrida), 'reposo': np.array(reposo)}
    CACHE.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(pickle.dumps(d))
    return d


def tangente(X, M=None):
    """Covarianzas OAS recentradas con M (por defecto, su propia media riemanniana) y proyectadas al
    espacio tangente en la identidad. Devuelve (rasgos, M)."""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import tangent_space
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.mean import mean_riemann
    C = Covariances('oas').fit_transform(X)
    M = mean_riemann(C) if M is None else M
    Mi = invsqrtm(M)
    return tangent_space(np.array([Mi @ c @ Mi for c in C]), np.eye(C.shape[1])), M


def centro(X):
    return tangente(X)[1]


def balancear(d, rng):
    """Tantos reposos como manos derechas en cada corrida, en orden cronologico."""
    idx = []
    for c in CORRIDAS_MI:
        uno = np.flatnonzero((d['corrida'] == c) & (d['y'] == 1))
        cero = np.flatnonzero((d['corrida'] == c) & (d['y'] == 0))
        idx += list(uno) + list(rng.choice(cero, min(len(cero), len(uno)), replace=False))
    return np.sort(idx)


def fuente(datos, personas, semilla=0):
    """Rasgos de las personas fuente, cada una recentrada con su propio centro."""
    Z, y = [], []
    for p in personas:
        i = balancear(datos[p], np.random.default_rng([semilla, p]))
        Z.append(tangente(datos[p]['X'][i])[0]); y.append(datos[p]['y'][i])
    return np.vstack(Z), np.concatenate(y)


def una(tarea):
    """Dejando fuera a una persona: BA en su corrida 12 de los tres decoders, para cada n."""
    from sklearn.linear_model import LogisticRegression
    objetivo, personas = tarea
    datos = {p: cargar(p) for p in personas}
    d = datos[objetivo]
    lr = lambda: LogisticRegression(max_iter=2000)
    Zf, yf = fuente(datos, [p for p in personas if p != objetivo])
    pre = lr().fit(Zf, yf)
    i = balancear(d, np.random.default_rng([1, objetivo]))
    cal, pru = i[d['corrida'][i] != CORRIDA_PRUEBA], i[d['corrida'][i] == CORRIDA_PRUEBA]
    Xp, yp = d['X'][pru], d['y'][pru]
    ba = lambda modelo, M: hw.exactitud_balanceada(yp, modelo.predict(tangente(Xp, M)[0]))
    filas = []
    for n in ENES:
        if n > len(cal):
            continue
        Xc, yc = d['X'][cal[:n]], d['y'][cal[:n]]
        M = centro(Xc) if n else centro(d['reposo'])              # lo unico que aporta la persona sin etiquetas
        r = {'persona': objetivo, 'n': n, 'transferido': ba(pre, M), 'cero': np.nan, 'afinado': np.nan}
        if n and len(set(yc)) == 2:
            Zc = tangente(Xc, M)[0]
            r['cero'] = ba(lr().fit(Zc, yc), M)
            w = np.r_[np.ones(len(yf)), np.full(n, PESO_PROPIO)]
            r['afinado'] = ba(lr().fit(np.vstack([Zf, Zc]), np.r_[yf, yc], sample_weight=w), M)
        filas.append(r)
    return filas


def medir(personas, salida=print):
    with ThreadPoolExecutor(8) as ex:                             # la descarga de PhysioNet es lenta: en paralelo
        list(ex.map(cargar, personas))
    with ProcessPoolExecutor(max_workers=min(12, len(personas))) as ex:
        filas = [f for fs in ex.map(una, [(p, personas) for p in personas]) for f in fs]
    (CARPETA / 'transferencia.pkl').write_bytes(pickle.dumps({'personas': personas, 'filas': filas}))
    return informe(salida)


def informe(salida=print):
    d = pickle.loads((CARPETA / 'transferencia.pkl').read_bytes())
    filas, personas = d['filas'], d['personas']
    salida(f'{len(personas)} personas de EEGMMIDB, dejando-una-fuera; BA en la corrida 12 de cada una (media +- EE)')
    tabla = {}
    for n in ENES:
        f = [x for x in filas if x['n'] == n]
        if not f:
            continue
        media = lambda v: (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))) if np.isfinite(v).all() else (np.nan, np.nan)
        tabla[n] = {k: media(np.array([x[k] for x in f])) for k in ('cero', 'transferido', 'afinado')}
        tabla[n]['dif'] = media(np.array([x['afinado'] - x['cero'] for x in f]))
        t = tabla[n]
        salida(f"  n = {n:2d}: desde cero {t['cero'][0]:.3f} +- {t['cero'][1]:.3f} | transferido {t['transferido'][0]:.3f} +- "
               f"{t['transferido'][1]:.3f} | afinado {t['afinado'][0]:.3f} +- {t['afinado'][1]:.3f} | afinado - cero "
               f"{t['dif'][0]:+.3f} +- {t['dif'][1]:.3f}")
    # ahorro: cuantos ensayos necesita "desde cero" para igualar al pre-entrenado con n0 ensayos propios
    cero = {n: t['cero'][0] for n, t in tabla.items() if n}
    for n0 in (0, 4, 8, 12):
        if n0 in tabla:
            meta = tabla[n0]['afinado'][0] if n0 else tabla[n0]['transferido'][0]
            alcanza = next((n for n in sorted(cero) if cero[n] >= meta), None)
            tabla[n0]['equivale'] = alcanza
            salida(f"  con {n0} ensayos propios el pre-entrenado da BA {meta:.3f}; desde cero hacen falta "
                   + (f'{alcanza} ensayos para igualarlo' if alcanza else f'mas de {max(cero)} ensayos'))
    graficar(tabla, len(personas))
    salida(f'Figura: {FIGURA}')
    return tabla


def graficar(tabla, personas):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, nombre, color in (('cero', 'desde cero (hoy)', '#7f7f7f'), ('transferido', 'pre-entrenado, solo recentrado', '#1f77b4'),
                             ('afinado', 'pre-entrenado + ensayos propios', '#ff7f0e')):
        n = [x for x in sorted(tabla) if np.isfinite(tabla[x][k][0])]
        ax.errorbar(n, [tabla[x][k][0] for x in n], [tabla[x][k][1] for x in n], marker='o', capsize=3, color=color, label=nombre)
    ax.axhline(0.5, color='k', lw=0.5, ls=':')
    ax.set_xlabel('ensayos de calibración de la persona'); ax.set_ylabel('BA en otra corrida')
    ax.set_title(f'Mano derecha imaginada contra reposo, 8 canales del Unicorn\nEEGMMIDB, {personas} personas, dejando-una-fuera (media ± EE)',
                 fontsize=10)
    ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGURA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURA, dpi=150); plt.close(fig)


def guardar_modelo(personas, salida=print):
    """El decoder pre-entrenado con todas las personas, para `orquestador.py real --preentrenado`."""
    datos = {p: cargar(p) for p in personas}
    Z, y = fuente(datos, personas)
    hw.guardar({'Z': Z.astype(np.float32), 'y': y, 'canales': config.CANALES_EEG, 'personas': len(personas),
                'banda': config.BANDA_MI, 'ventana_s': config.VENTANA_MI,
                'origen': 'EEGMMIDB (PhysioNet), corridas 4, 8 y 12: mano derecha imaginada contra reposo'},
               config.DECODER_PREENTRENADO)
    salida(f'Guardado modelos/{config.DECODER_PREENTRENADO}: {len(y)} ensayos de {len(personas)} personas')


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else '40'
    if arg == 'informe':
        return informe()
    if arg == 'modelo':
        d = pickle.loads((CARPETA / 'transferencia.pkl').read_bytes())
        return guardar_modelo(d['personas'])
    personas = [p for p in range(1, 110) if p not in EXCLUIDAS][:int(arg)]
    medir(personas)


if __name__ == '__main__':
    main()
