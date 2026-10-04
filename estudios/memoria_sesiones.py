"""Memoria entre sesiones: cuantos ensayos de calibracion ahorra arrancar de la sesion previa del
mismo piloto (orquestador.py --desde-sesion; memoria.py), frente a calibrar desde cero.

Validacion honesta, igual con todas las fuentes de datos:
  - La sesion PREVIA solo aporta la memoria (sus ensayos, recentrados con su propio centro).
  - De la sesion NUEVA, los primeros n ensayos calibran y los ULTIMOS prueban. Los de prueba son
    posteriores en el tiempo y nunca se usan para ajustar nada: ni el centro, ni el clasificador, ni
    la eleccion de canales, ni el umbral.
  - En la prueba el decoder recentra en linea con cada ventana sin etiqueta, como en el lazo.
  - El peso de los ensayos nuevos (config.MEMORIA_PESO_NUEVO) y los largos de la calibracion corta se
    fijaron antes de medir y no se ajustaron contra la prueba.

Decoders de MI que se comparan, con n ensayos de la sesion nueva:
  cero      el de hoy: hardware.DecoderIM solo con esos n ensayos, eligiendo canales
  memoria   --desde-sesion: rasgos de la sesion previa + los n de hoy (que pesan mas), recentrados con
            el centro de hoy. Con n = 0 el centro sale de ventanas de hoy SIN etiqueta
  previo    el decoder de la sesion previa tal cual (lo que hace --saltar-calibracion con modelos viejos)
Detectores de ErrP, igual: cero (n epocas de hoy), memoria (previas + n de hoy) y previo.

Fuentes:
  reales      los calibracion_mi_*.npz y calibracion_errp_*.npz que deja cada calibracion real en
              resultados/. Una pareja de sesiones es UNA medicion: se reporta con el intervalo de sus
              ensayos de prueba y se etiqueta como exploratoria.
  physionet   40 personas reales de EEGMMIDB (las de estudios/transferencia_physionet.py): memoria =
              corrida 4, sesion nueva = corrida 8, prueba = corrida 12. OJO: las tres corridas son del
              mismo dia y sin quitarse el gorro, minutos aparte; no son el piloto ni el Unicorn. Mide si
              la memoria ayuda entre corridas, no entre dias.
  gemelo      verificacion: otra sesion del mismo sujeto es otro ruido y otra ganancia por electrodo, un
              cambio que programamos nosotros. Dice que el codigo funciona, no cuanto ahorra una persona.

Uso: python estudios/memoria_sesiones.py lista
     python estudios/memoria_sesiones.py reales <mi_previa.npz> <mi_nueva.npz> [<errp_previa.npz> <errp_nueva.npz>]
     python estudios/memoria_sesiones.py physionet
     python estudios/memoria_sesiones.py gemelo [sujetos]
"""
import copy
import pickle
import sys
import time
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
import numpy as np

import config
import hardware as hw
import memoria

CARPETA = config.RESULTADOS / 'memoria_sesiones'
FIGURA = config.RAIZ / 'docs' / 'figuras' / 'memoria_sesiones.png'
ENES_MI = (0, 4, 8, 12, 16, 20, 24, 36)
ENES_ERRP = (40, 80, 120)         # cada punto elige entre 4 configuraciones del detector: minutos por sujeto
SIN_ETIQUETA = 6                  # ventanas de hoy sin etiqueta para recentrar cuando n = 0 (12 s de EEG)
PRUEBA_MIN_MI, PRUEBA_MIN_ERRP = 12, 40
VARIANTES = ('cero', 'memoria', 'previo')


def ba_mi(dec, X, y):
    """Predicciones de un decoder sobre ventanas nuevas en orden, recentrando en linea como el lazo."""
    dec = copy.deepcopy(dec)
    return np.array([int(dec.w0 @ dec.phi(x) + dec.c0 >= 0) for x in X])


def curva_mi(Xa, ya, Xc, yc, Xp, yp, enes=ENES_MI):
    """Una pareja de sesiones. a: la previa; c: ensayos de calibracion de la nueva, en orden;
    p: prueba (posteriores). Devuelve {n: {variante: predicciones sobre la prueba | None}}."""
    pre = memoria.mi_de_sesion(Xa, ya)
    previo = ba_mi(hw.DecoderIM().ajustar(Xa, ya, config.candidatos('decoder')), Xp, yp)
    filas = {}
    for n in enes:
        if n > len(yc):
            continue
        X, y = Xc[:n], yc[:n]
        r = {'cero': None, 'memoria': None, 'previo': previo}
        if n == 0:
            r['memoria'] = ba_mi(hw.DecoderIM().ajustar_desde(pre, reposo=Xc[:SIN_ETIQUETA]), Xp, yp)
        elif np.bincount(y, minlength=2).min() >= 2:
            r['memoria'] = ba_mi(hw.DecoderIM().ajustar_desde(pre, X, y, peso=config.MEMORIA_PESO_NUEVO), Xp, yp)
            try:
                r['cero'] = ba_mi(hw.DecoderIM().ajustar(X, y, config.candidatos('decoder')), Xp, yp)
            except ValueError:                             # muy pocos ensayos para elegir canales
                pass
        filas[n] = r
    return filas


def curva_errp(Xa, ya, Xc, yc, Xp, yp, enes=ENES_ERRP):
    """Lo mismo para el detector de ErrP. Ademas, para cada n, la BA que REPORTARIA el CP3 con memoria
    (validacion cruzada solo sobre las epocas de hoy), para compararla con la real en la prueba."""
    previo = hw.DetectorErrP().ajustar(Xa, ya, config.candidatos('detector'), evaluar=False)
    pred = lambda d: np.array([int(d.p_error(e) > d.umbral) for e in Xp])
    filas = {}
    for n in enes:
        if n > len(yc) or np.bincount(yc[:n], minlength=2).min() < 4:
            continue
        X, y = Xc[:n], yc[:n]
        mem = memoria.detector_con_memoria(previo, Xa, ya, X, y)
        cero = hw.DetectorErrP().ajustar(X, y, config.candidatos('detector'), evaluar=False)
        filas[n] = {'cero': pred(cero), 'memoria': pred(mem), 'previo': pred(previo), 'reportada': mem.ba}
    return filas


def resumir(parejas, yps, salida=print, que='BA'):
    """parejas: [{n: {variante: predicciones}}] (una por pareja de sesiones); yps: sus etiquetas de
    prueba. Con varias parejas: media +- error estandar entre parejas. Con una: BA e intervalo del 90 %
    de sus ensayos de prueba. Devuelve {n: {variante: (valor, dispersion)}}."""
    tabla = {}
    for n in sorted({n for p in parejas for n in p}):
        tabla[n] = {}
        for v in VARIANTES:
            con = [(p[n][v], y) for p, y in zip(parejas, yps) if n in p and p[n][v] is not None]
            if len(con) < max(1, len(parejas) // 2):        # variante que casi nadie pudo ajustar con ese n
                continue
            bas = np.array([hw.exactitud_balanceada(y, pred) for pred, y in con])
            if len(parejas) == 1:
                tabla[n][v] = (float(bas[0]), hw.intervalo_ba(con[0][1], con[0][0]))
            else:
                tabla[n][v] = (float(bas.mean()), float(bas.std(ddof=1) / np.sqrt(len(bas))), len(bas))
        rep = [p[n]['reportada'] for p in parejas if n in p and 'reportada' in p[n]]
        if rep:
            tabla[n]['reportada'] = float(np.mean(rep))
        if not any(v in tabla[n] for v in ('cero', 'memoria')):
            del tabla[n]
            continue
        # diferencia PAREADA (la misma pareja de sesiones con y sin memoria): lo que dice si es mas que ruido
        dif = np.array([hw.exactitud_balanceada(y, p[n]['memoria']) - hw.exactitud_balanceada(y, p[n]['cero'])
                        for p, y in zip(parejas, yps) if n in p and p[n]['cero'] is not None and p[n]['memoria'] is not None])
        if len(dif) > 1 and 'cero' in tabla[n]:
            tabla[n]['dif'] = (float(dif.mean()), float(dif.std(ddof=1) / np.sqrt(len(dif))), len(dif))
        f = lambda t: (f'{t[0]:.3f} [{t[1][0]:.2f}, {t[1][1]:.2f}]' if len(parejas) == 1 else f'{t[0]:.3f} +- {t[1]:.3f}')
        salida(f'  n = {n:3d}: ' + ' | '.join(f'{v} {f(tabla[n][v])}' for v in VARIANTES if v in tabla[n])
               + (f" | memoria - cero {tabla[n]['dif'][0]:+.3f} +- {tabla[n]['dif'][1]:.3f}" if 'dif' in tabla[n] else '')
               + (f" | el CP3 con memoria reportaria {tabla[n]['reportada']:.3f}" if rep else ''))
    return tabla


def ahorro(tabla, salida=print, unidad='ensayos'):
    """Cuantos ensayos necesita 'cero' para igualar a la memoria con n0, y con cuantos llega cada uno
    al umbral del checkpoint. Con una sola pareja es una lectura de puntos ruidosos: orientativa."""
    cero = {n: t['cero'][0] for n, t in tabla.items() if 'cero' in t}
    mem = {n: t['memoria'][0] for n, t in tabla.items() if 'memoria' in t}
    if not cero or not mem:
        return
    for n0 in sorted(mem)[:4]:
        alcanza = next((n for n in sorted(cero) if cero[n] >= mem[n0]), None)
        salida(f"  con {n0} {unidad} de hoy la memoria da BA {mem[n0]:.3f}; desde cero hacen falta "
               + (f'{alcanza} para igualarla (ahorra {alcanza - n0})' if alcanza is not None
                  else f'mas de {max(cero)} (el maximo medido)'))


def partir(X, y, prueba_min, enes):
    """Sesion nueva en orden: calibracion (hasta el mayor n que deja al menos prueba_min para probar) y prueba."""
    n_cal = max([n for n in enes if n <= len(y) - prueba_min], default=0)
    return X[:n_cal], y[:n_cal], X[n_cal:], y[n_cal:]


# ====================================================================== fuentes
def gemelo(sujetos=4, salida=print):
    import cerebro_sintetico as cs
    mi, errp, ymi, yerrp = [], [], [], []
    for s in range(sujetos):
        Xa, ya = cs.sesion_mi(40, semilla=s)
        Xb, yb = cs.sesion_mi(72, semilla=s, sesion=1)
        Xc, yc, Xp, yp = partir(Xb, yb, 36, ENES_MI)
        mi.append(curva_mi(Xa, ya, Xc, yc, Xp, yp)); ymi.append(yp)
        Ea, ea = cs.sesion_errp(120, semilla=s)
        Eb, eb = cs.sesion_errp(240, semilla=s, sesion=1)
        Ec, ec, Ep, ep = partir(Eb, eb, 120, ENES_ERRP)
        errp.append(curva_errp(Ea, ea, Ec, ec, Ep, ep)); yerrp.append(ep)
        salida(f'  sujeto {s} listo', flush=True)
    salida(f'GEMELO (verificacion; el cambio entre sesiones lo programamos nosotros), {sujetos} sujetos, media +- EE')
    salida(' MI: BA en 36 ensayos posteriores de la sesion nueva')
    t_mi = resumir(mi, ymi, salida)
    ahorro(t_mi, salida)
    salida(' ErrP: BA en 120 epocas posteriores de la sesion nueva')
    t_errp = resumir(errp, yerrp, salida)
    ahorro(t_errp, salida, 'epocas')
    return guardar('gemelo', {'mi': t_mi, 'errp': t_errp, 'n': sujetos})


def physionet(salida=print):
    import transferencia_physionet as tp
    d = pickle.loads((tp.CARPETA / 'transferencia.pkl').read_bytes())
    parejas, yps = [], []
    for p in d['personas']:
        x = tp.cargar(p)
        i = tp.balancear(x, np.random.default_rng([1, p]))       # mismas ventanas que el estudio de transferencia
        de = lambda c: (x['X'][i[x['corrida'][i] == c]], x['y'][i[x['corrida'][i] == c]])
        (Xa, ya), (Xc, yc), (Xp, yp) = de(4), de(8), de(12)
        if min(len(ya), len(yc), len(yp)) < 8:
            continue
        parejas.append(curva_mi(Xa, ya, Xc, yc, Xp, yp)); yps.append(yp)
    salida(f'PHYSIONET (personas reales; corridas del mismo dia, sin quitarse el gorro): {len(parejas)} personas, media +- EE')
    salida(' MI: memoria = corrida 4, calibracion = primeros n ensayos de la corrida 8, BA en la corrida 12')
    t = resumir(parejas, yps, salida)
    ahorro(t, salida)
    return guardar('physionet', {'mi': t, 'n': len(parejas)})


def reales(archivos, salida=print):
    """archivos: mi_previa.npz, mi_nueva.npz y, opcional, errp_previa.npz, errp_nueva.npz."""
    hora = lambda r: time.strftime('%d/%m %H:%M', time.localtime(Path(r).stat().st_mtime))
    a, b = np.load(archivos[0]), np.load(archivos[1])
    Xc, yc, Xp, yp = partir(b['X'], b['y'], PRUEBA_MIN_MI, ENES_MI)
    salida(f"DATOS REALES, UNA pareja de sesiones (EXPLORATORIO): previa {Path(archivos[0]).name} ({hora(archivos[0])}, "
           f"{len(a['y'])} ensayos), nueva {Path(archivos[1]).name} ({hora(archivos[1])}, {len(b['y'])} ensayos)")
    salida(f' MI: calibracion = primeros n ensayos de la nueva; BA [intervalo del 90 %] en sus ultimos {len(yp)}')
    if len(yp) < PRUEBA_MIN_MI or len(set(yp)) < 2:
        return salida('  la sesion nueva tiene muy pocos ensayos para apartar una prueba: no se mide')
    t = {'mi': resumir([curva_mi(a['X'], a['y'], Xc, yc, Xp, yp)], [yp], salida)}
    ahorro(t['mi'], salida)
    if len(archivos) >= 4:
        a, b = np.load(archivos[2]), np.load(archivos[3])
        Xc, yc, Xp, yp = partir(b['X'], b['y'], PRUEBA_MIN_ERRP, ENES_ERRP)
        salida(f" ErrP: previa {Path(archivos[2]).name} ({len(a['y'])} epocas), nueva {Path(archivos[3]).name} "
               f"({len(b['y'])}); BA [intervalo del 90 %] en sus ultimas {len(yp)}")
        if len(yc) and len(set(yp)) == 2:
            t['errp'] = resumir([curva_errp(a['X'], a['y'], Xc, yc, Xp, yp)], [yp], salida)
            ahorro(t['errp'], salida, 'epocas')
        else:
            salida('  la sesion nueva tiene muy pocas epocas para apartar una prueba: no se mide')
    salida('  Una pareja de sesiones no da un intervalo entre personas: son cifras de ESTA pareja.')
    return guardar('reales', dict(t, n=1, archivos=[Path(r).name for r in archivos]))


def lista(salida=print):
    """Las calibraciones guardadas en resultados/, de la mas nueva a la mas vieja. No dice de quien son:
    el gemelo y una persona dejan los mismos archivos."""
    for patron in ('calibracion_mi_*.npz', 'calibracion_errp_*.npz'):
        for r in sorted(config.RESULTADOS.glob(patron), reverse=True)[:12]:
            y = np.load(r)['y']
            salida(f"  {time.strftime('%a %d/%m %H:%M', time.localtime(r.stat().st_mtime))}  {r.name}  {len(y)} ensayos "
                   f"({int(y.sum())} de la clase 1)")


def guardar(nombre, datos):
    CARPETA.mkdir(parents=True, exist_ok=True)
    (CARPETA / f'{nombre}.pkl').write_bytes(pickle.dumps(datos))
    return datos


def graficar(salida=print):
    """Figura con lo ya medido (resultados/memoria_sesiones/*.pkl): BA de MI contra ensayos de hoy."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    titulos = {'reales': 'Datos reales del piloto: UNA pareja de sesiones (exploratorio)',
               'physionet': 'EEGMMIDB: personas reales, corridas del mismo día',
               'gemelo': 'Gemelo (verificación: cambio entre sesiones programado)'}
    hay = [n for n in titulos if (CARPETA / f'{n}.pkl').exists()]
    if not hay:
        return
    fig, ejes = plt.subplots(1, len(hay), figsize=(5.2 * len(hay), 3.8), squeeze=False)
    for ax, nombre in zip(ejes[0], hay):
        d = pickle.loads((CARPETA / f'{nombre}.pkl').read_bytes())
        for v, etiqueta, color in (('cero', 'desde cero (hoy)', '#7f7f7f'), ('memoria', 'con memoria de la sesión previa', '#ff7f0e'),
                                   ('previo', 'decoder previo sin recalibrar', '#1f77b4')):
            n = [x for x in sorted(d['mi']) if v in d['mi'][x]]
            val = [d['mi'][x][v] for x in n]
            err = ([[t[0] - t[1][0] for t in val], [t[1][1] - t[0] for t in val]] if d['n'] == 1 else [t[1] for t in val])
            ax.errorbar(n, [t[0] for t in val], err, marker='o', capsize=3, color=color, label=etiqueta)
        ax.axhline(config.MI_EXACTITUD_MIN, color='k', lw=0.6, ls=':')
        ax.set_xlabel('ensayos de calibración de la sesión nueva'); ax.set_ylabel('BA en ensayos posteriores')
        quienes = 'sujetos del gemelo' if nombre == 'gemelo' else 'personas'
        ax.set_title(titulos[nombre] + (f"\n{d['n']} {quienes}, media ± EE" if d['n'] > 1 else '\nintervalo del 90 % de sus ensayos de prueba'),
                     fontsize=9)
        ax.legend(fontsize=7, loc='best'); ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGURA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURA, dpi=150); plt.close(fig)
    salida(f'Figura: {FIGURA}')


def main():
    warnings.filterwarnings('ignore')                      # avisos de convergencia con muy pocos ensayos
    arg = sys.argv[1] if len(sys.argv) > 1 else 'lista'
    if arg == 'lista':
        return lista()
    if arg == 'reales':
        if len(sys.argv) not in (4, 6):
            raise SystemExit('uso: reales <mi_previa.npz> <mi_nueva.npz> [<errp_previa.npz> <errp_nueva.npz>]')
        reales(sys.argv[2:])
    elif arg == 'physionet':
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        physionet()
    elif arg == 'gemelo':
        gemelo(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    graficar()


if __name__ == '__main__':
    main()
