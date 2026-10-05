"""Por que el detector de ErrP no encontro nada en P001 (EXPLORATORIO: un participante, una sesion).

La sesion real del 5 de octubre se detuvo en el CP3: "ErrP: sens 0.06, espec 0.90, BA 0.48 con 120 epocas -> NO GO".
Este script contrasta con los datos, una por una, las hipotesis de por que, cada una con su control de etiquetas
permutadas (siempre DENTRO de estratos que respetan el diseno: bloque de 20 ensayos x cue, como los asigno
`Orquestador.calibrar_errp`):

  0  Que ortesis uso la sesion y si el movimiento se veia. La latencia ACK -> inicio de cada paso contra
     hardware.latencia_mecanica_simulada(seq, semilla) de 20 semillas (la real no depende de `seq`), y los eventos
     `mano` del flujo Estado (los publica solo `--mano-virtual`).
  1  Hay senal? Onda diferencia error - correcto por canal con intervalo bootstrap; prueba de permutacion con correccion
     por el maximo de |t| sobre canales x ventanas de 50 ms, y la de las ventanas Ne/Pe en Fz/Cz fijadas de antemano.
  1b El DetectorErrP completo contra la nula: las 4 AUC con que elige canales y vistas, con etiquetas reales y permutadas.
  1c Sensibilidad al preprocesamiento: otra banda, sin linea base, referencia al promedio.
  2  Desfase de la epoca: AUC de validacion cruzada (ventanas de 100 ms + LDA con encogimiento) con la epoca cortada
     de -300 a +500 ms del inicio del movimiento, contra la nula (y contra el maximo de la nula sobre los desfases).
  3  Calidad y fatiga: amplitudes, artefactos, giro de la cabeza, alfa occipital por bloque, primera contra segunda mitad.
  4  Diseno: errores por bloque y direccion, tiempos, cues; 4b respuesta occipital promedio al movimiento y al cue (control de
     visibilidad por EEG, debil).
  5  Control positivo: la onda diferencia del gemelo (cerebro_sintetico.plantilla_errp) sumada al EEG CONTINUO de P001 en
     los pasos erroneos. Si la cadena de calculo (marcas, filtro, linea base, detector) sirve con EEG real, tiene que
     encontrarla; a que amplitud deja de hacerlo dice cuanto podia ver este registro, y la curva de aprendizaje (n epocas hechas
     de instantes al azar del mismo registro) cuanto ayudan mas epocas SI el ErrP fuera de esa amplitud. Efecto minimo
     detectable por numero de errores, en el paso 1.

Entradas (cada una opcional: lo que falte se avisa y se omiten los pasos que la piden):
  --npz   calibracion_errp_*.npz (epocas de 1 s; paso 1, 3 parcial, 4)
  --jsonl sesion_xdf_*_estado.jsonl (eventos y marcadores; paso 0 y 4)
  --xdf   el XDF de LabRecorder (EEG continuo, IMU y marcadores; pasos 2, 3, 5)

  python estudios/diagnostico_errp_p001.py --xdf ruta/archivo.xdf [--rapido] [--sin-detector]

Guarda resultados/diagnostico_errp_p001.json y docs/figuras/diagnostico_errp_p001.png. Informe: docs/diagnostico_errp_p001.md.
EXPLORATORIO: una persona, una sesion; nada de esto se generaliza.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
import config
import hardware as hw

FS = config.FLUJOS['EEG'][2]
ANTES = -config.EPOCA_ERRP[0]                        # 0.2 s de linea base antes del inicio del movimiento
DESPUES = config.EPOCA_ERRP[1]
SESION = 'sub_P001_ses_S001_task_Default_run_001'
NPZ_ERRP = config.RESULTADOS / f'calibracion_errp_6_{SESION}.npz'
JSONL = config.RESULTADOS / f'sesion_xdf_6_{SESION}{config.SUFIJO_ESTADO}'
SALIDA_JSON = config.RESULTADOS / 'diagnostico_errp_p001.json'
FIGURA = config.RAIZ / 'docs' / 'figuras' / 'diagnostico_errp_p001.png'

TAM_BLOQUE = 20                                      # ensayos por bloque del plan (Orquestador.calibrar_errp)
VENTANAS_T = (0.10, 0.70, 0.05)                      # prueba de permutacion: ventanas de 50 ms de 100 a 700 ms
A_PRIORI = {'Ne': (0.20, 0.35), 'Pe': (0.30, 0.50)}  # ventanas fijadas antes de mirar los datos
CANALES_A_PRIORI = ('Fz', 'Cz')
VENTANAS_AUC = (0.10, 0.70, 0.10)                    # clasificador simple: ventanas de 100 ms
DESFASES_S = np.round(np.arange(-0.30, 0.5001, 0.05), 3)
SEMILLAS_ORTESIS = range(20)


# ============================ Diseno y permutaciones ============================
def estratos_del_diseno(y, direccion, tam_bloque=TAM_BLOQUE):
    """Estrato de cada ensayo = (bloque de `tam_bloque`, cue). El plan de la calibracion mezcla al azar, dentro de cada
    bloque, las parejas (cue, error?), asi que las etiquetas de error son intercambiables DENTRO de un (bloque, cue).
    direccion = cue si acerto, 1 - cue si fue error: cue = direccion XOR y."""
    y, direccion = np.asarray(y, dtype=int), np.asarray(direccion, dtype=int)
    return (np.arange(len(y)) // tam_bloque) * 2 + (direccion ^ y)


def permutar_en_estratos(y, estratos, rng):
    """Las etiquetas `y` mezcladas al azar dentro de cada estrato (la nula exacta de un diseno aleatorizado por bloques)."""
    salida = np.array(y, copy=True)
    for e in np.unique(estratos):
        i = np.flatnonzero(estratos == e)
        salida[i] = salida[rng.permutation(i)]
    return salida


def medias_por_ventana(X, desde_s, hasta_s, ancho_s, fs=FS, antes=ANTES):
    """(n, canales, ventanas) con la media de cada ventana de `ancho_s` entre `desde_s` y `hasta_s` tras el inicio del
    movimiento, y el inicio de cada ventana. X: (n, canales, muestras) de una epoca que empieza `antes` s antes."""
    inicios = np.arange(desde_s, hasta_s - 1e-9, ancho_s)
    cortes = [(int(round((antes + a) * fs)), int(round((antes + a + ancho_s) * fs))) for a in inicios]
    return np.stack([X[:, :, i:j].mean(axis=2) for i, j in cortes], axis=2), inicios


def t_welch(F, y):
    """t de Welch (error contra correcto) de cada columna de F (n, m)."""
    F, g = np.asarray(F, dtype=float), np.asarray(y) == 1
    a, b = F[g], F[~g]
    return (a.mean(0) - b.mean(0)) / np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))


def _t_welch_muchas(F, Y):
    """t de Welch para k etiquetados a la vez. F (n, m); Y (k, n) de 0/1. -> (k, m)."""
    n1 = Y.sum(1, keepdims=True)
    n0 = Y.shape[1] - n1
    S1, Q1 = Y @ F, Y @ (F ** 2)
    S0, Q0 = F.sum(0)[None] - S1, (F ** 2).sum(0)[None] - Q1
    m1, m0 = S1 / n1, S0 / n0
    v1, v0 = (Q1 - n1 * m1 ** 2) / (n1 - 1), (Q0 - n0 * m0 ** 2) / (n0 - 1)
    return (m1 - m0) / np.sqrt(v1 / n1 + v0 / n0)


def prueba_permutacion(F, y, estratos, n_perm, rng, umbral_t=2.0):
    """Prueba de permutacion de las etiquetas (dentro de `estratos`) sobre las columnas de F (n, m).

    p_sin_corregir: por columna. p_corregido: por maximo de |t| sobre TODAS las columnas (controla el error de tipo I
    de la familia). n_sobre_umbral: cuantas columnas pasan |t| > umbral_t y cuantas pasarian por azar (media de la nula)."""
    F = np.asarray(F, dtype=float)
    Y = np.stack([permutar_en_estratos(y, estratos, rng) for _ in range(n_perm)]).astype(float)
    t_obs = t_welch(F, y)
    T = np.abs(_t_welch_muchas(F, Y))
    cuentas = (T > umbral_t).sum(1)
    n_obs = int((np.abs(t_obs) > umbral_t).sum())
    return {'t': t_obs,
            'p_sin_corregir': (1 + (T >= np.abs(t_obs)[None]).sum(0)) / (1 + n_perm),
            'p_corregido': (1 + (T.max(1)[:, None] >= np.abs(t_obs)[None]).sum(0)) / (1 + n_perm),
            'max_nula': T.max(1), 'n_sobre_umbral': n_obs, 'n_sobre_umbral_nula': float(cuentas.mean()),
            'p_cuenta': float((1 + (cuentas >= n_obs).sum()) / (1 + n_perm))}


def bootstrap_onda_diferencia(X, y, n_boot, rng, nivel=0.95):
    """Onda diferencia error - correcto (canales x muestras) con su intervalo bootstrap por muestra (remuestreando los
    ensayos dentro de cada clase). -> (observada, bajo, alto, muestras (n_boot, canales, muestras))."""
    X1, X0 = X[y == 1], X[y == 0]
    obs = X1.mean(0) - X0.mean(0)
    d = np.empty((n_boot,) + obs.shape)
    for b in range(n_boot):
        d[b] = X1[rng.integers(0, len(X1), len(X1))].mean(0) - X0[rng.integers(0, len(X0), len(X0))].mean(0)
    bajo, alto = np.percentile(d, [(1 - nivel) / 2 * 100, (1 + nivel) / 2 * 100], axis=0)
    return obs, bajo, alto, d


def d_de_cohen(F, y):
    """d de Cohen (error contra correcto) de cada columna de F, con la desviacion combinada."""
    F, g = np.asarray(F, dtype=float), np.asarray(y) == 1
    a, b = F[g], F[~g]
    sp = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean(0) - b.mean(0)) / sp


def efecto_minimo_detectable(sd, n_error, n_correcto, alfa=0.05, potencia=0.80):
    """Diferencia de medias (en las unidades de `sd`) que una prueba de dos colas con nivel `alfa` detecta con la
    `potencia` dada: (z(1 - alfa/2) + z(potencia)) * sd * sqrt(1/n1 + 1/n0). Aproximacion normal, varianzas iguales."""
    from scipy.stats import norm
    return float((norm.ppf(1 - alfa / 2) + norm.ppf(potencia)) * sd * np.sqrt(1.0 / n_error + 1.0 / n_correcto))


def puntajes_lda(F_ent, y_ent, F_pru):
    """Puntaje de un LDA binario con encogimiento automatico (Ledoit-Wolf) entrenado con (F_ent, y_ent) y aplicado a F_pru. Es la cuenta de
    LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto') de sklearn (la covarianza de cada clase se estandariza y se encoge por
    separado, y se promedian con las frecuencias de las clases) sin su sobrecarga: unas 5 veces mas rapido, con los mismos puntajes
    (la prueba lo comprueba). El sesgo cambia de un pliegue a otro, y cuenta cuando se juntan los puntajes de todos los pliegues."""
    from sklearn.covariance import ledoit_wolf
    F_ent = np.asarray(F_ent, dtype=float)
    y_ent = np.asarray(y_ent)
    n = len(y_ent)
    cov, medias = 0.0, []
    for k in (0, 1):
        Fk = F_ent[y_ent == k]
        medias.append(Fk.mean(0))
        escala = Fk.std(0)
        escala[escala == 0] = 1.0
        cov = cov + (len(Fk) / n) * (escala[:, None] * ledoit_wolf((Fk - medias[-1]) / escala)[0] * escala[None, :])
    w = np.linalg.solve(cov, medias[1] - medias[0])
    sesgo = -0.5 * w @ (medias[1] + medias[0]) + np.log((y_ent == 1).sum() / (y_ent == 0).sum())
    return np.asarray(F_pru, dtype=float) @ w + sesgo


def auc_de_puntajes(y, puntaje):
    """AUC (probabilidad de que un error puntue mas que un correcto, con empates a medias) por rangos."""
    from scipy.stats import rankdata
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    r = rankdata(puntaje)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def auc_cv(F, y, repeticiones=2, pliegues=5, semilla=0):
    """AUC de validacion cruzada estratificada (`pliegues` pliegues, `repeticiones` veces con otra mezcla) de un LDA con encogimiento
    automatico sobre F (n, m); el AUC se calcula con los puntajes de todos los pliegues juntos. Ojo: con ruido puro este AUC sale
    algo POR DEBAJO de 0.5 (el entrenamiento de cada pliegue tiene de menos justo lo que la prueba tiene de mas), asi que se compara
    siempre con su nula de etiquetas permutadas, no con 0.5."""
    from sklearn.model_selection import StratifiedKFold
    y, F = np.asarray(y), np.asarray(F, dtype=float)
    aucs = []
    for r in range(repeticiones):
        puntaje = np.zeros(len(y))
        for ent, pru in StratifiedKFold(pliegues, shuffle=True, random_state=semilla + r).split(F, y):
            puntaje[pru] = puntajes_lda(F[ent], y[ent], F[pru])
        aucs.append(auc_de_puntajes(y, puntaje))
    return float(np.mean(aucs))


# ============================ Registro continuo ============================
def preparar_continuo(x, t, fs=FS, banda=config.BANDA_ERRP):
    """Filtra el EEG continuo UNA vez por tramo sin huecos: los mismos tramos y la misma cuenta que hardware.cortar_epoca
    (que filtra el tramo entero en cada epoca; aqui se filtra una vez y se corta muchas). -> lista de (tiempos, filtrado)."""
    saltos = np.flatnonzero(np.abs(np.diff(t)) > config.SALUD['hueco_max_s']) + 1
    bordes = np.r_[0, saltos, len(t)]
    tramos = []
    for a, b in zip(bordes[:-1], bordes[1:]):
        if b - a < 2 * fs or not np.isfinite(x[:, a:b]).all():
            continue
        xu, tu = hw.rejilla(x[:, a:b], t[a:b], fs)
        tramos.append((tu, hw.filtrar(xu, banda, fs)))
    return tramos


def cortar(tramos, t0, desfase_s=0.0, fs=FS, antes=ANTES, despues=DESPUES, linea_base=True):
    """Epoca (canales x muestras) alrededor de t0 + desfase_s, con la linea base de los `antes` s previos a ESA hora (si
    `linea_base`), o None si no cabe en un tramo. Con desfase 0 es igual a hardware.cortar_epoca."""
    ini, fin = t0 + desfase_s - antes, t0 + desfase_s + despues
    for tu, xf in tramos:
        if tu[0] <= ini + 1.0 / fs and tu[-1] >= fin - 1.0 / fs:
            n = int(round((antes + despues) * fs))
            i0 = int(round((ini - tu[0]) * fs))
            if i0 < 0 or i0 + n > xf.shape[1]:
                return None
            e = xf[:, i0:i0 + n]
            return e - e[:, :int(antes * fs)].mean(axis=1, keepdims=True) if linea_base else e.copy()
    return None


def cortar_todas(tramos, t0s, desfase_s=0.0, linea_base=True):
    """(n, canales, muestras) con las epocas de todos los t0s; si alguna no cabe, ValueError."""
    E = [cortar(tramos, t0, desfase_s, linea_base=linea_base) for t0 in t0s]
    if any(e is None for e in E):
        raise ValueError(f'hay epocas fuera del registro con el desfase {desfase_s:+.2f} s')
    return np.array(E)


def escanear_desfases(tramos, t0s, y, estratos, desfases, canales, n_perm, rng, repeticiones=2):
    """AUC de validacion cruzada (ventanas de 100 ms + LDA con encogimiento) con la epoca cortada en t0 + cada desfase, y su
    nula: las mismas cuentas con las etiquetas permutadas dentro de `estratos`. Con n_perm = 0 solo la curva observada.
    p_corregido compara cada desfase con el MAXIMO de la nula sobre todos los desfases (la familia entera)."""
    y = np.asarray(y)
    F = []
    for d in desfases:
        W, _ = medias_por_ventana(cortar_todas(tramos, t0s, d)[:, canales, :], *VENTANAS_AUC)
        F.append(W.reshape(len(y), -1))
    auc = np.array([auc_cv(f, y, repeticiones) for f in F])
    r = {'desfases_s': [float(d) for d in desfases], 'auc': auc.tolist(), 'n_caracteristicas': int(F[0].shape[1])}
    if n_perm > 0:
        nula = np.array([[auc_cv(f, yp, repeticiones) for f in F]
                         for yp in (permutar_en_estratos(y, estratos, rng) for _ in range(n_perm))])
        r.update({'nula_media': nula.mean(0).tolist(), 'nula_p95': np.percentile(nula, 95, axis=0).tolist(),
                  'nula_max_p95': float(np.percentile(nula.max(1), 95)),
                  'p_sin_corregir': ((1 + (nula >= auc[None]).sum(0)) / (1 + n_perm)).tolist(),
                  'p_corregido': ((1 + (nula.max(1)[:, None] >= auc[None]).sum(0)) / (1 + n_perm)).tolist(), 'n_perm': n_perm})
    return r


# ============================ Ortesis simulada o real ============================
def resumen_pasos(marcadores):
    """({seq: hora del ACK}, {seq: hora del inicio}) de los marcadores del jsonl ({'t', 'marcador'})."""
    ack, ini = {}, {}
    for m in marcadores:
        nombre = m['marcador']
        if nombre.startswith('paso_ack:'):
            ack[int(nombre.split(':')[1])] = m['t']
        elif nombre.startswith('paso_inicio:'):
            ini[int(nombre.split(':')[1])] = m['t']
    return ack, ini


def ortesis_parece_simulada(seqs, lat_s, semillas=SEMILLAS_ORTESIS):
    """La latencia ACK -> inicio del movimiento de cada paso, es la de OrtesisSimulada? Esa es una funcion de (semilla,
    seq): uniforme en config.LATENCIA_MECANICA_SIM_MS y reproducible. Con la ortesis real la latencia no depende de `seq`
    de esa forma, asi que la correlacion con la de la semilla correcta es ~1 (mas un desfase fijo de la telemetria) y con
    las otras semillas ~0. Veredicto: r >= 0.9 con una semilla y la mejor de las demas por debajo de r - 0.5.
    -> dict (simulada: True / False / None si la latencia es constante y no se puede decidir)."""
    lat = np.asarray(lat_s, dtype=float)
    if len(lat) < 10 or float(np.std(lat)) < 1e-9:
        return {'simulada': None, 'motivo': 'menos de 10 pasos o latencia constante'}
    por_semilla = {}
    for s in semillas:
        sim = np.array([hw.latencia_mecanica_simulada(q, s) for q in seqs])
        por_semilla[int(s)] = float(np.corrcoef(lat, sim)[0, 1])
    mejor = max(por_semilla, key=por_semilla.get)
    otras = [abs(v) for s, v in por_semilla.items() if s != mejor]
    sim = np.array([hw.latencia_mecanica_simulada(q, mejor) for q in seqs])
    dif = lat - sim
    return {'simulada': bool(por_semilla[mejor] >= 0.9 and por_semilla[mejor] - max(otras) >= 0.5), 'semilla': mejor,
            'r': por_semilla[mejor], 'r_otras_max_abs': float(max(otras)), 'r_por_semilla': por_semilla,
            'desfase_mediano_ms': float(np.median(dif) * 1e3), 'desfase_sd_ms': float(dif.std() * 1e3),
            'latencia_ms': [float(lat.min() * 1e3), float(np.median(lat) * 1e3), float(lat.max() * 1e3)]}


# ============================ Control positivo: ErrP inyectado ============================
class _RngFijo:
    """Un generador sin azar para sacar de cerebro_sintetico.plantilla_errp la onda SIN jitter de latencia ni de amplitud."""

    def normal(self, mu=0.0, sd=1.0):
        return mu

    def uniform(self, lo=0.0, hi=1.0):
        return (lo + hi) / 2


def onda_diferencia_gemelo(amplitud_uv):
    """(tiempos, onda) de 1 s: lo que el gemelo le suma a un movimiento ERRONEO y no a uno correcto (Ne ~250 ms, Pe ~360 ms,
    N tardia ~500 ms), con la amplitud `amplitud_uv` del gemelo (por defecto, 6)."""
    import cerebro_sintetico as cs
    rng = _RngFijo()
    onda = cs.plantilla_errp(True, amplitud_uv, rng) - cs.plantilla_errp(False, amplitud_uv, rng)
    return np.arange(len(onda)) / cs.FS, onda


def inyectar_errp(x, t, t0s, es_error, amplitud_uv, pesos):
    """Copia de x (canales x muestras, crudo) con la onda diferencia del gemelo sumada, en cada canal con su peso, a partir
    del inicio de los movimientos erroneos. t: hora de cada muestra. Todo lo demas (ruido, deriva, parpadeos) es de P001."""
    tiempos, onda = onda_diferencia_gemelo(amplitud_uv)
    salida = np.array(x, dtype=float, copy=True)
    pesos = np.asarray(pesos, dtype=float)
    for t0, error in zip(t0s, es_error):
        if not error:
            continue
        i0, i1 = np.searchsorted(t, [t0, t0 + tiempos[-1]])
        salida[:, i0:i1] += pesos[:, None] * np.interp(t[i0:i1] - t0, tiempos, onda, left=0.0, right=0.0)[None]
    return salida


# ============================ Calidad ============================
def giro_maximo(imu, t, t0s, antes=ANTES, despues=DESPUES):
    """Velocidad angular maxima (grados/s, los 3 ejes del giroscopio: igual que EntradaEEG.movimiento) en [t0 - antes, t0 + despues]."""
    salida = []
    for t0 in t0s:
        i0, i1 = np.searchsorted(t, [t0 - antes, t0 + despues])
        salida.append(float(np.abs(imu[3:6, i0:i1]).max()))
    return np.array(salida)


def fisher_2x2(a, b, c, d):
    """p de la prueba exacta de Fisher (dos colas) de la tabla [[a, b], [c, d]]."""
    from scipy.stats import fisher_exact
    return float(fisher_exact([[a, b], [c, d]])[1])


# ============================ Lectura ============================
def leer_jsonl(ruta):
    """(eventos, marcadores) de sesion_..._estado.jsonl: lineas {'t', 'evento'} y {'t', 'marcador'}."""
    eventos, marcadores = [], []
    with open(ruta, encoding='utf-8') as f:
        for linea in f:
            d = json.loads(linea)
            (eventos if 'evento' in d else marcadores).append(d)
    return eventos, marcadores


def leer_xdf(ruta):
    """Un dict con el EEG (8 x N), la IMU (6 x N), sus horas y los marcadores (hora, texto); la lectura es de desde_xdf.py."""
    import desde_xdf
    S = desde_xdf.leer(ruta)
    return {'x': np.asarray(S['EEG']['time_series'], dtype=float).T, 't': np.asarray(S['EEG']['time_stamps'], dtype=float),
            'imu': np.asarray(S['IMU']['time_series'], dtype=float).T,
            'marcas': [(float(ts), m[0]) for ts, m in zip(S['Marcadores']['time_stamps'], S['Marcadores']['time_series'])],
            'paso': np.asarray(S['Paso']['time_series'], dtype=float),
            'estado': [json.loads(m[0]) for m in S['Estado']['time_series']]}


def _f(v, d=3):
    return round(float(v), d)


# ============================ Pasos del diagnostico ============================
def paso0_ortesis(eventos, marcadores, salida):
    ack, ini = resumen_pasos(marcadores)
    seqs = sorted(set(ack) & set(ini))
    tipos = Counter(e['evento'].get('tipo') for e in eventos)
    cp = [e['evento']['texto'] for e in eventos if e['evento'].get('tipo') == 'checkpoint']
    r = {'pasos': len(seqs), 'eventos_por_tipo': dict(tipos), 'mano_virtual_eventos': tipos.get(config.MANO_VIRTUAL['evento'], 0),
         'eventos_de_lazo': tipos.get('paso', 0) + tipos.get('ajeno', 0), 'checkpoints': cp}
    if len(seqs) >= 10:
        r.update(ortesis_parece_simulada(seqs, [ini[s] - ack[s] for s in seqs]))
    salida['ortesis'] = r
    print('\n[0] Que ortesis uso la sesion y si el movimiento se veia')
    for c in cp:
        print('    ' + c)
    print(f"    eventos de Estado: {dict(tipos)} -> eventos `mano` (mano virtual): {r['mano_virtual_eventos']}; "
          f"eventos de lazo (paso/ajeno): {r['eventos_de_lazo']}")
    sig = inspect.signature(hw.OrtesisSimulada.__init__).parameters
    lat0, jit0 = sig['latencia_ms'].default, sig['jitter_ms'].default
    print(f"    (para comparar con el CP1: OrtesisSimulada con latencia {lat0:g} ms y jitter {jit0:g} ms da en teoria mediana {lat0:.1f}, MAD {0.6745 * jit0:.1f} y p95 {lat0 + 1.645 * jit0:.1f} ms; "
          'compatible, pero no basta: un ACK real tambien puede verse asi)')
    if r.get('simulada') is not None:
        lo, med, hi = r['latencia_ms']
        print(f"    latencia ACK -> inicio de {r['pasos']} pasos: {lo:.0f} / {med:.0f} / {hi:.0f} ms (min / mediana / max; la simulada va de "
              f"{config.LATENCIA_MECANICA_SIM_MS[0]:.0f} a {config.LATENCIA_MECANICA_SIM_MS[1]:.0f})")
        print(f"    correlacion con latencia_mecanica_simulada(seq, semilla={r['semilla']}): r = {r['r']:.4f} "
              f"(desfase fijo {r['desfase_mediano_ms']:.1f} ms, sd {r['desfase_sd_ms']:.1f}); con las otras 19 semillas |r| <= {r['r_otras_max_abs']:.3f}")
        print(f"    -> ortesis SIMULADA: {r['simulada']}" + ('; y ningun evento `mano`: no hay registro de nada que el piloto pudiera ver moverse'
                                                            if r['simulada'] and not r['mano_virtual_eventos'] else ''))
    else:
        print('    (sin pasos suficientes en el jsonl para decidirlo)')


def paso1_senal(X, y, direccion, estratos, a, rng, salida):
    n1, n0 = int(y.sum()), int((1 - y).sum())
    print(f'\n[1] Hay senal? ({n1} movimientos erroneos, {n0} correctos; epoca -0.2 a 0.8 s, filtro {config.BANDA_ERRP[0]:g}-{config.BANDA_ERRP[1]:g} Hz, '
          f'linea base de 200 ms; {a.perm} permutaciones dentro de bloque x cue)')
    obs, bajo, alto, d_boot = bootstrap_onda_diferencia(X, y, a.boot, rng)
    tt = np.arange(X.shape[2]) / FS - ANTES
    r = {'n_error': n1, 'n_correcto': n0}
    # pico de la onda diferencia en Fz/Cz/Pz (para describirla; la prueba es la de abajo)
    r['picos'] = {}
    for c in config.PAPELES['errp']:
        w = obs[config.CANALES_EEG.index(c)]
        sel = (tt >= 0.15) & (tt <= 0.60)
        i_min, i_max = np.argmin(np.where(sel, w, np.inf)), np.argmax(np.where(sel, w, -np.inf))
        r['picos'][c] = {'min_uv': _f(w[i_min]), 'min_ms': int(round(tt[i_min] * 1e3)), 'max_uv': _f(w[i_max]), 'max_ms': int(round(tt[i_max] * 1e3))}
        print(f"    {c}: en 150-600 ms la onda diferencia va de {w[i_min]:+.2f} uV (a {tt[i_min] * 1e3:.0f} ms) a {w[i_max]:+.2f} uV (a {tt[i_max] * 1e3:.0f} ms)")
    # ventanas fijadas de antemano: Ne y Pe en Fz y Cz
    F_ap, nombres = [], []
    for nombre, (d, h) in A_PRIORI.items():
        for c in CANALES_A_PRIORI:
            F_ap.append(X[:, config.CANALES_EEG.index(c), int(round((ANTES + d) * FS)):int(round((ANTES + h) * FS))].mean(axis=1))
            nombres.append((nombre, c, d, h))
    F_ap = np.array(F_ap).T
    pp = prueba_permutacion(F_ap, y, estratos, a.perm, rng)
    dc = d_de_cohen(F_ap, y)
    r['a_priori'] = []
    print('    ventanas fijadas de antemano (Ne 200-350 ms y Pe 300-500 ms en Fz y Cz; error - correcto):')
    for k, (nombre, c, d, h) in enumerate(nombres):
        i0, i1 = int(round((ANTES + d) * FS)), int(round((ANTES + h) * FS))
        ci = np.percentile(d_boot[:, config.CANALES_EEG.index(c), i0:i1].mean(axis=1), [2.5, 97.5])
        dif = float(F_ap[y == 1, k].mean() - F_ap[y == 0, k].mean())
        r['a_priori'].append({'ventana': nombre, 'canal': c, 'dif_uv': _f(dif), 'ic95': [_f(ci[0]), _f(ci[1])], 'd': _f(dc[k]),
                              't': _f(pp['t'][k]), 'p': _f(pp['p_sin_corregir'][k]), 'p_corregido': _f(pp['p_corregido'][k])})
        print(f"      {nombre} {c}: {dif:+.2f} uV, IC95 bootstrap [{ci[0]:+.2f}, {ci[1]:+.2f}], d = {dc[k]:+.2f}, t = {pp['t'][k]:+.2f}, "
              f"p = {pp['p_sin_corregir'][k]:.3f}, p corregido (max |t| de las 4) = {pp['p_corregido'][k]:.3f}")
    # todas las ventanas de 50 ms x 8 canales
    W, inicios = medias_por_ventana(X, *VENTANAS_T)
    pt = prueba_permutacion(W.reshape(len(y), -1), y, estratos, a.perm, rng)
    k_mejor = int(np.argmax(np.abs(pt['t'])))
    c_mejor, v_mejor = divmod(k_mejor, W.shape[2])
    r['todas'] = {'caracteristicas': int(W.shape[1] * W.shape[2]), 'max_abs_t': _f(np.abs(pt['t']).max()),
                  'canal': config.CANALES_EEG[c_mejor], 'ventana_ms': int(round(inicios[v_mejor] * 1e3)),
                  'p_corregido_min': _f(pt['p_corregido'].min()), 'p_sin_corregir_min': _f(pt['p_sin_corregir'].min()),
                  'pasan_corregido': int((pt['p_corregido'] < 0.05).sum()), 'pasan_sin_corregir': int((pt['p_sin_corregir'] < 0.05).sum()),
                  'n_t_mayor_2': pt['n_sobre_umbral'], 'n_t_mayor_2_nula': _f(pt['n_sobre_umbral_nula'], 1), 'p_cuenta': _f(pt['p_cuenta']),
                  'max_t_nula_p95': _f(np.percentile(pt['max_nula'], 95)), 'max_t_nula': pt['max_nula'][:2000].tolist()}
    print(f"    las {r['todas']['caracteristicas']} medias de 50 ms (8 canales x 12 ventanas de 100 a 700 ms): mayor |t| = {r['todas']['max_abs_t']:.2f} "
          f"({r['todas']['canal']}, ventana desde {r['todas']['ventana_ms']} ms); lo que alcanza el 5 % de la nula del maximo: {r['todas']['max_t_nula_p95']:.2f}")
    print(f"      pasan con correccion (p < 0.05): {r['todas']['pasan_corregido']}; sin corregir: {r['todas']['pasan_sin_corregir']} "
          f"(esperadas por azar sin corregir: {0.05 * r['todas']['caracteristicas']:.1f}); |t| > 2: {r['todas']['n_t_mayor_2']} "
          f"contra {r['todas']['n_t_mayor_2_nula']} de la nula (p = {r['todas']['p_cuenta']:.2f})")
    # cuanto efecto podia ver esta muestra
    sd = float(np.sqrt(np.mean([((n1 - 1) * F_ap[y == 1, k].var(ddof=1) + (n0 - 1) * F_ap[y == 0, k].var(ddof=1)) / (n1 + n0 - 2)
                                for k in range(F_ap.shape[1])])))
    r['efecto_minimo'] = {'sd_uv': _f(sd)}
    print(f'    desviacion entre ensayos de la media de una ventana a priori: {sd:.2f} uV')
    for ne in (n1, 60, 90, 120):
        nc = int(round(ne * n0 / n1))
        r['efecto_minimo'][str(ne)] = {'n_correcto': nc, 'uv_alfa05': _f(efecto_minimo_detectable(sd, ne, nc)),
                                       'uv_alfa0125': _f(efecto_minimo_detectable(sd, ne, nc, alfa=0.05 / 4)),
                                       'd_alfa05': _f(efecto_minimo_detectable(1.0, ne, nc))}
        e = r['efecto_minimo'][str(ne)]
        print(f"      con {ne:3d} errores y {nc:3d} correctos ({ne + nc} epocas al {100 * n1 / (n1 + n0):.0f} %): efecto minimo con 80 % de potencia "
              f"{e['uv_alfa05']:.2f} uV (d = {e['d_alfa05']:.2f}) a nivel 0.05; {e['uv_alfa0125']:.2f} uV con la correccion de 4 pruebas")
    # fiabilidad entre mitades de la onda diferencia en Fz/Cz/Pz (100-700 ms)
    c3 = config.indices('errp')
    sel = slice(int(round((ANTES + 0.1) * FS)), int(round((ANTES + 0.7) * FS)))
    mitad = np.arange(len(y)) < len(y) // 2

    def r_mitades(yy):
        d1 = (X[mitad & (yy == 1)][:, c3, sel].mean(0) - X[mitad & (yy == 0)][:, c3, sel].mean(0)).ravel()
        d2 = (X[~mitad & (yy == 1)][:, c3, sel].mean(0) - X[~mitad & (yy == 0)][:, c3, sel].mean(0)).ravel()
        return float(np.corrcoef(d1, d2)[0, 1])
    r_obs = r_mitades(y)
    nula = np.array([r_mitades(permutar_en_estratos(y, estratos, rng)) for _ in range(min(a.perm, 2000))])
    r['mitades'] = {'r': _f(r_obs), 'nula_media': _f(nula.mean()), 'nula_p95': _f(np.percentile(nula, 95)),
                    'p': _f((1 + (nula >= r_obs).sum()) / (1 + len(nula)))}
    print(f"    fiabilidad entre mitades (ondas diferencia de Fz/Cz/Pz, 100-700 ms; ensayos 1-60 contra 61-120): r = {r_obs:+.3f}; "
          f"nula {nula.mean():+.3f} (p95 {np.percentile(nula, 95):+.3f}), p = {r['mitades']['p']:.3f}")
    salida['senal'] = r
    return obs, bajo, alto


def paso1c_preprocesamiento(xdf, t0s, y, estratos, a, rng, salida):
    """La misma prueba de permutacion (96 medias de 50 ms, correccion por el maximo) con otros preprocesamientos: si la senal
    se hubiera escondido por el filtro, la linea base o la referencia, alguno de estos la mostraria."""
    print('\n[1c] Sensibilidad al preprocesamiento (96 medias de 50 ms de 100 a 700 ms; max |t| y p corregido por permutacion)')
    x, t = xdf['x'], xdf['t']
    variantes = [('la de config (1-10 Hz, linea base de 200 ms)', config.BANDA_ERRP, True, False),
                 ('sin linea base', config.BANDA_ERRP, False, False),
                 ('0.5-20 Hz', (0.5, 20.0), True, False),
                 ('1-6 Hz', (1.0, 6.0), True, False),
                 ('referencia al promedio de los 8 canales', config.BANDA_ERRP, True, True)]
    r = []
    for nombre, banda, linea_base, car in variantes:
        tramos = preparar_continuo(x - x.mean(0, keepdims=True) if car else x, t, banda=banda)
        W, inicios = medias_por_ventana(cortar_todas(tramos, t0s, linea_base=linea_base), *VENTANAS_T)
        pt = prueba_permutacion(W.reshape(len(y), -1), y, estratos, min(a.perm, 2000), rng)
        k = int(np.argmax(np.abs(pt['t'])))
        c, v = divmod(k, W.shape[2])
        r.append({'variante': nombre, 'max_abs_t': _f(np.abs(pt['t']).max()), 'canal': config.CANALES_EEG[c], 'ventana_ms': int(round(inicios[v] * 1e3)),
                  'p_corregido_min': _f(pt['p_corregido'].min())})
        print(f"    {nombre:48s} mayor |t| = {r[-1]['max_abs_t']:.2f} ({r[-1]['canal']}, desde {r[-1]['ventana_ms']} ms); p corregido = {r[-1]['p_corregido_min']:.3f}")
    salida['preprocesamiento'] = r


def paso2_desfases(tramos, t0s, y, estratos, a, rng, salida):
    print(f'\n[2] Desfase de la epoca (de {DESFASES_S[0] * 1e3:+.0f} a {DESFASES_S[-1] * 1e3:+.0f} ms del inicio del movimiento; '
          f'AUC de validacion cruzada 5 pliegues x 2, LDA con encogimiento, ventanas de 100 ms de 100 a 700 ms; {a.perm_auc} permutaciones)')
    r = {}
    for nombre, canales in (('Fz/Cz/Pz', config.indices('errp')), ('8 canales', list(range(8)))):
        e = escanear_desfases(tramos, t0s, y, estratos, DESFASES_S, canales, a.perm_auc, rng)
        r[nombre] = e
        i = int(np.argmax(e['auc']))
        print(f"    {nombre} ({e['n_caracteristicas']} caracteristicas): AUC maximo {e['auc'][i]:.3f} con desfase {e['desfases_s'][i] * 1e3:+.0f} ms; "
              f"en desfase 0: {e['auc'][list(DESFASES_S).index(0.0)]:.3f}" + (
                  f"; nula media {np.mean(e['nula_media']):.3f}, p95 del maximo sobre desfases {e['nula_max_p95']:.3f}; "
                  f"p corregido del maximo = {e['p_corregido'][i]:.3f} (sin corregir {e['p_sin_corregir'][i]:.3f})" if 'nula_media' in e else ''))
    salida['desfases'] = r


def paso3_calidad(X, y, estratos, xdf, t0s, cues_errp, a, rng, salida):
    from scipy.signal import welch
    print('\n[3] Calidad y fatiga')
    r = {}
    ptp = np.ptp(X, axis=2).max(axis=1)
    rms_epoca = np.sqrt((X ** 2).mean(axis=(1, 2)))
    r['rms_por_canal_uv'] = {c: _f(v, 2) for c, v in zip(config.CANALES_EEG, np.sqrt((X ** 2).mean(axis=(0, 2))))}
    r['ptp_epoca_uv'] = {'mediana': _f(np.median(ptp), 1), 'p95': _f(np.percentile(ptp, 95), 1), 'max': _f(ptp.max(), 1)}
    sobre = ptp > 100.0                                    # DetectorErrP.umbral_amp
    r['epocas_sobre_100uv'] = {'error': int(sobre[y == 1].sum()), 'correcto': int(sobre[y == 0].sum()),
                               'p_fisher': _f(fisher_2x2(int(sobre[y == 1].sum()), int((y == 1).sum() - sobre[y == 1].sum()),
                                                         int(sobre[y == 0].sum()), int((y == 0).sum() - sobre[y == 0].sum())))}
    print('    rms por canal de las epocas (1-10 Hz, uV): ' + ', '.join(f'{c} {v:.1f}' for c, v in r['rms_por_canal_uv'].items()))
    print(f"    pico a pico maximo por epoca: mediana {r['ptp_epoca_uv']['mediana']} uV, p95 {r['ptp_epoca_uv']['p95']}, max {r['ptp_epoca_uv']['max']}; "
          f"epocas sobre 100 uV: {r['epocas_sobre_100uv']['error']} de errores y {r['epocas_sobre_100uv']['correcto']} de correctos (Fisher p = {r['epocas_sobre_100uv']['p_fisher']:.2f})")
    # primera contra segunda mitad: rms y deriva de la amplitud con el ensayo
    from scipy.stats import spearmanr
    rho, p_rho = spearmanr(np.arange(len(y)), rms_epoca)
    r['rms_vs_ensayo'] = {'rho': _f(rho), 'p': _f(p_rho), 'primera_mitad_mediana_uv': _f(np.median(rms_epoca[:60]), 2),
                          'segunda_mitad_mediana_uv': _f(np.median(rms_epoca[60:]), 2)}
    print(f"    rms de la epoca (mediana): primeros 60 {np.median(rms_epoca[:60]):.2f} uV, ultimos 60 {np.median(rms_epoca[60:]):.2f} uV "
          f"(Spearman con el numero de ensayo {rho:+.2f}, p = {p_rho:.2f})")
    # AUC por mitad con su nula (ventanas de 100 ms, Fz/Cz/Pz y 8 canales)
    for nombre, canales in (('Fz/Cz/Pz', config.indices('errp')), ('8 canales', list(range(8)))):
        W, _ = medias_por_ventana(X[:, canales, :], *VENTANAS_AUC)
        F = W.reshape(len(y), -1)
        mitad = np.arange(len(y)) < 60
        f = lambda yy: (auc_cv(F[mitad], yy[mitad]), auc_cv(F[~mitad], yy[~mitad]))
        obs = f(y)
        nula = np.array([f(permutar_en_estratos(y, estratos, rng)) for _ in range(a.perm_auc)])
        r.setdefault('auc_por_mitad', {})[nombre] = {'primera': _f(obs[0]), 'segunda': _f(obs[1]), 'nula_media': _f(nula.mean()), 'nula_p95': _f(np.percentile(nula, 95)),
                                                     'p_primera': _f((1 + (nula[:, 0] >= obs[0]).sum()) / (1 + len(nula))),
                                                     'p_segunda': _f((1 + (nula[:, 1] >= obs[1]).sum()) / (1 + len(nula)))}
        e = r['auc_por_mitad'][nombre]
        print(f"    AUC {nombre}, primera mitad {e['primera']:.3f} (p = {e['p_primera']:.2f}), segunda {e['segunda']:.3f} (p = {e['p_segunda']:.2f}); nula media {e['nula_media']:.3f}, p95 {e['nula_p95']:.3f}")
    if xdf is not None:
        x, t, imu = xdf['x'], xdf['t'], xdf['imu']
        giro = giro_maximo(imu, t, t0s)
        r['giro_epoca_dps'] = {'mediana': _f(np.median(giro), 1), 'p95': _f(np.percentile(giro, 95), 1), 'max': _f(giro.max(), 1),
                               'sobre_umbral': int((giro > config.GIRO_ARTEFACTO_DPS).sum())}
        print(f"    giro maximo de la cabeza en cada epoca: mediana {r['giro_epoca_dps']['mediana']} grados/s, p95 {r['giro_epoca_dps']['p95']}, max {r['giro_epoca_dps']['max']}; "
              f"sobre {config.GIRO_ARTEFACTO_DPS:.0f}: {r['giro_epoca_dps']['sobre_umbral']}")
        # alfa occipital por bloque de 20 ensayos (cada bloque: del cue de su primer ensayo al ultimo movimiento)
        alfa = []
        for b in range(len(y) // TAM_BLOQUE):
            i0, i1 = b * TAM_BLOQUE, (b + 1) * TAM_BLOQUE - 1
            j0, j1 = np.searchsorted(t, [cues_errp[i0], t0s[i1] + 1.0])
            alfa.append(hw.potencia_alfa(x[:, j0:j1], FS))
        r['alfa_por_bloque'] = [_f(v / alfa[0], 2) for v in alfa]
        print('    alfa occipital (8-13 Hz, PO7/Oz/PO8) de cada bloque de 20 ensayos entre el del primer bloque: ' + ', '.join(f'{v:.2f}' for v in r['alfa_por_bloque'])
              + f" (el semaforo PILOTO avisa desde {config.SALUD['piloto_amarillo']})")
        # 60 Hz y deriva
        f_, P = welch(x, fs=FS, nperseg=int(4 * FS))
        red = P[:, (f_ > 58) & (f_ < 62)].sum(1) / P[:, (f_ > 1) & (f_ < 100)].sum(1)
        r['fraccion_60hz'] = {c: _f(v, 2) for c, v in zip(config.CANALES_EEG, red)}
        print('    fraccion de potencia en 58-62 Hz sobre 1-100 Hz de todo el registro: ' + ', '.join(f'{c} {v:.2f}' for c, v in r['fraccion_60hz'].items())
              + f" (umbral del CP1: {config.CP1_RED['umbral']}; el filtro de 1-10 Hz y el notch la quitan casi toda)")
    salida['calidad'] = r


def paso4_diseno(y, direccion, eventos, marcadores, salida):
    print('\n[4] Diseno')
    r = {'errores': int(y.sum()), 'epocas': int(len(y)), 'proporcion_error': _f(y.mean()),
         'errores_por_bloque': [int(y[i:i + TAM_BLOQUE].sum()) for i in range(0, len(y), TAM_BLOQUE)],
         'errores_por_direccion': {'cerrar': int(y[direccion == 1].sum()), 'abrir': int(y[direccion == 0].sum())}}
    seguidos = 0
    mejor = 0
    for v in y:
        seguidos = seguidos + 1 if v else 0
        mejor = max(mejor, seguidos)
    r['racha_maxima_de_errores'] = mejor
    print(f"    {r['errores']} errores en {r['epocas']} epocas ({100 * y.mean():.0f} %); por bloque de {TAM_BLOQUE}: {r['errores_por_bloque']}; "
          f"cerrar {r['errores_por_direccion']['cerrar']} / abrir {r['errores_por_direccion']['abrir']}; racha maxima de errores: {mejor}")
    cues = [m for m in marcadores if m['marcador'] in (config.CUE_CERRAR, config.CUE_RELAJA)][-len(y):]
    ack, ini = resumen_pasos(marcadores)
    seqs = sorted(set(ack) & set(ini))
    if len(cues) == len(y) and len(seqs) == len(y):
        c = np.array([m['t'] for m in cues])
        c_a = np.array([ack[s] for s in seqs]) - c
        a_i = np.array([ini[s] - ack[s] for s in seqs])
        r['tiempos_s'] = {'cue_a_ack': [_f(c_a.min()), _f(np.median(c_a)), _f(c_a.max())], 'periodo_ensayo_mediana': _f(np.median(np.diff(c))),
                          'ack_a_inicio_ms': [_f(a_i.min() * 1e3, 0), _f(np.median(a_i) * 1e3, 0), _f(a_i.max() * 1e3, 0)]}
        margen = np.median(np.diff(c)) - (np.median(c_a) + np.median(a_i) + DESPUES)
        r['tiempos_s']['cue_siguiente_tras_la_epoca'] = _f(margen)
        print(f"    cue -> ACK {c_a.min():.2f}-{c_a.max():.2f} s; ACK -> inicio {a_i.min() * 1e3:.0f}-{a_i.max() * 1e3:.0f} ms; un ensayo cada {np.median(np.diff(c)):.2f} s; "
              f"el cue siguiente llega {margen:.2f} s despues del final de la epoca (medianas)")
    visual = [e['evento'] for e in eventos if e['evento'].get('tipo') == 'cue']
    r['cues'] = {'n': len(visual), 'con_visual_false': sum(1 for e in visual if e.get('visual') is False)}
    if len(visual) >= len(y):                          # las etiquetas: error = el movimiento fue contra el cue (direccion = cue XOR error)
        meta = np.array([e['meta'] for e in visual[-len(y):]])
        r['etiquetas_coherentes_con_los_cues'] = bool(np.array_equal((meta == 1).astype(int), direccion ^ y))
        print(f"    las etiquetas (error = el movimiento fue contra el cue) {'coinciden' if r['etiquetas_coherentes_con_los_cues'] else 'NO COINCIDEN'} con los cues grabados en Estado")
    print(f"    cues en Estado: {r['cues']['n']}, de ellos sin senal visual (--cue-sin-visual): {r['cues']['con_visual_false']} -> la meta se mostro en pantalla")
    print(f'    movimiento de la calibracion: +-0.15 del recorrido en {config.DURACION_PASO_MS} ms (PASO_VISIBLE = {config.PASO_VISIBLE}); '
          'el paso era grande: la pregunta es si se VIO (paso 0), no su tamano')
    salida['diseno'] = r


def amplitud_pico_a_pico(tramos, tiempos, canales, ventana=(0.05, 0.40), fs=FS, antes=ANTES):
    """Pico a pico (uV) del promedio de las epocas alrededor de `tiempos`, promediando `canales`, dentro de `ventana`
    (s tras cada evento). Una respuesta evocada de fase fija sube; el ruido de fondo solo la deja en su piso."""
    E = np.array([cortar(tramos, t0) for t0 in tiempos])
    w = E.mean(axis=0)[canales].mean(axis=0)[int(round((antes + ventana[0]) * fs)):int(round((antes + ventana[1]) * fs))]
    return float(w.max() - w.min())


def paso4b_visibilidad(tramos, t0s, cues_errp, a, rng, salida):
    """Control de visibilidad POR EEG (debil, mirar con cuidado): si el piloto hubiera visto moverse algo, el promedio de los 120
    movimientos deberia dejar una respuesta occipital (PO7/Oz/PO8) sobre el piso del ruido. El piso se saca de 120 instantes al
    azar dentro de la misma fase. El cue (una palabra que cambia por otra del mismo color y tamano) casi no cambia la luminancia:
    es un control positivo pobre, asi que el resultado NO distingue 'no se vio' de 'no se midio'."""
    occ = config.indices('visual')
    mov, cue = amplitud_pico_a_pico(tramos, t0s, occ), amplitud_pico_a_pico(tramos, cues_errp, occ)
    lo, hi = cues_errp[0] + 0.5, t0s[-1] + 0.5
    nula = np.array([amplitud_pico_a_pico(tramos, rng.uniform(lo, hi, len(t0s)), occ) for _ in range(a.pseudo)])
    r = {'movimiento_uv': _f(mov), 'cue_uv': _f(cue), 'nula_media_uv': _f(nula.mean()), 'nula_p95_uv': _f(np.percentile(nula, 95)),
         'p_movimiento': _f((1 + (nula >= mov).sum()) / (1 + len(nula))), 'p_cue': _f((1 + (nula >= cue).sum()) / (1 + len(nula))), 'n_pseudo': len(nula)}
    salida['visibilidad_eeg'] = r
    print(f'\n[4b] Respuesta occipital promedio (PO7/Oz/PO8, 50-400 ms, pico a pico; {len(nula)} sorteos de {len(t0s)} instantes al azar en la misma fase)')
    pasan = [nombre for nombre, p_ in (('el movimiento', r['p_movimiento']), ('el cue', r['p_cue'])) if p_ < 0.05]
    print(f"    movimiento: {mov:.2f} uV (p = {r['p_movimiento']:.3f}); cue: {cue:.2f} uV (p = {r['p_cue']:.3f}); piso: media {nula.mean():.2f}, p95 {np.percentile(nula, 95):.2f} uV. "
          + (f"Pasa {' y '.join(pasan)} (p < 0.05); " if pasan else 'Ni uno ni otro pasan (p >= 0.05); ') + 'el control es pobre (ver la docstring): no prueba que no se vio')


def curva_de_aprendizaje(x, t, amplitud_uv, tamanos, repeticiones, rng, p_error=0.3, separacion_s=1.25, margen_s=15.0):
    """AUC de validacion cruzada (ventanas de 100 ms + LDA con encogimiento) con n epocas hechas del ruido de P001: n instantes al
    azar del registro entero con `separacion_s` entre ellos (que ni la epoca ni la onda inyectada se pisen), `p_error` de ellos con la
    onda del gemelo sumada. Dice cuanto mejora el AUC con mas epocas SI el ErrP de P001 fuera de esa amplitud (no lo sabemos).
    -> {n: (media y desviacion entre sorteos del AUC Fz/Cz/Pz, y lo mismo con 8 canales)}."""
    import cerebro_sintetico as cs
    casillas = np.arange(t[0] + margen_s, t[-1] - 2.0, separacion_s)
    salida = {}
    for n in tamanos:
        a3, a8 = [], []
        for _ in range(repeticiones):
            t0s = np.sort(rng.choice(casillas, size=n, replace=False))
            y = np.zeros(n, dtype=int)
            y[rng.choice(n, int(round(p_error * n)), replace=False)] = 1
            E = cortar_todas(preparar_continuo(inyectar_errp(x, t, t0s, y == 1, amplitud_uv, cs.W_ERRP), t), t0s)
            W, _ = medias_por_ventana(E, *VENTANAS_AUC)
            a3.append(auc_cv(W[:, config.indices('errp'), :].reshape(n, -1), y))
            a8.append(auc_cv(W.reshape(n, -1), y))
        salida[int(n)] = (float(np.mean(a3)), float(np.std(a3)), float(np.mean(a8)), float(np.std(a8)))
    return salida


def referencia_gemelo(amplitudes, semillas=(0, 1, 2, 3), n=120):
    """AUC de validacion cruzada (el MISMO clasificador de ventanas de 100 ms + LDA) en el banco offline del gemelo con
    esas amplitudes de ErrP (sin la theta, que este clasificador no mira): lo que el proyecto daba por bueno, para
    compararlo con la misma onda sumada al ruido real. -> {amplitud: (AUC Fz/Cz/Pz, AUC 8 canales)}, medias de las semillas."""
    import cerebro_sintetico as cs
    salida = {}
    for amp in amplitudes:
        a3, a8 = [], []
        for semilla in semillas:
            X, y = cs.sesion_errp(n, p_error=0.3, errp=amp, theta=0.0, semilla=semilla)
            W, _ = medias_por_ventana(X, *VENTANAS_AUC)
            a3.append(auc_cv(W[:, config.indices('errp'), :].reshape(n, -1), y))
            a8.append(auc_cv(W.reshape(n, -1), y))
        salida[float(amp)] = (float(np.mean(a3)), float(np.mean(a8)))
    return salida


def paso5_inyeccion(xdf, t0s, y, direccion, estratos, a, rng, salida):
    import cerebro_sintetico as cs
    print('\n[5] Control positivo: onda diferencia del gemelo sumada al EEG continuo de P001 en los pasos erroneos')
    print(f'    (forma del gemelo, pesos por canal {cs.W_ERRP.tolist()}; la amplitud es el argumento `--errp` del gemelo, por defecto 6: '
          f'pico de {onda_diferencia_gemelo(6.0)[1].max():.1f} uV y valle de {onda_diferencia_gemelo(6.0)[1].min():.1f} uV en el canal de peso 1; '
          'los ensayos correctos quedan IGUAL)')
    x, t = xdf['x'], xdf['t']
    c3 = config.indices('errp')
    ref = referencia_gemelo([amp for amp in a.amplitudes if amp in (0.0, 3.0, 6.0)])
    r = {'por_amplitud': {}, 'referencia_gemelo': {str(k): list(v) for k, v in ref.items()}}
    for amp in a.amplitudes:
        tramos = preparar_continuo(inyectar_errp(x, t, t0s, y == 1, amp, cs.W_ERRP), t)
        E = cortar_todas(tramos, t0s)
        W, _ = medias_por_ventana(E, *VENTANAS_T)
        pt = prueba_permutacion(W.reshape(len(y), -1), y, estratos, min(a.perm, 2000), rng)
        Wa, _ = medias_por_ventana(E[:, c3, :], *VENTANAS_AUC)
        auc3 = auc_cv(Wa.reshape(len(y), -1), y)
        Wb, _ = medias_por_ventana(E, *VENTANAS_AUC)
        auc8 = auc_cv(Wb.reshape(len(y), -1), y)
        r['por_amplitud'][str(amp)] = {'max_abs_t': _f(np.abs(pt['t']).max()), 'pasan_corregido': int((pt['p_corregido'] < 0.05).sum()),
                                       'p_corregido_min': _f(pt['p_corregido'].min()), 'auc_Fz_Cz_Pz': _f(auc3), 'auc_8canales': _f(auc8)}
        e = r['por_amplitud'][str(amp)]
        print(f"    amplitud {amp:4.1f} uV: mayor |t| = {e['max_abs_t']:5.2f}, ventanas que pasan con correccion {e['pasan_corregido']:2d} de 96 "
              f"(p corregido minimo {e['p_corregido_min']:.3f}); AUC Fz/Cz/Pz {auc3:.3f}, 8 canales {auc8:.3f}"
              + (f"   | gemelo, mismo clasificador: {ref[amp][0]:.3f} y {ref[amp][1]:.3f}" if amp in ref else ''))
    # el barrido de desfases tiene que encontrar un retraso conocido: el ErrP inyectado 150 ms DESPUES del inicio
    desfase = a.retraso_inyectado
    tramos = preparar_continuo(inyectar_errp(x, t, np.asarray(t0s) + desfase, y == 1, a.amplitud_retraso, cs.W_ERRP), t)
    e = escanear_desfases(tramos, t0s, y, estratos, DESFASES_S, c3, 0, rng)
    i = int(np.argmax(e['auc']))
    r['retraso'] = {'inyectado_ms': int(round(desfase * 1e3)), 'amplitud_uv': a.amplitud_retraso, 'mejor_desfase_ms': int(round(e['desfases_s'][i] * 1e3)),
                    'desfases_s': e['desfases_s'], 'auc': e['auc']}
    r['retraso']['auc_extremos'] = [_f(e['auc'][0]), _f(e['auc'][-1])]
    print(f"    barrido de desfases con el ErrP inyectado {desfase * 1e3:+.0f} ms despues del inicio ({a.amplitud_retraso:g} uV): AUC maximo {e['auc'][i]:.3f} con el desfase "
          f"{e['desfases_s'][i] * 1e3:+.0f} ms; en los extremos ({e['desfases_s'][0] * 1e3:+.0f} y {e['desfases_s'][-1] * 1e3:+.0f} ms) {e['auc'][0]:.3f} y {e['auc'][-1]:.3f}. "
          'Las ventanas de 100 ms cubren 600 ms: el maximo cae en una meseta ancha alrededor del retraso, no en un punto')
    r['curva_de_aprendizaje'] = {}
    for amp in a.amplitudes_curva:
        curva = curva_de_aprendizaje(x, t, amp, a.tamanos_curva, a.repeticiones_curva, rng)
        r['curva_de_aprendizaje'][str(amp)] = {str(n): list(v) for n, v in curva.items()}
        print(f'    mas epocas con ruido de P001 (instantes al azar de todo el registro, 30 % con {amp:g} uV inyectados; media +- sd de {a.repeticiones_curva} sorteos): '
              + '; '.join(f'{n} epocas: AUC {v[0]:.3f} +- {v[1]:.3f} (Fz/Cz/Pz) y {v[2]:.3f} +- {v[3]:.3f} (8 canales)' for n, v in curva.items()))
    if not a.sin_detector:
        r['detector_inyectado'] = {}
        for amp in a.amplitudes_detector:
            tramos = preparar_continuo(inyectar_errp(x, t, t0s, y == 1, amp, cs.W_ERRP), t)
            t_ini = time.time()
            d = hw.DetectorErrP().ajustar(cortar_todas(tramos, t0s), y, config.candidatos('detector'), direccion=direccion)
            r['detector_inyectado'][str(amp)] = {'sens': _f(d.sens, 2), 'espec': _f(d.espec, 2), 'ba': _f(d.ba, 2), 'eleccion': d.eleccion,
                                                 'puntajes': {k: _f(v) for k, v in d.puntajes.items()}}
            print(f"    DetectorErrP completo (el del CP3) con {amp:g} uV inyectados: sens {d.sens:.2f}, espec {d.espec:.2f}, BA {d.ba:.2f} ({d.eleccion}); "
                  'AUC de las 4 configuraciones ' + ', '.join(f'{k} {v:.2f}' for k, v in d.puntajes.items()) + f'   [{time.time() - t_ini:.0f} s]')
    salida['inyeccion'] = r


def paso_detector_nula(X, y, direccion, estratos, a, rng, salida):
    """Las 4 AUC de validacion cruzada con que DetectorErrP elige canales y vistas, con las etiquetas reales y permutadas."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    print(f'\n[1b] El propio DetectorErrP contra la nula ({a.perm_detector} permutaciones dentro de bloque x cue; las 4 configuraciones de config.candidatos)')
    det = hw.DetectorErrP()
    cand = config.candidatos('detector')
    cv = StratifiedKFold(4, shuffle=True, random_state=0)

    def aucs(yy):
        return {n: float(roc_auc_score(yy, cross_val_predict(det._pipe(*cand[n]), X, yy, method='predict_proba', cv=cv)[:, 1])) for n in cand}
    t0 = time.time()
    obs = aucs(y)
    r = {'auc': {n: _f(v) for n, v in obs.items()}, 'maximo_observado': _f(max(obs.values())), 'n_perm': a.perm_detector}
    print('    AUC con las etiquetas reales: ' + ', '.join(f'{n} {v:.3f}' for n, v in obs.items()) + f'   [{time.time() - t0:.0f} s]')
    if a.perm_detector > 0:
        maximos, todas = [], []
        for _ in range(a.perm_detector):
            p = aucs(permutar_en_estratos(y, estratos, rng))
            maximos.append(max(p.values()))
            todas.extend(p.values())
        maximos, todas = np.array(maximos), np.array(todas)
        r.update({'nula_media': _f(todas.mean()), 'nula_sd': _f(todas.std()), 'nula_maximo_media': _f(maximos.mean()), 'nula_maximo_p95': _f(np.percentile(maximos, 95)),
                  'p_maximo': _f((1 + (maximos >= max(obs.values())).sum()) / (1 + len(maximos)))})
        print(f"    con etiquetas permutadas: AUC media {todas.mean():.3f} (sd {todas.std():.3f}); el MAXIMO de las 4 (lo que elige el detector) {maximos.mean():.3f} de media, "
              f"p95 {np.percentile(maximos, 95):.3f}; el observado ({max(obs.values()):.3f}) tiene p = {r['p_maximo']:.2f}")
    salida['detector_nula'] = r


# ============================ Figura ============================
def figura(salida, obs, bajo, alto, ruta=FIGURA):
    """Cuatro paneles (colores: los tres primeros de la paleta categorica del proyecto de graficas, que validan entre si):
    evidencia de la ortesis, onda diferencia, barrido de desfases y control positivo."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    superficie, tinta, tinta2, rejilla, gris = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e1', '#8b8a85'
    azul, naranja, agua = '#2a78d6', '#eb6834', '#1baf7a'
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True, facecolor=superficie)
    for a_ in ax.ravel():
        a_.set_facecolor(superficie)
        a_.grid(True, color=rejilla, lw=0.8)
        a_.set_axisbelow(True)
        a_.spines[['top', 'right']].set_visible(False)
        a_.spines[['left', 'bottom']].set_color(gris)
        a_.tick_params(colors=tinta2)
    # (a) la latencia de la sesion contra la de OrtesisSimulada
    o = salida.get('ortesis', {})
    if o.get('simulada') is not None and 'marcadores_lat' in salida:
        seqs, lat = salida['marcadores_lat']
        sim = np.array([hw.latencia_mecanica_simulada(q, o['semilla']) for q in seqs]) * 1e3
        a_ = ax[0, 0]
        a_.scatter(sim, np.array(lat) * 1e3, s=30, color=azul, edgecolor=superficie, linewidth=1.0, zorder=3)
        lim = np.array([25.0, 160.0])
        a_.plot(lim, lim + o['desfase_mediano_ms'], color=gris, lw=1.2, zorder=2)
        a_.text(0.04, 0.95, f"r = {o['r']:.3f} con la semilla {o['semilla']}\n|r| \u2264 {o['r_otras_max_abs']:.2f} con las otras 19 semillas",
                transform=a_.transAxes, va='top', color=tinta, fontsize=9.5)
        a_.set_xlabel('lat. ACK \u2192 inicio de OrtesisSimulada (ms)', color=tinta2)
        a_.set_ylabel('lat. ACK \u2192 inicio de la sesi\u00f3n (ms)', color=tinta2)
        a_.set_title('(a) La ortesis de la sesi\u00f3n fue la simulada', color=tinta, fontsize=11, loc='left')
    # (b) onda diferencia con IC95 bootstrap por muestra
    a_ = ax[0, 1]
    tt = (np.arange(obs.shape[1]) / FS - ANTES) * 1e3
    for c, color in (('Fz', azul), ('Cz', naranja), ('Pz', agua)):
        i = config.CANALES_EEG.index(c)
        a_.fill_between(tt, bajo[i], alto[i], color=color, alpha=0.14, lw=0)
        a_.plot(tt, obs[i], color=color, lw=1.8, label=c)
    a_.axhline(0, color=gris, lw=1.0)
    a_.axvline(0, color=gris, lw=1.0, ls=':')
    a_.set_xlabel('ms desde el inicio del movimiento', color=tinta2)
    a_.set_ylabel('error \u2212 correcto (\u00b5V)', color=tinta2)
    a_.set_title('(b) Onda diferencia, IC95 bootstrap por muestra', color=tinta, fontsize=11, loc='left')
    a_.legend(frameon=False, labelcolor=tinta2)
    # (c) barrido de desfases: P001, nula y el mismo barrido con un ErrP conocido
    d3, d8 = salida.get('desfases', {}).get('Fz/Cz/Pz'), salida.get('desfases', {}).get('8 canales')
    if d3 and d8:
        a_ = ax[1, 0]
        x = np.array(d3['desfases_s']) * 1e3
        if 'nula_p95' in d3:
            a_.fill_between(x, d3['nula_media'], d3['nula_p95'], color=gris, alpha=0.22, lw=0, label='nula (Fz/Cz/Pz): media a p95')
            a_.axhline(max(d3['nula_max_p95'], d8.get('nula_max_p95', 0)), color=gris, lw=1.2, ls='--', label='p95 del m\u00e1ximo de la nula')
        a_.plot(x, d3['auc'], color=azul, lw=1.8, label='P001, Fz/Cz/Pz')
        a_.plot(x, d8['auc'], color=naranja, lw=1.8, label='P001, 8 canales')
        ret = salida.get('inyeccion', {}).get('retraso')
        if ret:
            a_.plot(np.array(ret['desfases_s']) * 1e3, ret['auc'], color=agua, lw=1.8, ls='--',
                    label=f"control: {ret['amplitud_uv']:g} \u00b5V inyectados a +{ret['inyectado_ms']} ms")
        a_.axhline(0.5, color=gris, lw=0.8)
        a_.set_xlabel('desfase de la \u00e9poca respecto al inicio (ms)', color=tinta2)
        a_.set_ylabel('AUC de validaci\u00f3n cruzada', color=tinta2)
        a_.set_title('(c) Barrido de desfases', color=tinta, fontsize=11, loc='left')
        a_.set_ylim(0.3, 1.0)
        a_.legend(frameon=False, fontsize=8, labelcolor=tinta2, loc='upper left')
    # (d) control positivo contra la amplitud
    inj = salida.get('inyeccion', {}).get('por_amplitud')
    if inj:
        a_ = ax[1, 1]
        amps = sorted(inj, key=float)
        xa = [float(v) for v in amps]
        a_.plot(xa, [inj[v]['auc_Fz_Cz_Pz'] for v in amps], 'o-', color=azul, lw=1.8, ms=6, label='P001 + onda, Fz/Cz/Pz')
        a_.plot(xa, [inj[v]['auc_8canales'] for v in amps], 's-', color=naranja, lw=1.8, ms=6, label='P001 + onda, 8 canales')
        ref = salida.get('inyeccion', {}).get('referencia_gemelo')
        if ref:
            xr = sorted(ref, key=float)
            a_.plot([float(v) for v in xr], [ref[v][0] for v in xr], 'D--', color=agua, lw=1.8, ms=6, label='gemelo + onda, Fz/Cz/Pz')
        a_.axhline(0.5, color=gris, lw=0.8)
        a_.set_xlabel('amplitud de la onda del gemelo sumada (\u00b5V)', color=tinta2)
        a_.set_ylabel('AUC de validaci\u00f3n cruzada', color=tinta2)
        a_.set_title('(d) Control positivo en el EEG de P001', color=tinta, fontsize=11, loc='left')
        a_.set_ylim(0.3, 1.0)
        a_.legend(frameon=False, fontsize=8, labelcolor=tinta2, loc='upper left')
    fig.suptitle('Diagn\u00f3stico del ErrP de P001 (EXPLORATORIO: un participante, una sesi\u00f3n)', color=tinta, fontsize=12, x=0.01, ha='left')
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=140, facecolor=superficie)
    plt.close(fig)
    return ruta


# ============================ Principal ============================
def _json(o):
    if isinstance(o, dict):
        return {str(k): _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--xdf', help='el XDF de LabRecorder (si falta, se omiten los pasos 2, 3 parcial y 5)')
    ap.add_argument('--npz', default=str(NPZ_ERRP))
    ap.add_argument('--jsonl', default=str(JSONL))
    ap.add_argument('--semilla', type=int, default=0)
    ap.add_argument('--perm', type=int, default=5000, help='permutaciones de las pruebas de ventanas (max |t|)')
    ap.add_argument('--perm-auc', dest='perm_auc', type=int, default=200, help='permutaciones del AUC (por desfase y por mitad)')
    ap.add_argument('--perm-detector', dest='perm_detector', type=int, default=20, help='permutaciones del DetectorErrP completo (~7 s cada una)')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--pseudo', type=int, default=300, help='sorteos de instantes al azar para el piso de la respuesta occipital')
    ap.add_argument('--amplitudes', type=float, nargs='+', default=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0], help='uV de la onda del gemelo inyectada')
    ap.add_argument('--amplitudes-detector', dest='amplitudes_detector', type=float, nargs='+', default=[6.0, 12.0],
                    help='uV con que se corre el DetectorErrP completo (~35 s cada una)')
    ap.add_argument('--amplitudes-curva', dest='amplitudes_curva', type=float, nargs='+', default=[3.0, 6.0], help='uV con que se arma la curva de aprendizaje')
    ap.add_argument('--tamanos-curva', dest='tamanos_curva', type=int, nargs='+', default=[120, 240, 360], help='numero de epocas de la curva de aprendizaje')
    ap.add_argument('--repeticiones-curva', dest='repeticiones_curva', type=int, default=4)
    ap.add_argument('--retraso-inyectado', dest='retraso_inyectado', type=float, default=0.15, help='s de retraso del ErrP inyectado en la prueba del barrido')
    ap.add_argument('--amplitud-retraso', dest='amplitud_retraso', type=float, default=6.0)
    ap.add_argument('--sin-detector', dest='sin_detector', action='store_true', help='omite lo que corre el DetectorErrP completo (lento)')
    ap.add_argument('--rapido', action='store_true', help='pocas permutaciones (revision en menos de un minuto; los p no son los del informe)')
    ap.add_argument('--sin-figura', dest='sin_figura', action='store_true')
    a = ap.parse_args(argv)
    if a.rapido:
        a.perm, a.perm_auc, a.perm_detector, a.boot, a.pseudo, a.amplitudes, a.sin_detector = 500, 30, 0, 300, 40, [0.0, 3.0, 6.0], True
        a.amplitudes_curva, a.tamanos_curva, a.repeticiones_curva = [6.0], [120, 240], 1
    warnings.filterwarnings('ignore')
    def sorteo(k):
        """Un generador por paso: omitir un paso (--sin-detector, sin XDF) no cambia los p de los demas."""
        return np.random.default_rng([a.semilla, k])
    salida = {'exploratorio': 'un participante, una sesion', 'argumentos': {k: v for k, v in vars(a).items() if k not in ('xdf', 'npz', 'jsonl')}}
    print('Diagnostico del ErrP de P001 (EXPLORATORIO: un participante, una sesion; los p son de permutacion, no de supuestos de normalidad)')

    eventos = marcadores = None
    if Path(a.jsonl).exists():
        eventos, marcadores = leer_jsonl(a.jsonl)
        paso0_ortesis(eventos, marcadores, salida)
        ack, ini = resumen_pasos(marcadores)
        seqs = sorted(set(ack) & set(ini))
        salida['marcadores_lat'] = (seqs, [ini[s] - ack[s] for s in seqs])
    else:
        print(f'\nFALTA {a.jsonl}: se omite el paso 0 (ortesis) y la parte de tiempos del paso 4. Se genera con desde_xdf.py.')

    if not Path(a.npz).exists():
        print(f'\nFALTA {a.npz}: sin las epocas de ErrP no hay pasos 1, 3 ni 4. Se genera con `python desde_xdf.py <archivo.xdf>`.')
        return 1
    d = np.load(a.npz)
    X, y, direccion = d['X'], d['y'].astype(int), d['direccion'].astype(int)
    estratos = estratos_del_diseno(y, direccion)
    print(f'\nEpocas: {X.shape} (n, canales, muestras), {int(y.sum())} errores; estratos del diseno: {len(np.unique(estratos))} de '
          f'{int(np.bincount(estratos).min())} a {int(np.bincount(estratos).max())} ensayos')

    xdf, t0s, cues_errp = None, None, None
    if a.xdf and Path(a.xdf).exists():
        try:
            xdf = leer_xdf(a.xdf)
        except ImportError as e:
            print(f'\nNo se pudo leer el XDF ({e}): falta pyxdf; se omiten los pasos 2, 3 (giro, alfa, 60 Hz) y 5.')
    else:
        print(f"\n{'FALTA el XDF ' + a.xdf if a.xdf else 'No se dio --xdf'}: se omiten el barrido de desfases (2), el giro, el alfa y el 60 Hz (3) y el control positivo (5).")
    if xdf is not None:
        ack_x = {int(m.split(':')[1]): ts for ts, m in xdf['marcas'] if m.startswith('paso_ack:')}
        ini_x = {int(m.split(':')[1]): ts for ts, m in xdf['marcas'] if m.startswith('paso_inicio:')}
        seqs = sorted(set(ack_x) & set(ini_x))
        t0s = np.array([ini_x[s] for s in seqs])
        cues_errp = np.array([ts for ts, m in xdf['marcas'] if m in (config.CUE_CERRAR, config.CUE_RELAJA)][-len(seqs):])
        if len(seqs) != len(y) or not np.array_equal(xdf['paso'][:, 0].astype(int), direccion):
            print(f'\nOJO: el XDF tiene {len(seqs)} pasos y la npz {len(y)} epocas, o las direcciones no coinciden: se omiten los pasos del XDF.')
            xdf = None
        else:
            tramos = preparar_continuo(xdf['x'], xdf['t'])
            E0 = cortar_todas(tramos, t0s)
            print(f"EEG continuo: {xdf['x'].shape[1]} muestras ({xdf['t'][-1] - xdf['t'][0]:.0f} s, {len(tramos)} tramo(s) sin huecos); las epocas recortadas del XDF "
                  f"{'coinciden con las de la npz' if np.allclose(E0, X, atol=1e-6) else 'DIFIEREN de las de la npz'} (max dif {np.abs(E0 - X).max():.2g} uV)")

    obs, bajo, alto = paso1_senal(X, y, direccion, estratos, a, sorteo(1), salida)
    if not a.sin_detector:
        paso_detector_nula(X, y, direccion, estratos, a, sorteo(2), salida)
    if xdf is not None:
        paso1c_preprocesamiento(xdf, t0s, y, estratos, a, sorteo(3), salida)
        paso2_desfases(tramos, t0s, y, estratos, a, sorteo(4), salida)
    paso3_calidad(X, y, estratos, xdf, t0s, cues_errp, a, sorteo(5), salida)
    if eventos is not None:
        paso4_diseno(y, direccion, eventos, marcadores, salida)
    if xdf is not None:
        paso4b_visibilidad(tramos, t0s, cues_errp, a, sorteo(6), salida)
        paso5_inyeccion(xdf, t0s, y, direccion, estratos, a, sorteo(7), salida)

    SALIDA_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SALIDA_JSON, 'w', encoding='utf-8') as f:
        json.dump(_json({k: v for k, v in salida.items() if k != 'marcadores_lat'}), f, ensure_ascii=False, indent=1)
    print(f'\nResultados: {SALIDA_JSON}')
    if not a.sin_figura:
        print(f'Figura: {figura(salida, obs, bajo, alto)}')
    o = salida.get('ortesis', {})
    if o.get('simulada') and not o.get('mano_virtual_eventos'):
        print('EXPLORATORIO: una persona, una sesion. Ortesis simulada y ninguna orden a la mano virtual: no hay registro de movimiento visible, asi que '
              'este resultado nulo no dice nada del ErrP de P001.')
    else:
        print('EXPLORATORIO: una persona, una sesion; un resultado nulo con pocos errores no se generaliza.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
