"""Gemelo personalizado (5 de octubre): ajusta el cerebro sintetico a las calibraciones REALES de un piloto y
predice como le ira al agente con el. Predice, no mide: la prediccion se compara despues con la sesion real.

Que se ajusta (4 cosas, con epocas que la calibracion real ya guarda: resultados/calibracion_mi_*.npz y
calibracion_errp_*.npz, que son ventanas ya filtradas como las del lazo):
  ganancia por canal   RMS en 8-30 Hz de las epocas de MI en reposo (clase 0), canal por canal, contra el del gemelo
  erd                  cociente de potencia en C3 entre imaginar cerrar y reposo (8-30 Hz)
  errp                 amplitud MEDIA (300-420 ms, la Pe) de la onda diferencia error - acierto, promediada en Fz/Cz/Pz. Se uso primero el pico a pico,
                       pero con pocas epocas el ruido lo infla (en el piloto de mentira daba 9 uV para un ErrP de 4): la media lo promedia
  parpadeos            fraccion de epocas de ErrP con amplitud mayor que el umbral de artefacto del detector

Cada ajuste es de SIMULACION: se simula el gemelo con una rejilla del parametro, se calcula el MISMO
estadistico que sobre los datos reales y se invierte por interpolacion (np.interp). Si el valor real cae
fuera de la rejilla, el parametro queda en el borde y el ajuste lo marca (`fuera_de_rejilla`).

LIMITES (se repiten en el informe):
  - Solo son 4 parametros de un gemelo que tiene muchos otros fijos (alfa, 1/f, red electrica, theta, la forma
    del ErrP, la fatiga): que el gemelo copie 4 estadisticos del piloto no es que se parezca al piloto.
  - Las epocas guardadas estan filtradas (8-30 y 1-10 Hz): el ruido de banda ancha no se puede ajustar.
  - Con ~40 ensayos de MI y ~120 de ErrP los estadisticos tienen incertidumbre: el informe da su intervalo (bootstrap).
  - La prediccion usa 4 sujetos gemelos x 4 lazos con ESTE piloto como parametros; la variabilidad entre
    sujetos del gemelo es la del azar de sus semillas, no la de la persona.

Uso:  python gemelo_personal.py --mi resultados/calibracion_mi_<n>.npz --errp resultados/calibracion_errp_<n>.npz
          [--sesion resultados/sesion_real_<fecha>.csv]   (la del lunes: se pone al lado de la prediccion)
      python gemelo_personal.py --demo                   (un piloto de mentira: comprueba que recupera lo que se le puso)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import config

CANALES = list(config.CANALES_EEG)
IDX = {c: i for i, c in enumerate(CANALES)}
FS = config.FLUJOS['EEG'][2]
BASE = {'erd': 0.25, 'errp': 6.0, 'parpadeos': 0.15, 'ganancia_canal': [1.0] * len(CANALES)}   # el gemelo de siempre
REJILLAS = {'erd': (0.04, 0.10, 0.18, 0.28, 0.40, 0.55, 0.70),
            'errp': (1.0, 2.0, 3.5, 5.0, 7.0, 9.0, 12.0),
            'parpadeos': (0.0, 0.03, 0.08, 0.15, 0.30, 0.50)}
VENTANA_ERRP_S = (0.30, 0.42)                 # tras el movimiento (la Pe del ErrP); la epoca guardada empieza 0.2 s antes
ANTES_S = -config.EPOCA_ERRP[0]


# ------------------------------------------------------------ estadisticos (los mismos en real y gemelo)
def estadisticos_mi(X, y):
    """RMS por canal en reposo y log del cociente de potencia en C3 (cerrar / reposo). X: (n, 8, muestras); y: 1 = cerrar."""
    X, y = np.asarray(X, dtype=float), np.asarray(y).astype(int)
    if y.min() == y.max() or min((y == 0).sum(), (y == 1).sum()) < 4:
        raise ValueError('hacen falta al menos 4 ensayos de MI de cada clase')
    pot = (X ** 2).mean(axis=2)                                   # (n, 8) potencia por epoca y canal
    c3 = IDX['C3']
    # RMS por canal con la MEDIANA de las epocas: un parpadeo en el 20-30 % de ellas infla la media (sobre todo en Fz)
    return {'rms_reposo': np.sqrt(np.median(pot[y == 0], axis=0)),
            'log_cociente_c3': float(np.log(pot[y == 1, c3].mean() / pot[y == 0, c3].mean())), 'n': len(y)}


def estadisticos_errp(X, y, umbral_amp=None):
    """Amplitud media de la Pe (300-420 ms) en la onda diferencia (error - acierto) de Fz/Cz/Pz, sin las epocas con artefacto, y fraccion de epocas con artefacto."""
    X, y = np.asarray(X, dtype=float), np.asarray(y).astype(int)
    if y.min() == y.max() or min((y == 0).sum(), (y == 1).sum()) < 4:
        raise ValueError('hacen falta al menos 4 epocas de ErrP con error y 4 sin el')
    if umbral_amp is None:
        import hardware as hw
        umbral_amp = hw.DetectorErrP().umbral_amp
    # la onda diferencia SIN las epocas con artefacto (las que el detector tambien descarta): un parpadeo en el 20-30 % de
    # ellas la ahoga en ruido y el pico a pico ya no dice nada del ErrP
    limpia = np.ptp(X, axis=2).max(axis=1) <= umbral_amp
    frac = float((~limpia).mean())
    if min((limpia & (y == 1)).sum(), (limpia & (y == 0)).sum()) < 4:        # sin epocas limpias no hay onda diferencia que medir
        return {'errp_pe': float('nan'), 'frac_artefacto': frac, 'umbral_amp': float(umbral_amp), 'n': len(y)}
    dif = X[limpia & (y == 1)].mean(axis=0) - X[limpia & (y == 0)].mean(axis=0)
    fcp = dif[[IDX['Fz'], IDX['Cz'], IDX['Pz']]].mean(axis=0)
    i0, i1 = int((ANTES_S + VENTANA_ERRP_S[0]) * FS), int((ANTES_S + VENTANA_ERRP_S[1]) * FS)
    return {'errp_pe': float(fcp[i0:i1].mean()), 'frac_artefacto': frac,
            'umbral_amp': float(umbral_amp), 'n': len(y)}


def bootstrap(f, X, y, n=200, semilla=0):
    """Intervalo del 90 % de un estadistico (f(X, y) -> numero) remuestreando epocas."""
    rng = np.random.default_rng(semilla)
    v = []
    for _ in range(n):
        i = rng.integers(len(y), size=len(y))
        try:
            v.append(f(X[i], y[i]))
        except ValueError:
            continue
    return (float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))) if v else (float('nan'), float('nan'))


# ------------------------------------------------------------ el gemelo, con parametros
def _gemelo(params, semilla, n_mi=0, n_errp=0):
    import cerebro_sintetico as cs
    k = dict(erd=params['erd'], errp=params['errp'], parpadeos=params['parpadeos'],
             ganancia_canal=params['ganancia_canal'], semilla=semilla)
    mi = cs.sesion_mi(n_mi, **k) if n_mi else None
    er = cs.sesion_errp(n_errp, **k) if n_errp else None
    return mi, er


def _invertir(real, rejilla, simulados):
    """Valor del parametro cuyo estadistico simulado iguala al real (interpolacion; fuera de la rejilla, el borde)."""
    x, y = np.asarray(simulados, dtype=float), np.asarray(rejilla, dtype=float)
    ok = np.isfinite(x)                                           # un punto de la rejilla sin epocas limpias no cuenta
    x, y = x[ok], y[ok]
    orden = np.argsort(x)
    # un estadistico simulado no siempre sube de forma estricta con el parametro (ruido): se ordena y se promedian empates
    x, y = x[orden], y[orden]
    fuera = bool(real < x[0] or real > x[-1])
    return float(np.interp(real, x, y)), fuera


def ajustar(est_mi, est_errp, semilla=0, n_mi=48, n_errp=80, pasadas=2, rejillas=None, salida=print):
    """Parametros del gemelo a partir de los estadisticos reales. Devuelve (params, detalle).
    Los cuatro ajustes se estorban un poco (los parpadeos inflan el RMS de Fz y el ruido de la onda diferencia), asi
    que se repiten `pasadas` veces, cada una contra el gemelo con los parametros de la anterior."""
    params, detalle = dict(BASE), {}
    for p in range(pasadas):
        salida(f'  pasada {p + 1} de {pasadas}')
        params, detalle = _pasada(params, est_mi, est_errp, semilla, n_mi, n_errp, rejillas or REJILLAS, salida)
    return params, detalle


def _pasada(params, est_mi, est_errp, semilla, n_mi, n_errp, REJILLAS, salida):
    detalle = {}
    # 1) ganancia por canal: el RMS en reposo del piloto entre el del gemelo con los parametros de ahora
    mi, _ = _gemelo(dict(params, ganancia_canal=BASE['ganancia_canal']), semilla, n_mi=n_mi)
    rms_gemelo = estadisticos_mi(*mi)['rms_reposo']
    g = np.asarray(est_mi['rms_reposo']) / rms_gemelo
    params['ganancia_canal'] = [float(v) for v in g]
    # que tan bien queda: el RMS del gemelo con esas ganancias contra el real (el gemelo tiene su propia mezcla entre electrodos)
    ajustado = estadisticos_mi(*_gemelo(params, semilla, n_mi=n_mi)[0])['rms_reposo']
    detalle['ganancia_canal'] = {'error_rms_max': float(np.max(np.abs(ajustado / np.asarray(est_mi['rms_reposo']) - 1))), 'real_rms_uv': [float(v) for v in est_mi['rms_reposo']], 'gemelo_base_rms_uv': [float(v) for v in rms_gemelo]}
    salida('  ganancia por canal: ' + ', '.join(f'{c} x{v:.2f}' for c, v in zip(CANALES, g)))
    # 2) ERD
    sim = [estadisticos_mi(*_gemelo(dict(params, erd=v), semilla, n_mi=n_mi)[0])['log_cociente_c3'] for v in REJILLAS['erd']]
    params['erd'], fuera = _invertir(est_mi['log_cociente_c3'], REJILLAS['erd'], sim)
    detalle['erd'] = {'estadistico_real': est_mi['log_cociente_c3'], 'rejilla': list(REJILLAS['erd']), 'simulados': sim, 'fuera_de_rejilla': fuera}
    salida(f"  erd {params['erd']:.2f} (log cociente C3 real {est_mi['log_cociente_c3']:+.2f}" + ('; FUERA DE LA REJILLA' if fuera else '') + ')')
    # 3) ErrP
    if not np.isfinite(est_errp['errp_pe']):
        salida('  errp: sin epocas limpias suficientes (parpadeos en casi todas): se deja el del gemelo estandar')
        detalle['errp'] = {'estadistico_real': float('nan'), 'rejilla': list(REJILLAS['errp']), 'simulados': [], 'fuera_de_rejilla': True,
                           'nota': 'sin epocas limpias'}
    else:
        sim = [estadisticos_errp(*_gemelo(dict(params, errp=v), semilla, n_errp=n_errp)[1], est_errp['umbral_amp'])['errp_pe']
               for v in REJILLAS['errp']]
        params['errp'], fuera = _invertir(est_errp['errp_pe'], REJILLAS['errp'], sim)
        detalle['errp'] = {'estadistico_real': est_errp['errp_pe'], 'rejilla': list(REJILLAS['errp']), 'simulados': sim, 'fuera_de_rejilla': fuera}
        salida(f"  errp {params['errp']:.1f} uV (Pe de la onda diferencia real {est_errp['errp_pe']:.1f}"
               + ('; FUERA DE LA REJILLA' if fuera else '') + ')')
    # 4) parpadeos
    sim = [estadisticos_errp(*_gemelo(dict(params, parpadeos=v), semilla, n_errp=n_errp)[1], est_errp['umbral_amp'])['frac_artefacto'] for v in REJILLAS['parpadeos']]
    params['parpadeos'], fuera = _invertir(est_errp['frac_artefacto'], REJILLAS['parpadeos'], sim)
    detalle['parpadeos'] = {'estadistico_real': est_errp['frac_artefacto'], 'rejilla': list(REJILLAS['parpadeos']), 'simulados': sim, 'fuera_de_rejilla': fuera}
    salida(f"  parpadeos {params['parpadeos']:.2f}/s (epocas de ErrP con artefacto: real {est_errp['frac_artefacto']:.0%}" + ('; FUERA DE LA REJILLA' if fuera else '') + ')')
    return params, detalle


# ------------------------------------------------------------ prediccion
def _tarea(t):
    """Un sujeto gemelo con los parametros dados: calibra como el camino real y corre los lazos."""
    nombre, params, sujeto, reps = t
    sys.path.insert(0, str(config.RAIZ / 'estudios'))
    import agente_lento as al
    import hardware as hw
    al.REGIMEN, al.TOPES = 'actual', 'ignorar'

    class MundoPersonal(al.Mundo2):
        def __init__(self, semilla):
            super().__init__(semilla)
            a = self.cer.a
            a.erd, a.errp, a.parpadeos = params['erd'], params['errp'], params['parpadeos']
            self.cer.ganancia = np.asarray(params['ganancia_canal'], dtype=float)
    al.Mundo2 = MundoPersonal
    m = MundoPersonal(sujeto)
    X, y = al.epocas_mi(m, np.random.default_rng(sujeto + 100))
    dec = hw.DecoderIM().ajustar(X, y, config.candidatos('decoder'))
    Xe, ye = al.epocas_errp(m, np.random.default_rng(sujeto + 300))
    det = hw.DetectorErrP().ajustar(Xe, ye, config.candidatos('detector'))
    mod = {'dec': dec, 'det': det, 'X_cal': Xe, 'y_cal': ye}
    filas = {k: [] for k in ('estatico', 'bayes')}
    for r in range(reps):
        for k, kw in (('estatico', {'cfg_agente': {'modo': 'estatico'}}), ('bayes', {})):
            filas[k] += [dict(x, sujeto=sujeto, rep=r) for x in al.lazo(mod, 1000 + 100 * sujeto + r, **kw)]
    return nombre, {'mi_ba': float(dec.ba), 'sens': float(det.sens), 'espec': float(det.espec), 'ba': float(det.ba)}, filas


def predecir(params, sujetos=4, reps=4, con_estandar=True, salida=print):
    """Predice con el gemelo personalizado (y, de referencia, el estandar): calibracion (MI BA, ErrP sens / espec / BA) y
    lazo (error antes y 2 min tras perturbar, sesiones que se recuperan). Devuelve {condicion: {...}}."""
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0, str(config.RAIZ / 'estudios'))
    import agente_lento as al
    conds = {'personal': params, **({'estandar': BASE} if con_estandar else {})}
    tareas = [(n, p, s, reps) for n, p in conds.items() for s in range(sujetos)]
    cal, filas = {n: [] for n in conds}, {n: {'estatico': [], 'bayes': []} for n in conds}
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, len(tareas))) as ex:
        for nombre, c, f in ex.map(_tarea, tareas):
            cal[nombre].append(c)
            for k in f:
                filas[nombre][k] += f[k]
    out = {}
    for n in conds:
        r = {'calibracion': {k: float(np.mean([c[k] for c in cal[n]])) for k in cal[n][0]}, 'sujetos': sujetos}
        for k in ('estatico', 'bayes'):
            ses = al.sesiones({'m': filas[n][k]}, 'm')
            rec = [x['rec'] for x in ses if x['rec'] is not None]
            r[k] = {'antes': float(np.mean([x['antes'] for x in ses])), 'err': float(np.mean([x['err'] for x in ses])),
                    'err_ee': float(np.std([x['err'] for x in ses], ddof=1) / np.sqrt(len(ses))),
                    'recuperan': len(rec), 'n': len(ses), 'mediana': float(np.median(rec)) if rec else None}
        out[n] = r
        salida(f"  {n}: MI BA {r['calibracion']['mi_ba']:.2f}, ErrP BA {r['calibracion']['ba']:.2f}; bayes error {r['bayes']['err']:.3f}, "
               f"se recuperan {r['bayes']['recuperan']}/{r['bayes']['n']}")
    return out


# ------------------------------------------------------------ informe
def de_la_sesion(ruta):
    """Lo mismo, medido en la sesion real (CSV): error del agente tras perturbar y recuperacion, con copiloto.Sesion."""
    import copiloto
    s = copiloto.Sesion(Path(ruta))
    e, rec = s.metrica('error'), s.metrica('recuperacion')
    return {'error_agente': e.get('error_agente'), 'error_sombra': e.get('error_sombra'),
            'recuperacion': rec if isinstance(rec, list) else None, 'resumen': s.resumen_sesion()}


def informe(params, detalle, est, pred, real=None, ruta=None):
    p = pred['personal']
    est_ = pred.get('estandar')
    L = ['# Gemelo personalizado: predicción para este piloto', '',
         '**Es una predicción hecha con un gemelo digital ajustado a la calibración real, no una medición. Compárala con la sesión real; no la presentes como resultado.**', '',
         '## Parámetros ajustados', '', '| Parámetro | Gemelo estándar | Este piloto | Estadístico real | Fuera de la rejilla |', '|---|---|---|---|---|',
         f"| ERD (C3) | {BASE['erd']:.2f} | {params['erd']:.2f} | log cociente C3 {detalle['erd']['estadistico_real']:+.2f} | {'SÍ' if detalle['erd']['fuera_de_rejilla'] else 'no'} |",
         f"| ErrP (µV) | {BASE['errp']:.1f} | {params['errp']:.1f} | amplitud de la Pe {detalle['errp']['estadistico_real']:.1f} µV | {'SÍ' if detalle['errp']['fuera_de_rejilla'] else 'no'} |",
         f"| Parpadeos (/s) | {BASE['parpadeos']:.2f} | {params['parpadeos']:.2f} | épocas con artefacto {detalle['parpadeos']['estadistico_real']:.0%} | {'SÍ' if detalle['parpadeos']['fuera_de_rejilla'] else 'no'} |",
         '', 'Ganancia por canal (RMS en reposo del piloto entre el del gemelo base): ' + ', '.join(f'{c} ×{v:.2f}' for c, v in zip(CANALES, params['ganancia_canal'])) + '.',
         f"Épocas usadas: {est['mi']['n']} ensayos de MI y {est['errp']['n']} de ErrP.", '',
         '## Predicción (4 sujetos gemelos × 4 lazos con estos parámetros)', '',
         '| | Gemelo estándar | Este piloto (predicción) |', '|---|---|---|']
    f = lambda r, k: r[k]
    L += [f"| BA del decoder de MI (calibración) | {est_['calibracion']['mi_ba']:.2f} | {p['calibracion']['mi_ba']:.2f} |" if est_ else '',
          f"| Detector de ErrP: sens / espec / BA | {est_['calibracion']['sens']:.2f} / {est_['calibracion']['espec']:.2f} / {est_['calibracion']['ba']:.2f} | {p['calibracion']['sens']:.2f} / {p['calibracion']['espec']:.2f} / {p['calibracion']['ba']:.2f} |" if est_ else '',
          f"| Error antes de perturbar (bayes) | {est_['bayes']['antes']:.3f} | {p['bayes']['antes']:.3f} |" if est_ else '',
          f"| Error en los 2 min tras perturbar: estático | {est_['estatico']['err']:.3f} | {p['estatico']['err']:.3f} |" if est_ else '',
          f"| Error en los 2 min tras perturbar: bayes | {est_['bayes']['err']:.3f} ± {est_['bayes']['err_ee']:.3f} | {p['bayes']['err']:.3f} ± {p['bayes']['err_ee']:.3f} |" if est_ else '',
          f"| Sesiones que se recuperan (bayes) | {est_['bayes']['recuperan']}/{est_['bayes']['n']} | {p['bayes']['recuperan']}/{p['bayes']['n']} |" if est_ else '',
          f"| Pasos hasta recuperar (mediana) | {'—' if est_['bayes']['mediana'] is None else format(est_['bayes']['mediana'], '.0f')} | {'—' if p['bayes']['mediana'] is None else format(p['bayes']['mediana'], '.0f')} |" if est_ else '']
    if real:
        e = real.get('error_agente')
        L += ['', '## La sesión real, al lado', '',
              f"- Error del agente en la sesión: {e:.3f}" + (f" (decoder sin aprender: {real['error_sombra']:.3f})" if real.get('error_sombra') is not None else '') if e is not None else '- La sesión no trae error medible.',
              '- Recuperación: ' + (json.dumps(real['recuperacion'], ensure_ascii=False) if real.get('recuperacion') else 'sin dato'),
              '- La sesión real es UNA sola (con su propio azar): una diferencia con la predicción de unos puntos de error no es un fallo del gemelo; el error de la predicción es del orden de su error estándar.']
    L += ['', '## Límites', '',
          '- 4 estadísticos ajustados de un gemelo con muchos parámetros fijos (alfa, 1/f, red eléctrica, theta, forma del ErrP, fatiga). Que copie esos 4 no es que se parezca al piloto.',
          '- Las épocas guardadas están filtradas (8–30 y 1–10 Hz): el ruido de banda ancha no se ajusta.',
          '- Los sujetos gemelos difieren por el azar de sus semillas, no por la persona: la dispersión de la predicción no es la del piloto.',
          '- **La recuperación es un umbral.** Con una BA del detector entre 0.60 y 0.70 el agente pasa de no recuperarse a recuperarse en la mayoría de las sesiones (curva de robustez del README): un error de 0.03 en la BA predicha puede cambiar la predicción de recuperación de 0/16 a 8/16 (así fue con el piloto de mentira). Lo más estable de la predicción son las BA de la calibración y el rango del error (0.45 a 0.50 tras perturbar); ancla la predicción con la BA REAL del CP3 del piloto.']
    texto = '\n'.join(x for x in L if x is not None)
    if ruta:
        Path(ruta).write_text(texto + '\n', encoding='utf-8')
    return texto


def estadisticos_de_archivos(mi_npz, errp_npz):
    mi, er = np.load(mi_npz), np.load(errp_npz)
    return {'mi': estadisticos_mi(mi['X'], mi['y']), 'errp': estadisticos_errp(er['X'], er['y'])}, (mi, er)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Gemelo personalizado a partir de la calibracion real de un piloto')
    ap.add_argument('--mi', help='resultados/calibracion_mi_<n>.npz')
    ap.add_argument('--errp', help='resultados/calibracion_errp_<n>.npz')
    ap.add_argument('--sesion', help='CSV de la sesion real que se compara con la prediccion')
    ap.add_argument('--demo', action='store_true', help='un piloto de mentira (del propio gemelo): comprueba que el ajuste lo recupera')
    ap.add_argument('--sujetos', type=int, default=4)
    ap.add_argument('--salida', default=str(config.RESULTADOS / 'gemelo_personal.md'))
    a = ap.parse_args(argv)
    if a.demo:
        pil = {'erd': 0.15, 'errp': 4.0, 'parpadeos': 0.30, 'ganancia_canal': [1.2, 0.8, 1.0, 1.4, 0.9, 1.1, 1.3, 0.7]}
        mi, er = _gemelo(pil, 77, n_mi=48, n_errp=120)
        est = {'mi': estadisticos_mi(*mi), 'errp': estadisticos_errp(*er)}
        print(f'Piloto de mentira (del gemelo): {pil}')
    else:
        if not (a.mi and a.errp):
            ap.error('hacen falta --mi y --errp (o --demo)')
        est, _ = estadisticos_de_archivos(a.mi, a.errp)
    params, detalle = ajustar(est['mi'], est['errp'])
    pred = predecir(params, a.sujetos)
    real = de_la_sesion(a.sesion) if a.sesion else None
    texto = informe(params, detalle, est, pred, real, a.salida)
    print(texto)
    if a.demo:                                    # la unica comprobacion posible sin un piloto real: el verdadero rendimiento del piloto de mentira
        print('\n## Comprobación con el piloto de mentira (su rendimiento verdadero en el gemelo, mismos 4 × 4)\n')
        verdad = predecir(pil, a.sujetos, con_estandar=False, salida=lambda *x: None)['personal']
        p = pred['personal']
        for k, nombre in (('estatico', 'estático'), ('bayes', 'bayes')):
            print(f"- {nombre}: predicho {p[k]['err']:.3f} ({p[k]['recuperan']}/{p[k]['n']} se recuperan) · verdadero {verdad[k]['err']:.3f} ({verdad[k]['recuperan']}/{verdad[k]['n']})")
        print(f"- calibración: MI BA predicha {p['calibracion']['mi_ba']:.2f} / verdadera {verdad['calibracion']['mi_ba']:.2f}; "
              f"ErrP BA {p['calibracion']['ba']:.2f} / {verdad['calibracion']['ba']:.2f}")
    print(f'\nInforme: {a.salida}')


if __name__ == '__main__':
    main()
