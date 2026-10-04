"""Control causal con sham en el gemelo (3 de octubre): criterio de aceptacion de --sham.

Mismo lazo sin LSL que estudios/paso_sin_movimiento.py (topes del recorrido, modo 'ignorar'), con
la sesion de `orquestador.py real --sham`: bloque estatico (30 pasos) y dos bloques adaptativos del
mismo largo, uno REAL y uno SHAM, en orden al azar. Cada bloque arranca con el agente reiniciado
(beta, varianza y prior), sin perturbacion, y recibe la suya (2.4 logits) en el mismo paso.

En el bloque sham el agente aprende con la fiabilidad fija en la calibrada (ni congela ni escala)
de una senal que no dice nada del error de cada paso. Se comparan las dos fuentes de esa senal
(config.SHAM_ERRP_FUENTES):

  nula         sin evidencia del ErrP: la tasa base de la calibracion (LLR = 0)
  recientes    los p_errp del mismo bloque, permutados entre los pasos recientes (lo que pidio Luis)
  calibracion  "sham ciego": p_errp sacados al azar de los de la calibracion (su distribucion y su tasa
               de error), sin relacion con el lazo actual

Una tercera, descartada: los p_errp del bloque estatico de la misma sesion, permutados. Quedaba en
medio (el sham se recuperaba en 3 a 8 de 16): cualquier senal con detecciones empuja a beta cuando
las decisiones estan cargadas a un lado.

Criterio (TAREAS.md): con 16 sesiones el bloque real se recupera en >= 12 y el sham en <= 3, y la
diferencia de error tras perturbar tiene un intervalo que excluye el 0.

SOLO es el gemelo: dice que hace el lazo con un piloto sintetico, no como sera con una persona.

Uso: python estudios/sham_gemelo.py [sujetos] [repeticiones] [pasos por bloque] [paso de la perturbacion]
     python estudios/sham_gemelo.py informe      tabla con lo ya corrido
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')        # un proceso por combinacion, un hilo cada uno
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import copy
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import config
import agente_lento as al
from agente_errp import AgenteErrP, ConfigAgente, ConfianzaDetector, SenalSham

CACHE = config.RESULTADOS / 'sham_gemelo'
FUENTES = config.SHAM_ERRP_FUENTES
REGIMENES = ('actual', 'ayer', 'debil')
META_BETA = 0.7 * config.PERTURBACION_LOGITS


def p_cal(mod):
    """Los p_errp de las epocas de calibracion, cada uno de un modelo que no vio esa epoca: lo que
    guarda DetectorErrP.ajustar en p_cv (los sujetos ya guardados en cache no lo traen)."""
    if 'p_cv' not in mod:
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        det = mod['det']
        mod['p_cv'] = getattr(det, 'p_cv', None)
        if mod['p_cv'] is None:
            cv = StratifiedKFold(4, shuffle=True, random_state=0)
            mod['p_cv'] = cross_val_predict(det._pipe(), mod['X_cal'], mod['y_cal'], method='predict_proba', cv=cv)[:, 1]
    return mod['p_cv']


def sesion(mod, semilla, pasos=config.SHAM_ERRP_PASOS, fuente='recientes', perturbar_en=config.SHAM_ERRP_PERTURBAR_EN, **cfg_agente):
    """Una sesion con --sham. Devuelve una fila por paso de los dos bloques adaptativos.
    cfg_agente: campos extra de ConfigAgente (estudios/prior_por_paso.py)."""
    dec, det = copy.deepcopy(mod['dec']), mod['det']
    m = al.Mundo2(semilla)
    m.avanzar(4.0)
    rng = np.random.default_rng(semilla + 7)             # metas y latencias
    sens, espec = (float(np.clip(v, 0.51, 0.99)) for v in (det.sens, det.espec))
    ag = AgenteErrP(dec.w0, dec.c0, ConfigAgente(modo='bayes', sens=sens, espec=espec, salida_detector='calibrada',
                                                 p_error_calibracion=det.p_error_cal, **cfg_agente))
    conf = ConfianzaDetector(sens, espec)
    orden_bloques = [str(v) for v in np.random.default_rng([semilla, 11]).permutation(config.BLOQUES_SHAM)]
    filas, angulo = [], config.PUNTO_MEDIO
    for k, (bloque, n) in enumerate([('estatico', al.PASOS_EST)] + [(b, pasos) for b in orden_bloques]):
        despl, orden, beta_pre = 0.0, [], None
        if bloque != 'estatico':
            ag.reiniciar()
            senal = SenalSham(fuente, semilla, reserva=p_cal(mod) if fuente == 'calibracion' else None)
        for t in range(n):
            if t % config.PASOS_ENSAYO == 0:
                if not orden:
                    orden = [int(v) for v in rng.permutation([1, -1])]
                meta = orden.pop()
                m.cer.meta = meta
                angulo = config.PUNTO_MEDIO
                m.avanzar(config.VENTANA_MI + config.ESPERA_PRIMER_PASO_S)
            if bloque != 'estatico' and t == perturbar_en:
                despl, beta_pre = -config.PERTURBACION_LOGITS, ag.beta
            d = ag.decidir(dec.phi(m.ventana_mi()), despl)
            erroneo = d.direccion != meta
            antes, angulo = angulo, float(np.clip(angulo + d.delta, 0, 1))
            if abs(angulo - antes) < config.PASO_VISIBLE - 1e-9:      # Orquestador.paso: ni epoca ni aprendizaje
                rng.uniform(*config.LATENCIA_MECANICA_SIM_MS)
                m.avanzar(config.EPOCA_ERRP[1])
                p, art = float('nan'), True
            else:
                e = al.mover(m, erroneo, rng)
                p, art = det.p_error(e), det.artefacto(e)
            fiab = conf(erroneo, bool(p > det.umbral), not art)      # sigue midiendo al detector
            sv, ev = conf.vivo()
            if bloque == 'sham':
                p_ag, s_ag, e_ag = (float('nan'), sens, espec) if art else senal(p, ag.cfg)
                ag.actualizar(p_ag, art, 1.0, s_ag, e_ag, peso=1.0)
            else:
                ag.actualizar(p, art, conf.fiabilidad_bruta, sv, ev, peso=fiab if bloque == 'real' else 0.0)
            if bloque != 'estatico':
                filas.append({'bloque': bloque, 'primero': k == 1, 't': t, 'post': despl != 0, 'erroneo': bool(erroneo),
                              'sombra': d.direccion_sombra != meta, 'beta': ag.beta, 'beta_pre': beta_pre,
                              'quieto': not np.isfinite(p), 'congelado': conf.congelado})
    return filas


def una(tarea):
    reg, fuente, sujeto, reps, pasos, en = tarea
    al.REGIMEN = reg
    mod = al.preparar(sujeto, salida=lambda *a: None)
    return reg, fuente, [dict(f, sujeto=sujeto, rep=r) for r in range(reps)
                         for f in sesion(mod, 1000 + 100 * sujeto + r, pasos, fuente, en)]


def por_sesion(filas, bloque, horizonte=None):
    """De cada sesion, lo que paso en ese bloque tras perturbar: error, sombra y pasos hasta recuperar
    (beta al 70 % de la perturbacion), mirando solo los primeros `horizonte` pasos tras perturbar."""
    out = []
    for clave in sorted({(f['sujeto'], f['rep']) for f in filas}):
        post = [f for f in filas if (f['sujeto'], f['rep']) == clave and f['bloque'] == bloque and f['post']][:horizonte]
        rec = np.flatnonzero(np.array([f['beta'] - f['beta_pre'] for f in post]) >= META_BETA)
        out.append({'err': np.mean([f['erroneo'] for f in post]), 'sombra': np.mean([f['sombra'] for f in post]),
                    'rec': int(rec[0]) + 1 if rec.size else None, 'primero': post[0]['primero'], 'n': len(post)})
    return out


def comparar(filas, horizonte=None, semilla=0):
    """Real contra sham, pareado por sesion: recuperadas, error y diferencia con intervalo del 90 %."""
    r, s = por_sesion(filas, 'real', horizonte), por_sesion(filas, 'sham', horizonte)
    d = np.array([b['err'] - a['err'] for a, b in zip(r, s)])
    rng = np.random.default_rng(semilla)
    medias = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(2000)]
    rec = [x['rec'] for x in r if x['rec'] is not None]
    return {'n': len(r), 'rec_real': len(rec), 'rec_sham': sum(x['rec'] is not None for x in s),
            'pasos_real': float(np.median(rec)) if rec else float('nan'),
            'err_real': float(np.mean([x['err'] for x in r])), 'err_sham': float(np.mean([x['err'] for x in s])),
            'sombra': float(np.mean([x['sombra'] for x in r + s])),
            'dif': float(d.mean()), 'ic': (float(np.quantile(medias, 0.05)), float(np.quantile(medias, 0.95))),
            'post': r[0]['n']}


def informe(salida=print):
    res = {}
    for ruta in sorted(CACHE.glob('corrida_*.pkl')):
        datos = pickle.loads(ruta.read_bytes())
        salida(f"\n{datos['pasos']} pasos por bloque, perturbacion en el paso {datos['en']} "
               f"({datos['sujetos']} sujetos x {datos['reps']} sesiones del gemelo)")
        for (reg, fuente), filas in sorted(datos['filas'].items()):
            c = res[datos['pasos'], datos['en'], reg, fuente] = comparar(filas)
            ok = c['rec_real'] >= 0.75 * c['n'] and c['rec_sham'] <= 0.1875 * c['n'] and c['ic'][0] > 0
            salida(f"  detector {reg:7s} sham '{fuente:10s}': recuperan real {c['rec_real']:2d}/{c['n']} (mediana "
                   f"{c['pasos_real']:.0f} pasos), sham {c['rec_sham']:2d}/{c['n']} | error tras perturbar real "
                   f"{c['err_real']:.3f}, sham {c['err_sham']:.3f} (sombra {c['sombra']:.2f}) | sham - real "
                   f"{c['dif']:+.3f} IC90 [{c['ic'][0]:+.3f}, {c['ic'][1]:+.3f}] -> {'CUMPLE' if ok else 'no cumple'}")
    return res


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'informe':
        return informe()
    sujetos = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    pasos = int(sys.argv[3]) if len(sys.argv) > 3 else config.SHAM_ERRP_PASOS
    en = int(sys.argv[4]) if len(sys.argv) > 4 else config.SHAM_ERRP_PERTURBAR_EN
    tareas = [(reg, fuente, s, reps, pasos, en) for reg in REGIMENES for fuente in FUENTES for s in range(sujetos)]
    filas = {}
    with ProcessPoolExecutor(max_workers=min(12, len(tareas))) as ex:
        for reg, fuente, f in ex.map(una, tareas):
            filas.setdefault((reg, fuente), []).extend(f)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f'corrida_{pasos}_{en}.pkl').write_bytes(pickle.dumps(
        {'pasos': pasos, 'en': en, 'sujetos': sujetos, 'reps': reps, 'filas': filas}))
    informe()


if __name__ == '__main__':
    main()
