"""Prior de error por paso contra el sham ciego (4 de octubre). EXPLORATORIO, solo el gemelo.

Hallazgo previo (estudios/sham_gemelo.py): el sham ciego (p_errp de la calibracion, a su tasa) deja
recuperarse al agente en 7 de 16 sesiones. Hipotesis: con el agente muy seguro y equivocado, cualquier
P_hat promedia el prior global y el gradiente empuja p' hacia el sin informacion real. Se prueba
`ConfigAgente.prior_por_paso`: el prior de cada paso es el error que el propio agente predice,
1 - max(p', 1 - p'), con un piso epsilon.

Se mide, con el lazo de estudios/sham_gemelo.py (gemelo sin LSL, topes del recorrido, 16 sesiones =
4 sujetos x 4, 80 pasos por bloque, perturbacion en el paso 10), para epsilon en {0, 0.05, 0.10, tasa global}
y la linea base (prior global): el bloque REAL (con el ErrP del detector) contra el bloque SHAM
ciego ('calibracion'); tambien con el sham 'nula' (el de la demo) para ver que no se rompe.
Criterio de aceptacion de TAREAS.md: real >= 12/16, sham <= 3/16 e intervalo de la diferencia sin el 0.

Control negativo obligatorio (CLAUDE.md) con el lazo de estudios/agente_lento.py (todo paso visible, sin
topes): `control` corre cada condicion con la evidencia real del ErrP y SIN ella (el agente recibe siempre la
tasa base: LLR = 0). Si sin evidencia se recupera, lo mueve el prior y no el ErrP.

Uso: python estudios/prior_por_paso.py [sujetos] [repeticiones] [pasos] [regimen]
     python estudios/prior_por_paso.py informe
     python estudios/prior_por_paso.py control [sujetos] [repeticiones] [regimen]
     python estudios/prior_por_paso.py velocidad      pasos hasta recuperarse del bloque real (mediana y p90), con lo ya corrido
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import config
import agente_lento as al
import sham_gemelo as sg

CACHE = config.RESULTADOS / 'prior_por_paso'
CONDICIONES = {'base (prior global)': {}, 'eps 0': {'prior_por_paso': True, 'piso_prior': 0.0},
               'eps 0.05': {'prior_por_paso': True, 'piso_prior': 0.05},
               'eps 0.10': {'prior_por_paso': True, 'piso_prior': 0.10},
               'eps 0.15': {'prior_por_paso': True, 'piso_prior': 0.15},
               'eps 0.20': {'prior_por_paso': True, 'piso_prior': 0.20},
               'eps = tasa global': {'prior_por_paso': True, 'piso_prior': None}}
FUENTES = ('calibracion', 'nula')


def una(tarea):
    reg, sujeto, reps, pasos = tarea
    al.REGIMEN = reg
    mod = al.preparar(sujeto, salida=lambda *a: None)
    out = {}
    for nombre, extra in CONDICIONES.items():
        for fuente in FUENTES:
            out[nombre, fuente] = [dict(f, sujeto=sujeto, rep=r) for r in range(reps)
                                   for f in sg.sesion(mod, 1000 + 100 * sujeto + r, pasos, fuente, **extra)]
    return reg, out


def control_una(tarea):
    reg, sujeto, reps = tarea
    al.REGIMEN = reg
    mod = al.preparar(sujeto, salida=lambda *a: None)
    return {(nombre, nulo): [al.resumen(al.lazo(mod, 1000 + 100 * sujeto + r, evidencia_nula=nulo, cfg_agente=extra))
                             for r in range(reps)]
            for nombre, extra in CONDICIONES.items() for nulo in (False, True)}


def control(sujetos, reps, reg, salida=print):
    res = {}
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, sujetos)) as ex:
        for out in ex.map(control_una, [(reg, s, reps) for s in range(sujetos)]):
            for k, v in out.items():
                res.setdefault(k, []).extend(v)
    salida(f"\nControl negativo, detector '{reg}', {sujetos * reps} sesiones del lazo de agente_lento.py (sin topes)")
    for (nombre, nulo), rr in res.items():
        rec = [x['pasos70'] for x in rr if x['pasos70'] is not None and x['pasos70'] <= al.VENTANA]    # los 2 min del CP4
        salida(f"  {nombre:20s} {'SIN evidencia del ErrP' if nulo else 'con el ErrP real     '}: recupera en los 2 min {len(rec):2d}/{len(rr)}"
               f" (mediana {np.median(rec) if rec else float('nan'):.0f} pasos) | error en los 2 min tras perturbar "
               f"{np.mean([x['err'] for x in rr]):.3f} (sombra {np.mean([x['sombra'] for x in rr]):.3f})")
    return res


def velocidad(filas):
    """Pasos hasta recuperarse (beta al 70 % de la perturbacion) del bloque REAL, una cifra por sesion.
    Una sesion que no se recupera en el bloque cuenta como infinito: si mas del 10 % no se recupera, el
    percentil 90 es infinito (no se alcanza)."""
    r = sg.por_sesion(filas, 'real')
    v = np.array([x['rec'] if x['rec'] is not None else np.inf for x in r], dtype=float)
    finito = np.where(np.isfinite(v), v, 1e9)         # np.quantile interpola mal con infinitos (inf - inf = nan)
    p90 = float(np.quantile(finito, 0.9))
    return {'n': len(v), 'recuperadas': int(np.isfinite(v).sum()), 'mediana': float(np.median(v)),
            'p90': p90 if p90 < 1e8 else float('inf'), 'pasos': v}


def comparar_velocidad(base, nuevo, semilla=0, remuestras=2000):
    """Mediana de `nuevo` contra la de `base` (cociente) con intervalo del 90 % por remuestreo de sesiones."""
    rng = np.random.default_rng(semilla)
    b, n = base['pasos'], nuevo['pasos']
    q = [np.median(n[rng.integers(len(n), size=len(n))]) / np.median(b[rng.integers(len(b), size=len(b))])
         for _ in range(remuestras)]
    return nuevo['mediana'] / base['mediana'], (float(np.quantile(q, 0.05)), float(np.quantile(q, 0.95)))


def informe_velocidad(condicion='eps 0.10', salida=print):
    """Lo ya corrido (corrida_actual_*.pkl): velocidad de recuperacion con la bandera contra sin ella."""
    out = {}
    for ruta in sorted(CACHE.glob('corrida_actual_*.pkl')):
        d = pickle.loads(ruta.read_bytes())
        salida(f"\nVelocidad de recuperacion, bloque real, {d['pasos']} pasos por bloque, {d['sujetos'] * d['reps']} sesiones del gemelo")
        for fuente in FUENTES:
            b, n = velocidad(d['filas']['base (prior global)', fuente]), velocidad(d['filas'][condicion, fuente])
            cociente, ic = comparar_velocidad(b, n)
            out[fuente] = (b, n, cociente, ic)
            salida(f"  (sesiones de la corrida con sham '{fuente}') sin bandera: recuperan {b['recuperadas']}/{b['n']}, mediana "
                   f"{b['mediana']:.1f} pasos, p90 {b['p90']:.1f} | {condicion}: {n['recuperadas']}/{n['n']}, mediana "
                   f"{n['mediana']:.1f}, p90 {n['p90']:.1f} | mediana x{cociente:.2f} IC90 [{ic[0]:.2f}, {ic[1]:.2f}]")
        todo = {k: np.concatenate([out[f][i]['pasos'] for f in FUENTES]) for k, i in (('base', 0), ('nuevo', 1))}
        junto = {k: {'pasos': v, 'mediana': float(np.median(v))} for k, v in todo.items()}
        c, ic = comparar_velocidad(junto['base'], junto['nuevo'])
        salida(f"  las dos corridas juntas ({len(todo['base'])} sesiones): mediana {junto['base']['mediana']:.1f} -> "
               f"{junto['nuevo']['mediana']:.1f} pasos (x{c:.2f}, IC90 [{ic[0]:.2f}, {ic[1]:.2f}]); p90 "
               f"{np.quantile(todo['base'], 0.9):.1f} -> {np.quantile(todo['nuevo'], 0.9):.1f}")
    return out


def informe(salida=print):
    for ruta in sorted(CACHE.glob('corrida_*.pkl')):
        d = pickle.loads(ruta.read_bytes())
        salida(f"\nDetector '{d['reg']}', {d['pasos']} pasos por bloque, {d['sujetos']} sujetos x {d['reps']} sesiones del gemelo")
        for (nombre, fuente), filas in d['filas'].items():
            c = sg.comparar(filas)
            ok = c['rec_real'] >= 0.75 * c['n'] and c['rec_sham'] <= 0.1875 * c['n'] and c['ic'][0] > 0
            salida(f"  {nombre:20s} sham {fuente:11s}: recuperan real {c['rec_real']:2d}/{c['n']} sham {c['rec_sham']:2d}/{c['n']}"
                   f" | error tras perturbar real {c['err_real']:.3f} sham {c['err_sham']:.3f}"
                   f" | sham - real {c['dif']:+.3f} IC90 [{c['ic'][0]:+.3f}, {c['ic'][1]:+.3f}] -> {'CUMPLE' if ok else 'no cumple'}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'informe':
        return informe()
    if len(sys.argv) > 1 and sys.argv[1] == 'velocidad':
        return informe_velocidad()
    if len(sys.argv) > 1 and sys.argv[1] == 'control':
        return control(int(sys.argv[2]) if len(sys.argv) > 2 else 4, int(sys.argv[3]) if len(sys.argv) > 3 else 4,
                       sys.argv[4] if len(sys.argv) > 4 else 'actual')
    sujetos = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    pasos = int(sys.argv[3]) if len(sys.argv) > 3 else config.SHAM_ERRP_PASOS
    reg = sys.argv[4] if len(sys.argv) > 4 else 'actual'
    filas = {}
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, sujetos)) as ex:
        for _, out in ex.map(una, [(reg, s, reps, pasos) for s in range(sujetos)]):
            for k, f in out.items():
                filas.setdefault(k, []).extend(f)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f'corrida_{reg}_{pasos}.pkl').write_bytes(pickle.dumps(
        {'reg': reg, 'pasos': pasos, 'sujetos': sujetos, 'reps': reps, 'filas': filas}))
    informe()


if __name__ == '__main__':
    main()
