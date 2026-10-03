"""Los pasos que no mueven la ortesis (3 de octubre).

Cada ensayo empieza en el punto medio y da 5 pasos de hasta 0.30 del recorrido: con un decoder
seguro la ortesis llega al tope en el segundo paso y los tres siguientes no la mueven. Tras la
perturbacion pasa lo mismo hacia el lado equivocado. Una persona no ve nada en esos pasos, asi
que no hay ErrP que leer. El gemelo de antes reaccionaba al marcador del paso aunque la ortesis
no se moviera, y por eso este problema no se veia.

Mismo lazo sin LSL que estudios/agente_lento.py, con los topes del recorrido (al.TOPES):

  ciego     el gemelo no reacciona a un paso que no se ve, pero el lazo lee su epoca y aprende
            de ella: lo que habria hecho el orquestador de antes con una persona
  ignorar   el lazo no lee esa epoca ni aprende de ella (config.IGNORAR_SIN_MOVIMIENTO, lo de hoy)

En los dos modos se corre el agente de hoy y su CONTROL NEGATIVO (sin la evidencia del ErrP), con
los tres detectores del estudio del agente lento. La columna 'sin topes' es ese estudio: todo paso
era un movimiento visible. Con el modo 'ignorar' se rehace la figura del control negativo y se
corre ademas el detector co-adaptativo (el lazo real lo lleva encendido).

SOLO es el gemelo: dice que hace el lazo con un piloto que no reacciona a lo que no ve, no como
sera con una persona.

Uso: python estudios/paso_sin_movimiento.py [sujetos] [repeticiones]
     python estudios/paso_sin_movimiento.py informe      tabla y figura con lo ya corrido
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')        # un proceso por combinacion, un hilo cada uno
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import config
import agente_lento as al

CACHE = config.RESULTADOS / 'paso_sin_movimiento'
REGIMENES = ('actual', 'ayer', 'debil')
MODOS = ('ciego', 'ignorar')
VARIANTES = (al.BASE, al.SIN_ERRP_HOY)
COADAPTA = 'calibrada (hoy) + detector co-adaptativo'
ARGUMENTOS = dict(al.VARIANTES, **{COADAPTA: {'coadaptar': True}})


def una(tarea):
    """Un regimen del gemelo con un modo de topes: el agente de hoy y su control, mismas sesiones."""
    reg, modo, sujetos, reps = tarea
    al.REGIMEN, al.TOPES = reg, modo
    variantes = VARIANTES + ((COADAPTA,) if modo == 'ignorar' else ())
    filas = {v: [] for v in variantes}
    for s in range(sujetos):
        mod = al.preparar(s, salida=lambda *a: None)
        for r in range(reps):
            for v in variantes:
                f = al.lazo(mod, 1000 + 100 * s + r, **ARGUMENTOS[v])
                filas[v] += [dict(x, sujeto=s, rep=r) for x in f]
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f'corrida_{modo}_{reg}.pkl').write_bytes(pickle.dumps(filas))
    return reg, modo


def cargar(modo, reg):
    if modo is None:                                  # el estudio del agente lento: sin topes
        ruta = al.CACHE / f'corrida_{reg}.pkl'
        return pickle.loads(ruta.read_bytes())['filas'] if ruta.exists() else None
    ruta = CACHE / f'corrida_{modo}_{reg}.pkl'
    return pickle.loads(ruta.read_bytes()) if ruta.exists() else None


def medidas(filas, variante=al.BASE):
    """Lo que paso en los 57 pasos (2 min) tras perturbar, por sesion."""
    ses = al.sesiones(filas, variante)
    v = [f for f in filas[variante] if f['post'] and f['t'] < al.PERTURBA_EN + al.VENTANA]
    vistos = [f for f in v if not f.get('quieto')]
    err = [f for f in vistos if f['erroneo'] and not f['art']]
    rec = [x['rec'] for x in ses if x['rec'] is not None]
    fin = [f for f in filas[variante] if f['post'] and f['t'] == al.PERTURBA_EN + al.VENTANA - 1]
    return {'n': len(ses), 'err': np.mean([x['err'] for x in ses]), 'ee': np.std([x['err'] for x in ses], ddof=1) / np.sqrt(len(ses)),
            'sombra': np.mean([x['sombra'] for x in ses]), 'rec': len(rec), 'pasos': np.median(rec) if rec else float('nan'),
            'quietos': np.mean([bool(f.get('quieto')) for f in v]),
            'quietos_err': np.mean([bool(f.get('quieto')) for f in v if f['erroneo']] or [np.nan]),
            'sens': np.mean([f['sens'] for f in fin]),
            'congelado': np.mean([bool(f.get('congelado', f['peso'] == 0)) for f in v]),
            'p_hat_err': np.mean([f['P_hat'] for f in err]) if err else float('nan')}


def informe(salida=print):
    datos = {}
    for reg in REGIMENES:
        salida(f"\nRegimen '{reg}' (agente de hoy; error en los 2 min tras perturbar, media +- EE entre sesiones)")
        for modo, nombre in ((None, 'sin topes (estudio del agente lento)'), ('ciego', 'topes, el lazo lee la epoca (antes)'),
                             ('ignorar', 'topes, el lazo ignora el paso (hoy)')):
            filas = cargar(modo, reg)
            if filas is None:
                continue
            a, c = medidas(filas), medidas(filas, al.SIN_ERRP_HOY)
            salida(f"  {nombre:38s} agente {a['err']:.3f} +- {a['ee']:.3f} (sombra {a['sombra']:.2f}) | recuperan "
                   f"{a['rec']:2d}/{a['n']} (mediana {a['pasos']:.0f} pasos) | control sin ErrP {c['rec']:2d}/{c['n']} "
                   f"(error {c['err']:.3f}) | pasos sin movimiento {a['quietos']:.0%} (de los errores {a['quietos_err']:.0%}) "
                   f"| sens viva al final {a['sens']:.2f} | congelado {a['congelado']:.0%} | P_hat de los errores vistos "
                   f"{a['p_hat_err']:.2f}")
            if modo == 'ignorar':
                datos[reg] = filas
                if COADAPTA in filas:
                    fijo, co = al.sesiones(filas, al.BASE), al.sesiones(filas, COADAPTA)
                    d = np.array([b['err'] - a_['err'] for a_, b in zip(fijo, co)])
                    k = medidas(filas, COADAPTA)
                    cambios = np.mean([max(f['version'] for f in filas[COADAPTA] if (f['sujeto'], f['rep']) == c) - 1
                                       for c in sorted({(f['sujeto'], f['rep']) for f in filas[COADAPTA]})])
                    salida(f"  {'  con el detector co-adaptativo':38s} agente {k['err']:.3f} ({d.mean():+.3f} +- "
                           f"{d.std(ddof=1) / np.sqrt(len(d)):.3f} contra el fijo; por sesion de {d.min():+.2f} a {d.max():+.2f}) "
                           f"| recuperan {k['rec']:2d}/{k['n']} (mediana {k['pasos']:.0f} pasos) | BA viva "
                           f"{al.ba_viva(filas, COADAPTA):.2f} contra {al.ba_viva(filas):.2f} | {cambios:.1f} cambios de modelo por sesion")
    if len(datos) == len(REGIMENES):
        nota = ('Los pasos que no mueven la órtesis (ya estaba en el tope) no producen ErrP en el gemelo y el agente '
                'no aprende de ellos.')
        salida(f'Figura: {al.graficar_control(datos, nota=nota)}')
    return datos


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'informe':
        return informe()
    sujetos = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tareas = [(reg, modo, sujetos, reps) for reg in REGIMENES for modo in MODOS]
    with ProcessPoolExecutor(max_workers=len(tareas)) as ex:
        for reg, modo in ex.map(una, tareas):
            print(f'  listo: {reg}, {modo}', flush=True)
    informe()


if __name__ == '__main__':
    main()
