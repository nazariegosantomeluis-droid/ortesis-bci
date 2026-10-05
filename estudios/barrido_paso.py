"""Barrido del tamano de paso y de los pasos por ensayo en el gemelo (5 de octubre). SOLO el gemelo.

Cuanto se mueve la ortesis en cada paso (ganancia = paso maximo, como en config: 0.30) y cuantos pasos
da cada ensayo antes de cambiar de meta (config.PASOS_ENSAYO = 5) decide tres cosas que se estorban:
cuantos pasos caen en el tope del recorrido (no mueven la ortesis: sin epoca de ErrP, sin aprendizaje),
si la ortesis termina cerrada o abierta del todo al acabar el ensayo, y si el agente se recupera de la
perturbacion. Se mide cada combinacion con el lazo de estudios/agente_lento.py (topes del recorrido,
modo 'ignorar', agente bayes de hoy, detector actual), 4 sujetos x 4 lazos = 16 sesiones por celda,
las mismas semillas en todas las celdas (1000 + 100 s + k).

  pasos en tope     fraccion de los pasos del bloque adaptativo que no mueven la ortesis (< PASO_VISIBLE)
  cierre completo   fraccion de los ensayos de CERRAR que acaban con la ortesis >= 0.95 y de los de
                    RELAJAR que acaban <= 0.05 (promedio de las dos)
  recuperan         sesiones de 16 con beta al 70 % de la perturbacion en los 57 pasos del CP4
  error             fraccion de pasos erroneos en esos 2 min

PASO_VISIBLE (0.08) no se barre: es lo que percibe el piloto. La duracion de un paso en el tiempo no se
modela (el gemelo reacciona al ACK), asi que un paso grande no cuesta tiempo aqui; en el firmware 1.2 un
paso de 0.30 tarda ~430 ms (90 grados/s sobre 130 grados).

Uso: python estudios/barrido_paso.py [sujetos] [repeticiones]
     python estudios/barrido_paso.py informe      tabla y figura con lo ya corrido
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
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

CACHE = config.RESULTADOS / 'barrido_paso'
PASOS = (0.15, 0.20, 0.30, 0.40)             # ganancia = paso maximo
POR_ENSAYO = (3, 5, 8)                       # pasos por ensayo
ACTUAL = (config.PASO_MAX, config.PASOS_ENSAYO)


def una(tarea):
    """Un sujeto, una celda del barrido."""
    paso, por_ensayo, sujeto, reps = tarea
    al.REGIMEN, al.TOPES = 'actual', 'ignorar'
    config.PASOS_ENSAYO = por_ensayo          # lo lee el lazo (cada tarea corre en su proceso)
    mod = al.preparar(sujeto, salida=lambda *a: None)
    filas = []
    for r in range(reps):
        filas += [dict(x, sujeto=sujeto, rep=r) for x in
                  al.lazo(mod, 1000 + 100 * sujeto + r, cfg_agente={'ganancia': paso, 'paso_max': paso})]
    return (paso, por_ensayo), filas


def correr(sujetos=4, reps=4):
    tareas = [(p, n, s, reps) for p in PASOS for n in POR_ENSAYO for s in range(sujetos)]
    datos = {}
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, len(tareas))) as ex:
        for celda, filas in ex.map(una, tareas):
            datos.setdefault(celda, []).extend(filas)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / 'corrida.pkl').write_bytes(pickle.dumps({'sujetos': sujetos, 'reps': reps, 'datos': datos}))
    return datos


def medir(filas, por_ensayo):
    """Las cuatro cifras de una celda a partir de las filas de sus sesiones."""
    ad = [f for f in filas if f['bloque'] == 'adaptativo']
    quietos = float(np.mean([f['quieto'] for f in ad]))
    cierre = {1: [], -1: []}
    for f in ad:
        if f['t'] % por_ensayo == por_ensayo - 1:                       # ultimo paso del ensayo
            cierre[f['meta']].append(f['angulo'] >= 0.95 if f['meta'] > 0 else f['angulo'] <= 0.05)
    completo = {k: float(np.mean(v)) if v else float('nan') for k, v in cierre.items()}
    ses = al.sesiones({'m': filas}, 'm')
    return {'quieto': quietos, 'cierra': completo[1], 'abre': completo[-1], 'completo': float(np.nanmean(list(completo.values()))),
            'recuperan': sum(x['rec'] is not None for x in ses), 'n': len(ses),
            'err': float(np.mean([x['err'] for x in ses])),
            'mediana': float(np.median([x['rec'] for x in ses if x['rec'] is not None])) if any(x['rec'] for x in ses) else None}


def tabla(m):
    lineas = ['| Paso máx. | Pasos por ensayo | Pasos en tope | Cierre completo (cerrar / relajar) | Se recuperan | Error 2 min | Pasos hasta recuperar (mediana) |',
              '|---|---|---|---|---|---|---|']
    for (p, n), x in sorted(m.items()):
        marca = ' (actual)' if (p, n) == ACTUAL else ''
        lineas.append(f"| {p:.2f}{marca} | {n} | {x['quieto']:.0%} | {x['completo']:.0%} ({x['cierra']:.0%} / {x['abre']:.0%}) | "
                      f"{x['recuperan']}/{x['n']} | {x['err']:.3f} | {'—' if x['mediana'] is None else format(x['mediana'], '.0f')} |")
    return lineas


def figura(m, ruta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tinta, tinta2 = '#0b0b0b', '#52514e'
    paneles = [('quieto', 'Pasos en el tope del recorrido', '{:.0%}', 'Blues', None),
               ('completo', 'Ensayos que terminan con la órtesis completa', '{:.0%}', 'Greens', None),
               ('recuperan', 'Sesiones que se recuperan (de 16)', '{:.0f}', 'Purples', None)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.patch.set_facecolor('#fcfcfb')
    for a, (clave, titulo, fmt, mapa, _) in zip(ax, paneles):
        z = np.array([[m[(p, n)][clave] for n in POR_ENSAYO] for p in PASOS])
        im = a.imshow(z, cmap=mapa, aspect='auto', origin='lower')
        for i, p in enumerate(PASOS):
            for j, n in enumerate(POR_ENSAYO):
                a.text(j, i, fmt.format(z[i, j]), ha='center', va='center', fontsize=12, fontweight='bold',
                       color='white' if z[i, j] > (z.min() + z.max()) / 2 else tinta)
        a.set_xticks(range(len(POR_ENSAYO)), [str(n) for n in POR_ENSAYO])
        a.set_yticks(range(len(PASOS)), [f'{p:.2f}' for p in PASOS])
        a.set_xlabel('pasos por ensayo', color=tinta2)
        a.set_ylabel('paso máximo (fracción del recorrido)', color=tinta2)
        a.set_title(titulo, color=tinta, fontsize=11)
        if ACTUAL[0] in PASOS and ACTUAL[1] in POR_ENSAYO:
            a.add_patch(plt.Rectangle((POR_ENSAYO.index(ACTUAL[1]) - 0.5, PASOS.index(ACTUAL[0]) - 0.5), 1, 1, fill=False, ec='#eb6834', lw=3))
    fig.suptitle('Barrido del tamaño de paso y los pasos por ensayo (gemelo digital; recuadro naranja = la configuración actual)',
                 color=tinta, fontsize=13, fontweight='bold', x=0.01, ha='left')
    fig.text(0.01, 0.01, 'Gemelo digital sin LSL, detector actual, agente bayes, topes del recorrido, 16 sesiones por celda con las mismas semillas. '
             'No son datos de una persona.', fontsize=8.5, color=tinta2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return ruta


def informe(salida=print):
    d = pickle.loads((CACHE / 'corrida.pkl').read_bytes())
    m = {celda: medir(filas, celda[1]) for celda, filas in d['datos'].items()}
    ruta = figura(m, config.RAIZ / 'docs' / 'figuras' / 'barrido_paso.png')
    lineas = tabla(m)
    (config.RAIZ / 'docs' / 'barrido_paso.md').write_text('# Barrido del tamaño de paso y los pasos por ensayo (gemelo digital)\n\n'
                                                           + '\n'.join(lineas) + '\n', encoding='utf-8')
    salida(f'{ruta}\n' + '\n'.join(lineas))
    return m


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'informe':
        return informe()
    sujetos = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    correr(sujetos, reps)
    informe()


if __name__ == '__main__':
    main()
