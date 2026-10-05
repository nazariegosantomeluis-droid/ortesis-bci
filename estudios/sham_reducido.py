"""Corrida reducida de estudios/sham_gemelo.py (4 de octubre): solo el detector actual y el sham sin evidencia
('actual', 'nula'), 4 sujetos x 4 sesiones, 80 pasos por bloque, perturbacion en el paso 10, con el
config.ESPERA_PRIMER_PASO_S de hoy. Sirve para repetir el criterio de aceptacion de --sham tras un cambio de
tiempos sin correr los demas detectores y fuentes (~1 min con 2 nucleos). No escribe en resultados/sham_gemelo.
SOLO es el gemelo: no dice como sera con una persona.

Uso: python estudios/sham_reducido.py"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / 'estudios'))
import config
import sham_gemelo as sg


def main():
    t0 = time.time()
    print(f'config.ESPERA_PRIMER_PASO_S = {config.ESPERA_PRIMER_PASO_S}  VENTANA_MI = {config.VENTANA_MI}  '
          f'DURACION_MI_S = {config.DURACION_MI_S}', flush=True)
    sujetos, reps, pasos, en = 4, 4, 80, 10
    tareas = [('actual', 'nula', s, reps, pasos, en) for s in range(sujetos)]
    filas = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for reg, fuente, f in ex.map(sg.una, tareas):
            filas.extend(f)
            print(f'  listo {reg}/{fuente}: {len(f)} filas ({time.time()-t0:.0f} s)', flush=True)
    c = sg.comparar(filas)
    ok = c['rec_real'] >= 0.75 * c['n'] and c['rec_sham'] <= 0.1875 * c['n'] and c['ic'][0] > 0
    print(f"\n{pasos} pasos por bloque, perturbacion en el paso {en} ({sujetos} sujetos x {reps} sesiones)")
    print(f"  detector actual sham 'nula': recuperan real {c['rec_real']}/{c['n']} (mediana {c['pasos_real']:.0f} pasos), "
          f"sham {c['rec_sham']}/{c['n']} | error tras perturbar real {c['err_real']:.3f}, sham {c['err_sham']:.3f} "
          f"(sombra {c['sombra']:.2f}) | sham - real {c['dif']:+.3f} IC90 [{c['ic'][0]:+.3f}, {c['ic'][1]:+.3f}] "
          f"-> {'CUMPLE' if ok else 'no cumple'}")
    print(f"  pasos tras perturbar por sesion (post): {c['post']}")
    print(f'  filas totales {len(filas)}; tiempo {time.time()-t0:.0f} s', flush=True)
    print('FIN', flush=True)


if __name__ == '__main__':
    main()
