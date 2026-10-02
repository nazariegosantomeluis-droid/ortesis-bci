"""Dos decisiones de la calibracion y el lazo de MI, medidas en el gemelo (sin LSL):

1. Espera tras la senal antes del primer paso: 0, 1 y 2 s extra (cuanto de la mejora de
   2 s conserva 1 s).
2. Minimo de ensayos de la calibracion secuencial de MI: 24 contra 36 (optimismo de la BA
   reportada contra el error real en el lazo).

Uso: python estudios/decisiones_mi.py [sujetos_espera] [sujetos_minimo]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from brecha_mi import Mundo, calibrar, lazo
from brecha_mi_calibracion import calibrar_secuencial

n_espera = int(sys.argv[1]) if len(sys.argv) > 1 else 8
n_minimo = int(sys.argv[2]) if len(sys.argv) > 2 else 16

print(f'1) espera extra tras la senal ({n_espera} sujetos, 300 pasos cada uno)')
res = {}
for extra in (0.0, 1.0, 2.0):
    err, prim = [], []
    for s in range(n_espera):
        m = Mundo(s)
        dec = calibrar(m, np.random.default_rng(s + 100))
        f = lazo(m, dec, np.random.default_rng(s + 200), ensayos=60, espera_extra=extra)
        err.append(f[:, 3].mean()); prim.append(f[f[:, 0] == 0, 3].mean())
    res[extra] = (np.mean(err), np.mean(prim))
    print(f'   +{extra:.0f} s: error {np.mean(err):.3f} +- {np.std(err):.3f} | primer paso {np.mean(prim):.3f}', flush=True)
mejora2, mejora1 = res[0.0][0] - res[2.0][0], res[0.0][0] - res[1.0][0]
print(f'   +1 s conserva {100 * mejora1 / mejora2:.0f} % de la mejora de +2 s' if mejora2 > 0 else '   +2 s no mejora')

print(f'2) minimo de ensayos de la calibracion secuencial ({n_minimo} sujetos)')
for n_min in (24, 36):
    rep, real, usados = [], [], []
    for s in range(n_minimo):
        m = Mundo(s)
        dec, n = calibrar_secuencial(m, np.random.default_rng(s + 100), n_min=n_min)
        f = lazo(m, dec, np.random.default_rng(s + 200), ensayos=60)
        rep.append(1 - dec.ba); real.append(f[:, 3].mean()); usados.append(n)
    print(f'   minimo {n_min}: ensayos usados (mediana) {np.median(usados):.0f} | error esperado por la BA reportada '
          f'{np.mean(rep):.3f} | error real {np.mean(real):.3f} | optimismo {np.mean(real) - np.mean(rep):+.3f}', flush=True)
