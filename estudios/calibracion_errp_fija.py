"""Verifica la calibracion de ErrP corregida: 120 epocas fijas y umbral de Neyman-Pearson
elegido con validacion anidada. Por sujeto del gemelo: 300 epocas; las primeras 120 calibran y
las 180 restantes miden lo real. Lo reportado debe coincidir con lo real.

Uso: python estudios/calibracion_errp_fija.py [sujetos]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
import numpy as np
import config, hardware as hw, cerebro_sintetico as cs

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
F = []
for s in range(N):
    X, y = cs.sesion_errp(300, semilla=s)
    det = hw.DetectorErrP().ajustar(X[:120], y[:120])
    p = np.array([det.p_error(e) for e in X[120:]]) > det.umbral
    yl = y[120:]
    sens, espec = p[yl == 1].mean(), 1 - p[yl == 0].mean()
    F.append((det.sens, det.espec, det.ba, sens, espec, 0.5 * (sens + espec), det.ba >= config.BA_MIN and det.espec >= config.ESPEC_MIN))
    f = F[-1]
    print(f'  sujeto {s:2d}: reportado sens {f[0]:.2f} espec {f[1]:.2f} BA {f[2]:.2f} | real sens {f[3]:.2f} espec {f[4]:.2f} '
          f'BA {f[5]:.2f} | CP3 {"GO" if f[6] else "NO GO"}', flush=True)
F = np.array(F, dtype=float)
print(f'{N} sujetos del gemelo, 120 epocas fijas, umbral anidado')
print(f'   reportado: sens {F[:, 0].mean():.2f}  espec {F[:, 1].mean():.2f}  BA {F[:, 2].mean():.2f}')
print(f'   real:      sens {F[:, 3].mean():.2f}  espec {F[:, 4].mean():.2f}  BA {F[:, 5].mean():.2f}')
print(f'   diferencia media de BA (reportada - real): {np.mean(F[:, 2] - F[:, 5]):+.3f} +- {np.std(F[:, 2] - F[:, 5]):.3f}')
print(f'   CP3 en GO: {int(F[:, 6].sum())} de {N}')
