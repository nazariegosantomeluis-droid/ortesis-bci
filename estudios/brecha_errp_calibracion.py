"""Brecha del detector de ErrP: optimismo de la calibracion SECUENCIAL.

Por sujeto del gemelo: 300 epocas. Las primeras (hasta 120) calibran con la regla del
orquestador (revisa cada 10 desde 40; GO si el limite inferior del IC90 de la BA >= 0.75 y
la especificidad >= 0.90; NO GO temprano si el limite superior < 0.70 con 60 o mas). Las 180
restantes hacen de "lazo": sensibilidad y especificidad reales con el umbral calibrado."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
import config, hardware as hw, cerebro_sintetico as cs

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
filas = []
for s in range(N):
    X, y = cs.sesion_errp(300, semilla=s)
    Xc, yc, Xl, yl = X[:120], y[:120], X[120:], y[120:]
    decision, det = 'fin', None
    for n in range(40, 121, 10):
        det = hw.DetectorErrP().ajustar(Xc[:n], yc[:n])
        lo, hi = hw.intervalo_ba(yc[:n], det.pred_cv)
        if lo >= config.BA_MIN and det.espec >= config.ESPEC_MIN:
            decision = 'go'
            break
        if hi < config.BA_MIN - 0.05 and n >= 60:
            decision = 'nogo'
            break
    p = np.array([det.p_error(e) for e in Xl]) > det.umbral
    sens_l, espec_l = p[yl == 1].mean(), 1 - p[yl == 0].mean()
    completo = hw.DetectorErrP().ajustar(Xc, yc)                 # si se hubieran usado las 120
    pc = np.array([completo.p_error(e) for e in Xl]) > completo.umbral
    filas.append((decision, n, det.sens, det.espec, det.ba, sens_l, espec_l,
                  completo.ba, 0.5 * (pc[yl == 1].mean() + 1 - pc[yl == 0].mean())))
    f = filas[-1]
    print(f'  sujeto {s:2d}: {f[0]:4s} con {f[1]:3d} epocas | reportado sens {f[2]:.2f} espec {f[3]:.2f} BA {f[4]:.2f} | '
          f'real sens {f[5]:.2f} espec {f[6]:.2f} BA {0.5 * (f[5] + f[6]):.2f} | con 120: reportada {f[7]:.2f}, real {f[8]:.2f}', flush=True)
for nombre in ('go', 'nogo', 'fin'):
    G = np.array([f[1:] for f in filas if f[0] == nombre], dtype=float)
    if len(G):
        print(f'[{nombre}: {len(G)} de {N}] epocas {np.median(G[:, 0]):.0f} | reportado sens {G[:, 1].mean():.2f} espec {G[:, 2].mean():.2f} '
              f'BA {G[:, 3].mean():.2f} | real sens {G[:, 4].mean():.2f} espec {G[:, 5].mean():.2f} BA {(0.5 * (G[:, 4] + G[:, 5])).mean():.2f} | '
              f'con las 120 epocas: BA real {G[:, 7].mean():.2f}')
