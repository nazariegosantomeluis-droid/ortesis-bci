"""Aceptacion de la Tarea 2 en el gemelo: el IIC (atenuacion de la N1 visual en PO7/Oz/PO8)
debe ordenar sesiones con embodiment 0.2, 0.5 y 0.8, y con embodiment 0 su intervalo debe
incluir 0. Cada sesion es independiente (su propia semilla). Uno de cada 10 movimientos es
ajeno, como en el lazo.

SOLO verifica el estimador: la atenuacion la programamos nosotros en el gemelo
(cerebro_sintetico.ATENUACION_MAX); no dice nada de si existe en una persona.

Uso: python estudios/embodiment_gemelo.py [sujetos] [movimientos]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
import numpy as np
from scipy.stats import spearmanr
import cerebro_sintetico as cs
import embodiment as emb

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
M = int(sys.argv[2]) if len(sys.argv) > 2 else 300
NIVELES = (0.0, 0.2, 0.5, 0.8)

R = {e: [] for e in NIVELES}
rho, ordenadas = [], 0
for s in range(N):
    for k, e in enumerate(NIVELES):
        semilla = 100 * s + k                                # sesiones independientes
        X, aj, ok = cs.sesion_embodiment(M, embodiment=e, semilla=semilla)
        R[e].append(emb.desde_epocas(X, aj, ok, fs=cs.FS, semilla=semilla).estimar())
    v = [R[e][-1]['iic'] for e in NIVELES[1:]]
    rho.append(spearmanr(NIVELES[1:], v)[0])
    ordenadas += v[0] < v[1] < v[2]
    print(f'  sujeto {s:2d}: ' + ', '.join(f"emb {e}: {R[e][-1]['iic']:+.2f}" for e in NIVELES)
          + f' | rho {rho[-1]:+.1f}', flush=True)

print(f'{N} sujetos del gemelo, {M} movimientos por sesion ({M // 10} ajenos), IIC con intervalo del 90 %:')
for e in NIVELES:
    iic = np.array([r['iic'] for r in R[e]])
    cubre0 = np.mean([r['ic'][0] < 0 < r['ic'][1] for r in R[e]])
    print(f'   embodiment {e}: IIC {iic.mean():+.2f} +- {iic.std():.2f} | intervalo incluye 0 en {cubre0:.0%}')
niv = np.repeat(NIVELES[1:], N)
iic = np.array([r['iic'] for e in NIVELES[1:] for r in R[e]])
rho_total = spearmanr(niv, iic)[0]
rng = np.random.default_rng(0)                               # intervalo: remuestreando sujetos
boot = []
for _ in range(2000):
    i = rng.integers(0, N, N)
    idx = np.concatenate([i + k * N for k in range(3)])
    boot.append(spearmanr(niv[idx], iic[idx])[0])
print(f'   Spearman por sujeto (0.2/0.5/0.8): media {np.mean(rho):+.2f}; orden perfecto en {ordenadas} de {N}')
print(f'   Spearman con todas las sesiones juntas: {rho_total:+.2f} '
      f'[{np.quantile(boot, 0.05):+.2f}, {np.quantile(boot, 0.95):+.2f}]')
