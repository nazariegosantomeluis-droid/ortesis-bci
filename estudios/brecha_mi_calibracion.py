"""Brecha en MI, segunda parte: cuanto es optimismo de la calibracion SECUENCIAL (se detiene
cuando el intervalo ya decide GO) y cuanto es ruido de un bloque estatico de 30 pasos."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
import config, hardware as hw
from brecha_mi import Mundo, lazo   # mismo directorio


def calibrar_secuencial(m, rng, n_max=60, n_min=24, espera=1.5, duracion=4.0):
    """Como BackendReal.calibrar_mi: revisa cada 6 ensayos desde 24 y para al decidir."""
    X, y = [], []
    for k in range(n_max):
        clase = 1 - y[-1] if (k % 2 and y) else int(rng.integers(2))
        m.avanzar(espera)
        m.cer.meta = 1 if clase else -1
        m.avanzar(duracion)
        X.append(m.ventana_mi()); y.append(clase)
        n = k + 1
        if n >= n_min and n % 6 == 0 or n == n_max:
            dec = hw.DecoderIM().ajustar(np.array(X), np.array(y))
            lo, hi = hw.intervalo_ba(np.array(y), dec.pred_cv)
            if lo >= config.MI_EXACTITUD_MIN or (hi < config.MI_EXACTITUD_MIN - 0.05 and n >= 1.5 * n_min):
                break
    return dec, n


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    filas = []
    for s in range(N):
        m = Mundo(s)
        dec, n = calibrar_secuencial(m, np.random.default_rng(s + 100))
        f = lazo(m, dec, np.random.default_rng(s + 200), ensayos=60)          # 300 pasos: error "verdadero"
        err = f[:, 3]
        bloques = [err[i:i + 30].mean() for i in range(0, 300, 30)]            # bloques de 30 pasos, como el estatico
        filas.append((dec.ba, n, err.mean(), np.std(bloques), max(bloques), err[f[:, 0] == 0].mean(), err[f[:, 0] > 0].mean()))
    F = np.array(filas)
    go = F[:, 0] >= config.MI_EXACTITUD_MIN
    print(f'{N} sujetos del gemelo; calibracion secuencial (para al decidir)')
    print(f'  ensayos usados: mediana {np.median(F[:, 1]):.0f}; GO en {go.sum()}/{N}')
    for nombre, sel in (('todos', np.ones(N, bool)), ('solo los que dieron GO', go)):
        G = F[sel]
        print(f'  [{nombre}] BA reportada {G[:, 0].mean():.3f} -> error esperado {1 - G[:, 0].mean():.3f} | '
              f'error real en 300 pasos {G[:, 2].mean():.3f} | brecha media {(G[:, 2] - (1 - G[:, 0])).mean():+.3f}')
        print(f'      primer paso del ensayo {G[:, 5].mean():.3f} vs pasos 2 a 5 {G[:, 6].mean():.3f}')
        print(f'      un bloque de 30 pasos: desviacion {G[:, 3].mean():.3f}; el peor de 10 bloques da en promedio {G[:, 4].mean():.3f}')
    paradas_24 = F[:, 1] == 24
    if paradas_24.any():
        G = F[paradas_24]
        print(f'  [los que pararon a los 24 ensayos: {paradas_24.sum()}] BA reportada {G[:, 0].mean():.3f}, error real {G[:, 2].mean():.3f}')
