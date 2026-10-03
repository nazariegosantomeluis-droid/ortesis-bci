"""Epoca del ErrP alineada al ACK contra alineada al inicio real del movimiento (gemelo).

La ortesis empieza a moverse entre 30 y 150 ms despues del ACK; el cerebro reacciona al ver
el movimiento. Se compara la BA del detector (validacion anidada, 120 epocas) cortando la epoca
en el ACK, en el inicio real y en el inicio detectado por telemetria (error de ~5 ms).
SOLO verifica que la alineacion funciona: el efecto lo programamos nosotros en el gemelo.

Uso: python estudios/alineacion_epoca.py [sujetos]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
import numpy as np
import config, hardware as hw, cerebro_sintetico as cs

FS = cs.FS
ANTES = -config.EPOCA_ERRP[0]


def epocas(semilla, n=120, p_error=0.3):
    cer = cs.Cerebro(cs._args(semilla=semilla))
    rng = np.random.default_rng(semilla + 2)
    t, X = 0.0, {'ack': [], 'inicio real': [], 'telemetria': []}
    y = []
    for k in range(n):
        err = bool(rng.random() < p_error)
        cer.meta = 1
        pre, t = cs._bloque(cer, t, 1.5)
        t_ack = t + 1 / FS
        lat = hw.latencia_mecanica_simulada(k, semilla)
        cer.movimiento(t_ack + lat, err)
        post, t = cs._bloque(cer, t, 1.4)
        x = np.hstack([pre, post])
        ts = (t - (x.shape[1] - 1) / FS) + np.arange(x.shape[1]) / FS
        for nombre, t0 in (('ack', t_ack), ('inicio real', t_ack + lat),
                           ('telemetria', t_ack + lat + rng.normal(0, 0.005))):
            X[nombre].append(hw.cortar_epoca(x, ts, t0, FS))
        y.append(int(err))
    return {k: np.array(v) for k, v in X.items()}, np.array(y)


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    res = {k: [] for k in ('ack', 'inicio real', 'telemetria')}
    for s in range(N):
        X, y = epocas(s)
        for k in res:
            res[k].append(hw.DetectorErrP().ajustar(X[k], y).ba)
        print(f'  sujeto {s}: ' + ', '.join(f'{k} {res[k][-1]:.2f}' for k in res), flush=True)
    print(f'{N} sujetos del gemelo (latencia mecanica de 30 a 150 ms; BA anidada con 120 epocas):')
    for k, v in res.items():
        print(f'   epoca alineada a {k:12s}: BA {np.mean(v):.3f} +- {np.std(v):.3f}')
