"""Investigacion de la brecha calibracion -> lazo (imaginacion motora), sin LSL.

Reproduce con el gemelo los tiempos del orquestador real: calibracion (ventana en estado
estable, sin movimientos de la ortesis) y lazo estatico (ventana justo tras la senal,
pasos cada ~0.9 s con ventanas traslapadas, respuestas cerebrales a cada movimiento,
recentrado en linea). Ablaciones: quitar una causa a la vez."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
import config, hardware as hw, cerebro_sintetico as cs

FS = cs.FS
PASO_S = 0.87                      # mediana entre pasos en la sesion de Luis


class Mundo:
    """El gemelo con un buffer continuo, para cortar ventanas como lo hace BackendReal."""

    def __init__(self, semilla):
        self.cer = cs.Cerebro(cs._args(semilla=semilla))
        self.t, self.x = 0.0, np.zeros((8, 0))

    def avanzar(self, seg):
        n = int(round(seg * FS))
        if n <= 0:
            return
        ts = self.t + np.arange(1, n + 1) / FS
        self.x = np.hstack([self.x, self.cer.generar(ts).T])[:, -int(8 * FS):]
        self.t = ts[-1]

    def ventana_mi(self):
        xf = hw.filtrar(self.x[:, -int((config.VENTANA_MI + 1.0) * FS):], config.BANDA_MI, FS)
        return xf[:, -int(config.VENTANA_MI * FS):]


def calibrar(m, rng, n=40, espera=1.5, duracion=4.0):
    X, y = [], []
    for k in range(n):
        clase = 1 - y[-1] if (k % 2 and y) else int(rng.integers(2))
        m.avanzar(espera)                       # el gemelo sigue en la clase anterior
        m.cer.meta = 1 if clase else -1
        m.avanzar(duracion)
        X.append(m.ventana_mi()); y.append(clase)
    return hw.DecoderIM().ajustar(np.array(X), np.array(y))


def lazo(m, dec, rng, ensayos=40, espera_extra=0.0, recentrar=True, erps=True, pasos=config.PASOS_ENSAYO):
    """Devuelve por paso: posicion en el ensayo, meta, p y error del decoder calibrado."""
    filas, orden = [], []
    for _ in range(ensayos):
        if not orden:
            orden = list(rng.permutation([1, -1]))
        meta = int(orden.pop())
        m.cer.meta = meta
        m.avanzar(config.VENTANA_MI + espera_extra)       # BackendReal.cue(): espera 2 s tras la senal
        for k in range(pasos):
            phi = dec.phi(m.ventana_mi(), actualizar_centro=recentrar)
            z = float(dec.w0 @ phi + dec.c0)
            d = 1 if z >= 0 else -1
            filas.append((k, meta, 1 / (1 + np.exp(-z)), int(d != meta)))
            if erps:
                if hasattr(m.cer, 'movimiento'):
                    m.cer.movimiento(m.t + 0.008, d != meta)   # la ortesis se mueve: N1 visual y, si falla, ErrP
                else:                                          # gemelo anterior (sin N1)
                    m.cer.eventos.append((m.t + 0.008, cs.plantilla_errp(d != meta, m.cer.a.errp, m.cer.rng)))
            m.avanzar(PASO_S)
    return np.array(filas)


CONDICIONES = [
    ('lazo tal como esta',                        dict()),
    ('(a) esperar 2 s mas tras la senal',         dict(espera_extra=2.0)),
    ('(b1) sin recentrado en linea',              dict(recentrar=False)),
    ('(b2) sin respuestas a los movimientos',     dict(erps=False)),
    ('(b1)+(b2)',                                 dict(recentrar=False, erps=False)),
    ('(a)+(b1)',                                  dict(espera_extra=2.0, recentrar=False)),
    ('(a)+(b1)+(b2)',                             dict(espera_extra=2.0, recentrar=False, erps=False)),
    ('(a)+(b1)+(b2), 1 paso por ensayo',          dict(espera_extra=2.0, recentrar=False, erps=False, pasos=1)),
]

if __name__ == '__main__':
    semillas = range(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    res = {n: [] for n, _ in CONDICIONES}
    bas, por_pos, p_pos = [], {n: [] for n, _ in CONDICIONES}, {n: [] for n, _ in CONDICIONES}
    for s in semillas:
        for nombre, kw in CONDICIONES:
            m = Mundo(s)                               # misma realizacion de calibracion en cada condicion
            dec = calibrar(m, np.random.default_rng(s + 100))
            if nombre == CONDICIONES[0][0]:
                bas.append(dec.ba)
            f = lazo(m, dec, np.random.default_rng(s + 200), **kw)
            res[nombre].append(f[:, 3].mean())
            npos = int(f[:, 0].max()) + 1
            por_pos[nombre].append([f[f[:, 0] == k, 3].mean() for k in range(npos)])
            p_pos[nombre].append([[f[(f[:, 0] == k) & (f[:, 1] == mt), 2].mean() for k in range(npos)] for mt in (-1, 1)])
    print(f'{len(list(semillas))} sujetos del gemelo (montaje Unicorn, decoder de 8 canales)')
    print(f'calibracion: BA {np.mean(bas):.3f} +- {np.std(bas):.3f}  -> error esperado {1 - np.mean(bas):.3f}')
    print(f"{'condicion':42s} {'error':>7s}   error por posicion del paso en el ensayo")
    for nombre, _ in CONDICIONES:
        e = np.array(res[nombre]); pp = np.mean(por_pos[nombre], axis=0)
        print(f'{nombre:42s} {e.mean():.3f} +- {e.std():.3f}   ' + ' '.join(f'{v:.2f}' for v in pp))
    for nombre in (CONDICIONES[0][0], CONDICIONES[2][0], CONDICIONES[3][0]):
        pm = np.mean(p_pos[nombre], axis=0)
        print(f'p media por paso [{nombre}]  relaja: ' + ' '.join(f'{v:.2f}' for v in pm[0])
              + '   cerrar: ' + ' '.join(f'{v:.2f}' for v in pm[1]))
