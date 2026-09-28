"""Simulador del lazo MI -> ortesis -> ErrP -> controlador.
Piloto sintetico para probar el agente sin casco ni persona.
Uso:  python simulador_lazo.py --sens 0.72 --espec 0.91
"""
import argparse
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import config


def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Decision:
    p_prima: float     # probabilidad de "cerrar"
    direccion: int     # +1 cerrar, -1 relajar
    dtheta: float      # grados que se mueve la ortesis


class PilotoSimulado:
    """Rasgos de imaginacion motora + detector ErrP imperfecto."""

    def __init__(self, dim=4, separacion=1.0, sens=0.70, espec=0.90,
                 p_artefacto=0.05, semilla_sujeto=0, semilla_ruido=1):
        u = np.random.default_rng(semilla_sujeto).normal(size=dim)
        self.u = u / np.linalg.norm(u)           # direccion "buena" del sujeto
        self.rng = np.random.default_rng(semilla_ruido)
        self.dim, self.sep = dim, separacion
        self.sens, self.espec, self.p_art = sens, espec, p_artefacto
        self.desplazamiento = np.zeros(dim)      # aqui vive la perturbacion

    def rasgos(self, y):
        """y = +1 (cerrar) o -1 (relajar)."""
        return (y * self.sep * self.u + self.desplazamiento
                + self.rng.normal(size=self.dim))

    def errp(self, erroneo):
        """Salida del detector de B2: (p_errp, artefacto).
        Siempre consume los mismos numeros aleatorios, para que dos
        controladores distintos vean exactamente el mismo piloto."""
        r = self.rng.random(3)
        b = self.rng.beta(2, 2)
        if r[0] < self.p_art:
            return float(r[2]), True
        detecta = r[1] < (self.sens if erroneo else 1 - self.espec)
        p = 0.5 + 0.5 * b if detecta else 0.5 * b   # p>0.5  <=>  detecta
        return float(p), False

    def perturbar(self, w, logits):
        """Desplaza los rasgos para que el decoder w pierda `logits`."""
        w = np.asarray(w, float)
        self.desplazamiento = -logits * w / np.dot(w, w)


def calibrar(piloto, n=80):
    """Simula la calibracion de B1: devuelve (w0, c0)."""
    y = np.repeat([1, -1], n // 2)
    X = np.array([piloto.rasgos(yi) for yi in y])
    clf = LogisticRegression().fit(X, (y > 0).astype(int))
    return clf.coef_[0], float(clf.intercept_[0])


class ControladorEstatico:
    """Decoder fijo de B1, sin aprendizaje. Linea base / decoder en la sombra."""

    def __init__(self, w, c, paso_max=10.0):
        self.w, self.c, self.paso_max = np.array(w, float), float(c), paso_max

    def decidir(self, phi):
        p = float(sigmoide(self.w @ phi + self.c))
        d = 1 if p >= 0.5 else -1
        return Decision(p, d, self.paso_max * abs(2 * p - 1))

    def actualizar(self, p_errp, artefacto, fiab):
        return {}


def simular(controlador, piloto, w_ref, pasos=600, pasos_ensayo=5,
            t_perturb=300, logits=config.PERTURBACION_LOGITS):
    reg = {'error': [], 'angulo': [], 'meta': [], 'beta': []}
    angulo, y = 45.0, 1
    for t in range(pasos):
        if t % pasos_ensayo == 0:                  # nuevo ensayo, nueva meta
            y = 1 if piloto.rng.random() < 0.5 else -1
        if t == t_perturb:
            piloto.perturbar(w_ref, logits)
        dec = controlador.decidir(piloto.rasgos(y))
        angulo = float(np.clip(angulo + dec.direccion * dec.dtheta, 0, 90))
        erroneo = dec.direccion != y
        p_errp, art = piloto.errp(erroneo)
        info = controlador.actualizar(p_errp, art, 1.0) or {}
        reg['error'].append(int(erroneo))
        reg['angulo'].append(angulo)
        reg['meta'].append(90 if y > 0 else 0)
        reg['beta'].append(info.get('beta', np.nan))
    return {k: np.array(v) for k, v in reg.items()}


def movil(x, w=30):
    return np.convolve(x, np.ones(w) / w, mode='valid')


def graficar(resultados, t_perturb):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for nombre, r in resultados.items():
        ax[0].plot(np.arange(len(movil(r['error']))) + 30, movil(r['error']),
                   label=nombre)
        ax[1].plot(r['angulo'], label=f'angulo {nombre}', lw=0.8)
    ax[1].plot(next(iter(resultados.values()))['meta'], 'k--', lw=0.6,
               label='meta')
    for a in ax:
        a.axvline(t_perturb, color='r', ls=':', label='perturbacion')
        a.legend(loc='upper right', fontsize=8)
    ax[0].set_ylabel('tasa de error (movil 30)')
    ax[1].set_ylabel('angulo (grados)')
    ax[1].set_xlabel('paso')
    fig.tight_layout()
    config.RESULTADOS.mkdir(exist_ok=True)
    fig.savefig(config.RESULTADOS / 'simulacion.png', dpi=120)
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sens', type=float, default=config.SENS)
    ap.add_argument('--espec', type=float, default=config.ESPEC)
    ap.add_argument('--pasos', type=int, default=600)
    ap.add_argument('--semilla', type=int, default=0)
    a = ap.parse_args()

    def nuevo_piloto(ruido):
        return PilotoSimulado(sens=a.sens, espec=a.espec,
                              semilla_sujeto=a.semilla, semilla_ruido=ruido)

    w0, c0 = calibrar(nuevo_piloto(ruido=99))
    t_p = a.pasos // 2
    controladores = {'estatico': ControladorEstatico(w0, c0)}
    # manana: controladores['agente'] = AgenteErrP(w0, c0, ...)

    resultados = {}
    for nombre, ctrl in controladores.items():
        r = simular(ctrl, nuevo_piloto(ruido=1), w0, a.pasos, t_perturb=t_p)
        resultados[nombre] = r
        print(f"{nombre:10s} error antes: {r['error'][:t_p].mean():.2f}   "
              f"despues: {r['error'][t_p:].mean():.2f}")
    graficar(resultados, t_p)


if __name__ == '__main__':
    main()