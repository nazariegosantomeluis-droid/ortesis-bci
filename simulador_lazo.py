"""Simulador del lazo MI -> ortesis -> ErrP -> controlador.
Piloto sintetico para probar el agente sin casco ni persona.
Uso:  python simulador_lazo.py --sens 0.72 --espec 0.91
      python simulador_lazo.py --falla_detector     (prueba el CUSUM)
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import config
from agente_errp import (Decision, sigmoide, AgenteErrP, ConfigAgente,
                         MonitorDetector)


class PilotoSimulado:
    """Rasgos de imaginacion motora + detector ErrP imperfecto."""

    def __init__(self, dim=4, separacion=1.0, sens=0.70, espec=0.90,
                 p_artefacto=0.05, semilla_sujeto=0, semilla_ruido=1):
        u = np.random.default_rng(semilla_sujeto).normal(size=dim)
        self.u = u / np.linalg.norm(u)
        self.rng = np.random.default_rng(semilla_ruido)
        self.dim, self.sep = dim, separacion
        self.sens, self.espec, self.p_art = sens, espec, p_artefacto
        self.sens_ok, self.espec_ok = sens, espec
        self.desplazamiento = np.zeros(dim)

    def rasgos(self, y):
        return (y * self.sep * self.u + self.desplazamiento
                + self.rng.normal(size=self.dim))

    def errp(self, erroneo):
        r = self.rng.random(3)
        b = self.rng.beta(2, 2)
        if r[0] < self.p_art:
            return float(r[2]), True
        detecta = r[1] < (self.sens if erroneo else 1 - self.espec)
        p = 0.5 + 0.5 * b if detecta else 0.5 * b
        return float(p), False

    def perturbar(self, w, logits):
        w = np.asarray(w, float)
        self.desplazamiento = -logits * w / np.dot(w, w)

    def detector_degradado(self, si):
        """Simula que el detector se arruina (p.ej. ruido de servos)."""
        self.sens, self.espec = (0.5, 0.55) if si else (self.sens_ok, self.espec_ok)


def calibrar(piloto, n=80):
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


def simular(controlador, piloto, w_ref, monitor=None, pasos=600,
            pasos_ensayo=5, t_perturb=300, logits=config.PERTURBACION_LOGITS,
            falla=None):
    campos = ['error', 'angulo', 'meta', 'beta', 'P_hat', 'cusum', 'fiab']
    reg = {k: [] for k in campos}
    angulo, y, orden = 45.0, 1, []
    for t in range(pasos):
        if t % pasos_ensayo == 0:        # nuevo ensayo; cerrar/relajar balanceados
            if not orden:
                orden = list(piloto.rng.permutation([1, -1]))
            y = int(orden.pop())
        if t == t_perturb:
            piloto.perturbar(w_ref, logits)
        if falla:
            piloto.detector_degradado(falla[0] <= t < falla[1])
        dec = controlador.decidir(piloto.rasgos(y))
        angulo = float(np.clip(angulo + dec.direccion * dec.dtheta, 0, 90))
        erroneo = dec.direccion != y
        p_errp, art = piloto.errp(erroneo)
        fiab = monitor(erroneo, p_errp > 0.5, not art) if monitor else 1.0
        info = controlador.actualizar(p_errp, art, fiab) or {}
        reg['error'].append(int(erroneo))
        reg['angulo'].append(angulo)
        reg['meta'].append(90 if y > 0 else 0)
        reg['beta'].append(info.get('beta', np.nan))
        reg['P_hat'].append(info.get('P_hat', np.nan))
        reg['cusum'].append(monitor.cusum if monitor else np.nan)
        reg['fiab'].append(fiab)
    return {k: np.array(v) for k, v in reg.items()}


def movil(x, w=30):
    return np.convolve(x, np.ones(w) / w, mode='valid')


def graficar(resultados, t_perturb, falla=None):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for nombre, r in resultados.items():
        ax[0].plot(np.arange(len(movil(r['error']))) + 30, movil(r['error']),
                   label=nombre)
        ax[1].plot(r['angulo'], label=f'angulo {nombre}', lw=0.8)
    ax[1].plot(next(iter(resultados.values()))['meta'], 'k--', lw=0.6,
               label='meta')
    if 'agente' in resultados:
        r = resultados['agente']
        ax[2].plot(r['beta'], label='beta (correccion del agente)')
        ax[2].plot(r['cusum'], label='CUSUM (nats)')
        ax[2].axhline(config.CUSUM_CONGELAR, color='gray', ls='--', lw=0.8,
                      label='umbral congelar')
        ax[2].axhline(config.PERTURBACION_LOGITS, color='g', ls=':', lw=0.8,
                      label='beta ideal tras perturbacion')
    for a in ax:
        a.axvline(t_perturb, color='r', ls=':', label='perturbacion')
        if falla:
            a.axvspan(*falla, color='orange', alpha=0.15)
        a.legend(loc='upper right', fontsize=7)
    ax[0].set_ylabel('tasa de error (movil 30)')
    ax[1].set_ylabel('angulo (grados)')
    ax[2].set_ylabel('beta / CUSUM')
    ax[2].set_xlabel('paso')
    fig.tight_layout()
    config.RESULTADOS.mkdir(exist_ok=True)
    fig.savefig(config.RESULTADOS / 'simulacion.png', dpi=120)
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sens', type=float, default=config.SENS)
    ap.add_argument('--espec', type=float, default=config.ESPEC)
    ap.add_argument('--eta', type=float, default=config.ETA_BETA)
    ap.add_argument('--pasos', type=int, default=600)
    ap.add_argument('--semilla', type=int, default=0)
    ap.add_argument('--falla_detector', action='store_true',
                    help='arruina el detector entre los pasos 420 y 480')
    ap.add_argument('--sin_grafica', action='store_true')
    a = ap.parse_args()

    def nuevo_piloto(ruido):
        return PilotoSimulado(sens=a.sens, espec=a.espec,
                              semilla_sujeto=a.semilla, semilla_ruido=ruido)

    w0, c0 = calibrar(nuevo_piloto(ruido=99))
    t_p = a.pasos // 2
    falla = (420, 480) if a.falla_detector else None
    cfg = ConfigAgente(eta_beta=a.eta, sens=a.sens, espec=a.espec)
    corridas = {
        'estatico': (ControladorEstatico(w0, c0), None),
        'agente':   (AgenteErrP(w0, c0, cfg),
                     MonitorDetector(sensibilidad=a.sens, especificidad=a.espec)),
    }
    resultados = {}
    for nombre, (ctrl, mon) in corridas.items():
        r = simular(ctrl, nuevo_piloto(ruido=a.semilla + 1), w0, mon, a.pasos,
                    t_perturb=t_p, falla=falla)
        resultados[nombre] = r
        k = t_p + int(round(120 / config.CICLO_S))   # 2 minutos despues
        e = r['error']
        print(f"{nombre:9s} error  antes: {e[:t_p].mean():.2f}   "
              f"primeros 2 min: {e[t_p:k].mean():.2f}   "
              f"despues de 2 min: {e[k:].mean():.2f}")
    if 'agente' in resultados:
        print(f"beta final: {resultados['agente']['beta'][-1]:.2f} "
              f"(ideal ~ {config.PERTURBACION_LOGITS})")
    if not a.sin_grafica:
        graficar(resultados, t_p, falla)


if __name__ == '__main__':
    main()
    