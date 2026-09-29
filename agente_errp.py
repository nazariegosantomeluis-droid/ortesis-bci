"""Agente que corrige el decoder de MI usando el ErrP como recompensa.

Idea: el decoder de B1 da  logit = w0.phi + c0.  El agente aprende un solo
parametro extra, beta, que se suma al logit:  logit = w0.phi + c0 + beta.
Una perturbacion de 2.4 logits es justo un corrimiento de ese sesgo, asi
que aprender beta basta para recuperarse, y con un solo parametro aprende
rapido aunque la recompensa (el ErrP) sea ruidosa.
"""
from dataclasses import dataclass
import numpy as np
import config


def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Decision:
    p_prima: float     # probabilidad de "cerrar" (ya corregida)
    direccion: int     # +1 cerrar, -1 relajar
    dtheta: float      # grados


@dataclass
class ConfigAgente:
    eta_beta: float = config.ETA_BETA   # tasa de aprendizaje de beta
    sens: float = config.SENS           # sensibilidad medida del detector (B2)
    espec: float = config.ESPEC         # especificidad medida del detector (B2)
    prior_error: float = 0.2            # tasa de error inicial supuesta
    alfa_prior: float = 0.05            # que tan rapido se actualiza ese prior
    beta_max: float = 5.0               # limite de seguridad para beta
    paso_max: float = 10.0              # grados maximos por paso


class AgenteErrP:
    def __init__(self, w0, c0, cfg=None):
        self.w, self.c = np.asarray(w0, float), float(c0)
        self.cfg = cfg or ConfigAgente()
        self.beta = 0.0
        self.prior = self.cfg.prior_error
        self._ultima = None

    def decidir(self, phi):
        p = float(sigmoide(self.w @ phi + self.c + self.beta))
        d = 1 if p >= 0.5 else -1
        self._ultima = Decision(p, d, self.cfg.paso_max * abs(2 * p - 1))
        return self._ultima

    def prob_error(self, p_errp):
        """P_hat: probabilidad (Bayes) de que el ultimo paso haya sido erroneo,
        dado lo que dijo el detector y su sensibilidad/especificidad."""
        s, e, pi = self.cfg.sens, self.cfg.espec, self.prior
        if p_errp > 0.5:
            num, den = s * pi, s * pi + (1 - e) * (1 - pi)
        else:
            num, den = (1 - s) * pi, (1 - s) * pi + e * (1 - pi)
        return num / den

    def actualizar(self, p_errp, artefacto, fiab):
        dec = self._ultima
        if dec is None or artefacto:
            return {'beta': self.beta, 'P_hat': np.nan, 'aprendio': False}
        P_hat = self.prob_error(p_errp)
        # q = probabilidad de que la intencion real fuera "cerrar"
        q = (1 - P_hat) if dec.direccion > 0 else P_hat
        # gradiente de la log-verosimilitud logistica respecto a beta
        self.beta += self.cfg.eta_beta * fiab * (q - dec.p_prima)
        self.beta = float(np.clip(self.beta, -self.cfg.beta_max, self.cfg.beta_max))
        self.prior += self.cfg.alfa_prior * (P_hat - self.prior)
        return {'beta': self.beta, 'P_hat': P_hat, 'aprendio': fiab > 0}


class MonitorDetector:
    """Vigila si el detector de ErrP sigue comportandose como se calibro.
    Como el orquestador conoce la meta, sabe si cada paso fue erroneo;
    compara eso con lo que dijo el detector (CUSUM en nats).
      H0: detector sano      (sens, espec calibradas)
      H1: detector degradado (casi al azar)"""

    def __init__(self, sensibilidad=config.SENS, especificidad=config.ESPEC,
                 sens_malo=0.5, espec_malo=0.6,
                 umbral_congelar=config.CUSUM_CONGELAR,
                 umbral_reanudar=config.CUSUM_REANUDAR):
        self.s0, self.e0 = sensibilidad, especificidad
        self.s1, self.e1 = sens_malo, espec_malo
        self.alto, self.bajo = umbral_congelar, umbral_reanudar
        self.cusum = 0.0
        self.congelado = False

    @staticmethod
    def _p(hay_errp, erroneo, s, e):
        if erroneo:
            return s if hay_errp else 1 - s
        return (1 - e) if hay_errp else e

    def __call__(self, erroneo, hay_errp, valido):
        """Devuelve la fiabilidad (0 a 1) con la que el agente debe aprender."""
        if valido:
            llr = np.log(self._p(hay_errp, erroneo, self.s1, self.e1)
                         / self._p(hay_errp, erroneo, self.s0, self.e0))
            # tope en 2x el umbral: si el detector se recupera, se reanuda pronto
            self.cusum = min(max(0.0, self.cusum + llr), 2 * self.alto)
        if self.cusum > self.alto:
            self.congelado = True
        elif self.cusum < self.bajo:
            self.congelado = False
        if self.congelado:
            return 0.0
        return float(np.clip(1 - self.cusum / self.alto, 0, 1))