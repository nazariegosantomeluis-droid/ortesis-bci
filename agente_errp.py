"""Agente que corrige en linea el decoder de imaginacion motora usando el ErrP.

    z   = w0 . phi + c0 + o          logit del decoder (o = perturbacion de la demo)
    p'  = sigma(z + beta)            probabilidad corregida de "cerrar"
    d   = +1 si p' >= 0.5, si no -1

El agente aprende un parametro, beta, tratado como el estado de un filtro de
Kalman (media + varianza). Ideas propias:

1. P_hat bayesiano con la confiabilidad VIVA del detector: la salida del ErrP
   se convierte en probabilidad posterior de error usando la sensibilidad y
   especificidad que el detector tiene AHORA (ConfianzaDetector), no las de la
   calibracion. Con un detector de probabilidades calibradas usa la salida
   completa y corrige el cambio de prior entre calibracion y lazo.
2. Varianza como tasa de aprendizaje: grande con incertidumbre, chica al
   converger. No hay eta que elegir.
3. Dos detectores de cambio que re-inflan la varianza:
   - sesgo de decisiones (rapido; asume bloques cerrar/relajar balanceados),
   - chequeo predictivo (general): el agente predice cuantos ErrP deberia
     provocar segun su propia confianza; si aparecen mas de los predichos de
     forma sostenida, esta seguro y equivocado, y vuelve a aprender rapido.
4. Paso visible: el paso nunca baja del umbral que el piloto percibe (se
   mide con el piloto; config.PASO_VISIBLE). Sin error visible no hay ErrP.
5. Cotas de seguridad: cambio maximo por paso y |beta| maximo.

Modos: 'bayes' (el de la demo), 'fijo' (eta constante, linea base) y
'estatico' (no aprende).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config

__all__ = ['sigmoide', 'logit', 'Decision', 'ConfigAgente', 'AgenteErrP', 'ConfianzaDetector']


def sigmoide(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return float(np.log(p / (1 - p)))


@dataclass
class Decision:
    p_prima: float
    direccion: int          # +1 cerrar, -1 relajar
    delta: float            # paso con signo, fraccion del rango (0 abierta, 1 cerrada)
    z: float                # logit crudo del decoder
    direccion_sombra: int   # lo que haria el decoder calibrado sin corregir
    explorando: bool        # el paso se agrando hasta el umbral visible


@dataclass
class ConfigAgente:
    modo: str = 'bayes'
    # detector de ErrP
    sens: float = config.SENS
    espec: float = config.ESPEC
    salida_detector: str = 'binaria'        # 'binaria' | 'calibrada'
    p_error_calibracion: float = 0.3        # tasa de errores en CAL_ERRP (para 'calibrada')
    umbral_detector: float = 0.5            # p_errp > umbral = "detecto error" (para 'binaria')
    # politica: paso informativo
    ganancia: float = 0.20
    paso_max: float = config.PASO_MAX
    paso_visible: float = config.PASO_VISIBLE
    # modo 'fijo'
    eta_beta: float = config.ETA_BETA
    # filtro de Kalman
    varianza_inicial: float = 0.5
    ruido_proceso: float = 1e-3
    varianza_reinicio: float = 1.5
    info_min: float = 0.05
    # detector de cambio 1: sesgo de decisiones
    usar_sesgo: bool = True
    alfa_sesgo: float = 0.05
    umbral_sesgo: float = 0.12
    refractario: int = 40
    # detector de cambio 2: chequeo predictivo del ErrP (CUSUM)
    holgura_predictiva: float = 0.05
    umbral_predictivo: float = 3.0
    # prior de error
    prior_error: float = 0.2
    alfa_prior: float = 0.03
    # seguridad
    max_delta_beta: float = 0.6
    beta_max: float = 6.0

    def __post_init__(self):
        if self.modo not in ('bayes', 'fijo', 'estatico'):
            raise ValueError("modo debe ser 'bayes', 'fijo' o 'estatico'")
        if self.salida_detector not in ('binaria', 'calibrada'):
            raise ValueError("salida_detector debe ser 'binaria' o 'calibrada'")
        if not 0 < self.paso_visible <= self.paso_max <= 1:
            raise ValueError('se requiere 0 < paso_visible <= paso_max <= 1')


class AgenteErrP:
    def __init__(self, w0, c0=0.0, cfg: ConfigAgente | None = None):
        self.cfg = cfg or ConfigAgente()
        self.w0 = np.atleast_1d(np.asarray(w0, dtype=float)).copy()
        self.c0 = float(c0)
        self.reiniciar()

    def reiniciar(self):
        c = self.cfg
        self.beta, self.var = 0.0, c.varianza_inicial
        self.prior = c.prior_error
        self.sesgo = 0.0
        self._desde_cambio = c.refractario
        self.cusum_pred = 0.0
        self.n_cambios = {'sesgo': 0, 'prediccion': 0}
        self._pendiente: Decision | None = None

    @property
    def umbral_b(self):
        """Umbral equivalente sobre la salida cruda: p > b  <=>  p' > 0.5."""
        return float(sigmoide(-self.beta))

    # ------------------------------------------------------------ politica
    def decidir(self, phi, desplazamiento=0.0) -> Decision:
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        if phi.shape != self.w0.shape:
            raise ValueError(f'phi debe tener forma {self.w0.shape}; llego {phi.shape}')
        c = self.cfg
        z = float(self.w0 @ phi + self.c0 + desplazamiento)
        p = float(sigmoide(z + self.beta))
        d = 1 if p >= 0.5 else -1
        confianza = c.ganancia * abs(2 * p - 1)
        mag = float(np.clip(max(confianza, c.paso_visible), 0.0, c.paso_max))
        dec = Decision(p, d, d * mag, z, 1 if z >= 0 else -1, c.paso_visible > confianza)
        self._pendiente = dec
        return dec

    # ------------------------------------------------------------ percepcion del ErrP
    def prob_error(self, p_errp, sens=None, espec=None, fiabilidad=1.0):
        """P_hat: posterior de que el paso fue erroneo."""
        c = self.cfg
        a_priori = logit(self.prior)
        if c.salida_detector == 'calibrada':
            llr = logit(p_errp) - logit(c.p_error_calibracion)   # corrige el cambio de prior
            return float(sigmoide(a_priori + fiabilidad * llr))
        s = c.sens if sens is None else sens
        e = c.espec if espec is None else espec
        llr = np.log(s / (1 - e)) if p_errp > c.umbral_detector else np.log((1 - s) / e)
        return float(sigmoide(a_priori + llr))

    # ------------------------------------------------------------ aprendizaje
    def actualizar(self, p_errp, artefacto=False, fiabilidad=1.0, sens=None, espec=None) -> dict:
        if self._pendiente is None:
            raise RuntimeError('llama a decidir() antes de actualizar()')
        dec, self._pendiente = self._pendiente, None
        c = self.cfg
        info = {'beta': self.beta, 'varianza': self.var, 'P_hat': float('nan'),
                'usada': False, 'cambio': ''}
        if p_errp is None or not np.isfinite(p_errp) or artefacto:
            return info

        fiab = float(np.clip(fiabilidad, 0, 1))
        P_hat = self.prob_error(p_errp, sens, espec, fiab)
        info['P_hat'] = P_hat
        q = (1 - P_hat) if dec.direccion > 0 else P_hat     # P(la intencion era cerrar)
        g = q - dec.p_prima                                  # residuo (innovacion)

        if c.modo != 'estatico' and fiab > 0:
            if c.modo == 'fijo':
                paso = c.eta_beta * fiab * g
            else:
                s_, e_ = (c.sens, c.espec) if sens is None else (sens, espec)
                info['cambio'] = self._detectar_cambio(dec, p_errp > 0.5, s_, e_)
                self.var += c.ruido_proceso
                h = dec.p_prima * (1 - dec.p_prima) * max((2 * q - 1) ** 2, c.info_min)
                self.var = 1.0 / (1.0 / self.var + fiab * h)
                paso = fiab * self.var * g
            paso = float(np.clip(paso, -c.max_delta_beta, c.max_delta_beta))
            self.beta = float(np.clip(self.beta + paso, -c.beta_max, c.beta_max))
            info['usada'] = True

        self.prior += c.alfa_prior * (P_hat - self.prior)
        info.update(beta=self.beta, varianza=self.var)
        return info

    def _detectar_cambio(self, dec, detectado, sens, espec):
        """Re-infla la varianza cuando el modelo deja de describir al piloto.

        sesgo:      p' deberia promediar 0.5 en bloques balanceados (rapido).
        prediccion: el agente predice su propia probabilidad de equivocarse,
                    1 - max(p', 1 - p'), y con ella cuantos ErrP deberian
                    aparecer. Si aparecen mas de los predichos de forma
                    sostenida (CUSUM), el agente esta seguro y equivocado:
                    el modelo fallo. Funciona aunque las metas no esten
                    balanceadas.
        """
        c = self.cfg
        self._desde_cambio += 1
        self.sesgo += c.alfa_sesgo * ((dec.p_prima - 0.5) - self.sesgo)
        err_pred = 1 - max(dec.p_prima, 1 - dec.p_prima)
        det_pred = sens * err_pred + (1 - espec) * (1 - err_pred)
        self.cusum_pred = max(0.0, self.cusum_pred + float(detectado) - det_pred - c.holgura_predictiva)
        if self._desde_cambio < c.refractario:
            return ''
        motivo = ''
        if c.usar_sesgo and abs(self.sesgo) > c.umbral_sesgo:
            motivo = 'sesgo'
        elif self.cusum_pred > c.umbral_predictivo:
            motivo = 'prediccion'
        if motivo:
            self.var = max(self.var, c.varianza_reinicio)
            self._desde_cambio = 0
            self.cusum_pred = 0.0
            self.n_cambios[motivo] += 1
        return motivo


class ConfianzaDetector:
    """Estima en vivo la sensibilidad y especificidad del detector de ErrP.

    En sesiones con meta conocida, cada epoca dice si el detector acerto. Se
    lleva un posterior Beta por sensibilidad y otro por especificidad, con
    olvido exponencial de los datos (no del prior de calibracion). De ahi sale
    el indice de Youden J = sens + espec - 1 (0 = el detector no informa).

    - El agente usa sens/espec VIVAS para su P_hat.
    - fiabilidad = J_vivo / J_calibracion en [0, 1] escala cuanto aprende.
    - Si J cae bajo `congelar` (relativo), el aprendizaje se congela; se
      reanuda al superar `reanudar` (histeresis).
    La meta solo juzga al detector; nunca le dice al agente hacia donde moverse.
    """

    def __init__(self, sensibilidad=config.SENS, especificidad=config.ESPEC,
                 peso_previo=3.0, olvido=0.93, congelar=0.5, reanudar=0.7):
        if sensibilidad + especificidad <= 1:
            raise ValueError('un detector con sens + espec <= 1 no informa')
        self.s0, self.e0, self.n0 = sensibilidad, especificidad, peso_previo
        self.j0 = sensibilidad + especificidad - 1
        self.olvido, self.u_cong, self.u_rean = olvido, congelar, reanudar
        self.err_det = self.err_tot = self.ok_nodet = self.ok_tot = 0.0
        self.congelado = False

    @property
    def sens(self):
        return (self.n0 * self.s0 + self.err_det) / (self.n0 + self.err_tot)

    @property
    def espec(self):
        return (self.n0 * self.e0 + self.ok_nodet) / (self.n0 + self.ok_tot)

    @property
    def youden(self):
        return self.sens + self.espec - 1

    @property
    def fiabilidad_bruta(self):
        return float(np.clip(self.youden / self.j0, 0, 1))

    def __call__(self, erroneo, detectado, valido=True) -> float:
        if valido:
            l = self.olvido
            self.err_det *= l; self.err_tot *= l; self.ok_nodet *= l; self.ok_tot *= l
            if erroneo:
                self.err_tot += 1
                self.err_det += bool(detectado)
            else:
                self.ok_tot += 1
                self.ok_nodet += not detectado
        f = self.fiabilidad_bruta
        if f < self.u_cong:
            self.congelado = True
        elif f > self.u_rean:
            self.congelado = False
        return 0.0 if self.congelado else f

    def vivo(self):
        """(sens, espec) acotados para usarse en Bayes."""
        return float(np.clip(self.sens, 0.51, 0.99)), float(np.clip(self.espec, 0.51, 0.99))
