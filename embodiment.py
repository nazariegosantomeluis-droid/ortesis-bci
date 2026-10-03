"""Tarea 2 minima: indice de integracion corporal (IIC). METRICA EXPLORATORIA, no validada.

Firma principal: atenuacion sensorial. Si el cerebro predice las consecuencias de sus propios
movimientos, la respuesta visual temprana (N1 occipital, PO7/Oz/PO8) a un movimiento propio de
la ortesis deberia ser menor que a uno ajeno (anunciado en pantalla, pero no decidido por el
piloto). El IIC es un tamano de efecto (d de Cohen):

    IIC = (N1 de los ajenos - N1 de los propios correctos) / desviacion estandar combinada

Solo entran los propios correctos: los ajenos van siempre hacia la meta, asi el contraste no se
confunde con la respuesta al error. Intervalo bootstrap remuestreando epocas dentro de cada
grupo; tendencia = IIC de la segunda mitad de la sesion menos el de la primera.
"""
import numpy as np

import config

FS = config.FLUJOS['EEG'][2]


def amplitud_n1(epoca, fs=FS, ventana=config.VENTANA_N1, canales=config.CANALES_N1,
                antes=-config.EPOCA_ERRP[0]):
    """Amplitud de la N1 en uV (positiva = N1 mayor): menos el promedio de PO7/Oz/PO8 en la
    ventana. La epoca viene filtrada y con linea base (hardware.cortar_epoca)."""
    i0, i1 = (int(round((antes + v) * fs)) for v in ventana)
    return float(-np.mean(np.asarray(epoca)[config.indices(canales), i0:i1]))


def _d(a, b):
    """d de Cohen de a contra b, con la desviacion combinada; 0 si no hay variacion."""
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / s) if s > 0 else 0.0


def _d_boot(a, b, rng, n_boot):
    """n_boot remuestreos de _d(a, b), remuestreando dentro de cada grupo."""
    A = a[rng.integers(0, len(a), (n_boot, len(a)))]
    B = b[rng.integers(0, len(b), (n_boot, len(b)))]
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * A.var(1, ddof=1) + (nb - 1) * B.var(1, ddof=1)) / (na + nb - 2))
    return (A.mean(1) - B.mean(1)) / np.where(s > 0, s, np.inf)


class IndiceEmbodiment:
    """Acumula la N1 de cada movimiento y estima el IIC con su intervalo y su tendencia."""

    def __init__(self, nivel=0.90, n_boot=1000, semilla=0, minimo=config.IIC_MIN_EPOCAS):
        self.nivel, self.n_boot, self.semilla, self.minimo = nivel, n_boot, semilla, minimo
        self.obs = []                       # (paso, amplitud, ajeno)

    def observar(self, paso, amplitud, ajeno, correcto=True):
        """Una epoca ajena, o propia (solo cuentan las correctas). Lo no finito no cuenta."""
        if amplitud is None or amplitud == '' or not np.isfinite(amplitud) or (not ajeno and not correcto):
            return
        self.obs.append((paso, float(amplitud), bool(ajeno)))

    @staticmethod
    def _grupos(obs):
        return (np.array([x for _, x, aj in obs if aj]), np.array([x for _, x, aj in obs if not aj]))

    def _alcanza(self, ajenos, propios, fraccion=1.0):
        return len(propios) >= self.minimo[0] * fraccion and len(ajenos) >= max(2, self.minimo[1] * fraccion)

    def estimar(self, con_ic=True):
        """{'iic', 'ic', 'n_propios', 'n_ajenos', 'tendencia', 'ic_tendencia'}. iic es None si
        faltan epocas; la tendencia, si falta alguna mitad. Sin con_ic solo los puntos."""
        a, p = self._grupos(self.obs)
        r = {'iic': None, 'ic': None, 'n_propios': len(p), 'n_ajenos': len(a),
             'tendencia': None, 'ic_tendencia': None}
        if not self._alcanza(a, p):
            return r
        rng = np.random.default_rng(self.semilla)
        cola = (1 - self.nivel) / 2
        intervalo = lambda v: tuple(float(q) for q in np.quantile(v, [cola, 1 - cola]))
        r['iic'] = _d(a, p)
        if con_ic:
            r['ic'] = intervalo(_d_boot(a, p, rng, self.n_boot))
        corte = np.median([k for k, _, _ in self.obs])
        (a1, p1), (a2, p2) = (self._grupos([o for o in self.obs if (o[0] > corte) == segunda])
                              for segunda in (False, True))
        if self._alcanza(a1, p1, 0.5) and self._alcanza(a2, p2, 0.5):
            r['tendencia'] = _d(a2, p2) - _d(a1, p1)
            if con_ic:
                r['ic_tendencia'] = intervalo(_d_boot(a2, p2, rng, self.n_boot) - _d_boot(a1, p1, rng, self.n_boot))
        return r


def desde_epocas(X, ajeno, correcto, fs=FS, **kw):
    """IndiceEmbodiment con las epocas X (n, canales, muestras) en orden de sesion."""
    ind = IndiceEmbodiment(**kw)
    for k, (x, aj, ok) in enumerate(zip(X, ajeno, correcto)):
        ind.observar(k, amplitud_n1(x, fs), bool(aj), bool(ok))
    return ind


def texto(r):
    """Resumen de una linea del IIC, con su etiqueta de exploratorio."""
    if r['iic'] is None:
        return (f"IIC (exploratorio): sin estimar ({r['n_propios']} propios correctos, {r['n_ajenos']} ajenos; "
                f"hacen falta {config.IIC_MIN_EPOCAS[0]} y {config.IIC_MIN_EPOCAS[1]})")
    s = f"IIC (exploratorio, atenuacion de la N1): {r['iic']:+.2f}"
    if r['ic']:
        s += ' [{:+.2f}, {:+.2f}]'.format(*r['ic'])
    s += f" con {r['n_propios']} propios correctos y {r['n_ajenos']} ajenos"
    if r['tendencia'] is not None:
        s += f"; tendencia {r['tendencia']:+.2f}" + (' [{:+.2f}, {:+.2f}]'.format(*r['ic_tendencia'])
                                                       if r['ic_tendencia'] else '')
    return s
