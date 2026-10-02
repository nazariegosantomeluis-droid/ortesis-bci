"""Ingenieria del caos para el lazo: fallas reproducibles por semilla.

PlanCaos no rompe nada por si mismo: solo responde "que falla toca ahora". Lo
consultan el gemelo (cerebro_sintetico.py --caos), OrtesisSimulada y BackendSim
(orquestador.py ... --caos), cada uno con su propio reloj.

  - Fallas por tiempo (corte_eeg, rafaga_parpadeos, canal): una linea de tiempo
    por tipo, generada con su propio generador. La respuesta no depende de
    cuando ni en que orden se pregunte.
  - Fallas por paso (ack_perdido, pico_latencia): dependen solo de
    (semilla, tipo, seq).

Las tasas del "caos estandar" viven en config.CAOS_ESTANDAR.
"""
import numpy as np

import config

_TIPOS = ['corte_eeg', 'rafaga_parpadeos', 'canal', 'ack_perdido', 'pico_latencia']
INICIO_S = 10.0          # sin fallas por tiempo antes de este instante
SEPARACION_S = 5.0       # minimo entre dos fallas del mismo tipo


class PlanCaos:
    def __init__(self, semilla, tasas=None):
        self.semilla, self.tasas = int(semilla), tasas or config.CAOS_ESTANDAR
        self._lineas = {}

    def _rng(self, tipo, *extra):
        return np.random.default_rng([self.semilla, _TIPOS.index(tipo), *extra])

    def _linea(self, tipo, hasta):
        """Eventos del tipo hasta el instante `hasta`; se extiende sin cambiar lo ya generado."""
        l = self._lineas.setdefault(tipo, {'rng': self._rng(tipo), 't': INICIO_S, 'ev': []})
        c = self.tasas[tipo]
        while l['t'] <= hasta:
            t0 = l['t'] + float(l['rng'].exponential(c['cada_s']))
            ev = (t0, float(l['rng'].uniform(*c['duracion_s'])))
            if tipo == 'canal':
                ev += (config.CANALES_EEG[int(l['rng'].integers(len(config.CANALES_EEG)))],
                       ['plano', 'ruidoso'][int(l['rng'].integers(2))])
            l['ev'].append(ev)
            l['t'] = ev[0] + ev[1] + SEPARACION_S
        return l['ev']

    def activo(self, tipo, t):
        """La falla por tiempo activa en t, o None.
        corte_eeg y rafaga_parpadeos: (t0, duracion); canal: (t0, duracion, electrodo, modo)."""
        for ev in self._linea(tipo, t):
            if ev[0] <= t < ev[0] + ev[1]:
                return ev
        return None

    def por_paso(self, tipo, seq):
        """ack_perdido -> bool; pico_latencia -> ms o None."""
        r, c = self._rng(tipo, int(seq)), self.tasas[tipo]
        if tipo == 'ack_perdido':
            return bool(r.random() < c['p'])
        return float(r.uniform(*c['ms'])) if r.random() < c['p'] else None
