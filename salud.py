"""Salud del lazo: un semaforo por subsistema y el retroceso de las reconexiones.

Vigilante no toca hardware ni LSL: recibe lecturas y un tiempo, y decide
colores. Asi se prueba completo sin casco. Los umbrales viven en config.SALUD.

Escalera de degradacion (escalon()):
  1. todo bien
  2. detector poco fiable o reloj dudoso: el lazo sigue, el agente no aprende
  3. EEG perdido o canal despegado: pausa segura
  4. ortesis perdida: pausa segura sin poder moverla (registro y aviso)
"""
import config

V, A, R = config.VERDE, config.AMARILLO, config.ROJO


class Retroceso:
    """Esperas de reconexion: inicial, x2, x2... hasta el maximo."""

    def __init__(self, inicial=config.RECONEXION_INICIAL_S, maximo=config.RECONEXION_MAX_S):
        self.inicial, self.maximo = inicial, maximo
        self.reiniciar()

    def reiniciar(self):
        self._espera = self.inicial

    def siguiente(self):
        e = self._espera
        self._espera = min(self.maximo, 2 * e)
        return e


class Vigilante:
    """Semaforo VERDE / AMARILLO / ROJO por subsistema.

    Lecturas que recibe actualizar() (las que falten no cambian su semaforo):
      eeg      = {'edad_s', 'tasa_hz', 'canales': {electrodo: motivo}}
      ortesis  = {'puerto_ok', 'acks_perdidos', 'latencia_ms'}
      reloj_ms = deriva del retraso del EEG contra su linea base
      detector = {'fiabilidad', 'congelado'}
    """

    def __init__(self, umbrales=None):
        self.u = umbrales or config.SALUD
        self.colores = {s: V for s in config.SUBSISTEMAS}
        self.detalle = {s: '' for s in config.SUBSISTEMAS}
        self._canal_malo = False
        self._t_verde = None

    def actualizar(self, t, eeg=None, ortesis=None, reloj_ms=None, detector=None):
        """Devuelve los cambios de color [(subsistema, color)], para publicarlos como marcador."""
        previos = dict(self.colores)
        if eeg is not None:
            self._eeg(eeg)
        if ortesis is not None:
            self._ortesis(ortesis)
        if reloj_ms is not None:
            self._poner('reloj', self._nivel(abs(reloj_ms), 'reloj_amarillo_ms', 'reloj_rojo_ms'),
                        f'deriva {reloj_ms:+.0f} ms')
        if detector is not None:
            color = R if detector['congelado'] else (
                A if detector['fiabilidad'] < self.u['detector_amarillo'] else V)
            self._poner('detector', color, f"fiabilidad {detector['fiabilidad']:.2f}")
        # el detector no cuenta para reanudar: en pausa no hay pasos con que recuperarlo
        bien = self.colores['eeg'] == V and self.colores['ortesis'] == V and self.colores['reloj'] != R
        if not bien:
            self._t_verde = None
        elif self._t_verde is None:
            self._t_verde = t
        return [(s, c) for s, c in self.colores.items() if c != previos[s]]

    def _nivel(self, x, k_amarillo, k_rojo):
        return R if x >= self.u[k_rojo] else (A if x >= self.u[k_amarillo] else V)

    def _poner(self, sub, color, texto):
        self.colores[sub] = color
        self.detalle[sub] = '' if color == V else texto

    def _eeg(self, e):
        self._canal_malo = bool(e['canales'])
        if e['canales']:
            return self._poner('eeg', R, '; '.join(f'{c} {m}' for c, m in e['canales'].items()))
        c_edad = self._nivel(e['edad_s'], 'eeg_edad_amarillo_s', 'eeg_edad_rojo_s')
        if c_edad != V:
            return self._poner('eeg', c_edad, f"sin muestras hace {e['edad_s']:.1f} s")
        nominal = config.FLUJOS['EEG'][2]
        c_tasa = self._nivel(abs(e['tasa_hz'] - nominal) / nominal, 'eeg_tasa_amarillo', 'eeg_tasa_rojo')
        self._poner('eeg', c_tasa, f"tasa {e['tasa_hz']:.0f} Hz")

    def _ortesis(self, o):
        if not o['puerto_ok']:
            return self._poner('ortesis', R, 'puerto caido')
        if o['acks_perdidos'] >= self.u['acks_rojo']:
            return self._poner('ortesis', R, f"{o['acks_perdidos']} ACK perdidos seguidos")
        if o['acks_perdidos'] >= self.u['acks_amarillo']:
            return self._poner('ortesis', A, 'ACK perdido')
        if o['latencia_ms'] >= self.u['latencia_pico_ms']:
            return self._poner('ortesis', A, f"latencia {o['latencia_ms']:.0f} ms")
        self._poner('ortesis', V, '')

    def codigo(self):
        """Una letra por subsistema, en el orden del contrato (columna 'salud' del CSV)."""
        return ''.join(self.colores[s][0] for s in config.SUBSISTEMAS)

    def motivo_pausa(self):
        """None, 'eeg', 'canal' u 'ortesis'."""
        if self.colores['ortesis'] == R:
            return 'ortesis'
        if self.colores['eeg'] == R:
            return 'canal' if self._canal_malo else 'eeg'
        return None

    def escalon(self):
        if self.colores['ortesis'] == R:
            return 4
        if self.colores['eeg'] == R:
            return 3
        return 2 if R in (self.colores['reloj'], self.colores['detector']) else 1

    def listo_para_reanudar(self, t):
        return self._t_verde is not None and t - self._t_verde >= self.u['verde_para_reanudar_s']
