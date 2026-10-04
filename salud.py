"""Salud del lazo: un semaforo por subsistema y el retroceso de las reconexiones.

Vigilante no toca hardware ni LSL: recibe lecturas y un tiempo, y decide
colores. Asi se prueba completo sin casco. Los umbrales viven en config.SALUD.

Escalera de degradacion (escalon()):
  1. todo bien
  2. detector poco fiable o reloj dudoso: el lazo sigue, el agente no aprende
  3. EEG perdido o canal despegado: pausa segura
  4. ortesis perdida: pausa segura sin poder moverla (registro y aviso)
"""
from collections import deque

import numpy as np

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


class RelojContador:
    """Reconstruye la hora de cada muestra con el contador de muestras del casco.

    Por Bluetooth las muestras llegan a rafagas: la hora de llegada tiene decenas de ms
    de jitter y no sirve para cortar una epoca. El contador si: la muestra c se tomo en
    desfase + c / fs. El desfase se estima como el MINIMO de (llegada - c / fs) en una
    ventana deslizante: el retardo de transporte nunca es negativo, asi que su minimo es
    un retardo fijo, y la ventana deja seguir la deriva del cristal del casco.

    - Las muestras perdidas quedan como un hueco en la hora (no se disimulan) y se cuentan.
    - Si el contador retrocede, el casco se reinicio: se vuelve a anclar y se cuenta.
    - El desfase se corrige de a poco (20 % de un periodo por muestra): la hora nunca
      retrocede.
    modulo: None para un contador que no da la vuelta (Unicorn); 256 para el del Cyton o el
    de la placa sintetica de BrainFlow.
    """

    def __init__(self, fs, ventana_s=30.0, modulo=None):
        self.fs, self.ventana_s, self.modulo = float(fs), ventana_s, modulo
        self.perdidas = self.reinicios = 0
        self._candidatos = deque()                # (llegada, llegada - c / fs) por bloque
        self._desfase = None
        self._crudo = self._abs = None            # ultimo contador visto y su valor desenrollado
        self._ultima_hora = -np.inf

    def _anclar(self):
        self._candidatos.clear()
        self._desfase = None

    def estampar(self, contadores, t_llegada):
        """Hora (mismo reloj que t_llegada) de cada muestra del bloque que acaba de llegar."""
        c = np.asarray(contadores, dtype=np.int64)
        if c.size == 0:
            return np.empty(0)
        absolutos = np.empty(c.size, dtype=np.int64)
        for i, v in enumerate(c):                 # desenrollar el contador y contar perdidas
            if self._crudo is None:
                self._abs = int(v) if self.modulo is None else 0
            else:
                salto = int(v) - self._crudo
                if self.modulo is not None:
                    salto %= self.modulo
                    salto = salto or 1            # contador repetido: se toma como la siguiente
                if salto <= 0:                    # retrocedio: el casco se reinicio
                    self.reinicios += 1
                    self._anclar()
                    salto = 1
                self.perdidas += salto - 1
                self._abs += salto
            self._crudo = int(v)
            absolutos[i] = self._abs
        candidato = t_llegada - absolutos[-1] / self.fs
        if self._desfase is None:                 # primer bloque, o tras un reinicio
            base = max(candidato, self._ultima_hora + 1 / self.fs - absolutos[0] / self.fs)
            self._desfase = base
        self._candidatos.append((t_llegada, candidato))
        while t_llegada - self._candidatos[0][0] > self.ventana_s:
            self._candidatos.popleft()
        objetivo = min(v for _, v in self._candidatos)
        paso = 0.2 / self.fs                      # correccion maxima por muestra
        horas = np.empty(c.size)
        for i, a in enumerate(absolutos):
            self._desfase += float(np.clip(objetivo - self._desfase, -paso, paso))
            horas[i] = max(self._desfase + a / self.fs, self._ultima_hora + 1e-6)
            self._ultima_hora = horas[i]
        return horas


class RegistroHuecos:
    """Cuenta los huecos del flujo de la placa, para medir el dongle con el casco real.

    - Rafagas de muestras perdidas: saltos del contador de la placa. El del Unicorn no da
      la vuelta (modulo=None); el del Cyton y el de la placa sintetica van de 0 a 255
      (modulo=256). Una rafaga de N muestras dura N / fs.
    - Silencios de llegada: los datos tardaron mas de `silencio_s` en llegar al puente.
    resumen() se imprime cada `periodo_s` y da la ventana reciente y el total de la sesion.
    """

    def __init__(self, fs, modulo=256, silencio_s=0.1, periodo_s=30.0):
        self.fs, self.modulo, self.silencio_s, self.periodo_s = fs, modulo, silencio_s, periodo_s
        self.huecos, self.silencios = [], []      # ventana actual: (t, muestras, s) y (t, s)
        self.muestras = 0                         # recibidas en toda la sesion
        self.total_rafagas = self.total_perdidas = self.total_silencios = self.reinicios = 0
        self._ultimo = self._t_bloque = self._t0 = self._t_resumen = None

    def reiniciar_contador(self):
        """Tras reconectar la placa su contador empieza de nuevo: ese salto no es un hueco."""
        self._ultimo = self._t_bloque = None

    def bloque(self, contador, t):
        """contador: el canal de numero de paquete del bloque recien leido; t: cuando llego (s)."""
        c = [int(v) for v in contador]
        if not c:
            return
        if self._t0 is None:
            self._t0 = self._t_resumen = t
        if self._t_bloque is not None and t - self._t_bloque >= self.silencio_s:
            self.silencios.append((t, t - self._t_bloque))
            self.total_silencios += 1
        previo = self._ultimo
        for v in c:
            if previo is not None and v != previo:           # un contador repetido no es un hueco
                if self.modulo is not None:
                    perdidas = (v - previo - 1) % self.modulo
                elif v > previo:
                    perdidas = v - previo - 1
                else:                                        # sin vuelta y retrocede: reinicio del casco
                    perdidas = 0
                    self.reinicios += 1
                if perdidas:
                    self.huecos.append((t, perdidas, perdidas / self.fs))
                    self.total_rafagas += 1
                    self.total_perdidas += perdidas
            previo = v
        self._ultimo, self._t_bloque = previo, t
        self.muestras += len(c)

    def toca_resumen(self, t):
        return self._t_resumen is not None and t - self._t_resumen >= self.periodo_s

    def resumen(self, t):
        """Texto de la ventana que termina en t (y la reinicia) mas el acumulado de la sesion."""
        dur = [1000 * d for _, _, d in self.huecos]
        perdidas = sum(n for _, n, _ in self.huecos)
        txt = f'  huecos (ultimos {t - self._t_resumen:.0f} s): {len(dur)} rafagas, {perdidas} muestras perdidas'
        if dur:
            txt += f', media {sum(dur) / len(dur):.0f} ms, max {max(dur):.0f} ms'
        txt += f'; {len(self.silencios)} silencios de llegada'
        if self.silencios:
            txt += f' (max {1000 * max(d for _, d in self.silencios):.0f} ms)'
        minutos = max(t - self._t0, 1e-9) / 60
        total = self.muestras + self.total_perdidas
        txt += (f' | sesion: {self.total_rafagas} rafagas ({self.total_rafagas / minutos:.1f} por min), '
                f'{self.total_perdidas} muestras perdidas ({100 * self.total_perdidas / max(total, 1):.2f} %), '
                f'{self.total_silencios} silencios'
                + (f', {self.reinicios} reinicios del contador' if self.reinicios else ''))
        self.huecos, self.silencios, self._t_resumen = [], [], t
        return txt


class Vigilante:
    """Semaforo VERDE / AMARILLO / ROJO por subsistema.

    Lecturas que recibe actualizar() (las que falten no cambian su semaforo):
      eeg      = {'edad_s', 'tasa_hz', 'canales': {electrodo: motivo}}
      ortesis  = {'puerto_ok', 'acks_perdidos', 'latencia_ms'} (y, por Wi-Fi, 'paro' y 'bloqueo')
      reloj_ms = deriva del retraso del EEG contra su linea base
      detector = {'fiabilidad', 'congelado', 'epocas'}
      alfa     = potencia alfa occipital (semaforo PILOTO)

    El detector empieza en CALENTANDO: con menos de `detector_epocas_min` epocas
    validas su fiabilidad viva todavia es ruido, asi que no opina ni emite cambios.
    Esto solo afecta al semaforo; el congelamiento del aprendizaje no cambia.
    El piloto tambien: mide su linea base de alfa durante `piloto_base_s` y despues compara.
    Solo avisa (somnolencia, ojos cerrados o desconexion de la tarea): no pausa ni cuenta
    en la escalera de degradacion.
    """

    def __init__(self, umbrales=None):
        self.u = umbrales or config.SALUD
        self.colores = {s: V for s in config.SUBSISTEMAS}
        self.colores['detector'] = self.colores['piloto'] = config.CALENTANDO
        self.detalle = {s: '' for s in config.SUBSISTEMAS}
        self._canal_malo = False
        self._t_verde = None
        self._alfa_t0, self._alfa_base, self._alfa_linea = None, [], None
        self.alfa_rel = None             # ultimo alfa occipital contra su linea base (None: aun calienta)

    def actualizar(self, t, eeg=None, ortesis=None, reloj_ms=None, detector=None, alfa=None):
        """Devuelve los cambios de color [(subsistema, color)], para publicarlos como marcador."""
        previos = dict(self.colores)
        if eeg is not None:
            self._eeg(eeg)
        if ortesis is not None:
            self._ortesis(ortesis)
        if reloj_ms is not None:
            self._poner('reloj', self._nivel(abs(reloj_ms), 'reloj_amarillo_ms', 'reloj_rojo_ms'),
                        f'deriva {reloj_ms:+.0f} ms')
        if detector is not None and detector['epocas'] >= self.u['detector_epocas_min']:
            color = R if detector['congelado'] else (
                A if detector['fiabilidad'] < self.u['detector_amarillo'] else V)
            self._poner('detector', color, f"fiabilidad {detector['fiabilidad']:.2f}")
        if alfa is not None and alfa > 0:
            self._piloto(t, alfa)
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
        # solo cuenta el deficit: una rafaga de muestras atrasadas que llegan juntas no es una falla
        nominal = config.FLUJOS['EEG'][2]
        c_tasa = self._nivel(max(0.0, nominal - e['tasa_hz']) / nominal, 'eeg_tasa_amarillo', 'eeg_tasa_rojo')
        self._poner('eeg', c_tasa, f"tasa {e['tasa_hz']:.0f} Hz")

    def _piloto(self, t, alfa):
        if self._alfa_t0 is None:
            self._alfa_t0 = t
        if t - self._alfa_t0 < self.u['piloto_base_s']:
            self._alfa_base.append(alfa)                 # sigue en CALENTANDO
            return
        if self._alfa_linea is None:
            self._alfa_linea = float(np.median(self._alfa_base)) if self._alfa_base else alfa
        r = self.alfa_rel = alfa / self._alfa_linea
        self._poner('piloto', self._nivel(r, 'piloto_amarillo', 'piloto_rojo'),
                    f'alfa occipital x{r:.1f} de su linea base (somnolencia u ojos cerrados?)')

    def _ortesis(self, o):
        if not o['puerto_ok']:
            return self._poner('ortesis', R, 'puerto caido')
        if o.get('paro'):                                   # Wi-Fi: el paro de emergencia de la ortesis esta oprimido
            return self._poner('ortesis', R, 'paro de emergencia oprimido')
        if o['acks_perdidos'] >= self.u['acks_rojo']:
            return self._poner('ortesis', R, f"{o['acks_perdidos']} ACK perdidos seguidos")
        if o['acks_perdidos'] >= self.u['acks_amarillo']:
            return self._poner('ortesis', A, 'ACK perdido')
        if o.get('bloqueo'):                                # Wi-Fi: la corriente de los servos paso del limite
            return self._poner('ortesis', A, 'bloqueo por corriente: la ortesis limita el cierre')
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
