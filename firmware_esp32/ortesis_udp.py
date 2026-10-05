"""Cliente UDP (Wi-Fi) de referencia para la Órtesis Adaptrode (firmware 1.2 o más nuevo). Solo usa la biblioteca estándar.
Para el cable USB, que tiene menos latencia, está OrtesisUSB en ortesis_usb.py, con la misma interfaz.

Pensado para envolverse en hardware.py con la misma interfaz que OrtesisSerial:
  - mover(cierre, pulgar, p) manda una orden con seq nuevo, espera el ACK y devuelve la ESTAMPA del movimiento
    en el reloj de la laptop (time.monotonic, en segundos):
        {'seq', 't_envio', 't_aplicado', 'rtt_ms', 'metodo': 'ack' | 'respaldo'}
    't_aplicado' sale de la hora de la ESP32 en el ACK, convertida al reloj de la laptop. Si el ACK no llega,
    el respaldo es t_envio + la latencia medida, con metodo='respaldo' (márquenlo así en el CSV).
  - Un hilo de LATIDO reenvía el estado cada 100 ms para que la vigilancia del firmware (0.5 s) no abra la mano
    entre pasos. Cada latido lleva seq nuevo, así que también trae su ACK: con ellos se sincronizan los relojes.
  - p (probabilidad del decoder) viaja en cada latido y sube la barra del nervio de luz; errp() lo hace destellar en rojo.
  - telemetria() devuelve la última telemetría con 't_local' (hora de la muestra en el reloj de la laptop).
  - on_paro(funcion) avisa cuando cambia el paro de emergencia (por ejemplo, para pasar a PAUSA_SEGURA).
"""
import json
import socket
import statistics
import threading
import time
from collections import deque


class OrtesisUDP:
    def __init__(self, ip='192.168.4.1', puerto=8888, latido_s=0.1, espera_ack_s=0.15, reloj=time.monotonic):
        self.dest = (ip, puerto)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.settimeout(0.05)
        self._iniciar(latido_s, espera_ack_s, reloj)

    # ------------------------------------------------------------------ transporte (OrtesisUSB lo reemplaza)
    def _tx(self, datos):
        self.s.sendto(datos, self.dest)

    def _rx(self):
        try:
            return self.s.recvfrom(2048)[0]
        except socket.timeout:
            return None

    def _cerrar_transporte(self):
        self.s.close()

    def _iniciar(self, latido_s, espera_ack_s, reloj):
        self.latido_s, self.espera_ack_s, self.reloj = latido_s, espera_ack_s, reloj
        self.estado = {'cierre': 0.0, 'pulgar': 0.0, 'p': 0.5}
        self.seq = 0
        self._enviados = {}                       # seq → hora de envío
        self._acks = {}                           # seq → (hora de llegada del ACK, t_ms de la ESP32)
        self._muestras = deque(maxlen=300)        # (rtt, desfase) de los últimos ~30 s, para seguir la deriva del reloj
        self.tel, self.t_tel, self.paro = {}, None, None
        self._cb_paro = None
        self._cv = threading.Condition()
        self._vivo = True
        threading.Thread(target=self._escuchar, daemon=True).start()
        threading.Thread(target=self._latir, daemon=True).start()

    # ------------------------------------------------------------------ envío
    def _enviar(self, extra=None):
        with self._cv:
            self.seq += 1
            seq = self.seq
            mensaje = dict(self.estado, seq=seq, **(extra or {}))
            self._enviados[seq] = self.reloj()
            if len(self._enviados) > 600:
                for k in sorted(self._enviados)[:200]:
                    self._enviados.pop(k, None)
                    self._acks.pop(k, None)
        try:
            self._tx(json.dumps(mensaje).encode())
        except OSError:
            pass
        return seq

    def _latir(self):
        while self._vivo:
            self._enviar()
            time.sleep(self.latido_s)

    # ------------------------------------------------------------------ recepción
    def _escuchar(self):
        while self._vivo:
            try:
                datos = self._rx()
            except OSError:
                if not self._vivo:
                    break
                time.sleep(0.05)
                continue
            if not datos:
                continue
            t = self.reloj()
            try:
                m = json.loads(datos)
            except ValueError:                     # por USB también llegan los mensajes de texto de la ESP32
                continue
            if not isinstance(m, dict):
                continue
            if 'ack' in m:
                with self._cv:
                    te = self._enviados.get(m['ack'])
                    if te is not None:
                        rtt = t - te
                        self._muestras.append((rtt, m['t_ms'] / 1000.0 - (te + rtt / 2)))
                        self._acks[m['ack']] = (t, m['t_ms'])
                        self._cv.notify_all()
            else:
                self.tel, self.t_tel = m, t
                paro = bool(m.get('paro'))
                if paro != self.paro:
                    self.paro = paro
                    if self._cb_paro:
                        try:
                            self._cb_paro(paro)
                        except Exception as e:          # un error del usuario no debe tumbar la recepción
                            print('on_paro falló:', e)

    # ------------------------------------------------------------------ reloj
    def desfase(self):
        """Desfase reloj ESP32 − reloj laptop (s), tomado de la muestra con menor RTT de los últimos ~30 s."""
        with self._cv:
            if not self._muestras:
                return None
            return min(self._muestras)[1]

    def a_local(self, t_ms):
        d = self.desfase()
        return None if d is None else t_ms / 1000.0 - d

    def latencia(self):
        """RTT en ms de los últimos ~30 s: mínimo, mediana y percentil 95."""
        with self._cv:
            r = sorted(x[0] * 1000 for x in self._muestras)
        if not r:
            return None
        return {'min': round(r[0], 1), 'mediana': round(statistics.median(r), 1),
                'p95': round(r[min(len(r) - 1, int(0.95 * len(r)))], 1), 'n': len(r)}

    # ------------------------------------------------------------------ interfaz
    def mover(self, cierre=None, pulgar=None, p=None, esperar=True):
        for k, v in (('cierre', cierre), ('pulgar', pulgar), ('p', p)):
            if v is not None:
                self.estado[k] = max(0.0, min(1.0, float(v)))
        seq = self._enviar()
        te = self._enviados[seq]
        ack = None
        if esperar:
            with self._cv:
                self._cv.wait_for(lambda: seq in self._acks, timeout=self.espera_ack_s)
                ack = self._acks.get(seq)
        if ack:
            t_llega, t_ms = ack
            t_apl = self.a_local(t_ms)
            if t_apl is None:
                t_apl = te + (t_llega - te) / 2
            return {'seq': seq, 't_envio': te, 't_aplicado': t_apl, 'rtt_ms': round((t_llega - te) * 1000, 1), 'metodo': 'ack'}
        lat = self.latencia()
        media_s = (lat['mediana'] / 2000.0) if lat else 0.01
        return {'seq': seq, 't_envio': te, 't_aplicado': te + media_s, 'rtt_ms': None, 'metodo': 'respaldo'}

    def set_p(self, p):
        self.estado['p'] = max(0.0, min(1.0, float(p)))      # viaja en el siguiente latido

    def errp(self):
        self._enviar({'errp': 1})

    def vibrar(self, ms=120):
        self._enviar({'vib': int(ms)})

    def calibrar(self, servo, abierta, cerrada):
        self._enviar({'cmd': 'calibrar', 'servo': servo, 'abierta': abierta, 'cerrada': cerrada})

    def on_paro(self, funcion):
        self._cb_paro = funcion

    def telemetria(self):
        tel = dict(self.tel)
        if 't_ms' in tel:
            tel['t_local'] = self.a_local(tel['t_ms'])
        if self.t_tel is not None:
            tel['edad_s'] = round(self.reloj() - self.t_tel, 3)
        return tel

    def conectada(self, max_edad_s=0.5):
        return self.t_tel is not None and self.reloj() - self.t_tel < max_edad_s

    def cerrar(self, abrir=True):
        if abrir:
            self.mover(0.0, 0.0, 0.5, esperar=False)
            time.sleep(0.2)
        self._vivo = False
        time.sleep(0.1)
        self._cerrar_transporte()
