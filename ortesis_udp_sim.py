"""Firmware 1.2 de la ortesis Adaptrode simulado en la laptop: habla el mismo UDP que la ESP32
(config.py, seccion Ortesis por Wi-Fi), para probar hardware.OrtesisUDP y el orquestador sin la placa.

Modela lo que importa del firmware: ACK inmediato con su propio reloj en ms (otro origen y 30 ppm de
deriva), telemetria cada 100 ms, vigilancia (sin ordenes durante 0.5 s abre la mano despacio), servos
con velocidad limitada y, opcionalmente, el paro de emergencia y el bloqueo. Cada cambio de meta
empieza a moverse `latencia_mecanica_simulada(seq, semilla)` despues del ACK: la misma que usa
OrtesisSimulada y con la que el gemelo (cerebro_sintetico.py) ancla el ErrP.

NO es el firmware: no mide corriente ni fuerza (siempre 0) ni calibra. Solo prueba el protocolo y el reloj.

Uso:  python ortesis_udp_sim.py [--ip 127.0.0.1] [--puerto 8888] [--semilla 0]
      (el orquestador:  python orquestador.py real --ortesis-udp 127.0.0.1)
"""
import argparse
import json
import socket
import threading
import time

import config
import hardware

VEL_MAX = 90.0                  # grados por segundo (firmware: CFG['vel_max'])
VEL_RELAJAR = 30.0              # al abrir por vigilancia
VIGILANCIA_S = 0.5
ABIERTA, CERRADA = 20.0, 150.0  # angulos de fabrica de los dedos


class FirmwareSimulado:
    def __init__(self, ip='127.0.0.1', puerto=config.PUERTO_ORTESIS_UDP, semilla=0, periodo_s=0.02,
                 origen_ms=532_118, deriva_ppm=30.0, reloj=time.monotonic, mecanica=True):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind((ip, puerto))
        self.s.setblocking(False)
        self.puerto = self.s.getsockname()[1]
        self.semilla, self.periodo, self.reloj, self.mecanica = semilla, periodo_s, reloj, mecanica
        self.origen_ms, self.deriva, self.t0 = origen_ms, deriva_ppm * 1e-6, reloj()
        self.frac = 0.0                                  # fraccion real de los dedos
        self.meta, self.meta_pendiente = 0.0, None       # meta vigente y (hora de aplicarla, meta)
        self.vigilancia, self.paro, self.bloqueo = True, False, False
        self.perder_acks = set()                         # seq cuyo ACK no se manda (pruebas)
        self.cliente, self.ultimo_cmd, self.ultimo_seq_ack, self.ultima_tel = None, None, None, 0.0
        self.recibidos, self.acks = [], []               # (seq, hora de llegada) y (seq, t_ms) para las pruebas
        self.angulos_aplicados = []                      # (t_ms en que la meta empezo a regir, meta, seq)
        self._vivo = False

    def t_ms(self):
        """Reloj de la ESP32: milisegundos desde un origen propio, con deriva."""
        return int(self.origen_ms + (self.reloj() - self.t0) * 1000 * (1 + self.deriva))

    def _aplicar_meta(self, meta, seq):
        self.meta = meta
        self.angulos_aplicados.append((self.t_ms(), meta, seq))

    def _recibir(self, ahora):
        while True:
            try:
                datos, direccion = self.s.recvfrom(512)
            except (BlockingIOError, OSError):
                return
            try:
                m = json.loads(datos)
            except ValueError:
                continue
            self.cliente, self.ultimo_cmd = direccion, ahora
            seq = m.get('seq')
            self.recibidos.append((seq, ahora))
            if 'cierre' in m:
                meta = min(1.0, max(0.0, float(m['cierre'])))
                vigente = self.meta_pendiente[1] if self.meta_pendiente else self.meta
                if abs(meta - vigente) > 1e-9:
                    # un cambio de meta de un paso empieza tras la latencia mecanica; un latido, ya
                    demora = hardware.latencia_mecanica_simulada(seq, self.semilla) \
                        if self.mecanica and seq is not None and seq < config.UDP_SEQ_LATIDO else 0.0
                    self.meta_pendiente = (ahora + demora, meta, seq)
            if seq is not None and seq != self.ultimo_seq_ack:
                self.ultimo_seq_ack = seq
                if seq not in self.perder_acks:
                    t_ms = self.t_ms()
                    self.acks.append((seq, t_ms))
                    self.s.sendto(json.dumps({'ack': seq, 't_ms': t_ms}).encode(), direccion)

    def paso(self, ahora=None):
        """Un ciclo del firmware (20 ms): ordenes, vigilancia, servos y telemetria."""
        ahora = self.reloj() if ahora is None else ahora
        self._recibir(ahora)
        if self.meta_pendiente and ahora >= self.meta_pendiente[0]:
            _, meta, seq = self.meta_pendiente
            self.meta_pendiente = None
            self._aplicar_meta(meta, seq)
        self.vigilancia = self.ultimo_cmd is None or ahora - self.ultimo_cmd > VIGILANCIA_S
        meta = 0.0 if (self.vigilancia or self.paro) else self.meta
        vel = VEL_RELAJAR if self.vigilancia else VEL_MAX
        tope = vel * self.periodo / (CERRADA - ABIERTA)
        self.frac += max(-tope, min(tope, meta - self.frac))
        if self.cliente is not None and ahora - self.ultima_tel >= 0.1:
            self.ultima_tel = ahora
            tel = {'seq': self.ultimo_seq_ack or 0, 'cierre': round(self.frac, 3), 'pulgar': round(self.frac, 3),
                   'i_ma': 0, 'fuerza': 0.0, 'paro': self.paro, 'bloqueo': self.bloqueo, 'vigilancia': self.vigilancia,
                   'ang': {'dedos': [ABIERTA, CERRADA], 'pulgar': [30.0, 120.0]}, 'msg': '', 'ver': '1.2-sim',
                   't_ms': self.t_ms()}
            try:
                self.s.sendto(json.dumps(tel).encode(), self.cliente)
            except OSError:
                pass

    def correr(self):
        self._vivo = True
        while self._vivo:
            t = self.reloj()
            self.paso(t)
            time.sleep(max(0.0, self.periodo - (self.reloj() - t)))

    def iniciar(self):
        """En un hilo, para las pruebas."""
        threading.Thread(target=self.correr, daemon=True).start()
        return self

    def detener(self):
        self._vivo = False
        time.sleep(self.periodo * 2)
        self.s.close()


def main():
    ap = argparse.ArgumentParser(description='Firmware 1.2 de la ortesis simulado (UDP).')
    ap.add_argument('--ip', default='127.0.0.1')
    ap.add_argument('--puerto', type=int, default=config.PUERTO_ORTESIS_UDP)
    ap.add_argument('--semilla', type=int, default=0, help='la misma que la de la ortesis simulada y el gemelo (0)')
    ap.add_argument('--sin-mecanica', action='store_true', help='el movimiento empieza al llegar la orden')
    a = ap.parse_args()
    fw = FirmwareSimulado(a.ip, a.puerto, a.semilla, mecanica=not a.sin_mecanica)
    print(f'Firmware simulado 1.2 escuchando en {a.ip}:{fw.puerto}  (Ctrl+C para salir)', flush=True)
    try:
        fw.correr()
    except KeyboardInterrupt:
        print()


if __name__ == '__main__':
    main()
