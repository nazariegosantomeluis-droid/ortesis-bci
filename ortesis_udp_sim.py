"""Firmware de la ortesis Adaptrode simulado en la laptop: habla el mismo JSON que la ESP32 (config.py,
secciones Ortesis por Wi-Fi y por USB), para probar hardware.OrtesisUDP, hardware.OrtesisUSB y el
orquestador sin la placa. Por UDP (por omision) o por un cable USB en memoria (usb=True, PuertoSerieSimulado:
solo para las pruebas; un COM de verdad necesitaria un par de puertos virtuales).

Modela lo que importa del firmware: ACK inmediato con su propio reloj en ms (otro origen y 30 ppm de
deriva), telemetria cada 100 ms por Wi-Fi o cada 200 ms por USB (esta solo mientras llegan ordenes por el
cable, con 'ang' y 'ver' una vez por segundo y el saludo de texto al arrancar, como el 1.3), vigilancia (sin
ordenes durante 0.5 s abre la mano despacio), servos con velocidad limitada y, opcionalmente, el paro de
emergencia y el bloqueo. Cada cambio de meta empieza a moverse `latencia_mecanica_simulada(seq, semilla)`
despues del ACK: la misma que usa OrtesisSimulada y con la que el gemelo (cerebro_sintetico.py) ancla el ErrP.

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
USB_TELEMETRIA_S = 0.2          # firmware 1.3: telemetria por USB cada 200 ms...
USB_ORDENES_VIVAS_S = 2.0       # ...mientras la ultima orden por USB tenga menos de 2 s (usb_hasta)


class PuertoSerieSimulado:
    """El cable USB en memoria. Del lado de la PC se ve como pyserial (write, read con timeout, in_waiting,
    close); del lado de la ESP32, `lineas_esp` y `escribir_esp`. Con `caido` = True la PC ve un puerto
    que falla (cable suelto) y lo que se escribe se pierde."""

    def __init__(self, timeout=0.05):
        self.a_esp, self.a_pc, self.timeout, self.caido = b'', b'', timeout, False
        self._cv = threading.Condition()

    def _revisar(self):
        if self.caido:
            raise OSError('puerto caido')

    def write(self, datos):
        with self._cv:
            self._revisar()
            self.a_esp += bytes(datos)
        return len(datos)

    @property
    def in_waiting(self):
        self._revisar()
        return len(self.a_pc)

    def read(self, n=1):
        with self._cv:
            self._cv.wait_for(lambda: self.a_pc or self.caido, timeout=self.timeout)
            self._revisar()
            datos, self.a_pc = self.a_pc[:n], self.a_pc[n:]
            return datos

    def close(self):
        pass

    def lineas_esp(self):
        with self._cv:
            *lineas, self.a_esp = self.a_esp.split(b'\n')
        return [l for l in lineas if l.strip()]

    def escribir_esp(self, texto):
        with self._cv:
            if not self.caido:
                self.a_pc += (texto + '\r\n').encode()     # MicroPython termina las lineas con \r\n
                self._cv.notify_all()


class FirmwareSimulado:
    def __init__(self, ip='127.0.0.1', puerto=config.PUERTO_ORTESIS_UDP, semilla=0, periodo_s=0.02,
                 origen_ms=532_118, deriva_ppm=30.0, reloj=time.monotonic, mecanica=True, usb=False):
        self.s = self.serie = None
        if usb:
            self.serie, self.puerto = PuertoSerieSimulado(), None
            self.serie.escribir_esp('Adaptrode 1.3-sim - DEVKIT_V1 - red Adaptrode - USB si')   # el saludo: texto, no JSON
        else:
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
        self.usb_hasta, self.n_tel_usb = None, 0         # USB: hasta cuando hay telemetria y cuantas van
        self.recibidos, self.acks = [], []               # (seq, hora de llegada) y (seq, t_ms) para las pruebas
        self.p, self.destellos = None, []                # nivel del nervio de luz y hora de llegada de cada mensaje con errp
        self.descartar = None                            # pruebas: funcion(mensaje) -> True si ese datagrama se pierde en el Wi-Fi
        self.angulos_aplicados = []                      # (t_ms en que la meta empezo a regir, meta, seq)
        self._vivo = False

    def t_ms(self):
        """Reloj de la ESP32: milisegundos desde un origen propio, con deriva."""
        return int(self.origen_ms + (self.reloj() - self.t0) * 1000 * (1 + self.deriva))

    def _aplicar_meta(self, meta, seq):
        self.meta = meta
        self.angulos_aplicados.append((self.t_ms(), meta, seq))

    def _llegadas(self):
        """(mensaje en bytes, direccion) de lo que llego en este ciclo; por USB la direccion es None."""
        if self.serie is not None:
            return [(l, None) for l in self.serie.lineas_esp()]
        out = []
        while True:
            try:
                out.append(self.s.recvfrom(512))
            except (BlockingIOError, OSError):
                return out

    def _contestar(self, texto, direccion):
        if self.serie is not None:
            self.serie.escribir_esp(texto)
        else:
            self.s.sendto(texto.encode(), direccion)

    def _recibir(self, ahora):
        for datos, direccion in self._llegadas():
            try:
                m = json.loads(datos)
            except ValueError:
                continue
            if not isinstance(m, dict):
                continue
            if self.descartar is not None and self.descartar(m):
                continue                                 # perdido antes de llegar: la ESP32 no se entera
            self.ultimo_cmd = ahora
            if self.serie is None:
                self.cliente = direccion
            else:
                self.usb_hasta = ahora + USB_ORDENES_VIVAS_S
            seq = m.get('seq')
            if 'p' in m:
                self.p = float(m['p'])
            if m.get('errp'):
                self.destellos.append(ahora)
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
                    self._contestar(json.dumps({'ack': seq, 't_ms': t_ms}), direccion)

    def _telemetria(self, completa=True):
        tel = {'seq': self.ultimo_seq_ack or 0, 'cierre': round(self.frac, 3), 'pulgar': round(self.frac, 3),
               'i_ma': 0, 'fuerza': 0.0, 'paro': self.paro, 'bloqueo': self.bloqueo, 'vigilancia': self.vigilancia}
        if completa:
            tel.update(ang={'dedos': [ABIERTA, CERRADA], 'pulgar': [30.0, 120.0]}, ver='1.3-sim')
        tel.update(msg='', t_ms=self.t_ms())
        return json.dumps(tel)

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
        if self.serie is not None:
            if self.usb_hasta is not None and ahora < self.usb_hasta and ahora - self.ultima_tel >= USB_TELEMETRIA_S:
                self.ultima_tel, self.n_tel_usb = ahora, self.n_tel_usb + 1
                self.serie.escribir_esp(self._telemetria(completa=self.n_tel_usb % 5 == 1))
        elif self.cliente is not None and ahora - self.ultima_tel >= 0.1:
            self.ultima_tel = ahora
            try:
                self.s.sendto(self._telemetria().encode(), self.cliente)
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
        if self.s is not None:
            self.s.close()


def main():
    ap = argparse.ArgumentParser(description='Firmware de la ortesis simulado (UDP).')
    ap.add_argument('--ip', default='127.0.0.1')
    ap.add_argument('--puerto', type=int, default=config.PUERTO_ORTESIS_UDP)
    ap.add_argument('--semilla', type=int, default=0, help='la misma que la de la ortesis simulada y el gemelo (0)')
    ap.add_argument('--sin-mecanica', action='store_true', help='el movimiento empieza al llegar la orden')
    a = ap.parse_args()
    fw = FirmwareSimulado(a.ip, a.puerto, a.semilla, mecanica=not a.sin_mecanica)
    print(f'Firmware simulado escuchando en {a.ip}:{fw.puerto}  (Ctrl+C para salir)', flush=True)
    try:
        fw.correr()
    except KeyboardInterrupt:
        print()


if __name__ == '__main__':
    main()
