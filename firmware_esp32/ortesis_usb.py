"""Cliente por cable USB para la Órtesis Adaptrode (firmware 1.3 o más nuevo). Misma interfaz que OrtesisUDP:
mover(), set_p(), errp(), vibrar(), calibrar(), telemetria(), on_paro(), latencia(), conectada() y cerrar().

Es el camino de menor latencia: la ida y vuelta queda en unos milisegundos y casi no varía, y la laptop
conserva su Wi-Fi normal (con internet). Necesita pyserial:   pip install pyserial

    from ortesis_usb import OrtesisUSB
    print(OrtesisUSB.puertos())        # para encontrar el puerto, por ejemplo COM5 o /dev/ttyUSB0
    o = OrtesisUSB('COM5')
"""
import threading
import time

from ortesis_udp import OrtesisUDP


class OrtesisUSB(OrtesisUDP):
    def __init__(self, puerto, baudios=115200, latido_s=0.1, espera_ack_s=0.15, reloj=time.monotonic):
        import serial
        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate, self.ser.timeout = puerto, baudios, 0.05
        self.ser.dtr = False                      # que abrir el puerto no reinicie la ESP32 (en la mayoría de las placas)
        self.ser.rts = False
        self.ser.open()
        self._buf, self._tx_candado = b'', threading.Lock()
        self._iniciar(latido_s, espera_ack_s, reloj)

    def _tx(self, datos):
        with self._tx_candado:
            self.ser.write(datos + b'\n')

    def _rx(self):
        trozo = self.ser.readline()
        if not trozo:
            return None
        self._buf += trozo
        if not self._buf.endswith(b'\n'):
            if len(self._buf) > 4096:
                self._buf = b''
            return None
        linea, self._buf = self._buf.strip(), b''
        return linea or None

    def _cerrar_transporte(self):
        self.ser.close()

    @staticmethod
    def puertos():
        from serial.tools import list_ports
        return [(p.device, p.description) for p in list_ports.comports()]
