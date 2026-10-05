#!/usr/bin/env python3
"""Prueba la órtesis desde la laptop, sin casco y sin instalar nada (solo Python 3).

Por Wi-Fi:  conecta la laptop a la red «Adaptrode» (clave adaptrode2026) y corre   python probar_esp32.py
Por cable USB (menos latencia; necesita  pip install pyserial):
     python probar_esp32.py --puertos         muestra los puertos para saber cuál es la ESP32
     python probar_esp32.py --usb COM5        (en Mac o Linux, algo como /dev/ttyUSB0)
Luego escribe una orden y presiona Enter:
     c 0.5              cierra los dedos a la mitad (0 = abierta, 1 = cerrada)
     p 0.3              mueve el pulgar al 30 %
     a                  abre todo                 x        cierra todo
     v                  vibra 200 ms              e        destello rojo (como un ErrP)
     t                  muestra la telemetría     ciclo    abre y cierra 3 veces, despacio
     lat                latencia del Wi-Fi (tiempo de ida y vuelta de los ACK)
     cal dedos 20 140   cambia los ángulos abierta y cerrada de un servo y los guarda en la ESP32
     espera 2           espera 2 segundos (útil con --ordenes)
     q                  salir: la órtesis se abre sola en medio segundo
También puedes mandar varias órdenes de una vez:   python probar_esp32.py --ordenes "c 1; espera 2; t; a"
"""
import argparse
import json
import socket
import threading
import time
from collections import deque


class Ortesis:
    def __init__(self, ip='192.168.4.1', puerto=8888, usb=None):
        self.ser, self.s = None, None
        if usb:                                  # cable USB: líneas JSON a 115200 baudios
            import serial
            self.ser = serial.Serial()
            self.ser.port, self.ser.baudrate, self.ser.timeout = usb, 115200, 0.2
            self.ser.dtr = self.ser.rts = False  # que abrir el puerto no reinicie la ESP32
            self.ser.open()
            self._buf = b''
        else:                                    # Wi-Fi: paquetes UDP
            self.dest = (ip, puerto)
            self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.s.settimeout(0.2)
        self.estado = {'cierre': 0.0, 'pulgar': 0.0, 'p': 0.0}
        self.tel, self.t_tel, self.seq, self.vivo = None, 0.0, 0, True
        self.enviados, self.rtts = {}, deque(maxlen=300)
        self.candado = threading.Lock()
        threading.Thread(target=self._latido, daemon=True).start()
        threading.Thread(target=self._escuchar, daemon=True).start()

    def _tx(self, datos):
        if self.ser:
            self.ser.write(datos + b'\n')
        else:
            self.s.sendto(datos, self.dest)

    def _rx(self):
        if not self.ser:
            return self.s.recvfrom(2048)[0]
        self._buf += self.ser.readline()
        if not self._buf.endswith(b'\n'):
            if len(self._buf) > 4096:
                self._buf = b''
            raise socket.timeout
        linea, self._buf = self._buf.strip(), b''
        return linea

    def _mandar(self, mensaje):
        with self.candado:
            self.seq += 1
            mensaje = dict(mensaje, seq=self.seq)
            self.enviados[self.seq] = time.time()
            if len(self.enviados) > 600:
                for k in sorted(self.enviados)[:200]:
                    del self.enviados[k]
            try:
                self._tx(json.dumps(mensaje).encode())
            except OSError:
                pass

    def _latido(self):                     # cada 100 ms repite la orden actual para que no se active la vigilancia
        while self.vivo:
            self._mandar(self.estado)
            time.sleep(0.1)

    def _escuchar(self):
        while self.vivo:
            try:
                datos = self._rx()
                m, t = json.loads(datos), time.time()
                if not isinstance(m, dict):
                    continue
                if 'ack' in m:                     # firmware 1.2: ACK de cada orden, sirve para medir la latencia
                    te = self.enviados.get(m['ack'])
                    if te is not None:
                        self.rtts.append((t - te) * 1000)
                else:
                    self.tel, self.t_tel = m, t
            except (socket.timeout, OSError, ValueError):
                pass

    def mover(self, cierre=None, pulgar=None):
        if cierre is not None:
            self.estado['cierre'] = max(0.0, min(1.0, cierre))
        if pulgar is not None:
            self.estado['pulgar'] = max(0.0, min(1.0, pulgar))
        self.estado['p'] = self.estado['cierre']
        self._mandar(dict(self.estado, vib=120))

    def vibrar(self, ms=200):
        self._mandar(dict(self.estado, vib=ms))

    def destello(self):
        self._mandar(dict(self.estado, errp=1))

    def calibrar(self, servo, abierta, cerrada):
        self._mandar({'cmd': 'calibrar', 'servo': servo, 'abierta': abierta, 'cerrada': cerrada})

    def conectada(self):
        return self.tel is not None and time.time() - self.t_tel < 1.0

    def cerrar(self):
        self.mover(0.0, 0.0)
        time.sleep(0.3)
        self.vivo = False


def si_no(v):
    return 'SÍ' if v else 'no'


def texto_telemetria(o):
    if not o.conectada():
        return ('No llega telemetría. Por Wi-Fi: ¿la laptop está en la red «Adaptrode»? Por USB: ¿es el puerto correcto '
                'y está cerrado Thonny? ¿La ESP32 está encendida y con el firmware 1.3?')
    t = o.tel
    ang = t.get('ang', {})
    linea = (f"dedos {t['cierre']:.2f} · pulgar {t['pulgar']:.2f} · {t['i_ma']} mA · fuerza {t['fuerza']:.2f} · "
             f"paro {si_no(t['paro'])} · bloqueo {si_no(t['bloqueo'])} · vigilancia {si_no(t['vigilancia'])}")
    if ang:
        linea += ' · ángulos ' + ', '.join(f'{k} {a:g}–{c:g}' for k, (a, c) in ang.items())
    if t.get('msg'):
        linea += f" · {t['msg']}"
    return linea


def ejecutar_orden(o, linea):
    partes = linea.strip().split()
    if not partes:
        return True
    cmd, args = partes[0].lower(), partes[1:]
    try:
        if cmd == 'q':
            return False
        elif cmd == 'c':
            o.mover(cierre=float(args[0]))
        elif cmd == 'p':
            o.mover(pulgar=float(args[0]))
        elif cmd == 'a':
            o.mover(0.0, 0.0)
        elif cmd == 'x':
            o.mover(1.0, 1.0)
        elif cmd == 'v':
            o.vibrar()
        elif cmd == 'e':
            o.destello()
        elif cmd == 't':
            print(texto_telemetria(o))
        elif cmd == 'lat':
            r = sorted(o.rtts)
            if not r:
                print('Todavía no llegan ACK (el firmware debe ser 1.2 o más nuevo).')
            else:
                print(f'Ida y vuelta por {"USB" if o.ser else "Wi-Fi"}: mínimo {r[0]:.1f} ms · mediana {r[len(r) // 2]:.1f} ms · '
                      f'95 % por debajo de {r[int(0.95 * (len(r) - 1))]:.1f} ms ({len(r)} muestras)')
        elif cmd == 'espera':
            time.sleep(float(args[0]))
        elif cmd == 'ciclo':
            for i in range(3):
                print(f'  ciclo {i + 1}: cerrar'); o.mover(1.0, 1.0); time.sleep(2.5)
                print(f'  ciclo {i + 1}: abrir'); o.mover(0.0, 0.0); time.sleep(2.5)
        elif cmd == 'cal':
            servo, abierta, cerrada = args[0].lower(), float(args[1]), float(args[2])
            o.calibrar(servo, abierta, cerrada)
            time.sleep(0.4)
            print(texto_telemetria(o))
        else:
            print('No conozco esa orden. Escribe una de: c p a x v e t lat ciclo cal espera q')
    except (IndexError, ValueError):
        print('Faltan números o no se entienden. Ejemplos: «c 0.5», «cal dedos 20 140».')
    return True


def main():
    ap = argparse.ArgumentParser(description='Prueba la órtesis Adaptrode por Wi-Fi.')
    ap.add_argument('--ip', default='192.168.4.1', help='IP de la ESP32 (192.168.4.1 si estás en su red)')
    ap.add_argument('--puerto', type=int, default=8888)
    ap.add_argument('--usb', metavar='PUERTO', help='usar el cable USB en vez del Wi-Fi (por ejemplo COM5)')
    ap.add_argument('--puertos', action='store_true', help='mostrar los puertos USB disponibles y salir')
    ap.add_argument('--ordenes', help='órdenes separadas por «;» para correr sin teclear')
    a = ap.parse_args()
    if a.puertos:
        from serial.tools import list_ports
        for p in list_ports.comports():
            print(f'{p.device:14s} {p.description}')
        print('La ESP32 suele aparecer como «CP210x» o «CH340».')
        return
    o = Ortesis(a.ip, a.puerto, usb=a.usb)
    for _ in range(20):
        if o.conectada():
            break
        time.sleep(0.1)
    print(('Conectada: ' if o.conectada() else 'Aviso: ') + texto_telemetria(o))
    try:
        if a.ordenes:
            for linea in a.ordenes.split(';'):
                if not ejecutar_orden(o, linea):
                    break
                time.sleep(0.3)
        else:
            print('Órdenes: c 0.5 · p 0.3 · a · x · v · e · t · lat · ciclo · cal dedos 20 140 · q para salir')
            while ejecutar_orden(o, input('> ')):
                pass
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        o.cerrar()
        print('Listo: dejé de mandar órdenes; la órtesis se abre sola.')


if __name__ == '__main__':
    main()
