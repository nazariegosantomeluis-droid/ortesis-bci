"""Firmware de la Órtesis Adaptrode · ESP32 DevKit V1 (o ESP32-C3 SuperMini) · MicroPython 1.22 o más nuevo.

TODO EN ESTE ÚNICO ARCHIVO
  - Mueve los dos servos (dedos y pulgar) con velocidad limitada.
  - Pinta el nervio de luz, zumba los vibradores y lee el paro, la corriente y el sensor de fuerza.
  - Recibe órdenes por dos caminos a la vez y contesta por el mismo camino por el que llegaron:
      · Cable USB a la laptop: una línea JSON por orden, a 115200 baudios. Es el camino de menor latencia.
      · Wi-Fi: crea la red «Adaptrode» (clave adaptrode2026) y escucha UDP en el puerto 8888.
  - Prueba rápida sin laptop: deja apretado el botón BOOT de la placa 1 segundo (luz, zumbido y servos al 50 %).

PROTECCIONES
  - Vigilancia: si no llegan órdenes durante medio segundo, abre la mano despacio.
  - Bloqueo: si la corriente pasa de 1.8 A más de 0.3 s, retrocede un poco y limita el cierre 2 s.
  - Paro de emergencia: corta la corriente de los servos. Mientras está oprimido, las metas vuelven a «abierta»,
    así que al soltarlo la mano no salta a cerrarse. La ESP32 sigue prendida para avisar.
  - Si el programa se detiene (un error o Stop en Thonny), apaga la señal de los servos y los vibradores.

ÓRDENES (laptop → ESP32): el mismo JSON por USB (una línea) o por Wi-Fi (un paquete UDP)
  {"seq": 12, "cierre": 0.35, "pulgar": 0.2, "p": 0.71, "errp": 0, "vib": 120}
      cierre, pulgar: 0 = abierta, 1 = cerrada · p: nivel del nervio (0 a 1) · errp: 1 = destello rojo · vib: ms
  {"cmd": "calibrar", "servo": "dedos", "abierta": 20, "cerrada": 150}    → se guarda en calibracion.json
ACK (ESP32 → laptop), en cuanto aplica una orden con seq nuevo:
  {"ack": 12, "t_ms": 532118}      t_ms = reloj de la ESP32 en el momento de aplicarla (para alinear el ErrP)
TELEMETRÍA (ESP32 → laptop): cada 100 ms por Wi-Fi y cada 200 ms por USB
  {"seq": 12, "cierre": 0.33, "pulgar": 0.2, "i_ma": 410, "fuerza": 0.12, "paro": false, "bloqueo": false,
   "vigilancia": false, "ang": {"dedos": [20, 150], "pulgar": [30, 120]}, "msg": "", "ver": "1.3", "t_ms": 532200}
  Por USB, "ang" y "ver" van una vez por segundo para que las líneas sean cortas.
"""
import json
import select
import socket
import sys
import time

import neopixel
import network
from machine import ADC, PWM, Pin

try:
    ticks_ms, ticks_diff, dormir_ms = time.ticks_ms, time.ticks_diff, time.sleep_ms
except AttributeError:          # CPython: permite probar este mismo archivo en la laptop
    ticks_ms = lambda: int(time.monotonic() * 1000)
    ticks_diff = lambda a, b: a - b
    dormir_ms = lambda ms: time.sleep(ms / 1000)

VERSION = '1.3'
PLACA = 'DEVKIT_V1'             # ESP32 DevKit V1 de 30 pines; con la ESP32-C3 SuperMini usen 'C3_SUPERMINI'
PINES = {
    'DEVKIT_V1': {'pin_servo': {'dedos': 18, 'pulgar': 19}, 'pin_led': 23, 'pin_vib': (25, 26),
                  'pin_fuerza': 34, 'pin_corriente': 35, 'pin_paro': 32, 'pin_boton': 0},  # lecturas en ADC1
    'C3_SUPERMINI': {'pin_servo': {'dedos': 5, 'pulgar': 6}, 'pin_led': 7, 'pin_vib': (10, 21),
                     'pin_fuerza': 0, 'pin_corriente': 1, 'pin_paro': 3, 'pin_boton': 9},
}
CFG = {
    'ssid': 'Adaptrode', 'clave': 'adaptrode2026', 'puerto': 8888,
    'canal': 6,                 # canal de Wi-Fi: 1, 6 u 11; cámbienlo si hay muchas redes alrededor
    'usb': True,                # escuchar también órdenes por el cable USB
    'periodo_ms': 10,           # el ciclo corre a 100 Hz: una orden espera 10 ms como máximo
    'n_led': 7,
    'angulos': {'dedos': (20, 150), 'pulgar': (30, 120)},   # (abierta, cerrada) de fábrica; la calibración guardada manda
    'archivo_cal': 'calibracion.json',
    'vel_max': 90.0,            # grados por segundo en operación
    'vel_relajar': 30.0,        # grados por segundo al abrir por vigilancia
    'vigilancia_ms': 500,
    'shunt_ohm': 0.1, 'i_bloqueo_ma': 1800, 'bloqueo_ms': 300, 'retroceso': 0.10, 'bloqueo_dura_ms': 2000,
    'paro_uv': 1_000_000,       # divisor 10k/10k tras el paro: menos de 1 V = paro oprimido
}
CFG.update(PINES[PLACA])
DETENER = False                 # las pruebas lo ponen en True para salir del ciclo
SERIE_E = SERIE_S = None        # en la laptop, las pruebas conectan aquí una terminal simulada
SERVOS, ESTADO = {}, {}


class Servo:
    def __init__(self, pin, abierta, cerrada):
        self.pwm = PWM(Pin(pin), freq=50)
        self.abierta, self.cerrada = abierta, cerrada
        self.frac = self.meta = 0.0                      # arranca en «abierta»
        self.limite, self.limite_hasta = 1.0, 0
        self._escribir()

    def angulo(self):
        return self.abierta + (self.cerrada - self.abierta) * self.frac

    def _escribir(self):
        self.pwm.duty_ns(int(500_000 + self.angulo() / 180.0 * 2_000_000))   # 0.5 ms = 0°, 2.5 ms = 180°

    def paso(self, dt, vel, ahora):
        if self.limite < 1.0 and ticks_diff(ahora, self.limite_hasta) > 0:
            self.limite = 1.0                                    # termina el retroceso por bloqueo
        meta = min(self.meta, self.limite)
        tope = vel * dt / max(1.0, abs(self.cerrada - self.abierta))
        self.frac += max(-tope, min(tope, meta - self.frac))
        self._escribir()

    def apagar(self):
        try:
            self.pwm.deinit()                                    # sin pulsos el servo no hace fuerza
        except Exception:
            pass


class Serie:
    """Líneas JSON por el cable USB: es la misma consola que usa Thonny."""
    def __init__(self, entrada, salida):
        self.e, self.s, self.buf = entrada, salida, ''
        self.poll = select.poll()
        self.poll.register(self.e, select.POLLIN)

    def lineas(self):
        listas = []
        for _ in range(512):                                     # a lo más 512 caracteres por ciclo
            if not self.poll.poll(0):
                break
            c = self.e.read(1)
            if not c:
                break
            if c == '\n':
                if self.buf:
                    listas.append(self.buf)
                self.buf = ''
            elif c != '\r':
                self.buf = self.buf + c if len(self.buf) < 600 else ''
        return listas

    def escribir(self, texto):
        try:
            self.s.write(texto + '\n')
        except Exception:
            pass


def acotar(v):
    return max(0.0, min(1.0, float(v)))


def aviso(texto, ahora):
    ESTADO['msg'], ESTADO['msg_hasta'] = texto, ahora + 1500
    print(texto)


def cargar_calibracion():
    try:
        with open(CFG['archivo_cal']) as f:
            datos = json.load(f)
        for nombre in ('dedos', 'pulgar'):
            if nombre in datos:
                CFG['angulos'][nombre] = (float(datos[nombre][0]), float(datos[nombre][1]))
        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def guardar_calibracion():
    try:
        with open(CFG['archivo_cal'], 'w') as f:
            json.dump({k: [s.abierta, s.cerrada] for k, s in SERVOS.items()}, f)
        return True
    except OSError:
        return False


def calibrar(m, ahora):
    nombre = m.get('servo')
    if nombre not in SERVOS:
        return aviso('cal_error: el servo debe ser dedos o pulgar', ahora)
    s = SERVOS[nombre]
    try:
        a = min(180.0, max(0.0, float(m.get('abierta', s.abierta))))
        c = min(180.0, max(0.0, float(m.get('cerrada', s.cerrada))))
    except (TypeError, ValueError):
        return aviso('cal_error: los angulos deben ser numeros', ahora)
    if abs(c - a) < 20:
        return aviso('cal_error: abierta y cerrada deben separarse al menos 20 grados', ahora)
    s.abierta, s.cerrada = a, c
    aviso('cal_ok ' + nombre + (' guardada' if guardar_calibracion() else ' (no se pudo guardar)'), ahora)


def procesar(m, ahora):
    if m.get('cmd') == 'calibrar':
        return calibrar(m, ahora)
    if 'cierre' in m:
        SERVOS['dedos'].meta = acotar(m['cierre'])
    if 'pulgar' in m:
        SERVOS['pulgar'].meta = acotar(m['pulgar'])
    if 'p' in m:
        ESTADO['p'] = acotar(m['p'])
    if m.get('errp'):
        ESTADO['destello_hasta'] = ahora + 300
    if m.get('vib'):
        ESTADO['vib_hasta'] = ahora + min(int(m['vib']), 500)
    ESTADO['seq'] = m.get('seq', ESTADO['seq'])


def recibir(m, ahora, contestar, clave_ack):
    """Aplica una orden de la laptop y manda su ACK por el mismo camino por el que llegó."""
    ESTADO['prueba_desde'] = None                    # una orden de la laptop cancela la prueba local
    procesar(m, ahora)
    sq = m.get('seq')
    if sq is not None and sq != ESTADO[clave_ack]:
        ESTADO[clave_ack] = sq
        contestar(json.dumps({'ack': sq, 't_ms': ticks_ms()}))


def prueba_local(apretado, ahora, ultimo_cmd):
    """Botón BOOT apretado 1 s: la barra de luz sube y baja, zumban los vibradores y los servos van al 50 % y regresan."""
    if apretado:
        if ESTADO['boton_desde'] is None:
            ESTADO['boton_desde'] = ahora
        elif ESTADO['prueba_desde'] is None and ticks_diff(ahora, ESTADO['boton_desde']) > 1000:
            ESTADO['prueba_desde'] = ahora
            aviso('prueba local: luz, vibradores y servos al 50 %', ahora)
    else:
        ESTADO['boton_desde'] = None
    if ESTADO['prueba_desde'] is None:
        return ultimo_cmd
    t = ticks_diff(ahora, ESTADO['prueba_desde'])
    if t > 6000:
        ESTADO['prueba_desde'] = None
        for s in SERVOS.values():
            s.meta = 0.0
        return ultimo_cmd
    for s in SERVOS.values():
        s.meta = 0.5 if 1500 <= t < 3500 else 0.0
    ESTADO['p'] = t / 1500 if t < 1500 else (1.0 if t < 3500 else max(0.0, 1.0 - (t - 3500) / 2500))
    if t < 200 or 1500 <= t < 1700:
        ESTADO['vib_hasta'] = ahora + 150
    return ahora                                     # mientras dura la prueba, la vigilancia no la interrumpe


def pintar_nervio(led, n, ahora):
    """Barra que sube con p (azul = abrir, verde agua = cerrar); destello rojo con un ErrP;
    ámbar en vigilancia; rojo fijo con el paro oprimido."""
    if ESTADO['paro']:
        colores = [(90, 0, 0)] * n
    elif ticks_diff(ESTADO['destello_hasta'], ahora) > 0:
        colores = [(200, 10, 10)] * n
    elif ESTADO['vigilancia']:
        colores = [(80, 45, 0)] * n
    else:
        p = ESTADO['p']
        color = (0, int(60 + 180 * p), int(220 - 120 * p))
        nivel = p * n
        colores = [tuple(int(c * (1.0 if i < nivel else 0.12)) for c in color) for i in range(n)]
    for i in range(n):
        led[i] = colores[i]
    led.write()


def telemetria(ahora, completa=True):
    tel = {'seq': ESTADO['seq'], 'cierre': round(SERVOS['dedos'].frac, 3), 'pulgar': round(SERVOS['pulgar'].frac, 3)}
    for k in ('i_ma', 'fuerza', 'paro', 'bloqueo', 'vigilancia'):
        tel[k] = ESTADO[k]
    if completa:
        tel['ang'] = {k: [s.abierta, s.cerrada] for k, s in SERVOS.items()}
        tel['ver'] = VERSION
    tel['msg'] = ESTADO['msg'] if ticks_diff(ESTADO['msg_hasta'], ahora) > 0 else ''
    tel['t_ms'] = ticks_ms()
    return json.dumps(tel)


def ejecutar(ciclos=None, periodo_ms=None):
    periodo_ms = periodo_ms or CFG['periodo_ms']
    cargar_calibracion()
    ap = network.WLAN(network.AP_IF)
    ap.active(True)
    try:
        ap.config(essid=CFG['ssid'], password=CFG['clave'], authmode=3, channel=CFG['canal'])
    except Exception:
        ap.config(essid=CFG['ssid'], password=CFG['clave'], authmode=3)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', CFG['puerto']))
    sock.setblocking(False)
    serie = None
    if CFG['usb']:
        try:
            serie = Serie(SERIE_E or sys.stdin, SERIE_S or sys.stdout)
        except Exception:
            serie = None
    try:
        boton = Pin(CFG['pin_boton'], Pin.IN, Pin.PULL_UP)
    except Exception:
        boton = None
    for nombre, pin in CFG['pin_servo'].items():
        SERVOS[nombre] = Servo(pin, *CFG['angulos'][nombre])
    led = neopixel.NeoPixel(Pin(CFG['pin_led']), CFG['n_led'])
    vibs = [Pin(p, Pin.OUT, value=0) for p in CFG['pin_vib']]
    adc_f, adc_i, adc_p = ADC(Pin(CFG['pin_fuerza'])), ADC(Pin(CFG['pin_corriente'])), ADC(Pin(CFG['pin_paro']))
    adc_f.atten(ADC.ATTN_11DB)
    adc_i.atten(ADC.ATTN_0DB)
    adc_p.atten(ADC.ATTN_11DB)
    ahora = ticks_ms()
    ESTADO.update(seq=0, p=0.5, destello_hasta=ahora, vib_hasta=ahora, vigilancia=True, paro=False, bloqueo=False,
                  i_ma=0, fuerza=0.0, msg='', msg_hasta=ahora, seq_ack_udp=None, seq_ack_usb=None,
                  usb_hasta=ahora - 1, prueba_desde=None, boton_desde=None)
    try:
        ip = ap.ifconfig()[0]
    except Exception:
        ip = '192.168.4.1'
    print('Adaptrode', VERSION, '·', PLACA, '· red', CFG['ssid'], '· IP', ip, '· canal', CFG['canal'],
          '· USB', 'sí' if serie else 'no')
    ultimo_cmd, ultimo_tel, ultimo_tel_usb, n_usb = ahora - 10_000, ahora, ahora, 0
    t_sobre, cliente, n = None, None, 0
    try:
        while not DETENER and (ciclos is None or n < ciclos):
            n += 1
            ahora = ticks_ms()
            try:
                try:                                         # 1a) órdenes por Wi-Fi
                    while True:
                        datos, direccion = sock.recvfrom(512)
                        m = json.loads(datos)
                        if isinstance(m, dict):
                            cliente, ultimo_cmd = direccion, ahora
                            recibir(m, ahora, lambda t: sock.sendto(t.encode(), direccion), 'seq_ack_udp')
                except (OSError, ValueError):
                    pass
                if serie is not None:                        # 1b) órdenes por el cable USB
                    for linea in serie.lineas():
                        try:
                            m = json.loads(linea)
                        except ValueError:
                            continue
                        if isinstance(m, dict):
                            ultimo_cmd, ESTADO['usb_hasta'] = ahora, ahora + 2000
                            recibir(m, ahora, serie.escribir, 'seq_ack_usb')
                if boton is not None:                        # 2) prueba local con el botón BOOT
                    ultimo_cmd = prueba_local(boton.value() == 0, ahora, ultimo_cmd)
                ESTADO['vigilancia'] = ticks_diff(ahora, ultimo_cmd) > CFG['vigilancia_ms']
                if ESTADO['vigilancia']:                     # 3) sin órdenes: abrir despacio
                    for s in SERVOS.values():
                        s.meta = 0.0
                ESTADO['i_ma'] = int(adc_i.read_uv() / CFG['shunt_ohm'] / 1000)      # 4) sensores
                ESTADO['fuerza'] = round(min(1.0, adc_f.read_uv() / 3_300_000), 3)
                ESTADO['paro'] = adc_p.read_uv() < CFG['paro_uv']
                if ESTADO['paro']:                           # 5) paro: al soltarlo, que no salte a cerrar
                    for s in SERVOS.values():
                        s.meta = s.frac = 0.0
                if ESTADO['i_ma'] > CFG['i_bloqueo_ma']:    # 6) bloqueo: retrocede y limita un rato
                    if t_sobre is None:
                        t_sobre = ahora
                    elif ticks_diff(ahora, t_sobre) > CFG['bloqueo_ms']:
                        for s in SERVOS.values():
                            s.limite = max(0.0, s.frac - CFG['retroceso'])
                            s.limite_hasta = ahora + CFG['bloqueo_dura_ms']
                        ESTADO['bloqueo'], t_sobre = True, None
                else:
                    t_sobre = None
                    if all(s.limite >= 1.0 for s in SERVOS.values()):
                        ESTADO['bloqueo'] = False
                vel = CFG['vel_relajar'] if ESTADO['vigilancia'] else CFG['vel_max']
                for s in SERVOS.values():                    # 7) mover, vibrar y pintar
                    s.paso(periodo_ms / 1000, vel, ahora)
                zumba = ticks_diff(ESTADO['vib_hasta'], ahora) > 0
                for v in vibs:
                    v.value(1 if zumba else 0)
                pintar_nervio(led, CFG['n_led'], ahora)
                if cliente is not None and ticks_diff(ahora, ultimo_tel) >= 100:      # 8) telemetría
                    ultimo_tel = ahora
                    try:
                        sock.sendto(telemetria(ahora).encode(), cliente)
                    except OSError:
                        pass
                if serie is not None and ticks_diff(ESTADO['usb_hasta'], ahora) > 0 and ticks_diff(ahora, ultimo_tel_usb) >= 200:
                    ultimo_tel_usb, n_usb = ahora, n_usb + 1
                    serie.escribir(telemetria(ahora, completa=(n_usb % 5 == 1)))
            except Exception as e:                           # un error raro no debe dejar la mano trabada
                print('error en el ciclo:', repr(e))
            dormir_ms(max(0, periodo_ms - ticks_diff(ticks_ms(), ahora)))
    finally:
        for s in SERVOS.values():
            s.apagar()
        for v in vibs:
            v.value(0)
        try:
            for i in range(CFG['n_led']):
                led[i] = (0, 0, 0)
            led.write()
        except Exception:
            pass
        sock.close()


if __name__ == '__main__':
    ejecutar()
