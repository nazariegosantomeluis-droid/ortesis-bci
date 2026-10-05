#!/usr/bin/env python3
"""Prueba sin hardware: corre main.py en la laptop con machine, network y neopixel simulados.
Parte A, por Wi-Fi con OrtesisUDP: ACK, latido, paro, ErrP, respaldo sin ACK y vigilancia.
Parte B, por USB con una terminal simulada: prueba local con el botón BOOT, y ACK, telemetría y paro con OrtesisUSB.
Uso (en Linux o Mac):  python firmware_esp32/pruebas/prueba_firmware.py
"""
import os, pty, sys, tempfile, threading, time, tty
from pathlib import Path
AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(AQUI / 'simulados'), str(AQUI.parent)]
import machine
import main as fw
from ortesis_udp import OrtesisUDP
from ortesis_usb import OrtesisUSB

fw.CFG['archivo_cal'] = os.path.join(tempfile.mkdtemp(), 'calibracion.json')
machine.ADC.valores_uv.update({32: 2_500_000, 35: 30_000, 3: 2_500_000, 1: 30_000})   # paro suelto, 0.3 A
aplicado = {}                                   # seq → hora real (ms) en que el firmware aplicó la orden
_procesar = fw.procesar
def procesar_espia(m, ahora):
    if 'seq' in m and m['seq'] not in aplicado:
        aplicado[m['seq']] = fw.ticks_ms()
    return _procesar(m, ahora)
fw.procesar = procesar_espia

class Terminal:                                 # lado ESP32 de una terminal simulada: lo que manda la laptop llega aquí
    def __init__(self, fd): self.fd = fd
    def fileno(self): return self.fd
    def read(self, n=1):
        try:
            return os.read(self.fd, n).decode('utf-8', 'replace')
        except OSError:
            return ''
    def write(self, s): os.write(self.fd, s.encode())
maestro, esclavo = pty.openpty(); tty.setraw(esclavo)
fw.SERIE_E = fw.SERIE_S = Terminal(maestro)

def arrancar():
    fw.DETENER = False; fw.SERVOS.clear()
    h = threading.Thread(target=fw.ejecutar, daemon=True); h.start(); time.sleep(0.3)
    return h

def estampas(cliente, metas):
    errores = []
    for meta in metas:
        e = cliente.mover(cierre=meta, p=meta)
        assert e['metodo'] == 'ack', e
        errores.append(abs(e['t_aplicado'] - aplicado[e['seq']] / 1000) * 1000)
        time.sleep(0.4)
    assert max(errores) < 10, errores
    return max(errores)

# ============================== Parte A: Wi-Fi
hilo = arrancar()
o = OrtesisUDP('127.0.0.1')
cambios_paro = []
o.on_paro(cambios_paro.append)
time.sleep(1.0)
print(f'1. Wi-Fi · ACK: 5 movimientos estampados, error máximo {estampas(o, (1.0, 0.0, 0.6, 0.2, 1.0)):.1f} ms · RTT {o.latencia()}')
vig = []
for _ in range(30):
    vig.append(fw.ESTADO['vigilancia']); time.sleep(0.1)
assert not any(vig) and fw.SERVOS['dedos'].frac > 0.95, (any(vig), fw.SERVOS['dedos'].frac)
print(f'2. Wi-Fi · latido: 3 s entre pasos sin vigilancia; dedos en {fw.SERVOS["dedos"].frac:.2f}')
tel = o.telemetria()
assert tel['t_local'] is not None and abs(tel['t_local'] - (o.reloj() - tel['edad_s'])) < 0.15, tel
machine.ADC.valores_uv[32] = machine.ADC.valores_uv[3] = 0; time.sleep(0.35)
assert cambios_paro[-1] is True and o.telemetria()['paro'] is True and fw.SERVOS['dedos'].frac == 0.0, cambios_paro
machine.ADC.valores_uv[32] = machine.ADC.valores_uv[3] = 2_500_000; time.sleep(0.35)
assert cambios_paro[-1] is False, cambios_paro
print(f'3. Wi-Fi · paro: on_paro avisó {cambios_paro}; con el paro las metas vuelven a abierta')
o.errp(); time.sleep(0.1)
assert fw.ticks_diff(fw.ESTADO['destello_hasta'], fw.ticks_ms()) > 0
print('4. Wi-Fi · errp: el nervio destella en rojo')
sin_ack = OrtesisUDP('127.0.0.1', puerto=8899, espera_ack_s=0.1)
e = sin_ack.mover(cierre=0.5)
assert e['metodo'] == 'respaldo' and e['rtt_ms'] is None, e
sin_ack.cerrar(abrir=False)
print('5. Wi-Fi · respaldo: sin ACK, la estampa usa la hora de envío más la latencia y queda marcada')
o.mover(cierre=1.0); time.sleep(1.8)
o._vivo = False; time.sleep(0.8)
fa = fw.SERVOS['dedos'].frac; time.sleep(0.6); fb = fw.SERVOS['dedos'].frac
assert fw.ESTADO['vigilancia'] and fb < fa, (fa, fb)
print(f'6. Wi-Fi · vigilancia: sin latido se abre despacio ({fa:.2f} → {fb:.2f})')
fw.DETENER = True; hilo.join(2)

# ============================== Parte B: botón BOOT y cable USB
aplicado.clear()
hilo = arrancar()
machine.Pin.entradas[0] = 0; time.sleep(1.2); machine.Pin.entradas[0] = 1          # BOOT apretado 1.2 s
assert fw.ESTADO['prueba_desde'] is not None
time.sleep(2.8)
fr, vig = fw.SERVOS['dedos'].frac, fw.ESTADO['vigilancia']
assert fr > 0.4 and not vig, (fr, vig)
time.sleep(3.3)
assert fw.ESTADO['prueba_desde'] is None and fw.SERVOS['dedos'].meta == 0.0
print(f'7. botón BOOT: la prueba local llevó los servos a {fr:.2f} sin laptop y terminó sola')
u = OrtesisUSB(os.ttyname(esclavo))
cambios_usb = []
u.on_paro(cambios_usb.append)
time.sleep(1.0)
assert u.conectada(), u.tel
print(f'8. USB · ACK: 3 movimientos estampados, error máximo {estampas(u, (1.0, 0.0, 0.6)):.1f} ms · RTT {u.latencia()}')
tel = u.telemetria()
assert 'cierre' in tel and tel['t_local'] is not None, tel
machine.ADC.valores_uv[32] = machine.ADC.valores_uv[3] = 0; time.sleep(0.5)
assert cambios_usb and cambios_usb[-1] is True, cambios_usb
machine.ADC.valores_uv[32] = machine.ADC.valores_uv[3] = 2_500_000; time.sleep(0.5)
assert cambios_usb[-1] is False, cambios_usb
print(f'9. USB · telemetría con hora local y paro avisado {cambios_usb}')
u.mover(cierre=0.3, p=0.3); time.sleep(0.3); u.cerrar(); time.sleep(0.9)
assert fw.ESTADO['vigilancia'], fw.ESTADO['vigilancia']
print('10. USB · al cerrar el cliente, la vigilancia toma el control y abre la mano')
fw.DETENER = True; hilo.join(2)
print('Todo bien.')
