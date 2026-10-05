"""Sensor de fuerza resistivo FSR402 (circular) en la ortesis: de la lectura del firmware a newtons.

El firmware 1.3 lee GPIO34 (divisor de voltaje con 10 kOhm a GND) y manda `fuerza` = V/3.3 en la telemetria. Aqui se
convierte a resistencia y a fuerza con la curva de `config.FSR_*`. Funciones puras, sin hardware ni LSL.

  python fuerza.py --puerto 192.168.4.1     # imprime la fuerza en vivo por Wi-Fi (Ctrl+C para salir)

ORIENTATIVO: sin calibrar con pesas la cifra en newtons tiene un error de hasta ~30 % (ver config.FSR_*).
"""
import argparse
import math
import time

import config


def resistencia_ohm(fraccion, r_divisor=config.FSR_R_DIVISOR_OHM):
    """Resistencia del FSR para una lectura V/3.3 (FSR arriba, resistencia fija a GND). None si no hay contacto."""
    if fraccion is None or fraccion < config.FSR_UMBRAL_TOQUE:
        return None
    return r_divisor * (1.0 / min(float(fraccion), 0.999) - 1.0)


def _curva(curva=None):
    """(a, b) de F = a * (1/R_kohm)^b por los dos puntos de la hoja de datos, o de una calibracion propia."""
    (f1, r1), (f2, r2) = curva or (config.FSR_CURVA_N_1KOHM, config.FSR_CURVA_N_2)
    b = math.log(f2 / f1) / math.log(r1 / r2)
    return f1 / (1.0 / r1) ** b, b


def fuerza_n(fraccion, curva=None):
    """Fuerza en newtons (0.0 sin contacto)."""
    r = resistencia_ohm(fraccion)
    if r is None:
        return 0.0
    a, b = _curva(curva)
    return a * (1000.0 / r) ** b


def calibrar(pares):
    """Curva (punto debil, punto fuerte) a partir de [(newtons, fraccion), ...] medidos con pesas; sirve como `curva`."""
    pts = sorted((n, resistencia_ohm(f) / 1000.0) for n, f in pares if resistencia_ohm(f))
    if len(pts) < 2 or pts[0][0] == pts[-1][0]:
        raise ValueError('hacen falta al menos 2 pesas distintas con contacto')
    return pts[0], pts[-1]


def texto(fraccion, curva=None):
    """Linea imprimible: lectura cruda, resistencia y fuerza."""
    r = resistencia_ohm(fraccion)
    if r is None:
        return f'fuerza {fraccion or 0.0:.3f} (sin contacto) · 0.00 N'
    n = fuerza_n(fraccion, curva)
    aviso = ' (saturado: no confiable)' if n > config.FSR_MAX_N else ''
    return f'fuerza {fraccion:.3f} · FSR {r / 1000:.1f} kOhm · {n:.2f} N ({n / 9.80665 * 1000:.0f} gf){aviso}'


def main():
    import hardware as hw
    ap = argparse.ArgumentParser(description='Fuerza del FSR402 en vivo (telemetria de la ESP32 por Wi-Fi)')
    ap.add_argument('--puerto', default=config.IP_ORTESIS_UDP, help='IP de la ESP32')
    ap.add_argument('--udp-puerto', type=int, default=config.PUERTO_ORTESIS_UDP)
    a = ap.parse_args()
    o = hw.OrtesisUDP(a.puerto, a.udp_puerto)
    try:
        while True:
            time.sleep(0.2)
            t = o.ultima_tel if o.conectada() else None
            print(texto(t.get('fuerza', 0.0)) if t else 'sin telemetria: ¿la laptop esta en la red Adaptrode?', flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        o.cerrar()


if __name__ == '__main__':
    main()
