"""Mide la latencia mecanica real de la ortesis (del ACK al inicio del movimiento) con la
telemetria T del ESP32, para alinear la epoca del ErrP el domingo.

  python verificar_ortesis.py --puerto COM4            con el ESP32
  python verificar_ortesis.py --simulada               con la ortesis simulada (para probar)

Mueve la ortesis entre 0.3 y 0.7 varias veces y, para cada movimiento, busca en la telemetria
el primer cambio del angulo tras el ACK. Reporta cuantos inicios se detectaron y la
distribucion de la latencia mecanica. El informe queda en resultados/verificacion_ortesis.json.
"""
import argparse
import json
import time

import numpy as np

import config
import hardware as hw


def medir(ortesis, movimientos=30, pausa_s=0.8, salida=print):
    lat, como = [], []
    for k in range(movimientos):
        seq, t_ack, _ = ortesis.mover(0.3 if k % 2 else 0.7)
        if t_ack is None:
            como.append('sin_ack')
            continue
        t0, forma = ortesis.inicio_movimiento(seq, t_ack, espera_s=0.6 if pausa_s else 0.0)
        como.append(forma)
        if forma == 'telemetria':
            lat.append(1000 * (t0 - t_ack))
        time.sleep(pausa_s)
    lat = np.array(lat)
    r = {'movimientos': movimientos, 'detectados': int(sum(c == 'telemetria' for c in como)),
         'sin_ack': int(sum(c == 'sin_ack' for c in como))}
    if lat.size:
        r.update(mediana_ms=float(np.median(lat)), p5_ms=float(np.quantile(lat, 0.05)),
                 p95_ms=float(np.quantile(lat, 0.95)), max_ms=float(lat.max()))
    if r['detectados'] == 0:
        r['veredicto'] = ('FALLA: no se vio el inicio del movimiento. Revisa que el firmware mande la '
                          'telemetria T,<t_us>,<angulo>,<fsr> y el t_us en el ACK (A,<seq>,<t_us>).')
    elif r['detectados'] < 0.8 * movimientos:
        r['veredicto'] = 'AVISO: el inicio se vio en menos del 80 % de los movimientos; se usara ACK + latencia media.'
    else:
        r['veredicto'] = (f"OK: latencia mecanica mediana {r['mediana_ms']:.0f} ms (p5 {r['p5_ms']:.0f}, "
                          f"p95 {r['p95_ms']:.0f}); la epoca del ErrP se alinea al inicio real.")
    salida(f"  inicios detectados: {r['detectados']} de {movimientos}" +
           (f"; latencia mecanica mediana {r['mediana_ms']:.0f} ms, p5 {r['p5_ms']:.0f}, p95 {r['p95_ms']:.0f}, "
            f"maxima {r['max_ms']:.0f}" if lat.size else ''))
    salida('  ' + r['veredicto'])
    return r


def main():
    ap = argparse.ArgumentParser(description='Latencia mecanica de la ortesis con la telemetria del ESP32')
    ap.add_argument('--puerto', default=config.PUERTO_ORTESIS)
    ap.add_argument('--simulada', action='store_true')
    ap.add_argument('--movimientos', type=int, default=30)
    a = ap.parse_args()
    ortesis = hw.OrtesisSimulada() if a.simulada else hw.OrtesisSerial(a.puerto)
    try:
        r = medir(ortesis, a.movimientos)
    finally:
        ortesis.cerrar()
    config.RESULTADOS.mkdir(exist_ok=True)
    ruta = config.RESULTADOS / 'verificacion_ortesis.json'
    ruta.write_text(json.dumps(dict(r, t=time.time(), simulada=a.simulada), indent=1), encoding='utf-8')
    print(f'Informe: {ruta}')


if __name__ == '__main__':
    main()
