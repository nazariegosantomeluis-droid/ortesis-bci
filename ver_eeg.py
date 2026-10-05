"""Visor del EEG en vivo SIN orquestador: para probar el casco solo.

tablero.py no sirve para esto: escucha el flujo 'Estado', que solo publica el orquestador. Este
visor lee el EEG (y la IMU) directo de LSL con hardware.EntradaEEG y dibuja los 8 canales
(1-40 Hz, ultimos 5 s). Cada canal lleva lo mismo que mide el CP1: uV rms, fraccion de 60 Hz y
OK / REVISAR. Arriba: alfa occipital (sube con los ojos cerrados) y giroscopio (sube al asentir).

Uso
  python puente_lsl.py --placa unicorn      terminal 1 (o python cerebro_sintetico.py)
  python ver_eeg.py                         terminal 2
  python ver_eeg.py --fuente unicornlsl     con la app UnicornLSL de g.tec
  python ver_eeg.py --escala 200            eje vertical de +-200 uV (por defecto 100)
"""
import argparse
import sys

import numpy as np

import config
import hardware as hw

VISIBLE_S = 5.0          # segundos dibujados
MARGEN_S = 1.0           # se filtra un segundo de mas y se tira: ahi vive el borde del filtro
CALIDAD_S = 5.0          # ventana del rms y del 60 Hz de cada canal
ALFA_S = 4.0             # lo minimo que pide hardware.potencia_alfa


def resumen(entrada):
    """Lo que se dibuja, leido de una EntradaEEG (o de algo con _crudo, calidad, movimiento, ultimo_t,
    edad y fs): {'trazos' (canales x muestras, uV filtrados), 'filas' (calidad por canal), 'malos'
    ({canal: motivo}), 'alfa', 'giro', 'edad'}. Sin datos todavia: trazos None."""
    fs = entrada.fs
    x, _ = entrada._crudo(VISIBLE_S + MARGEN_S)
    r = {'trazos': None, 'filas': [], 'malos': {}, 'alfa': None, 'giro': None, 'edad': entrada.edad()}
    if x.size == 0 or x.shape[1] < int((MARGEN_S + 1.0) * fs):
        return r
    centrado = x - np.median(x, axis=1, keepdims=True)          # el crudo trae ~200 mV de continua
    r['trazos'] = hw.filtrar(centrado, (1.0, 40.0), fs)[:, int(MARGEN_S * fs):]
    r['filas'] = entrada.calidad(CALIDAD_S)
    r['malos'] = hw.revisar_canales(x, fs)
    r['alfa'] = hw.potencia_alfa(entrada._crudo(ALFA_S + 0.1)[0], fs)
    t = entrada.ultimo_t()
    r['giro'] = entrada.movimiento(t - 1.0, t)
    return r


def texto_canal(fila, motivo=None):
    """Titulo de un canal y si esta bien: ('Fz   8 uV rms | 60 Hz 12 % | OK', True)."""
    ok = fila['ok'] and not motivo
    estado = 'OK' if ok else 'REVISAR' + (f' ({motivo})' if motivo else '')
    return f"{fila['canal']}  {fila['rms_uv']:.0f} uV rms | 60 Hz {100 * fila['red']:.0f} % | {estado}", ok


def texto_cabecera(r, alfa_base=None):
    if r['trazos'] is None:
        return 'Esperando muestras de EEG...'
    partes = [f"{sum(texto_canal(f, r['malos'].get(f['canal']))[1] for f in r['filas'])}/{len(r['filas'])} canales OK"]
    if r['alfa'] is not None:
        partes.append(f"alfa occipital {r['alfa']:.1f} uV2/Hz"
                      + (f" (x{r['alfa'] / alfa_base:.1f} del inicio)" if alfa_base else ''))
    partes.append('sin IMU' if r['giro'] is None else f"giroscopio {r['giro']:.0f} grados/s")
    if r['edad'] > config.SALUD['eeg_edad_rojo_s']:
        partes.append(f"SIN MUESTRAS hace {r['edad']:.0f} s")
    return '   |   '.join(partes)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Visor del EEG en vivo, sin orquestador')
    ap.add_argument('--fuente', choices=sorted(config.FUENTES_EEG), default='puente')
    ap.add_argument('--eeg-nombre', dest='eeg_nombre', help='nombre del flujo (por defecto, el de la fuente)')
    ap.add_argument('--escala', type=float, default=100.0, help='eje vertical, en uV')
    ap.add_argument('--captura', help='guarda una imagen del visor y sale')
    ap.add_argument('--segundos', type=float, default=8.0, help='con --captura: cuanto espera antes de guardar')
    a = ap.parse_args(argv)
    try:
        entrada = hw.EntradaEEG(segundos=VISIBLE_S + MARGEN_S + 5.0, nombre=a.eeg_nombre, fuente=a.fuente)
    except RuntimeError as e:
        print(f'{e}\nEste visor NO necesita el orquestador, pero si una fuente de EEG: en otra terminal,\n'
              '  python puente_lsl.py --placa unicorn     (el casco)\n'
              '  python cerebro_sintetico.py              (sin casco: el gemelo)')
        return 1

    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    ventana = QtWidgets.QWidget()
    ventana.setWindowTitle(f'EEG en vivo ({entrada.etiqueta}) - sin orquestador')
    caja = QtWidgets.QVBoxLayout(ventana)
    cabecera = QtWidgets.QLabel('Esperando muestras de EEG...')
    cabecera.setStyleSheet('font-size: 15px; padding: 4px;')
    caja.addWidget(cabecera)
    lienzo = pg.GraphicsLayoutWidget()
    caja.addWidget(lienzo)
    graficas, curvas = [], []
    for i, canal in enumerate(config.CANALES_EEG):
        g = lienzo.addPlot(row=i, col=0)
        g.setYRange(-a.escala, a.escala)
        g.setMouseEnabled(x=False, y=False)
        g.hideAxis('bottom') if i < len(config.CANALES_EEG) - 1 else g.setLabel('bottom', 'segundos')
        g.setTitle(canal)
        if graficas:
            g.setXLink(graficas[0])
        graficas.append(g)
        curvas.append(g.plot(pen='#1f77b4'))
    base = []                                     # la primera alfa medida: referencia para "ojos cerrados"

    def refrescar():
        r = resumen(entrada)
        if r['alfa'] is not None and not base:
            base.append(r['alfa'])
        cabecera.setText(texto_cabecera(r, base[0] if base else None))
        if r['trazos'] is None:
            return
        t = np.arange(r['trazos'].shape[1]) / entrada.fs - r['trazos'].shape[1] / entrada.fs
        for curva, g, y, fila in zip(curvas, graficas, r['trazos'], r['filas']):
            curva.setData(t, y)
            titulo, ok = texto_canal(fila, r['malos'].get(fila['canal']))
            g.setTitle(titulo, color='#2ca02c' if ok else '#d62728')

    reloj = QtCore.QTimer()
    reloj.timeout.connect(refrescar)
    reloj.start(200)
    ventana.resize(1100, 900)
    ventana.show()
    if a.captura:
        QtCore.QTimer.singleShot(int(a.segundos * 1000), lambda: (ventana.grab().save(a.captura), app.quit()))
    app.exec()
    entrada.cerrar()
    return 0


if __name__ == '__main__':
    sys.exit(main())
