"""60 Hz del CP1: cuantas veces fallaria el CP1 por azar, con una ventana y con la mediana de varias (EXPLORATORIO).

Toma un XDF de LabRecorder con el EEG del casco (quieto), calcula la fraccion de potencia en 58-62 Hz sobre 1-100 Hz de
cada canal en ventanas de `ventana_s` que avanzan de a `paso_s`, y simula los dos procedimientos del CP1:

  una ventana   como era: falla si algun canal pasa de config.CP1_RED['umbral'] en la ventana
  robusto       hardware.EntradaEEG.calidad_robusta: los canales que pasan del umbral en la primera ventana se miden en
                config.CP1_RED['ventanas'] ventanas seguidas y deciden por la mediana

Con P001 (616 s, 4 de octubre): una ventana 15.6 % de los inicios, robusto 6.5 % (re-mide en 15.6 %). Es una persona, una
sesion, y los inicios se traslapan: es una estimacion de que tan seguido pasaria, no una tasa.

  python estudios/red_cp1.py ruta/archivo.xdf [--ventana 10] [--paso 2]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
import config
sys.path.insert(0, str(Path(__file__).resolve().parent))   # xdf_minimo


def fraccion_red(x, fs):
    """Por canal, la fraccion de potencia en 58-62 Hz sobre 1-100 Hz (la misma cuenta que hardware.EntradaEEG.calidad)."""
    f = np.fft.rfftfreq(x.shape[1], 1 / fs)
    X = (x - x.mean(1, keepdims=True)) * np.hanning(x.shape[1])
    P = np.abs(np.fft.rfft(X, axis=1)) ** 2
    return P[:, (f > config.RED_HZ - 2) & (f < config.RED_HZ + 2)].sum(1) / (P[:, (f > 1) & (f < 100)].sum(1) + 1e-12)


def series(x, fs, ventana_s=10.0, paso_s=2.0):
    """(ventanas x canales) de fracciones, cada ventana de `ventana_s` s y una cada `paso_s` s."""
    n, paso = int(ventana_s * fs), int(paso_s * fs)
    return np.array([fraccion_red(x[:, i:i + n], fs) for i in range(0, x.shape[1] - n + 1, paso)])


def simular(C, ventana_s=10.0, paso_s=2.0, umbral=None, ventanas=None):
    """Para cada inicio posible: (falla con una ventana, falla con el procedimiento robusto, re-mide). C: salida de series()."""
    umbral = config.CP1_RED['umbral'] if umbral is None else umbral
    ventanas = config.CP1_RED['ventanas'] if ventanas is None else ventanas
    salto = int(round(ventana_s / paso_s))
    filas = []
    for i in range(0, len(C) - salto * (ventanas - 1)):
        primera = C[i]
        sosp = primera >= umbral
        if not sosp.any():
            filas.append((False, False, False))
            continue
        seguidas = C[[i + j * salto for j in range(ventanas)]][:, sosp]
        filas.append((True, bool((np.median(seguidas, axis=0) >= umbral).any()), True))
    return np.array(filas, dtype=bool)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('archivo')
    ap.add_argument('--ventana', type=float, default=10.0)
    ap.add_argument('--paso', type=float, default=2.0)
    a = ap.parse_args(argv)
    import xdf_minimo
    por_nombre = {s['nombre']: s for s in xdf_minimo.leer(a.archivo).values()}
    eeg = por_nombre['EEG']
    x, fs = eeg['x'].astype(float).T, eeg['fs']
    C = series(x, fs, a.ventana, a.paso)
    r = simular(C, a.ventana, a.paso)
    print(f'{Path(a.archivo).name}: {x.shape[1] / fs:.0f} s de EEG, {len(C)} ventanas de {a.ventana:.0f} s, {len(r)} inicios simulados')
    print('  canal   mediana  p10-p90        ventanas >= umbral')
    for k, c in enumerate(config.CANALES_EEG):
        print(f'  {c:>4}    {np.median(C[:, k]):.2f}    {np.percentile(C[:, k], 10):.2f}-{np.percentile(C[:, k], 90):.2f}      {np.mean(C[:, k] >= config.CP1_RED["umbral"]):5.1%}')
    print(f'  falla el CP1 por 60 Hz: una ventana {np.mean(r[:, 0]):.1%} | robusto (mediana de {config.CP1_RED["ventanas"]}) {np.mean(r[:, 1]):.1%}'
          f' | re-mide en {np.mean(r[:, 2]):.1%} de los inicios (+{(config.CP1_RED["ventanas"] - 1) * a.ventana:.0f} s)')
    print('EXPLORATORIO: una persona, una sesion; los inicios se traslapan.')


if __name__ == '__main__':
    main()
