"""Puente BrainFlow -> LSL: publica el flujo 'EEG' del contrato.

Uso
  python puente_lsl.py                                  placa sintetica (sin casco)
  python puente_lsl.py --placa cyton --puerto COM3      casco real
  python puente_lsl.py --placa cyton --puerto COM3 --grabar resultados/sesion.csv
  python puente_lsl.py --placa playback --archivo resultados/sesion.csv   (plan B)
  python puente_lsl.py --placa cyton --puerto COM3 --impedancias   mide antes de transmitir

Impedancias: el Cyton inyecta 6 nA a 31.25 Hz en cada electrodo (comando
z<canal>01Z); la amplitud de esa senal da la impedancia de contacto. Se
guardan en resultados/impedancias.json y el orquestador las usa en el CP1.
"""
import argparse
import json
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from pylsl import StreamOutlet, local_clock

import config


def medir_impedancias(board, eeg, fs, simulada=False):
    """kOhm por canal. Formula del lead-off del ADS1299: Z = sqrt(2)*Vrms/6nA - 2.2 kOhm (serie)."""
    res = {}
    if simulada:
        rng = np.random.default_rng()
        res = {c: float(rng.uniform(4, 14)) for c in config.CANALES_EEG[:len(eeg)]}
    else:
        sos = butter(4, (27, 35), btype='bandpass', fs=fs, output='sos')
        for i, canal in enumerate(config.CANALES_EEG[:len(eeg)], start=1):
            board.config_board(f'z{i}01Z')
            time.sleep(2.0)
            d = board.get_current_board_data(int(fs))[eeg[i - 1]]
            board.config_board(f'z{i}00Z')
            v = sosfiltfilt(sos, d - d.mean())[int(0.2 * fs):]
            res[canal] = float(max(0.0, (np.sqrt(2) * v.std() * 1e-6 / 6e-9 - 2200) / 1000))
            time.sleep(0.3)
    print('Impedancias' + (' (SIMULADAS: placa sintetica)' if simulada else '') + ':')
    for c, z in res.items():
        print(f'  {c:>4}: {z:6.1f} kOhm  {"bien" if z <= config.IMPEDANCIA_MAX_KOHM else "ALTA: mas gel / presionar"}')
    config.RESULTADOS.mkdir(exist_ok=True)
    config.IMPEDANCIAS_JSON.write_text(json.dumps(
        {'t': time.time(), 'simulada': simulada, 'kohm': res}, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--placa', choices=['sintetica', 'cyton', 'playback'], default='sintetica')
    ap.add_argument('--puerto', help='puerto del dongle del Cyton (p. ej. COM3)')
    ap.add_argument('--archivo', help='archivo de BrainFlow para --placa playback')
    ap.add_argument('--grabar', help='ademas guarda los datos crudos en este archivo (para playback)')
    ap.add_argument('--impedancias', action='store_true', help='mide impedancias antes de transmitir')
    a = ap.parse_args()

    BoardShim.disable_board_logger()
    params = BrainFlowInputParams()
    if a.placa == 'sintetica':
        board_id = BoardIds.SYNTHETIC_BOARD.value
    elif a.placa == 'cyton':
        if not a.puerto:
            raise SystemExit('Falta --puerto (Administrador de dispositivos -> Puertos COM).')
        board_id = BoardIds.CYTON_BOARD.value
        params.serial_port = a.puerto
    else:
        if not a.archivo:
            raise SystemExit('Falta --archivo para playback.')
        board_id = BoardIds.PLAYBACK_FILE_BOARD.value
        params.file = a.archivo
        params.master_board = BoardIds.CYTON_BOARD.value

    datos_id = params.master_board if a.placa == 'playback' else board_id
    eeg = BoardShim.get_eeg_channels(datos_id)[:8]
    fs = BoardShim.get_sampling_rate(datos_id)
    if fs != config.FLUJOS['EEG'][2]:
        print(f'Aviso: la placa muestrea a {fs} Hz y el contrato dice {config.FLUJOS["EEG"][2]} Hz')

    board = BoardShim(board_id, params)
    board.prepare_session()
    if a.placa == 'playback':
        board.config_board('loopback_true')
    board.start_stream(45000, f'file://{a.grabar}:w' if a.grabar else '')
    if a.impedancias and a.placa != 'playback':
        medir_impedancias(board, eeg, fs, simulada=(a.placa == 'sintetica'))
        board.get_board_data()                  # descarta lo medido con la corriente de prueba
    outlet = StreamOutlet(config.crear_info('EEG'), chunk_size=10)
    print(f'Publicando EEG ({a.placa}, {len(eeg)} canales, {fs} Hz). Ctrl+C para parar.')

    n_total, t_ini = 0, time.time()
    try:
        while True:
            d = board.get_board_data()
            if d.shape[1]:
                # la ultima muestra del bloque se estampa "ahora"; LSL reparte el resto
                outlet.push_chunk(d[eeg, :].T.tolist(), local_clock())
                n_total += d.shape[1]
            if time.time() - t_ini > 5:
                print(f'  {n_total / (time.time() - t_ini):.0f} muestras/s')
                n_total, t_ini = 0, time.time()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print('Deteniendo...')
    finally:
        board.stop_stream()
        board.release_session()


if __name__ == '__main__':
    main()
