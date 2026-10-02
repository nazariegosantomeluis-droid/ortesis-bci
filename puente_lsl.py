"""Puente BrainFlow -> LSL: publica los flujos 'EEG' e 'IMU' del contrato.

Uso
  python puente_lsl.py --placa unicorn                          casco real (g.tec Unicorn Hybrid Black)
  python puente_lsl.py --placa unicorn --serie UN-2023.01.01    si hay varios cascos cerca
  python puente_lsl.py --placa unicorn --grabar resultados/sesion.csv
  python puente_lsl.py                                          placa sintetica (sin casco)
  python puente_lsl.py --placa playback --archivo resultados/sesion.csv   (plan B)
  python puente_lsl.py --placa cyton --puerto COM3 [--impedancias]        (hardware anterior)

Unicorn: el casco se empareja con SU dongle (no con el Bluetooth de la laptop) y solo una
aplicacion puede conectarse a la vez (cierra la Unicorn Suite). No mide impedancias: el CP1
usa la calidad de senal por canal. Antes de la demo: python verificar_unicorn.py brainflow.

La hora de cada muestra se reconstruye con el contador de la placa (salud.RelojContador) y
cada 30 s se imprime el registro de huecos (perdidas de Bluetooth) y la bateria.
"""
import argparse
import json
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from pylsl import StreamOutlet, local_clock

import config
from salud import RegistroHuecos, RelojContador, Retroceso

SILENCIO_MAX_S = 2.0     # sin datos de la placa durante este tiempo: se reconecta


def reconectar(board, retroceso, grabar=None):
    """Libera la sesion de BrainFlow y la vuelve a preparar, con retroceso exponencial,
    hasta que la placa responda (dongle desconectado, casco apagado). El flujo LSL no
    se toca: el orquestador solo ve un silencio.

    OJO: probado solo con la placa sintetica; NO probado con el Cyton real."""
    for soltar in (board.stop_stream, board.release_session):
        try:
            soltar()
        except Exception:
            pass
    while True:
        espera = retroceso.siguiente()
        print(f'  [puente] placa sin datos: reintento en {espera:.1f} s', flush=True)
        time.sleep(espera)
        try:
            board.prepare_session()
            board.start_stream(45000, f'file://{grabar}:a' if grabar else '')   # :a = no pisa lo grabado
            retroceso.reiniciar()
            print('  [puente] placa recuperada', flush=True)
            return
        except Exception:
            try:
                board.release_session()
            except Exception:
                pass


def plan_placa(placa, serie=None, puerto=None, archivo=None, maestra='unicorn'):
    """Todo lo que el puente necesita saber de una placa, sacado del descriptor de BrainFlow
    (no conecta nada): identificador, parametros y en que fila esta cada cosa."""
    ids = {'sintetica': BoardIds.SYNTHETIC_BOARD.value, 'cyton': BoardIds.CYTON_BOARD.value,
           'unicorn': BoardIds.UNICORN_BOARD.value}
    params, real = BrainFlowInputParams(), placa
    if placa == 'playback':
        if not archivo:
            raise SystemExit('Falta --archivo para playback.')
        board_id, real = BoardIds.PLAYBACK_FILE_BOARD.value, maestra
        params.file, params.master_board = archivo, ids[maestra]
    else:
        board_id = ids[placa]
        if placa == 'cyton':
            if not puerto:
                raise SystemExit('Falta --puerto (Administrador de dispositivos -> Puertos COM).')
            params.serial_port = puerto
        if placa == 'unicorn' and serie:
            params.serial_number = serie          # opcional: solo hace falta con varios cascos cerca
    d = BoardShim.get_board_descr(ids[real])
    acc, gyr, otros = d.get('accel_channels'), d.get('gyro_channels'), d.get('other_channels')
    return {'board_id': board_id, 'params': params, 'placa': real, 'fs': d['sampling_rate'],
            'eeg': d['eeg_channels'][:8], 'imu': acc + gyr if acc and gyr else None,
            'bateria': d.get('battery_channel'), 'contador': d['package_num_channel'],
            'validez': otros[0] if real == 'unicorn' and otros else None,
            # el contador del Unicorn no da la vuelta; el del Cyton y el de la placa sintetica, cada 256
            'modulo': None if real == 'unicorn' else 256}


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
    ap.add_argument('--placa', choices=['sintetica', 'unicorn', 'cyton', 'playback'], default='sintetica')
    ap.add_argument('--serie', help='numero de serie del Unicorn (UN-XXXX.XX.XX); opcional con un solo casco')
    ap.add_argument('--puerto', help='puerto del dongle del Cyton (p. ej. COM3)')
    ap.add_argument('--archivo', help='archivo de BrainFlow para --placa playback')
    ap.add_argument('--maestra', choices=['unicorn', 'cyton'], default='unicorn',
                    help='placa con la que se grabo el archivo de playback')
    ap.add_argument('--grabar', help='ademas guarda los datos crudos en este archivo (para playback)')
    ap.add_argument('--impedancias', action='store_true', help='mide impedancias antes de transmitir (solo Cyton)')
    a = ap.parse_args()

    BoardShim.disable_board_logger()
    plan = plan_placa(a.placa, a.serie, a.puerto, a.archivo, a.maestra)
    eeg, fs = plan['eeg'], plan['fs']
    if fs != config.FLUJOS['EEG'][2]:
        print(f'Aviso: la placa muestrea a {fs} Hz y el contrato dice {config.FLUJOS["EEG"][2]} Hz')

    board = BoardShim(plan['board_id'], plan['params'])
    board.prepare_session()
    if a.placa == 'playback':
        board.config_board('loopback_true')
    board.start_stream(45000, f'file://{a.grabar}:w' if a.grabar else '')
    if a.impedancias and plan['placa'] == 'unicorn':
        print('El Unicorn no mide impedancias: el CP1 usa la calidad de senal por canal.')
    elif a.impedancias and a.placa != 'playback':
        medir_impedancias(board, eeg, fs, simulada=(a.placa == 'sintetica'))
        board.get_board_data()                  # descarta lo medido con la corriente de prueba
    outlet = StreamOutlet(config.crear_info('EEG'), chunk_size=10)
    salida_imu = StreamOutlet(config.crear_info('IMU'), chunk_size=10) if plan['imu'] else None
    print(f"Publicando EEG{' e IMU' if salida_imu else ''} ({plan['placa']}, {len(eeg)} canales, {fs} Hz). "
          f'Ctrl+C para parar.', flush=True)
    # registro de huecos: muestras perdidas (contador de la placa) y silencios de llegada.
    # Sirve para medir el Bluetooth con el casco real.
    huecos = RegistroHuecos(fs, modulo=plan['modulo'])
    # la hora de cada muestra se reconstruye con el contador de la placa (sin el jitter de los
    # bloques de llegada); una perdida queda como un hueco en la hora. El orquestador no usa
    # el suavizado de marcas de LSL: confia en esta hora.
    reloj = RelojContador(fs, modulo=plan['modulo'])

    n_total, t_ini, bateria, invalidas = 0, time.time(), None, 0
    t_dato, retroceso = time.time(), Retroceso()
    try:
        while True:
            try:
                d = board.get_board_data()
            except Exception:                   # la placa dejo de responder
                d = np.empty((0, 0))
            if d.shape[1]:
                horas = list(reloj.estampar(d[plan['contador']], local_clock()))
                outlet.push_chunk(d[eeg, :].T.tolist(), horas)
                if salida_imu:
                    salida_imu.push_chunk(d[plan['imu'], :].T.tolist(), horas)
                if plan['bateria'] is not None:
                    bateria = float(d[plan['bateria'], -1])
                if plan['validez'] is not None:
                    invalidas += int((d[plan['validez']] != 1).sum())
                n_total += d.shape[1]
                t_dato = time.time()
                huecos.bloque(d[plan['contador']], t_dato)
            elif time.time() - t_dato > SILENCIO_MAX_S:
                reconectar(board, retroceso, a.grabar)
                t_dato = time.time()
                huecos.reiniciar_contador()
            if huecos.toca_resumen(time.time()):
                print(huecos.resumen(time.time())
                      + (f' | bateria {bateria:.0f} %' if bateria is not None else '')
                      + (f' | {invalidas} muestras no validas' if invalidas else ''), flush=True)
            if time.time() - t_ini > 5:
                print(f'  {n_total / (time.time() - t_ini):.0f} muestras/s')
                n_total, t_ini = 0, time.time()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print('Deteniendo...')
    finally:
        for soltar in (board.stop_stream, board.release_session):
            try:
                soltar()
            except Exception:
                pass


if __name__ == '__main__':
    main()
