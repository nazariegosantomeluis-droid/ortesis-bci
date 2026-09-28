import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from pylsl import StreamInfo, StreamOutlet

BOARD_ID = BoardIds.SYNTHETIC_BOARD.value   # luego: BoardIds.CYTON_BOARD.value
params = BrainFlowInputParams()
# params.serial_port = 'COM3'               # solo con la Cyton real

board = BoardShim(BOARD_ID, params)
eeg_ch = BoardShim.get_eeg_channels(BOARD_ID)[:8]
fs = BoardShim.get_sampling_rate(BOARD_ID)

from config import crear_info
outlet = StreamOutlet(crear_info('EEG'))

board.prepare_session()
board.start_stream()
print(f"Publicando EEG ({len(eeg_ch)} canales, {fs} Hz). Ctrl+C para parar.")
try:
    while True:
        data = board.get_board_data()          # (canales, muestras)
        if data.shape[1] > 0:
            outlet.push_chunk(data[eeg_ch, :].T.tolist())
        time.sleep(0.02)
except KeyboardInterrupt:
    print("Deteniendo...")
finally:
    board.stop_stream()
    board.release_session()