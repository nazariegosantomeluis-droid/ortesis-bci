from pylsl import StreamInlet, resolve_byprop

streams = resolve_byprop('name', 'EEG', timeout=5)
if not streams:
    raise SystemExit("No encontré el stream 'EEG'. ¿Está corriendo puente_lsl.py?")

inlet = StreamInlet(streams[0])
print("Conectado a:", streams[0].name(), "|", streams[0].nominal_srate(), "Hz")
for _ in range(10):
    chunk, ts = inlet.pull_chunk(timeout=1.0)
    print(f"{len(chunk)} muestras recibidas")