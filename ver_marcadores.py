"""Escucha los flujos Marcadores y Paso del orquestador y los imprime."""
from pylsl import StreamInlet, resolve_byprop

def abrir(nombre):
    s = resolve_byprop('name', nombre, timeout=10)
    if not s:
        raise SystemExit(f"No encontre '{nombre}'. ¿Esta corriendo orquestador.py?")
    return StreamInlet(s[0])

marc, paso = abrir('Marcadores'), abrir('Paso')
print('Escuchando... Ctrl+C para salir')
while True:
    m, _ = marc.pull_sample(timeout=0.0)
    if m:
        print('MARCADOR:', m[0])
    p, _ = paso.pull_sample(timeout=0.0)
    if p:
        print(f"PASO: p'={p[0]:.2f} dir={int(p[1]):+d} dtheta={p[2]:.1f}")