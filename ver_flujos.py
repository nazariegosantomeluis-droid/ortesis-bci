"""Diagnostico de LSL: lista los flujos visibles y muestra lo que publica el orquestador.

Uso:  python ver_flujos.py             lista flujos y escucha Marcadores + Paso
      python ver_flujos.py --solo_lista
"""
import argparse
import json
from pylsl import StreamInlet, resolve_streams, resolve_byprop

ap = argparse.ArgumentParser()
ap.add_argument('--solo_lista', action='store_true')
a = ap.parse_args()

flujos = resolve_streams(wait_time=2.0)
print(f'{len(flujos)} flujo(s) en la red:')
for s in flujos:
    print(f'  {s.name():12s} tipo={s.type():8s} canales={s.channel_count()} '
          f'Hz={s.nominal_srate():g} id={s.source_id()} host={s.hostname()}')
if a.solo_lista:
    raise SystemExit

entradas = {}
for nombre in ('Marcadores', 'Paso', 'Estado'):
    s = resolve_byprop('name', nombre, timeout=3)
    if s:
        entradas[nombre] = StreamInlet(s[0])
if not entradas:
    raise SystemExit('No encontre flujos del orquestador. ¿Esta corriendo orquestador.py?')
print('Escuchando... Ctrl+C para salir')
while True:
    for nombre, inl in entradas.items():
        m, t = inl.pull_sample(timeout=0.02)
        if m is None:
            continue
        if nombre == 'Marcadores':
            print(f'{t:12.3f}  MARCADOR  {m[0]}')
        elif nombre == 'Paso':
            print(f"{t:12.3f}  PASO      p'={m[0]:.2f} dir={int(m[1]):+d} delta={m[2]:+.2f}")
        elif json.loads(m[0]).get('tipo') == 'checkpoint':
            print(f'{t:12.3f}  CHECKPOINT {json.loads(m[0])["texto"]}')
