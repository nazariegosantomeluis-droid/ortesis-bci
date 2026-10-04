"""Plan B: repite en el tablero una sesion grabada, sin casco ni calibracion.

Cada sesion del orquestador deja, junto a su CSV, resultados/sesion_..._estado.jsonl con todo lo
que publico en el flujo Estado (senales, pasos, semaforos, checkpoints, IIC) y la hora de cada
cosa. Este programa lo vuelve a publicar en el flujo Estado, al ritmo original o mas rapido: el
tablero lo muestra igual que en vivo. Con --puerto, ademas la ortesis repite los mismos angulos.

  python tablero.py                                                      terminal 1
  python repetir_sesion.py resultados/sesion_real_XXXX_estado.jsonl      terminal 2
  python repetir_sesion.py --ultima --velocidad 2 --puerto COM4          la mas reciente, al doble, con la ortesis

Por defecto empieza en el lazo (se salta los minutos de calibracion, pero muestra sus checkpoints);
--completa repite todo. Es una REPETICION: nada se decide en vivo, y hay que decirlo al mostrarla.
"""
import argparse
import json
import time
from pathlib import Path

import config

ESPERA_MAX_S = 3.0          # ninguna espera entre eventos dura mas que esto (pausas, calibracion)


def leer(ruta):
    """[(hora, evento)] de un archivo _estado.jsonl, en orden."""
    eventos = []
    with open(ruta, encoding='utf-8') as f:
        for linea in f:
            if linea.strip():
                d = json.loads(linea)
                if 'evento' in d:                 # las lineas de marcadores no se repiten
                    eventos.append((d['t'], d['evento']))
    return eventos


def ultima(backend='real'):
    """La sesion grabada mas reciente de ese backend, o None."""
    rutas = sorted(config.RESULTADOS.glob(f'sesion_{backend}_*{config.SUFIJO_ESTADO}'), key=lambda r: r.stat().st_mtime)
    return rutas[-1] if rutas else None


def desde_el_lazo(eventos):
    """Los checkpoints de la calibracion y, despues, todo desde la senal del primer ensayo del lazo."""
    primero = next((i for i, (_, e) in enumerate(eventos) if e['tipo'] in ('paso', 'ajeno')), None)
    if primero is None:
        return eventos
    inicio = max((i for i in range(primero) if eventos[i][1]['tipo'] == 'cue'), default=primero)
    t0 = eventos[inicio][0]
    return [(t0, e) for _, e in eventos[:inicio] if e['tipo'] == 'checkpoint'] + eventos[inicio:]


def repetir(ruta, velocidad=1.0, completa=False, publicar=None, ortesis=None, dormir=time.sleep, salida=print):
    """Vuelve a publicar la sesion. velocidad: 1 = ritmo original, 2 = al doble, 0 = sin esperas.
    publicar(evento): por defecto, el flujo LSL Estado. Devuelve cuantos eventos publico."""
    eventos = leer(ruta)
    if not completa:
        eventos = desde_el_lazo(eventos)
    if publicar is None:
        from pylsl import StreamOutlet
        flujo = StreamOutlet(config.crear_info('Estado'))
        salida('Esperando al tablero (python tablero.py)...')
        if not flujo.wait_for_consumers(15.0):
            salida('  nadie escucha el flujo Estado; se repite de todos modos.')
        publicar = lambda e: flujo.push_sample([json.dumps(e)])
    pasos = sum(e['tipo'] in ('paso', 'ajeno') for _, e in eventos)
    salida(f'REPETICION de {Path(ruta).name}: {len(eventos)} eventos, {pasos} pasos, velocidad x{velocidad:g}. '
           f'Nada se decide en vivo.')
    previo = eventos[0][0] if eventos else 0.0
    for t, e in eventos:
        if velocidad > 0:
            dormir(min(max(t - previo, 0.0) / velocidad, ESPERA_MAX_S))
        previo = t
        publicar(e)
        if e['tipo'] == 'checkpoint':
            salida('  ' + e['texto'])
        elif e['tipo'] == 'cue':
            salida('  >>> ' + ('CERRAR' if e['meta'] > 0 else 'RELAJA'))
        elif e['tipo'] in ('paso', 'ajeno') and ortesis is not None:
            ortesis.mover(e['angulo'])
    if ortesis is not None:
        ortesis.mover(config.POSICION_SEGURA, config.PAUSA_DURACION_MS)
    salida('Fin de la repeticion.')
    return len(eventos)


def main():
    ap = argparse.ArgumentParser(description='Plan B: repite una sesion grabada en el tablero')
    ap.add_argument('archivo', nargs='?', help=f'resultados/sesion_..{config.SUFIJO_ESTADO}')
    ap.add_argument('--ultima', action='store_true', help='la sesion grabada mas reciente')
    ap.add_argument('--backend', choices=['real', 'sim'], default='real', help='con --ultima: de que tipo')
    ap.add_argument('--velocidad', type=float, default=1.0, help='1 = ritmo original; 2 = al doble; 0 = sin esperas')
    ap.add_argument('--completa', action='store_true', help='tambien la calibracion (por defecto, desde el lazo)')
    ap.add_argument('--puerto', help='mueve la ortesis a los angulos grabados (por ejemplo COM4)')
    a = ap.parse_args()
    ruta = ultima(a.backend) if a.ultima else a.archivo
    if not ruta or not Path(ruta).exists():
        raise SystemExit('No hay sesion grabada que repetir. Da un archivo _estado.jsonl o usa --ultima.')
    ortesis = None
    if a.puerto:
        import hardware as hw
        ortesis = hw.OrtesisSerial(a.puerto)
    try:
        repetir(ruta, a.velocidad, a.completa, ortesis=ortesis)
    finally:
        if ortesis is not None:
            ortesis.cerrar()


if __name__ == '__main__':
    main()
