"""Calibraciones y registro de una sesion grabada con LabRecorder (XDF) -> los archivos que usan el resto de las herramientas.

Lee los flujos EEG, Marcadores, Paso y Estado y deja, junto al XDF o en --salida:
  calibracion_mi_<nombre>.npz      X (n, 8, 500): ventana de MI de 2 s (ultimos 2 s de los 4 s tras cada cue), filtrada 8-30 Hz,
                                   como la del orquestador; y: 1 = cerrar. TODOS los cues, incluidos los ensayos que el orquestador
                                   reintento (el XDF no dice cuales se aceptaron): se descartan los que tienen artefacto de amplitud.
  calibracion_errp_<nombre>.npz    X (n, 8, 250): epoca de ErrP (-0.2 a 0.8 s del inicio del movimiento), filtrada 1-10 Hz con linea
                                   base; y: 1 = el movimiento contradijo al cue; direccion.
  sesion_xdf_<nombre>_estado.jsonl los eventos del flujo Estado (checkpoints, cues, pasos) y los marcadores, como `Salidas.estado`.
Si la grabacion no tiene el lazo adaptativo (solo calibracion), no hay CSV de pasos: las herramientas que lo piden dicen «no hay dato».

Uso: python desde_xdf.py <archivo.xdf> [--salida carpeta]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config

FS = config.FLUJOS['EEG'][2]


def leer(ruta):
    import pyxdf
    datos, _ = pyxdf.load_xdf(str(ruta))
    return {s['info']['name'][0]: s for s in datos}


def ventanas_mi(S, duracion=config.DURACION_MI_S, umbral_amp=150.0):
    import hardware as hw
    x, t = np.asarray(S['EEG']['time_series'], dtype=float).T, np.asarray(S['EEG']['time_stamps'], dtype=float)
    X, y = [], []
    marcas = [(ts, m[0]) for ts, m in zip(S['Marcadores']['time_stamps'], S['Marcadores']['time_series'])]
    bloques = [ts for ts, m in marcas if m.startswith('bloque:')]
    fin_mi = bloques[1] if len(bloques) > 1 else np.inf                          # los cues de la calibracion de MI: antes del 2.o bloque
    for ts, m in marcas:
        if m not in (config.CUE_CERRAR, config.CUE_RELAJA) or ts >= fin_mi:
            continue
        fin = ts + duracion
        i = np.searchsorted(t, [fin - config.VENTANA_MI - 1.0, fin])           # 1 s de margen para el filtro
        seg = x[:, i[0]:i[1]]
        if seg.shape[1] < int((config.VENTANA_MI + 1.0) * FS) - 5 or t[min(i[1], len(t)) - 1] < fin - 0.05:
            continue
        v = hw.filtrar(seg, config.BANDA_MI, FS)[:, -int(config.VENTANA_MI * FS):]
        if np.ptp(v, axis=1).max() > umbral_amp:                                # parpadeo o movimiento grande
            continue
        X.append(v)
        y.append(int(m == config.CUE_CERRAR))
    return np.array(X), np.array(y)


def epocas_errp(S):
    import hardware as hw
    x, t = np.asarray(S['EEG']['time_series'], dtype=float).T, np.asarray(S['EEG']['time_stamps'], dtype=float)
    marcas = sorted([(ts, m[0]) for ts, m in zip(S['Marcadores']['time_stamps'], S['Marcadores']['time_series'])])
    pasos = list(zip(S['Paso']['time_stamps'], np.asarray(S['Paso']['time_series'])))
    inicios = {m.split(':')[1]: ts for ts, m in marcas if m.startswith('paso_inicio:')}
    X, y, d = [], [], []
    k = 0
    for ts, m in marcas:
        if not m.startswith('paso_ack:'):
            continue
        seq = m.split(':')[1]
        cue = [mm for tt, mm in marcas if tt <= ts and mm in (config.CUE_CERRAR, config.CUE_RELAJA)]
        paso = [p for tp, p in pasos if abs(tp - ts) < 1.0]
        if not cue or not paso or seq not in inicios:
            continue
        meta = 1 if cue[-1] == config.CUE_CERRAR else 0
        dirpaso = int(paso[-1][0])
        e = hw.cortar_epoca(x, t, inicios[seq], FS)
        if e is None:
            continue
        X.append(e)
        y.append(int(dirpaso != meta))
        d.append(dirpaso)
    return np.array(X), np.array(y), np.array(d)


def estado_jsonl(S, ruta):
    t0 = S['EEG']['time_stamps'][0]
    filas = [(ts, {'t': float(ts - t0), 'evento': json.loads(m[0])}) for ts, m in zip(S['Estado']['time_stamps'], S['Estado']['time_series'])]
    filas += [(ts, {'t': float(ts - t0), 'marcador': m[0]}) for ts, m in zip(S['Marcadores']['time_stamps'], S['Marcadores']['time_series'])]
    with open(ruta, 'w', encoding='utf-8') as f:
        for _, l in sorted(filas, key=lambda z: z[0]):
            f.write(json.dumps(l, ensure_ascii=False) + '\n')


def convertir(ruta_xdf, salida=None):
    ruta_xdf = Path(ruta_xdf)
    salida = Path(salida) if salida else ruta_xdf.parent
    nombre = ruta_xdf.stem.split('_eeg')[0][-40:].replace('-', '_')
    S = leer(ruta_xdf)
    Xm, ym = ventanas_mi(S)
    Xe, ye, de = epocas_errp(S)
    salida.mkdir(parents=True, exist_ok=True)
    np.savez(salida / f'calibracion_mi_{nombre}.npz', X=Xm, y=ym)
    np.savez(salida / f'calibracion_errp_{nombre}.npz', X=Xe, y=ye, direccion=de)
    estado_jsonl(S, salida / f'sesion_xdf_{nombre}{config.SUFIJO_ESTADO}')
    return {'mi': salida / f'calibracion_mi_{nombre}.npz', 'errp': salida / f'calibracion_errp_{nombre}.npz',
            'estado': salida / f'sesion_xdf_{nombre}{config.SUFIJO_ESTADO}', 'n_mi': len(ym), 'n_errp': len(ye),
            'pasos_lazo': sum(1 for m in S['Estado']['time_series'] if json.loads(m[0]).get('tipo') == 'paso'),
            'duracion_s': float(S['EEG']['time_stamps'][-1] - S['EEG']['time_stamps'][0])}


def main():
    ap = argparse.ArgumentParser(description='XDF de LabRecorder -> calibraciones y registro de estado')
    ap.add_argument('xdf')
    ap.add_argument('--salida')
    r = convertir(**{'ruta_xdf': ap.parse_args().xdf, 'salida': ap.parse_args().salida})
    print(r)


if __name__ == '__main__':
    main()
