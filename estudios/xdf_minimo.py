"""Lector minimo de archivos XDF (los que graba LabRecorder), sin pyxdf: lo justo para leer EEG y marcadores.

Devuelve {id: {nombre, tipo, canales, fs, formato, t, x, ...}}; `t` es la hora LSL de cada muestra (las que no traen marca
se reconstruyen con la tasa nominal) y NO aplica los desfases de reloj del pie del archivo. Pensado para estudios/; para
analisis serios usar pyxdf.
"""
import struct, re
import numpy as np
import xml.etree.ElementTree as ET

def _varlen(f):
    n = f.read(1)[0]
    if n == 1: return f.read(1)[0]
    if n == 4: return struct.unpack('<I', f.read(4))[0]
    if n == 8: return struct.unpack('<Q', f.read(8))[0]
    raise ValueError(n)

TIPOS = {'float32': ('<f4', 4), 'double64': ('<f8', 8), 'int32': ('<i4', 4), 'int16': ('<i2', 2), 'int8': ('<i1', 1), 'int64': ('<i8', 8)}

def leer(ruta):
    flujos = {}
    with open(ruta, 'rb') as f:
        assert f.read(4) == b'XDF:'
        while True:
            try:
                n = _varlen(f)
            except IndexError:
                break
            tag = struct.unpack('<H', f.read(2))[0]
            cuerpo = f.read(n - 2)
            if tag == 2:
                sid = struct.unpack('<I', cuerpo[:4])[0]
                xml = ET.fromstring(cuerpo[4:].decode('utf-8', 'replace'))
                flujos[sid] = dict(info=xml, nombre=xml.findtext('name'), tipo=xml.findtext('type'),
                                   canales=int(xml.findtext('channel_count')), fs=float(xml.findtext('nominal_srate')),
                                   formato=xml.findtext('channel_format'), t=[], x=[], offsets=[], pie=None)
            elif tag == 3:
                sid = struct.unpack('<I', cuerpo[:4])[0]
                s = flujos[sid]
                import io
                g = io.BytesIO(cuerpo[4:])
                m = _varlen(g)
                for _ in range(m):
                    ts_bytes = g.read(1)[0]
                    ts = struct.unpack('<d', g.read(8))[0] if ts_bytes == 8 else None
                    if s['formato'] == 'string':
                        L = _varlen(g)
                        val = g.read(L).decode('utf-8', 'replace')
                        s['x'].append(val)
                    else:
                        dt, sz = TIPOS[s['formato']]
                        val = np.frombuffer(g.read(sz * s['canales']), dtype=dt)
                        s['x'].append(val)
                    s['t'].append(ts)
            elif tag == 4:
                sid = struct.unpack('<I', cuerpo[:4])[0]
                col, off = struct.unpack('<dd', cuerpo[4:20])
                flujos[sid]['offsets'].append((col, off))
            elif tag == 6:
                sid = struct.unpack('<I', cuerpo[:4])[0]
                flujos[sid]['pie'] = ET.fromstring(cuerpo[4:].decode('utf-8', 'replace'))
    # marcas de tiempo: rellenar las que faltan (regulares) con la tasa nominal
    for s in flujos.values():
        t = s['t']
        if s['formato'] != 'string':
            s['x'] = np.array(s['x'])
        # interpolacion de marcas ausentes
        arr = np.array([np.nan if v is None else v for v in t], float)
        if len(arr) and np.isnan(arr).any() and s['fs'] > 0:
            idx = np.arange(len(arr))
            ok = ~np.isnan(arr)
            # sin marca explicita: la hora es la ultima marca conocida + k/fs
            ult = np.where(ok, idx, -1); ult = np.maximum.accumulate(ult)
            base = np.where(ult >= 0, arr[np.clip(ult, 0, None)], np.nan)
            arr = base + (idx - ult) / s['fs']
        s['t'] = arr
    return flujos
