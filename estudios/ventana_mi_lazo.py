"""Ventana de decision de MI: que parte del ensayo mira cada paso del lazo (EXPLORATORIO).

Con el casco real (P001, 4 de octubre), la desincronizacion (ERD) aparecio de 2 a 4 s despues de la senal y la
calibracion decide con los ULTIMOS 2 s de una senal de 4 s, o sea [2, 4]. Este estudio responde dos cosas:

  sesion    de una sesion grabada (resultados/sesion_..._estado.jsonl): en que segundos tras la senal cae la ventana de
            MI de cada paso del ensayo (la orden del paso 1 sale a ~4 s; cada paso siguiente, ~1 s despues) y cuanto de
            [2, 4] cubre. Sirve para el gemelo y para la sesion real del domingo.
  real      de un XDF de LabRecorder con una calibracion de MI: la exactitud balanceada (BA) del decoder C3/Cz/C4 segun
            la ventana tras la senal, con prueba de permutaciones. Un participante, pocos ensayos: es una tendencia,
            no una estimacion.

  python estudios/ventana_mi_lazo.py sesion [--ultima | resultados/sesion_real_..._estado.jsonl]
  python estudios/ventana_mi_lazo.py real ruta/archivo.xdf
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
import config

OBJETIVO_S = config.MI_VENTANA_OBJETIVO_S
VENTANAS = ((0, 2), (0.5, 2.5), (1, 3), (1.5, 3.5), (2, 4))


def solapamiento(ini, fin, objetivo=OBJETIVO_S):
    """Segundos de [ini, fin] que caen dentro del objetivo."""
    return max(0.0, min(fin, objetivo[1]) - max(ini, objetivo[0]))


def ventanas_por_paso(registros):
    """registros: lineas de un _estado.jsonl ({'t', 'marcador'} o {'t', 'evento'}). Solo cuenta el lazo (tras el primer
    marcador bloque:LAZO_*). Devuelve {paso del ensayo (1..): [segundos del cue a la orden de cada ensayo]}: la ventana de
    MI de ese paso termina cuando sale la orden y dura config.VENTANA_MI."""
    cue, k, lazo, por = None, 0, False, {}
    for d in registros:
        m = d.get('marcador')
        if m is None:
            continue
        if m.startswith('bloque:LAZO'):
            lazo = True
        if not lazo:
            continue
        if m in (config.CUE_CERRAR, config.CUE_RELAJA):
            cue, k = d['t'], 0
        elif cue is not None and (m.startswith('paso_ack') or m.startswith('paso_quieto')):
            k += 1
            por.setdefault(k, []).append(d['t'] - cue)
    return por


def primera_ventana_esperada():
    """[inicio, fin] en segundos tras la senal de la ventana del primer paso, segun config (la espera del cue)."""
    fin = config.VENTANA_MI + config.ESPERA_PRIMER_PASO_S
    return fin - config.VENTANA_MI, fin


def tabla_sesion(por):
    filas = []
    for k in sorted(por):
        fin = float(np.mean(por[k]))
        filas.append({'paso': k, 'n': len(por[k]), 'ini': fin - config.VENTANA_MI, 'fin': fin,
                      'cubre_s': solapamiento(fin - config.VENTANA_MI, fin)})
    return filas


def ba_por_ventana(x_filtrado, t, cues, fs, ventanas=VENTANAS, canales=None, repeticiones=12, permutaciones=100, semilla=0):
    """BA (validacion cruzada, media de `repeticiones` barajas) del decoder de MI en cada ventana tras la senal, y el
    p de permutar las etiquetas. x_filtrado: canales x muestras ya con la banda de MI; t: hora de cada muestra; cues:
    [(hora de la senal, meta +1/-1)]. Devuelve [(ini, fin, ba, p, nulo_medio)]."""
    import hardware as hw
    canales = config.indices('mi') if canales is None else canales
    y = np.array([m > 0 for _, m in cues], dtype=int)
    rng = np.random.default_rng(semilla)

    def ba(X, yy, reps):
        out = []
        for r in range(reps):
            p = np.random.default_rng(r).permutation(len(yy))
            out.append(hw.DecoderIM().ajustar(X[p], yy[p], {'c': canales}).ba)
        return float(np.mean(out))
    filas = []
    for a, b in ventanas:
        X = []
        for c, _ in cues:
            i0 = int(np.searchsorted(t, c + a))
            X.append(x_filtrado[:, i0:i0 + int(round((b - a) * fs))])
        X = np.array(X)
        real = ba(X, y, repeticiones)
        nulo = [ba(X, rng.permutation(y), 1) for _ in range(permutaciones)]
        filas.append((a, b, real, (1 + sum(n >= real for n in nulo)) / (1 + permutaciones), float(np.mean(nulo))))
    return filas


def _sesion(a):
    ruta = max(config.RESULTADOS.glob('sesion_real_*' + config.SUFIJO_ESTADO), key=lambda p: p.stat().st_mtime) \
        if a.ultima or not a.archivo else Path(a.archivo)
    registros = [json.loads(l) for l in ruta.read_text(encoding='utf-8').splitlines() if l.strip()]
    filas = tabla_sesion(ventanas_por_paso(registros))
    print(f'{ruta.name}: ventana de MI de cada paso del ensayo, en segundos tras la senal (la calibracion usa {OBJETIVO_S})')
    for f in filas:
        print(f"  paso {f['paso']}: [{f['ini']:.2f}, {f['fin']:.2f}] s   cubre {f['cubre_s']:.2f} de {OBJETIVO_S[1] - OBJETIVO_S[0]:.0f} s   (n = {f['n']})")
    if not filas:
        print('  (sin pasos del lazo en esta sesion)')
    return filas


def _real(a):
    import hardware as hw
    import xdf_minimo
    fl = xdf_minimo.leer(a.archivo)
    por_nombre = {s['nombre']: s for s in fl.values()}
    eeg, mar = por_nombre['EEG'], por_nombre['Marcadores']
    x, t, fs = eeg['x'].astype(float).T, eeg['t'], eeg['fs']
    nm = [m[0] if not isinstance(m, str) else m for m in mar['x']]
    tm = np.array(mar['t'])
    ini = next((tt for tt, n in zip(tm, nm) if n == config.m_bloque('CAL_MI')), tm[0])
    fin = next((tt for tt, n in zip(tm, nm) if n == config.m_bloque('CAL_ERRP')), tm[-1])
    cues = [(tt, 1 if n == config.CUE_CERRAR else -1) for tt, n in zip(tm, nm)
            if n in (config.CUE_CERRAR, config.CUE_RELAJA) and ini <= tt < fin]
    xf = hw.filtrar(x, config.BANDA_MI, fs)
    print(f'{Path(a.archivo).name}: {len(cues)} ensayos de MI ({sum(m > 0 for _, m in cues)} de cerrar), decoder C3/Cz/C4, banda {config.BANDA_MI}')
    print('  ventana tras la senal   BA     p (100 permutaciones)   BA con etiquetas permutadas')
    for a_, b_, ba, p, nulo in ba_por_ventana(xf, t, cues, fs):
        print(f'  [{a_:>3}, {b_:>3}] s            {ba:.3f}  p = {p:.3f}                {nulo:.3f}')
    print('EXPLORATORIO: un participante; se miraron 5 ventanas (el p no esta corregido) y los ensayos solo duran 4 s,\n'
          'asi que no se pueden probar ventanas mas tardias.')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='orden', required=True)
    p = sub.add_parser('sesion')
    p.add_argument('archivo', nargs='?')
    p.add_argument('--ultima', action='store_true')
    p = sub.add_parser('real')
    p.add_argument('archivo')
    a = ap.parse_args(argv)
    return _sesion(a) if a.orden == 'sesion' else _real(a)


if __name__ == '__main__':
    main()
