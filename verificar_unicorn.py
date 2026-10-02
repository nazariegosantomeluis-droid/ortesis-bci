"""Comprueba, con el casco puesto y en menos de 5 minutos, lo que no se pudo verificar sin el
g.tec Unicorn Hybrid Black, y dice que fuente de EEG usar.

  python verificar_unicorn.py brainflow [--serie UN-XXXX.XX.XX]    conecta por BrainFlow (dongle)
  python verificar_unicorn.py lsl [--nombre <nombre del flujo>]    lee la app UnicornLSL
  python verificar_unicorn.py ambas                                 primero BrainFlow, luego LSL

OJO: solo UNA aplicacion puede conectarse al casco a la vez. Para 'brainflow' cierra la
Unicorn Suite; para 'lsl' abre UnicornLSL, conecta el casco y pulsa Start. Con 'ambas' el
programa te avisa cuando cambiar.

Guia de ~50 s por fuente: quieto con ojos abiertos, parpadear, ojos cerrados y mover la cabeza.
Comprueba:
  - tasa de muestreo (250 Hz) y que el contador avance de 1 en 1 (y cuantas muestras se pierden);
  - el orden de los canales: el contador esta donde se espera, el acelerometro mide ~1 g en
    reposo, la bateria esta entre 0 y 100 y la validez es 0 o 1;
  - las unidades del EEG (microvolts) y su offset de continua;
  - que Fz sea el canal 1 (los parpadeos son maximos ahi) y que el alfa con ojos cerrados sea
    occipital (PO7, Oz, PO8): asi se confirma el orden dentro del EEG;
  - que el giroscopio responda al mover la cabeza.
El veredicto queda en pantalla y en resultados/verificacion_unicorn.json.
"""
import argparse
import json
import time

import numpy as np

import config

CRITICAS = ['tasa', 'contador', 'acelerometro', 'bateria', 'validez', 'eeg_unidades']


def fases(auto=False):
    """(clave, segundos, instruccion). auto=True: sin guia, para probar con el gemelo."""
    if auto:
        return [('reposo', 6.0, ''), ('parpadeo', 4.0, ''), ('cerrados', 4.0, ''), ('cabeza', 4.0, '')]
    return [('reposo', 15.0, 'QUIETO, ojos abiertos, mirando al frente'),
            ('parpadeo', 10.0, 'PARPADEA fuerte, una vez por segundo'),
            ('cerrados', 15.0, 'CIERRA los ojos y relajate'),
            ('cabeza', 8.0, 'ABRE los ojos y di que si y que no con la cabeza')]


# ------------------------------------------------------------ lectura
def _grabar(leer_bloque, fases_, salida):
    """Graba todas las fases. leer_bloque() -> (muestras x canales, llegadas). Devuelve x, llegada y fase."""
    xs, ts, fs_ = [], [], []
    for k, (clave, segundos, texto) in enumerate(fases_):
        if texto:
            salida(f'  >>> {texto}  ({segundos:.0f} s)')
        t_fin = time.time() + segundos
        while time.time() < t_fin:
            x, t = leer_bloque()
            if len(t):
                xs.append(np.asarray(x, dtype=float)); ts.append(np.asarray(t, dtype=float))
                fs_.append(np.full(len(t), k))
            time.sleep(0.02)
    if not xs:
        raise RuntimeError('no llego ninguna muestra')
    return np.vstack(xs).T, np.concatenate(ts), np.concatenate(fs_)


def leer_lsl(nombre=None, tipo='Data', fases=None, salida=print):
    """Graba el flujo de la app UnicornLSL (por nombre o, si no se da, por tipo)."""
    from pylsl import StreamInlet, resolve_byprop
    clave, valor = ('name', nombre) if nombre else ('type', tipo)
    s = resolve_byprop(clave, valor, timeout=15)
    if not s:
        raise RuntimeError(f'no encontre un flujo LSL con {clave}={valor}. ¿Esta abierta la app UnicornLSL y en Start?')
    info = s[0]
    entrada = StreamInlet(info)
    x, llegada, fase = _grabar(lambda: entrada.pull_chunk(timeout=0.0), fases, salida)
    f = config.FUENTES_EEG['unicornlsl']
    return {'fuente': 'lsl', 'nombre': info.name(), 'x': x, 'llegada': llegada, 'fase': fase,
            'fs': float(info.nominal_srate()) or 250.0, 'modulo': None, 'fases': [c for c, _, _ in fases],
            'mapa': {k: f[k] for k in ('eeg', 'imu', 'bateria', 'contador', 'validez')}}


def leer_brainflow(serie=None, fases=None, salida=print, placa='unicorn'):
    """Graba directo del casco con BrainFlow (placa='sintetica' solo para probar este camino)."""
    import puente_lsl
    from brainflow.board_shim import BoardShim
    BoardShim.disable_board_logger()
    plan = puente_lsl.plan_placa(placa, serie=serie)
    board = BoardShim(plan['board_id'], plan['params'])
    board.prepare_session()
    try:
        board.start_stream(45000, '')
        time.sleep(1.0)
        board.get_board_data()

        def bloque():
            d = board.get_board_data()
            return d.T, np.full(d.shape[1], time.time())
        x, llegada, fase = _grabar(bloque, fases, salida)
    finally:
        for soltar in (board.stop_stream, board.release_session):
            try:
                soltar()
            except Exception:
                pass
    return {'fuente': 'brainflow', 'nombre': serie, 'x': x, 'llegada': llegada, 'fase': fase,
            'fs': float(plan['fs']), 'modulo': plan['modulo'], 'fases': [c for c, _, _ in fases],
            'mapa': {k: plan[k] for k in ('eeg', 'imu', 'bateria', 'contador', 'validez')}}


# ------------------------------------------------------------ comprobaciones
def _parece_contador(v, modulo=None):
    d = np.diff(v)
    if modulo:
        d = d % modulo
    return bool(len(d) and np.all(d == np.round(d)) and np.all(d >= 1) and np.mean(d == 1) > 0.8)


def evaluar(datos):
    """Lista de {'clave', 'estado' (OK | AVISO | FALLA), 'texto'} a partir de lo grabado."""
    import hardware as hw
    x, fs, m, fase = datos['x'], datos['fs'], datos['mapa'], datos['fase']
    nombres = datos['fases']
    de = lambda clave: fase == nombres.index(clave) if clave in nombres else np.zeros(len(fase), bool)
    res = []

    def anotar(clave, estado, texto):
        res.append({'clave': clave, 'estado': estado, 'texto': texto})

    # --- contador y tasa
    c = x[m['contador']]
    if _parece_contador(c, datos['modulo']):
        d = np.diff(c) % datos['modulo'] if datos['modulo'] else np.diff(c)
        perdidas, total = int((d - 1).sum()), int(d.sum()) + 1
        rafagas = int((d > 1).sum())
        anotar('contador', 'OK' if perdidas <= 0.01 * total else 'AVISO',
               f'avanza de 1 en 1; {perdidas} muestras perdidas de {total} ({100 * perdidas / total:.2f} %) en {rafagas} rafagas'
               + (f', la mayor de {1000 * (d.max() - 1) / fs:.0f} ms' if rafagas else ''))
        duracion = datos['llegada'][-1] - datos['llegada'][0]
        tasa = (total - 1) / duracion if duracion > 0 else 0.0
        anotar('tasa', 'OK' if abs(tasa - 250) < 0.03 * 250 else 'FALLA', f'{tasa:.1f} muestras por segundo segun el contador (se esperan 250)')
        hora = c / fs                                         # jitter de llegada contra la hora del contador
        retraso = datos['llegada'] - hora
        anotar('jitter', 'OK', f'jitter de llegada por Bluetooth: p95 {1000 * (np.quantile(retraso, .95) - retraso.min()):.0f} ms '
                               f'(la hora de cada muestra se reconstruye con el contador)')
    else:
        otros = [k for k in range(x.shape[0]) if k != m['contador'] and _parece_contador(x[k], datos['modulo'])]
        anotar('contador', 'FALLA', f"el canal {m['contador']} no avanza de 1 en 1: no es el contador"
               + (f'; el contador parece estar en el canal {otros[0]}' if otros else '; ningun canal lo parece'))
        anotar('tasa', 'FALLA', 'sin contador no se puede medir la tasa real')
    # --- acelerometro y giroscopio
    quieto = de('reposo') if de('reposo').any() else np.ones(len(fase), bool)
    if m['imu']:
        g = float(np.median(np.linalg.norm(x[m['imu'][:3]][:, quieto], axis=0)))
        if 0.9 <= g <= 1.1:
            anotar('acelerometro', 'OK', f'{g:.2f} g en reposo')
        elif 9.0 <= g <= 10.6:
            anotar('acelerometro', 'AVISO', f'{g:.2f} en reposo: esta en m/s2, no en g (hay que dividir entre 9.81)')
        elif 900 <= g <= 1100:
            anotar('acelerometro', 'AVISO', f'{g:.0f} en reposo: esta en mg, no en g (hay que dividir entre 1000)')
        else:
            anotar('acelerometro', 'FALLA', f'{g:.3g} en reposo: no son ~1 g; esos canales no son el acelerometro')
        giro = np.abs(x[m['imu'][3:]])
        reposo = float(np.median(giro[:, quieto]))
        mov = float(giro[:, de('cabeza')].max()) if de('cabeza').any() else 0.0
        if reposo > 5:
            anotar('giroscopio', 'AVISO', f'{reposo:.1f} grados/s en reposo: ¿la cabeza estaba quieta? ¿son otras unidades?')
        elif mov > 20:
            anotar('giroscopio', 'OK', f'{reposo:.1f} grados/s en reposo y hasta {mov:.0f} al mover la cabeza')
        else:
            anotar('giroscopio', 'AVISO', f'no se vio movimiento de cabeza (maximo {mov:.1f} grados/s)')
    else:
        anotar('acelerometro', 'FALLA', 'la fuente no trae IMU')
    # --- bateria y validez
    if m['bateria'] is not None:
        b = x[m['bateria']]
        anotar('bateria', 'OK' if 0 < b.min() <= b.max() <= 100 else 'FALLA',
               f'{b[-1]:.0f} %' if 0 < b.min() <= b.max() <= 100 else f'valores de {b.min():.3g} a {b.max():.3g}: no es un porcentaje de bateria')
        if 0 < b[-1] < 30:
            anotar('bateria_baja', 'AVISO', f'bateria al {b[-1]:.0f} %: cargala antes de la demo')
    else:
        anotar('bateria', 'FALLA', 'la fuente no trae bateria')
    if m['validez'] is not None:
        v = x[m['validez']]
        if set(np.unique(v)) <= {0.0, 1.0}:
            anotar('validez', 'OK' if v.mean() > 0.99 else 'AVISO', f'{100 * v.mean():.1f} % de las muestras marcadas como validas')
        else:
            anotar('validez', 'FALLA', f'valores {np.unique(v)[:5]}: no es un indicador 0 / 1')
    else:
        anotar('validez', 'FALLA', 'la fuente no trae indicador de validez')
    # --- EEG: unidades, offset y orden
    eeg = x[m['eeg']]
    if eeg.shape[1] > 2 * fs and np.isfinite(eeg).all():
        rms = hw.filtrar(eeg[:, quieto], (1.0, 40.0), fs).std(axis=1)
        med, offset = float(np.median(rms)), float(np.abs(eeg.mean(axis=1)).max())
        detalle = ', '.join(f'{c} {r:.0f}' for c, r in zip(config.CANALES_EEG, rms))
        if 1.0 <= med <= 150.0:
            anotar('eeg_unidades', 'OK', f'microvolts: RMS de 1 a 40 Hz por canal: {detalle}')
        elif 1e-6 <= med <= 150e-6:
            anotar('eeg_unidades', 'FALLA', f'RMS mediano {med:.2e}: el EEG parece venir en volts, no en microvolts')
        else:
            anotar('eeg_unidades', 'FALLA', f'RMS mediano {med:.3g} uV fuera de lo plausible (electrodos sin contacto u otras unidades): {detalle}')
        anotar('eeg_offset', 'OK' if offset < config.SALUD['canal_saturado_uv'] else 'AVISO',
               f'offset de continua maximo {offset / 1000:.1f} mV (el umbral de saturacion es {config.SALUD["canal_saturado_uv"] / 1000:.0f} mV)')
        malos = hw.revisar_canales(eeg[:, quieto][:, -int(2 * fs):], fs)
        if malos:
            anotar('eeg_canales', 'AVISO', 'canales con problema ahora: ' + '; '.join(f'{c} {v}' for c, v in malos.items()))
        if de('parpadeo').any():
            pp = np.ptp(hw.filtrar(eeg[:, de('parpadeo')], (1.0, 10.0), fs), axis=1)
            mayor = config.CANALES_EEG[int(np.argmax(pp))]
            if pp.max() < 40:
                anotar('parpadeo_fz', 'AVISO', 'no se vieron parpadeos claros: no se pudo confirmar donde esta Fz')
            else:
                anotar('parpadeo_fz', 'OK' if mayor == 'Fz' else 'AVISO',
                       f'los parpadeos son maximos en {mayor} ({pp.max():.0f} uV)'
                       + ('' if mayor == 'Fz' else ': se esperaba Fz; revisa el orden de los canales de EEG'))
        if de('cerrados').any() and de('cerrados').sum() > 2 * fs:
            alfa = hw.filtrar(eeg[:, de('cerrados')], (8.0, 12.0), fs).var(axis=1)
            occ, resto = alfa[config.indices('alfa')].mean(), alfa[config.indices(['Fz', 'C3', 'Cz', 'C4'])].mean()
            anotar('alfa_occipital', 'OK' if occ > resto else 'AVISO',
                   f'alfa con ojos cerrados: occipital / fronto-central = {occ / resto:.1f}'
                   + ('' if occ > resto else ': se esperaba mas alfa atras; revisa el orden de los canales de EEG'))
    else:
        anotar('eeg_unidades', 'FALLA', 'no hay EEG suficiente o tiene valores no finitos')
    return res


def veredicto(por_fuente, nombre):
    """Que fuente usar el dia de la demo, segun las comprobaciones criticas de cada una."""
    def pasa(res):
        estados = {r['clave']: r['estado'] for r in res}
        return all(estados.get(k) == 'OK' for k in CRITICAS)
    lineas = []
    if 'brainflow' in por_fuente and pasa(por_fuente['brainflow']):
        lineas.append('VEREDICTO: usa BrainFlow (fuente principal):')
        lineas.append(f"    python puente_lsl.py --placa unicorn{' --serie ' + nombre if nombre else ''}")
        lineas.append('    python orquestador.py real --puerto COM4')
    elif 'lsl' in por_fuente and pasa(por_fuente['lsl']):
        lineas.append('VEREDICTO: usa la app UnicornLSL (respaldo), con el casco conectado y en Start:')
        lineas.append(f"    python orquestador.py real --puerto COM4 --fuente unicornlsl{' --eeg-nombre ' + nombre if nombre else ''}")
    else:
        fallas = sorted({r['clave'] for res in por_fuente.values() for r in res
                         if r['clave'] in CRITICAS and r['estado'] != 'OK'})
        lineas.append('VEREDICTO: NO uses todavia ninguna de las fuentes probadas. Falla: ' + ', '.join(fallas) + '.')
        lineas.append('    Corrige el mapa de canales en config.FUENTES_EEG (o las unidades) y vuelve a verificar.')
    avisos = [f"{f}: {r['clave']}" for f, res in por_fuente.items() for r in res if r['estado'] == 'AVISO']
    if avisos:
        lineas.append('  Avisos a revisar: ' + '; '.join(avisos))
    return '\n'.join(lineas)


def main():
    ap = argparse.ArgumentParser(description='Verificacion del Unicorn Hybrid Black con el casco puesto')
    ap.add_argument('fuente', choices=['brainflow', 'lsl', 'ambas'])
    ap.add_argument('--serie', help='numero de serie del casco (UN-XXXX.XX.XX); opcional con un solo casco')
    ap.add_argument('--nombre', help='nombre del flujo de UnicornLSL (por defecto se busca por tipo Data)')
    ap.add_argument('--auto', action='store_true', help='sin guia (para probar con el gemelo)')
    a = ap.parse_args()
    por_fuente, nombre = {}, a.serie or a.nombre
    for fuente in (['brainflow', 'lsl'] if a.fuente == 'ambas' else [a.fuente]):
        print(f'\n=== {fuente} ===')
        if a.fuente == 'ambas' and fuente == 'lsl':
            input('Ahora abre UnicornLSL, conecta el casco, pulsa Start y luego Enter aqui... ')
        try:
            datos = (leer_brainflow(a.serie, fases(a.auto)) if fuente == 'brainflow'
                     else leer_lsl(a.nombre, fases=fases(a.auto)))
        except Exception as e:
            print(f'  FALLA    conexion: {e}')
            por_fuente[fuente] = [{'clave': k, 'estado': 'FALLA', 'texto': 'sin conexion'} for k in CRITICAS]
            continue
        nombre = nombre or datos['nombre']
        por_fuente[fuente] = evaluar(datos)
        for r in por_fuente[fuente]:
            print(f"  {r['estado']:8s} {r['clave']}: {r['texto']}")
    texto = veredicto(por_fuente, nombre)
    print('\n' + texto)
    config.RESULTADOS.mkdir(exist_ok=True)
    ruta = config.RESULTADOS / 'verificacion_unicorn.json'
    ruta.write_text(json.dumps({'t': time.time(), 'nombre': nombre, 'por_fuente': por_fuente, 'veredicto': texto},
                               indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'\nInforme: {ruta}')


if __name__ == '__main__':
    main()
