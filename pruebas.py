"""Pruebas automaticas, sin hardware. Correlas antes de cada commit.

Uso:  python pruebas.py              pruebas rapidas (~1 min)
      python pruebas.py --completa   ademas el lazo real contra el cerebro sintetico (~5 min)
"""
import argparse
import csv
import json
import subprocess
import sys
import time
import traceback

import numpy as np

import config

# Las pruebas nunca llaman a la API real, haya o no llave en la maquina: usan ApiSimulada. Con una
# llave en .env, tablero_salud le preguntaba al copiloto de verdad (y fallaba: esperaba la plantilla).
import os
os.environ.pop('ANTHROPIC_API_KEY', None)
config.ARCHIVO_ENV = config.RAIZ / '.env.pruebas'            # no existe: ia.cliente() da None

RESULTADOS = []


def prueba(fn):
    def envoltura():
        t0 = time.time()
        try:
            detalle = fn() or ''
            RESULTADOS.append(True)
            print(f'  OK    {fn.__name__:32s} {time.time() - t0:5.1f} s  {detalle}')
        except Exception as e:
            RESULTADOS.append(False)
            print(f'  FALLA {fn.__name__:32s} {e}')
            traceback.print_exc(limit=2)
    envoltura.__name__ = fn.__name__
    return envoltura


# ------------------------------------------------------------ contrato
@prueba
def contrato():
    for nombre in config.FLUJOS:
        info = config.crear_info(nombre)
        assert info.name() == nombre
    assert len(set(config.COLUMNAS_CSV)) == len(config.COLUMNAS_CSV)
    for e, destinos in config.TRANSICIONES.items():
        assert e in config.ESTADOS and set(destinos) <= set(config.ESTADOS), e
    for e in config.ESTADOS:                 # de cualquier estado se llega a EVALUACION
        vistos, pila = set(), [e]
        while pila:
            x = pila.pop()
            vistos.add(x)
            pila += [d for d in config.TRANSICIONES[x] if d not in vistos]
        assert 'EVALUACION' in vistos, e
    assert 'PAUSA_SEGURA' in config.ESTADOS
    for e in ('LAZO_ESTATICO', 'LAZO_ADAPTATIVO', 'APRENDIZAJE_CONGELADO', 'PERTURBACION'):
        assert 'PAUSA_SEGURA' in config.TRANSICIONES[e], e
    assert set(config.TRANSICIONES['PAUSA_SEGURA']) == {
        'LAZO_ESTATICO', 'LAZO_ADAPTATIVO', 'APRENDIZAJE_CONGELADO', 'EVALUACION'}
    assert config.m_salud('eeg', config.ROJO) == 'salud:eeg:ROJO'
    assert config.COLUMNAS_CSV[-6:] == ['salud', 'excluido', 'alineacion', 'ajeno', 'n1_uv', 'iic']
    # Tarea 2: movimientos ajenos anunciados, fuera del analisis del agente
    assert config.m_paso_ajeno(3) == 'paso_ajeno:3' and config.AVISO_AJENO == 'aviso_ajeno'
    assert config.m_paso_quieto(3) == 'paso_quieto:3' and config.SIN_MOVIMIENTO == 'sin_movimiento'
    # control causal: los dos bloques tienen marcador y columna en el CSV
    assert [config.m_bloque(b) for b in config.BLOQUES_SHAM] == ['bloque:real', 'bloque:sham']
    assert 'bloque' in config.COLUMNAS_CSV and config.SHAM_ERRP_FUENTE in config.SHAM_ERRP_FUENTES
    assert 60 <= config.SHAM_ERRP_PASOS <= 80 and config.SHAM_ERRP_PERTURBAR_EN % config.PASOS_ENSAYO == 0
    assert config.IGNORAR_SIN_MOVIMIENTO is True
    assert 'ajeno' in config.MOTIVOS_EXCLUSION and config.CANALES_N1 == config.PAPELES['visual']
    # montaje del Unicorn Hybrid Black y el papel de cada sensor
    assert config.CANALES_EEG == ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
    assert config.PAPELES == {'mi': ['C3', 'Cz', 'C4'], 'errp': ['Fz', 'Cz', 'Pz'],
                              'visual': ['PO7', 'Oz', 'PO8'], 'alfa': ['PO7', 'Oz', 'PO8']}
    assert config.indices('mi') == [1, 2, 3] and config.indices(['Oz', 'Fz']) == [6, 0]
    assert config.FLUJOS['IMU'][1] == len(config.CANALES_IMU) == 6
    for nombre, f in config.FUENTES_EEG.items():      # mapa de canales de cada fuente de EEG
        usados = f['eeg'] + (f['imu'] or []) + [f[k] for k in ('bateria', 'contador', 'validez') if f[k] is not None]
        assert len(f['eeg']) == 8 and len(set(usados)) == len(usados) and max(usados) < f['canales'], nombre
    assert config.FUENTES_EEG['puente']['nombre'] == 'EEG' and config.FUENTES_EEG['puente']['contador'] is None
    lsl = config.FUENTES_EEG['unicornlsl']             # la app de g.tec: 17 canales de tipo 'Data'
    assert (lsl['tipo'], lsl['canales'], lsl['contador'], lsl['validez']) == ('Data', 17, 15, 16)
    return f'{len(config.FLUJOS)} flujos, {len(config.ESTADOS)} estados'


# ------------------------------------------------------------ salud
@prueba
def vigilante():
    from salud import Vigilante
    A, R = config.AMARILLO, config.ROJO
    bien_eeg = {'edad_s': 0.02, 'tasa_hz': 250.0, 'canales': {}}
    bien_ort = {'puerto_ok': True, 'acks_perdidos': 0, 'latencia_ms': 8.0}
    v = Vigilante()
    # calentamiento: con menos de 15 epocas validas el detector no opina ni emite cambios,
    # aunque el ConfianzaDetector ya este congelado
    minimo = config.SALUD['detector_epocas_min']
    assert v.colores['detector'] == config.CALENTANDO
    for n in (0, 5, minimo - 1):
        assert v.actualizar(0.0, eeg=bien_eeg, ortesis=bien_ort, reloj_ms=0.0,
                            detector={'fiabilidad': 0.3, 'congelado': True, 'epocas': n}) == []
    assert v.codigo() == 'VVVCC' and v.escalon() == 1 and v.motivo_pausa() is None   # piloto: gris, sin alfa
    # al terminar de calentar publica su primer color real
    assert v.actualizar(0.0, detector={'fiabilidad': 1.0, 'congelado': False, 'epocas': minimo})         == [('detector', config.VERDE)]
    assert v.codigo() == 'VVVVC' and v.escalon() == 1 and v.motivo_pausa() is None
    # corte de EEG: rojo inmediato, con marcador de cambio
    assert v.actualizar(1.0, eeg={**bien_eeg, 'edad_s': 1.4}) == [('eeg', R)]
    assert v.motivo_pausa() == 'eeg' and v.escalon() == 3
    # vuelve: hacen falta 3 s continuos en verde
    v.actualizar(2.0, eeg=bien_eeg)
    assert not v.listo_para_reanudar(4.9) and v.listo_para_reanudar(5.0)
    # canal despegado: dice cual y por que
    v.actualizar(6.0, eeg={**bien_eeg, 'canales': {'C3': 'plano', 'Fz': 'ruidoso'}})
    assert v.motivo_pausa() == 'canal' and 'C3 plano' in v.detalle['eeg'] and 'Fz ruidoso' in v.detalle['eeg']
    v.actualizar(7.0, eeg=bien_eeg)
    # tasa real: solo cuenta el deficit (una rafaga de muestras atrasadas no es una falla)
    for tasa, color in ((400.0, config.VERDE), (215.0, A), (180.0, R), (250.0, config.VERDE)):
        v.actualizar(7.5, eeg={**bien_eeg, 'tasa_hz': tasa})
        assert v.colores['eeg'] == color, (tasa, v.colores['eeg'])
    # ortesis: 1 ACK perdido amarillo, 3 rojo, pico de latencia amarillo, puerto caido rojo
    v.actualizar(8.0, ortesis={**bien_ort, 'acks_perdidos': 1})
    assert v.colores['ortesis'] == A and v.motivo_pausa() is None
    v.actualizar(9.0, ortesis={**bien_ort, 'acks_perdidos': 3})
    assert v.motivo_pausa() == 'ortesis' and v.escalon() == 4
    v.actualizar(10.0, ortesis={**bien_ort, 'latencia_ms': 150.0})
    assert v.colores['ortesis'] == A
    v.actualizar(11.0, ortesis={**bien_ort, 'puerto_ok': False})
    assert v.colores['ortesis'] == R
    v.actualizar(12.0, ortesis=bien_ort)
    # reloj rojo y detector congelado: escalon 2, sin pausa
    v.actualizar(13.0, reloj_ms=60.0)
    assert v.colores['reloj'] == R and v.escalon() == 2 and v.motivo_pausa() is None
    assert not v.listo_para_reanudar(99.0)            # el reloj en rojo bloquea la salida
    v.actualizar(14.0, reloj_ms=5.0, detector={'fiabilidad': 0.2, 'congelado': True, 'epocas': 40})
    assert v.colores['detector'] == R and v.escalon() == 2
    assert v.listo_para_reanudar(17.0)                # el detector no la bloquea
    return 'semaforos, detalle del electrodo, verde continuo y escalones'


@prueba
def semaforo_piloto():
    """Semaforo PILOTO (solo avisa): alfa occipital (PO7/Oz/PO8) contra su linea base de los
    primeros piloto_base_s. Gris mientras mide la linea base; nunca pausa ni cambia el escalon.
    En el gemelo la fatiga sube el alfa y el semaforo la ve."""
    import hardware as hw, cerebro_sintetico as cs
    from salud import Vigilante
    v = Vigilante()
    assert v.colores['piloto'] == config.CALENTANDO
    base = config.SALUD['piloto_base_s']
    for t in np.arange(0.0, base, 1.0):
        assert v.actualizar(t, alfa=10.0) == [] and v.colores['piloto'] == config.CALENTANDO
    cambios = []
    for t, alfa in ((base, 12.0), (base + 1, 20.0), (base + 2, 30.0)):
        cambios += v.actualizar(t, alfa=alfa)
    assert cambios == [('piloto', config.VERDE), ('piloto', config.AMARILLO), ('piloto', config.ROJO)], cambios
    assert 'alfa' in v.detalle['piloto'] and v.motivo_pausa() is None and v.escalon() == 1
    assert len(v.codigo()) == len(config.SUBSISTEMAS) and v.codigo()[-1] == 'R'
    v.actualizar(base + 3, alfa=None)                    # sin lectura no cambia
    assert v.colores['piloto'] == config.ROJO
    # gemelo: con fatiga sube el alfa de PO7/Oz/PO8 (amplitud x2 a los 15 min con fatiga 1)
    r = {}
    for fatiga in (0.0, 1.0):
        cer = cs.Cerebro(cs._args(semilla=2, fatiga=fatiga))
        cer.t0_sesion = 0.0                              # el reloj del banco empieza en 0
        x0, _ = cs._bloque(cer, 0.0, 20.0)
        x1, _ = cs._bloque(cer, 900.0, 20.0)
        r[fatiga] = hw.potencia_alfa(x1, cs.FS) / hw.potencia_alfa(x0, cs.FS)
    assert r[0.0] < config.SALUD['piloto_amarillo'] and r[1.0] > config.SALUD['piloto_rojo'], r
    return (f"gris -> verde -> amarillo -> rojo sin pausar; gemelo a los 15 min: alfa x{r[0.0]:.1f} "
            f"sin fatiga y x{r[1.0]:.1f} con fatiga 1")


@prueba
def retroceso():
    from salud import Retroceso
    r = Retroceso(0.5, 8.0)
    assert [r.siguiente() for _ in range(6)] == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]
    r.reiniciar()
    assert r.siguiente() == 0.5
    return '0.5, 1, 2, 4, 8, 8 y reinicio'


# ------------------------------------------------------------ agente
@prueba
def agente_basico():
    from agente_errp import AgenteErrP, ConfigAgente
    ag = AgenteErrP(np.ones(3), 0.0)
    try:
        ag.actualizar(0.9)
        raise AssertionError('debio exigir decidir() antes')
    except RuntimeError:
        pass
    try:
        ag.decidir(np.ones(4))
        raise AssertionError('debio rechazar phi con forma incorrecta')
    except ValueError:
        pass
    for phi in (np.zeros(3), np.full(3, 5.0), np.full(3, -5.0)):
        d = ag.decidir(phi)
        assert config.PASO_VISIBLE - 1e-9 <= abs(d.delta) <= config.PASO_MAX + 1e-9, d
        ag.actualizar(0.2)
    b = ag.beta
    ag.decidir(np.zeros(3))
    ag.actualizar(0.9, artefacto=True)
    assert ag.beta == b, 'con artefacto no debe aprender'
    est = AgenteErrP(np.ones(3), 0.0, ConfigAgente(modo='estatico'))
    for _ in range(50):
        est.decidir(np.random.randn(3))
        est.actualizar(0.9)
    assert est.beta == 0.0
    return 'paso visible, artefactos, modo estatico'


@prueba
def comparacion_baselines():
    """La tabla y la figura de comparacion con baselines (estudios/comparacion_baselines.py): cifras por sesion
    a partir de filas conocidas, tabla en es y en con las mismas cifras y sin mezclar idiomas, y la figura se dibuja."""
    import sys
    import tempfile
    from pathlib import Path
    sys.path.insert(0, str(config.RAIZ / 'estudios'))
    import comparacion_baselines as cb

    def filas(recupera_en, err=0.3):
        """Una sesion: 30 pasos antes de perturbar y 77 despues; beta sube 0.1 por paso desde `recupera_en` (None: nunca)."""
        out = [{'sujeto': 0, 'rep': 0, 'bloque': 'adaptativo', 'post': False, 't': t, 'erroneo': t % 5 == 0, 'sombra': False,
                'beta': 0.0, 'beta_antes': 0.0} for t in range(30)]
        for t in range(77):
            b = 0.0 if recupera_en is None else max(0.0, (t - recupera_en + 1) * 0.1) + (cb.al.META_BETA if t >= recupera_en else 0.0)
            out.append({'sujeto': 0, 'rep': 0, 'bloque': 'adaptativo', 'post': True, 't': 40 + t, 'erroneo': t % 2 == 0, 'sombra': True,
                        'beta': b, 'beta_antes': 0.0})
        return out
    m = {k: cb.metricas(filas(None if k in ('estatico', 'sin_errp') else 10)) for k in cb.METODOS}
    assert m['estatico']['recuperan'] == 0 and m['estatico']['mediana'] is None
    assert m['bayes']['recuperan'] == 1 and m['bayes']['mediana'] == 11 and abs(m['bayes']['antes'] - 0.2) < 1e-9
    assert abs(m['bayes']['err'] - 29 / 57) < 1e-9 or abs(m['bayes']['err'] - 0.5) < 0.02
    es, n = cb.tabla(m, 'es')
    en, _ = cb.tabla(m, 'en')
    assert n == 1 and len(es) == len(en) == 2 + len(cb.METODOS)
    assert 'Bayes (el nuestro)' in es[4] and 'Bayes (ours)' in en[4] and 'ours' not in ' '.join(es) and 'nuestro' not in ' '.join(en)
    assert es[2].count('\u2014') == 1 and en[2].count('\u2014') == 1 and '1/1' in es[4] and '1/1' in en[4]       # estatico: sin pasos de recuperacion
    # mismas cifras en los dos idiomas
    cifras = lambda lineas: [c.strip() for l in lineas[2:] for c in l.split('|')[2:-1]]
    assert cifras(es) == cifras(en)
    datos = {'actual': {k: filas(None if k in ('estatico', 'sin_errp') else 10) for k in cb.METODOS}}
    with tempfile.TemporaryDirectory() as d:
        for idioma in ('es', 'en'):
            ruta_md, mm = cb.informe_md(datos, idioma, Path(d) / f'{idioma}.md')
            assert ruta_md.read_text(encoding='utf-8').startswith('# ')
            png = cb.figura(mm, idioma, Path(d) / f'{idioma}.png')
            assert png.exists() and png.stat().st_size > 20_000
    return 'tabla y figura en es y en con las mismas cifras'


@prueba
def barrido_paso():
    """Las cifras del barrido de paso (estudios/barrido_paso.py) a partir de filas conocidas: pasos en tope,
    cierre completo por tipo de ensayo, recuperacion, y la tabla marca la configuracion actual."""
    import sys
    import tempfile
    from pathlib import Path
    sys.path.insert(0, str(config.RAIZ / 'estudios'))
    import barrido_paso as bp
    # 2 ensayos de 5 pasos (cerrar y relajar) en el bloque adaptativo; el de cerrar acaba en 1.0, el de relajar en 0.2
    filas = []
    for t in range(10):
        cerrar = t < 5
        ang = [0.5, 0.8, 1.0, 1.0, 1.0][t] if cerrar else [0.5, 0.3, 0.2, 0.2, 0.2][t - 5]
        filas.append({'sujeto': 0, 'rep': 0, 'bloque': 'adaptativo', 'post': False, 't': t, 'meta': 1 if cerrar else -1,
                      'angulo': ang, 'quieto': t in (3, 4, 8, 9), 'erroneo': False, 'sombra': False, 'beta': 0.0, 'beta_antes': 0.0})
    for t in range(10, 80):                      # tras perturbar: beta llega a la meta en el paso 20 despues
        filas.append({'sujeto': 0, 'rep': 0, 'bloque': 'adaptativo', 'post': True, 't': t, 'meta': 1, 'angulo': 0.5, 'quieto': False,
                      'erroneo': t % 2 == 0, 'sombra': True, 'beta': 0.1 * (t - 9), 'beta_antes': 0.0})
    x = bp.medir(filas, 5)
    assert abs(x['quieto'] - 4 / 80) < 1e-9
    # ensayos de cerrar que terminan completos: el del paso 4 (1.0) de 15 (con los 14 de despues de perturbar, a 0.5); relajar: 0 de 1
    assert abs(x['cierra'] - 1 / 15) < 1e-9 and x['abre'] == 0.0 and abs(x['completo'] - 1 / 30) < 1e-9, x
    assert x['recuperan'] == 1 and x['n'] == 1 and x['mediana'] is not None
    m = {celda: x for celda in ((0.15, 3), bp.ACTUAL)}
    lineas = bp.tabla(m)
    assert len(lineas) == 4 and '(actual)' in lineas[3] and '(actual)' not in lineas[2]
    with tempfile.TemporaryDirectory() as d:
        mm = {(p, n): x for p in bp.PASOS for n in bp.POR_ENSAYO}
        png = bp.figura(mm, Path(d) / 'b.png')
        assert png.exists() and png.stat().st_size > 20_000
    return 'pasos en tope, cierre completo y recuperacion por celda; tabla y figura'


@prueba
def gemelo_personal():
    """gemelo_personal.py: los estadisticos de la calibracion real, la inversion por simulacion (con un gemelo de juguete
    de forma cerrada, para que sea rapida y exacta), el informe y la lectura de la sesion real."""
    import tempfile
    from pathlib import Path
    import gemelo_personal as gp
    C3, FZ, CZ, PZ = (gp.IDX[c] for c in ('C3', 'Fz', 'Cz', 'Pz'))
    # estadisticos de datos conocidos
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 20)
    X = rng.normal(size=(40, 8, 200)) * np.arange(1, 9)[None, :, None]               # RMS 1..8 por canal
    X[y == 1, C3] *= 0.8                                                             # cerrar baja la amplitud en C3 a 0.8 (potencia 0.64)
    e = gp.estadisticos_mi(X, y)
    assert np.allclose(e['rms_reposo'], np.arange(1, 9), rtol=0.2) and abs(e['log_cociente_c3'] - np.log(0.64)) < 0.15
    ye = np.tile([0, 1, 0, 0], 30)
    Xe = rng.normal(size=(120, 8, 250)) * 0.5
    bump = np.zeros(250); bump[120:160] = 1.0                                         # cubre la ventana de la Pe (300 a 420 ms tras el movimiento)
    Xe[np.ix_(ye == 1, [FZ, CZ, PZ])] += 6.0 * bump
    Xe[:12, FZ, 20] += 400.0                                                          # 12 epocas con un parpadeo enorme
    s = gp.estadisticos_errp(Xe, ye, 100.0)
    assert abs(s['errp_pe'] - 6.0) < 1.0 and abs(s['frac_artefacto'] - 0.1) < 1e-9, s      # la Pe (300-420 ms) cae dentro del bump de 6 uV
    for malo in (lambda: gp.estadisticos_mi(X[:3], y[:3]), lambda: gp.estadisticos_errp(Xe, np.zeros(120, int), 100.0)):
        try:
            malo(); raise AssertionError('debia rechazar')
        except ValueError:
            pass
    lo, hi = gp.bootstrap(lambda a, b: gp.estadisticos_errp(a, b, 100.0)['errp_pe'], Xe, ye, n=20)
    assert lo < s['errp_pe'] + 2 and hi > s['errp_pe'] - 2
    # la inversion: monotona, y el borde se marca
    v, fuera = gp._invertir(0.5, [0, 1, 2], [0.0, 1.0, 2.0])
    assert abs(v - 0.5) < 1e-9 and not fuera and gp._invertir(5.0, [0, 1, 2], [0.0, 1.0, 2.0]) == (2.0, True)

    def falso(params, semilla, n_mi=0, n_errp=0):
        """Un gemelo de juguete: RMS 6 x ganancia, ERD en C3, ErrP en Fz/Cz/Pz y parpadeos en una fraccion de las epocas."""
        r = np.random.default_rng(semilla)
        mi = er = None
        if n_mi:
            ym = np.tile([0, 1], n_mi // 2)
            Xm = r.normal(size=(n_mi, 8, 100)) * (6 * np.asarray(params['ganancia_canal']))[None, :, None]
            Xm[ym == 1, C3] *= 1 - params['erd']
            mi = (Xm, ym)
        if n_errp:
            yy = np.tile([0, 1, 0, 0], n_errp // 4)
            Xx = r.normal(size=(n_errp, 8, 250)) * 0.5
            Xx[np.ix_(yy == 1, [FZ, CZ, PZ])] += params['errp'] * bump
            Xx[r.random(n_errp) < params['parpadeos'] * 2, FZ, 20] += 400.0
            er = (Xx, yy)
        return mi, er
    verdad = {'erd': 0.30, 'errp': 6.0, 'parpadeos': 0.15, 'ganancia_canal': [1.0, 1.5, 0.8, 1.2, 1.0, 0.9, 1.1, 1.3]}
    mi, er = falso(verdad, 5, n_mi=60, n_errp=160)
    est = {'mi': gp.estadisticos_mi(*mi), 'errp': gp.estadisticos_errp(*er, 100.0)}
    original, gp._gemelo = gp._gemelo, falso
    try:
        params, detalle = gp.ajustar(est['mi'], est['errp'], n_mi=60, n_errp=160, pasadas=2, salida=lambda *a: None)
    finally:
        gp._gemelo = original
    assert abs(params['erd'] - 0.30) < 0.05 and abs(params['errp'] - 6.0) < 1.0 and abs(params['parpadeos'] - 0.15) < 0.08, params
    assert np.allclose(params['ganancia_canal'], verdad['ganancia_canal'], rtol=0.25) and detalle['ganancia_canal']['error_rms_max'] < 0.05
    assert not any(detalle[k]['fuera_de_rejilla'] for k in ('erd', 'errp', 'parpadeos'))
    # un piloto fuera de lo que el gemelo sabe hacer queda marcado
    est_raro = {'mi': est['mi'], 'errp': dict(est['errp'], errp_pe=80.0)}
    gp._gemelo = falso
    try:
        _, det2 = gp.ajustar(est_raro['mi'], est_raro['errp'], n_mi=60, n_errp=160, pasadas=1, salida=lambda *a: None)
    finally:
        gp._gemelo = original
    assert det2['errp']['fuera_de_rejilla']
    # el informe y la sesion real al lado
    pred = {k: {'calibracion': {'mi_ba': 0.8, 'sens': 0.7, 'espec': 0.9, 'ba': 0.8}, 'sujetos': 4,
                'estatico': {'antes': 0.2, 'err': 0.49, 'err_ee': 0.01, 'recuperan': 0, 'n': 16, 'mediana': None},
                'bayes': {'antes': 0.15, 'err': 0.37, 'err_ee': 0.02, 'recuperan': 14, 'n': 16, 'mediana': 28.0}} for k in ('personal', 'estandar')}
    with tempfile.TemporaryDirectory() as d:
        orq, _ = _sesion_sham(3, 'sham-real')
        real = gp.de_la_sesion(orq.ruta_csv)
        assert real['error_agente'] is not None and isinstance(real['recuperacion'], list)
        txt = gp.informe(params, detalle, est, pred, real, Path(d) / 'g.md')
        assert (Path(d) / 'g.md').exists() and 'predicción' in txt and 'no una medición' in txt and 'La sesión real, al lado' in txt
        assert 'FUERA' not in txt and f"{params['erd']:.2f}" in txt
    return f"recupera erd {params['erd']:.2f} (0.30), errp {params['errp']:.1f} (6.0), parpadeos {params['parpadeos']:.2f} (0.15) con un gemelo de juguete; marca lo fuera de rejilla"


@prueba
def ia_sesion():
    """ia_sesion.py: las 6 preguntas, el co-investigador y el informe sobre una sesion grabada (simulador, sin API) y la
    auditoria de cifras: lo que sale de las herramientas se verifica, una cifra inventada y un paso que no existe se marcan."""
    import tempfile
    from pathlib import Path
    import copiloto
    import ia_sesion as isn
    orq, _ = _sesion_sham(3, 'sham-real')
    o = isn.auditar(orq.ruta_csv, None)
    assert len(o['preguntas']) == 6 and all(q['respuesta'] and q['origen'] == 'reglas' for q in o['preguntas'])
    assert o['api'] is False and o['coinvestigador']['valida'] and o['coinvestigador']['origen'] == 'reglas'
    assert o['coinvestigador']['coincide_con_reglas'] is True                       # sin API, la propuesta ES la de las reglas
    # lo que dice el copiloto se verifica contra la sesion: ninguna respuesta cita pasos que no existen
    for q in o['preguntas']:
        a = q['auditoria']
        assert not a['pasos_inexistentes'], q
        assert q['sin_dato'] or a['cifras'] == 0 or len(a['verificadas']) >= 0.8 * a['cifras'], (q['pregunta'], a)
    assert sum(len(q['auditoria']['verificadas']) for q in o['preguntas']) >= 10
    # la auditoria detecta lo inventado
    s = copiloto.Sesion(Path(orq.ruta_csv))
    mala = isn.auditar_texto('Se recupero en 999 pasos, en el paso 4000, con un error de 0.123.', s)
    assert '999' in mala['no_encontradas'] and '0.123' in mala['no_encontradas'] and mala['pasos_inexistentes'] == [4000], mala
    err = s.metrica('error', 'tras_perturbacion')
    bien = isn.auditar_texto(f"El error del agente fue {err['error_agente']:.2f} y el de la sombra {err['error_sombra']:.2f} (pasos {err['pasos'][0]} a {err['pasos'][1]}).", s)
    assert not bien['no_encontradas'] and not bien['pasos_inexistentes'], bien
    md = isn.a_markdown(o)
    assert md.count('###') == 6 and 'sin API' in md and 'Co-investigador' in md and 'pendientes de una persona' in md
    return f"6 preguntas auditadas ({sum(len(q['auditoria']['verificadas']) for q in o['preguntas'])} cifras verificadas), propuesta valida, cifras inventadas detectadas"
def reporte_detector():
    """Reporte del detector al final de CAL_ERRP: cifras exactas con datos conocidos, la figura en es y en, el detector
    guarda sus probabilidades de validacion cruzada, y un fallo del reporte nunca lanza."""
    import tempfile
    from pathlib import Path
    import cerebro_sintetico as cs
    import hardware as hw
    import reporte_detector as rd
    # 10 errores y 20 aciertos con puntajes conocidos; umbral 0.5
    y = np.r_[np.ones(10, int), np.zeros(20, int)]
    p = np.r_[np.linspace(0.45, 0.95, 10), np.linspace(0.05, 0.60, 20)]
    r = rd.calcular(p, y, 0.5, None)
    esperado = {'tp': 9, 'fn': 1, 'fp': int((p[10:] > 0.5).sum())}
    assert (r['tp'], r['fn'], r['fp']) == (esperado['tp'], esperado['fn'], esperado['fp']), r
    assert r['tn'] == 20 - r['fp'] and abs(r['sens'] - 0.9) < 1e-9 and abs(r['espec'] - r['tn'] / 20) < 1e-9
    assert abs(r['ba'] - 0.5 * (r['sens'] + r['espec'])) < 1e-9 and 0.8 < r['auc'] < 1.0 and 0 <= r['ece'] <= 1
    mp, fr, nc = r['confiabilidad']
    assert nc.sum() == 30 and np.all(np.diff(mp) >= 0) and np.all((fr >= 0) & (fr <= 1))        # casillas con las 30 epocas, de menor a mayor
    # con las decisiones de la validacion anidada (las que da el CP3), esas mandan en sens / espec / falsos positivos
    anid = np.zeros(30, int); anid[:5] = 1; anid[10] = 1
    r2 = rd.calcular(p, y, 0.5, anid)
    assert (r2['tp'], r2['fn'], r2['fp'], r2['tn']) == (5, 5, 1, 19)
    for malo in ((p, np.ones(30, int)), (p[:5], y)):
        try:
            rd.calcular(malo[0], malo[1], 0.5, None)
            raise AssertionError('debia rechazar')
        except ValueError:
            pass
    # un detector de verdad: guarda p_cv y la figura se dibuja en los dos idiomas
    X, ye = cs.sesion_errp(60, semilla=3)
    det = hw.DetectorErrP().ajustar(X, ye, config.candidatos('detector'))
    assert det.p_cv is not None and len(det.p_cv) == len(ye) and 0 <= det.p_cv.min() and det.p_cv.max() <= 1
    with tempfile.TemporaryDirectory() as d:
        ruta, res = rd.desde_detector(det, Path(d) / 'det.png')
        assert ruta is not None and ruta.stat().st_size > 30_000 and abs(res['umbral'] - det.umbral) < 1e-12
        assert abs(res['sens'] - det.sens) < 1e-9 and abs(res['espec'] - det.espec) < 1e-9       # las del CP3, no las de p_cv
        assert rd.figura(res, Path(d) / 'en.png', 'en', det.eleccion).stat().st_size > 30_000
        # nunca lanza: sin probabilidades, o con un detector roto
        det2 = hw.DetectorErrP()
        ruta2, motivo = rd.desde_detector(det2, Path(d) / 'no.png')
        assert ruta2 is None and isinstance(motivo, str) and not (Path(d) / 'no.png').exists()
        assert rd.desde_detector(None, Path(d) / 'no.png')[0] is None
        # el gancho de CAL_ERRP: la bandera existe y el metodo del backend dibuja y avisa sin lanzar
        import orquestador
        assert orquestador.argumentos(['real']).sin_reporte_detector is False
        assert orquestador.argumentos(['real', '--sin-reporte-detector']).sin_reporte_detector is True
        viejo, config.RESULTADOS = config.RESULTADOS, Path(d)
        try:
            orquestador.BackendReal.reporte_detector(type('B', (), {'detector': det})(), 123)
            assert (Path(d) / 'detector_errp_123.png').exists()
            orquestador.BackendReal.reporte_detector(type('B', (), {'detector': hw.DetectorErrP()})(), 124)    # sin p_cv: avisa, no lanza
            assert not (Path(d) / 'detector_errp_124.png').exists()
        finally:
            config.RESULTADOS = viejo
    return f"umbral {det.umbral:.2f}, sens {det.sens:.2f}, espec {det.espec:.2f}: figura de confiabilidad, ROC y umbral (es y en)"


@prueba
def prior_por_paso():
    """Prior de error por paso (encendido por defecto con epsilon 0.10): el error que el agente predice, con un piso."""
    from agente_errp import AgenteErrP, ConfigAgente
    assert config.PRIOR_POR_PASO is True and config.PISO_PRIOR_PASO == 0.10                # el defecto medido el 4 de octubre
    assert ConfigAgente().prior_por_paso is True and ConfigAgente().piso_prior == 0.10
    assert ConfigAgente(prior_por_paso=False).prior_por_paso is False                     # y se puede apagar
    kw = dict(salida_detector='calibrada', p_error_calibracion=0.3)
    phi = np.array([6.0])                                    # decoder muy seguro: p' ~ 0.998, error predicho ~ 0.002

    def p_hat(**c):
        ag = AgenteErrP([1.0], 0.0, ConfigAgente(**kw, **c))
        d = ag.decidir(phi)
        return ag.actualizar(0.9, False, 1.0)['P_hat'], d.p_prima, ag
    base, p_prima, _ = p_hat(prior_por_paso=False)
    assert p_prima > 0.99
    for piso, esperado in ((0.0, 0.0), (0.05, 0.05), (0.10, 0.10), (None, 0.2)):         # None: la tasa global (prior 0.2)
        ph, _, ag = p_hat(prior_por_paso=True, piso_prior=piso)
        prior = max(1 - max(p_prima, 1 - p_prima), esperado)
        assert abs(ag.prior_del_paso() - ag.prior) < 1e-12                               # sin paso pendiente: el global
        llr = np.log(0.9 / 0.1) - np.log(0.3 / 0.7)
        previsto = float(1 / (1 + np.exp(-(np.log(max(prior, 1e-6) / (1 - max(prior, 1e-6))) + llr))))
        assert abs(ph - previsto) < 1e-6, (piso, ph, previsto)
        assert ph <= base + 1e-12                                                                 # seguro y con prior bajo: duda menos del paso
    # con un paso dudoso (p' ~ 0.5) el error predicho (0.5) pasa por encima del piso
    ag = AgenteErrP([1.0], 0.0, ConfigAgente(**kw, prior_por_paso=True, piso_prior=0.05))
    d = ag.decidir(np.array([0.0]))
    assert abs(ag.prior_del_paso(d) - 0.5) < 1e-9
    for malo in (-0.1, 0.5):
        try:
            ConfigAgente(prior_por_paso=True, piso_prior=malo)
            raise AssertionError('debio rechazar el piso')
        except ValueError:
            pass
    # velocidad de recuperacion (estudios/prior_por_paso.py): no recuperada = infinito, y el p90 lo dice
    sys.path.insert(0, str(config.RAIZ / 'estudios'))
    import prior_por_paso as pp
    series = {0: [0.5, 1.0, 1.7, 2.0, 2.0], 1: [0.1, 0.5, 1.0, 1.5, 1.7], 2: [0.1] * 5}     # recupera en 3, en 5 y nunca (meta 1.68)
    filas = [{'sujeto': s, 'rep': 0, 'bloque': 'real', 'primero': True, 'post': True, 'beta_pre': 0.0, 'beta': b, 'erroneo': False, 'sombra': False}
             for s, bs in series.items() for b in bs]
    v = pp.velocidad(filas)
    assert v['n'] == 3 and v['recuperadas'] == 2 and v['mediana'] == 5 and v['p90'] == np.inf, v
    return 'encendido por defecto (epsilon 0.10); con epsilon 0, 0.05, 0.10 y tasa global el prior de cada paso es el error predicho con piso'


@prueba
def p_hat_refleja_errp():
    """P_hat dice lo que vio el detector aunque el agente no este aprendiendo (bloque estatico,
    aprendizaje congelado, reloj en ROJO). Antes, con salida calibrada, quedaba igual al prior."""
    import orquestador
    from agente_errp import AgenteErrP, ConfigAgente
    ag = AgenteErrP(np.ones(3), 0.0, ConfigAgente(salida_detector='calibrada', p_error_calibracion=0.3))
    prior = ag.prior
    ag.decidir(np.zeros(3))
    alto = ag.actualizar(0.95, False, fiabilidad=1.0, peso=0.0)          # el detector vio un error claro
    assert alto['P_hat'] > prior + 0.3 and ag.beta == 0.0 and not alto['usada'], alto
    ag.decidir(np.zeros(3))
    bajo = ag.actualizar(0.05, False, fiabilidad=1.0, peso=0.0)
    assert bajo['P_hat'] < prior - 0.05 and ag.beta == 0.0, bajo
    ag.decidir(np.zeros(3))
    dudoso = ag.actualizar(0.95, False, fiabilidad=0.4, peso=0.0)        # detector poco fiable: se le cree menos
    assert ag.prior < dudoso['P_hat'] < alto['P_hat'], dudoso
    ag.decidir(np.zeros(3))
    assert ag.actualizar(0.95, False, fiabilidad=1.0)['usada'] and ag.beta != 0.0    # sin peso: aprende como siempre

    # en el orquestador: bloque estatico con salida calibrada
    class Calibrada(orquestador.BackendSim):
        def preparar(self, orq):
            return dict(super().preparar(orq), salida='calibrada')
    a = orquestador.argumentos(['sim', '--ciclo', '0', '--pasos_estatico', '90', '--pasos_adaptativo', '10',
                                '--sin_perturbacion'])       # ~1 de cada 3 pasos no mueve la ortesis: sin P_hat
    orq = orquestador.Orquestador(Calibrada(a), a)
    orquestador.correr(orq, a)
    est = [f for f in orq.filas if f['estado'] == 'LAZO_ESTATICO' and f['P_hat'] != '']
    ph = np.array([f['P_hat'] for f in est]); err = np.array([f['error_verdadero'] for f in est])
    assert len(est) > 40 and ph.std() > 0.1, ph.std()
    assert ph[err == 1].mean() > ph[err == 0].mean() + 0.08, (ph[err == 1].mean(), ph[err == 0].mean())
    assert len({f['beta'] for f in orq.filas if f['estado'] == 'LAZO_ESTATICO'}) == 1      # y aun asi no aprende
    return (f'bloque estatico: P_hat {ph[err == 1].mean():.2f} en errores y {ph[err == 0].mean():.2f} en aciertos, '
            f'sin aprender')


@prueba
def agente_aprende():
    import simulador_lazo as S
    M = {m: np.array([S.metricas(S.correr(m, s), 300) for s in range(12)]).mean(0)
         for m in ('estatico', 'fijo', 'bayes')}
    assert M['bayes'][2] < M['estatico'][2] - 0.08, M
    assert M['bayes'][1] <= M['fijo'][1] + 0.01, M
    return (f"tras perturbar: estatico {M['estatico'][2]:.2f}, fijo {M['fijo'][2]:.2f}, "
            f"bayes {M['bayes'][2]:.2f}; primeros 2 min fijo {M['fijo'][1]:.2f} vs bayes {M['bayes'][1]:.2f}")


@prueba
def agente_sin_sesgo():
    """Con metas no balanceadas se apaga el detector de sesgo: el chequeo predictivo
    debe seguir recuperando mejor que no aprender."""
    import simulador_lazo as S
    M = {m: np.array([S.metricas(S.correr(m, s, usar_sesgo=False), 300) for s in range(12)]).mean(0)
         for m in ('estatico', 'bayes')}
    assert M['bayes'][2] < M['estatico'][2] - 0.08, M
    return f"tras perturbar: estatico {M['estatico'][2]:.2f}, bayes sin sesgo {M['bayes'][2]:.2f}"


@prueba
def confianza_detector():
    import simulador_lazo as S
    dentro, fuera, fiab = [], [], []
    for s in range(12):
        r = S.correr('bayes', s, falla=(420, 480))
        dentro.append((r['fiab'][440:480] == 0).mean())
        fiab.append(r['fiab'][440:480].mean())
        fuera.append((r['fiab'][:400] == 0).mean())
    assert np.mean(fiab) < 0.4 and np.mean(fuera) < 0.05, (np.mean(fiab), np.mean(fuera))
    return (f'en la falla: congelado {np.mean(dentro):.0%}, aprende al {np.mean(fiab):.0%}; '
            f'congelado en falso {np.mean(fuera):.1%}')


# ------------------------------------------------------------ orquestador
@prueba
def maquina_estados():
    from orquestador import MaquinaEstados

    class Nula:
        def marcador(self, *a):
            pass
    m = MaquinaEstados(Nula())
    m.ir_a('CAL_MI')
    try:
        m.ir_a('LAZO_ADAPTATIVO')
        raise AssertionError('debio rechazar CAL_MI -> LAZO_ADAPTATIVO')
    except ValueError:
        pass
    return 'rechaza transiciones invalidas'


@prueba
def orquestador_sim():
    import orquestador
    a = orquestador.argumentos(['sim', '--ciclo', '0', '--pasos_estatico', '20', '--pasos_adaptativo', '90'])
    orq = orquestador.Orquestador(orquestador.BackendSim(a), a)
    orquestador.correr(orq, a)                     # (no se busca "el CSV nuevo": puede haber otras sesiones a la vez)
    filas = list(csv.DictReader(open(orq.ruta_csv)))
    assert len(filas) == 110 and list(filas[0]) == config.COLUMNAS_CSV
    assert {'LAZO_ESTATICO', 'LAZO_ADAPTATIVO'} <= {f['estado'] for f in filas}
    return f'{len(filas)} filas con el esquema del contrato'


def _backend_con_fallas(orquestador, a, fallas, cortar_en_tic=None):
    """BackendSim con fallas programadas a mano y un tope de tics, para que una pausa
    que no termina sea una falla de la prueba y no un cuelgue."""
    class ConFallas(orquestador.BackendSim):
        tics = 0

        def esperar(self, dt):
            self.tics += 1
            if cortar_en_tic is not None and self.tics >= cortar_en_tic:
                raise KeyboardInterrupt
            assert self.tics < 5000, 'la pausa segura no termina'
            super().esperar(dt)
    b = ConFallas(a)
    b.fallas = fallas
    return b


@prueba
def pausa_segura():
    import orquestador
    a = orquestador.argumentos(['sim', '--ciclo', '0', '--pasos_estatico', '20', '--pasos_adaptativo', '60',
                                '--sin_perturbacion', '--ajenos-cada', '0'])   # fallas por seq de pasos propios
    # corte de EEG de 4 s que empieza a media epoca (t=59 s); el movimiento a posicion segura
    # de esa pausa (seq 30) sale con pico de latencia; C3 plano 5 s en t=100 s; 3 ACK perdidos
    b = _backend_con_fallas(orquestador, a, {
        'corte_eeg': [(59.0, 4.0)], 'canal': [(100.0, 5.0, 'C3', 'plano')],
        'ack_perdido': {70, 71, 72}, 'pico_latencia': {10: 150.0, 30: 200.0}})
    orq = orquestador.Orquestador(b, a)
    orquestador.correr(orq, a)
    filas = orq.filas
    assert list(filas[0]) == config.COLUMNAS_CSV
    pausas = [f for f in filas if f['estado'] == 'PAUSA_SEGURA']
    assert [f['excluido'] for f in pausas] == ['pausa:eeg', 'pausa:canal', 'pausa:ortesis'], pausas
    lazo = [f for f in filas if f['estado'] != 'PAUSA_SEGURA']
    assert len(lazo) == 80                                   # las pausas no consumen pasos
    motivos = [f['excluido'] for f in lazo if f['excluido']]
    assert motivos == ['epoca_invalida', 'sin_ack', 'sin_ack', 'sin_ack'], motivos
    # el agente no aprende en pausa ni en pasos excluidos: beta igual a la fila anterior
    for i, f in enumerate(filas):
        if f['excluido'] and i:
            assert f['beta'] == filas[i - 1]['beta'], (i, f)
        assert len(f['salud']) == len(config.SUBSISTEMAS) and set(f['salud']) <= set('VARC'), f
    # el detector calienta sus primeras 15 epocas validas sin emitir marcadores de salud; un paso
    # que no movio la ortesis no tiene epoca y no cuenta
    con_epoca = [not f['excluido'] and f['artefacto'] == 0 for f in filas]
    assert not all(con_epoca[:30]) and filas[0]['salud'][3] == 'C'
    for i, f in enumerate(filas[:40]):
        assert (f['salud'][3] == 'C') == (sum(con_epoca[:i]) < 15), (i, f['salud'], sum(con_epoca[:i]))
    i_det = next(i for i, m in enumerate(orq.salidas.marcadores) if m.startswith('salud:detector:'))
    assert sum(m.startswith('paso_ack:') for m in orq.salidas.marcadores[:i_det]) >= 15
    assert orq.fsm.estado == 'EVALUACION'
    estados = [e for _, e in orq.fsm.historial]
    for i, e in enumerate(estados):
        if e == 'PAUSA_SEGURA':
            assert estados[i - 1] == estados[i + 1], estados   # regresa al estado previo
    marc = orq.salidas.marcadores
    for m in ('salud:eeg:ROJO', 'salud:eeg:VERDE', 'salud:ortesis:AMARILLO', 'salud:ortesis:ROJO',
              'bloque:PAUSA_SEGURA'):
        assert m in marc, m
    assert 'C3 plano' in ' '.join(orq.avisos_salud)          # dice que electrodo y por que
    assert orq.excluidos == {'pausa:eeg': 1, 'pausa:canal': 1, 'pausa:ortesis': 1,
                             'epoca_invalida': 1, 'sin_ack': 3}, orq.excluidos
    # Ctrl+C dentro de una pausa que no termina: llega a EVALUACION y cierra el CSV
    b2 = _backend_con_fallas(orquestador, a, {'corte_eeg': [(10.0, 1e9)]}, cortar_en_tic=3)
    orq2 = orquestador.Orquestador(b2, a)
    orquestador.correr(orq2, a)
    assert orq2.fsm.estado == 'EVALUACION' and orq2.f_csv.closed
    assert orq2.filas[-1]['excluido'] == 'pausa:eeg'
    return f'3 pausas (eeg, canal, ortesis) con regreso al estado previo; excluidos {orq.excluidos}'


@prueba
def cp1_robusto():
    """CP1 de la ortesis con 40 movimientos y tres metricas robustas (MAD, p95 y ACK perdidos):
    un pico aislado no lo tumba, un jitter tipico alto si. Y el caos actua solo desde el lazo."""
    import types
    import orquestador
    import hardware as hw
    import cerebro_sintetico as cs
    rng = np.random.default_rng(0)
    assert config.CP1_MOVIMIENTOS == 40
    normal = list(rng.normal(8, 1.5, 39))
    ok = hw.evaluar_latencias(normal + [300.0])               # 39 normales y un pico de 300 ms
    assert ok['ok'], ok
    malo = hw.evaluar_latencias(list(rng.normal(60, 40, 40)))   # jitter tipico de 40 ms
    assert not malo['ok'] and malo['mad_ms'] > config.CP1_MAD_MAX_MS, malo
    perdidos = hw.evaluar_latencias(normal[:35] + [float('nan')] * 5)   # 12.5 % de ACK perdidos
    assert not perdidos['ok'] and abs(perdidos['perdidos'] - 0.125) < 1e-9, perdidos
    assert 'MAD' in ok['texto'] and 'p95' in ok['texto']
    # caos desde el lazo (por defecto) o desde la calibracion
    assert orquestador.argumentos(['real']).caos_desde == 'lazo'
    b = orquestador.BackendReal.__new__(orquestador.BackendReal)
    b.a = types.SimpleNamespace(caos=1, caos_nivel='estandar', caos_desde='lazo')
    b.ortesis, b.hw = hw.OrtesisSimulada(), hw
    b.activar_caos('calibracion')
    assert b.ortesis.caos is None
    b.activar_caos('lazo')
    assert b.ortesis.caos is not None
    cer = cs.Cerebro(cs._args(caos=1, caos_desde='lazo'))
    assert cer.t_caos(1e6) is None                            # sin la senal del lazo no hay caos
    cer._marcador(config.m_bloque('LAZO_ESTATICO'), 500.0)
    assert cer.t_caos(530.0) == 30.0
    desde_ya = cs.Cerebro(cs._args(caos=1, caos_desde='calibracion'))
    assert desde_ya.t_caos(desde_ya.t0_sesion + 7) == 7
    return f"pico aislado: GO ({ok['texto']}); jitter de 40 ms: NO GO; el caos espera al lazo"


@prueba
def calibracion_repeticiones():
    import orquestador
    b = orquestador.BackendReal.__new__(orquestador.BackendReal)   # sin hardware
    intentos = []
    assert b._ensayo_con_reintentos(lambda: intentos.append(1), 'ensayo') is None
    assert len(intentos) == 1 + config.CAL_REPETICIONES_MAX
    cuenta = iter([None, None, 'dato'])
    assert b._ensayo_con_reintentos(lambda: next(cuenta), 'ensayo') == 'dato'
    return f'maximo {config.CAL_REPETICIONES_MAX} repeticiones por ensayo y aviso'


# ------------------------------------------------------------ persistencia
@prueba
def repetir_sesion():
    """Plan B: cada sesion deja grabado, junto a su CSV, todo lo que publico en el flujo Estado
    (con su hora). repetir_sesion.py lo vuelve a publicar: el tablero muestra la sesion igual que
    en vivo y, si se quiere, la ortesis repite los mismos angulos. Sin casco ni calibracion."""
    import json
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from pyqtgraph.Qt import QtWidgets
    import orquestador
    import repetir_sesion as rs
    import tablero
    a = orquestador.argumentos(['sim', '--ciclo', '0', '--semilla', '5', '--pasos_estatico', '20',
                                '--pasos_adaptativo', '60'])
    orq = orquestador.Orquestador(orquestador.BackendSim(a), a)
    orquestador.correr(orq, a)
    ruta = orq.ruta_csv.with_name(orq.ruta_csv.stem + config.SUFIJO_ESTADO)
    grabados = rs.leer(ruta)
    tipos = [e['tipo'] for _, e in grabados]
    assert tipos.count('paso') + tipos.count('ajeno') == len(orq.filas) == 80 and tipos.count('ajeno') == 6, tipos
    assert tipos.count('cue') == 16 and 'checkpoint' in tipos and 'iic' in tipos
    assert all(t2 >= t1 for (t1, _), (t2, _) in zip(grabados, grabados[1:]))
    # la repeticion publica lo mismo y en el mismo orden; la ortesis va a los mismos angulos
    publicados, angulos, esperas = [], [], []

    class Ortesis:
        def mover(self, fraccion, dur_ms=None):
            angulos.append(fraccion)
            return 1, 0.0, 1.0
    n = rs.repetir(ruta, velocidad=2.0, completa=True, publicar=publicados.append, ortesis=Ortesis(),
                   dormir=esperas.append, salida=lambda *_: None)
    assert n == len(grabados) and publicados == [e for _, e in grabados]
    assert angulos[:-1] == [e['angulo'] for _, e in grabados if e['tipo'] in ('paso', 'ajeno')]
    assert angulos[-1] == config.POSICION_SEGURA           # al terminar, la ortesis queda abierta
    assert all(0 <= d <= rs.ESPERA_MAX_S for d in esperas)
    assert sum(esperas) <= (grabados[-1][0] - grabados[0][0]) / 2.0 + 1e-6
    # por defecto empieza en el lazo: se salta la calibracion, pero conserva sus checkpoints
    corto = []
    rs.repetir(ruta, velocidad=0, publicar=corto.append, salida=lambda *_: None)
    assert len(corto) <= len(publicados) and [e['tipo'] for e in corto].count('paso') == tipos.count('paso')
    # el tablero la muestra completa sin errores
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    t = tablero.Tablero()
    try:
        t.timer.stop()
        for e in publicados:
            t._procesar(e)
        t._dibujar()
        assert 'paso 80' in t.lbl_estado.text(), t.lbl_estado.text()
    finally:
        t.close()
    assert rs.ultima('sim').name == ruta.name
    return f'{n} eventos grabados y repetidos ({tipos.count("paso")} pasos, {tipos.count("ajeno")} ajenos); el tablero y la ortesis los siguen'


@prueba
def plan_b_sin_sesion_detenida():
    """repetir_sesion.tiene_pasos / sesiones / ultima, sin Qt ni LSL: el plan B nunca es una sesion que un NO GO detuvo
    antes del lazo (solo checkpoint y detenida), un archivo vacio o roto. Cuenta tanto 'paso' como 'ajeno', separa los
    backends 'real' y 'sim' y lee config.RESULTADOS al llamarla (no al importar)."""
    import json
    import tempfile
    from pathlib import Path
    import detencion
    import repetir_sesion as rs
    suf = config.SUFIJO_ESTADO
    with tempfile.TemporaryDirectory() as d:
        carpeta = Path(d)

        def sesion(nombre, tipos, mtime, **extra):
            return _grabar_estado_falso(carpeta / (nombre + suf), tipos, mtime=mtime, **extra)
        ajena = sesion('sesion_real_E', ['cue', 'ajeno'], 500)                       # solo un movimiento ajeno: llego al lazo
        buena = sesion('sesion_real_A', ['checkpoint', 'cue', 'paso', 'paso'], 1000)
        detenida = sesion('sesion_real_B', ['checkpoint', 'detenida'], 2000, detenida=detencion.evento(3, 'ErrP: BA 0.60', {'ba': 0.6}))
        vacia = carpeta / ('sesion_real_C' + suf)
        vacia.write_text('', encoding='utf-8')
        os.utime(vacia, (3000, 3000))
        rota = carpeta / ('sesion_real_D' + suf)                                      # la linea de paso quedo a medias
        rota.write_text('{"t": 1.0, "evento": {"tipo": "paso", \n', encoding='utf-8')
        os.utime(rota, (4000, 4000))
        no_json = carpeta / ('sesion_real_H' + suf)
        no_json.write_text('esto no es json\n\n', encoding='utf-8')
        os.utime(no_json, (4100, 4100))
        # lineas que dicen "paso" sin ser un paso: un checkpoint con esa clave y un marcador llamado asi
        engano = carpeta / ('sesion_real_I' + suf)
        engano.write_text(json.dumps({'t': 1.0, 'evento': {'tipo': 'checkpoint', 'paso': 3, 'ajeno': 1}}) + '\n'
                          + json.dumps({'t': 2.0, 'marcador': 'paso'}) + '\n', encoding='utf-8')
        os.utime(engano, (4200, 4200))
        sim = sesion('sesion_sim_F', ['checkpoint', 'paso'], 5000)
        (carpeta / 'sesion_real_G.csv').write_text('a,b\n1,2\n', encoding='utf-8')   # otro archivo de la sesion: no cuenta
        # una escritura interrumpida despues de un paso valido: la sesion ya llego al lazo
        cortada = carpeta / ('sesion_real_J' + suf)
        cortada.write_text(json.dumps({'t': 1.0, 'evento': {'tipo': 'paso', 'paso': 1}}) + '\n{"t": 2.0, "evento": {"tipo": "pa', encoding='utf-8')
        os.utime(cortada, (600, 600))

        # tiene_pasos: paso y ajeno cuentan; checkpoint y detenida, vacio, roto, un archivo que no existe, no
        assert rs.tiene_pasos(buena) and rs.tiene_pasos(ajena) and rs.tiene_pasos(sim) and rs.tiene_pasos(cortada)
        for no in (detenida, vacia, rota, no_json, engano, carpeta / 'no_existe_estado.jsonl'):
            assert not rs.tiene_pasos(no), no.name
        # sesiones: solo las que llegaron al lazo, de la mas vieja a la mas reciente, y cada backend por separado
        assert rs.sesiones('real', carpeta) == [ajena, cortada, buena], [r.name for r in rs.sesiones('real', carpeta)]
        assert rs.sesiones('sim', carpeta) == [sim] and rs.sesiones('otro', carpeta) == []
        assert rs.sesiones('real', carpeta / 'no_existe') == []
        # ultima(): la buena aunque la detenida (y los archivos vacios y rotos) sean mas nuevos; lee config.RESULTADOS al llamarla
        anterior = config.RESULTADOS
        config.RESULTADOS = carpeta
        try:
            assert rs.ultima('real') == buena and rs.ultima('sim') == sim and rs.ultima() == buena
            os.utime(buena, (550, 550))                                               # la buena deja de ser la ultima: gana la otra mas nueva
            assert rs.ultima('real') == cortada
            os.utime(buena, (1000, 1000))
            # solo detenidas, vacias o rotas: no hay ultima (None), no una sesion que no sirve
            for nombre in (buena, ajena, cortada):
                nombre.unlink()
            assert rs.ultima('real') is None and rs.sesiones('real') == [] and rs.ultima('sim') == sim
        finally:
            config.RESULTADOS = anterior
        assert config.RESULTADOS == anterior
    return 'detenida, vacia, rota y enganos no cuentan; paso y ajeno si; real y sim separados; ultima() usa config.RESULTADOS'


@prueba
def modelos_del_dia():
    """Dos protecciones para el dia de la demo. (1) Al cargar modelos ya calibrados se dice hace
    cuanto se calibraron y, si son viejos, se avisa: en modelos/ pueden haber quedado los del
    gemelo o los de otro piloto. (2) --solo-errp repite solo la calibracion de ErrP con el decoder
    de MI ya calibrado (si falla el CP3 no hay que repetir los 5 minutos de MI)."""
    import os
    import tempfile
    import types
    from pathlib import Path
    import orquestador
    carpeta, original = Path(tempfile.mkdtemp()), config.MODELOS
    config.MODELOS = carpeta
    try:
        assert orquestador.edad_modelos_h() is None                 # no hay modelos
        for nombre in ('decoder_im.pkl', 'detector_errp.pkl'):
            (carpeta / nombre).write_bytes(b'x')
        assert orquestador.edad_modelos_h() < 0.01
        viejo = time.time() - 30 * 3600
        os.utime(carpeta / 'decoder_im.pkl', (viejo, viejo))        # cuenta el mas viejo de los dos
        assert 29.9 < orquestador.edad_modelos_h() < 30.1
        assert 'OJO' in orquestador.texto_edad_modelos() and '30' in orquestador.texto_edad_modelos()
        os.utime(carpeta / 'decoder_im.pkl', None)
        assert 'OJO' not in orquestador.texto_edad_modelos()
    finally:
        config.MODELOS = original
    # --solo-errp: carga el decoder, no calibra MI y si calibra ErrP
    a = orquestador.argumentos(['real', '--solo-errp'])
    assert a.solo_errp and not orquestador.argumentos(['real']).solo_errp
    llamadas = []
    b = orquestador.BackendReal.__new__(orquestador.BackendReal)
    b.a = a
    b.hw = types.SimpleNamespace(cargar=lambda n: llamadas.append(('cargar', n)) or types.SimpleNamespace(
        ba=0.8, w0=np.ones(3), c0=0.0))
    b.revisar = lambda orq: True
    b.calibrar_mi = lambda orq: llamadas.append('mi') or True

    def calibrar_errp(orq):
        llamadas.append('errp')
        b.detector = types.SimpleNamespace(sens=0.7, espec=0.9, p_error_cal=0.3, umbral=0.4)
        return True
    b.calibrar_errp = calibrar_errp
    b.preparar_coadaptacion = lambda orq: llamadas.append('coadapta')
    estados = []
    orq = types.SimpleNamespace(fsm=types.SimpleNamespace(ir_a=estados.append))
    p = b.preparar(orq)
    assert llamadas == [('cargar', 'decoder_im.pkl'), 'errp', 'coadapta'], llamadas
    assert estados == ['CAL_MI', 'CAL_ERRP'] and p['umbral'] == 0.4 and p['salida'] == 'calibrada'
    return 'edad de los modelos con aviso si son viejos; --solo-errp repite solo la calibracion de ErrP'


@prueba
def instantanea_estado():
    """El estado del agente y del ConfianzaDetector sobrevive a un viaje por JSON (la
    trayectoria futura es identica), y la instantanea se escribe de forma atomica."""
    import json
    import orquestador
    from agente_errp import AgenteErrP, ConfianzaDetector

    def avanzar(ag, co, rng, n):
        betas = []
        for _ in range(n):
            ag.decidir(rng.normal(size=3))
            err = rng.random() < 0.3
            det = rng.random() < (0.7 if err else 0.1)
            fiab = co(err, det, True)
            betas.append(ag.actualizar(0.9 if det else 0.1, False, fiab, *co.vivo())['beta'])
        return betas
    rng = np.random.default_rng(0)
    ag, co = AgenteErrP(np.ones(3)), ConfianzaDetector()
    avanzar(ag, co, rng, 80)
    copia = json.loads(json.dumps({'a': ag.a_dict(), 'c': co.a_dict(), 'rng': rng.bit_generator.state}))
    ag2, co2, rng2 = AgenteErrP(np.ones(3)), ConfianzaDetector(), np.random.default_rng()
    ag2.desde_dict(copia['a']); co2.desde_dict(copia['c']); rng2.bit_generator.state = copia['rng']
    assert avanzar(ag, co, rng, 80) == avanzar(ag2, co2, rng2, 80)
    assert (co.congelado, co.n_validas, ag.n_cambios, ag.var) == (co2.congelado, co2.n_validas, ag2.n_cambios, ag2.var)
    ruta = config.RESULTADOS / 'prueba_instantanea.json'
    try:
        assert orquestador.guardar_instantanea(ruta, {'paso': 7, 'beta': np.float64(0.5)})
        assert orquestador.guardar_instantanea(ruta, {'paso': 8, 'beta': np.float64(0.5)})   # reemplaza
        assert json.loads(ruta.read_text()) == {'paso': 8, 'beta': 0.5}
        assert not ruta.with_suffix('.tmp').exists()
    finally:
        ruta.unlink(missing_ok=True)
    # el backend real guarda y repone seq de la ortesis y el centro M del decoder (modelos de mentira)
    import types
    import hardware as hw
    b = orquestador.BackendReal.__new__(orquestador.BackendReal)
    b.hw = types.SimpleNamespace(cargar=lambda nombre: types.SimpleNamespace(M=None, nombre=nombre))
    b.ortesis = hw.OrtesisSimulada()
    b.restaurar({'seq': 57, 'M': (2 * np.eye(8)).tolist()})
    assert b.ortesis.seq == 57 and np.array_equal(b.decoder.M, 2 * np.eye(8))
    assert b.detector.nombre == 'detector_errp.pkl'
    assert json.loads(json.dumps(b.instantanea())) == {'seq': 57, 'M': (2 * np.eye(8)).tolist()}
    # un fallo al guardar avisa, pero no lanza: el lazo no se cae por esto
    assert orquestador.guardar_instantanea(config.RESULTADOS / 'no_existe' / 'x.json', {}) is False
    return 'agente y confianza identicos tras JSON; escritura atomica; guardar nunca lanza'


@prueba
def reanudar():
    """Se mata el proceso a mitad de sesion; --reanudar continua la misma sesion y el
    mismo CSV, y el resultado es identico al de una sesion sin interrumpir."""
    import json
    import orquestador
    base = ['--semilla', '4', '--pasos_estatico', '20', '--pasos_adaptativo', '90', '--caos', '3']
    a = orquestador.argumentos(['sim', '--ciclo', '0'] + base)
    ref = orquestador.Orquestador(orquestador.BackendSim(a), a)
    orquestador.correr(ref, a)
    referencia = list(csv.DictReader(open(ref.ruta_csv)))
    config.ESTADO_SESION_JSON.unlink(missing_ok=True)

    p = subprocess.Popen([sys.executable, 'orquestador.py', 'sim', '--ciclo', '0.05'] + base,
                         cwd=config.RAIZ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t_fin, paso = time.time() + 60, 0
        while paso < 40 and time.time() < t_fin:           # se espera a que vaya a media sesion
            time.sleep(0.05)
            try:
                paso = json.loads(config.ESTADO_SESION_JSON.read_text())['paso']
            except (OSError, ValueError):
                pass
    finally:
        p.kill()                                           # cierre inesperado
        p.wait()
    inst = json.loads(config.ESTADO_SESION_JSON.read_text())
    assert 40 <= inst['paso'] < len(referencia) and not inst['terminada'], inst['paso']

    r = subprocess.run([sys.executable, 'orquestador.py', 'sim', '--reanudar', '--ciclo', '0'],
                       cwd=config.RAIZ, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-1500:]
    filas = list(csv.DictReader(open(inst['ruta_csv'])))
    # beta y el numero de paso continuan exactamente donde se quedaron
    assert float(filas[inst['paso'] - 1]['beta']) == round(inst['agente']['beta'], 4)
    assert len(filas) == len(referencia)
    quitar = lambda f: {k: v for k, v in f.items() if k not in ('t_iso', 't_lsl')}
    distintas = [i for i, (x, y) in enumerate(zip(filas, referencia)) if quitar(x) != quitar(y)]
    assert not distintas, f'filas distintas a la sesion sin interrumpir: {distintas[:5]}'
    assert json.loads(config.ESTADO_SESION_JSON.read_text())['terminada']
    # sesion ya terminada, o sin instantanea: mensaje claro y sin traza
    for caso, texto in (('terminada', 'ya termino'), ('ausente', 'No hay sesion')):
        if caso == 'ausente':
            config.ESTADO_SESION_JSON.unlink()
        r2 = subprocess.run([sys.executable, 'orquestador.py', 'sim', '--reanudar'],
                            cwd=config.RAIZ, capture_output=True, text=True, timeout=60)
        assert r2.returncode == 2 and 'Traceback' not in r2.stderr and texto in r2.stdout + r2.stderr, (caso, r2.stderr[-300:])
    return (f"matado en el paso {inst['paso']} de {len(referencia)}; reanudado identico a la "
            f"sesion sin interrumpir (con caos)")


# ------------------------------------------------------------ control causal con sham
@prueba
def senal_sham():
    """Lo que recibe el agente en el bloque sham. 'nula': LLR = 0 con las dos salidas del detector
    (P_hat queda en el prior). 'recientes': los mismos valores en otro orden, cada uno una sola vez,
    y sin el del paso actual; se guarda y se restaura para --reanudar."""
    from agente_errp import AgenteErrP, ConfigAgente, SenalSham
    for salida in ('calibrada', 'binaria'):
        ag = AgenteErrP([1.0], 0.0, ConfigAgente(salida_detector=salida))
        p, s, e = SenalSham('nula')(0.97, ag.cfg)
        assert abs(ag.prob_error(p, s, e, 1.0) - ag.prior) < 1e-12, salida
    cfg = ConfigAgente()
    senal, entra = SenalSham('recientes', semilla=1, memoria=8), [k / 100 for k in range(1, 61)]
    sale = [senal(p, cfg)[0] for p in entra]
    assert all(np.isnan(v) for v in sale[:8]) and not any(np.isnan(v) for v in sale[8:])
    dados = sale[8:]
    assert len(set(dados)) == len(dados) and set(dados) | set(senal.guardados) == set(entra)
    assert all(d != p for d, p in zip(sale, entra)), 'nunca entrega el p_errp del propio paso'
    assert np.mean([abs(entra.index(d) - k) for k, d in enumerate(sale) if not np.isnan(d)]) < 20   # son recientes
    # 'calibracion' (sham ciego, solo en el estudio): permutaciones sucesivas de los p_errp de la calibracion
    ciego = SenalSham('calibracion', semilla=2, reserva=[0.1, 0.2, 0.9])
    tres = [ciego(0.5, cfg)[0] for _ in range(6)]
    assert sorted(tres[:3]) == sorted(tres[3:]) == [0.1, 0.2, 0.9] and 'calibracion' not in config.SHAM_ERRP_FUENTES_LAZO
    copia = SenalSham('recientes').desde_dict(senal.a_dict())
    assert [copia(0.5, cfg)[0] for _ in range(5)] == [senal(0.5, cfg)[0] for _ in range(5)]
    try:
        SenalSham('otra')
        raise AssertionError('acepto una fuente desconocida')
    except ValueError:
        pass
    import hardware as hw
    lo, hi = hw.intervalo_diferencia([1] * 40, [0] * 40)
    assert lo == hi == 1.0
    lo, hi = hw.intervalo_diferencia([0, 1] * 20, [1, 0] * 20)
    assert lo <= 0 <= hi
    return "'nula' deja P_hat en el prior; 'recientes' permuta sin repetir ni adelantar; intervalo de la diferencia"


def _sesion_sham(semilla, orden, fuente=config.SHAM_ERRP_FUENTE, extra=()):
    import orquestador
    a = orquestador.argumentos(['sim', '--ciclo', '0', '--semilla', str(semilla), '--pasos_estatico', '20', '--sham',
                                '--sham-orden', orden, '--sham-fuente', fuente, *extra])
    orq = orquestador.Orquestador(orquestador.BackendSim(a), a)
    eventos = []
    estado = orq.salidas.estado
    orq.salidas.estado = lambda **d: (eventos.append(d), estado(**d))[1]
    orquestador.correr(orq, a)
    return orq, eventos


@prueba
def orquestador_sham():
    """--sham: dos bloques del mismo largo con el agente reiniciado y su propia perturbacion en el
    mismo paso; en el sham no se congela y la fiabilidad queda fija; EVALUACION los compara; el
    flujo Estado no dice cual es cual hasta el final (el tablero lo oculta)."""
    n, en = config.SHAM_ERRP_PASOS, config.SHAM_ERRP_PERTURBAR_EN
    orq, eventos = _sesion_sham(3, 'sham-real')
    filas = list(csv.DictReader(open(orq.ruta_csv)))
    assert list(filas[0]) == config.COLUMNAS_CSV
    assert [f['bloque'] for f in filas] == [''] * 20 + ['sham'] * n + ['real'] * n
    m = orq.salidas.marcadores
    assert m.index('bloque:sham') < m.index('bloque:real') and m.count(config.PERTURBACION_ON) == 2
    for nombre, ini in (('sham', 20), ('real', 20 + n)):
        b = orq.sham['bloques'][nombre]
        assert b['inicio'] == ini and b['t_perturbacion'] == ini + en and abs(b['beta_pre']) < 1.0, (nombre, b)
        # arranca reiniciado: tras su primer paso la varianza es, como mucho, la inicial mas el ruido de un paso
        c = orq.agente.cfg
        assert float(filas[ini]['varianza_beta']) <= c.varianza_inicial + c.ruido_proceso + 1e-4
    sham = [f for f in filas if f['bloque'] == 'sham']
    assert {f['estado'] for f in sham} <= {'LAZO_ADAPTATIVO', 'PERTURBACION'}          # nunca congelado
    assert {f['fiabilidad'] for f in sham} == {'1.0'}
    # fuente 'nula': P_hat es solo el prior del paso (con el prior por paso, el error que el agente predice
    # con su piso: depende de p', no del error verdadero); sin la bandera seria el mismo prior en todos los pasos
    con = [f for f in sham if f['P_hat']]
    assert con
    if orq.agente.cfg.prior_por_paso:
        piso = orq.agente.cfg.piso_prior
        assert all(abs(float(f['P_hat']) - max(1 - max(float(f['p_prima']), 1 - float(f['p_prima'])), piso)) < 2e-3 for f in con)
    else:
        assert len({f['P_hat'] for f in con}) == 1
    c = orq.comparacion_sham
    assert c['orden'] == ['sham', 'real'] and c['ic_dif'][0] <= c['dif'] <= c['ic_dif'][1]
    assert orq.error_post['agente'] == c['real']['agente']                      # el CP4 es el del bloque real
    pasos = [e for e in eventos if e['tipo'] == 'paso']
    assert [e['bloque'] for e in pasos] == [None] * 20 + ['A'] * n + ['B'] * n
    assert [e['tipo'] for e in eventos if e['tipo'] in ('bloque_sham', 'sham')] == ['bloque_sham', 'bloque_sham', 'sham']
    # 12 sesiones del simulador con el orden alternado: beta se recupera en el real y casi nunca en el
    # sham. (En el simulador la perturbacion sube poco el error, 0.30 de la sombra contra 0.50 en el
    # gemelo, asi que aqui se compara la recuperacion; la diferencia de error se mide en el gemelo.)
    rec = {'real': 0, 'sham': 0}
    for s in range(12):
        o, _ = _sesion_sham(100 + s, ('real-sham', 'sham-real')[s % 2])
        for b in rec:
            rec[b] += o.comparacion_sham[b]['pasos'] is not None
    assert rec['real'] >= rec['sham'] + 5 and rec['sham'] <= 2, rec
    # la fuente 'recientes' tambien corre; el orden al azar queda registrado
    o, _ = _sesion_sham(5, 'real-sham', 'recientes')
    assert len({f['P_hat'] for f in csv.DictReader(open(o.ruta_csv)) if f['bloque'] == 'sham' and f['P_hat']}) > 1
    import orquestador
    a = orquestador.argumentos(['sim', '--sham'])
    assert a.sham_orden is None and sorted(orquestador.Orquestador(orquestador.BackendSim(a), a).sham['orden']) == ['real', 'sham']
    return f"simulador, 12 sesiones: beta se recupera en el bloque real {rec['real']}/12 y en el sham {rec['sham']}/12"


@prueba
def reanudar_sham():
    """Una sesion --sham interrumpida a mitad del segundo bloque se reanuda identica (misma senal sham)."""
    import orquestador
    for fuente in config.SHAM_ERRP_FUENTES_LAZO:
        ref, _ = _sesion_sham(7, 'real-sham', fuente)
        referencia = list(csv.DictReader(open(ref.ruta_csv)))
        a = orquestador.argumentos(['sim', '--ciclo', '0', '--semilla', '7', '--pasos_estatico', '20', '--sham',
                                    '--sham-orden', 'real-sham', '--sham-fuente', fuente])
        orq = orquestador.Orquestador(orquestador.BackendSim(a), a)
        corte = 20 + config.SHAM_ERRP_PASOS + 33

        def guardar(terminada=False, orq=orq):
            orquestador.Orquestador.guardar(orq, terminada)
            if len(orq.filas) == corte:
                raise KeyboardInterrupt
        orq.guardar = guardar
        orquestador.correr(orq, a)
        inst = orquestador.cargar_instantanea()
        assert inst['paso'] == corte and inst['sham']['actual'] == 'sham' and inst['prog']['bloque'] == 'sham'
        a2 = orquestador.argumentos_reanudados(inst, orquestador.argumentos(['sim', '--reanudar', '--ciclo', '0']))
        orq2 = orquestador.Orquestador(orquestador.BackendSim(a2), a2, inst)
        orquestador.correr(orq2, a2)
        filas = list(csv.DictReader(open(orq2.ruta_csv)))
        quitar = lambda f: {k: v for k, v in f.items() if k not in ('t_iso', 't_lsl')}
        distintas = [i for i, (x, y) in enumerate(zip(filas, referencia)) if quitar(x) != quitar(y)]
        assert len(filas) == len(referencia) and not distintas, (fuente, len(filas), distintas[:5])
        assert orq2.comparacion_sham['real']['agente'] == ref.comparacion_sham['real']['agente']
    return 'interrumpida en el paso 33 del bloque sham y reanudada: CSV identico con las dos fuentes'


@prueba
def controles_especificidad():
    """Lo que se le agrego a los dos controles de jusren. Fisher por direccion: avisa cuando las falsas
    alarmas se cargan a una direccion y casi nunca cuando no. --control-reposo: su bloque sham corre
    dentro del orquestador real, tras calibrar, con el EEG y la ortesis de la sesion."""
    import types
    import bloque_sham
    import hardware as hw
    import orquestador
    rng = np.random.default_rng(0)
    avisos = {'igual': 0, 'cargado': 0}
    for _ in range(200):
        y, d = (rng.random(120) < 0.3).astype(int), rng.integers(0, 2, 120)
        for caso, fa in (('igual', (0.10, 0.10)), ('cargado', (0.02, 0.30))):       # falsas alarmas abrir, cerrar
            pred = np.where(y == 1, rng.random(120) < 0.7, rng.random(120) < np.where(d == 1, fa[1], fa[0])).astype(int)
            avisos[caso] += hw.fisher_por_direccion(y, pred, d)['avisa']
    assert avisos['igual'] <= 16 and avisos['cargado'] >= 170, avisos
    assert hw.fisher_por_direccion([0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0])['p'] == 1.0
    # --control-reposo: el orquestador real corre bloque_sham.correr con lo suyo y publica el resultado
    class EEG:
        fs = 250
        ventana = lambda self, seg, pm=0.0, hm=None: (rng.normal(0, 5, (8, int(seg * 250))), None)
        ultimo_t = lambda self: 0.0
        movimiento = lambda self, t0, t1: None
    Xmi = rng.normal(0, 5, (40, 8, 500))
    dec = hw.DecoderIM().ajustar(Xmi, np.arange(40) % 2)
    b = orquestador.BackendReal.__new__(orquestador.BackendReal)
    b.hw, b.eeg, b.decoder, b.ortesis = hw, EEG(), dec, hw.OrtesisSimulada(latencia_ms=0.1, jitter_ms=0.0)
    eventos, marcas = [], []
    orq = types.SimpleNamespace(salidas=types.SimpleNamespace(marcador=marcas.append, estado=lambda **d: eventos.append(d)))
    dormir = bloque_sham.time.sleep
    bloque_sham.time.sleep = lambda s: None
    try:
        r = b.control_reposo(orq)
    finally:
        bloque_sham.time.sleep = dormir
        b.ortesis.cerrar()
    assert marcas == [config.CONTROL_REPOSO] and eventos[0]['tipo'] == 'control_reposo' and eventos[0]['n'] == config.SHAM_PASOS
    assert r['pasa'] in (True, False) and len(b.reposo['filas']) == config.SHAM_PASOS and b.ortesis.seq >= 2 * config.SHAM_PASOS
    assert orquestador.argumentos(['real', '--control-reposo']).control_reposo
    return (f"Fisher por direccion: avisa {avisos['cargado']}/200 con sesgo y {avisos['igual']}/200 sin el; --control-reposo corre "
            f"el bloque sham de jusren dentro del orquestador ({config.SHAM_PASOS} movimientos, AUC {r['auc']:.2f})")


# ------------------------------------------------------------ IA: API simulada
class ApiSimulada:
    """La API de mensajes, de mentira: devuelve en orden las respuestas del guion (o lo que devuelva
    cada funcion del guion al recibir la peticion) y guarda las peticiones para revisarlas."""

    def __init__(self, *guion):
        self.guion, self.peticiones, self.messages = list(guion), [], self

    def create(self, **kw):
        self.peticiones.append(kw)
        r = self.guion.pop(0)
        if isinstance(r, Exception):
            raise r
        return r(kw) if callable(r) else r


def _resp(*bloques, fin='end_turn'):
    import types
    return types.SimpleNamespace(stop_reason=fin, content=[
        types.SimpleNamespace(type='text', text=b) if isinstance(b, str)
        else types.SimpleNamespace(type='tool_use', name=b[0], input=b[1], id=f'tu_{k}') for k, b in enumerate(bloques)])


def _sesion_copiloto():
    import copiloto
    import orquestador
    a = orquestador.argumentos(['sim', '--ciclo', '0', '--semilla', '4', '--pasos_estatico', '20', '--caos', '3',
                                '--falla_detector'])
    orq = orquestador.Orquestador(orquestador.BackendSim(a), a)
    orquestador.correr(orq, a)
    return orq, copiloto.Sesion(orq.ruta_csv)


@prueba
def copiloto_herramientas():
    """Las herramientas del copiloto devuelven las cifras de la sesion (las mismas de EVALUACION), con
    los pasos que usaron; si el dato no existe lo dicen; las seis preguntas de TAREAS.md se responden
    con plantillas, sin API."""
    import json
    import copiloto
    orq, s = _sesion_copiloto()
    r = s.metrica('error', 'tras_perturbacion')
    assert (r['error_agente'], r['error_sombra']) == (orq.error_post['agente'], orq.error_post['sombra'])
    assert tuple(r['ic90_agente']) == tuple(orq.error_post['ic_agente']) and r['pasos'][0] == orq.t_perturbacion + 1
    rec = s.metrica('recuperacion')[0]
    assert rec['paso_perturbacion'] == orq.t_perturbacion + 1 and rec.get('pasos') == orq.error_post['pasos']
    exc = s.metrica('excluidos')
    assert {m: d['n'] for m, d in exc['por_motivo'].items()} == orq.excluidos and exc['excluidos'] == sum(orq.excluidos.values())
    assert all(orq.filas[p - 1]['excluido'] == m for m, d in exc['por_motivo'].items() for p in d['pasos'])
    res = s.resumen_sesion()
    assert res['pasos_validos'] == len(orq.filas) - exc['excluidos'] and res['sin_movimiento'] == orq.sin_movimiento
    assert res['bloques']['estatico']['n'] + res['bloques']['adaptativo']['n'] == res['pasos_validos']
    assert any('CP4' in c for c in res['checkpoints'])
    # eventos: la falla del detector congela el aprendizaje; cada fila de pausa es un evento con su causa
    cong = s.eventos_de(tipo='congelamiento')
    congeladas = [f['paso'] for f in s.filas if f['estado'] == 'APRENDIZAJE_CONGELADO']
    assert cong and cong[0]['paso'] == congeladas[0] and max(e['hasta_paso'] for e in cong) >= congeladas[-1]
    pausas = s.eventos_de(tipo='pausa')
    assert len(pausas) == sum(str(f['excluido']).startswith('pausa:') for f in s.filas) > 0
    assert [e['paso'] for e in s.eventos_de(tipo='perturbacion')] == [orq.t_perturbacion + 1]
    assert all(40 <= e['paso'] <= 60 for e in s.eventos_de(40, 60)) and s.eventos_de(tipo='semaforo')
    ba = s.metrica('ba_viva')
    assert ba['en_el_paso'] == s.filas_de()[-1]['paso'] and abs(ba['ba_viva'] - (orq.confianza.sens + orq.confianza.espec) / 2) < 0.01
    fi = s.metrica('fiabilidad', 'adaptativo')
    assert fi['pasos_congelados'] == len([p for p in congeladas if not s.filas[p - 1]['excluido']]) and fi['minima'] == 0.0
    sp = s.metrica('sin_errp_por_paso')
    assert sp['chico']['errores'] + sp['grande']['errores'] > 20 and sp['nota'] == 'deteccion registrada por paso'
    # filas puntuales con limite; lo que no existe se dice, lo mal pedido falla con un mensaje claro
    f = s.pasos(1, 200, ['beta', 'excluido'])
    assert len(f['filas']) == config.COPILOTO_MAX_FILAS and f['truncado'] and set(f['filas'][0]) == {'paso', 'beta', 'excluido'}
    assert s.pasos(5)['filas'][0]['paso'] == 5 and copiloto.SIN_DATO in s.pasos(9000)['motivo']
    assert copiloto.SIN_DATO in s.metrica('alfa')['motivo'] and copiloto.SIN_DATO in s.metrica('error', 'sham')['motivo']
    for mala in (lambda: s.metrica('felicidad'), lambda: s.metrica('error', 'otro'), lambda: s.pasos(1, 2, ['eeg_crudo']),
                 lambda: s.eventos_de(tipo='fiesta')):
        try:
            mala()
            raise AssertionError('acepto una peticion invalida')
        except ValueError:
            pass
    comp = s.comparar_sesiones()
    assert comp['sesiones'][-1]['sesion'] == s.ruta.name and len(comp['sesiones']) == 2
    assert copiloto.SIN_DATO in s.comparar_sesiones(['no_existe.csv'])['motivo']
    json.dumps(copiloto.ia.sanear(res))                             # todo lo que sale es JSON
    # las seis preguntas, por plantillas
    n = congeladas[len(congeladas) // 2]
    t = {q: copiloto.responder(s, q)[0] for q in (
        f'por que se congelo el aprendizaje en el paso {n}?', 'cuanto tardo en recuperarse tras la perturbacion?',
        'el agente le gano a la sombra y con que certeza?', 'hubo senales de fatiga?',
        'cuantos pasos se excluyeron y por que?', 'como se compara con la sesion anterior?',
        'por que se congelo el aprendizaje en el paso 5?', 'por que se congelo en el paso 99999?')}
    r1, r2, r3, r4, r5, r6, r7, r8 = t.values()
    assert f'paso {n} cae en un congelamiento' in r1 and 'sens viva' in r1 and 'falsas alarmas' in r1, r1
    assert f'paso {orq.t_perturbacion + 1}' in r2 and ('se recupero en' in r2 or 'NO se recupero' in r2), r2
    assert f"agente {orq.error_post['agente']:.2f}" in r3 and 'sombra - agente' in r3 and ('certeza' in r3), r3
    assert copiloto.SIN_DATO in r4 and 'fatiga' in r4, r4
    assert f"{exc['excluidos']} de {len(orq.filas)}" in r5 and all(m in r5 for m in orq.excluidos), r5
    assert s.ruta.name in r6 and 'error_agente' in r6, r6
    assert 'no estaba congelado' in r7 and copiloto.SIN_DATO in r8
    return (f"cifras iguales a EVALUACION (agente {orq.error_post['agente']:.2f}, sombra {orq.error_post['sombra']:.2f}); "
            f"{len(cong)} congelamientos, {len(pausas)} pausas; 6 preguntas respondidas con sus pasos; lo que falta dice '{copiloto.SIN_DATO}'")


@prueba
def copiloto_api_simulada():
    """Con la API simulada: el copiloto usa las herramientas y devuelve lo que diga el modelo; una
    pregunta sin datos produce 'no hay dato'; si la API falla o declina responden las plantillas; a
    la API no sale ni una ruta ni una senal cruda; una propuesta fuera de rango se rechaza; el
    informe deja la propuesta pendiente de una persona."""
    import json
    import copiloto
    import ia
    _, s = _sesion_copiloto()
    eco = lambda kw: _resp('Respuesta: ' + ' / '.join(b['content'] for b in kw['messages'][-1]['content']))
    api = ApiSimulada(_resp(('metrica', {'nombre': 'alfa'}), ('eventos', {'tipo': 'congelamiento'}),
                            ('metrica', {'nombre': 'felicidad'}), fin='tool_use'), eco)
    txt, origen, usadas = copiloto.responder(s, 'hubo senales de fatiga?', api)
    assert origen == 'api' and copiloto.SIN_DATO in txt and 'aprendizaje congelado del paso' in txt, txt
    assert [u[0] for u in usadas] == ['metrica', 'eventos', 'metrica']
    p1, p2 = api.peticiones
    assert p1['model'] == config.IA_MODELO == 'claude-opus-5-5' and len(p1['tools']) == 5 and copiloto.SIN_DATO in p1['system']
    assert 'tool_choice' not in p1 and 'thinking' not in p1 and 'temperature' not in p1     # el modelo los rechaza
    resultados = p2['messages'][-1]['content']
    assert [r['is_error'] for r in resultados] == [False, False, True] and len(p2['messages']) == 3
    enviado = json.dumps([p['messages'] for p in api.peticiones], default=lambda o: o.__dict__)
    assert str(config.RAIZ.parent) not in enviado and 'Users' not in enviado, 'salio una ruta de la maquina'
    # la API falla, declina o da vueltas sin fin: plantillas, y se dice
    for guion in ([ConnectionError('sin red')], [_resp(fin='refusal')], [_resp(('pasos', {'desde': 1}), fin='tool_use')] * 20):
        txt, origen, _ = copiloto.responder(s, 'cuantos pasos se excluyeron y por que?', ApiSimulada(*guion))
        assert origen == 'reglas' and 'Se excluyeron' in txt and 'la IA no respondio' in txt, txt
    # sanear: ni senales crudas ni rutas
    try:
        ia.sanear({'eeg': [0.1] * 1000})
        raise AssertionError('dejo pasar una senal cruda')
    except ValueError:
        pass
    assert ia.sanear({'ruta': str(s.ruta), 'p': s.ruta, 'x': float('nan')}) == {'ruta': s.ruta.name, 'p': s.ruta.name, 'x': None}
    # propuestas: esquema fijo y rangos seguros
    buena = {'accion': 'ajustar_parametro', 'parametro': 'paso_visible', 'valor': 0.1, 'justificacion': 'los pasos chicos no se ven'}
    casos = {'valida': (buena, True), 'fuera de rango': ({**buena, 'parametro': 'paso_max', 'valor': 0.9}, False),
             'parametro prohibido': ({**buena, 'parametro': 'beta_max'}, False), 'sin valor': ({**buena, 'valor': None}, False),
             'accion inventada': ({**buena, 'accion': 'apagar'}, False), 'campo de mas': ({**buena, 'extra': 1}, False),
             'continuar con valor': ({**buena, 'accion': 'continuar'}, False), 'pausa larga': ({**buena, 'accion': 'pausa', 'parametro': 'pausa_s', 'valor': 9000}, False),
             'pausa': ({**buena, 'accion': 'pausa', 'parametro': 'pausa_s', 'valor': 60}, True),
             'recalibrar': ({'accion': 'recalibrar', 'parametro': None, 'valor': None, 'justificacion': 'BA viva 0.55'}, True),
             'valor no numerico': ({**buena, 'valor': 'mucho'}, False), 'sin justificacion': ({**buena, 'justificacion': ' '}, False)}
    for nombre, (p, esperado) in casos.items():
        assert ia.validar_propuesta(p)[0] is esperado, (nombre, ia.validar_propuesta(p))
    resumen = copiloto.resumen_para_propuesta(s, 'adaptativo')
    assert resumen['pasos'] == len(s.filas_de('adaptativo')) and 0 < resumen['fraccion_congelado'] < 1
    r = ia.proponer(resumen, ApiSimulada(_resp(json.dumps(casos['fuera de rango'][0]))))
    assert r['origen'] == 'reglas' and r['valida'] and 'fuera del rango seguro' in r['rechazada_api']['motivo'], r
    api = ApiSimulada(_resp(json.dumps(buena)))
    r = ia.proponer(resumen, api)
    assert r['origen'] == 'api' and r['propuesta'] == buena and api.peticiones[0]['output_config']['format']['type'] == 'json_schema'
    assert json.loads(api.peticiones[0]['messages'][0]['content'])['pasos'] == resumen['pasos']
    assert ia.proponer(resumen, ApiSimulada(TimeoutError('lenta')))['origen'] == 'reglas'
    reglas = ia.propuesta_por_reglas
    assert reglas({'pasos': 5})['accion'] == 'continuar' and reglas({'pasos': 90, 'alfa_rel': 2.0})['accion'] == 'pausa'
    assert reglas({'pasos': 90, 'ba_viva': 0.52})['accion'] == 'recalibrar'
    assert reglas({'pasos': 90, 'fraccion_excluidos': 0.4, 'excluidos': {'pausa:eeg': 30}})['accion'] == 'pausa'
    chico = reglas({'pasos': 90, 'ba_viva': 0.8, 'sin_errp_por_paso': {'chico': {'errores': 12, 'sin_errp': 0.7},
                                                                       'grande': {'errores': 10, 'sin_errp': 0.2}}})
    assert chico['parametro'] == 'paso_visible' and chico['valor'] == 0.1
    for p in (reglas({'pasos': 90, 'alfa_rel': 2.0}), chico, reglas({'pasos': 90, 'ba_viva': 0.9})):
        assert ia.validar_propuesta(p) == (True, ''), p
    # informe: cuatro Markdown, figuras y una propuesta que espera a una persona
    copiloto.ruta_propuestas(s.ruta).unlink(missing_ok=True)
    interp = {k: f'INTERPRETACION {k}' for k in copiloto.ESQUEMA_INTERPRETACION['required']}
    api = ApiSimulada(_resp(json.dumps(interp)), _resp(json.dumps({**buena, 'valor': 5})))
    inf = copiloto.informe(s, cli=api)
    md = [p for p in inf['archivos'] if p.suffix == '.md']
    assert len(md) == 4 and all(p.exists() and p.stat().st_size > 200 for p in inf['archivos']) and inf['origen_texto'] == 'api'
    for p in md:
        txt = p.read_text(encoding='utf-8')
        assert 'INTERPRETACION ' + '_'.join(p.stem.split('_')[-2:]) in txt, p.name
        assert ('"accion"' in txt) == ('terapeuta' in p.name) and '_fig_sesion.png' in txt
    assert 'Requiere la aprobación' in md[0].read_text(encoding='utf-8') and 'Requires approval' in md[1].read_text(encoding='utf-8')
    assert inf['propuesta']['origen'] == 'reglas' and inf['propuesta']['rechazada_api']         # la de la API (valor 5) no paso
    pend = copiloto.pendiente(s.ruta, 'proxima_sesion')
    assert pend and pend['id'] == inf['propuesta']['id'] and copiloto.main(['--sesion', str(s.ruta), '--decidir', 'rechazar']) == 0
    assert copiloto.pendiente(s.ruta) is None
    registros = [l['registro'] for l in copiloto.leer_propuestas(s.ruta)]
    assert registros == ['propuesta', 'decision', 'efecto'] and copiloto.leer_propuestas(s.ruta)[1]['decision'] == 'rechazada'
    sin = copiloto.informe(s, cli=None)
    assert sin['origen_texto'] == 'reglas' and 'movimientos' in sin['archivos'][2].read_text(encoding='utf-8')
    return ('herramientas por la API simulada, "no hay dato" cuando falta, plantillas si la API falla, nada crudo sale, '
            '12 propuestas validadas contra los rangos, informe en 4 Markdown con propuesta pendiente')


@prueba
def coinvestigador_entre_bloques():
    """--coinvestigador: al terminar cada bloque hay una propuesta registrada con su decision y su
    efecto. Aprobada, cambia el parametro (dentro del rango seguro); rechazada o sin decision, nada
    cambia; una propuesta de la API fuera de rango nunca llega al operador; recalibrar aprobado
    termina la sesion en orden; entre los dos bloques del control causal no se ajusta nada."""
    import json
    import copiloto
    import orquestador

    def sesion(guion, decide, extra=()):
        a = orquestador.argumentos(['sim', '--ciclo', '0', '--semilla', '2', '--pasos_estatico', '40', '--pasos_adaptativo',
                                    '90', '--coinvestigador', '--coinvestigador-espera', '2', *extra])
        orq = orquestador.Orquestador(orquestador.BackendSim(a), a)
        orq.cliente_ia = ApiSimulada(*[_resp(json.dumps(g)) for g in guion]) if guion else None
        eventos, estado = [], orq.salidas.estado
        orq.salidas.estado = lambda **d: (eventos.append(d), estado(**d))[1]
        if decide is not None:                      # como el boton del tablero: escribe la decision en el registro
            orq.al_proponer = lambda p: eventos[-1]['espera_s'] > 0 and copiloto.decidir(orq.ruta_csv, p['id'], decide, por='prueba')
        orquestador.correr(orq, a)
        return orq, [l for l in copiloto.leer_propuestas(orq.ruta_csv)], eventos

    p = lambda accion, parametro=None, valor=None: {'accion': accion, 'parametro': parametro, 'valor': valor,
                                                   'justificacion': 'por las cifras del bloque'}
    sube = p('ajustar_parametro', 'paso_visible', 0.1)
    # aprobada: el parametro cambia para el bloque siguiente y todo queda registrado
    orq, reg, ev = sesion([sube, p('continuar')], True)
    assert orq.agente.cfg.paso_visible == 0.1 and orq.ajustes == {'paso_visible': 0.1} and len(orq.filas) == 130
    assert [l['registro'] for l in reg] == ['propuesta', 'decision', 'efecto', 'propuesta'], [l['registro'] for l in reg]
    assert reg[0]['origen'] == 'api' and reg[1]['decision'] == 'aprobada' and reg[2]['efecto'] == 'paso_visible: 0.08 -> 0.1'
    r = reg[0]['resumen']
    assert r['bloque'] == 'estatico' and r['pasos'] == 40 and {'exactitud', 'error_sombra', 'ba_viva', 'fiabilidad_media',
                                                                'excluidos', 'alfa_rel', 'sin_errp_por_paso'} <= set(r)
    chicos = [abs(float(f['delta'])) for f in orq.filas[40:] if not f['excluido'] and f['alineacion'] != config.SIN_MOVIMIENTO]
    assert min(chicos) >= 0.1 - 1e-9 > config.PASO_VISIBLE, min(chicos)          # el efecto se ve en el lazo
    tipos = [e['tipo'] for e in ev if e['tipo'] in ('propuesta', 'decision')]
    assert tipos == ['propuesta', 'decision', 'propuesta'] and ev[[e['tipo'] for e in ev].index('propuesta')]['csv'] == orq.ruta_csv.name
    enviado = json.loads(orq.cliente_ia.peticiones[0]['messages'][0]['content'])
    assert enviado['pasos'] == 40 and 'Users' not in json.dumps(enviado)
    # rechazada, y sin decision: nada cambia
    for decide, decision in ((False, 'rechazada'), (None, 'sin_decision')):
        orq, reg, _ = sesion([sube, p('continuar')], decide)
        assert orq.agente.cfg.paso_visible == config.PASO_VISIBLE and not orq.ajustes and len(orq.filas) == 130
        assert reg[1]['decision'] == decision and 'nada cambio' in reg[2]['efecto'], reg[1:3]
    # la API propone algo fuera de rango: se descarta y deciden las reglas; sin API, reglas tambien
    orq, reg, _ = sesion([p('ajustar_parametro', 'paso_max', 0.95), p('continuar')], True)
    assert reg[0]['origen'] == 'reglas' and 'fuera del rango seguro' in reg[0]['rechazada_api']['motivo']
    assert orq.agente.cfg.paso_max == config.PASO_MAX
    orq, reg, _ = sesion(None, True)
    assert [l['origen'] for l in reg if l['registro'] == 'propuesta'] == ['reglas', 'reglas'] and len(orq.filas) == 130
    # recalibrar aprobado: la sesion termina en orden tras el bloque estatico; pausa aprobada: sigue
    orq, reg, _ = sesion([p('recalibrar')], True)
    assert len(orq.filas) == 40 and orq.fsm.estado == 'EVALUACION' and '--solo-errp' in reg[2]['efecto']
    orq, reg, _ = sesion([p('pausa', 'pausa_s', 45), p('continuar')], True)
    assert len(orq.filas) == 130 and reg[2]['efecto'].startswith('pausa de 45 s')
    # paso visible mayor que el paso maximo: aprobado, pero incoherente -> no se aplica
    orq, reg, _ = sesion([p('ajustar_parametro', 'paso_max', 0.15), sube, p('continuar')], True, ['--sham', '--sham-orden', 'real-sham'])
    efectos = [l['efecto'] for l in reg if l['registro'] == 'efecto']
    assert efectos[0] == 'paso_max: 0.3 -> 0.15' and 'control causal' in efectos[1], efectos       # entre A y B no se toca
    bloques = [l['resumen']['bloque'] for l in reg if l['registro'] == 'propuesta']
    assert bloques == ['estatico', 'bloque A', 'bloque B'], bloques                                 # el ciego se mantiene
    # una sesion sin la bandera no propone nada ni crea el registro
    a = orquestador.argumentos(['sim', '--ciclo', '0', '--pasos_estatico', '20', '--pasos_adaptativo', '30'])
    orq = orquestador.Orquestador(orquestador.BackendSim(a), a)
    orquestador.correr(orq, a)
    assert not copiloto.ruta_propuestas(orq.ruta_csv).exists()
    return ('aprobada: paso_visible 0.08 -> 0.10 y el lazo lo usa; rechazada o sin decision: nada cambia; fuera de rango: '
            'descartada; recalibrar termina en orden; en el control causal se mantiene el ciego')


@prueba
def narrador_jurado():
    """El narrador saca de los eventos de una sesion lo que vale la pena contar (checkpoints,
    perturbacion, recuperacion, congelamiento, pausas, control causal), en espanol o ingles. Con la
    API simulada usa su frase si es corta y no trae cifras ajenas al evento; si la API falla, tarda o
    inventa un numero, plantilla. Durante el control causal no cuenta nada que delate el bloque real."""
    import json
    import narrador as nr
    orq, _ = _sesion_copiloto()                                   # caos + falla del detector: de todo un poco
    registro = orq.ruta_csv.with_name(orq.ruta_csv.stem + config.SUFIJO_ESTADO)
    eventos = [json.loads(l)['evento'] for l in registro.read_text(encoding='utf-8').splitlines() if '"evento"' in l]
    n = nr.Narrador('es')
    hechos = [h for e in eventos for h in n.observar(e)]
    tipos = [h['evento'] for h in hechos]
    assert tipos.count('perturbacion') == 1 and tipos.count('pausa') == tipos.count('reanudado') >= 3, tipos
    assert 'congelamiento' in tipos and 'checkpoint' in tipos, tipos
    contados = {t: tipos.count(t) for t in sorted(set(tipos))}
    pert = next(h for h in hechos if h['evento'] == 'perturbacion')
    assert pert['paso'] == orq.t_perturbacion + 1
    if orq.error_post['pasos'] is not None:                       # misma recuperacion que EVALUACION (en filas, con pausas)
        rec = next(h for h in hechos if h['evento'] == 'recuperacion')
        assert rec['pasos'] >= orq.error_post['pasos']
    frases = {idioma: [nr.Narrador(idioma).plantilla(h) for h in hechos] for idioma in ('es', 'en')}
    assert all(f and len(f) <= config.NARRADOR_MAX_CARACTERES and '{' not in f for fs in frases.values() for f in fs)
    assert any('Pausa segura' in f for f in frases['es']) and any('Safe pause' in f for f in frases['en'])
    assert all(nr.frase_valida(f, h) for fs in frases.values() for f, h in zip(fs, hechos)), 'una plantilla trae cifras ajenas'
    # API simulada: frase buena -> se usa; con un numero inventado, larga, vacia, rechazo o error -> plantilla
    h = {'evento': 'recuperacion', 'paso': 60, 'pasos': 26, 'segundos': 55}
    api = ApiSimulada(_resp('El agente volvio a acertar en 26 pasos, unos 55 segundos, leyendo el cerebro del piloto.'),
                      _resp('El agente se recupero en 12 pasos.'), _resp('x' * 400), _resp(''), _resp('frase', fin='refusal'),
                      TimeoutError('lenta'))
    n = nr.Narrador('es', api)
    assert n.frase(h) == ('El agente volvio a acertar en 26 pasos, unos 55 segundos, leyendo el cerebro del piloto.', 'api')
    for _ in range(5):
        assert n.frase(h) == (n.plantilla(h), 'plantilla')
    p = api.peticiones[0]
    assert p['model'] == config.IA_MODELO and json.loads(p['messages'][0]['content']) == h and 'espanol' in p['system']
    assert p['output_config'] == {'effort': 'low'}
    assert n.frase(h, nacio=time.time() - 60) == (n.plantilla(h), 'plantilla') and len(api.peticiones) == 6   # evento viejo: ni pregunta
    assert nr.frase_valida('Fiabilidad del 70 %.', {'f': 0.7}) and not nr.frase_valida('En 3 pasos.', {'pasos': 26})
    # control causal: los pasos traen la letra del bloque; el narrador no dice cual se recupero hasta el final
    o, ev = _sesion_sham(3, 'real-sham')
    n = nr.Narrador('en')
    hechos = [x for e in ev for x in n.observar(e)]
    tipos = [x['evento'] for x in hechos]
    assert tipos.count('perturbacion') == 2 and 'recuperacion' not in tipos and 'congelamiento' not in tipos, tipos
    assert 'sham' in tipos
    assert 'Causal control' in n.plantilla(next(x for x in hechos if x['evento'] == 'sham'))
    assert config.FLUJOS['Narracion'][0] == 'Markers'
    return f"sesion con caos y falla del detector: {contados}; en el control causal no delata el bloque real"


# ------------------------------------------------------------ tablero
@prueba
def tablero_flechas():
    """--flechas: la contribucion de cada ErrP al cambio de beta (tamano y direccion), sin pantalla;
    el reinicio de beta entre bloques del control causal no cuenta como flecha, y en el ciego queda oculto."""
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from pyqtgraph.Qt import QtWidgets
    import tablero
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    paso = {'tipo': 'paso', 'estado': 'LAZO_ADAPTATIVO', 'meta': 1, 'angulo': 0.5, 'p_crudo': 0.6, 'b': 0.5,
            'p_prima': 0.6, 'P_hat': 0.7, 'error': 1, 'error_sombra': 1, 'sd_beta': 0.5, 'youden': 0.6,
            'fiabilidad': 1.0, 'congelado': False, 'cambio': '', 'latencia_ms': 5.0, 'perturbado': False,
            'salud': {'eeg': 'VERDE', 'ortesis': 'VERDE', 'reloj': 'VERDE', 'detector': 'VERDE'}}
    # la funcion pura
    a = {**paso, 'paso': 1, 'beta': 0.0, 'bloque': None}
    assert tablero.contribucion_beta(None, a) == 0.0 and tablero.contribucion_beta(0.0, a, None) == 0.0
    assert abs(tablero.contribucion_beta(0.0, {**a, 'paso': 2, 'beta': 0.4}, a) - 0.4) < 1e-12
    assert abs(tablero.contribucion_beta(0.4, {**a, 'paso': 3, 'beta': 0.1}, {**a, 'paso': 2}) + 0.3) < 1e-12
    assert tablero.contribucion_beta(0.4, {**a, 'paso': 1, 'beta': 0.0}, {**a, 'paso': 9}) == 0.0          # sesion nueva
    assert tablero.contribucion_beta(1.2, {**a, 'paso': 5, 'beta': 0.0, 'bloque': 'B'}, {**a, 'paso': 4, 'bloque': 'A'}) == 0.0
    # el tablero
    t = tablero.Tablero(flechas=True)
    sin = tablero.Tablero()
    try:
        t.timer.stop(); sin.timer.stop()
        assert len(t.p) == 6 and len(sin.p) == 5 and not hasattr(sin, 'lbl_flecha')        # apagado por defecto
        betas = [(0.0, True), (0.5, True), (0.3, True), (0.3, False), (-0.2, True)]
        for i, (b, det) in enumerate(betas, 1):
            t._procesar({**paso, 'paso': i, 'beta': b, 'detectado': det})
        t._dibujar()
        assert np.allclose(list(t.d['db']), [0.0, 0.5, -0.2, 0.0, -0.5]), list(t.d['db'])
        assert 'RELAJAR' in t.lbl_flecha.text() and '-0.50' in t.lbl_flecha.text(), t.lbl_flecha.text()
        assert len(t.c_puntas.data) == 3                                                    # 3 flechas: los pasos con cambio
        # control causal ciego: sin flechas hasta revelar
        t._procesar({'tipo': 'bloque_sham', 'letra': 'A', 'nombre': 'sham', 'pasos': 80, 'fuente': 'nula'})
        t._procesar({**paso, 'paso': 6, 'beta': -0.3, 'detectado': False, 'bloque': 'A'})
        t._dibujar()
        assert 'oculto' in t.p[5].titleLabel.text and len(t.c_puntas.data) == 0
        t.btn_sham.setChecked(True)
        t._dibujar()
        assert 'oculto' not in t.p[5].titleLabel.text and len(t.c_puntas.data) > 0
    finally:
        t.close(); sin.close()
    return 'flechas con tamano y direccion; ciego en el control causal; apagado por defecto'


@prueba
def tablero_salud():
    """El tablero (sin pantalla) pinta los cuatro semaforos y PAUSA_SEGURA en rojo con el electrodo."""
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from pyqtgraph.Qt import QtWidgets
    import tablero
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    C = tablero.COLORES_SALUD
    t = tablero.Tablero()
    try:
        t.timer.stop()
        assert set(t.semaforos) == {'eeg', 'ortesis', 'detector', 'piloto'}
        assert C[config.CALENTANDO] in t.semaforos['detector'].styleSheet()      # gris al arrancar
        assert C[config.CALENTANDO] in t.semaforos['piloto'].styleSheet()
        t._procesar({'tipo': 'salud', 'estado': 'PAUSA_SEGURA', 'motivo': 'canal', 'escalon': 3,
                     'colores': {'eeg': 'ROJO', 'ortesis': 'VERDE', 'reloj': 'VERDE', 'detector': 'AMARILLO'},
                     'detalle': {'eeg': 'C3 plano', 'ortesis': '', 'reloj': '', 'detector': 'fiabilidad 0.60'}})
        assert C['ROJO'] in t.semaforos['eeg'].styleSheet()
        assert C['VERDE'] in t.semaforos['ortesis'].styleSheet()
        assert C['AMARILLO'] in t.semaforos['detector'].styleSheet()
        txt = t.lbl_estado.text()
        assert 'PAUSA SEGURA' in txt and 'canal' in txt and 'C3 plano' in txt, txt
        assert C['ROJO'] in t.lbl_estado.styleSheet()
        assert 'C3 plano' in t.semaforos['eeg'].toolTip()
        paso = {'tipo': 'paso', 'paso': 1, 'estado': 'LAZO_ADAPTATIVO', 'meta': 1, 'angulo': 0.5,
                'p_crudo': 0.6, 'b': 0.5, 'p_prima': 0.6, 'P_hat': None, 'error': 0, 'error_sombra': 0,
                'beta': 0.1, 'sd_beta': 0.5, 'youden': 0.6, 'fiabilidad': 1.0, 'congelado': False,
                'cambio': '', 'latencia_ms': None, 'perturbado': False, 'excluido': 'sin_ack',
                'salud': {'eeg': 'VERDE', 'ortesis': 'AMARILLO', 'reloj': 'VERDE', 'detector': 'CALENTANDO'}}
        t._procesar(paso)
        assert 'sin ACK' in t.lbl_estado.text() and 'PAUSA' not in t.lbl_estado.text()
        assert C['VERDE'] in t.semaforos['eeg'].styleSheet()
        assert C['AMARILLO'] in t.semaforos['ortesis'].styleSheet()
        assert C[config.CALENTANDO] in t.semaforos['detector'].styleSheet()
        viejo = {k: v for k, v in paso.items() if k not in ('salud', 'excluido')}   # sesion grabada antigua
        t._procesar({**viejo, 'paso': 2, 'latencia_ms': 8.0})
        t._dibujar()
        # Tarea 2: aviso del movimiento ajeno, la meta de vuelta tras el y el IIC (exploratorio)
        t._procesar({'tipo': 'cue', 'meta': 1})
        t._procesar({'tipo': 'aviso_ajeno', 'meta': 1})
        assert t.lbl_cue.text() == 'AUTOMATICO'
        r = {'iic': 0.42, 'ic': [0.05, 0.8], 'n_propios': 80, 'n_ajenos': 12, 'tendencia': None, 'ic_tendencia': None}
        t._procesar({'tipo': 'ajeno', 'paso': 3, 'estado': 'LAZO_ADAPTATIVO', 'meta': 1, 'angulo': 0.65,
                     'iic': r, 'salud': paso['salud']})
        assert t.lbl_cue.text() == 'CERRAR'
        assert '+0.42' in t.lbl_iic.text() and 'exploratorio' in t.lbl_iic.text() and '12' in t.lbl_iic.text()
        t._procesar({**paso, 'paso': 4, 'iic': {**r, 'iic': None, 'ic': None}})
        assert 'sin estimar' in t.lbl_iic.text()
        # control causal: ciego hasta que el operador pulsa 'Revelar bloques'
        assert t.btn_sham.isHidden() and t.lbl_sham.text() == ''
        t._procesar({'tipo': 'bloque_sham', 'letra': 'A', 'nombre': 'sham', 'pasos': 80, 'fuente': 'nula'})
        assert 'bloque A' in t.lbl_sham.text() and 'SHAM' not in t.lbl_sham.text() and not t.btn_sham.isHidden()
        b = lambda err, pasos: {'agente': err, 'sombra': 0.5, 'ic_agente': [err - 0.1, err + 0.1],
                                'ic_sombra': [0.4, 0.6], 'pasos': pasos, 'seg': pasos and pasos * 2.1}
        t._procesar({'tipo': 'sham', 'orden': ['sham', 'real'], 'fuente': 'nula', 'dif': 0.14, 'ic_dif': [0.02, 0.26],
                     'solo_real': True, 'real': b(0.33, 26), 'sham': b(0.47, None)})
        txt = t.lbl_sham.text()
        assert 'Bloque A: error 0.47' in txt and 'NO se recupero' in txt and '26 pasos' in txt, txt
        assert 'SHAM' not in txt and 'REAL' not in txt and 'sham - real' not in txt, txt
        t.btn_sham.setChecked(True)
        txt = t.lbl_sham.text()
        assert 'Bloque A = SHAM' in txt and 'Bloque B = REAL' in txt and 'sham - real +0.14' in txt, txt
        assert not hasattr(t, 'caja')                             # la caja del copiloto solo con --copiloto
        assert t.lbl_narrador.isHidden()                          # la franja del narrador solo con --narrador
        t._frase({'texto': 'El agente se recuperó en 26 pasos.', 'evento': 'recuperacion', 'origen': 'plantilla'})
        assert '26 pasos' in t.lbl_narrador.text()
        # co-investigador: la propuesta con Aprobar y Rechazar; el boton escribe la decision en el registro
        import copiloto
        csv_prueba = config.RESULTADOS / 'sesion_sim_prueba_tablero.csv'
        copiloto.ruta_propuestas(csv_prueba).unlink(missing_ok=True)
        prop = {'accion': 'ajustar_parametro', 'parametro': 'paso_visible', 'valor': 0.1, 'justificacion': 'pasos chicos sin ErrP'}
        assert t.btn_aprobar.isHidden() and t.btn_rechazar.isHidden()
        t._procesar({'tipo': 'propuesta', 'id': 'tras:estatico#1', 'csv': csv_prueba.name, 'bloque': 'estatico',
                     'propuesta': prop, 'origen': 'api', 'valida': True, 'espera_s': 60.0})
        assert 'AJUSTAR PARAMETRO paso_visible = 0.1' in t.lbl_propuesta.text() and 'Claude' in t.lbl_propuesta.text()
        assert not t.btn_aprobar.isHidden() and not t.btn_rechazar.isHidden()
        t.btn_aprobar.click()
        reg = copiloto.leer_propuestas(csv_prueba)
        assert reg[-1]['decision'] == 'aprobada' and reg[-1]['id'] == 'tras:estatico#1' and t.btn_aprobar.isHidden()
        t._procesar({'tipo': 'decision', 'id': 'tras:estatico#1', 'decision': 'aprobada', 'efecto': 'paso_visible: 0.08 -> 0.1'})
        assert 'APROBADA: paso_visible: 0.08 -> 0.1' in t.lbl_propuesta.text()
        t._procesar({'tipo': 'propuesta', 'id': 'x#2', 'csv': csv_prueba.name, 'bloque': 'adaptativo', 'propuesta': prop,
                     'origen': 'reglas', 'valida': True, 'espera_s': 0.0})          # ultimo bloque: sin botones
        assert t.btn_aprobar.isHidden() and 'reglas' in t.lbl_propuesta.text()
        copiloto.ruta_propuestas(csv_prueba).unlink()
        t2 = tablero.Tablero(copiloto=True)
        try:
            t2.timer.stop()
            t2._responder('cuantos pasos se excluyeron y por que?')   # lo que corre el otro hilo
            assert t2.lbl_copiloto.text() == ''
            t2._mostrar_respuesta()
            assert 'sesion_' in t2.lbl_copiloto.text() and ('excluyeron' in t2.lbl_copiloto.text()
                                                            or 'no pudo responder' in t2.lbl_copiloto.text())
        finally:
            t2.close()
    finally:
        t.close()
    return 'cuatro semaforos (gris al calentar), PAUSA SEGURA en rojo con el electrodo y la causa; IIC y AUTOMATICO'


# ------------------------------------------------------------ caos
@prueba
def plan_caos():
    from caos import PlanCaos
    a, b, c = PlanCaos(7), PlanCaos(7), PlanCaos(8)
    ts = np.arange(0, 600, 0.1)
    for tipo in ('corte_eeg', 'rafaga_parpadeos', 'canal', 'perdida_bt'):
        ea = [a.activo(tipo, t) for t in ts]
        assert ea == [b.activo(tipo, t) for t in ts[::-1]][::-1], tipo      # no depende del orden de consulta
        ev = {e for e in ea if e}
        assert 2 <= len(ev) <= 40, (tipo, len(ev))
        lo, hi = config.CAOS_ESTANDAR[tipo]['duracion_s']
        assert all(lo <= e[1] <= hi for e in ev), tipo
    canales = {e[2:] for e in (a.activo('canal', t) for t in ts) if e}
    assert all(c_ in config.CANALES_EEG and m in ('plano', 'ruidoso') for c_, m in canales), canales
    perdidos = [a.por_paso('ack_perdido', s) for s in range(2000)]
    assert perdidos == [b.por_paso('ack_perdido', s) for s in range(2000)]
    assert 0.015 < np.mean(perdidos) < 0.05, np.mean(perdidos)
    picos = [p for p in (a.por_paso('pico_latencia', s) for s in range(2000)) if p]
    assert all(80 <= p <= 300 for p in picos) and 0.03 < len(picos) / 2000 < 0.08
    assert [c.activo('corte_eeg', t) for t in ts] != [a.activo('corte_eeg', t) for t in ts]
    # caos leve: del orden de una falla cada 2 a 3 minutos, sumando todos los tipos
    fallas, horas = 0, 0
    for semilla in range(20):
        leve = PlanCaos(semilla, config.CAOS['leve'])
        for tipo in ('corte_eeg', 'rafaga_parpadeos', 'canal', 'perdida_bt'):
            leve.activo(tipo, 3600.0)
            fallas += sum(1 for e in leve._lineas[tipo]['ev'] if e[0] < 3600.0)
        pasos = int(3600 / config.CICLO_S)
        fallas += sum(leve.por_paso('ack_perdido', s) for s in range(pasos))
        fallas += sum(leve.por_paso('pico_latencia', s) is not None for s in range(pasos))
        horas += 1
    cada_min = 60.0 * horas / fallas
    assert 2.0 <= cada_min <= 3.0, cada_min
    assert config.CAOS['estandar'] is config.CAOS_ESTANDAR
    return f'reproducible por semilla e independiente del orden; caos leve: una falla cada {cada_min:.1f} min'


def _sesion_caos(semilla, caos, nivel='estandar'):
    import orquestador
    argv = ['sim', '--ciclo', '0', '--semilla', str(semilla), '--pasos_estatico', '60',
            '--pasos_adaptativo', '300', '--caos-nivel', nivel] + (['--caos', str(caos)] if caos is not None else [])
    a = orquestador.argumentos(argv)
    orq = orquestador.Orquestador(_backend_con_fallas(orquestador, a, {}), a)
    orquestador.correr(orq, a)
    return orq


@prueba
def caos_sim():
    """Sesion simulada con el caos estandar: termina sin excepcion, ninguna epoca tomada
    durante un corte de EEG llega al agente y el agente no aprende en PAUSA_SEGURA."""
    orq = _sesion_caos(0, caos=1)
    assert orq.fsm.estado == 'EVALUACION' and orq.f_csv.closed
    por_seq = {f['seq']: f for f in orq.filas if f['seq'] != ''}
    assert orq.b.epocas_en_corte, 'el caos estandar debe producir al menos una epoca en un corte'
    for seq in orq.b.epocas_en_corte:
        f = por_seq[seq]
        assert f['excluido'] == 'epoca_invalida' and f['P_hat'] == '', f     # sin P_hat: el agente no la vio
    n_pausas = 0
    for i, f in enumerate(orq.filas[1:], 1):
        n_pausas += f['estado'] == 'PAUSA_SEGURA'
        if f['excluido']:
            assert f['beta'] == orq.filas[i - 1]['beta'], (i, f)
    assert n_pausas >= 3, n_pausas
    assert sum(1 for f in orq.filas if f['estado'] != 'PAUSA_SEGURA') == 360
    assert set(orq.excluidos) <= set(config.MOTIVOS_EXCLUSION), orq.excluidos
    # perdidas de Bluetooth: las epocas que las cruzan quedan invalidas; la ventana de MI las tolera
    assert orq.b.epocas_en_perdida, 'el caos estandar debe producir epocas cruzadas por una perdida de Bluetooth'
    for seq in orq.b.epocas_en_perdida:
        assert por_seq[seq]['excluido'] == 'epoca_invalida', por_seq[seq]
    # el nivel leve excluye mucho menos (semilla 3: con la 1 sus pocas fallas caen en pasos sin epoca)
    leve = _sesion_caos(0, caos=3, nivel='leve')
    por_fallas = lambda o: sum(n for m, n in o.excluidos.items() if m != 'ajeno')   # los ajenos no son fallas
    assert 0 < por_fallas(leve) < por_fallas(orq) / 3, (leve.excluidos, orq.excluidos)
    return f'360 pasos y {n_pausas} pausas sin excepcion; excluidos {orq.excluidos}'


@prueba
def caos_agente_vs_sombra():
    """EXPLORATORIO, sobre el simulador (no es una persona): con el caos estandar el
    agente sigue claramente por debajo de la sombra tras la perturbacion."""
    ag, so, exc = [], [], {}
    for s in range(12):
        orq = _sesion_caos(s, caos=100 + s)
        ag.append(orq.error_post['agente']); so.append(orq.error_post['sombra'])
        for k, v in orq.excluidos.items():
            exc[k] = exc.get(k, 0) + v
    assert np.mean(ag) < np.mean(so) - 0.05, (np.mean(ag), np.mean(so))
    return (f'simulador, 12 sujetos, ~2 min tras perturbar: agente {np.mean(ag):.2f} vs sombra '
            f'{np.mean(so):.2f}; excluidos por motivo {exc}')


# ------------------------------------------------------------ hardware (sin hardware)
@prueba
def modelos_hardware():
    import hardware as hw
    rng = np.random.default_rng(0)
    ch = 8
    # decoder + recentrado riemanniano ante un cambio de mezcla de canales
    y = np.repeat([0, 1], 40)
    X = rng.normal(size=(80, ch, 500)); X[y == 1, 2] *= 2.0
    dec = hw.DecoderIM(paso_recentrado=0.05).ajustar(X, y)
    A = np.eye(ch) + 0.6 * rng.normal(size=(ch, ch))
    yt = np.tile([0, 1], 60)
    Xt = rng.normal(size=(120, ch, 500)); Xt[yt == 1, 2] *= 2.0
    Xt = np.einsum('ij,njt->nit', A, Xt)
    acc = np.mean([(dec.w0 @ dec.phi(x) + dec.c0 > 0) == t for x, t in zip(Xt, yt)][60:])
    assert dec.ba > 0.8 and acc > 0.85, (dec.ba, acc)
    # detector de ErrP calibrado + rareza
    t = np.arange(250) / 250
    erp = 8 * np.exp(-((t - 0.5) ** 2) / 0.005)
    Y = np.repeat([0, 1], 40)
    E = rng.normal(0, 3, size=(80, ch, 250)); E[Y == 1, 6] += erp
    det = hw.DetectorErrP().ajustar(E, Y)
    raro = E[0].copy(); raro[3] *= 6
    assert det.ba > 0.8 and det.artefacto(raro) and not det.artefacto(E[1]), det.ba
    # una ventana con NaN no produce rasgos ni mueve el centro del recentrado
    M0 = dec.M.copy()
    assert dec.phi(np.full((ch, 500), np.nan)) is None and np.array_equal(dec.M, M0)
    lo, hi = hw.intervalo_ba(Y, det.pred_cv)
    assert lo <= det.ba <= hi
    o = hw.OrtesisSimulada()
    lat = np.array([o.mover(0.3 + 0.04 * k)[2] for k in range(10)])     # latencia del ACK en ms (tercer valor)
    media, sd = float(lat.mean()), float(lat.std())
    assert 3 < media < 20
    return (f'recentrado: {acc:.0%} tras mezclar canales; ErrP BA {det.ba:.2f}, rareza ok; '
            f'ACK {media:.1f}+-{sd:.1f} ms')


@prueba
def decoder_preentrenado():
    """El decoder de MI puede arrancar de uno pre-entrenado con OTROS sujetos (DecoderIM.ajustar_desde):
    sin ensayos propios ya decide (el piloto solo aporta su centro, de EEG sin etiquetas), y con pocos
    ensayos propios no queda peor que calibrar desde cero. Aqui los otros sujetos son del gemelo, asi
    que solo se prueba el mecanismo; lo medido con personas esta en estudios/transferencia_physionet.py."""
    import os
    import cerebro_sintetico as cs
    import hardware as hw
    import orquestador
    sys.path.insert(0, str(config.RAIZ / 'estudios'))
    import transferencia_physionet as tp
    Z, y = zip(*[(tp.tangente(X)[0], yy) for X, yy in (cs.sesion_mi(40, semilla=s) for s in range(1, 6))])
    pre = {'Z': np.vstack(Z).astype(np.float32), 'y': np.concatenate(y), 'canales': config.CANALES_EEG, 'personas': 5,
           'origen': 'gemelo (prueba)'}
    cero, propio, afinado = [], [], []
    for s in (9, 10, 11, 12):
        X, yy = cs.sesion_mi(100, semilla=s)
        Xc, yc, Xp, yp = X[:16], yy[:16], X[40:], yy[40:]
        ba = lambda d: hw.exactitud_balanceada(yp, np.array([int(d.w0 @ d.phi(x, actualizar_centro=False) + d.c0 > 0) for x in Xp]))
        d0 = hw.DecoderIM().ajustar_desde(pre, reposo=X[:20])                      # ni una etiqueta del sujeto
        assert d0.pred_cv is None and np.isnan(d0.ba) and d0.w0.shape == (36,) and d0.eleccion == 'pre-entrenado'
        d16 = hw.DecoderIM().ajustar_desde(pre, Xc, yc)
        assert len(d16.pred_cv) == 16 and 0 <= d16.ba <= 1 and d16.phi(np.full_like(Xp[0], np.nan)) is None
        cero.append(ba(d0)); afinado.append(ba(d16)); propio.append(ba(hw.DecoderIM().ajustar(Xc, yc, config.candidatos('decoder'))))
    assert np.mean(cero) > 0.75 and np.mean(afinado) > np.mean(propio) - 0.03, (cero, propio, afinado)
    # la calibracion real lo usa con --preentrenado; el modelo se guarda y se carga como los demas
    assert orquestador.argumentos(['real', '--preentrenado']).preentrenado and not orquestador.argumentos(['real']).preentrenado
    nombre = 'decoder_preentrenado_prueba.pkl'
    hw.guardar(pre, nombre)
    assert hw.cargar(nombre)['personas'] == 5
    os.remove(config.MODELOS / nombre)
    return (f'gemelo, 4 sujetos nuevos: sin ensayos propios BA {np.mean(cero):.2f}; con 16 ensayos, desde cero '
            f'{np.mean(propio):.2f} y pre-entrenado + propios {np.mean(afinado):.2f}')


@prueba
def detector_umbral_anidado():
    """El umbral de Neyman-Pearson se elige dentro de cada pliegue (validacion anidada): la BA
    que reporta la calibracion coincide con la de epocas nuevas. Verifica que no hay sesgo; con
    120 epocas el sesgo del umbral elegido sobre los mismos puntajes era chico (+0.03): el grande
    venia de la parada secuencial (ver calibracion_errp_fija)."""
    import hardware as hw
    rng = np.random.default_rng(1)
    t = np.arange(250) / 250
    erp = 2.2 * np.exp(-((t - 0.5) ** 2) / 0.005)                # senal debil: BA real ~0.7

    def epocas(n):
        y = (rng.random(n) < 0.3).astype(int)
        E = rng.normal(0, 3, size=(n, 8, 250))
        E[y == 1, 6] += erp
        E[y == 1, 2] += 0.6 * erp
        return E, y
    difs = []
    for _ in range(2):
        E, y = epocas(120)
        det = hw.DetectorErrP().ajustar(E, y)
        Et, yt = epocas(600)
        p = np.array([det.p_error(e) for e in Et]) > det.umbral
        difs.append(det.ba - 0.5 * (p[yt == 1].mean() + 1 - p[yt == 0].mean()))
        assert len(det.pred_cv) == len(y)
    assert abs(np.mean(difs)) < 0.06, difs
    return f'BA reportada menos BA real en epocas nuevas: {np.mean(difs):+.3f}'


@prueba
def seleccion_canales_vistas():
    """La calibracion elige por validacion cruzada (anidada para reportar) entre los canales del
    papel y los 8, y entre dos y tres vistas del detector (la tercera: potencia theta en Fz y Cz),
    y registra la eleccion. Datos sinteticos donde se sabe cual deberia ganar."""
    import hardware as hw
    rng = np.random.default_rng(3)
    todos = list(range(8))
    # decoder: la diferencia entre clases esta en Oz, fuera de C3/Cz/C4 -> deben ganar los 8
    y = np.repeat([0, 1], 30)
    X = rng.normal(size=(60, 8, 500)); X[y == 1, 6] *= 2.0
    dec = hw.DecoderIM().ajustar(X, y, candidatos={'C3/Cz/C4': config.indices('mi'), '8 canales': todos})
    assert dec.eleccion == '8 canales' and dec.ba > 0.8, (dec.eleccion, dec.ba, dec.puntajes)
    assert dec.phi(X[0]).shape == dec.w0.shape
    # decoder: diferencia en C3 -> los 3 motores bastan (y gana el de menos canales si empatan)
    X2 = rng.normal(size=(60, 8, 500)); X2[y == 1, 1] *= 2.0
    dec2 = hw.DecoderIM().ajustar(X2, y, candidatos={'C3/Cz/C4': config.indices('mi'), '8 canales': todos})
    assert dec2.ba > 0.8 and dec2.canales == (config.indices('mi') if dec2.eleccion == 'C3/Cz/C4' else todos)
    # detector: el error solo deja un estallido theta sin fase fija (el promedio no lo ve) -> tres vistas
    t = np.arange(250) / 250
    ye = (rng.random(100) < 0.35).astype(int)
    E = rng.normal(0, 3, size=(100, 8, 250))
    for i in np.flatnonzero(ye):
        burst = 6.0 * np.exp(-((t - 0.6) ** 2) / (2 * 0.1 ** 2)) * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2 * np.pi))
        E[i, config.indices(['Fz', 'Cz'])] += burst
    cand = {'dos vistas': (todos, 'dos'), 'tres vistas': (todos, 'tres')}
    det = hw.DetectorErrP().ajustar(E, ye, candidatos=cand)
    assert det.eleccion == 'tres vistas' and det.ba > 0.7, (det.eleccion, det.ba, det.puntajes)
    assert 0 <= det.p_error(E[0]) <= 1 and set(det.puntajes) == set(cand)
    return (f"decoder: {dec.eleccion} (BA {dec.ba:.2f}) y {dec2.eleccion} (BA {dec2.ba:.2f}); "
            f"detector con theta: {det.eleccion} (BA {det.ba:.2f})")


@prueba
def detector_coadaptativo():
    """Detector co-adaptativo con el gemelo: cada epoca valida del lazo se puntua con el modelo
    vigente ANTES de usarse para entrenar (evaluacion secuencial honesta); cada 20 epocas nuevas
    se re-entrena en otro hilo; el modelo nuevo se prueba en sombra y solo reemplaza al vigente
    si no empeora. La BA secuencial crece con las epocas del lazo y el ciclo no se bloquea."""
    import hardware as hw
    import cerebro_sintetico as cs
    X, y = cs.sesion_errp(340, semilla=4)
    det = hw.DetectorErrP().ajustar(X[:40], y[:40])               # calibracion corta: hay de donde mejorar
    cambios = []
    co = hw.DetectorCoadaptativo(det, X[:40], y[:40], cada=20, prueba=config.COADAPTAR_PRUEBA, al_cambiar=cambios.append)
    lento = 0.0
    for k, (e, err) in enumerate(zip(X[40:], y[40:])):
        t0 = time.perf_counter()
        p = co.actual.p_error(e)
        co.observar(e, bool(err), p, artefacto=False)            # nunca espera al entrenamiento
        lento = max(lento, time.perf_counter() - t0)
        if k % 20 == 19:
            co.esperar()       # en el lazo real 20 epocas son ~18 s, mas de lo que tarda re-entrenar
    co.esperar()
    primeras, ultimas = co.ba_secuencial(desde=0, hasta=100), co.ba_secuencial(desde=-100)
    assert ultimas > primeras + 0.03, (primeras, ultimas, co.version, co.descartes)
    assert co.version > 1 and cambios and cambios[-1] is co.actual and lento < 0.25, (co.version, lento)
    # un candidato peor se descarta
    malo = hw.DetectorCoadaptativo(det, X[:40], y[:40], cada=20, prueba=20)
    malo._entrenar = lambda Xs, ys: hw.DetectorErrP().ajustar(np.array(Xs), np.random.default_rng(0).permutation(ys))
    for k, (e, err) in enumerate(zip(X[40:200], y[40:200])):
        malo.observar(e, bool(err), malo.actual.p_error(e), artefacto=False)
        if k % 20 == 19:
            malo.esperar()
    malo.esperar()
    assert malo.descartes >= 1 and malo.actual is det, (malo.descartes, malo.version)
    return (f'BA secuencial en vivo: {primeras:.2f} en las primeras 100 epocas del lazo, {ultimas:.2f} en las ultimas 100; '
            f'{co.version - 1} cambios de modelo, {co.descartes} descartados; paso mas lento {1000 * lento:.0f} ms')


@prueba
def coadaptativo_no_detiene_el_lazo():
    """Nada de lo que falle en el detector co-adaptativo detiene el lazo: un re-entrenamiento que
    lanza una excepcion, un candidato que falla al puntuar o un aviso de cambio que falla quedan
    registrados (co.errores) y el lazo sigue con el detector vigente; con 3 fallos seguidos la
    co-adaptacion se apaga sola. Cerrar no espera para siempre a un re-entrenamiento colgado.
    --sin-coadaptativo la apaga desde el principio."""
    import types
    import hardware as hw
    import orquestador

    class Falso:                                              # detector de mentira, perfecto
        umbral, canales, vistas, sens, espec, p_error_cal = 0.5, [0], 'dos', 0.7, 0.9, 0.3

        def p_error(self, e):
            return float(e[0])

    class Roto(Falso):
        def p_error(self, e):
            raise ValueError('modelo corrupto')
    epoca = lambda err: np.array([0.8 if err else 0.2])
    det = Falso()

    def lazo(co, n):
        for k in range(n):                                    # como el lazo: observar nunca lanza
            err = k % 3 == 0
            co.observar(epoca(err), err, co.actual.p_error(epoca(err)))
            co.esperar()
        return co
    nuevo = lambda **k: hw.DetectorCoadaptativo(det, [epoca(0), epoca(1)], [0, 1], cada=10, prueba=6, **k)
    # 1) el re-entrenamiento lanza una excepcion; a la tercera seguida se apaga
    co = nuevo()

    def revienta(X, y):
        raise RuntimeError('sin memoria')
    co._entrenar = revienta
    lazo(co, 25)
    assert co.actual is det and co.activo and len(co.errores) == 2 and 'sin memoria' in co.errores[0], co.errores
    lazo(co, 35)
    assert len(co.errores) == 3 and not co.activo and co.actual is det, (co.errores, co.activo)
    assert len(co.historial) == 60                            # la BA en vivo se sigue midiendo
    # 2) el candidato falla al puntuar una epoca en la prueba en sombra
    co = nuevo()
    co._entrenar = lambda X, y: Roto()
    lazo(co, 15)
    assert co.actual is det and co.candidato is None and 'modelo corrupto' in co.errores[0], co.errores
    # 3) el aviso de cambio (umbral, agente, confianza) falla: el cambio de modelo se deshace
    def aviso_roto(d):
        if d is not det:
            raise KeyError('umbral')
    co = nuevo(al_cambiar=aviso_roto)
    co._entrenar = lambda X, y: Falso()
    lazo(co, 20)
    assert co.actual is det and co.version == 1 and 'umbral' in co.errores[0], (co.version, co.errores)
    # 4) un re-entrenamiento colgado no bloquea el cierre
    co = nuevo()
    co._entrenar = lambda X, y: time.sleep(1.5) or Falso()
    for k in range(10):
        co.observar(epoca(k % 3 == 0), k % 3 == 0, 0.2)
    t0 = time.perf_counter()
    co.esperar(timeout=0.05)
    assert time.perf_counter() - t0 < 0.5
    # 5) el interruptor
    a = orquestador.argumentos(['real', '--sin-coadaptativo'])
    assert a.sin_coadaptativo and not orquestador.argumentos(['real']).sin_coadaptativo
    b = orquestador.BackendReal.__new__(orquestador.BackendReal)
    b.a, b.hw, b.detector = a, hw, det
    b.preparar_coadaptacion(types.SimpleNamespace(detector_cambiado=None))
    assert b.coadapta is None
    return 'excepcion al entrenar, candidato roto y aviso roto: registrados y el lazo sigue; 3 fallos seguidos lo apagan'


@prueba
def inicio_movimiento():
    """La epoca del ErrP se alinea al inicio REAL del movimiento: la telemetria T del ESP32 dice
    cuando el angulo empieza a cambiar (interpolando entre muestras), y su reloj se convierte al de
    la PC con los pares (t_us del ACK, hora de llegada). Sin telemetria, ACK + latencia media."""
    import hardware as hw
    import verificar_ortesis as vo
    o = hw.OrtesisSimulada(timeout_ack=0.0)                  # latencia mecanica de 30 a 150 ms
    errores, alineaciones = [], set()
    for k in range(10):                                       # como en el lazo: un movimiento termina
        time.sleep(0.45)                                      # antes de mandar el siguiente
        seq, t_ack, _ = o.mover(0.3 if k % 2 else 0.7)
        t0, como = o.inicio_movimiento(seq, t_ack, espera_s=0.0)
        verdad = t_ack + hw.latencia_mecanica_simulada(seq)
        errores.append(t0 - verdad); alineaciones.add(como)
    assert alineaciones == {'telemetria'} and np.abs(errores).max() < 0.012, (alineaciones, np.abs(errores).max())
    assert 0.03 <= np.median(o.latencias_mecanicas) <= 0.15
    # sin telemetria: ACK mas la latencia media medida (o el ACK solo, si no hay ninguna medida)
    sin = hw.OrtesisSimulada(timeout_ack=0.0, telemetria=False)
    seq, t_ack, _ = sin.mover(0.6)
    assert sin.inicio_movimiento(seq, t_ack, espera_s=0.0) == (t_ack, 'ack')
    o.con_telemetria = False                                  # el ESP32 deja de mandar telemetria
    time.sleep(0.45)
    seq, t_ack, _ = o.mover(0.2)
    t0, como = o.inicio_movimiento(seq, t_ack, espera_s=0.0)
    assert como == 'ack+latencia' and abs(t0 - t_ack - np.mean(o.latencias_mecanicas)) < 1e-9
    # el CSV dice que alineacion se uso en cada paso
    assert 'alineacion' in config.COLUMNAS_CSV
    # verificar_ortesis.py mide la latencia mecanica para el domingo
    r = vo.medir(hw.OrtesisSimulada(timeout_ack=0.0), movimientos=6, pausa_s=0.45, salida=lambda *a: None)
    assert r['detectados'] == 6 and 30 <= r['mediana_ms'] <= 150 and r['veredicto'].startswith('OK'), r
    return (f'inicio por telemetria con error maximo {1000 * np.abs(errores).max():.1f} ms; latencia mecanica '
            f'mediana {1000 * np.median(o.latencias_mecanicas):.0f} ms; sin telemetria: ACK + latencia media')


@prueba
def rechazo_por_cabeza():
    """Movimiento de cabeza (giroscopio): la epoca del ErrP y la ventana de MI con movimiento se
    marcan como artefacto (el agente no aprende de ese paso y el decoder no se recentra), sin
    pausa; en calibracion, el ensayo se repite."""
    import types
    import orquestador
    import hardware as hw
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 30)
    X = rng.normal(size=(60, 8, 500)); X[y == 1, 1] *= 2.0
    dec = hw.DecoderIM().ajustar(X, y)
    det = hw.DetectorErrP().ajustar(rng.normal(0, 3, (60, 8, 250)), np.r_[np.ones(20, int), np.zeros(40, int)])

    class EEG:
        giro = 0.0
        fs = 250.0

        def movimiento(self, t0, t1):
            return self.giro

        def epoca(self, t0):
            return rng.normal(0, 3, (8, 250))

        def ventana(self, segundos, *a):
            return rng.normal(size=(8, int(segundos * 250))), None

        def ultimo_t(self):
            return 100.0

        def lecturas(self):
            return {'canales': {}}
    b = orquestador.BackendReal.__new__(orquestador.BackendReal)
    b.hw, b.eeg, b.decoder, b.detector, b.coadapta = hw, EEG(), dec, det, None
    p, art, exc = b.errp(1, 100.0, False, 0.1)
    assert not art and exc == ''
    b.eeg.giro = 80.0                                        # asiente con la cabeza
    p, art, exc = b.errp(2, 100.0, False, 0.1)
    assert art and exc == '', (art, exc)
    M0 = dec.M.copy()
    phi = b.phi(1)
    assert phi is not None and np.array_equal(dec.M, M0) and b.mov_mi   # decide, pero no se recentra
    b.eeg.giro = 0.0
    b.phi(1)
    assert not np.array_equal(dec.M, M0) and not b.mov_mi
    b.eeg.giro = 80.0
    assert b._cabeza_movida(99.0, 100.0) and 'cabeza' in b.falla
    return f'con giro de 80 grados/s: epoca marcada como artefacto y sin recentrado (umbral {config.GIRO_ARTEFACTO_DPS:.0f} grados/s)'


@prueba
def cierre_completo():
    """La ortesis cierra (y abre) completa: cada ensayo empieza en el punto medio y el paso maximo
    es 0.30. Antes cerraba completa solo en el 27 % de los ensayos de "cerrar" (simulador)."""
    import orquestador
    a = orquestador.argumentos(['sim', '--ciclo', '0', '--semilla', '2', '--pasos_estatico', '60',
                                '--pasos_adaptativo', '200', '--sin_perturbacion'])
    orq = orquestador.Orquestador(orquestador.BackendSim(a), a)
    marcas, publicar = [], orq.salidas.marcador
    orq.salidas.marcador = lambda txt, *r: (marcas.append(txt), publicar(txt, *r))[1]
    orquestador.correr(orq, a)
    lazo = [f for f in orq.filas if f['estado'] != 'PAUSA_SEGURA']
    fin = [(lazo[i]['meta'], lazo[i]['angulo']) for i in range(config.PASOS_ENSAYO - 1, len(lazo), config.PASOS_ENSAYO)]
    cierra = np.mean([ang >= 0.95 for m, ang in fin if m == 1])
    abre = np.mean([ang <= 0.05 for m, ang in fin if m == -1])
    assert config.PASO_MAX == 0.30 and config.CENTRAR_ENSAYO
    assert cierra > 0.5 and abre > 0.5, (cierra, abre)
    # un centrado (con su marcador) antes de cada cue, y ninguno a medio ensayo
    assert marcas.count(config.CENTRADO) == len(fin), (marcas.count(config.CENTRADO), len(fin))
    assert all(marcas[i + 1] in (config.CUE_CERRAR, config.CUE_RELAJA)
               for i, m in enumerate(marcas) if m == config.CENTRADO)
    return f'ensayos que terminan cerrados del todo: {cierra:.0%}; abiertos del todo: {abre:.0%}'


@prueba
def paso_sin_movimiento():
    """Un paso que no mueve la ortesis (ya estaba en el tope) no informa: nadie ve nada, asi que
    no hay ErrP que leer. El agente no aprende de el, no cuenta como deteccion fallida para la
    confianza del detector ni entra al IIC, y el gemelo no reacciona. Sigue contando como
    decision (acierto o error) en el analisis."""
    import orquestador
    import cerebro_sintetico as cs
    a = orquestador.argumentos(['sim', '--ciclo', '0', '--semilla', '2', '--pasos_estatico', '40',
                                '--pasos_adaptativo', '200'])
    orq = orquestador.Orquestador(orquestador.BackendSim(a), a)
    previa = config.PERTURBACION_LOGITS
    config.PERTURBACION_LOGITS = 6.0     # perturbacion enorme: la ortesis se va al tope EQUIVOCADO y ahi se queda
    try:
        orquestador.correr(orq, a)
    finally:
        config.PERTURBACION_LOGITS = previa
    f = orq.filas
    quietos = [i for i, x in enumerate(f) if x['alineacion'] == config.SIN_MOVIMIENTO]
    assert len(quietos) > 10, len(quietos)
    for i in quietos:
        x, ant = f[i], f[i - 1]
        assert x['angulo'] in (0.0, 1.0), x                       # solo pasa en un tope
        assert x['excluido'] == '' and x['artefacto'] == 1 and x['P_hat'] == '' and x['n1_uv'] == '', x
        assert x['beta'] == ant['beta'], (i, x['beta'], ant['beta'])                 # no aprendio
        assert (x['sens_viva'], x['espec_viva']) == (ant['sens_viva'], ant['espec_viva']), (i, x, ant)
    movidos = [x for x in f if x['alineacion'] != config.SIN_MOVIMIENTO and not x['excluido']]
    assert all(x['P_hat'] != '' or x['artefacto'] == 1 for x in movidos)
    errores_quietos = sum(f[i]['error_verdadero'] for i in quietos)
    assert errores_quietos >= 3, errores_quietos    # el caso que importa: errores que nadie vio
    assert orq.agente.beta > 3.0, orq.agente.beta   # y aun asi aprende, con los pasos que si se ven
    # su marcador es paso_quieto, no paso_ack: quien corte epocas con los marcadores no lo toma
    marc = orq.salidas.marcadores
    for i in quietos:
        assert config.m_paso_quieto(f[i]['seq']) in marc and config.m_paso_ack(f[i]['seq']) not in marc, f[i]
    assert sum(m.startswith('paso_quieto:') for m in marc) == len(quietos)
    assert orq.sin_movimiento == len(quietos)        # EVALUACION dice cuantos fueron
    # el gemelo solo reacciona a paso_ack: un paso que no movio la ortesis no le provoca nada
    cer = cs.Cerebro(cs._args(semilla=0))
    cer.meta, cer.dir_paso = 1, -1
    n = len(cer.eventos)
    cer._marcador(config.m_paso_quieto(3), 5.0)
    assert len(cer.eventos) == n
    cer._marcador(config.m_paso_ack(4), 6.0)
    assert len(cer.eventos) > n
    return (f'{len(quietos)} de {len(f)} pasos sin movimiento ({errores_quietos} eran errores): sin aprendizaje ni '
            f'cuenta para la confianza del detector; el gemelo no reacciona')


@prueba
def iic_estimador():
    """IIC (Tarea 2, exploratorio): tamano de efecto de la N1 a movimientos ajenos contra
    propios correctos, con intervalo bootstrap; sin estimacion con pocas epocas; tendencia
    entre las dos mitades de la sesion; amplitud de la N1 en PO7/Oz/PO8."""
    import embodiment as emb
    rng = np.random.default_rng(0)
    ind = emb.IndiceEmbodiment(semilla=1)
    assert ind.estimar()['iic'] is None
    # uno de cada 10 ajeno; N1 ajena 4 uV y propia atenuada a 2.8 uV, ruido 2 uV: d verdadero 0.6
    for k in range(300):
        ajeno = k % 10 == 1
        ind.observar(k, (4.0 if ajeno else 2.8) + rng.normal(0, 2.0), ajeno=ajeno, correcto=True)
    r = ind.estimar()
    assert (r['n_ajenos'], r['n_propios']) == (30, 270), r
    assert 0 < r['ic'][0] < 0.6 < r['ic'][1], r
    # sin atenuacion el intervalo incluye 0; los propios erroneos y lo no finito no cuentan
    ind0 = emb.IndiceEmbodiment(semilla=1)
    for k in range(300):
        ajeno = k % 10 == 1
        ind0.observar(k, 4.0 + rng.normal(0, 2.0), ajeno=ajeno)
    ind0.observar(300, 1.0, ajeno=False, correcto=False)
    ind0.observar(301, float('nan'), ajeno=True)
    r0 = ind0.estimar()
    assert r0['ic'][0] < 0 < r0['ic'][1] and (r0['n_ajenos'], r0['n_propios']) == (30, 270), r0
    # tendencia: sin atenuacion en la primera mitad y d = 1 en la segunda
    ind2 = emb.IndiceEmbodiment(semilla=1)
    for k in range(300):
        ajeno = k % 10 == 1
        propia = 4.0 if k < 150 else 2.0
        ind2.observar(k, (4.0 if ajeno else propia) + rng.normal(0, 2.0), ajeno=ajeno)
    r2 = ind2.estimar()
    assert r2['tendencia'] > 0.5 and r2['ic_tendencia'][0] < r2['tendencia'] < r2['ic_tendencia'][1], r2
    # amplitud: una N1 de -5 uV en la ventana, solo en PO7/Oz/PO8, da +5 (positiva = N1 mayor)
    e = np.zeros((8, 250))
    i0, i1 = (int(round((-config.EPOCA_ERRP[0] + v) * 250)) for v in config.VENTANA_N1)
    e[config.indices(config.CANALES_N1), i0:i1] = -5.0
    e[config.indices('errp'), :] = 30.0                  # lo fronto-central no entra
    assert abs(emb.amplitud_n1(e, 250) - 5.0) < 1e-9
    return f"d {r['iic']:.2f} [{r['ic'][0]:.2f}, {r['ic'][1]:.2f}] (verdadero 0.6); sin atenuacion {r0['iic']:+.2f}"


@prueba
def gemelo_embodiment():
    """Gemelo con --embodiment: la N1 a movimientos propios se atenua segun el embodiment y la
    de los ajenos no. Con la misma semilla el ruido es identico entre niveles: el IIC debe
    crecer con el embodiment. La aceptacion estadistica, con sesiones independientes, esta en
    estudios/embodiment_gemelo.py."""
    import cerebro_sintetico as cs, embodiment as emb
    cer = cs.Cerebro(cs._args(semilla=0, embodiment=1.0))
    cer.meta, cer.dir_paso = 1, 1
    n = len(cer.eventos)
    cer._marcador(config.m_paso_ajeno(7), 10.0)          # el gemelo ve el movimiento ajeno
    assert len(cer.eventos) > n
    iic, n1 = {}, {}
    for e in (0.0, 0.5, 1.0):
        X, aj, ok = cs.sesion_embodiment(150, embodiment=e, semilla=4)
        a = np.array([emb.amplitud_n1(x, cs.FS) for x in X])
        n1[e] = (a[aj].mean(), a[~aj & ok].mean())
        iic[e] = emb.desde_epocas(X, aj, ok, fs=cs.FS).estimar(con_ic=False)['iic']
    assert iic[0.0] < iic[0.5] < iic[1.0], iic
    assert np.isclose(n1[0.0][0], n1[1.0][0])           # la N1 ajena no cambia con el embodiment
    caida = 1 - n1[1.0][1] / n1[0.0][1]                  # la propia baja ~ATENUACION_MAX (0.5)
    assert 0.3 < caida < 0.7, caida
    return f"IIC {iic[0.0]:+.2f} / {iic[0.5]:+.2f} / {iic[1.0]:+.2f} con embodiment 0 / 0.5 / 1; N1 propia -{caida:.0%}"


@prueba
def orquestador_ajenos():
    """Tarea 2: uno de cada 10 pasos del lazo adaptativo es un movimiento ajeno anunciado y
    hacia la meta; el agente no aprende de el, queda fuera del analisis y el IIC sale en el CSV."""
    import orquestador
    a = orquestador.argumentos(['sim', '--ciclo', '0', '--semilla', '3', '--pasos_estatico', '40',
                                '--pasos_adaptativo', '300', '--sin_perturbacion', '--embodiment', '0.8'])
    orq = orquestador.Orquestador(orquestador.BackendSim(a), a)
    marcas, publicar = [], orq.salidas.marcador
    orq.salidas.marcador = lambda txt, *r: (marcas.append(txt), publicar(txt, *r))[1]
    orquestador.correr(orq, a)
    f = orq.filas
    est = [x for x in f if x['estado'] == 'LAZO_ESTATICO']
    aj = [x for x in f if x['ajeno'] == 1]
    assert est and not any(x['ajeno'] for x in est)
    assert len(aj) == 300 // 10, len(aj)
    assert all(x['excluido'] == 'ajeno' and x['direccion'] == x['meta'] and abs(x['delta']) == config.PASO_AJENO
               for x in aj), aj[:3]
    for i, x in enumerate(f):                            # el agente no aprende de los ajenos
        if x['ajeno'] == 1:
            assert x['beta'] == f[i - 1]['beta'], (i, x['beta'], f[i - 1]['beta'])
    assert sum(m.startswith('paso_ajeno:') for m in marcas) == marcas.count(config.AVISO_AJENO) == len(aj)
    r = orq.iic
    assert r['n_ajenos'] == len(aj) and r['iic'] > 0 and r['ic'][0] > 0, r
    assert f[-1]['iic'] != '' and all(x['n1_uv'] != '' for x in aj)
    return f"{len(aj)} ajenos; IIC {r['iic']:.2f} [{r['ic'][0]:.2f}, {r['ic'][1]:.2f}] con embodiment 0.8 (simulado)"


@prueba
def cuestionario():
    """Cuestionario de la Tarea 2: tres afirmaciones de 1 a 7; si la respuesta no vale se
    vuelve a preguntar; se guarda junto a la sesion con el IIC."""
    import json, tempfile
    from pathlib import Path
    import orquestador
    respuestas = iter(['5', '9', 'x', '7', '1'])
    ruta = Path(tempfile.mkdtemp()) / 'sesion_cuestionario.json'
    r = orquestador.cuestionario(ruta, leer=lambda _: next(respuestas), iic={'iic': 0.4}, salida=lambda *_: None)
    assert [x['respuesta'] for x in r['items']] == [5, 7, 1]
    guardado = json.loads(ruta.read_text(encoding='utf-8'))
    assert [x['afirmacion'] for x in guardado['items']] == config.CUESTIONARIO and guardado['iic'] == {'iic': 0.4}
    return '3 respuestas guardadas'


@prueba
def calibracion_errp_fija():
    """La calibracion de ErrP usa siempre todas las epocas pedidas (120 por defecto): sin GO ni
    NO GO tempranos, que con pocos datos elegian estimados inflados por suerte."""
    import types
    import orquestador
    import hardware as hw
    assert orquestador.argumentos(['real']).ensayos_errp == 120
    rng = np.random.default_rng(0)
    t = np.arange(250) / 250
    ultimo = {'meta': 1, 'dir': 1}

    class Salidas:
        def marcador(self, txt, t=None):
            if txt in (config.CUE_CERRAR, config.CUE_RELAJA):
                ultimo['meta'] = 1 if txt == config.CUE_CERRAR else -1

        def estado(self, **k):
            pass
        paso = types.SimpleNamespace(push_sample=lambda x: ultimo.update(dir=int(x[1])))

    class EEG:                                            # ErrP enorme: con la regla vieja daba GO a las 40
        def epoca(self, t0):
            e = rng.normal(0, 2, size=(8, 250))
            if ultimo['dir'] != ultimo['meta']:
                e[2] += 8 * np.exp(-((t - 0.5) ** 2) / 0.005)
            return e

        def lecturas(self):
            return {'canales': {}}
    b = orquestador.BackendReal.__new__(orquestador.BackendReal)
    b.hw, b.eeg, b.ortesis, b.detector = hw, EEG(), hw.OrtesisSimulada(latencia_ms=0.1, jitter_ms=0.0), None
    b.a = types.SimpleNamespace(ensayos_errp=60, p_error=0.3, espera=0.0, forzar=True)
    import cue
    orq = types.SimpleNamespace(salidas=Salidas(), audio_cue=cue.Audio())
    posiciones, mover = [], b.ortesis.mover
    b.ortesis.mover = lambda fraccion, *r: (posiciones.append(fraccion), mover(fraccion, *r))[1]
    assert b.calibrar_errp(orq)
    assert len(b.detector.y_cal) == 60 and b.detector.ba > 0.9, (len(b.detector.y_cal), b.detector.ba)
    # cada ensayo mueve la ortesis de verdad: si el movimiento no cabe en el recorrido, antes
    # vuelve al centro (sin movimiento no hay nada que ver, y la epoca no tendria ErrP)
    saltos = np.abs(np.diff(posiciones))
    ensayos = np.isclose(saltos, 0.15)
    assert ensayos.sum() == 60, (ensayos.sum(), np.round(saltos, 2))
    assert all(np.isclose(posiciones[i + 1], 0.5) for i in np.flatnonzero(~ensayos)), np.round(posiciones, 2)
    assert min(posiciones) >= 0.1 - 1e-9 and max(posiciones) <= 0.9 + 1e-9
    pd = b.detector.por_direccion                          # B2: el desglose por direccion llega al detector
    assert pd['cerrar']['n'] + pd['abrir']['n'] == 60 and min(pd['cerrar']['n'], pd['abrir']['n']) >= 25, pd
    return (f'usa las 60 epocas pedidas aunque el detector es casi perfecto (BA {b.detector.ba:.2f}); '
            f'los 60 ensayos mueven la ortesis ({int((~ensayos).sum())} vueltas al centro)')


@prueba
def errp_por_direccion():
    """B2: sensibilidad y especificidad del detector por direccion (cerrar / abrir). La diferencia de
    especificidad pasa de config.ESPEC_DIF_MAX solo avisa (con ~40 aciertos por direccion el azar solo
    ya da ~0.06): un detector sesgado a una direccion hace que el agente aprenda mal."""
    import hardware as hw
    import cerebro_sintetico as cs
    y = np.r_[np.zeros(40), np.ones(10), np.zeros(40), np.ones(10)].astype(int)
    d = np.r_[np.ones(50), np.zeros(50)].astype(int)                     # primera mitad cerrar, segunda abrir

    def con_falsas_alarmas(n_cerrar, n_abrir):
        pred = y.copy()
        pred[np.flatnonzero((y == 0) & (d == 1))[:n_cerrar]] = 1
        pred[np.flatnonzero((y == 0) & (d == 0))[:n_abrir]] = 1
        return hw.metricas_por_direccion(y, pred, d)
    parejo, sesgado = con_falsas_alarmas(4, 4), con_falsas_alarmas(4, 12)
    assert abs(parejo['cerrar']['espec'] - 0.90) < 1e-9 and parejo['dif_espec'] == 0 and not parejo['avisa'], parejo
    assert abs(sesgado['abrir']['espec'] - 0.70) < 1e-9 and abs(sesgado['dif_espec'] - 0.20) < 1e-9 and sesgado['avisa']
    assert sesgado['cerrar']['sens'] == 1.0 and sesgado['cerrar']['n'] == 50
    una = hw.metricas_por_direccion(y[:50], y[:50], np.ones(50, dtype=int))      # falta una direccion
    assert np.isnan(una['abrir']['espec']) and not una['avisa']
    # integrado: ajustar() desglosa las predicciones de su validacion anidada
    X, ye = cs.sesion_errp(60, semilla=0)
    dire = np.random.default_rng(0).permutation([1, 0] * 30)
    det = hw.DetectorErrP().ajustar(X, ye, direccion=dire)
    pd = det.por_direccion
    assert pd['cerrar']['n'] + pd['abrir']['n'] == 60
    for k in ('cerrar', 'abrir'):
        assert 0 <= pd[k]['espec'] <= 1 and 0 <= pd[k]['sens'] <= 1, pd
    assert not hasattr(hw.DetectorErrP().ajustar(X, ye), 'por_direccion')           # sin direccion no lo calcula
    return (f"espec cerrar {pd['cerrar']['espec']:.2f} / abrir {pd['abrir']['espec']:.2f} "
            f"(dif {pd['dif_espec']:.2f}); un sesgo de 0.20 si avisa")


@prueba
def bloque_sham():
    """B1: el bloque sham detecta un p(t) que sigue a la ortesis (control positivo: un piloto que imagina lo
    que hace la ortesis) y deja pasar a un piloto en reposo. Gemelo, 40 pasos por sesion."""
    import hardware as hw
    import cerebro_sintetico as cs
    import bloque_sham
    rng = np.random.default_rng(0)
    # evaluador: separado -> FALLA; independiente -> PASA; pocos datos -> None
    d = rng.permutation([1, 0] * 20)
    assert hw.evaluar_sham(0.2 + 0.6 * d + rng.normal(0, 0.05, 40), d)['pasa'] is False
    assert hw.evaluar_sham(rng.uniform(size=40), d)['pasa'] is True
    assert hw.evaluar_sham([0.5] * 5, [1, 0, 1, 0, 1])['pasa'] is None
    # contra el gemelo
    Xmi, ymi = cs.sesion_mi(40, semilla=0)
    dec = hw.DecoderIM().ajustar(Xmi, ymi)
    p_de = lambda X: 1 / (1 + np.exp(-np.array([dec.w0 @ dec.phi(x, actualizar_centro=False) + dec.c0 for x in X])))

    def sham(sigue, semilla):
        X, direccion = cs.sesion_sham(40, sigue=sigue, semilla=semilla)
        return hw.evaluar_sham(p_de(X), direccion)
    nulos = [sham(False, s) for s in range(4)]
    controles = [sham(True, s) for s in range(2)]
    falsas = sum(r['pasa'] is False for r in nulos)
    assert falsas <= 1, [round(r['auc'], 2) for r in nulos]                   # nominal: 5 % de falsas alarmas
    assert all(r['pasa'] is False for r in controles), [round(r['auc'], 2) for r in controles]
    # el script: pasos balanceados, ortesis que se mueve, y pasos excluidos si el EEG no esta fresco
    class EEG:
        fs, hay = 250, True

        def ventana(self, seg, pm=0.0, hm=None):
            return (rng.normal(0, 5, (8, int(seg * 250))), None) if self.hay else (None, None)

        def ultimo_t(self):
            return 0.0

        def movimiento(self, t0, t1):
            return None
    ort = hw.OrtesisSimulada(latencia_ms=0.1, jitter_ms=0.0)
    filas, r = bloque_sham.correr(EEG(), ort, dec, pasos=8, semilla=0, reposo_s=0.0, despues_s=0.0, salida=lambda *a: None)
    assert len(filas) == 8 and sum(f['direccion'] for f in filas) == 4 and not any(f['excluido'] for f in filas)
    assert all(0 < f['p'] < 1 for f in filas) and ort.seq >= 16               # centrado + movimiento por paso
    eeg_mal = EEG(); eeg_mal.hay = False
    filas, r = bloque_sham.correr(eeg_mal, ort, dec, pasos=4, semilla=0, reposo_s=0.0, despues_s=0.0, salida=lambda *a: None)
    assert all(f['excluido'] == 'eeg_no_fresco' for f in filas) and r['pasa'] is None
    return (f"gemelo: piloto en reposo AUC {min(r['auc'] for r in nulos):.2f}-{max(r['auc'] for r in nulos):.2f} "
            f"({falsas} de 4 con falsa alarma); piloto que sigue a la ortesis AUC "
            f"{min(r['auc'] for r in controles):.2f}-{max(r['auc'] for r in controles):.2f} (2 de 2 detectados)")


@prueba
def intervalo_por_ensayos():
    """El intervalo del 90 % de una tasa de error se calcula remuestreando ENSAYOS (5 pasos con la
    misma meta), no pasos sueltos: los errores de un mismo ensayo estan correlacionados."""
    import hardware as hw
    rng = np.random.default_rng(0)
    independientes = (rng.random(300) < 0.2).astype(int)
    lo, hi = hw.intervalo_error(independientes, nivel=0.90)
    assert lo < independientes.mean() < hi and 0.05 < hi - lo < 0.13, (lo, hi)
    correlacionados = np.repeat((rng.random(60) < 0.2).astype(int), 5)   # ensayos enteros bien o mal
    lo2, hi2 = hw.intervalo_error(correlacionados, nivel=0.90)
    assert hi2 - lo2 > 1.5 * (hi - lo), ((lo, hi), (lo2, hi2))
    assert hw.intervalo_error([], nivel=0.9) == (0.0, 1.0)
    return f'pasos independientes: [{lo:.2f}, {hi:.2f}]; ensayos enteros: [{lo2:.2f}, {hi2:.2f}]'


@prueba
def senal_valida():
    import hardware as hw
    fs, rng = 250, np.random.default_rng(0)
    x = rng.normal(0, 10, size=(8, 1000)); t = 100 + np.arange(1000) / fs
    assert hw.revisar_canales(x, fs) == {}
    malo = x.copy(); malo[2] = 5.0; malo[7] *= 40; malo[0, 10] = 1.5 * config.SALUD['canal_saturado_uv']
    assert hw.revisar_canales(malo, fs) == {'Fz': 'saturado', 'Cz': 'plano', 'PO8': 'ruidoso'}
    assert hw.revisar_canales(np.empty((8, 0)), fs) == {}                           # buffer vacio
    recien = x.copy(); recien[4, -125:] = 0.0                                       # se despego hace 0.5 s
    assert hw.revisar_canales(recien, fs) == {'Pz': 'plano'}
    assert hw.ventana_valida(x, t, fs, edad_s=0.05, segundos=3.0)
    assert not hw.ventana_valida(x, t, fs, edad_s=2.0, segundos=3.0)               # rancia
    assert not hw.ventana_valida(x[:, :300], t[:300], fs, 0.05, 3.0)               # incompleta
    assert not hw.ventana_valida(np.empty((0,)), np.empty(0), fs, 0.05, 3.0)       # buffer vacio
    con_nan = x.copy(); con_nan[1, 500] = np.nan
    assert not hw.ventana_valida(con_nan, t, fs, 0.05, 3.0)
    t_hueco = t.copy(); t_hueco[620:] += 1.5
    assert not hw.ventana_valida(x, t_hueco, fs, 0.05, 3.0)                        # cruza un corte
    t_atras = t.copy(); t_atras[620:] -= 0.5
    assert not hw.ventana_valida(x, t_atras, fs, 0.05, 3.0)                        # salto hacia atras
    t0 = t[600]
    e = hw.cortar_epoca(x, t, t0, fs)
    assert e is not None and e.shape == (8, 250)
    assert hw.cortar_epoca(x, t_hueco, t0, fs) is None                              # corte dentro de la epoca
    assert hw.cortar_epoca(x, t_hueco, t[300], fs) is not None                      # epoca limpia antes del corte
    assert hw.cortar_epoca(con_nan, t, t[450], fs) is None
    assert hw.cortar_epoca(x, t, t[-10], fs) is None                                # faltan muestras
    assert hw.cortar_epoca(x, t, t[0] - 5.0, fs) is None                            # t0 fuera del buffer
    # perdidas de Bluetooth chicas y repartidas (5 %): la ventana estricta no vale; la de MI si
    quedan = np.sort(rng.choice(1000, 950, replace=False))
    xl, tl = x[:, quedan], t[quedan]
    u = config.SALUD
    assert not hw.ventana_valida(xl, tl, fs, 0.05, 3.0)
    assert hw.ventana_valida(xl, tl, fs, 0.05, 3.0, u['mi_perdida_max'], u['mi_hueco_max_s'])
    hoyo = np.r_[0:700, 800:1000]                                                   # falta 0.4 s seguido
    assert not hw.ventana_valida(x[:, hoyo], t[hoyo], fs, 0.05, 3.0, u['mi_perdida_max'], u['mi_hueco_max_s'])
    # rejilla: las muestras con hora vuelven a una rejilla uniforme
    onda = np.sin(2 * np.pi * 5 * (t - t[0]))[None, :] * np.ones((8, 1)) * 20
    xu, tu = hw.rejilla(onda[:, quedan], tl, fs)
    assert abs(np.diff(tu) - 1 / fs).max() < 1e-9 and np.abs(xu - 20 * np.sin(2 * np.pi * 5 * (tu - t[0]))).max() < 1.0
    # epoca con 3 muestras perdidas dentro (12 ms): sale completa y alineada con la de los datos completos
    falta = np.r_[0:640, 643:1000]
    e_ok, e_falta = hw.cortar_epoca(onda, t, t0, fs), hw.cortar_epoca(onda[:, falta], t[falta], t0, fs)
    assert e_falta is not None and e_falta.shape == e_ok.shape == (8, 250)
    assert np.abs(e_falta - e_ok).max() < 1.5, np.abs(e_falta - e_ok).max()
    return 'canales, ventana y epoca validadas por tiempo; perdidas chicas toleradas e interpoladas'


@prueba
def deriva_reloj():
    """La deriva del reloj se mide contra una mediana movil: detecta un cambio de desfase,
    pero el reloj nunca queda en ROJO para siempre (ni por un transitorio al volver los datos)."""
    import hardware as hw
    rng = np.random.default_rng(0)
    rojo = config.SALUD['reloj_rojo_ms']
    estable = list(0.020 + rng.normal(0, 0.003, 600))
    assert hw.deriva_reloj([]) == 0.0 and hw.deriva_reloj(estable[:10]) == 0.0      # aun sin datos
    assert abs(hw.deriva_reloj(estable)) < 5
    salto = estable + list(0.080 + rng.normal(0, 0.003, 40))                         # el desfase cambia 60 ms
    assert hw.deriva_reloj(salto) > rojo                                             # se nota de inmediato
    asentado = estable + list(0.080 + rng.normal(0, 0.003, 900))                     # ...y pasa a ser lo normal
    assert abs(hw.deriva_reloj(asentado)) < 5
    transitorio = list(-0.030 + rng.normal(0, 0.003, 60)) + list(0.020 + rng.normal(0, 0.003, 200))
    assert abs(hw.deriva_reloj(transitorio)) < 5                                     # arranque raro tras un corte
    return 'detecta el cambio de desfase y lo absorbe; un transitorio inicial no deja el reloj en ROJO'


@prueba
def ortesis_sin_ack():
    import hardware as hw

    class PlanFijo:                       # pierde los ACK 2, 3 y 4; pico de latencia en el 5
        def por_paso(self, tipo, seq):
            if tipo == 'ack_perdido':
                return 2 <= seq <= 4
            return 120.0 if seq == 5 else None
    o = hw.OrtesisSimulada(caos=PlanFijo(), timeout_ack=0.0)
    r = [o.mover(0.5) for _ in range(4)]
    assert [x[0] for x in r] == [1, 2, 3, 4]                   # seq nunca se reinicia
    assert r[0][1] is not None and r[1][1] is None and np.isnan(r[1][2])
    assert o.lecturas()['acks_perdidos'] == 3 and o.lecturas()['puerto_ok']
    seq, t_ack, lat = o.mover(0.5)
    assert seq == 5 and t_ack is not None and lat > 100 and o.lecturas()['acks_perdidos'] == 0
    return 'mover() no lanza; cuenta ACK perdidos, conserva seq y mide el pico de latencia'


class _ESP32Falso:
    """Puerto serie de mentira con el protocolo M/A del contrato y fallas a pedido."""

    def __init__(self, mundo):
        self.mundo, self.pendiente = mundo, b''

    def write(self, datos):
        if self.mundo['caido']:
            raise OSError('puerto caido')
        _, seq, _, _ = datos.decode().strip().split(',')
        if int(seq) in self.mundo['sin_ack']:
            return
        self.pendiente += self.mundo['basura'] + f'A,{seq},123\n'.encode()

    def read(self, n):
        if self.mundo['caido']:
            raise OSError('puerto caido')
        time.sleep(0.005)
        datos, self.pendiente = self.pendiente, b''
        return datos

    def close(self):
        pass


@prueba
def ortesis_udp():
    """OrtesisUDP (Wi-Fi) contra el firmware simulado en localhost, sin placa: el t_ack queda en el reloj
    de LSL (pylsl.local_clock) aunque la ESP32 tenga otro origen y deriva, los latidos no tocan el seq de
    los pasos, un ACK perdido no bloquea, y el paro de emergencia pone la ortesis en ROJO."""
    import argparse
    import math
    import numpy as np
    from pylsl import local_clock
    import hardware
    import orquestador
    import ortesis_udp_sim as sim
    from salud import Vigilante
    # el estimador de inicio por rampa, con telemetria a 10 Hz y un servo a 692 unidades/s
    vel, inicio = 692.0, 1_000_000 + 37_000
    tele = [(1_000_000 + 100_000 * k, 500.0 + max(0.0, (1_000_000 + 100_000 * k - inicio) / 1e6) * vel) for k in range(-1, 4)]
    est = hardware.inicio_por_rampa(tele, 1_000_000, vel)
    assert abs(est - inicio) < 1.0, (est, inicio)                       # exacto con rampa constante
    assert abs(hardware.inicio_por_telemetria(tele, 1_000_000) - inicio) > 20_000      # la interpolacion lineal lo adelanta
    assert hardware.inicio_por_rampa(tele[:2], 1_000_000, vel) is None   # sin movimiento visible
    fw = sim.FirmwareSimulado(puerto=0, origen_ms=7_654_321).iniciar()     # su reloj en ms no se parece al de LSL
    try:
        # el backend crea la ortesis de Wi-Fi con el reloj de LSL, no con time.monotonic
        por_defecto = hardware.OrtesisUDP('127.0.0.1', fw.puerto)
        assert por_defecto._clock is local_clock                                          # tambien por defecto
        por_defecto.cerrar()
        a = argparse.Namespace(ortesis_sim=False, ortesis_udp='127.0.0.1', udp_puerto=fw.puerto, puerto='COMX')
        o = orquestador.crear_ortesis(hardware, a)
        assert isinstance(o, hardware.OrtesisUDP) and o._clock is local_clock and o._clock is not time.monotonic
        assert o.conectada()
        time.sleep(1.0)                                                                    # varios latidos: reloj sincronizado
        desvios, rtts = [], []
        for k in range(8):
            antes = local_clock()
            seq, t_ack, rtt = o.mover(0.3 if k % 2 else 0.6)
            despues = local_clock()
            assert seq == k + 1, seq                                                       # los latidos numeran aparte
            assert antes - 0.005 <= t_ack <= despues + 0.005, (k, antes, t_ack, despues)    # en el reloj de LSL
            assert o.metodo == 'ack', o.metodo
            assert 0 <= rtt < 100 and abs(rtt - (despues - antes) * 1000) < 5
            desvios.append(t_ack - (antes + despues) / 2); rtts.append(rtt)
            time.sleep(0.3)
        assert fw.vigilancia is False and sum(1 for s_, _ in fw.recibidos if s_ and s_ > config.UDP_SEQ_LATIDO) > 10
        assert max(abs(d) for d in desvios) < 0.025, desvios         # ACK en el ciclo de 20 ms de la ESP32
        # inicio del movimiento por telemetria: cerca del que aplico el firmware (su ciclo es de 20 ms); con el
        # servo quieto (en el lazo los pasos van a ~2 s, y a 90 grados/s un paso de 0.3 dura 0.4 s)
        time.sleep(1.5)
        seq, t_ack, _ = o.mover(0.9)
        t0, como = o.inicio_movimiento(seq, t_ack)
        lat = hardware.latencia_mecanica_simulada(seq, 0)
        assert como in ("telemetria", "ack+latencia") and t_ack <= t0 <= t_ack + 0.5, (como, t0 - t_ack)
        if como == 'telemetria':
            assert abs(t0 - (t_ack + lat)) < 0.06, (t0 - t_ack, lat)
        # el mismo cliente con OTRO reloj (el de LSL mas 5000 s): el t_ack sale en ese reloj
        otro = lambda: local_clock() + 5000.0
        fw2 = sim.FirmwareSimulado(puerto=0, origen_ms=42).iniciar()
        try:
            o2 = hardware.OrtesisUDP('127.0.0.1', fw2.puerto, reloj=otro)
            time.sleep(0.8)
            antes = otro(); seq2, t2, _ = o2.mover(0.5); despues = otro()
            assert antes - 0.005 <= t2 <= despues + 0.005, (antes, t2, despues)
            o2.cerrar()
        finally:
            fw2.detener()
        # ACK perdido: sin ACK, no se cuelga y el siguiente paso funciona
        fw.perder_acks.add(o.seq + 1)
        t = time.time(); seq3, t_ack3, rtt3 = o.mover(0.4)
        assert t_ack3 is None and math.isnan(rtt3) and o.acks_perdidos == 1 and time.time() - t < 0.5
        assert o.mover(0.5)[1] is not None and o.acks_perdidos == 0 and o.seq == seq3 + 1
        # paro de emergencia: el Vigilante lo pone en ROJO con su motivo
        v = Vigilante()
        bien = {'edad_s': 0.02, 'tasa_hz': 250.0, 'canales': {}}
        v.actualizar(0.0, eeg=bien, ortesis=o.lecturas(), reloj_ms=0.0)
        assert v.colores['ortesis'] == config.VERDE, v.detalle
        fw.paro = True
        time.sleep(0.4)
        v.actualizar(1.0, eeg=bien, ortesis=o.lecturas(), reloj_ms=0.0)
        assert o.lecturas()['paro'] and v.colores['ortesis'] == config.ROJO and 'paro' in v.detalle['ortesis']
        fw.paro = False
        v2 = Vigilante()
        v2.actualizar(0.0, eeg=bien, ortesis=dict(o.lecturas(), paro=False, bloqueo=True), reloj_ms=0.0)
        assert v2.colores['ortesis'] == config.AMARILLO and 'bloqueo' in v2.detalle['ortesis']
        o.cerrar()
    finally:
        fw.detener()
    # sin placa: no se cuelga; sin telemetria la ortesis cuenta como desconectada
    muerto = hardware.OrtesisUDP('127.0.0.1', 9, esperar_telemetria_s=0.2)
    t = time.time(); seq, t_ack, _ = muerto.mover(0.5)
    assert t_ack is None and not muerto.conectada() and muerto.lecturas()['puerto_ok'] is False and time.time() - t < 0.5
    muerto.cerrar()
    return (f'reloj de LSL (desvio del ACK {max(abs(d) for d in desvios) * 1000:.0f} ms, RTT {np.median(rtts):.0f} ms); '
            f'seq de pasos sin saltos; ACK perdido sin bloqueo; paro en ROJO; sin placa no se cuelga')


@prueba
def ortesis_udp_nervio():
    """Cada paso manda p = p' del agente (set_p) y, si el detector marca un ErrP en una epoca sin artefacto,
    un destello rojo (errp): sesion simulada del orquestador con la ortesis UDP contra el firmware simulado."""
    import hardware
    import orquestador
    import ortesis_udp_sim as sim
    fw = sim.FirmwareSimulado(puerto=0).iniciar()
    try:
        o = hardware.OrtesisUDP('127.0.0.1', fw.puerto)
        # a mano: set_p viaja con la siguiente orden, y tambien en el latido sin mover nada
        o.set_p(0.83)
        time.sleep(0.4)
        assert fw.p == 0.83 and not fw.destellos
        o.errp()
        time.sleep(0.6)
        assert len(fw.destellos) == 1 + config.UDP_ERRP_LATIDOS       # el inmediato y los latidos siguientes
        # sin ortesis con nervio (USB, simulada) el backend no hace nada
        assert not hasattr(hardware.OrtesisSimulada(), 'set_p')
        orquestador.BackendReal.nervio(type('B', (), {'ortesis': hardware.OrtesisSimulada()})(), 0.5)
        orquestador.BackendReal.destello(type('B', (), {'ortesis': hardware.OrtesisSimulada()})())

        class ConNervio(orquestador.BackendSim):
            """El simulador del lazo, con el nervio de luz de la ortesis UDP."""
            def nervio(self, p):
                self.p_enviadas.append(p)
                o.set_p(p)

            def destello(self):
                self.n_destellos += 1
                self.t_destellos.append(time.monotonic())
                o.errp()

            def mover(self, fraccion):
                time.sleep(0.12)                 # lo que tarda un paso real: el latido y la orden llegan al firmware
                return super().mover(fraccion)
        a = orquestador.argumentos(['sim', '--ciclo', '0', '--semilla', '4', '--pasos_estatico', '10', '--pasos_adaptativo', '40',
                                    '--sin_perturbacion', '--ajenos-cada', '0'])
        b = ConNervio(a)
        b.p_enviadas, b.n_destellos, b.t_destellos = [], 0, []
        orq = orquestador.Orquestador(b, a)
        eventos = []
        estado = orq.salidas.estado
        orq.salidas.estado = lambda **d: (eventos.append(d), estado(**d))[1]
        orquestador.correr(orq, a)
        time.sleep(0.3)
        pasos = [e for e in eventos if e['tipo'] == 'paso']
        # p' de cada paso del lazo (no la posicion segura ni otros movimientos) llego a la ortesis
        assert len(b.p_enviadas) == len(pasos) > 30 and np.allclose(b.p_enviadas, [e['p_prima'] for e in pasos])
        assert 0.0 <= min(b.p_enviadas) and max(b.p_enviadas) <= 1.0 and np.std(b.p_enviadas) > 0.05
        assert fw.p == o.estado['p'] == b.p_enviadas[-1] or abs(fw.p - b.p_enviadas[-1]) < 1e-9
        # un destello por ErrP detectado en una epoca SIN artefacto (la fila del CSV dice si lo tuvo)
        por_paso = {e['paso']: e for e in pasos}
        marcados = sum(1 for i, f in enumerate(orq.filas, 1) if por_paso[i]['detectado'] and not int(f['artefacto']))
        con_art = sum(1 for i, f in enumerate(orq.filas, 1) if por_paso[i]['detectado'] and int(f['artefacto']))
        assert marcados > 3 and con_art >= 1 and b.n_destellos == marcados, (marcados, con_art, b.n_destellos)
        for t_e in b.t_destellos:                                       # y cada uno llego (localhost), en cuanto se mando
            assert any(t_e - 0.01 <= d <= t_e + 0.25 for d in fw.destellos), t_e
        o.cerrar()
    finally:
        fw.detener()
    return f'{len(pasos)} pasos: p\' llego en cada uno; {marcados} destellos rojos, uno por ErrP detectado sin artefacto ({con_art} con artefacto, sin destello)'


@prueba
def destello_errp_con_perdidas():
    """El destello de ErrP es un datagrama y el Wi-Fi del evento pierde: errp tambien viaja en los latidos
    siguientes. Firmware simulado que descarta datagramas (en rafaga y al azar), con y sin la repeticion."""
    import hardware
    import ortesis_udp_sim as sim

    def llegadas(errp_latidos, descartar, eventos, cada=0.5):
        """Cuantos de `eventos` ErrP llegaron al firmware: hubo un mensaje con errp en los 0.45 s siguientes."""
        fw = sim.FirmwareSimulado(puerto=0).iniciar()
        try:
            o = hardware.OrtesisUDP('127.0.0.1', fw.puerto, errp_latidos=errp_latidos)
            time.sleep(0.4)
            fw.descartar = descartar
            t_ev = []
            for _ in range(eventos):
                t_ev.append(time.monotonic())
                o.errp()
                time.sleep(cada)
            time.sleep(0.2)
            n = sum(1 for t in t_ev if any(t - 0.01 <= d <= t + 0.45 for d in fw.destellos))
            o.cerrar()
            return n
        finally:
            fw.detener()
    # rafaga: se pierden el envio inmediato y los dos primeros latidos de cada ErrP; el tercero llega
    def rafaga():
        estado = {'t': -1.0, 'n': 0}

        def descartar(m):
            if not m.get('errp'):
                return False
            ahora = time.monotonic()
            estado['n'] = 0 if ahora - estado['t'] > 0.3 else estado['n'] + 1        # n-esimo mensaje del mismo ErrP
            estado['t'] = ahora
            return estado['n'] < 3
        return descartar
    assert llegadas(0, rafaga(), 4) == 0                                   # un solo datagrama: se pierden todos
    assert llegadas(config.UDP_ERRP_LATIDOS, rafaga(), 4) == 4             # con latidos, llegan todos
    # al azar: cada datagrama se pierde con probabilidad 0.5 (con 3 repeticiones, 1 - 0.5^4 = 94 % de los ErrP llegan)
    def azar(semilla):
        rng = np.random.default_rng(semilla)
        return lambda m: bool(rng.random() < 0.5)
    sin = llegadas(0, azar(1), 12)
    con = llegadas(config.UDP_ERRP_LATIDOS, azar(1), 12)
    assert con >= 9 and con > sin, (con, sin)
    return f'perdiendo el 50 % de los datagramas llegan {con} de 12 ErrP con repeticion contra {sin} sin ella; en rafaga de 3, 4 de 4 contra 0 de 4'


@prueba
def ortesis_serial_reconecta():
    """OrtesisSerial contra un ESP32 de mentira: lineas corruptas, ACK perdido, puerto caido y reapertura."""
    import hardware as hw
    mundo = {'caido': False, 'sin_ack': {2}, 'basura': b'A,xx,\n\xff\xfeT,1\n', 'aperturas': 0}

    def abrir():
        mundo['aperturas'] += 1
        if mundo['caido']:
            raise OSError('no existe el puerto')
        return _ESP32Falso(mundo)
    o = hw.OrtesisSerial(abrir=abrir, timeout_ack=0.1)
    try:
        seq, t_ack, lat = o.mover(0.5)
        assert seq == 1 and t_ack is not None and lat < 100        # la basura no mata al lector
        assert o.mover(0.5)[1] is None and o.lecturas()['acks_perdidos'] == 1
        mundo['caido'] = True                                      # se reinicia el ESP32
        assert o.mover(0.5)[1] is None
        time.sleep(0.1)
        assert not o.lecturas()['puerto_ok']
        mundo['caido'] = False
        t_fin = time.time() + 5
        while not o.lecturas()['puerto_ok'] and time.time() < t_fin:
            time.sleep(0.05)
        assert o.lecturas()['puerto_ok'] and mundo['aperturas'] >= 2
        seq, t_ack, _ = o.mover(0.5)
        assert seq == 4 and t_ack is not None and o.lecturas()['acks_perdidos'] == 0
    finally:
        o.cerrar()
    return f"sobrevive a basura, ACK perdido y puerto caido; reabre solo ({mundo['aperturas']} aperturas), seq continua"


@prueba
def reconexion_eeg():
    """El flujo de EEG muere y vuelve como instancia nueva con OTRO desfase de reloj:
    EntradaEEG se reconecta, el reloj no queda en ROJO, la pausa puede terminar y
    ninguna ventana ni epoca cruza el hueco."""
    import threading
    import hardware as hw
    from pylsl import StreamInfo, StreamOutlet, local_clock
    from salud import Vigilante
    fs, nombre = 250, 'EEG_prueba'
    rng = np.random.default_rng(0)
    pub = {'outlet': None, 'desfase': 0.0, 't': local_clock(), 'vivo': True}
    candado = threading.Lock()

    def crear(desfase):
        with candado:
            pub['desfase'], pub['t'] = desfase, local_clock()
            pub['outlet'] = StreamOutlet(StreamInfo(nombre, 'EEG', 8, fs, 'float32', ''), chunk_size=10)

    def publicar():
        while pub['vivo']:
            with candado:
                n = int((local_clock() - pub['t']) * fs)
                if pub['outlet'] is not None and n > 0:
                    pub['t'] += n / fs
                    pub['outlet'].push_chunk(rng.normal(0, 10, size=(n, 8)).tolist(), pub['t'] + pub['desfase'])
            time.sleep(0.02)

    def vigilar(v, eeg, segundos, hasta=None):
        """Alimenta al Vigilante cada 0.1 s; devuelve los colores vistos de eeg y reloj."""
        vistos, t_fin = [], time.time() + segundos
        while time.time() < t_fin:
            l = eeg.lecturas()
            v.actualizar(time.time(), eeg=l, reloj_ms=l['reloj_ms'],
                         ortesis={'puerto_ok': True, 'acks_perdidos': 0, 'latencia_ms': 8.0})
            vistos.append((v.colores['eeg'], v.colores['reloj']))
            if hasta is not None and hasta():
                break
            time.sleep(0.1)
        return vistos

    crear(0.0)
    threading.Thread(target=publicar, daemon=True).start()
    eeg = hw.EntradaEEG(segundos=10.0, timeout=8.0, nombre=nombre)
    try:
        v = Vigilante()
        vigilar(v, eeg, 12.0, hasta=lambda: v.listo_para_reanudar(time.time()))
        assert v.listo_para_reanudar(time.time()), (v.colores, v.detalle)
        assert eeg.ventana(3.0)[0] is not None
        # se cae el flujo
        with candado:
            pub['outlet'] = None
        t_corte = local_clock()
        vistos = vigilar(v, eeg, 1.5)
        assert v.motivo_pausa() == 'eeg', (v.colores, v.detalle)
        assert eeg.ventana(3.0) == (None, None)
        # vuelve como instancia nueva, estampando 0.25 s adelantado
        crear(0.25)
        vistos = vigilar(v, eeg, 20.0, hasta=lambda: v.listo_para_reanudar(time.time()))
        assert v.listo_para_reanudar(time.time()), (v.colores, v.detalle)   # se puede salir de la pausa
        assert eeg.reconexiones == 1, eeg.reconexiones
        rojo_reloj = [r for _, r in vistos if r == config.ROJO]
        assert not rojo_reloj, f'el reloj quedo en ROJO {len(rojo_reloj)} lecturas tras reconectar'
        x, t = eeg.ventana(3.0)
        assert x is not None and t[0] > t_corte + 1.0 and np.abs(np.diff(t)).max() <= config.SALUD['hueco_max_s']
        assert eeg.epoca(t_corte + 0.5) is None           # t0 dentro del hueco
        assert eeg.epoca(t_corte - 0.3) is None           # la epoca cruzaria el hueco
        assert eeg.epoca(t[-1] - 1.0) is not None         # una epoca nueva y limpia si sale
    finally:
        pub['vivo'] = False
        eeg.cerrar()
    return 'reconecta con otro desfase; reloj sin ROJO; sale de la pausa; nada cruza el hueco'


@prueba
def silencio_sin_recrear():
    """El flujo enmudece 2.5 s y sigue con la MISMA instancia (un tiron del Bluetooth). La hora
    de cada muestra viene de la fuente (reconstruida por contador), no del suavizado de LSL:
    al volver los datos las marcas siguen alineadas con el reloj, sin vaciar el buffer."""
    import threading
    import hardware as hw
    from pylsl import StreamInfo, StreamOutlet, local_clock
    fs, nombre = 250, 'EEG_prueba3'
    out = StreamOutlet(StreamInfo(nombre, 'EEG', 8, fs, 'float32', ''), chunk_size=10)
    pub = {'vivo': True, 'callado': False, 't': local_clock()}
    rng = np.random.default_rng(0)

    def publicar():
        while pub['vivo']:
            ahora = local_clock()
            if pub['callado']:
                pub['t'] = ahora                   # al volver no rellena el hueco (como el gemelo)
            else:
                n = int((ahora - pub['t']) * fs)
                if n > 0:
                    horas = pub['t'] + np.arange(1, n + 1) / fs
                    pub['t'] = horas[-1]
                    out.push_chunk(rng.normal(0, 10, size=(n, 8)).tolist(), list(horas))
            time.sleep(0.02)
    threading.Thread(target=publicar, daemon=True).start()
    eeg = hw.EntradaEEG(segundos=10.0, timeout=8.0, nombre=nombre)
    retraso = lambda: local_clock() - eeg.ultimo_t()
    try:
        time.sleep(4.0)
        antes = retraso()
        assert antes < 0.1, antes
        n_antes = eeg._crudo(10.0)[1].size
        pub['callado'] = True
        t_corte = local_clock()
        time.sleep(2.5)
        pub['callado'] = False
        time.sleep(1.0)
        despues = retraso()
        assert abs(despues - antes) < 0.05, f'marcas desfasadas {1000 * (despues - antes):.0f} ms tras el silencio'
        assert eeg.reconexiones == 0 and eeg._crudo(10.0)[1].size > n_antes       # el buffer no se vacia
        assert eeg.ventana(3.0) == (None, None)            # la ventana cruzaria el hueco
        assert eeg.epoca(t_corte + 2.0) is None            # la epoca cruzaria el hueco
        time.sleep(3.0)
        x, t = eeg.ventana(3.0)
        assert x is not None and t[0] > t_corte + 2.0
        assert abs(eeg.lecturas()['reloj_ms']) < config.SALUD['reloj_rojo_ms']
        assert eeg.epoca(t[-1] - 1.0) is not None
    finally:
        pub['vivo'] = False
        eeg.cerrar()
    return f'tras 2.5 s de silencio las marcas coinciden con el reloj (diferencia {1000 * (despues - antes):+.0f} ms), sin vaciar el buffer'


@prueba
def dos_flujos_eeg():
    """Dos flujos con el mismo nombre en la red (por ejemplo, un gemelo olvidado en otra
    terminal): EntradaEEG elige el mas reciente, NO salta al otro mientras el suyo viva y,
    si el suyo muere, espera a uno mas nuevo en vez de conectarse al viejo."""
    import threading
    import hardware as hw
    from pylsl import StreamInfo, StreamOutlet, local_clock
    fs, nombre = 250, 'EEG_prueba2'
    viejo = StreamOutlet(StreamInfo(nombre, 'EEG', 8, fs, 'float32', ''), chunk_size=10)
    time.sleep(0.3)
    nuevo = StreamOutlet(StreamInfo(nombre, 'EEG', 8, fs, 'float32', ''), chunk_size=10)
    pub = {'vivo': True, 'callado': False, 't': local_clock(), 'nuevo': nuevo, 'valor': 0.0}
    del nuevo
    candado = threading.Lock()

    def publicar():                            # el viejo vale 1000 uV, el nuevo 0: asi se distinguen
        while pub['vivo']:
            with candado:
                n = int((local_clock() - pub['t']) * fs)
                if n > 0:
                    pub['t'] += n / fs
                    viejo.push_chunk(np.full((n, 8), 1000.0).tolist(), pub['t'])
                    if not pub['callado'] and pub['nuevo'] is not None:
                        pub['nuevo'].push_chunk(np.full((n, 8), pub['valor']).tolist(), pub['t'])
            time.sleep(0.02)
    threading.Thread(target=publicar, daemon=True).start()
    eeg = hw.EntradaEEG(segundos=10.0, timeout=8.0, nombre=nombre)
    try:
        time.sleep(1.5)
        assert eeg._crudo(1.0)[0].max() == 0.0, 'debio elegir el flujo mas reciente'
        pub['callado'] = True                  # su flujo enmudece 2.5 s, pero sigue vivo
        time.sleep(2.5)
        assert eeg.lecturas()['edad_s'] > 1.0
        pub['callado'] = False
        time.sleep(2.5)
        x, _ = eeg._crudo(1.0)
        assert eeg.reconexiones == 0 and x.max() == 0.0, (eeg.reconexiones, x.max())
        assert eeg.lecturas()['edad_s'] < 0.5
        assert 200 < eeg.lecturas()['tasa_hz'] < 400, eeg.lecturas()['tasa_hz']   # por muestras llegadas (una rafaga al volver puede pasar de 250)
        # su flujo muere: el viejo sigue ahi, pero ya fue descartado al arrancar; no se conecta a el
        with candado:
            pub['nuevo'] = None
        time.sleep(4.0)
        assert eeg.reconexiones == 0 and eeg.lecturas()['edad_s'] > 3.0, eeg.reconexiones
        # aparece una instancia mas nueva (el puente o el gemelo que volvio): a esa si
        with candado:
            pub['valor'] = 7.0
            pub['nuevo'] = StreamOutlet(StreamInfo(nombre, 'EEG', 8, fs, 'float32', ''), chunk_size=10)
        t_fin = time.time() + 15
        while eeg.reconexiones == 0 and time.time() < t_fin:
            time.sleep(0.1)
        time.sleep(1.0)
        assert eeg.reconexiones == 1 and eeg._crudo(0.5)[0].max() == 7.0
    finally:
        pub['vivo'] = False
        eeg.cerrar()
    return 'elige el mas reciente; no salta mientras el suyo viva; si muere, espera a uno mas nuevo'


@prueba
def entrada_unicorn():
    """EntradaEEG contra las fuentes de un Unicorn (gemelo): el flujo del puente, con IMU aparte,
    y el flujo unico de la app UnicornLSL (17 canales, resuelto por tipo o por nombre, hora por
    contador). Con perdidas de Bluetooth la ventana de MI sigue saliendo.

    Dos cosas del gemelo son al azar y la prueba las espera en vez de suponerlas (por suponerlas
    fallaba a veces: 1 de 2 corridas de --completa el 3 de octubre). Con 40 perdidas por minuto se
    pierde ~7 % de las muestras, pero a rachas: la ventana de MI (tolera 10 %) deja de salir hasta
    7 s seguidos. Y la cabeza puede pasar 7 s quieta. Durante una perdida la ultima muestra tiene
    hasta 0.26 s: el retraso es el menor de varias lecturas (un desfase de la hora por contador se
    veria en todas)."""
    import hardware as hw
    from pylsl import local_clock

    def con_gemelo(extra, **entrada):
        g = subprocess.Popen([sys.executable, 'cerebro_sintetico.py', '--semilla', '5', '--cabeza', '0.6',
                              '--perdidas-bt', '40'] + extra, cwd=config.RAIZ,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            eeg = hw.EntradaEEG(segundos=20.0, timeout=20.0, **entrada)
            try:
                time.sleep(7.0)
                x, t = eeg._crudo(6.0)
                l = eeg.lecturas()
                for _ in range(30):                # hasta 15 s: las perdidas vienen a rachas
                    ventana_mi = eeg.ventana(3.0, config.SALUD['mi_perdida_max'], config.SALUD['mi_hueco_max_s'])[0]
                    if ventana_mi is not None:
                        break
                    time.sleep(0.5)
                x, t = eeg._crudo(18.0)            # todo lo grabado: asi siempre hay algun hueco
                giro = eeg.movimiento(t[-1] - 10.0, t[-1])
                for _ in range(20):                # hasta 10 s mas: la cabeza se mueve al azar
                    if giro is not None and giro > 20:
                        break
                    time.sleep(0.5)
                    x, t = eeg._crudo(18.0)
                    giro = eeg.movimiento(t[-1] - 10.0, t[-1])
                retrasos = [local_clock() - t[-1]]
                for _ in range(4):                 # fuera de una perdida de Bluetooth en curso
                    time.sleep(0.1)
                    retrasos.append(local_clock() - eeg.ultimo_t())
                retraso = min(retrasos)
                return x, t, l, ventana_mi, giro, retraso
            finally:
                eeg.cerrar()
        finally:
            g.terminate()
            g.wait()

    resumen = []
    for nombre, extra, entrada in (
            ('puente', [], {}),
            ('unicornlsl por tipo', ['--formato', 'unicornlsl'], {'fuente': 'unicornlsl'}),
            ('unicornlsl por nombre', ['--formato', 'unicornlsl', '--nombre-lsl', 'UN-PRUEBA'],
             {'fuente': 'unicornlsl', 'nombre': 'UN-PRUEBA'})):
        x, t, l, ventana_mi, giro, retraso = con_gemelo(extra, **entrada)
        assert x.shape[0] == 8 and x.shape[1] > 1000, (nombre, x.shape)
        d = np.diff(t)
        assert abs(np.median(d) - 0.004) < 1e-4 and d.min() > 0, (nombre, np.median(d), d.min())
        normales = d[d < 0.018]
        assert np.abs(normales - 0.004).max() < 0.0015, (nombre, np.abs(normales - 0.004).max())   # sin jitter
        assert (d > 0.018).sum() >= 1, f'{nombre}: las perdidas de Bluetooth deben verse como huecos'
        assert -0.05 < retraso < 0.25, (nombre, retraso)
        # el offset de continua del casco no debe parecer un canal saturado ni plano
        assert not set(l['canales'].values()) & {'saturado', 'plano'}, (nombre, l)
        assert l['edad_s'] < 0.3 and 180 < l['tasa_hz'] < 300, (nombre, l)
        assert ventana_mi is not None and ventana_mi.shape == (8, 750), nombre
        assert giro is not None and giro > 20, (nombre, giro)                     # la cabeza se movio
        resumen.append(f'{nombre}: {int((d > 0.018).sum())} huecos')
    # un flujo con otro numero de canales no es esa fuente: error claro, no datos mezclados
    g = subprocess.Popen([sys.executable, 'cerebro_sintetico.py'], cwd=config.RAIZ,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        try:
            hw.EntradaEEG(timeout=20.0, fuente='unicornlsl', nombre='EEG')
            raise AssertionError('debio rechazar un flujo de 8 canales como fuente unicornlsl')
        except RuntimeError as e:
            assert '17' in str(e), e
    finally:
        g.terminate()
        g.wait()
    return '; '.join(resumen) + '; hora por contador sin jitter, IMU y ventana de MI con perdidas'


@prueba
def puente_hora_por_contador():
    """puente_lsl.py (placa sintetica de BrainFlow) estampa cada muestra con la hora reconstruida
    por el contador de la placa: sin el jitter de los bloques de llegada."""
    from pylsl import StreamInlet, resolve_byprop, proc_clocksync, local_clock
    g = subprocess.Popen([sys.executable, 'puente_lsl.py'], cwd=config.RAIZ,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        s = resolve_byprop('name', 'EEG', timeout=20)
        assert s, 'no aparecio el flujo EEG del puente'
        s2 = resolve_byprop('name', 'IMU', timeout=10)
        assert s2 and s2[0].channel_count() == 6, 'el puente debe publicar la IMU en su propio flujo'
        e, ts, t_fin = StreamInlet(s[0], processing_flags=proc_clocksync), [], time.time() + 6
        imu, ti = StreamInlet(s2[0], processing_flags=proc_clocksync), []
        while time.time() < t_fin:
            ts.extend(e.pull_chunk(timeout=0.05)[1])
            ti.extend(imu.pull_chunk(timeout=0.0)[1])
        retraso = local_clock() - ts[-1]
    finally:
        g.terminate()
        g.wait()
    d = np.diff(ts)[250:]
    assert len(ts) > 1000 and np.abs(d - 0.004).max() < 0.001, np.abs(d - 0.004).max()
    assert -0.05 < retraso < 0.3, retraso
    assert len(ti) > 1000 and np.abs(np.diff(ti)[250:] - 0.004).max() < 0.001      # la IMU, con la misma hora
    # el plan de cada placa sale del descriptor de BrainFlow (sin conectar nada)
    import puente_lsl
    u = puente_lsl.plan_placa('unicorn', serie='UN-2023.01.01')
    assert u['params'].serial_number == 'UN-2023.01.01' and u['fs'] == 250 and u['modulo'] is None
    assert (u['eeg'], u['imu'], u['bateria'], u['contador'], u['validez']) == \
        (list(range(8)), list(range(8, 14)), 14, 15, 16), u
    assert puente_lsl.plan_placa('unicorn')['params'].serial_number == ''       # la serie es opcional
    c = puente_lsl.plan_placa('cyton', puerto='COM3')
    assert c['imu'] is None and c['modulo'] == 256 and len(c['eeg']) == 8
    assert puente_lsl.plan_placa('playback', archivo='x.csv')['modulo'] is None  # se reproduce un Unicorn
    return f'{len(ts)} muestras con paso de 4 ms exacto (desvio maximo {1000 * np.abs(d - 0.004).max():.2f} ms)'


@prueba
def verificar_unicorn():
    """verificar_unicorn.py comprueba lo no verificado del casco (orden de canales, unidades,
    contador, acelerometro, bateria, validez) y da un veredicto. Aqui se prueba con el gemelo en
    formato UnicornLSL, con un mapa de canales equivocado y con la placa sintetica de BrainFlow."""
    import verificar_unicorn as vu
    g = subprocess.Popen([sys.executable, 'cerebro_sintetico.py', '--formato', 'unicornlsl', '--nombre-lsl',
                          'UN-PRUEBA', '--cabeza', '0.5', '--parpadeos', '0.8'], cwd=config.RAIZ,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        datos = vu.leer_lsl(nombre=None, tipo='Data', fases=vu.fases(auto=True), salida=lambda *a: None)
    finally:
        g.terminate()
        g.wait()
    assert datos['nombre'] == 'UN-PRUEBA' and datos['x'].shape[0] == 17
    res = vu.evaluar(datos)
    estados = {r['clave']: r['estado'] for r in res}
    for critica in vu.CRITICAS:
        assert estados[critica] == 'OK', (critica, [r for r in res if r['clave'] == critica])
    assert estados['alfa_occipital'] == 'OK' and estados['parpadeo_fz'] in ('OK', 'AVISO')
    texto = vu.veredicto({'lsl': res}, datos['nombre'])
    assert '--fuente unicornlsl' in texto and 'UN-PRUEBA' in texto, texto
    # con el orden de canales equivocado (contador y bateria cambiados) lo dice
    mal = dict(datos, mapa=dict(datos['mapa'], contador=14, bateria=15))
    mal_res = {r['clave']: r for r in vu.evaluar(mal)}
    assert mal_res['contador']['estado'] == 'FALLA' and 'canal 15' in mal_res['contador']['texto'], mal_res['contador']
    assert 'NO uses' in vu.veredicto({'lsl': list(mal_res.values())}, 'UN-PRUEBA')
    # el camino de BrainFlow corre de punta a punta con la placa sintetica (que NO es un Unicorn: debe fallar algo)
    sint = vu.leer_brainflow(serie=None, fases=vu.fases(auto=True)[:1], salida=lambda *a: None, placa='sintetica')
    sres = vu.evaluar(sint)
    assert sint['x'].shape[1] > 500 and any(r['estado'] == 'FALLA' for r in sres)
    assert vu.veredicto({'brainflow': res}, None).count('puente_lsl.py --placa unicorn') == 1
    return 'gemelo en formato UnicornLSL: criticas en OK y veredicto; detecta un mapa de canales equivocado'


@prueba
def registro_huecos():
    """El registro de huecos del puente: rafagas de paquetes perdidos (saltos del contador de
    paquetes de la placa, modulo 256) y silencios de llegada, con su resumen cada 30 s."""
    from salud import RegistroHuecos
    r = RegistroHuecos(fs=250, silencio_s=0.55)
    r.bloque(np.arange(0, 100), t=0.10)                       # sin perdidas
    r.bloque(np.arange(100, 250), t=0.50)
    r.bloque(np.r_[250:256, 0:20], t=1.00)                    # la vuelta 255 -> 0 no es un hueco
    assert r.huecos == [] and r.muestras == 276
    r.bloque(np.arange(30, 60), t=1.20)                       # faltan 20..29: 10 muestras = 40 ms
    r.bloque(np.r_[60:70, 95:120], t=1.40)                    # faltan 70..94 dentro del bloque: 100 ms
    assert [(n, round(d, 3)) for _, n, d in r.huecos] == [(10, 0.04), (25, 0.1)], r.huecos
    r.bloque(np.arange(120, 130), t=2.00)                     # los datos tardaron 0.6 s en llegar
    assert [round(d, 2) for _, d in r.silencios] == [0.6], r.silencios
    txt = r.resumen(t=30.0)
    for trozo in ('2 rafagas', '35 muestras', 'max 100 ms', '1 silencio', '600 ms'):
        assert trozo in txt, (trozo, txt)
    assert '0 rafagas' in r.resumen(t=60.0) and 'sesion: 2 rafagas' in r.resumen(t=60.0)   # ventana nueva
    assert not r.toca_resumen(t=70.0) and r.toca_resumen(t=91.0)
    # contador sin vuelta (Unicorn): un salto son perdidas; si retrocede, el casco se reinicio
    u = RegistroHuecos(fs=250, modulo=None)
    u.bloque(np.arange(1000, 1100), t=0.1)
    u.bloque(np.arange(1130, 1200), t=0.2)                    # 30 perdidas
    u.bloque(np.arange(0, 50), t=0.3)                         # reinicio: no cuenta como perdida
    assert (u.total_perdidas, u.total_rafagas, u.reinicios) == (30, 1, 1), (u.total_perdidas, u.reinicios)
    assert '1 reinicio' in u.resumen(t=30.0)
    return 'cuenta rafagas de paquetes perdidos y silencios; resumen por ventana y de la sesion'


def _bluetooth(fs, contadores, rng, latencia=0.020, deriva_ppm=0.0, t0=1000.0):
    """Bloques (contadores, hora de llegada) como los entrega un casco por Bluetooth: cada
    muestra se toma en t0 + c / fs (con el cristal del casco algo desviado) y llega en
    rafagas, con un retardo minimo fijo mas uno variable."""
    verdad = t0 + np.asarray(contadores) / (fs * (1 + deriva_ppm * 1e-6))
    bloques, i = [], 0
    while i < len(contadores):
        n = int(rng.integers(3, 30))
        retardo = latencia + rng.exponential(0.015) + (0.3 if rng.random() < 0.03 else 0.0)
        bloques.append((contadores[i:i + n], verdad[min(i + n, len(contadores)) - 1] + retardo))
        i += n
    return bloques, verdad


@prueba
def reloj_contador():
    """La hora de cada muestra se reconstruye con el contador del casco, no con la hora de
    llegada por Bluetooth: sin jitter, con las perdidas visibles como huecos y siguiendo la
    deriva del cristal."""
    from salud import RelojContador
    fs, rng = 250, np.random.default_rng(0)
    # 1) flujo normal con llegadas a rafagas (y algunas muy tardias)
    c = np.arange(0, 250 * 120)
    bloques, verdad = _bluetooth(fs, c, rng)
    reloj = RelojContador(fs)
    t = np.concatenate([reloj.estampar(b, llegada) for b, llegada in bloques])
    assert np.all(np.diff(t) > 0)
    err = (t - verdad)[250 * 20:] - 0.020                       # tras 20 s; 20 ms = retardo minimo, constante
    assert np.abs(err).max() < 0.004, np.abs(err).max()
    assert np.abs(np.diff(t)[250 * 20:] - 1 / fs).max() < 0.0005   # sin jitter entre muestras
    # 2) perdida de Bluetooth: faltan 50 muestras; el hueco queda en la hora, no se disimula
    c2 = np.r_[0:5000, 5050:10000]
    bloques, verdad = _bluetooth(fs, c2, rng)
    reloj = RelojContador(fs)
    t = np.concatenate([reloj.estampar(b, llegada) for b, llegada in bloques])
    saltos = np.diff(t)
    assert abs(saltos.max() - 51 / fs) < 0.002 and (saltos > 0.02).sum() == 1
    assert reloj.perdidas == 50 and reloj.reinicios == 0
    # 3) el casco se reinicia: el contador vuelve a 0; la hora sigue adelante y se cuenta el reinicio
    reloj = RelojContador(fs)
    a, va = _bluetooth(fs, np.arange(0, 3000), rng, t0=1000.0)
    b, vb = _bluetooth(fs, np.arange(0, 3000), rng, t0=1020.0)
    t = np.concatenate([reloj.estampar(x, llegada) for x, llegada in a + b])
    assert np.all(np.diff(t) > 0) and reloj.reinicios == 1
    assert np.abs((t[3000:] - vb)[1500:] - 0.020).max() < 0.02
    # 4) deriva del cristal: 100 ppm durante 10 minutos (60 ms acumulados) no se acumula en el error
    for ppm in (100.0, -100.0):
        c = np.arange(0, 250 * 600)
        bloques, verdad = _bluetooth(fs, c, rng, deriva_ppm=ppm)
        reloj = RelojContador(fs)
        t = np.concatenate([reloj.estampar(x, llegada) for x, llegada in bloques])
        err = (t - verdad)[250 * 60:] - 0.020
        assert np.abs(err).max() < 0.008, (ppm, np.abs(err).max())
    # 5) contador con vuelta (placa sintetica o Cyton: 0..255)
    reloj = RelojContador(fs, modulo=256)
    bloques, verdad = _bluetooth(fs, np.arange(0, 5000), rng)
    t = np.concatenate([reloj.estampar(np.asarray(x) % 256, llegada) for x, llegada in bloques])
    assert np.all(np.diff(t) > 0) and reloj.perdidas == 0 and np.abs(np.diff(t) - 1 / fs).max() < 0.002
    assert reloj.estampar([], 5000.0).size == 0
    return 'hora por contador: sin jitter, perdidas como huecos, reinicios y deriva de 100 ppm bajo 8 ms'


@prueba
def puente_reconecta():
    """La reconexion de la placa en puente_lsl.py, con la placa sintetica de BrainFlow.
    NO sustituye la prueba con el Cyton real."""
    import puente_lsl
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from salud import Retroceso
    BoardShim.disable_board_logger()
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, BrainFlowInputParams())
    board.prepare_session()
    board.start_stream(45000, '')
    try:
        board.release_session()                           # "se desconecto el dongle"
        puente_lsl.reconectar(board, Retroceso(0.1, 0.2), grabar=None)
        time.sleep(0.5)
        assert board.get_board_data().shape[1] > 50
    finally:
        try:
            board.release_session()
        except Exception:
            pass
    return 'la placa sintetica vuelve a entregar datos tras reconectar (Cyton real: sin probar)'


@prueba
def cerebro_sintetico():
    """El gemelo del piloto produce MI decodificable y un ErrP con la forma y latencia correctas."""
    import cerebro_sintetico as cs
    import hardware as hw
    X, y = cs.sesion_mi(40, semilla=3)
    ba_mi = hw.DecoderIM().ajustar(X, y).ba
    # montaje del Unicorn: el ERD de imaginar la mano derecha es maximo en C3 y el alfa es occipital
    potencia = lambda sel: X[sel].var(axis=(0, 2))
    erd = potencia(y == 1) / potencia(y == 0)
    assert config.CANALES_EEG[int(np.argmin(erd))] == 'C3', dict(zip(config.CANALES_EEG, erd.round(2)))
    alfa = potencia(y == 0)
    assert alfa[config.indices('alfa')].min() > 2 * alfa[config.indices(['Fz'])[0]], alfa.round(1)
    X, y = cs.sesion_errp(120, semilla=3)
    t = np.arange(X.shape[2]) / cs.FS + config.EPOCA_ERRP[0]
    dif = X[y == 1, config.CANALES_EEG.index('Cz')].mean(0) - X[y == 0, config.CANALES_EEG.index('Cz')].mean(0)
    pe, ne = dif[np.argmin(abs(t - 0.36))], dif[np.argmin(abs(t - 0.25))]
    det = hw.DetectorErrP().ajustar(X, y)
    # tras un error, mas potencia theta (4-8 Hz) en Fz y Cz entre 200 y 600 ms
    theta = hw._potencia_theta(X).mean(axis=1)
    assert theta[y == 1].mean() > theta[y == 0].mean() + 0.15, (theta[y == 1].mean(), theta[y == 0].mean())
    # el ErrP es fronto-central: mayor en los canales de su papel (Fz, Cz, Pz) que en los occipitales
    amp = np.abs(X[y == 1].mean(0) - X[y == 0].mean(0)).max(axis=1)
    assert amp[config.indices('errp')].min() > amp[config.indices('visual')].max(), amp.round(1)
    # det.ba > 0.70 (antes 0.75): el montaje del Unicorn tiene 3 electrodos fronto-centrales, no 5
    assert 0.65 < ba_mi < 1.0 and pe > 2 and ne < 0 and det.ba > 0.70, (ba_mi, pe, ne, det.ba)
    return f'MI BA {ba_mi:.2f}; ErrP Pe {pe:+.1f} uV, Ne {ne:+.1f} uV; detector BA {det.ba:.2f} (espec {det.espec:.2f})'


@prueba
def gemelo_unicorn():
    """El gemelo se comporta como un Unicorn: N1 visual occipital ante cada movimiento, IMU con
    movimientos de cabeza que ensucian el EEG, contador con perdidas de Bluetooth, y los dos
    formatos de salida (flujos del puente y flujo unico de la app UnicornLSL)."""
    import cerebro_sintetico as cs
    from pylsl import StreamInlet, resolve_byprop, proc_clocksync
    oz, fz = config.CANALES_EEG.index('Oz'), config.CANALES_EEG.index('Fz')
    # 1) respuesta visual: N1 ~170 ms en occipital aunque el movimiento sea correcto. Se compara
    #    con la MISMA realizacion sin N1 (tiene su propio generador), para no medir ruido.
    X, y = cs.sesion_errp(80, semilla=1)
    X0, _ = cs.sesion_errp(80, semilla=1, n1=0.0)
    t = np.arange(X.shape[2]) / cs.FS + config.EPOCA_ERRP[0]
    efecto = np.median(X[y == 0], axis=0) - np.median(X0[y == 0], axis=0)
    v = (t > 0.10) & (t < 0.25)
    n1, lat = efecto[oz, v].min(), t[v][np.argmin(efecto[oz, v])]
    en_n1 = efecto[:, np.argmin(np.abs(t - lat))]
    assert n1 < -2.0 and 0.14 < lat < 0.20, (n1, lat)
    assert en_n1[config.indices('visual')].max() < -2.0 and abs(en_n1[fz]) < 1.0, en_n1.round(2)
    # 2) IMU: ~1 g en reposo; al mover la cabeza gira el giroscopio y el EEG se ensucia
    cer, tt, eeg, imu = cs.Cerebro(cs._args(cabeza=0.5, semilla=2)), 0.0, [], []
    for _ in range(120):
        x, tt = cs._bloque(cer, tt, 0.5)
        eeg.append(x); imu.append(cer.imu)
    eeg, imu = np.hstack(eeg), np.vstack(imu)
    assert imu.shape == (eeg.shape[1], 6) and cer.contador == eeg.shape[1]
    assert 0.95 < np.median(np.linalg.norm(imu[:, :3], axis=1)) < 1.05
    moviendo = np.abs(imu[:, 3:]).max(axis=1) > 20
    assert 0.02 < moviendo.mean() < 0.8, moviendo.mean()
    assert np.abs(eeg[:, moviendo]).mean() > 2 * np.abs(eeg[:, ~moviendo]).mean()
    quieto = cs.Cerebro(cs._args())
    cs._bloque(quieto, 0.0, 10.0)
    assert np.abs(quieto.imu[:, 3:]).max() < 5

    def escuchar(extra, clave, valor, segundos, otro=None):
        """Lanza el gemelo, toma `segundos` de su flujo (y de `otro`) y lo cierra."""
        g = subprocess.Popen([sys.executable, 'cerebro_sintetico.py', '--semilla', '5'] + extra, cwd=config.RAIZ,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            s = resolve_byprop(clave, valor, timeout=20)
            assert s, f'no aparecio el flujo {clave}={valor}'
            entradas = [StreamInlet(s[0], processing_flags=proc_clocksync)]
            if otro:
                s2 = resolve_byprop('name', otro, timeout=10)
                assert s2, f'no aparecio el flujo {otro}'
                entradas.append(StreamInlet(s2[0], processing_flags=proc_clocksync))
            datos = [([], []) for _ in entradas]
            t_fin = time.time() + segundos
            while time.time() < t_fin:
                for e, (xs, ts) in zip(entradas, datos):
                    x, tm = e.pull_chunk(timeout=0.05)
                    xs.extend(x); ts.extend(tm)
            return [e.info() for e in entradas], [(np.array(x), np.array(tm)) for x, tm in datos]
        finally:
            g.terminate()
            g.wait()

    # 3) formato del puente: flujos EEG e IMU; una perdida de Bluetooth es un hueco en la hora
    infos, ((x, ts), (xi, ti)) = escuchar(['--perdidas-bt', '120'], 'name', 'EEG', 7.0, otro='IMU')
    assert infos[1].type() == 'IMU' and infos[1].channel_count() == 6 and xi.shape[1] == 6
    d = np.diff(ts)
    assert abs(np.median(d) - 1 / cs.FS) < 0.0005
    huecos = d[d > 0.018]
    assert len(huecos) >= 3 and huecos.max() < 0.6, huecos      # dos perdidas pueden encadenarse
    comun = (max(ts[0], ti[0]), min(ts[-1], ti[-1]))             # EEG e IMU: las mismas muestras
    dentro = lambda tm: int(((tm >= comun[0]) & (tm <= comun[1])).sum())
    assert dentro(ts) > 1000 and abs(dentro(ts) - dentro(ti)) <= 5, (dentro(ts), dentro(ti))
    assert (np.abs(x.mean(axis=0)) > 100).sum() >= 6, x.mean(axis=0)       # offset de continua, como el casco
    # 4) formato UnicornLSL: un flujo 'Data' de 17 canales sin etiquetas, con contador
    f = config.FUENTES_EEG['unicornlsl']
    infos, ((x, ts),) = escuchar(['--formato', 'unicornlsl', '--nombre-lsl', 'UN-PRUEBA', '--perdidas-bt', '120'],
                                 'type', f['tipo'], 6.0)
    info = infos[0]
    assert (info.name(), info.source_id(), info.channel_count(), info.nominal_srate()) == ('UN-PRUEBA', 'UN-PRUEBA', 17, 250)
    assert info.desc().child('channels').empty()
    pasos = np.diff(x[:, f['contador']])
    assert pasos.min() >= 1 and (pasos == 1).mean() > 0.9 and (pasos > 4).any(), np.unique(pasos)
    assert 0.9 < np.median(np.linalg.norm(x[:, f['imu'][:3]], axis=1)) < 1.1
    assert 0 < x[:, f['bateria']].min() <= x[:, f['bateria']].max() <= 100 and (x[:, f['validez']] == 1).all()
    return (f'N1 occipital {n1:.1f} uV a {1000 * lat:.0f} ms; IMU y artefacto de cabeza; perdidas de Bluetooth '
            f'como huecos ({len(huecos)}) y en el contador; formatos puente y UnicornLSL')


@prueba
def lazo_real_sintetico():
    puente = subprocess.Popen([sys.executable, 'cerebro_sintetico.py'], cwd=config.RAIZ,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(2)
        try:
            r = subprocess.run([sys.executable, 'orquestador.py', 'real', '--ortesis-sim', '--forzar',
                                '--ensayos_mi', '24', '--min_mi', '24', '--duracion_mi', '2.5',
                                '--espera', '0.3', '--ensayos_errp', '60',
                                '--seg_revision', '3',
                                '--pasos_estatico', '5', '--pasos_adaptativo', '10'],
                               cwd=config.RAIZ, capture_output=True, text=True, timeout=400)
        except subprocess.TimeoutExpired as e:      # que se vea en que iba, no solo que tardo
            salida = e.stdout.decode(errors='ignore') if isinstance(e.stdout, bytes) else (e.stdout or '')
            raise AssertionError('no termino en 400 s; ultimas lineas:\n' + '\n'.join(
                l for l in salida.splitlines() if 'INFO' not in l)[-1500:])
        assert r.returncode == 0, r.stderr[-1500:]
        assert 'EVALUACION' in r.stdout and '[CP3]' in r.stdout, r.stdout[-1500:]
    finally:
        puente.terminate()
    cps = [l for l in r.stdout.splitlines() if l.startswith('[CP')]
    pausas = r.stdout.count('[PAUSA SEGURA] motivo')
    excluidos = next(l.strip() for l in r.stdout.splitlines() if 'excluidos del analisis' in l)
    assert pausas == 0, f'{pausas} pausas seguras en una corrida sin fallas:\n' + '\n'.join(
        l for l in r.stdout.splitlines() if '[salud]' in l or 'PAUSA' in l)
    return ('cerebro sintetico + ortesis simulada: ' + ' | '.join(
        c.split(']')[0][1:] + (' GO' if 'GO' in c and 'NO GO' not in c else ' NO GO') for c in cps)
        + f'; sin pausas en falso; {excluidos}')


@prueba
def lazo_real_caos():
    """El camino real contra el gemelo con el caos estandar (reutiliza los modelos que
    acaba de calibrar lazo_real_sintetico). Cifras del gemelo, no de una persona."""
    puente = subprocess.Popen([sys.executable, 'cerebro_sintetico.py', '--caos', '1'], cwd=config.RAIZ,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(2)
        r = subprocess.run([sys.executable, 'orquestador.py', 'real', '--ortesis-sim', '--forzar',
                            '--saltar-calibracion', '--caos', '1', '--seg_revision', '3',
                            '--pasos_estatico', '30', '--pasos_adaptativo', '120'],
                           cwd=config.RAIZ, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr[-1500:]
        assert 'EVALUACION' in r.stdout and 'Traceback' not in r.stderr, r.stderr[-1500:]
    finally:
        puente.terminate()
    pausas = r.stdout.count('[PAUSA SEGURA] motivo')
    assert pausas >= 1, 'el caos estandar debio provocar al menos una pausa segura'
    assert r.stdout.count('se reanuda') == pausas, 'alguna pausa no termino'
    linea = lambda clave: next((l.strip() for l in r.stdout.splitlines() if clave in l), 'sin dato')
    return f"{pausas} pausas, todas reanudadas; {linea('excluidos del analisis')}; {linea('[CP4]')}"


@prueba
def parpadeos_cruzan_bloques():
    """En vivo el gemelo genera bloques de ~20 ms (5 muestras) y un parpadeo dura 0.3 s: tiene
    que seguir de un bloque al otro. Antes se recortaba al bloque y casi no habia parpadeos."""
    import cerebro_sintetico as cs
    fz = config.CANALES_EEG.index('Fz')

    def correr(parpadeo):
        cer, t, xs = cs.Cerebro(cs._args(parpadeos=0.0, semilla=3)), 0.0, []
        if parpadeo:
            cer.parpadeos_vivos.append(0.5)          # un parpadeo que empieza a los 0.5 s
        for _ in range(100):                          # 2 s en bloques de 5 muestras
            x, t = cs._bloque(cer, t, 0.02)
            assert x.shape[1] == 5
            xs.append(x)
        return np.hstack(xs)
    # misma realizacion con y sin el parpadeo: la diferencia es el parpadeo solo
    d = correr(True) - correr(False)
    forma = np.abs(d[fz])
    assert 100 < forma.max() < 160, forma.max()                       # ~120 uV en Fz (la mezcla entre electrodos lo ajusta)
    assert 65 <= (forma > 1.0).sum() <= 80, (forma > 1.0).sum()       # ~0.3 s a 250 Hz, no 5 muestras
    assert np.abs(d[fz]).max() > 3 * np.abs(d[config.CANALES_EEG.index('Oz')]).max()   # frontal
    assert len(cs.Cerebro(cs._args()).parpadeos_vivos) == 0
    # con la tasa de verdad, en bloques de 5 muestras, los parpadeos aparecen y son maximos en Fz
    import hardware as hw
    cer, t, xs = cs.Cerebro(cs._args(parpadeos=0.6, semilla=1)), 0.0, []
    for _ in range(1000):                             # 20 s
        x, t = cs._bloque(cer, t, 0.02)
        xs.append(x)
    pp = np.ptp(hw.filtrar(np.hstack(xs), (1.0, 10.0), cs.FS), axis=1)
    assert int(np.argmax(pp)) == fz and pp[fz] > 80, pp.round(0)
    return f'parpadeo de {int((forma > 1.0).sum())} muestras en bloques de 5; en vivo, maximo en Fz ({pp[fz]:.0f} uV pico a pico)'


@prueba
def lazo_real_memoria():
    """Memoria entre sesiones por el camino real contra el gemelo (reutiliza los modelos que calibro
    lazo_real_sintetico): una sesion deja su memoria con --guardar-memoria y la siguiente arranca de
    ella con --desde-sesion y las dos calibraciones cortas. Cifras del gemelo, no de una persona."""
    import memoria
    comun = ['--ortesis-sim', '--forzar', '--guardar-memoria', '--seg_revision', '3', '--pasos_estatico', '5',
             '--sin-cuestionario']
    puente = subprocess.Popen([sys.executable, 'cerebro_sintetico.py'], cwd=config.RAIZ,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(2)

        def sesion(*extra):
            r = subprocess.run([sys.executable, 'orquestador.py', 'real', *comun, *extra],
                               cwd=config.RAIZ, capture_output=True, text=True, timeout=400)
            assert r.returncode == 0 and 'Traceback' not in r.stderr, r.stderr[-1500:]
            ruta = next(l.split('Memoria de la sesion: ')[1].split('  (')[0] for l in r.stdout.splitlines()
                        if 'Memoria de la sesion: ' in l)
            return r.stdout, ruta
        _, ruta = sesion('--saltar-calibracion', '--pasos_adaptativo', '30')
        m = memoria.cargar(ruta)
        salida, ruta2 = sesion('--desde-sesion', ruta, '--duracion_mi', '2.5', '--espera', '0.3', '--pasos_adaptativo', '10')
    finally:
        puente.terminate()
    cp = {n: next(l for l in salida.splitlines() if l.startswith(f'[CP{n}]')) for n in (2, 3)}
    n, k = config.MEMORIA_ENSAYOS_MI, config.MEMORIA_EPOCAS_ERRP
    assert 'Calibracion corta de MI' in salida and 'largo fijo' in cp[2] and 'memoria de sesion' in cp[2], cp[2]
    assert f'con {k} epocas' in cp[3] and 'memoria de' in cp[3], cp[3]                 # el CP3, solo con las de hoy
    m2 = memoria.cargar(ruta2)                                                         # la memoria nueva lleva las dos sesiones
    assert len(m2['mi']['y']) > len(m['mi']['y']) and len(m2['errp']['y']) >= len(m['errp']['y']) + k - 5
    return (f"sesion 1 deja {len(m['mi']['y'])} ensayos de MI y {len(m['errp']['y'])} epocas; sesion 2 calibra con {n} y {k}: "
            + ' | '.join(c.split(' -> ')[0][:60] + ' -> ' + c.split(' -> ')[-1] for c in cp.values()))


# ------------------------------------------------------------ estado del sistema
@prueba
def estado_sistema():
    """Revision previa y franja del tablero: cada revision es una funcion pura sobre lecturas, asi
    que se prueba sin casco, sin ortesis, sin red y sin llave. Toda falla trae su solucion en una linea."""
    import os
    import tempfile
    from pathlib import Path
    import estado_sistema as es
    import puente_lsl
    todos = []

    def estados(resultados):
        todos.extend(resultados)
        return {r['clave']: r['estado'] for r in resultados}
    f = lambda nombre, tipo, canales=8, hz=250, host='pc': {'nombre': nombre, 'tipo': tipo, 'canales': canales, 'hz': hz, 'host': host}
    # flujos: sin EEG, dos fuentes a la vez, flujos repetidos y tasa baja
    assert estados(es.revisar_flujos([f('Estado', 'Markers', 1, 0)])) == {'flujos': 'FALLA'}
    dos = es.revisar_flujos([f('EEG', 'EEG'), f('UN-2023.01.01', 'Data', 17, host='otra')])
    assert estados(dos)['fuentes_eeg'] == 'FALLA' and 'otra' in dos[1]['texto'] and 'cierra' in dos[1]['solucion']
    e = estados(es.revisar_flujos([f('EEG', 'EEG'), f('IMU', 'IMU', 6), f('Estado', 'Markers', 1, 0), f('Estado', 'Markers', 1, 0)],
                                  {'EEG': (249.0, 250.0), 'IMU': (180.0, 250.0)}))
    assert e == {'flujos': 'OK', 'fuentes_eeg': 'OK', 'flujos_repetidos': 'FALLA', 'tasas': 'FALLA'}, e
    assert estados(es.revisar_flujos([f('EEG', 'EEG')], {'EEG': (None, 250.0)}))['tasas'] == 'FALLA'
    # casco: bateria, validez, perdidas por contador y calidad por canal
    bien = [{'canal': c, 'rms_uv': 10.0, 'red': 0.1, 'saturado': 0.0, 'ok': True} for c in config.CANALES_EEG]
    casco = lambda **k: estados(es.revisar_casco({'bateria': 80.0, 'validas': 1.0, 'perdidas': 0.0, 'calidad': bien,
                                                  'edad_s': 0.05, **k}))
    assert set(casco().values()) == {'OK'} and set(casco()) == {'bateria', 'validez', 'perdidas', 'canales'}
    assert casco(bateria=20.0)['bateria'] == 'AVISO' and casco(bateria=10.0)['bateria'] == 'FALLA'
    assert casco(bateria=None, validas=None) == {'bateria': 'AVISO', 'perdidas': 'OK', 'canales': 'OK'}
    assert casco(validas=0.9)['validez'] == 'AVISO'
    assert casco(perdidas=0.02)['perdidas'] == 'AVISO' and casco(perdidas=0.2)['perdidas'] == 'FALLA'
    assert casco(edad_s=4.0)['perdidas'] == 'FALLA'
    malo = es.revisar_casco({'bateria': 80.0, 'validas': 1.0, 'perdidas': 0.0, 'edad_s': 0.05,
                             'calidad': bien[:2] + [dict(bien[2], rms_uv=140.0, ok=False)] + bien[3:]})[-1]
    assert malo['estado'] == 'FALLA' and 'Cz' in malo['texto'] and 'electrodo' in malo['solucion']
    t = np.r_[np.arange(250), np.arange(275, 500)] / 250.0          # faltan 25 de 500 muestras
    assert abs(es.fraccion_perdida(t, 250) - 25 / 500) < 1e-9 and es.fraccion_perdida(t[:250], 250) == 0.0
    # lo que deja puente_lsl.py --estado: vigente se usa, viejo no
    original = config.ESTADO_PUENTE_JSON
    config.ESTADO_PUENTE_JSON = Path(tempfile.mkdtemp()) / 'estado_puente.json'
    try:
        assert es.leer_estado_puente() is None
        d = puente_lsl.estado_puente('unicorn', 64.0, 5000, 10)
        config.ESTADO_PUENTE_JSON.write_text(json.dumps(d))
        assert es.leer_estado_puente()['bateria'] == 64.0
        config.ESTADO_PUENTE_JSON.write_text(json.dumps(dict(d, t=time.time() - 120)))
        assert es.leer_estado_puente() is None
    finally:
        config.ESTADO_PUENTE_JSON = original
    # procesos: el lanzador del .venv y su hijo cuentan una vez; una terminal que menciona el guion, no
    py = r'C:\ortesis-bci\.venv\Scripts\python.exe'
    procesos = [(10, 1, f'{py} cerebro_sintetico.py --caos 1'), (11, 10, f'{py} cerebro_sintetico.py --caos 1'),
                (12, 1, 'bash -c "python cerebro_sintetico.py & python puente_lsl.py"'),
                (13, 1, f'"{py}" -u "C:\\ortesis-bci\\tablero.py" --narrador'), (14, 1, r'"C:\LabRecorder\LabRecorder.exe"')]
    assert es.vivos(procesos, 'cerebro_sintetico.py') == [11] and es.vivos(procesos, 'puente_lsl.py') == []
    assert es.vivos(procesos, 'tablero.py') == [13] and es.vivos(procesos, 'LabRecorder') == [14]
    assert estados(es.revisar_procesos(procesos)) == {'procesos': 'OK'}
    assert estados(es.revisar_procesos(procesos[:4])) == {'procesos': 'AVISO'}          # sin LabRecorder
    dos = es.revisar_procesos(procesos + [(20, 1, f'{py} puente_lsl.py --placa unicorn')])[0]
    todos.append(dos)
    assert dos['estado'] == 'FALLA' and '/PID 20' in dos['solucion'] and 'nunca taskkill /IM' in dos['solucion']
    assert estados(es.revisar_procesos(None)) == {'procesos': 'AVISO'}
    # ACK: los umbrales del CP1
    assert estados(es.revisar_ack([8.0] * 3)) == {'ack': 'AVISO'}
    assert estados(es.revisar_ack([8.0, 9.0] * 10)) == {'ack': 'OK'}
    assert estados(es.revisar_ack([8.0, 60.0, 30.0, 95.0] * 5)) == {'ack': 'FALLA'}
    # laptop: cargador, suspension (powercfg en cualquier idioma) y tapa
    salida = ('Indice de configuracion de corriente alterna actual: 0x00000708\n'
              'Indice de configuracion de corriente continua actual: 0x00000384\n')
    assert es.indices_powercfg('Minimo: 0x00000000\nMaximo: 0xffffffff\n' + salida) == (1800, 900)
    assert es.indices_powercfg('') is None and es.indices_powercfg(None) is None
    assert set(estados(es.revisar_laptop(True, 100, {'suspension': (0, 900), 'tapa': (0, 0)})).values()) == {'OK'}
    r = es.revisar_laptop(True, 100, {'suspension': (1800, 900), 'tapa': (1, 1)})
    assert estados(r) == {'cargador': 'OK', 'suspension': 'FALLA', 'tapa': 'AVISO'}
    assert '30 min' in r[1]['texto'] and r[1]['solucion'] == 'powercfg /change standby-timeout-ac 0'
    r = es.revisar_laptop(False, 41, {'suspension': (0, 900), 'tapa': None})
    assert estados(r) == {'cargador': 'FALLA', 'suspension': 'FALLA'} and '41 %' in r[0]['texto']
    assert r[1]['solucion'].endswith('standby-timeout-dc 0')
    assert estados(es.revisar_laptop(None, None, {'suspension': None, 'tapa': None})) == {'cargador': 'AVISO', 'suspension': 'AVISO'}
    assert [estados(es.revisar_disco(g))['disco'] for g in (0.5, 3.0, 50.0)] == ['FALLA', 'AVISO', 'OK']
    # API: una llamada minima con el modelo del contrato y sin parametros que ese modelo rechaza
    api = ApiSimulada(_resp('ok'))
    assert estados(es.revisar_api(api)) == {'env': 'OK', 'api': 'OK'}
    pet = api.peticiones[0]
    assert pet['model'] == config.IA_MODELO and not {'temperature', 'thinking', 'tool_choice'} & set(pet)
    AuthenticationError = type('AuthenticationError', (Exception,), {})
    r = es.revisar_api(ApiSimulada(AuthenticationError('401 invalid x-api-key')))
    assert estados(r) == {'env': 'OK', 'api': 'FALLA'} and 'llave' in r[1]['solucion']
    assert estados(es.revisar_api(ApiSimulada(TimeoutError('lenta')))) == {'env': 'OK', 'api': 'FALLA'}
    assert estados(es.revisar_api(ApiSimulada(), llamar=False)) == {'env': 'OK'}
    llave, env = os.environ.pop('ANTHROPIC_API_KEY', None), config.ARCHIVO_ENV
    config.ARCHIVO_ENV = Path(tempfile.mkdtemp()) / '.env'
    try:                                                    # sin llave no es una falla: la IA usa plantillas
        r = es.revisar_api()
        assert estados(r) == {'env': 'AVISO'} and 'ANTHROPIC_API_KEY' in r[0]['solucion']
    finally:
        config.ARCHIVO_ENV = env
        if llave is not None:
            os.environ['ANTHROPIC_API_KEY'] = llave
    # git
    g = {'rama': 'main', 'version': 'v-demo-13-gce4c633', 'sucios': 0, 'atras': 0}
    assert estados(es.revisar_git(g)) == {'git': 'OK'} and 'v-demo' in es.revisar_git(g)[0]['texto']
    assert estados(es.revisar_git(dict(g, sucios=2))) == {'git': 'AVISO'}
    assert estados(es.revisar_git(dict(g, atras=3))) == {'git': 'AVISO'}
    assert estados(es.revisar_git(dict(g, version=''))) == {'git': 'AVISO'}
    # toda revision que no esta bien dice que hacer, en una linea; las que estan bien, nada
    for r in todos:
        assert bool(r['solucion']) == (r['estado'] != 'OK') and '\n' not in r['solucion'], r
    assert '-> ' in es.texto(todos)
    # la franja del tablero: apagada por defecto; con resultados, las fallas primero y con su solucion
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from pyqtgraph.Qt import QtWidgets
    import tablero
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    t = tablero.Tablero()
    try:
        t.timer.stop()
        assert t.lbl_sistema.isHidden()
        t._franja_sistema(es.revisar_laptop(False, 41, {'suspension': (0, 0), 'tapa': None}) + es.revisar_disco(3.0))
        txt = t.lbl_sistema.text()
        assert 'conecta el cargador' in txt and txt.index('FALLA') < txt.index('AVISO'), txt
        assert tablero.COLORES_SISTEMA['FALLA'] in txt and 'cargador' in t.lbl_sistema.toolTip()
    finally:
        t.close()
    n = {e: sum(r['estado'] == e for r in todos) for e in ('OK', 'AVISO', 'FALLA')}
    return f"{len(todos)} revisiones puras ({n['FALLA']} fallas y {n['AVISO']} avisos, todos con solucion); franja del tablero apagada por defecto"


@prueba
def estado_sistema_lsl():
    """La revision contra el gemelo como la app UnicornLSL (bateria, validez y perdidas van en el flujo)
    y la alerta de dos fuentes de EEG al aparecer otra. Cifras del gemelo."""
    import estado_sistema as es
    from pylsl import StreamInfo, StreamOutlet
    gemelo = subprocess.Popen([sys.executable, 'cerebro_sintetico.py', '--formato', 'unicornlsl', '--cabeza', '0'],
                              cwd=config.RAIZ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    monitor = es.Monitor()
    try:
        time.sleep(3)
        r = {x['clave']: x for x in monitor.ciclo(espera_s=config.ESTADO_SISTEMA['ventana_s'] + 1)}
        assert r['bateria']['estado'] == 'OK' and '87' in r['bateria']['texto'], r['bateria']
        assert r['validez']['estado'] == 'OK' and r['fuentes_eeg']['estado'] == 'OK', (r['validez'], r['fuentes_eeg'])
        assert r['perdidas']['estado'] == 'OK' and r['tasas']['estado'] == 'OK', (r['perdidas'], r['tasas'])
        assert r['canales']['estado'] == 'OK', r['canales']
        assert {'procesos', 'ack', 'cargador', 'suspension', 'disco', 'env', 'git'} <= set(r), sorted(r)
        otra = StreamOutlet(StreamInfo('EEG_olvidado', 'EEG', 8, 250, 'float32', 'estado-prueba'))
        time.sleep(0.5)
        dos = {x['clave']: x for x in es.revisar_flujos(es.listar_flujos())}
        assert dos['fuentes_eeg']['estado'] == 'FALLA' and 'EEG_olvidado' in dos['fuentes_eeg']['texto'], dos
        del otra
    finally:
        monitor.cerrar()
        gemelo.terminate()
    return f"gemelo como UnicornLSL: {r['bateria']['texto']}, {r['perdidas']['texto']}; dos fuentes de EEG -> FALLA"


# ------------------------------------------------------------ memoria entre sesiones
@prueba
def memoria_sesiones():
    """Memoria entre sesiones (--guardar-memoria y --desde-sesion, apagados por defecto), con dos
    sesiones del mismo sujeto del gemelo: otro ruido y otra ganancia por electrodo, un cambio que
    programamos nosotros. Aqui solo se prueba el mecanismo; lo medido esta en estudios/memoria_sesiones.py."""
    import copy
    import tempfile
    import types
    from pathlib import Path
    import cerebro_sintetico as cs
    import hardware as hw
    import memoria
    import orquestador
    from agente_errp import AgenteErrP, ConfianzaDetector
    a = orquestador.argumentos(['real'])
    assert a.desde_sesion is None and not a.guardar_memoria                        # apagado por defecto
    assert config.MEMORIA_ENSAYOS_MI < a.min_mi and config.MEMORIA_EPOCAS_ERRP < a.ensayos_errp
    # el gemelo sin 'sesion' es el de siempre; con sesion = 1, el mismo sujeto en otra sesion
    X0 = cs.sesion_mi(3, semilla=3)[0]
    assert np.array_equal(X0, cs.sesion_mi(3, semilla=3, sesion=0)[0])
    assert not np.allclose(X0, cs.sesion_mi(3, semilla=3, sesion=1)[0])
    # sesion previa -> memoria -> archivo
    s = 1
    Xa, ya = cs.sesion_mi(40, semilla=s)
    Ea, ea = cs.sesion_errp(100, semilla=s)
    dec = hw.DecoderIM().ajustar(Xa, ya, config.candidatos('decoder'))
    det = hw.DetectorErrP(canales=config.indices('errp'), vistas='tres').ajustar(Ea, ea, evaluar=False)
    det.sens, det.espec, det.ba, det.eleccion = 0.7, 0.9, 0.8, 'Fz/Cz/Pz, tres vistas'
    ag = AgenteErrP(dec.w0, dec.c0)
    ag.beta = 2.1                                           # la beta final: con la perturbacion de la demo compensada
    m = memoria.construir(dec, det, memoria.mi_de_sesion(Xa, ya), Ea, ea, ag.a_dict(), ConfianzaDetector().a_dict(),
                          beta_sin_perturbar=0.15, origen='sesion_real_prueba.csv')
    carpeta = Path(tempfile.mkdtemp())
    ruta = memoria.guardar(carpeta / 'sesion_real_prueba.csv', m)
    assert ruta.name == 'sesion_real_prueba' + config.SUFIJO_MEMORIA
    m = memoria.cargar(carpeta / 'sesion_real_prueba.csv')                         # por el CSV o por el archivo
    assert memoria.cargar(ruta)['origen'] == 'sesion_real_prueba.csv' and 'MISMO piloto' in memoria.texto(m)
    assert m['mi']['Z'].shape == (40, 36) and m['agente']['beta'] == 2.1
    assert memoria.beta_inicial(m, 6.0) == 0.15, 'el agente parte de la beta de antes de perturbar, nunca de la final'
    assert memoria.beta_inicial({'agente': {'beta_sin_perturbar': None}}, 6.0) is None
    assert memoria.beta_inicial({'agente': {'beta_sin_perturbar': 9.0}}, 6.0) == 6.0
    for mala, error in ((carpeta / 'no_existe.csv', FileNotFoundError), (None, ValueError)):
        if mala is None:                                    # memoria grabada con otro montaje
            mala = memoria.guardar(carpeta / 'otro.csv', dict(m, mi=dict(m['mi'], canales=['Cz'])))
        try:
            memoria.cargar(mala)
            assert False, 'debio rechazarla'
        except error:
            pass
    # MI: calibracion corta de la sesion nueva; se prueba en ensayos posteriores que no ajustaron nada
    n = config.MEMORIA_ENSAYOS_MI
    Xb, yb = cs.sesion_mi(60, semilla=s, sesion=1)
    Xp, yp = Xb[24:], yb[24:]
    ba = lambda d: hw.exactitud_balanceada(yp, np.array([int(d.w0 @ d.phi(x) + d.c0 >= 0) for x in Xp]))
    con = hw.DecoderIM().ajustar_desde(m['mi'], Xb[:n], yb[:n], peso=config.MEMORIA_PESO_NUEVO)
    assert con.eleccion == 'memoria de sesion' and len(con.pred_cv) == n and 0 <= con.ba <= 1   # CP2: solo ensayos de hoy
    b_con = ba(con)
    b_cero = ba(hw.DecoderIM().ajustar(Xb[:n], yb[:n], config.candidatos('decoder')))
    b_previo = ba(copy.deepcopy(dec))
    assert b_con > 0.65 and b_con >= b_cero - 0.05, (b_con, b_cero, b_previo)
    encadenada = memoria.mi_de_sesion(Xb[:n], yb[:n], m['mi'])                     # la memoria de la sesion nueva lleva las dos
    assert encadenada['Z'].shape == (40 + n, 36) and memoria.mi_de_sesion() is None
    # ErrP: epocas previas + las de hoy; lo que ve el CP3 son SOLO las de hoy
    k = config.MEMORIA_EPOCAS_ERRP
    Eb, eb = cs.sesion_errp(100, semilla=s, sesion=1)
    d = memoria.detector_con_memoria(m['detector'], m['errp']['X'], m['errp']['y'], Eb[:k], eb[:k], np.arange(k) % 2)
    assert len(d.pred_cv) == k and np.array_equal(d.y_cal, eb[:k]) and 0 <= d.ba <= 1
    assert d.canales == det.canales and d.vistas == det.vistas and 'memoria de 100 epocas' in d.eleccion
    assert set(d.por_direccion) >= {'cerrar', 'abrir'}
    b_det = hw.exactitud_balanceada(eb[k:], np.array([int(d.p_error(e) > d.umbral) for e in Eb[k:]]))
    assert b_det > 0.6, b_det
    # el orquestador: --desde-sesion carga la memoria y acorta las dos calibraciones; el agente parte de su beta
    b = orquestador.BackendReal.__new__(orquestador.BackendReal)
    b.a, b.hw, vistos = orquestador.argumentos(['real', '--desde-sesion', str(ruta)]), hw, []
    b.revisar = lambda orq: True
    b.calibrar_mi = lambda orq: vistos.append((b.a.ensayos_mi, b.a.min_mi)) or setattr(b, 'decoder', dec) or True
    b.calibrar_errp = lambda orq: vistos.append(b.a.ensayos_errp) or setattr(b, 'detector', d) or True
    b.preparar_coadaptacion = lambda orq: None
    falso = types.SimpleNamespace(inst=None, b=b, a=b.a, fsm=types.SimpleNamespace(ir_a=lambda e: None))
    assert orquestador.Orquestador.preparar(falso)
    assert vistos == [(n, n), k] and b.memoria['origen'] == 'sesion_real_prueba.csv', vistos
    assert falso.agente.beta == 0.15 and falso.agente.var == falso.agente.cfg.varianza_inicial
    # --guardar-memoria: del agente, la beta de antes de perturbar; nunca lanza; el simulador no aplica
    b.cal_mi, b.coadapta = (Xb[:n], yb[:n]), None
    original = config.MODELOS
    config.MODELOS = carpeta
    try:
        np.savez(carpeta / 'detector_errp_datos.npz', X=Eb[:k], y=eb[:k])
        falso.ruta_csv, falso.sham, falso.beta_pre, falso.confianza = carpeta / 'sesion_real_dos.csv', None, 0.4, ConfianzaDetector()
        falso.agente.beta = 2.3
        orquestador.Orquestador.guardar_memoria(falso)
        m2 = memoria.cargar(falso.ruta_csv)
        assert memoria.beta_inicial(m2, 6.0) == 0.4 and len(m2['mi']['y']) == 40 + n and len(m2['errp']['y']) == k
        falso.sham = {'orden': ['real', 'sham']}            # control causal: dos perturbaciones, no hay una beta limpia
        orquestador.Orquestador.guardar_memoria(falso)
        assert memoria.beta_inicial(memoria.cargar(falso.ruta_csv), 6.0) is None
        b.datos_memoria = lambda: 1 / 0
        orquestador.Orquestador.guardar_memoria(falso)      # un fallo al guardar no rompe el final de la sesion
        orquestador.Orquestador.guardar_memoria(types.SimpleNamespace(b=object()))
        # memoria de una sesion ya corrida sin la bandera (la del domingo): con lo que quedo en modelos/ y resultados/
        resultados, instantanea = config.RESULTADOS, config.ESTADO_SESION_JSON
        config.RESULTADOS, config.ESTADO_SESION_JSON = carpeta, carpeta / 'estado_sesion.json'
        try:
            hw.guardar(dec, 'decoder_im.pkl')
            hw.guardar(det, 'detector_errp.pkl')
            np.savez(carpeta / 'calibracion_mi_100.npz', X=Xa, y=ya)
            np.savez(carpeta / 'calibracion_mi_200.npz', X=Xb, y=yb)               # de otra calibracion: no es la del decoder
            config.ESTADO_SESION_JSON.write_text(json.dumps({
                'ruta_csv': 'resultados/sesion_real_tres.csv', 'terminada': True, 'sham': None, 'beta_pre': 0.3,
                'agente': ag.a_dict(), 'confianza': ConfianzaDetector().a_dict()}))
            m3, notas, csv_ = memoria.desde_archivos()
            assert csv_.name == 'sesion_real_tres.csv' and len(m3['mi']['y']) == 40 and memoria.beta_inicial(m3, 6.0) == 0.3
            assert any('calibracion_mi_100.npz' in t for t in notas)
            m4 = memoria.desde_archivos('resultados/sesion_real_otra.csv')[0]      # la instantanea es de otra sesion
            assert memoria.beta_inicial(m4, 6.0) is None and m4['confianza'] is None
        finally:
            config.RESULTADOS, config.ESTADO_SESION_JSON = resultados, instantanea
    finally:
        config.MODELOS = original
    return (f'gemelo, 1 sujeto en otra sesion: con {n} ensayos de MI, BA desde cero {b_cero:.2f}, con memoria {b_con:.2f} '
            f'(decoder previo sin recalibrar {b_previo:.2f}); detector con memoria y {k} epocas de hoy BA {b_det:.2f} '
            f'(el CP3 reportaria {d.ba:.2f})')


# ------------------------------------------------------------ modo demo automatico (demo.py)
def _grabar_estado_falso(ruta, tipos, mtime=None, **extra):
    """Escribe una sesion grabada falsa (_estado.jsonl, como Salidas._registrar): una linea {'t', 'evento'} por evento,
    con la hora creciente, y la fija con `mtime` si se da. `tipos` son tipos de evento ('paso', 'ajeno', 'checkpoint',
    'detenida'...); un 'paso' o 'ajeno' trae su numero. Para probar el plan B sin correr una sesion."""
    import json
    lineas = []
    for k, tipo in enumerate(tipos):
        ev = {'tipo': tipo, **extra.get(tipo, {})}
        if tipo in ('paso', 'ajeno'):
            ev.setdefault('paso', k + 1)
        lineas.append(json.dumps({'t': float(k + 1), 'evento': ev}))
        lineas.append(json.dumps({'t': float(k + 1), 'marcador': f'm{k}'}))          # los marcadores tambien van en el archivo
    ruta.write_text('\n'.join(lineas) + '\n', encoding='utf-8')
    if mtime is not None:
        os.utime(ruta, (mtime, mtime))
    return ruta


class _ProcesosFalsos:
    """Hace de demo.Procesos sin lanzar nada: anota que se pidio, en que orden, y si se cerro."""

    def __init__(self, carpeta, mueren=()):
        from pathlib import Path
        self.carpeta, self.pedidos, self.mueren, self.cerrado = Path(carpeta), [], set(mueren), False

    def lanzar(self, nombre, script, args):
        self.pedidos.append((nombre, script, list(args)))

    def vivo(self, nombre):
        return nombre not in self.mueren

    def cola(self, nombre, lineas=12):
        return f'cola de {nombre}'

    def cerrar_todos(self, espera_s=8.0):
        self.cerrado = True
        return {n: 'limpio' for n, _, _ in self.pedidos}


@prueba
def demo_comandos():
    """Los comandos de cada plan: puros, sin banderas que el piloto no pidio, y con banderas que existen."""
    import demo
    a = demo.argumentos(['lanzar', '--plan', 'casco', '--ortesis-sim', '--serie', 'UN-2019.05.51', '--narrador', '--copiloto', '--idioma', 'en'])
    cmd = demo.comandos('casco', a, ahora=lambda: 0)
    assert cmd['fuente'][0] == 'puente_lsl.py' and cmd['fuente'][1][:4] == ['--placa', 'unicorn', '--serie', 'UN-2019.05.51']
    assert '--grabar' in cmd['fuente'][1] and cmd['fuente'][1][-1].endswith('.csv')
    assert cmd['tablero'] == ('tablero.py', ['--copiloto', '--narrador']) and cmd['narrador'] == ('narrador.py', ['--idioma', 'en'])
    assert cmd['orquestador'] == ('orquestador.py', ['real', '--ortesis-sim'])
    # la ortesis real lleva su puerto; sin tablero ni narrador no hay esos procesos
    a = demo.argumentos(['lanzar', '--plan', 'gemelo', '--puerto', 'COM7', '--sin-tablero'])
    cmd = demo.comandos('gemelo', a)
    assert cmd['fuente'] == ('cerebro_sintetico.py', []) and 'tablero' not in cmd and 'narrador' not in cmd
    assert cmd['orquestador'] == ('orquestador.py', ['real', '--puerto', 'COM7'])
    # la app UnicornLSL es la fuente: no se lanza ningun puente
    a = demo.argumentos(['lanzar', '--plan', 'unicornlsl', '--ortesis-sim', '--eeg-nombre', 'UN-1'])
    cmd = demo.comandos('unicornlsl', a)
    assert 'fuente' not in cmd and cmd['orquestador'][1] == ['real', '--ortesis-sim', '--fuente', 'unicornlsl', '--eeg-nombre', 'UN-1']
    # lo de tras `--` llega tal cual al orquestador; sin extras, nada de --forzar ni --saltar-calibracion
    a = demo.argumentos(['lanzar', '--plan', 'gemelo', '--ortesis-sim', '--', '--sham', '--preentrenado'])
    assert a.extras == ['--sham', '--preentrenado']
    assert demo.comandos('gemelo', a, a.extras)['orquestador'][1][-2:] == ['--sham', '--preentrenado']
    for plan in demo.PLANES:
        a = demo.argumentos(['lanzar', '--plan', plan, '--ortesis-sim', '--narrador'])
        todo = [x for _, args in demo.comandos(plan, a).values() for x in args]
        assert '--forzar' not in todo and '--saltar-calibracion' not in todo, plan
    # sin abreviaturas: `--ortesis` es del orquestador, no una forma corta de --ortesis-sim
    a = demo.argumentos(['lanzar', '--ortesis', 'x'])
    assert not a.ortesis_sim and a.extras == ['--ortesis', 'x']
    try:
        demo.comandos('otro', a)
        raise AssertionError('un plan desconocido debia fallar')
    except ValueError:
        pass
    # el cierre limpio pasa por `demo.py hijo`, que corre el script del proyecto
    argv = demo.argv_de('orquestador.py', ['real'])
    assert argv[1].endswith('demo.py') and argv[2:] == ['hijo', 'orquestador.py', 'real']
    # contrato con los demas scripts: las banderas que demo.py usa existen en ellos
    esperadas = {'puente_lsl.py': ['--placa', '--serie', '--grabar'], 'orquestador.py': ['--ortesis-sim', '--puerto', '--fuente', '--eeg-nombre'],
                 'tablero.py': ['--copiloto', '--narrador', '--flechas'], 'narrador.py': ['--idioma'],
                 'repetir_sesion.py': ['--ultima', '--velocidad', '--puerto']}
    for script, banderas in esperadas.items():
        texto = (config.RAIZ / script).read_text(encoding='utf-8')
        for b in banderas:
            assert f"'{b}'" in texto, f'{script} ya no tiene la bandera {b}, que demo.py usa'


@prueba
def demo_ortesis_udp():
    """demo.py --ortesis-udp se comporta como el orquestador: la misma IP y puerto llegan a orquestador.py (y a
    repetir_sesion.py en el plan B), sin pyserial ni verificacion USB, excluyente con --ortesis-sim."""
    import demo
    import orquestador
    a = demo.argumentos(['lanzar', '--plan', 'gemelo', '--ortesis-udp', '127.0.0.1', '--udp-puerto', '9000'])
    cmd = demo.comandos('gemelo', a)
    assert cmd['orquestador'] == ('orquestador.py', ['real', '--ortesis-udp', '127.0.0.1', '--udp-puerto', '9000'])
    o = orquestador.argumentos(cmd['orquestador'][1])                       # el orquestador entiende lo que demo.py le manda
    assert o.ortesis_udp == '127.0.0.1' and o.udp_puerto == 9000 and not o.ortesis_sim
    # sin valor: la IP de fabrica de la ESP32 en su red
    a = demo.argumentos(['lanzar', '--plan', 'gemelo', '--ortesis-udp'])
    assert a.ortesis_udp == config.IP_ORTESIS_UDP and a.udp_puerto == config.PUERTO_ORTESIS_UDP
    assert orquestador.argumentos(demo.comandos('gemelo', a)['orquestador'][1]).ortesis_udp == config.IP_ORTESIS_UDP
    # lo de tras `--` sigue llegando al orquestador
    a = demo.argumentos(['lanzar', '--plan', 'gemelo', '--ortesis-udp', '--', '--sham'])
    assert demo.comandos('gemelo', a, a.extras)['orquestador'][1][-1] == '--sham'
    # excluyente con la ortesis simulada, y por defecto nada cambia
    for orden in ('lanzar', 'preflight'):
        try:
            demo.argumentos([orden, '--ortesis-sim', '--ortesis-udp'])
            raise AssertionError('debia fallar')
        except SystemExit:
            pass
    assert demo.comandos('gemelo', demo.argumentos(['lanzar', '--puerto', 'COM7']))['orquestador'][1] == ['real', '--puerto', 'COM7']
    # preflight: sin pyserial, sin verificacion USB, y el aviso dice a donde apunta
    vistos = []
    demo.revisar_dependencias('gemelo', False, True, vistos.append, ortesis_udp='192.168.4.1')
    assert 'serial' not in vistos
    vistos.clear()
    demo.revisar_dependencias('gemelo', False, True, vistos.append)
    assert 'serial' in vistos
    r = demo.revisar_puerto('COM4', False, ortesis_udp='127.0.0.1', udp_puerto=8888)[0]
    assert r['estado'] == 'AVISO' and 'ortesis_udp_sim.py' in r['que_hacer']
    r = demo.revisar_puerto('COM4', False, ortesis_udp='192.168.4.1')[0]
    assert r['estado'] == 'AVISO' and '192.168.4.1:8888' in r['texto'] and 'Adaptrode' in r['que_hacer']
    assert demo.revisar_puerto('COM4', True, ortesis_udp='192.168.4.1')[0]['estado'] == 'OK'          # la simulada manda
    r = demo.revisar_verificaciones('gemelo', False, ortesis_udp='192.168.4.1')[0]
    assert r['clave'] == 'verif_ortesis' and r['estado'] == 'AVISO' and 'Wi-Fi' in r['texto']
    # la bandera existe en los scripts a los que demo.py se la pasa
    for script in ('orquestador.py', 'repetir_sesion.py'):
        texto = (config.RAIZ / script).read_text(encoding='utf-8')
        assert "'--ortesis-udp'" in texto and "'--udp-puerto'" in texto, script
    # el plan B: la ortesis repite por Wi-Fi
    visto = {}
    a = demo.argumentos(['planb', '--ortesis-udp', '10.0.0.5'])
    assert a.ortesis_udp == '10.0.0.5'
    return 'las banderas y el preflight de la ortesis UDP coinciden con las del orquestador'


@prueba
def demo_firmware_simulado():
    """demo.py con el plan gemelo y --ortesis-udp en esta laptop lanza tambien ortesis_udp_sim.py (antes que el
    orquestador); con la ESP32 real, otro plan o la ortesis simulada, no."""
    import tempfile
    from pathlib import Path

    import demo
    cmd = demo.comandos('gemelo', demo.argumentos(['lanzar', '--plan', 'gemelo', '--ortesis-udp', '127.0.0.1', '--udp-puerto', '9000']))
    assert cmd['firmware'] == ('ortesis_udp_sim.py', ['--ip', '127.0.0.1', '--puerto', '9000'])
    assert demo.comandos('gemelo', demo.argumentos(['lanzar', '--plan', 'gemelo', '--ortesis-udp', 'localhost']))['firmware'][1][:2] == ['--ip', 'localhost']
    for argv, plan in ((['--plan', 'gemelo', '--ortesis-udp', '192.168.4.1'], 'gemelo'),     # la ESP32 real
                       (['--plan', 'casco', '--ortesis-udp', '127.0.0.1'], 'casco'),           # sin el gemelo no es una demo de mentira
                       (['--plan', 'gemelo', '--ortesis-sim'], 'gemelo'),
                       (['--plan', 'gemelo', '--puerto', 'COM7'], 'gemelo')):
        assert 'firmware' not in demo.comandos(plan, demo.argumentos(['lanzar', *argv])), argv
    # el script existe y acepta lo que demo.py le pasa
    texto = (config.RAIZ / 'ortesis_udp_sim.py').read_text(encoding='utf-8')
    assert "'--ip'" in texto and "'--puerto'" in texto
    # el preflight lo sabe: con el gemelo es OK; con otro plan, el aviso de siempre
    assert demo.revisar_puerto('COM4', False, ortesis_udp='127.0.0.1', plan='gemelo')[0]['estado'] == 'OK'
    assert demo.revisar_puerto('COM4', False, ortesis_udp='127.0.0.1', plan='casco')[0]['estado'] == 'AVISO'
    # lanzar: firmware, fuente y orquestador en ese orden; si el firmware se cierra al arrancar, error claro y nada mas
    eeg = lambda: [{'name': 'EEG', 'type': 'EEG', 'host': 'h'}]
    bien = [demo.rev('x', demo.OK, 'bien')]
    with tempfile.TemporaryDirectory() as d:
        res, lineas, foreground = Path(d), [], []

        def lanzar(procesos):
            a = demo.argumentos(['lanzar', '--plan', 'gemelo', '--ortesis-udp', '127.0.0.1', '--sin-tablero'])
            foreground.clear()
            codigo = demo.lanzar(a, [], salida=lineas.append, procesos=procesos, resolver=eeg, hacer_preflight=lambda a: bien,
                                 resultados=res, dormir=lambda s: None, correr_foreground=lambda argv: foreground.append(argv) or 0)
            return codigo
        p = _ProcesosFalsos(res / 'logs')
        assert lanzar(p) == 0 and [n for n, _, _ in p.pedidos] == ['firmware', 'fuente'] and p.cerrado
        assert foreground[0][2:5] == ['hijo', 'orquestador.py', 'real'] and '--ortesis-udp' in foreground[0] and '127.0.0.1' in foreground[0]
        p = _ProcesosFalsos(res / 'logs2', mueren=['firmware'])
        assert lanzar(p) == 3 and not foreground and p.cerrado and any('ortesis_udp_sim.py se cerro' in l and 'cola de firmware' in l for l in lineas)
        assert [n for n, _, _ in p.pedidos] == ['firmware']
    return 'el firmware simulado se lanza con el gemelo y 127.0.0.1, antes que la fuente; no con la ESP32 real'


@prueba
def demo_revisiones():
    """Cada comprobacion del preflight, con git, importaciones, red y archivos falsos."""
    import json
    import tempfile
    from pathlib import Path

    import demo
    from verificar_unicorn import CRITICAS

    def estados(revs):
        return {r['clave']: r['estado'] for r in revs}

    def git_falso(describe='v-demo', sucio='', fetch=0, nuevos='0'):
        def git(*args, **kw):
            return {'describe': (0, describe), 'status': (0, sucio), 'fetch': (fetch, ''), 'rev-list': (0, nuevos)}.get(args[0], (1, ''))
        return git
    assert estados(demo.revisar_codigo(git_falso())) == {'version': 'OK', 'cambios': 'OK', 'origin': 'OK'}
    assert estados(demo.revisar_codigo(git_falso(describe='v-demo-26-gabc')))['version'] == 'AVISO'
    assert estados(demo.revisar_codigo(git_falso(sucio=' M orquestador.py')))['cambios'] == 'AVISO'
    assert estados(demo.revisar_codigo(git_falso(fetch=1)))['origin'] == 'AVISO'
    nuevo = demo.revisar_codigo(git_falso(nuevos='3'))[2]
    assert nuevo['estado'] == 'AVISO' and '3 commit' in nuevo['texto'] and 'Luis' in nuevo['que_hacer']
    assert demo.revisar_codigo(lambda *a, **k: (1, ''))[0]['estado'] == 'AVISO'

    vistos = []

    def importar(nombre):
        vistos.append(nombre)
        if nombre == 'brainflow':
            raise OSError('no carga la DLL')
    r = demo.revisar_dependencias('gemelo', True, True, importar)[0]
    assert r['estado'] == 'OK' and not {'brainflow', 'serial', 'pyqtgraph'} & set(vistos)
    vistos.clear()
    r = demo.revisar_dependencias('casco', False, False, importar)[0]
    assert r['estado'] == 'FALLA' and 'brainflow (OSError)' in r['texto'] and {'serial', 'pyqtgraph'} <= set(vistos) and r['que_hacer']

    def flujo(nombre, tipo='EEG', host='laptop'):
        return {'name': nombre, 'type': tipo, 'host': host}
    assert estados(demo.revisar_flujos('gemelo', lambda: [flujo('Otro', 'X')])) == {'flujo_eeg': 'OK'}
    r = demo.revisar_flujos('gemelo', lambda: [flujo('EEG', host='compa')])[0]
    assert r['estado'] == 'FALLA' and 'compa' in r['texto']
    assert estados(demo.revisar_flujos('gemelo', lambda: [flujo('Paso', 'Paso')]))['flujos_orquestador'] == 'FALLA'
    assert estados(demo.revisar_flujos('unicornlsl', lambda: []))['app_unicornlsl'] == 'FALLA'
    assert estados(demo.revisar_flujos('unicornlsl', lambda: [flujo('UN-1', 'Data')]))['app_unicornlsl'] == 'OK'

    def sin_red():
        raise RuntimeError('sin liblsl')
    assert demo.revisar_flujos('casco', sin_red)[0]['estado'] == 'FALLA'

    assert demo.revisar_puerto('COM4', True)[0]['estado'] == 'OK'
    assert demo.revisar_puerto('com4', False, lambda: ['COM3', 'COM4'])[0]['estado'] == 'OK'
    r = demo.revisar_puerto('COM4', False, lambda: ['COM3'])[0]
    assert r['estado'] == 'FALLA' and 'COM3' in r['texto']
    assert demo.revisar_puerto('COM4', False, sin_red)[0]['estado'] == 'AVISO'

    with tempfile.TemporaryDirectory() as d:
        carpeta, ahora = Path(d), (lambda: 1_000_000.0)

        def guardar(nombre, datos):
            (carpeta / nombre).write_text(json.dumps(datos), encoding='utf-8')

        def unicorn(fuente, mala=None, hace_h=1.0):
            return {'t': ahora() - hace_h * 3600,
                    'por_fuente': {fuente: [{'clave': k, 'estado': 'FALLA' if k == mala else 'OK', 'texto': ''} for k in CRITICAS]}}
        assert estados(demo.revisar_verificaciones('casco', True, carpeta, ahora)) == {'verif_casco': 'AVISO'}          # no hay archivo
        guardar('verificacion_unicorn.json', unicorn('brainflow'))
        assert estados(demo.revisar_verificaciones('casco', True, carpeta, ahora)) == {'verif_casco': 'OK'}
        assert estados(demo.revisar_verificaciones('unicornlsl', True, carpeta, ahora)) == {'verif_casco': 'AVISO'}     # solo probo brainflow
        guardar('verificacion_unicorn.json', unicorn('brainflow', mala='contador'))
        r = demo.revisar_verificaciones('casco', True, carpeta, ahora)[0]
        assert r['estado'] == 'FALLA' and 'contador' in r['texto']
        guardar('verificacion_unicorn.json', unicorn('lsl', hace_h=5.0))
        assert 'hace 5.0 h' in demo.revisar_verificaciones('unicornlsl', True, carpeta, ahora)[0]['texto']
        assert demo.revisar_verificaciones('gemelo', True, carpeta, ahora) == []                                         # sin casco ni ortesis

        def ortesis(veredicto, simulada=False, hace_h=1.0):
            guardar('verificacion_ortesis.json', {'t': ahora() - hace_h * 3600, 'simulada': simulada, 'veredicto': veredicto})
            return demo.revisar_verificaciones('gemelo', False, carpeta, ahora)[0]
        assert ortesis('OK: latencia mecanica mediana 90 ms')['estado'] == 'OK'
        assert ortesis('AVISO: el inicio se vio en menos del 80 %')['estado'] == 'AVISO'
        assert ortesis('FALLA: no se vio el inicio del movimiento')['estado'] == 'FALLA'
        assert ortesis('OK: simulada', simulada=True)['estado'] == 'AVISO'
        assert ortesis('OK: vieja', hace_h=9.0)['estado'] == 'AVISO'

        assert demo.revisar_plan_b(carpeta / 'no_existe')[0]['estado'] == 'AVISO'
        assert demo.revisar_plan_b(carpeta)[0]['estado'] == 'AVISO'
        # una sesion detenida por un NO GO (solo checkpoint y detenida, ningun paso) no es plan B, aunque sea la unica o la mas nueva
        sufijo = config.SUFIJO_ESTADO
        solo_detenida = carpeta / 'solo_detenida'
        solo_detenida.mkdir()
        _grabar_estado_falso(solo_detenida / ('sesion_real_20261004_120000' + sufijo), ['checkpoint', 'detenida'], mtime=2000)
        r = demo.revisar_plan_b(solo_detenida)[0]
        assert r['estado'] == 'AVISO' and 'no hay ninguna sesion real grabada' in r['texto'] and 'no cuenta' in r['que_hacer'], r
        # la buena (con al menos un paso), mas vieja que la detenida: se elige la buena
        buena = _grabar_estado_falso(carpeta / ('sesion_real_20261004_100000' + sufijo), ['checkpoint', 'paso'], mtime=1000)
        r = demo.revisar_plan_b(carpeta)[0]
        assert r['estado'] == 'OK' and 'sesion_real_20261004_100000' in r['texto'] and '1 sesion(es)' in r['texto']
        _grabar_estado_falso(carpeta / ('sesion_real_20261004_120000' + sufijo), ['checkpoint', 'detenida'], mtime=2000)
        r = demo.revisar_plan_b(carpeta)[0]
        assert r['estado'] == 'OK' and '1 sesion(es)' in r['texto'] and buena.name in r['texto'] and '120000' not in r['texto'], r
        # el plan B de una sesion simulada no vale como respaldo de la demo (aunque tenga pasos)
        _grabar_estado_falso(carpeta / ('sesion_sim_20261004_110000' + sufijo), ['checkpoint', 'paso'], mtime=3000)
        r = demo.revisar_plan_b(carpeta)[0]
        assert '1 sesion(es)' in r['texto'] and buena.name in r['texto'] and 'sesion_sim' not in r['texto'], r

        assert demo.revisar_disco(carpeta, lambda ruta: type('U', (), {'free': 10 * 10 ** 9})())[0]['estado'] == 'OK'
        r = demo.revisar_disco(carpeta, lambda ruta: type('U', (), {'free': 100 * 10 ** 6})())[0]
        assert r['estado'] == 'FALLA' and r['que_hacer']

    # la llave de la API: sin red y sin llave el preflight no se cae
    assert demo.revisar_llave()[0]['clave'] == 'api'

    # el informe: solo muestra "que hacer" cuando algo no esta OK, y dice si se puede seguir
    lineas = []
    seguir = demo.imprimir_revisiones([demo.rev('a', demo.OK, 'bien', 'no se ve'), demo.rev('b', demo.AVISO, 'regular', 'se ve')], lineas.append)
    assert seguir and 'no se ve' not in '\n'.join(lineas) and 'se ve' in '\n'.join(lineas) and '1 OK, 1 aviso(s), 0 falla(s)' in lineas[-1]
    assert not demo.imprimir_revisiones([demo.rev('c', demo.FALLA, 'mal')], lambda s: None)


@prueba
def demo_limpiar_modelos():
    """--limpiar-modelos mueve (no borra), deja el decoder preentrenado y el aviso de modelos viejos desaparece."""
    import tempfile
    from pathlib import Path

    import demo
    with tempfile.TemporaryDirectory() as d:
        modelos, estado = Path(d) / 'modelos', Path(d) / 'estado_sesion.json'
        modelos.mkdir()
        for nombre in ('decoder_mi.pkl', 'detector_errp.pkl', 'detector_errp_datos.npz', config.DECODER_PREENTRENADO, 'LEEME.txt'):
            (modelos / nombre).write_text(nombre, encoding='utf-8')
        estado.write_text('{}', encoding='utf-8')
        r = demo.revisar_modelos(modelos, ahora=lambda: time.time() + 7200)[0]
        assert r['estado'] == 'AVISO' and '3 archivo' in r['texto'] and 'hace 2.0 h' in r['texto']   # el preentrenado no cuenta
        destino, movidos = demo.limpiar_modelos(modelos, estado, ahora=lambda: 1_000_000.0)
        assert sorted(movidos) == ['decoder_mi.pkl', 'detector_errp.pkl', 'detector_errp_datos.npz', 'estado_sesion.json']
        for nombre in movidos:
            assert (destino / nombre).exists(), f'{nombre} se perdio en lugar de moverse'
        assert destino.parent.name == '_anteriores' and not estado.exists()
        assert sorted(p.name for p in modelos.glob('*') if p.is_file()) == ['LEEME.txt', config.DECODER_PREENTRENADO]
        assert demo.revisar_modelos(modelos)[0]['estado'] == 'OK'
        assert demo.limpiar_modelos(modelos, estado) == (None, [])                                   # segunda vez: nada que mover
        assert demo.limpiar_modelos(Path(d) / 'no_existe', estado) == (None, [])
        # lo movido no cuenta como modelo viejo aunque este dentro de modelos/
        assert demo.revisar_modelos(modelos)[0]['estado'] == 'OK'


@prueba
def demo_esperar_flujo():
    """Esperar el flujo EEG: aparece, se agota el tiempo, o el proceso muere (y se ve su cola)."""
    import demo
    t = [0.0]

    def dormir(s):
        t[0] += s
    llamadas = []

    def aparece_a_la_tercera():
        llamadas.append(1)
        return [{'name': 'EEG', 'type': 'EEG', 'host': 'h'}] if len(llamadas) >= 3 else []
    procesos = _ProcesosFalsos('x')
    assert demo.esperar_flujo('EEG', aparece_a_la_tercera, procesos, 'fuente', 30.0, dormir, lambda: t[0]) is True
    assert len(llamadas) == 3 and t[0] == 2.0
    try:
        demo.esperar_flujo('EEG', lambda: [], procesos, 'fuente', 5.0, dormir, lambda: t[0])
        raise AssertionError('debia agotar el tiempo')
    except demo.ErrorDemo as e:
        assert 'no aparecio el flujo EEG en 5 s' in str(e) and 'cola de fuente' in str(e)
    try:
        demo.esperar_flujo('EEG', lambda: [], _ProcesosFalsos('x', mueren=['fuente']), 'fuente', 30.0, dormir, lambda: t[0])
        raise AssertionError('debia avisar que el proceso murio')
    except demo.ErrorDemo as e:
        assert 'se cerro antes de publicar' in str(e) and 'cola de fuente' in str(e)


@prueba
def demo_procesos():
    """Procesos de verdad: cierre limpio por PID (el script ve su KeyboardInterrupt), el terco se termina, el que ya salio se anota."""
    import tempfile
    from pathlib import Path

    import demo
    limpio = 'import time\nprint("INFO ruido", flush=True)\nprint("listo", flush=True)\ntry:\n    while True:\n        time.sleep(0.1)\nexcept KeyboardInterrupt:\n    print("cerrado limpio", flush=True)\n'
    terco = ('import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\n'
             'if hasattr(signal, "SIGBREAK"):\n    signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n'
             'print("listo", flush=True)\nwhile True:\n    time.sleep(0.1)\n')
    corto = 'print("listo", flush=True)\n'

    def esperar_log(procesos, nombre, texto, segundos=40):
        t_fin = time.time() + segundos
        while time.time() < t_fin:
            if texto in procesos.cola(nombre):
                return
            time.sleep(0.2)
        raise AssertionError(f'{nombre} no escribio {texto!r}: ' + procesos.cola(nombre))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for nombre, codigo in (('limpio', limpio), ('terco', terco), ('corto', corto)):
            (d / f'{nombre}.py').write_text(codigo, encoding='utf-8')
        a = demo.Procesos(d / 'logs_a')
        pl = a.lanzar('limpio', str(d / 'limpio.py'), [])
        pc = a.lanzar('corto', str(d / 'corto.py'), [])
        esperar_log(a, 'limpio', 'listo')
        esperar_log(a, 'corto', 'listo')
        assert 'ruido' not in a.cola('limpio') and a.vivo('limpio') and not a.vivo('nadie')
        pc.wait(timeout=30)
        cierre = a.cerrar_todos(espera_s=30.0)
        assert cierre == {'limpio': 'limpio', 'corto': 'ya habia salido (0)'}, cierre
        assert pl.poll() is not None and pc.poll() is not None and a.hijos == {}
        assert 'cerrado limpio' in (d / 'logs_a' / 'limpio.log').read_text(encoding='utf-8')
        b = demo.Procesos(d / 'logs_b')
        pt = b.lanzar('terco', str(d / 'terco.py'), [])
        esperar_log(b, 'terco', 'listo')
        assert b.cerrar_todos(espera_s=1.0) == {'terco': 'terminate'} and pt.poll() is not None


@prueba
def demo_lanzar_simulado():
    """lanzar y planb con procesos falsos: el orden, las paradas del preflight, los extras, el cierre y la bitacora."""
    import json
    import tempfile
    from pathlib import Path

    import demo
    bien, mal = [demo.rev('x', demo.OK, 'bien')], [demo.rev('y', demo.FALLA, 'mal', 'arreglalo')]
    eeg = lambda: [{'name': 'EEG', 'type': 'EEG', 'host': 'h'}]
    with tempfile.TemporaryDirectory() as d:
        res, lineas, foreground = Path(d), [], []

        def correr(codigo=0, error=None):
            def f(argv):
                foreground.append(argv)
                if error:
                    raise error
                return codigo
            return f

        def lanzar(argv, extras=None, preflight=bien, procesos=None, **kw):
            a = demo.argumentos(['lanzar', *argv])
            procesos = procesos or _ProcesosFalsos(res / 'logs')
            foreground.clear()
            lineas.clear()
            kw.setdefault('correr_foreground', correr())
            codigo = demo.lanzar(a, a.extras if extras is None else extras, salida=lineas.append, procesos=procesos, resolver=eeg,
                                 hacer_preflight=lambda a: preflight, resultados=res, dormir=lambda s: None, **kw)
            return codigo, procesos
        bitacoras = lambda: sorted(res.glob('demo_*.json'))
        # una FALLA detiene todo: nada se lanza y queda anotado
        codigo, p = lanzar(['--plan', 'gemelo', '--ortesis-sim'], preflight=mal)
        assert codigo == 2 and not p.pedidos and not foreground and not p.cerrado
        assert json.loads(bitacoras()[-1].read_text(encoding='utf-8'))['salida'] == 'detenido por el preflight'
        # con --ignorar-fallas sigue, y la bitacora lo dice
        codigo, p = lanzar(['--plan', 'gemelo', '--ortesis-sim', '--ignorar-fallas'], preflight=mal)
        assert codigo == 0 and len(foreground) == 1 and json.loads(bitacoras()[-1].read_text(encoding='utf-8'))['ignoro_fallas'] is True
        # el camino normal: fuente, luego narrador y tablero, luego el orquestador en primer plano
        codigo, p = lanzar(['--plan', 'gemelo', '--ortesis-sim', '--narrador'], extras=['--sham', '--preentrenado'])
        assert codigo == 0 and [n for n, _, _ in p.pedidos] == ['fuente', 'narrador', 'tablero'] and p.cerrado
        argv = foreground[0]
        assert argv[2:5] == ['hijo', 'orquestador.py', 'real'] and argv[-2:] == ['--sham', '--preentrenado']
        assert '--forzar' not in argv and '--saltar-calibracion' not in argv
        b = json.loads(bitacoras()[-1].read_text(encoding='utf-8'))
        assert b['plan'] == 'gemelo' and b['extras_orquestador'] == ['--sham', '--preentrenado'] and b['codigo_orquestador'] == 0
        assert b['cierre_de_procesos'] == {'fuente': 'limpio', 'narrador': 'limpio', 'tablero': 'limpio'} and b['comandos']['orquestador'][0] == 'orquestador.py'
        assert 'Bitacora' in lineas[-1] and any('gemelo digital, no una persona' in l for l in lineas)
        # el codigo de salida del orquestador es el de la demo (un CP en NO GO se ve, no se esconde)
        assert lanzar(['--plan', 'gemelo', '--ortesis-sim', '--sin-tablero'], correr_foreground=correr(codigo=1))[0] == 1
        # un NO GO detuvo la sesion (el orquestador sale con config.SALIDA_NO_GO): el tablero se queda abierto con el aviso
        # hasta que el operador pulse Enter; si no, la demo lo cerraria en el mismo instante y nadie lo veria
        esperas, falsos = [], _ProcesosFalsos(res / 'logs')
        codigo, p = lanzar(['--plan', 'gemelo', '--ortesis-sim'], correr_foreground=correr(codigo=config.SALIDA_NO_GO),
                           procesos=falsos, esperar_operador=lambda: esperas.append(falsos.cerrado))      # aun no se cerro nada
        assert codigo == config.SALIDA_NO_GO and esperas == [False] and p.cerrado and any('tablero sigue abierto' in l for l in lineas)
        b = json.loads(bitacoras()[-1].read_text(encoding='utf-8'))
        assert b['detenido_por_no_go'] is True and b['codigo_orquestador'] == config.SALIDA_NO_GO
        # sin tablero vivo no hay nada que mantener abierto; una sesion normal o interrumpida tampoco espera
        for kw in (dict(procesos=_ProcesosFalsos(res / 'logs', mueren=['tablero'])), dict(correr_foreground=correr(codigo=0))):
            kw.setdefault('correr_foreground', correr(codigo=config.SALIDA_NO_GO))
            codigo, p = lanzar(['--plan', 'gemelo', '--ortesis-sim'], esperar_operador=lambda: esperas.append('no debia esperar'), **kw)
            assert esperas == [False] and p.cerrado, esperas
        # Ctrl+C o sin teclado durante la espera: se cierra todo igual y el codigo sigue siendo el del NO GO
        for error in (KeyboardInterrupt(), EOFError()):
            def interrumpe(error=error):
                raise error
            codigo, p = lanzar(['--plan', 'gemelo', '--ortesis-sim'], correr_foreground=correr(codigo=config.SALIDA_NO_GO), esperar_operador=interrumpe)
            assert codigo == config.SALIDA_NO_GO and p.cerrado
        # sin `esperar_operador` decide la terminal: _hay_terminal mira sys.stdin y _esperar_enter es un input() (probados con falsos)
        import builtins

        class Entrada:
            def __init__(self, tty):
                self.tty = tty

            def isatty(self):
                return self.tty
        stdin, hay_terminal, input_real = sys.stdin, demo._hay_terminal, builtins.input
        llamadas = []

        def input_falso(*args):
            llamadas.append((args, falsos.cerrado))
            return ''
        try:
            sys.stdin = Entrada(True)
            assert demo._hay_terminal() is True
            sys.stdin = Entrada(False)
            assert demo._hay_terminal() is False
            sys.stdin = None                                       # pythonw o un proceso sin stdin
            assert demo._hay_terminal() is False
            falsos = _ProcesosFalsos(res / 'logs')
            builtins.input = input_falso
            demo._esperar_enter()
            assert llamadas == [((), False)], llamadas                # un solo input(), sin texto
            # sin terminal: lo dice, no espera, cierra todo y el codigo sigue siendo el del NO GO
            demo._hay_terminal = lambda: False
            llamadas.clear()
            falsos = _ProcesosFalsos(res / 'logs')
            codigo, p = lanzar(['--plan', 'gemelo', '--ortesis-sim'], correr_foreground=correr(codigo=config.SALIDA_NO_GO), procesos=falsos)
            assert codigo == config.SALIDA_NO_GO and not llamadas and p.cerrado, llamadas
            assert any('Sin terminal' in l for l in lineas) and not any('tablero sigue abierto' in l for l in lineas), lineas
            assert json.loads(bitacoras()[-1].read_text(encoding='utf-8'))['detenido_por_no_go'] is True
            # con terminal: espera UN Enter con el tablero todavia abierto y despues cierra todo
            demo._hay_terminal = lambda: True
            falsos = _ProcesosFalsos(res / 'logs')
            codigo, p = lanzar(['--plan', 'gemelo', '--ortesis-sim'], correr_foreground=correr(codigo=config.SALIDA_NO_GO), procesos=falsos)
            assert codigo == config.SALIDA_NO_GO and llamadas == [((), False)] and p.cerrado, llamadas
            assert any('tablero sigue abierto' in l for l in lineas) and not any('Sin terminal' in l for l in lineas), lineas
            # con terminal pero sin teclado a media espera (EOF o Ctrl+C): se cierra igual
            for error in (EOFError(), KeyboardInterrupt()):
                def input_roto(*args, error=error):
                    raise error
                builtins.input = input_roto
                falsos = _ProcesosFalsos(res / 'logs')
                codigo, p = lanzar(['--plan', 'gemelo', '--ortesis-sim'], correr_foreground=correr(codigo=config.SALIDA_NO_GO), procesos=falsos)
                assert codigo == config.SALIDA_NO_GO and p.cerrado
            # una sesion normal no espera a nadie, haya o no terminal
            builtins.input = input_falso
            llamadas.clear()
            falsos = _ProcesosFalsos(res / 'logs')
            codigo, p = lanzar(['--plan', 'gemelo', '--ortesis-sim'], correr_foreground=correr(codigo=0), procesos=falsos)
            assert codigo == 0 and not llamadas and p.cerrado
        finally:
            sys.stdin, demo._hay_terminal, builtins.input = stdin, hay_terminal, input_real
        assert demo._hay_terminal is hay_terminal and builtins.input is input_real
        # Ctrl+C: 130, y los procesos de fondo se cierran igual
        codigo, p = lanzar(['--plan', 'gemelo', '--ortesis-sim'], correr_foreground=correr(error=KeyboardInterrupt()))
        assert codigo == 130 and p.cerrado
        # la fuente muere antes de publicar EEG: error claro, el orquestador no arranca, se cierra todo
        codigo, p = lanzar(['--plan', 'gemelo', '--ortesis-sim'], procesos=_ProcesosFalsos(res / 'logs', mueren=['fuente']))
        assert codigo == 3 and not foreground and p.cerrado and any('cola de fuente' in l for l in lineas)
        # un tablero que se cierra al arrancar no detiene la sesion: avisa y sigue
        codigo, p = lanzar(['--plan', 'gemelo', '--ortesis-sim'], procesos=_ProcesosFalsos(res / 'logs', mueren=['tablero']))
        assert codigo == 0 and len(foreground) == 1 and any('AVISO: tablero se cerro' in l for l in lineas)
        # con la app UnicornLSL no hay fuente que lanzar ni flujo que esperar
        codigo, p = lanzar(['--plan', 'unicornlsl', '--ortesis-sim', '--sin-tablero'])
        assert codigo == 0 and p.pedidos == [] and foreground[0][-2:] == ['--fuente', 'unicornlsl']
        assert len(bitacoras()) >= 6                                                                       # ninguna se pisa

        # plan B: sin sesion grabada no abre nada; con una, tablero y repetir_sesion --ultima
        a = demo.argumentos(['planb', '--ortesis-sim'])
        vacia = res / 'vacia'
        vacia.mkdir()
        p = _ProcesosFalsos(res / 'logs')
        assert demo.planb(a, salida=lineas.append, resultados=vacia, procesos=p, dormir=lambda s: None, correr_foreground=correr()) == 1
        assert p.pedidos == [] and 'No hay plan B' in lineas[-1]
        # una sesion que un NO GO detuvo antes del lazo (solo checkpoint y detenida) no es plan B: planb no abre nada
        solo = res / 'solo_detenida'
        solo.mkdir()
        _grabar_estado_falso(solo / ('sesion_real_20261004_120000' + config.SUFIJO_ESTADO), ['checkpoint', 'detenida'], mtime=2000)
        foreground.clear()
        assert demo.planb(a, salida=lineas.append, resultados=solo, procesos=p, dormir=lambda s: None, correr_foreground=correr()) == 1
        assert p.pedidos == [] and not foreground and 'No hay plan B' in lineas[-1]
        # con una buena (tiene pasos), aunque la detenida sea mas nueva, si hay plan B
        _grabar_estado_falso(res / ('sesion_real_20261004_100000' + config.SUFIJO_ESTADO), ['checkpoint', 'paso'], mtime=1000)
        _grabar_estado_falso(res / ('sesion_real_20261004_120000' + config.SUFIJO_ESTADO), ['checkpoint', 'detenida'], mtime=2000)
        foreground.clear()
        assert demo.planb(a, salida=lineas.append, resultados=res, procesos=p, dormir=lambda s: None, correr_foreground=correr()) == 0
        assert p.pedidos == [('tablero', 'tablero.py', [])] and p.cerrado
        assert foreground[0][3:6] == ['repetir_sesion.py', '--ultima', '--velocidad'] and '--puerto' not in foreground[0]
        foreground.clear()
        demo.planb(demo.argumentos(['planb', '--puerto', 'COM9']), salida=lineas.append, resultados=res, procesos=_ProcesosFalsos(res / 'l2'),
                   dormir=lambda s: None, correr_foreground=correr())
        assert foreground[0][-2:] == ['--puerto', 'COM9']


@prueba
def demo_gemelo_en_vivo():
    """Con LSL de verdad: demo.Procesos lanza el gemelo, espera su flujo EEG, el preflight lo ve y se cierra por PID."""
    import tempfile
    from pathlib import Path

    import demo
    antes = [f for f in demo._resolver_lsl(1.0) if f['name'] == 'EEG']
    assert not antes, 'hay un flujo EEG ajeno en la red; cierralo antes de probar'
    with tempfile.TemporaryDirectory() as d:
        procesos = demo.Procesos(Path(d))
        try:
            p = procesos.lanzar('fuente', 'cerebro_sintetico.py', [])
            demo.esperar_flujo('EEG', demo._resolver_lsl, procesos, 'fuente', 60.0)
            visto = demo.revisar_flujos('gemelo')[0]
            assert visto['clave'] == 'flujo_eeg' and visto['estado'] == 'FALLA', visto        # un segundo EEG se rechazaria
        finally:
            cierre = procesos.cerrar_todos(espera_s=15.0)
        assert cierre['fuente'] == 'limpio' and p.poll() is not None, cierre
    despues = demo.revisar_flujos('gemelo')[0]
    assert despues['estado'] == 'OK', despues
    return 'gemelo lanzado, visto y cerrado por PID'


@prueba
def toques_electrodos():
    """La prueba de toques de verificar_unicorn.py: el pico debe salir en el canal tocado. Datos
    SINTETICOS (ruido, 60 Hz, offset de continua y golpecitos de 150 uV en el canal tocado, 40 uV en sus
    vecinos): valida la logica, no cuanto responde un electrodo real. Cubre el caso que importa: C3 y C4
    intercambiados."""
    import verificar_unicorn as vu
    fs = 250.0
    nombres = [n for n, _, _ in vu.fases(solo_toques=True)]
    segs = [s for _, s, _ in vu.fases(solo_toques=True)]
    assert nombres[0] == 'reposo' and nombres[1:5] == ['toque_Fz', 'suelta_Fz', 'toque_C3', 'suelta_C3'] and len(nombres) == 17
    assert [n for n, _, _ in vu.fases(toques=True)][:4] == ['reposo', 'parpadeo', 'cerrados', 'cabeza'] and 'toque_PO8' in [n for n, _, _ in vu.fases(toques=True)]
    assert 'toque_Fz' not in [n for n, _, _ in vu.fases()]
    assert all('IZQUIERDA' in t for n, _, t in vu.fases(solo_toques=True) if n == 'toque_C3') and any('DERECHA' in t for n, _, t in vu.fases(solo_toques=True))
    fase = np.concatenate([np.full(int(s * fs), k) for k, s in enumerate(segs)])

    def senal(semilla=0, intercambio=(), sin_toque=(), debil=()):
        rng = np.random.default_rng(semilla)
        n = len(fase)
        t = np.arange(n) / fs
        x = rng.normal(0, 8, (8, n)) + 210_000.0 + 20 * np.sin(2 * np.pi * 60 * t)
        for k, c in enumerate(config.CANALES_EEG):
            if c in sin_toque:
                continue
            ini = np.where(fase == nombres.index(f'toque_{c}'))[0]
            for t0 in ini[int(0.6 * fs)::int(0.33 * fs)]:                  # golpecitos desde 0.6 s, tres por segundo
                m = min(int(0.12 * fs), n - t0)
                golpe = np.exp(-np.arange(m) / (0.03 * fs)) * np.sin(2 * np.pi * 9 * np.arange(m) / fs)
                x[k, t0:t0 + m] += (60 if c in debil else 150) * golpe
                for v in (k - 1, k + 1):
                    if 0 <= v < 8:
                        x[v, t0:t0 + m] += (35 if c in debil else 40) * golpe
        for a, b in intercambio:                                            # como si las filas del casco vinieran cruzadas
            i, j = config.CANALES_EEG.index(a), config.CANALES_EEG.index(b)
            x[[i, j]] = x[[j, i]]
        return x
    est = lambda res: {r['clave']: r['estado'] for r in res}
    # bien puesto: los 8 en OK y el resumen en OK, con su matriz
    res = vu.evaluar_toques(senal(), fase, nombres, fs)
    assert est(res)['toques'] == 'OK' and all(est(res)[f'toque_{c}'] == 'OK' for c in config.CANALES_EEG), res
    mat = np.array(res[-1]['matriz'])
    assert mat.shape == (8, 8) and all(np.argmax(mat[k]) == k for k in range(8))
    # C3 y C4 intercambiados: lo dice con sus nombres, en FALLA, y el veredicto no deja usar la fuente
    res = vu.evaluar_toques(senal(intercambio=[('C3', 'C4')]), fase, nombres, fs)
    e = est(res)
    assert e['toques'] == 'FALLA' and e['toque_C3'] == 'FALLA' and e['toque_C4'] == 'FALLA' and e['toque_Cz'] == 'OK', res
    assert 'C3 y C4' in res[-1]['texto'] and 'intercambiados' in res[-1]['texto'], res[-1]['texto']
    assert 'tocaste C3 y respondio mas C4' in [r for r in res if r['clave'] == 'toque_C3'][0]['texto']
    crit = [{'clave': k, 'estado': 'OK', 'texto': ''} for k in vu.CRITICAS]
    assert 'usa BrainFlow' in vu.veredicto({'brainflow': crit + [dict(r, estado='OK') for r in res[-1:]]}, None)
    v = vu.veredicto({'brainflow': crit + res}, None)
    assert 'NO uses' in v and 'toques' in v and 'C3 y C4' in v, v
    # un electrodo que no se toco (o no se vio) no se da por bueno ni por malo: AVISO
    res = vu.evaluar_toques(senal(sin_toque=['Pz']), fase, nombres, fs)
    e = est(res)
    assert e['toque_Pz'] == 'AVISO' and e['toques'] == 'AVISO' and e['toque_Cz'] == 'OK', res
    # toque flojo, con los vecinos casi igual de fuertes: AVISO (margen), nunca FALLA
    res = vu.evaluar_toques(senal(debil=['Oz']), fase, nombres, fs)
    assert est(res)['toque_Oz'] in ('AVISO', 'OK') and est(res)['toques'] != 'FALLA'
    # sin reposo ni pausas con que comparar: no inventa nada
    assert vu.evaluar_toques(senal(), fase, ['toque_Fz'] * 1, fs)[0]['estado'] == 'AVISO'
    # la fuente completa pasa por evaluar() (17 canales como la app de g.tec) y los cruzados la tumban
    n = len(fase)

    def datos17(x):
        z = np.zeros((17, n))
        z[:8] = x
        z[8:11] = np.array([[0.0], [0.0], [1.0]])                           # ~1 g en reposo
        z[11:14] = 1.0                                                      # giroscopio quieto
        z[14], z[15], z[16] = 90.0, np.arange(n), 1.0
        return {'x': z, 'fs': fs, 'fase': fase, 'fases': nombres, 'llegada': np.arange(n) / fs, 'modulo': None,
                'mapa': {k: config.FUENTES_EEG['unicornlsl'][k] for k in ('eeg', 'imu', 'bateria', 'contador', 'validez')}}
    ok = {r['clave']: r for r in vu.evaluar(datos17(senal()))}
    assert ok['toques']['estado'] == 'OK' and ok['eeg_unidades']['estado'] == 'OK' and ok['contador']['estado'] == 'OK', ok
    cruz = {r['clave']: r for r in vu.evaluar(datos17(senal(intercambio=[('C3', 'C4')])))}
    assert cruz['toques']['estado'] == 'FALLA' and 'NO uses' in vu.veredicto({'lsl': list(cruz.values())}, 'X')
    # el preflight de demo.py tambien lo ve
    import demo
    import json
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / 'verificacion_unicorn.json').write_text(json.dumps(
            {'t': 1_000_000.0, 'por_fuente': {'brainflow': [{'clave': k, 'estado': 'OK', 'texto': ''} for k in vu.CRITICAS]
                                              + [{'clave': 'toques', 'estado': 'FALLA', 'texto': 'x'}]}}), encoding='utf-8')
        r = demo.revisar_verificaciones('casco', True, Path(d), lambda: 1_000_100.0)[0]
        assert r['estado'] == 'FALLA' and 'toques' in r['texto'], r
    return 'C3 y C4 cruzados: FALLA con sus nombres; bien puesto 8/8; sin toque o flojo: AVISO (datos sinteticos)'


@prueba
def senal_neutra():
    """CERRAR y RELAJA se ven (y suenan) igual salvo la palabra (o el orden de los dos tonos). Sin Qt: el
    estilo del tablero sale de cue.estilo(), que no recibe la meta, y aqui se lee el codigo de Tablero._cue."""
    import ast
    import inspect
    from pathlib import Path
    import cue
    import orquestador
    # lo que se ve: misma hoja de estilo para las dos metas (la funcion ni siquiera recibe la meta)
    assert not inspect.signature(cue.estilo).parameters
    est = cue.estilo()
    assert f"font-size:{config.CUE_VISUAL['px']}px" in est and config.CUE_VISUAL['color'] in est and 'monospace' in est
    assert '#d62728' not in est and '#1f77b4' not in est                            # ni el rojo ni el azul de antes
    t1, t2 = cue.texto(1), cue.texto(-1)
    assert (t1, t2) == ('CERRAR', 'RELAJA') and len(t1) == len(t2)                    # misma cantidad de letras
    assert cue.texto(1, visual=False) == cue.texto(-1, visual=False) == config.CUE_VISUAL['neutro']
    assert orquestador.linea_cue(1) == '    >>> CERRAR' and orquestador.linea_cue(-1) == '    >>> RELAJA'
    # el tablero usa esas dos funciones y no lleva un color propio segun la meta
    fuente = open(Path(__file__).with_name('tablero.py'), encoding='utf-8').read()
    arbol = ast.parse(fuente)
    cuerpo = [n for c in arbol.body if isinstance(c, ast.ClassDef) and c.name == 'Tablero'
              for n in c.body if isinstance(n, ast.FunctionDef) and n.name == '_cue'][0]
    codigo = ast.get_source_segment(fuente, cuerpo)
    assert 'cue.estilo()' in codigo and 'cue.texto(' in codigo and '#' not in codigo.split('"""')[2], codigo
    # lo que se oye: los mismos dos tonos en orden contrario, misma duracion y energia, sin clic
    a, b = cue.tonos(1), cue.tonos(-1)
    assert a == b[::-1] and a[0][0] < a[1][0] and sum(ms for _, ms in a) == sum(ms for _, ms in b)
    wa, wb = np.array(cue.onda(1), dtype=float), np.array(cue.onda(-1), dtype=float)
    assert len(wa) == len(wb) and abs((wa ** 2).sum() / (wb ** 2).sum() - 1) < 0.01
    assert abs(wa[0]) < 5 and abs(wa[-1]) < 5 and np.abs(wa).max() <= 0.4 * 32767 + 1
    # --cue-audio: apagado por defecto y solo con el casco; suena en un hilo; un fallo no detiene nada
    assert not orquestador.audio_activo(orquestador.argumentos(['real']))
    assert orquestador.audio_activo(orquestador.argumentos(['real', '--cue-audio']))
    assert not orquestador.audio_activo(orquestador.argumentos(['sim', '--cue-audio']))
    sonado, avisos = [], []
    mudo = cue.Audio(False, reproducir=lambda m: sonado.append(m) or True)
    mudo.sonar(1)
    assert sonado == [] and mudo.sonadas == []
    activo = cue.Audio(True, reproducir=lambda m: sonado.append(m) or True, avisar=avisos.append)
    activo.sonar(1)
    activo.esperar()
    activo.sonar(-1)
    activo.esperar()
    assert sonado == [1, -1] and avisos == [] and activo.fallo is None
    roto = cue.Audio(True, reproducir=lambda m: 1 / 0, avisar=avisos.append)
    roto.sonar(1)
    roto.esperar()
    roto.sonar(-1)
    roto.esperar()
    assert len(avisos) == 1 and 'ZeroDivisionError' in avisos[0] and 'visual sigue' in avisos[0], avisos     # una vez
    sin_audio = cue.Audio(True, reproducir=lambda m: False, avisar=avisos.append)
    sin_audio.sonar(1)
    sin_audio.esperar()
    assert len(avisos) == 2 and sin_audio.fallo
    # el evento de Estado: --cue-sin-visual lo dice y el tablero muestra solo el '+'
    assert orquestador.evento_cue(orquestador.argumentos(['real']), 1) == {'tipo': 'cue', 'meta': 1}
    ev = orquestador.evento_cue(orquestador.argumentos(['real', '--cue-sin-visual']), -1)
    assert ev == {'tipo': 'cue', 'meta': -1, 'visual': False}
    assert orquestador.evento_cue(type('A', (), {})(), 1) == {'tipo': 'cue', 'meta': 1}   # Namespace de prueba sin la bandera
    return 'CERRAR/RELAJA: mismo estilo, misma forma de linea, tonos espejo con la misma energia; audio opcional que no detiene la sesion'


@prueba
def decoder_canales_mi():
    """El decoder de MI usa solo C3, Cz y C4 por defecto (--decoder-canales mi). En el gemelo no cuesta CP2;
    que quite una pista visual de los canales posteriores NO se puede ver aqui (el gemelo no la tiene): sale de
    los datos del casco real (un participante, exploratorio)."""
    import cerebro_sintetico as cs
    import hardware as hw
    import orquestador
    assert orquestador.argumentos(['real']).decoder_canales == config.DECODER_CANALES_DEFECTO == 'mi'
    assert orquestador.argumentos(['real', '--decoder-canales', 'auto']).decoder_canales == 'auto'
    solo = config.candidatos('decoder', 'mi')
    assert list(solo) == ['C3/Cz/C4'] and solo['C3/Cz/C4'] == config.indices('mi') == [1, 2, 3]
    assert list(config.candidatos('decoder', 'todos')) == ['8 canales']
    assert list(config.candidatos('decoder')) == list(config.candidatos('decoder', 'auto')) == ['C3/Cz/C4', '8 canales']   # estudios y banco: como siempre
    try:
        config.candidatos('decoder', 'pz')
        raise AssertionError('debia rechazar un valor desconocido')
    except ValueError:
        pass
    # el decoder queda con C3/Cz/C4, ignora cualquier otro canal (aunque sea un desastre) y pasa CP2 en el gemelo
    bas = []
    for semilla in range(4):
        X, y = cs.sesion_mi(40, semilla=semilla)
        d = hw.DecoderIM().ajustar(X, y, config.candidatos('decoder', 'mi'))
        assert d.canales == [1, 2, 3] and d.eleccion == 'C3/Cz/C4' and list(d.puntajes) == ['C3/Cz/C4']
        roto = X.copy()
        roto[:, [0, 4, 5, 6, 7]] = 1e6 * np.random.default_rng(semilla).normal(size=roto[:, [0, 4, 5, 6, 7]].shape)
        assert np.allclose(d.phi(roto[0], actualizar_centro=False), d.phi(X[0], actualizar_centro=False)), 'los canales fuera de C3/Cz/C4 no deben entrar'
        bas.append(d.ba)
    assert np.mean(bas) >= config.MI_EXACTITUD_MIN, bas
    return f'C3/Cz/C4: BA {np.mean(bas):.2f} en el gemelo (4 sujetos, 40 ensayos); los demas canales no entran'


@prueba
def mano_virtual_logica():
    """La mano virtual sin pantalla: geometria, animacion, eventos de Estado, dibujo contra un pintor de registro,
    el espejo de la ortesis y que la ventana Qt hable el mismo idioma que el pintor. La parte Qt no corre aqui."""
    import ast
    import types
    from pathlib import Path
    import cue
    import demo
    import hardware as hw
    import mano_virtual as mv
    import orquestador
    # geometria: la punta de cada dedo baja y se acerca a la palma al cerrar, sin saltos; cabe en la caja
    prev = None
    for c in np.linspace(0, 1, 11):
        pts = mv.punta_de_dedos(c)
        if prev is not None:
            if c <= 0.81:                                    # al final la yema se mete hacia la palma y sube un poco: es el puno
                assert all(p[1] <= q[1] + 1e-9 for p, q in zip(pts[:4], prev[:4])), f'las puntas de los dedos suben al cerrar ({c:.1f})'
            assert max(np.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(pts, prev)) < 0.45, 'salto de una punta entre dos cierres'
        prev = pts
    abierta, cerrada = mv.punta_de_dedos(0.0), mv.punta_de_dedos(1.0)
    assert all(p[1] > mv.PALMA[-1][1] + 0.5 for p in abierta[:4]), 'abierta, los dedos salen de la palma'
    assert all(p[1] < mv.PALMA[-1][1] - 0.1 for p in cerrada[:4]), 'cerrada, las puntas quedan por debajo de los nudillos'
    xmin, ymin, xmax, ymax = mv.CAJA
    for c in (0.0, 0.5, 1.0):
        for f in mv.primitivas(c):
            for x, y in f[1]:
                assert xmin <= x <= xmax and ymin <= y <= ymax, (c, f[0], x, y)
    # animacion: arranca de golpe (en 25 ms ya recorrio mas del 15 % de una orden de 250 ms), llega y no retrocede en el tiempo
    an = mv.Animacion(0.0)
    an.ir_a(1.0, 0.25, 10.0)
    assert an.valor(9.9) == 0.0 and an.valor(10.0) == 0.0 and an.valor(10.025) > 0.15 and an.valor(10.125) > 0.7
    assert abs(an.valor(10.25) - 1.0) < 1e-12 and an.valor(99.0) == 1.0
    an.ir_a(0.0, 0.25, 10.125)                              # una orden nueva con la anterior a medias: sale de donde esta
    assert abs(an.valor(10.125) - 0.75) < 1e-9 and an.valor(10.4) == 0.0
    an.ir_a(5.0, 0.2, 11.0)
    assert an.destino() == 1.0, 'el destino se recorta a 0..1'
    # estado: el evento 'mano' manda; 'paso' y 'ajeno' solo valen mientras no haya llegado ninguno
    st = mv.EstadoMano()
    st.procesar({'tipo': 'paso', 'angulo': 0.8}, 100.0)
    assert st.cierre(100.3) == 0.8 and not st.exacto
    st.procesar({'tipo': 'mano', 'angulo': 0.2, 'ms': 250, 'inicio': 100.4}, 100.35)
    assert st.exacto and st.cierre(100.39) == 0.8 and abs(st.cierre(100.66) - 0.2) < 1e-9, 'arranca en el inicio que dice el orquestador'
    st.procesar({'tipo': 'paso', 'angulo': 1.0}, 101.0)
    assert abs(st.cierre(101.5) - 0.2) < 1e-9, "con el evento 'mano', el 'paso' ya no mueve la mano"
    st.procesar({'tipo': 'mano', 'angulo': 1.0, 'ms': 250, 'inicio': 7.0}, 102.0)          # de otro reloj: se ignora, arranca ya
    assert st.cierre(102.3) == 1.0 and st.cierre(102.01) > 0.2
    st.procesar({'tipo': 'mano', 'angulo': 0.0, 'ms': 5}, 103.0)                              # 5 ms se alarga a lo minimo visible
    assert 0.2 < st.cierre(103.06) and st.cierre(103.2) == 0.0
    for raro in ({}, {'tipo': 'salud'}, {'tipo': 'paso'}, {'tipo': 'checkpoint', 'ok': False}):
        st.procesar(raro, 104.0)                                                              # ni se cae ni mueve la mano
    assert st.cierre(105.0) == 0.0
    # la senal de arriba: la misma para las dos metas, salvo la palabra; AUTOMATICO en un movimiento ajeno
    st = mv.EstadoMano()
    assert st.rotulo()[0] == ''
    st.procesar({'tipo': 'cue', 'meta': 1}, 1.0)
    r1 = st.rotulo()
    st.procesar({'tipo': 'cue', 'meta': -1}, 2.0)
    r2 = st.rotulo()
    assert (r1[0], r2[0]) == ('CERRAR', 'RELAJA') and r1[1] == r2[1], 'mismo color para las dos metas'
    st.procesar({'tipo': 'cue', 'meta': 1, 'visual': False}, 3.0)
    assert st.rotulo()[0] == cue.texto(1, visual=False) == '+'
    st.procesar({'tipo': 'aviso_ajeno'}, 4.0)
    assert st.rotulo()[0] == 'AUTOMATICO'
    st.procesar({'tipo': 'ajeno', 'angulo': 0.5}, 5.0)
    assert st.rotulo()[0] == '+'
    # dibujo: todo cabe en el cuadro (con su grosor) en pantallas anchas, altas y chicas, y la barra sigue al cierre
    for w, h in ((1920, 1080), (1080, 1920), (800, 600)):
        for c in (0.0, 0.5, 1.0):
            e = mv.EstadoMano()
            e.animacion = mv.Animacion(c)
            e.procesar({'tipo': 'cue', 'meta': 1 if c >= 0.5 else -1}, 0.0)
            pintor = mv.PintorRegistro()
            mv.dibujar(pintor, w, h, e, 1.0)
            assert pintor.ordenes[0] == ('rellenar', mv.COLOR_FONDO)
            for o in pintor.ordenes:
                if o[0] == 'poligono':
                    assert all(-1 <= x <= w + 1 and -1 <= y <= h + 1 for x, y in o[1]), (w, h, c)
                elif o[0] == 'cadena':
                    g = o[2]
                    assert all(g / 2 <= x <= w - g / 2 and g / 2 <= y <= h - g / 2 for x, y in o[1]), (w, h, c, 'la mano se sale del cuadro')
            textos = [o[1] for o in pintor.ordenes if o[0] == 'texto']
            assert ('CERRAR' if c >= 0.5 else 'RELAJA') in textos and f'{c * 100:.0f} % cerrada' in textos
            barra = [o for o in pintor.ordenes if o[0] == 'poligono' and o[2] == mv.COLOR_BANDA and o[1][0][1] == o[1][1][1] and len(o[1]) == 4][-1]
            assert abs((barra[1][1][0] - barra[1][0][0]) - c * 0.5 * w) < 1e-6
    e = mv.EstadoMano()
    pintor = mv.PintorRegistro()
    mv.dibujar(pintor, 800, 600, e, 0.0, conectado=False)
    assert 'Esperando al orquestador...' in [o[1] for o in pintor.ordenes if o[0] == 'texto']
    # la imagen de verificacion no necesita Qt
    import tempfile
    ruta = Path(tempfile.mkdtemp()) / 'mano.png'
    e.animacion = mv.Animacion(1.0)
    mv.imagen(e, ruta, 300, 350)
    assert ruta.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n' and ruta.stat().st_size > 3000
    # espejo de la ortesis: publica cada orden con su inicio y deja todo lo demas de la ortesis como estaba
    pub = []
    sim = hw.OrtesisSimulada(latencia_ms=0.1, jitter_ms=0.0, semilla=3)
    espejo = mv.espejar(sim, lambda **d: pub.append(d), lambda seq: hw.latencia_mecanica_simulada(seq, sim.semilla))
    assert espejo is sim and isinstance(sim, hw.OrtesisSimulada)
    seq, t_ack, _ = sim.mover(0.7)
    seq2, t_ack2, _ = sim.mover(0.5, config.CENTRADO_DURACION_MS)
    assert [p['tipo'] for p in pub] == ['mano', 'mano'] and pub[0]['angulo'] == 0.7 and pub[0]['seq'] == seq and pub[0]['ack']
    assert pub[0]['ms'] == config.DURACION_PASO_MS and pub[1]['ms'] == config.CENTRADO_DURACION_MS
    lo, hi = config.LATENCIA_MECANICA_SIM_MS
    assert lo / 1000 <= pub[0]['inicio'] - t_ack <= hi / 1000 + 1e-9 and sim.angulo == 0.5
    sim.seq = 40                                             # la reanudacion fija el seq en la ortesis, no en el espejo
    assert sim.mover(0.4)[0] == 41 and pub[-1]['seq'] == 41
    perdida = types.SimpleNamespace(mover=lambda f, d=250: (9, None, None))             # un ACK perdido tambien se publica
    pub2 = []
    mv.espejar(perdida, lambda **d: pub2.append(d))
    assert perdida.mover(0.3)[0] == 9 and pub2[0]['inicio'] is None and pub2[0]['ack'] is False
    roto = types.SimpleNamespace(mover=lambda f, d=250: (1, 5.0, None))
    mv.espejar(roto, lambda **d: 1 / 0)
    assert roto.mover(0.1) == (1, 5.0, None), 'un fallo al publicar no detiene la sesion'
    # el orquestador lo cablea con --mano-virtual (apagado por defecto) y el simulador avisa que no tiene ortesis
    assert not orquestador.argumentos(['real']).mano_virtual and orquestador.argumentos(['real', '--mano-virtual']).mano_virtual
    avisos, eventos = [], []
    backend = types.SimpleNamespace(hw=hw, ortesis=hw.OrtesisSimulada(latencia_ms=0.1, jitter_ms=0.0))
    assert orquestador.espejar_ortesis(backend, types.SimpleNamespace(estado=lambda **d: eventos.append(d)), avisar=avisos.append) is True
    backend.ortesis.mover(0.6)
    assert eventos and eventos[0]['tipo'] == 'mano' and 'mano_virtual.py' in avisos[0]
    assert orquestador.espejar_ortesis(types.SimpleNamespace(hw=hw), None, avisar=avisos.append) is False and 'simulador' in avisos[-1]
    # demo.py: la abre con --pantalla y lanza el orquestador con --mano-virtual; Qt se exige tambien con --sin-tablero
    a = demo.argumentos(['lanzar', '--plan', 'gemelo', '--ortesis-sim', '--mano-virtual', '--pantalla', '1'])
    cmd = demo.comandos('gemelo', a)
    assert cmd['mano'] == ('mano_virtual.py', ['--pantalla', '1']) and cmd['orquestador'][1] == ['real', '--ortesis-sim', '--mano-virtual']
    assert 'mano' not in demo.comandos('gemelo', demo.argumentos(['lanzar', '--plan', 'gemelo', '--ortesis-sim']))
    vistos = []
    demo.revisar_dependencias('gemelo', True, True, importar=vistos.append, mano_virtual=True)
    assert 'pyqtgraph' in vistos
    # la ventana Qt traduce el mismo pintor que se probo aqui: mismos metodos, y el codigo al menos se puede leer
    fuente = Path(mv.__file__).read_text(encoding='utf-8')
    arbol = ast.parse(fuente)
    clases = {n.name: n for f in ast.walk(arbol) if isinstance(f, ast.FunctionDef) and f.name == '_crear_ventana'
              for n in ast.walk(f) if isinstance(n, ast.ClassDef)}
    metodos = lambda c: {m.name for m in c.body if isinstance(m, ast.FunctionDef) and not m.name.startswith('_')}
    assert metodos(clases['PintorQt']) == metodos(ast.parse(fuente).body[[getattr(n, 'name', '') for n in arbol.body].index('PintorRegistro')]) - {'ordenes'}
    import importlib
    spec = importlib.util.find_spec('pyqtgraph')
    return 'cierre 0-1 sin saltos, cabe en 3 formatos de pantalla, espejo con inicio = ACK + latencia mecanica' + ('' if spec else ' (la ventana Qt no se pudo correr aqui)')


@prueba
def mano_virtual_ventana_falsa():
    """Corre el codigo de la ventana Qt contra un Qt de mentira que solo anota las llamadas: atrapa nombres mal escritos,
    argumentos que no cuadran y errores de logica de la ventana. NO prueba que Qt de verdad dibuje eso (hay que abrirla)."""
    import sys
    import types
    import mano_virtual as mv

    llamadas = []

    class Anota:
        """Cualquier atributo es una funcion que anota su llamada y devuelve otro Anota."""
        def __init__(self, nombre='Qt'):
            self._n = nombre

        def __getattr__(self, k):
            if k.startswith('__'):
                raise AttributeError(k)
            return Anota(f'{self._n}.{k}')

        def __call__(self, *a, **kw):
            llamadas.append((self._n, a, kw))
            return Anota(self._n + '()')

    class Widget:
        def __init__(self):
            self.cerrado, self.pantalla_completa, self._w, self._h = False, False, 900, 600

        def width(self):
            return self._w

        def height(self):
            return self._h

        def __getattr__(self, k):
            if k.startswith('__'):
                raise AttributeError(k)
            return lambda *a, **kw: llamadas.append((k, a, kw))

        def close(self):
            self.cerrado = True

        def isFullScreen(self):
            return self.pantalla_completa

        def showFullScreen(self):
            self.pantalla_completa = True

        def showNormal(self):
            self.pantalla_completa = False

    class Timer:
        def __init__(self):
            self.timeout = Anota('timeout')
            self.periodo = None

        def start(self, ms):
            self.periodo = ms

    class Pintor(Anota):
        def viewport(self):
            return 'viewport'
    Qt = types.SimpleNamespace(SolidLine=1, RoundCap=2, RoundJoin=3, NoBrush=4, AlignCenter=5, Key_Escape=27, Key_F=70)
    QtCore = types.SimpleNamespace(Qt=Qt, QTimer=Timer, QPointF=lambda x, y: (x, y), QRectF=lambda *a: a)
    QtGui = types.SimpleNamespace(QColor=lambda c: c, QPen=lambda *a: ('pluma',) + a, QBrush=lambda c: ('brocha', c),
                                  QPolygonF=lambda p: list(p), QPainterPath=lambda: Anota('ruta'),
                                  QFont=lambda *a: Anota('fuente'), QPainter=type('QPainter', (Pintor,), {'Antialiasing': 1}))
    pantalla = types.SimpleNamespace(geometry=lambda: types.SimpleNamespace(x=lambda: 1920, y=lambda: 0))
    QtWidgets = types.SimpleNamespace(QWidget=Widget, QApplication=types.SimpleNamespace(screens=lambda: [pantalla, pantalla]))
    qt = types.ModuleType('pyqtgraph.Qt')
    qt.QtCore, qt.QtGui, qt.QtWidgets = QtCore, QtGui, QtWidgets
    pg = types.ModuleType('pyqtgraph')
    pg.Qt = qt
    previos = {k: sys.modules.get(k) for k in ('pyqtgraph', 'pyqtgraph.Qt')}
    sys.modules['pyqtgraph'], sys.modules['pyqtgraph.Qt'] = pg, qt
    try:
        st = mv.EstadoMano()
        v = mv._crear_ventana(st, demo=True, ventana=False, pantalla=1)
        assert v.pantalla_completa and ('move', (1920, 0), {}) in llamadas and v.timer.periodo == 8
        v._cuadro()                                          # el guion de --demo da la primera orden
        assert st.ordenes == 1 and st.rotulo()[0] == 'CERRAR'
        for _ in range(3):
            llamadas.clear()
            v.paintEvent(None)
            nombres = [n for n, _, _ in llamadas]
            assert 'viewport' not in nombres and any(n.endswith('drawPolygon') for n in nombres) and any(n.endswith('drawPath') for n in nombres)
            assert any(n.endswith('drawText') for n in nombres) and any(n.endswith('fillRect') for n in nombres)
        v.keyPressEvent(types.SimpleNamespace(key=lambda: 70))
        assert not v.pantalla_completa
        v.keyPressEvent(types.SimpleNamespace(key=lambda: 70))
        assert v.pantalla_completa
        v.keyPressEvent(types.SimpleNamespace(key=lambda: 27))
        assert v.cerrado
        w = mv._crear_ventana(mv.EstadoMano(), demo=False, ventana=True)
        assert not w.pantalla_completa and ('resize', (900, 900), {}) in llamadas
    finally:
        for k, m in previos.items():
            if m is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = m
    return 'la ventana corre entera contra un Qt de mentira (demo, dibujo, teclas, pantalla 1); falta abrirla con Qt de verdad'


@prueba
def ventana_mi_lazo():
    """La ventana de MI del primer paso del lazo es la de la calibracion ([2, 4] s tras la senal), el estudio la mide
    desde los marcadores de una sesion y mide la BA por ventana en un EEG continuo con el ERD solo de 2 s en adelante."""
    sys.path.insert(0, str(config.RAIZ / 'estudios'))
    import ventana_mi_lazo as vm
    # las constantes: la calibracion decide con los ultimos VENTANA_MI s de DURACION_MI_S y el primer paso del lazo con lo mismo
    assert config.MI_VENTANA_OBJETIVO_S == (2.0, 4.0)
    assert (config.DURACION_MI_S - config.VENTANA_MI, config.DURACION_MI_S) == config.MI_VENTANA_OBJETIVO_S
    assert vm.primera_ventana_esperada() == config.MI_VENTANA_OBJETIVO_S
    assert config.VENTANA_MI + config.ESPERA_PRIMER_PASO_S == config.DURACION_MI_S
    import orquestador
    assert orquestador.argumentos(['real']).duracion_mi == config.DURACION_MI_S
    # solapamiento con [2, 4]
    assert abs(vm.solapamiento(1.05, 3.05) - 1.05) < 1e-9 and vm.solapamiento(2, 4) == 2.0 and vm.solapamiento(4.8, 6.8) == 0.0
    # marcadores de un lazo con 2 ensayos (el centrado y las ordenes de CP1 y de la calibracion de ErrP no cuentan)
    reg = [{'t': 0.5, 'marcador': config.CUE_CERRAR}, {'t': 0.9, 'marcador': 'paso_ack:1'},
           {'t': 5.0, 'marcador': 'bloque:LAZO_ESTATICO'}]
    for base in (10.0, 20.0):
        reg += [{'t': base, 'marcador': config.CUE_CERRAR}, {'t': base + 4.05, 'marcador': 'paso_ack:7'},
                {'t': base + 5.1, 'marcador': 'paso_quieto:8'}, {'t': base + 6.2, 'marcador': 'paso_ack:9'},
                {'t': base + 6.3, 'evento': {'tipo': 'paso'}}]
    por = vm.ventanas_por_paso(reg)
    assert sorted(por) == [1, 2, 3] and all(len(v) == 2 for v in por.values()), por
    filas = vm.tabla_sesion(por)
    assert abs(filas[0]['ini'] - 2.05) < 1e-9 and abs(filas[0]['fin'] - 4.05) < 1e-9 and filas[0]['cubre_s'] > 1.94
    assert filas[1]['cubre_s'] < 1.0 and filas[2]['cubre_s'] == 0.0
    # BA por ventana: EEG continuo de 250 Hz con 36 ensayos de 4 s; el ERD (potencia 8-30 Hz en C3) solo de 2 s en adelante
    fs = 250.0
    rng = np.random.default_rng(0)
    n = int(fs * 36 * 7)
    x = rng.normal(0, 1.0, (8, n))
    t = np.arange(n) / fs
    cues = [(3.0 + 7.0 * k, 1 if k % 2 == 0 else -1) for k in range(36)]
    ritmo = np.sin(2 * np.pi * 11 * t)
    for c, meta in cues:
        if meta > 0:                                            # una clase deja un ritmo de 11 Hz en C3, pero solo de 2 a 4 s
            i0, i1 = int((c + 2.0) * fs), int((c + 4.0) * fs)
            x[1, i0:i1] += 3.0 * ritmo[i0:i1]
    import hardware as hw
    xf = hw.filtrar(x, config.BANDA_MI, fs)
    filas = vm.ba_por_ventana(xf, t, cues, fs, ventanas=((0, 2), (2, 4)), repeticiones=3, permutaciones=20)
    ba_ant, ba_tras = filas[0][2], filas[1][2]
    assert ba_tras > 0.85 and ba_ant < 0.75 and filas[1][3] < 0.1 and filas[0][3] > 0.1, filas
    return f'primer paso = [2, 4] s; el estudio mide pasos y ventanas (BA {ba_ant:.2f} antes del ERD, {ba_tras:.2f} sobre el)'


@prueba
def cp1_red_robusto():
    """CP1, 60 Hz: una ventana mala no tumba el CP1 (decide la mediana de 3 ventanas para los canales sospechosos), un canal
    contaminado de verdad sigue fallando, y lo que no es 60 Hz (rms, saturacion) no se toca. EEG sintetico."""
    import types
    import hardware as hw
    fs, seg = 250.0, 4.0
    n = int(seg * fs)
    t = np.arange(n) / fs
    nombres = config.CANALES_EEG
    assert config.CP1_RED == {'umbral': 0.5, 'ventanas': 3}

    def eeg_falso(amplitudes, saturar=()):
        """amplitudes[w][canal] = uV del 60 Hz en la ventana w (la ultima se repite). Cada llamada a _crudo avanza una ventana."""
        estado = {'w': -1}

        def _crudo(segundos):
            estado['w'] += 1
            rng = np.random.default_rng(estado['w'])
            a = amplitudes[min(estado['w'], len(amplitudes) - 1)]
            x = rng.normal(0, 12.0, (8, n)) + 210_000.0 + np.array(a)[:, None] * np.sin(2 * np.pi * 60.0 * t)
            for k in saturar:
                x[k] += 1e6
            return x, t
        yo = types.SimpleNamespace(fs=fs, _crudo=_crudo)
        yo.calidad = lambda s: hw.EntradaEEG.calidad(yo, s)
        return yo

    limpio, alto = [3.0] * 8, [3.0] * 8
    alto[2] = 30.0                                              # Cz con el 60 Hz alto (fraccion ~0.8)
    esperas, avisos = [], []
    corre = lambda yo: hw.EntradaEEG.calidad_robusta(yo, seg, esperar=esperas.append, avisar=avisos.append)
    # una sola ventana de 60 Hz alto (la primera) y luego normal: pasa por la mediana. Con la medicion de siempre, fallaba.
    yo = eeg_falso([alto, limpio, limpio])
    una = hw.EntradaEEG.calidad(eeg_falso([alto]), seg)
    assert not una[2]['ok'] and una[2]['red'] > 0.5 and all(f['ok'] for k, f in enumerate(una) if k != 2), [f['red'] for f in una]
    filas = corre(yo)
    assert all(f['ok'] for f in filas) and 'red_ventanas' in filas[2] and len(filas[2]['red_ventanas']) == 3
    assert filas[2]['red_ventanas'][0] > 0.5 > filas[2]['red_ventanas'][1] and filas[2]['red'] == np.median(filas[2]['red_ventanas'])
    assert esperas == [seg, seg] and len(avisos) == 1 and 'Cz' in avisos[0] and 'mediana de 3' in avisos[0]
    assert all('red_ventanas' not in f for k, f in enumerate(filas) if k != 2), 'solo se re-miden los sospechosos'
    # contaminado de verdad (en las tres ventanas, o en dos de tres): falla, y dice cual
    esperas.clear(), avisos.clear()
    filas = corre(eeg_falso([alto, alto, alto]))
    assert [f['canal'] for f in filas if not f['ok']] == ['Cz'] and filas[2]['red'] > 0.5
    filas = corre(eeg_falso([alto, limpio, alto]))
    assert not filas[2]['ok'], 'dos de tres ventanas altas: la mediana tambien falla'
    # todo limpio: no espera nada
    esperas.clear(), avisos.clear()
    filas = corre(eeg_falso([limpio]))
    assert all(f['ok'] for f in filas) and esperas == [] and avisos == []
    # el que no estaba sospechoso en la primera ventana no se re-mide (y asi se documenta)
    filas = corre(eeg_falso([limpio, alto, alto]))
    assert all(f['ok'] for f in filas) and esperas == []
    # lo que no es 60 Hz sigue igual: un canal saturado o con rms fuera de rango falla aunque el 60 Hz sea bajo
    filas = corre(eeg_falso([limpio], saturar=[4]))
    assert not filas[4]['ok'] and filas[4]['saturado'] > 0
    assert not hw.canal_ok(2.0, 0.1, 0.0) and not hw.canal_ok(80.0, 0.1, 0.0) and hw.canal_ok(10.0, 0.49, 0.0) and not hw.canal_ok(10.0, 0.5, 0.0)
    # sin ventanas extra (config) se comporta como calidad()
    previo = dict(config.CP1_RED)
    config.CP1_RED['ventanas'] = 1
    try:
        esperas.clear()
        assert not corre(eeg_falso([alto]))[2]['ok'] and esperas == []
    finally:
        config.CP1_RED.update(previo)
    # el simulador de estudios/red_cp1.py: un pico de una ventana tumba a la medicion de siempre y no al procedimiento robusto
    sys.path.insert(0, str(config.RAIZ / 'estudios'))
    import red_cp1
    C = np.full((60, 8), 0.2)
    C[5, 2] = 0.9                                               # un solo pico en Cz
    C[20:50, 3] = 0.7                                           # C4 alto durante 30 ventanas seguidas (sostenido)
    r = red_cp1.simular(C, ventana_s=10.0, paso_s=2.0, umbral=0.5, ventanas=3)           # 50 inicios; las ventanas extra van +5 y +10 filas
    assert len(r) == 50 and r[5].tolist() == [True, False, True] and r[0].tolist() == [False, False, False]
    assert r[25].tolist() == [True, True, True]
    assert r[:, 0].sum() == 1 + 30 and r[:, 1].sum() == 25, (r[:, 0].sum(), r[:, 1].sum())   # el pico aislado ya no falla; lo sostenido, si
    # el orquestador usa la robusta
    fuente = open(config.RAIZ / 'orquestador.py', encoding='utf-8').read()
    assert 'self.eeg.calidad_robusta(' in fuente and 'self.eeg.calidad(self.a.seg_revision)' not in fuente
    return 'una ventana alta se corrige con la mediana de 3; contaminacion sostenida, saturacion y rms fuera de rango siguen fallando'


@prueba
def detencion_texto():
    """detencion.py: que dice la parada de cada CP (el arbol de docs/DOMINGO.md, seccion 5), con los numeros que decidieron el NO GO."""
    import json
    import detencion as dt
    todo = lambda l: ' '.join(l)
    # CP1: segun lo que fallo, y con el COM solo si la ortesis va por USB
    p = dt.pasos(1, {'senal_ok': False, 'canales_malos': ['Fz', 'Cz'], 'latencia_ok': True})
    assert 'Fz, Cz' in p[0] and 'gel' in p[0] and 'demo.py planb' in todo(p) and 'ESP32' not in todo(p)
    p = dt.pasos(1, {'senal_ok': True, 'canales_malos': [], 'latencia_ok': False}, 'COM4')
    assert 'verificar_ortesis.py --puerto COM4' in todo(p) and '--ortesis-sim' in todo(p) and 'gel' not in todo(p)
    assert 'verificar_ortesis' not in todo(dt.pasos(1, {'latencia_ok': False}, None)) and 'ortesis' in todo(dt.pasos(1, {'latencia_ok': False}))
    p = dt.pasos(1, {'senal_ok': False, 'canales_malos': ['C3'], 'latencia_ok': False}, 'COM4')
    assert 'gel' in todo(p) and 'COM4' in todo(p)
    # CP1 sin muestras de EEG: el casco no esta enviando, no hay electrodos que reacomodar
    p = dt.pasos(1, {'senal_ok': False, 'canales_malos': [], 'sin_muestras': True, 'latencia_ok': True})
    assert 'No llegaron muestras' in todo(p) and 'ver_flujos.py' in todo(p) and 'REVISAR' not in todo(p) and 'gel' not in todo(p), p
    p = dt.pasos(1, {'senal_ok': False, 'canales_malos': [], 'sin_muestras': True, 'latencia_ok': False}, 'COM4')
    assert 'No llegaron muestras' in todo(p) and 'verificar_ortesis.py --puerto COM4' in todo(p) and 'REVISAR' not in todo(p), p
    assert 'REVISAR' in todo(dt.pasos(1, {'senal_ok': False, 'canales_malos': [], 'sin_muestras': False}))      # mal sin canal nombrado
    # CP1 con la ortesis simulada: la latencia mala es de la laptop, y no se sugiere pasar a la simulada porque ya lo es
    p = dt.pasos(1, {'senal_ok': True, 'canales_malos': [], 'latencia_ok': False}, None, simulada=True)
    assert 'simulada' in todo(p) and 'saturada' in todo(p) and '--ortesis-sim' not in todo(p) and 'verificar_ortesis' not in todo(p), p
    assert '--ortesis-sim' in todo(dt.pasos(1, {'latencia_ok': False}, None, simulada=False))
    assert dt.evento(1, 'x', {'latencia_ok': False}, None, True)['que_hacer'] == p              # evento() la pasa tal cual
    # CP2
    p = dt.pasos(2, {'ba': 0.65})
    assert 'cambia de piloto' in todo(p) and '--forzar' in todo(p) and 'sin moverla' in todo(p)
    assert 'ningun ensayo valido' in todo(dt.pasos(2, {'sin_ensayos': True}))
    # CP3: las ramas del DOMINGO.md
    p = todo(dt.pasos(3, {'ba': 0.60, 'espec': 0.95}))
    assert 'no informa' in p and '--solo-errp' in p and 'planb' in p and '--saltar-calibracion' not in p
    p = todo(dt.pasos(3, {'ba': 0.80, 'espec': 0.80}))
    assert 'falsas alarmas' in p and '--solo-errp' in p and '--saltar-calibracion' not in p
    for ba, espec in ((0.70, 0.95), (0.85, 0.87), (0.70, 0.80), (0.65, 0.90)):       # dos caminos
        p = todo(dt.pasos(3, {'ba': ba, 'espec': espec}))
        assert '--solo-errp' in p and '--saltar-calibracion' in p and 'Dos caminos' in p and 'CP4 puede dar NO GO' in p, (ba, espec)
    assert '--solo-errp' in todo(dt.pasos(3, {'sin_epocas': True}))
    # los cortes del CP3 son los de config (no numeros sueltos en detencion.py): moverlos mueve las ramas, y en el corte exacto no cambia de rama
    corte_ba, corte_espec = config.DETENCION_CP3_BA_NO_INFORMA, config.DETENCION_CP3_ESPEC_FALSAS_ALARMAS
    assert 'no informa' in todo(dt.pasos(3, {'ba': corte_ba - 0.01, 'espec': 0.95})) and 'no informa' not in todo(dt.pasos(3, {'ba': corte_ba, 'espec': 0.95}))
    ba_alta = config.BA_MIN + 0.05
    assert 'falsas alarmas' in todo(dt.pasos(3, {'ba': ba_alta, 'espec': corte_espec - 0.01}))
    assert 'falsas alarmas' not in todo(dt.pasos(3, {'ba': ba_alta, 'espec': corte_espec}))
    try:
        config.DETENCION_CP3_BA_NO_INFORMA, config.DETENCION_CP3_ESPEC_FALSAS_ALARMAS = 0.50, 0.70
        assert 'no informa' not in todo(dt.pasos(3, {'ba': 0.60, 'espec': 0.95})) and 'Dos caminos' in todo(dt.pasos(3, {'ba': 0.60, 'espec': 0.95}))
        assert 'falsas alarmas' not in todo(dt.pasos(3, {'ba': 0.80, 'espec': 0.80}))
        assert 'no informa' in todo(dt.pasos(3, {'ba': 0.45, 'espec': 0.95}))
    finally:
        config.DETENCION_CP3_BA_NO_INFORMA, config.DETENCION_CP3_ESPEC_FALSAS_ALARMAS = corte_ba, corte_espec
    # el DOMINGO.md dice los mismos cortes: un umbral movido en config.py sin tocar el arbol (o al reves) se detecta
    dom = (config.RAIZ / 'docs' / 'DOMINGO.md').read_text(encoding='utf-8')
    assert '### CP3' in dom and '### CP4' in dom, 'el arbol del DOMINGO.md cambio de forma: ajusta esta prueba'
    cp3 = dom.split('### CP3')[1].split('### CP4')[0]
    titulo = cp3.split('\n')[0]
    assert f'{config.BA_MIN:.2f}' in titulo and f'{config.ESPEC_MIN:.2f}' in titulo, titulo
    # cada corte aparece en CADA rama del arbol que lo usa (el 0.65 en dos, el 0.85 en dos): cambiar uno solo de los dos numeros se detecta
    import re
    llano = ' '.join(re.sub(r'[*_`]', '', cp3).split())                          # sin negritas ni saltos de linea
    ba, es, ba_min, es_min = (f'{x:.2f}' for x in (corte_ba, corte_espec, config.BA_MIN, config.ESPEC_MIN))
    for frase in (f'BA entre {ba} y {ba_min}', f'BA por debajo de {ba}', f'especificidad entre {es} y {es_min}', f'especificidad < {es}'):
        assert frase in llano, f'DOMINGO.md, seccion CP3, ya no dice "{frase}": el arbol y config.DETENCION_CP3_* deben decir lo mismo'
    # lo que dice el texto existe: cada bandera esta en el arbol del DOMINGO.md y en el orquestador
    fuente = (config.RAIZ / 'orquestador.py').read_text(encoding='utf-8')
    for bandera in ('--solo-errp', '--saltar-calibracion', '--forzar', '--ortesis-sim'):
        assert bandera in dom and f"'{bandera}'" in fuente, bandera
    assert 'verificar_ortesis.py' in dom and (config.RAIZ / 'verificar_ortesis.py').exists()
    # sin CP conocido no inventa uno
    ev = dt.evento(None, 'se detuvo', None)
    assert ev['titulo'] == 'Sesion detenida' and ev['n'] is None and 'DOMINGO.md' in todo(ev['que_hacer'])
    assert dt.pasos(4) == dt.pasos(None)
    # el evento: titulo con el CP, serializable, todo ASCII (regla del proyecto) y escapado para el tablero
    ev = dt.evento(3, 'ErrP: sens 0.83, espec 0.87, BA 0.85 con 120 epocas (<b>x</b> & y)', {'ba': 0.85, 'espec': 0.87})
    assert ev['tipo'] == 'detenida' and ev['n'] == 3 and ev['titulo'] == 'Sesion detenida en CP3' and len(ev['que_hacer']) == 3
    assert json.loads(json.dumps(ev)) == ev
    for n, datos in ((1, {'senal_ok': False, 'canales_malos': ['Fz']}), (2, {'ba': 0.6}), (3, {'ba': 0.7, 'espec': 0.88}), (3, {'ba': 0.5})):
        assert all(ord(c) < 128 for c in json.dumps(dt.evento(n, 'x', datos, 'COM4'), ensure_ascii=False))
    h = dt.html(ev)
    assert 'Sesion detenida en CP3: ErrP: sens 0.83' in h and '&lt;b&gt;x&lt;/b&gt; &amp; y' in h and '<b>x</b>' not in h
    assert all(dt.html(dt.evento(3, 'x', {'ba': 0.7, 'espec': 0.88})).count(f) >= 1 for f in ('--solo-errp', '--saltar-calibracion'))
    c = dt.texto_consola(ev)
    assert c.startswith('Sesion detenida en CP3: ErrP') and 'Que hacer:' in c and '--solo-errp' in c and 'DOMINGO.md' in c
    return 'CP1 (con y sin muestras, simulada) a CP3 con sus ramas del arbol, cortes iguales a config y a DOMINGO.md, sin CP no inventa, escapado y ASCII'


@prueba
def detencion_orquestador():
    """Un NO GO que detiene la sesion se anuncia en Estado (tipo 'detenida') y el orquestador sale con config.SALIDA_NO_GO."""
    import contextlib
    import io
    import types
    import detencion as dt
    import orquestador
    visto = []
    sal = types.SimpleNamespace(estado=lambda **d: visto.append(d), no_go=None)
    silencio = lambda: contextlib.redirect_stdout(io.StringIO())
    # checkpoint(): solo un NO GO sin --forzar deja dicho el motivo
    with silencio():
        assert orquestador.checkpoint(sal, 3, True, 'bien', False, datos={'ba': 0.9}) and sal.no_go is None
        assert orquestador.checkpoint(sal, 3, False, 'mal', True, datos={'ba': 0.6}) and sal.no_go is None
        assert not orquestador.checkpoint(sal, 4, False, 'lento', False, informativo=True) and sal.no_go is None
        assert orquestador.checkpoint(sal, 2, False, 'x', False, informativo=True) is False and sal.no_go is None
        assert not orquestador.checkpoint(sal, 3, False, 'ErrP: sens 0.83, espec 0.87, BA 0.85', False, datos={'ba': 0.85, 'espec': 0.87})
    assert sal.no_go == {'n': 3, 'motivo': 'ErrP: sens 0.83, espec 0.87, BA 0.85', 'datos': {'ba': 0.85, 'espec': 0.87}}
    assert [e['ok'] for e in visto if e['tipo'] == 'checkpoint'][-1] is False
    # anunciar_detencion(): lo publica y lo dice en la terminal
    orq = types.SimpleNamespace(salidas=sal, fsm=types.SimpleNamespace(estado='CAL_ERRP'))
    a = orquestador.argumentos(['real', '--puerto', 'COM9'])
    visto.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev = orquestador.anunciar_detencion(orq, a)
    assert visto == [ev] and ev['tipo'] == 'detenida' and ev['n'] == 3 and 'espec 0.87' in ev['motivo'], visto
    assert '--saltar-calibracion' in buf.getvalue() and 'Sesion detenida en CP3' in buf.getvalue()
    # CP1 por latencia: el COM de la ortesis por USB, y nada de COM si va simulada o por Wi-Fi AUNQUE se haya dado --puerto
    sal.no_go = {'n': 1, 'motivo': 'latencia', 'datos': {'senal_ok': True, 'canales_malos': [], 'latencia_ok': False}}
    for args, hay_com, sugiere_sim, verificar in ((['real', '--puerto', 'COM9'], True, True, True),
                                                   (['real', '--puerto', 'COM9', '--ortesis-sim'], False, False, False),
                                                   (['real', '--puerto', 'COM9', '--ortesis-udp'], False, True, False),
                                                   (['real', '--ortesis-udp', '--puerto', 'COM9'], False, True, False),
                                                   (['real'], False, True, True)):                     # sin --puerto: el de config
        with silencio():
            ev = orquestador.anunciar_detencion(orq, orquestador.argumentos(args))
        texto = ' '.join(ev['que_hacer'])
        assert ev['n'] == 1 and ev['titulo'] == 'Sesion detenida en CP1', ev
        assert ('COM9' in texto) == hay_com, (args, texto)
        assert ('verificar_ortesis.py' in texto) == verificar and ('--ortesis-sim' in texto) == sugiere_sim, (args, texto)
        if args == ['real']:
            assert config.PUERTO_ORTESIS in texto, texto
        if '--ortesis-sim' in args:                         # la simulada no puede sugerirse a si misma
            assert 'simulada' in texto, texto
    # sin motivo dicho: el CP sale del estado de la maquina; sin estado conocido, no inventa
    sal.no_go = None
    for estado, n in (('IMPEDANCIAS', 1), ('CAL_MI', 2), ('CAL_ERRP', 3), ('LAZO_ESTATICO', None)):
        orq.fsm.estado = estado
        with silencio():
            ev = orquestador.anunciar_detencion(orq, a)
        assert ev['n'] == n and 'consola dice por que' in ev['motivo'], (estado, ev)

    # correr(): devuelve SALIDA_NO_GO si preparar() se detuvo, y publica la parada antes de cerrar
    class Parado:
        prog, sham, lista = None, None, False

        def __init__(self):
            self.salidas = types.SimpleNamespace(estado=lambda **d: visto.append(d), no_go={'n': 2, 'motivo': 'MI: BA 0.55', 'datos': {'ba': 0.55}})
            self.fsm = types.SimpleNamespace(estado='CAL_MI', ir_a=lambda n: None)
            self.evaluados = self.cerrados = 0

        def preparar(self):
            return None

        def evaluar(self):
            self.evaluados += 1

        def cerrar(self):
            self.cerrados += 1
    p = Parado()
    visto.clear()
    with silencio():
        codigo = orquestador.correr(p, a)
    assert codigo == config.SALIDA_NO_GO == 4 and [e['tipo'] for e in visto] == ['detenida'] and visto[0]['n'] == 2
    assert p.evaluados == 1 and p.cerrados == 1                       # sigue cerrando como siempre
    # si el aviso mismo falla (Estado caido, por ejemplo) la sesion igual se cierra y sale con 4
    def estado_roto(**d):
        raise OSError('Estado caido')
    p = Parado()
    p.salidas.estado = estado_roto
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert orquestador.correr(p, a) == config.SALIDA_NO_GO
    assert p.evaluados == 1 and p.cerrados == 1 and 'no se pudo anunciar la detencion' in buf.getvalue() and 'Estado caido' in buf.getvalue()
    # una sesion completa sale con 0 (las pruebas del simulador la corren de verdad)
    b = orquestador.argumentos(['sim', '--ciclo', '0', '--pasos_estatico', '6', '--pasos_adaptativo', '6', '--sin_perturbacion'])
    with silencio():
        assert orquestador.correr(orquestador.Orquestador(orquestador.BackendSim(b), b), b) == 0
    # main(): ese codigo es el de salida del proceso
    originales = orquestador.BackendSim, orquestador.Orquestador, orquestador.correr
    try:
        orquestador.BackendSim, orquestador.Orquestador = (lambda a: None), (lambda *x: None)
        for devuelve, sale in ((config.SALIDA_NO_GO, config.SALIDA_NO_GO), (0, None)):
            orquestador.correr = lambda orq, a, d=devuelve: d
            try:
                orquestador.main(['sim'])
                codigo = None
            except SystemExit as e:
                codigo = e.code
            assert codigo == sale, (devuelve, codigo)
    finally:
        orquestador.BackendSim, orquestador.Orquestador, orquestador.correr = originales
    # el cableado, con AST: toda llamada a checkpoint() de un CP que detiene (1, 2 o 3) pasa `datos=`; sin eso detencion.pasos
    # no sabe que fallo y cae en el texto generico. El CP4 solo informa y no lo necesita
    import ast
    arbol = ast.parse((config.RAIZ / 'orquestador.py').read_text(encoding='utf-8'))
    con_datos = {}
    for nodo in ast.walk(arbol):
        nombre = getattr(nodo.func, 'id', getattr(nodo.func, 'attr', None)) if isinstance(nodo, ast.Call) else None
        if nombre == 'checkpoint' and len(nodo.args) >= 2 and isinstance(nodo.args[1], ast.Constant):
            con_datos.setdefault(nodo.args[1].value, []).append('datos' in {k.arg for k in nodo.keywords})
    assert {1, 2, 3, 4} <= set(con_datos), con_datos                    # si no hay llamadas, esta revision no miraria nada
    for n in (1, 2, 3):
        assert all(con_datos[n]), f'una llamada a checkpoint() del CP{n} no pasa datos=: {con_datos}'

    # el cableado de BackendReal, con falsos (sin casco, sin ortesis y sin tardar): CP1, CP2 y CP3 dejan el motivo y los numeros
    import hardware

    class EEGFalso:
        def __init__(self, filas):
            self.filas = filas

        def calidad_robusta(self, segundos, esperar=None, avisar=None):
            return self.filas

    class OrtesisFalsa:
        def __init__(self, con_ack=True):
            self.con_ack, self.movimientos = con_ack, []

        def mover(self, fraccion, dur_ms=None):
            self.movimientos.append(fraccion)
            return len(self.movimientos), (1.0 if self.con_ack else None), 8.0 + len(self.movimientos) % 2     # ACK de 8 a 9 ms

    def fila(canal, ok=True):
        return {'canal': canal, 'rms_uv': 12.0, 'red': 0.1 if ok else 0.7, 'saturado': 0.0, 'ok': ok,
                **({} if ok else {'red_ventanas': [0.7, 0.6, 0.7]})}

    def backend(*argv, filas=(), con_ack=True, hw=hardware):
        b = object.__new__(orquestador.BackendReal)                       # sin __init__: no conecta a nada
        b.a = orquestador.argumentos(['real', '--ortesis-sim', '--seg_revision', '0', *argv])
        b.hw, b.eeg, b.ortesis, b.decoder, b.detector = hw, EEGFalso(list(filas)), OrtesisFalsa(con_ack), None, None
        sal = types.SimpleNamespace(estado=lambda **d: visto.append(d), no_go=None, marcador=lambda *a, **k: None)
        return b, types.SimpleNamespace(salidas=sal, fsm=types.SimpleNamespace(estado='IMPEDANCIAS', ir_a=lambda n: pasos_fsm.append(n)))
    pasos_fsm = []
    buenas, una_mala = [fila('Fz'), fila('C3')], [fila('Fz'), fila('C3', False), fila('Oz', False)]
    esperado = {'ok': dict(senal_ok=True, canales_malos=[], sin_muestras=False, latencia_ok=True),
                'canal': dict(senal_ok=False, canales_malos=['C3', 'Oz'], sin_muestras=False, latencia_ok=True),
                'vacia': dict(senal_ok=False, canales_malos=[], sin_muestras=True, latencia_ok=True),
                'sin_ack': dict(senal_ok=True, canales_malos=[], sin_muestras=False, latencia_ok=False),
                'ambos': dict(senal_ok=False, canales_malos=['C3', 'Oz'], sin_muestras=False, latencia_ok=False)}
    casos = (('canal', una_mala, True), ('vacia', [], True), ('sin_ack', buenas, False), ('ambos', una_mala, False))
    movimientos = config.CP1_MOVIMIENTOS
    config.CP1_MOVIMIENTOS = 2
    try:
        with silencio():
            b, o = backend(filas=buenas)
            assert b.revisar(o) is True and o.salidas.no_go is None and len(b.ortesis.movimientos) == 2
            assert [e['ok'] for e in visto if e['tipo'] == 'checkpoint'][-1] is True
            for clave, filas, con_ack in casos:
                b, o = backend(filas=filas, con_ack=con_ack)
                assert b.revisar(o) is False, clave
                ng = o.salidas.no_go
                assert ng['n'] == 1 and ng['datos'] == esperado[clave], (clave, ng)
                assert ('revisar C3, Oz' in ng['motivo']) == (clave in ('canal', 'ambos')) and 'calidad de senal ok' in ng['motivo'], ng
                # y de ahi sale el texto del tablero: lo que fallo, no el generico
                dicho = ' '.join(dt.pasos(1, ng['datos']))
                assert ('No llegaron muestras' in dicho) == (clave == 'vacia') and ('REVISAR' in dicho) is False, (clave, dicho)
                assert ('Reacomoda C3, Oz' in dicho) == (clave in ('canal', 'ambos')), (clave, dicho)
                assert ('Revisa la alimentacion' in dicho) == (not esperado[clave]['latencia_ok']), (clave, dicho)
                # con --forzar el NO GO no detiene y no queda nada que anunciar
                b, o = backend('--forzar', filas=filas, con_ack=con_ack)
                assert b.revisar(o) is True and o.salidas.no_go is None, clave
    finally:
        config.CP1_MOVIMIENTOS = movimientos
    assert config.CP1_MOVIMIENTOS == movimientos

    # CP2 y CP3 sin ensayos o sin epocas: no hay modelo con que seguir, asi que paran aunque haya --forzar y sin pasar por checkpoint()
    sin_decoder = types.SimpleNamespace(cargar=lambda nombre: types.SimpleNamespace(ba=0.9))       # --solo-errp lo carga de modelos/
    for forzar in ([], ['--forzar']):
        visto.clear()
        pasos_fsm.clear()
        with silencio():
            b, o = backend('--ensayos_mi', '0', *forzar)
            assert b.calibrar_mi(o) is False
            assert o.salidas.no_go['n'] == 2 and o.salidas.no_go['datos'] == {'sin_ensayos': True}, o.salidas.no_go
            assert 'ningun ensayo valido' in ' '.join(orquestador.anunciar_detencion(o, b.a)['que_hacer'])
            b, o = backend('--ensayos_mi', '0', *forzar)
            b.revisar = lambda orq: True                                   # CP1 ya paso: preparar() sigue a la calibracion de MI
            assert b.preparar(o) is None and pasos_fsm == ['CAL_MI'] and o.salidas.no_go['n'] == 2, (forzar, pasos_fsm)
            b, o = backend('--ensayos_errp', '0', *forzar)
            assert b.calibrar_errp(o) is False
            assert o.salidas.no_go['n'] == 3 and o.salidas.no_go['datos'] == {'sin_epocas': True}, o.salidas.no_go
            assert 'ninguna epoca valida' in ' '.join(orquestador.anunciar_detencion(o, b.a)['que_hacer'])
            pasos_fsm.clear()
            b, o = backend('--solo-errp', '--ensayos_errp', '0', *forzar, hw=sin_decoder)
            b.revisar = lambda orq: True
            assert b.preparar(o) is None and pasos_fsm == ['CAL_MI', 'CAL_ERRP'] and o.salidas.no_go['n'] == 3, (forzar, pasos_fsm)
        assert not [e for e in visto if e['tipo'] == 'checkpoint'], 'estas rutas no pasan por checkpoint()'
    # el contrato: ese codigo no choca con los otros que usa el proyecto
    assert config.SALIDA_NO_GO not in (0, 1, 2, 3, 130)
    return 'NO GO sin --forzar: se anuncia con el CP y que hacer (cableado de revisar/calibrar_mi/calibrar_errp y de checkpoint(datos=)), y el proceso sale con el codigo 4; --forzar y los CP informativos no paran'


@prueba
def tablero_detenida():
    """El aviso grande de 'Sesion detenida en CPx' del tablero: aparece con el evento 'detenida', ocupa el lugar de los
    paneles y se quita cuando empieza otra sesion. Con un Qt de mentira (corre sin pantalla ni PyQt); si PyQt esta, tambien con el de verdad."""
    import importlib
    import sys
    import types
    import detencion as dt

    class Anota:
        def __getattr__(self, k):
            if k.startswith('__'):
                raise AttributeError(k)
            return Anota()

        def __call__(self, *a, **kw):
            return Anota()

    class Etiqueta:
        """Un QLabel que recuerda lo que se le puso; cualquier otro metodo de Qt es un no-op."""
        def __init__(self, texto=''):
            self._t, self._oculto, self._estilo = texto, False, ''

        def setText(self, t):
            self._t = t

        def text(self):
            return self._t

        def setVisible(self, v):
            self._oculto = not v

        def isHidden(self):
            return self._oculto

        def setStyleSheet(self, s):
            self._estilo = s

        def styleSheet(self):
            return self._estilo

        def __getattr__(self, k):
            if k.startswith('__'):
                raise AttributeError(k)
            return Anota()                                          # setWordWrap, toggled.connect, ...

    class Grafica(Etiqueta):
        def addPlot(self, **kw):
            return Anota()

    class Capa:
        def __getattr__(self, k):
            if k.startswith('__'):
                raise AttributeError(k)
            return lambda *a, **kw: 0

    class Ventana:
        """Un QWidget: solo los metodos que el tablero usa; un nombre mal escrito en el tablero aqui SI falla."""
        def __init__(self, *a):
            pass

        def resize(self, *a):
            pass

        def setWindowTitle(self, *a):
            pass

    class Reloj:
        timeout = Anota()

        def start(self, ms):
            pass

        def stop(self):
            pass

    class Constantes:
        def __getattr__(self, k):
            return k

    def con_qt(QtCore, QtWidgets, pg):
        """Corre la comprobacion con ese Qt y devuelve el tablero ya probado."""
        tab_viejo = sys.modules.pop('tablero', None)
        previos = {k: sys.modules.get(k) for k in ('pyqtgraph', 'pyqtgraph.Qt')}
        if pg is not None:
            qt = types.ModuleType('pyqtgraph.Qt')
            qt.QtCore, qt.QtWidgets = QtCore, QtWidgets
            pg.Qt = qt
            sys.modules['pyqtgraph'], sys.modules['pyqtgraph.Qt'] = pg, qt
        try:
            tablero = importlib.import_module('tablero')
            tablero.resolve_byprop = lambda *a, **k: []              # que el hilo de conexion no busque flujos de verdad
            t = tablero.Tablero()
            try:
                t.timer.stop()
                assert t.lbl_detenida.isHidden() and not t.g.isHidden() and not t.lbl_detenida.text()
                ev = dt.evento(3, 'ErrP: sens 0.83, espec 0.87, BA 0.85 con 120 epocas (x)', {'ba': 0.85, 'espec': 0.87})
                t._procesar({'tipo': 'checkpoint', 'n': 3, 'ok': False, 'texto': '[CP3] ErrP: ... -> NO GO'})
                t._procesar(ev)
                assert not t.lbl_detenida.isHidden() and t.g.isHidden()                       # el aviso ocupa el lugar de los paneles
                txt = t.lbl_detenida.text()
                assert 'Sesion detenida en CP3' in txt and 'espec 0.87' in txt and '--solo-errp' in txt and '--saltar-calibracion' in txt, txt
                assert 'CP3' in t.lbl_estado.text() and tablero.COLORES_SALUD[config.ROJO] in t.lbl_estado.styleSheet()
                t._procesar(ev)                                                             # repetido: sigue igual
                assert not t.lbl_detenida.isHidden() and t.g.isHidden()
                t._procesar({'tipo': 'salud', 'estado': 'IMPEDANCIAS', 'motivo': '', 'escalon': 0, 'colores': {}, 'detalle': {}})
                assert not t.lbl_detenida.isHidden()                                        # otros eventos no lo quitan
                for nuevo in ({'tipo': 'checkpoint', 'n': 1, 'ok': True, 'texto': '[CP1] ok'}, {'tipo': 'cue', 'meta': 1},
                              {'tipo': 'paso', 'paso': 1, 'estado': 'LAZO_ESTATICO', 'meta': 1, 'angulo': 0.5, 'p_crudo': 0.6,
                               'b': 0.5, 'p_prima': 0.6, 'P_hat': None, 'error': 0, 'error_sombra': 0, 'beta': 0.1, 'sd_beta': 0.5,
                               'youden': 0.6, 'fiabilidad': 1.0, 'congelado': False, 'cambio': '', 'latencia_ms': None,
                               'perturbado': False}):
                    t._procesar(ev)
                    assert not t.lbl_detenida.isHidden()
                    t._procesar(nuevo)                                                      # empezo otra sesion: el aviso se va
                    assert t.lbl_detenida.isHidden() and not t.g.isHidden(), nuevo['tipo']
                t._procesar({'tipo': 'detenida', 'n': 1, 'titulo': 'Sesion detenida en CP1', 'motivo': '<script>x</script>',
                             'que_hacer': ['a & b']})                                       # el motivo es texto libre: se escapa
                assert '&lt;script&gt;' in t.lbl_detenida.text() and '<script>' not in t.lbl_detenida.text()
            finally:
                if hasattr(t, 'close'):
                    t.close()
        finally:
            sys.modules.pop('tablero', None)
            if tab_viejo is not None:
                sys.modules['tablero'] = tab_viejo
            for k, v in previos.items():
                if pg is None:
                    break
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    pg = types.ModuleType('pyqtgraph')
    for nombre in ('setConfigOptions', 'mkPen', 'FillBetweenItem', 'InfiniteLine', 'PlotCurveItem', 'ScatterPlotItem'):
        setattr(pg, nombre, Anota())
    pg.GraphicsLayoutWidget = Grafica
    QtCore = types.SimpleNamespace(Qt=Constantes(), QTimer=Reloj)
    QtWidgets = types.SimpleNamespace(QWidget=Ventana, QVBoxLayout=lambda *a: Capa(), QHBoxLayout=lambda *a: Capa(), QLabel=Etiqueta,
                                      QPushButton=lambda *a: Etiqueta(*a), QLineEdit=lambda *a: Etiqueta())
    real = False
    con_qt(QtCore, QtWidgets, pg)
    try:
        import pyqtgraph                                                                    # noqa: F401  (esta el de verdad?)
        from pyqtgraph.Qt import QtWidgets as QtW
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        app = QtW.QApplication.instance() or QtW.QApplication([])                           # noqa: F841
        real = True
    except Exception:
        pass
    if real:
        con_qt(None, None, None)
    return ('aviso grande, paneles ocultos y se quita con otra sesion (Qt de mentira + el de verdad)' if real else
            'aviso grande, paneles ocultos y se quita con otra sesion (solo con un Qt de mentira: falta probarlo con PyQt)')


# Tres niveles: las rapidas no tocan la red ni esperan en tiempo real (reloj virtual o
# datos sinteticos); --lsl agrega las que levantan el gemelo o el puente y esperan en
# tiempo real (o que tardan mas de un minuto); --completa agrega las sesiones reales contra el gemelo.
RAPIDAS = ['contrato', 'vigilante', 'semaforo_piloto', 'retroceso', 'agente_basico', 'p_hat_refleja_errp', 'prior_por_paso', 'ia_sesion', 'gemelo_personal', 'diagnostico_errp_p001', 'barrido_paso', 'comparacion_baselines', 'reporte_detector', 'agente_aprende', 
           'agente_sin_sesgo', 'confianza_detector', 'maquina_estados', 'orquestador_sim', 'pausa_segura',
           'calibracion_repeticiones', 'calibracion_errp_fija', 'errp_por_direccion', 'bloque_sham', 'cp1_robusto', 'seleccion_canales_vistas',
           'coadaptativo_no_detiene_el_lazo', 'inicio_movimiento', 'rechazo_por_cabeza', 'parpadeos_cruzan_bloques', 'cierre_completo',
           'paso_sin_movimiento',
           'iic_estimador', 'gemelo_embodiment',
           'orquestador_ajenos', 'cuestionario', 'deriva_reloj', 'plan_caos', 'caos_sim',
           'caos_agente_vs_sombra', 'senal_sham', 'orquestador_sham', 'reanudar_sham', 'controles_especificidad', 'copiloto_herramientas', 'copiloto_api_simulada', 'coinvestigador_entre_bloques', 'narrador_jurado', 'tablero_salud', 'tablero_flechas', 'repetir_sesion', 'plan_b_sin_sesion_detenida', 'modelos_del_dia', 'instantanea_estado', 'modelos_hardware',
           'detector_umbral_anidado', 'decoder_preentrenado', 'intervalo_por_ensayos', 'senal_valida', 'ortesis_sin_ack',
           'ortesis_serial_reconecta', 'ortesis_udp', 'ortesis_udp_nervio', 'destello_errp_con_perdidas', 'registro_huecos', 'reloj_contador', 'puente_reconecta', 'cerebro_sintetico',
           'estado_sistema', 'memoria_sesiones',
           'demo_comandos', 'demo_ortesis_udp', 'demo_firmware_simulado', 'demo_revisiones', 'demo_limpiar_modelos', 'demo_esperar_flujo', 'demo_procesos', 'demo_lanzar_simulado', 'toques_electrodos', 'senal_neutra', 'decoder_canales_mi', 'mano_virtual_logica', 'mano_virtual_ventana_falsa', 'ventana_mi_lazo', 'cp1_red_robusto', 'detencion_texto', 'detencion_orquestador', 'tablero_detenida']
CON_LSL = ['detector_coadaptativo', 'reanudar', 'reconexion_eeg', 'silencio_sin_recrear', 'dos_flujos_eeg', 'entrada_unicorn',
           'verificar_unicorn', 'puente_hora_por_contador', 'gemelo_unicorn', 'estado_sistema_lsl', 'demo_gemelo_en_vivo']
LAZO_REAL = ['lazo_real_sintetico', 'lazo_real_caos', 'lazo_real_memoria']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lsl', action='store_true', help='agrega las pruebas con LSL en tiempo real (~2.5 min)')
    ap.add_argument('--completa', action='store_true', help='todas, con las sesiones reales contra el gemelo (~10 min)')
    a = ap.parse_args()
    nombres = RAPIDAS + (CON_LSL if a.lsl or a.completa else []) + (LAZO_REAL if a.completa else [])
    print('Pruebas ortesis-bci' + (' (completas)' if a.completa else ' (con LSL)' if a.lsl else ' (rapidas)'))
    for nombre in nombres:
        globals()[nombre]()
    ok = sum(RESULTADOS)
    print(f'\n{ok}/{len(RESULTADOS)} pruebas pasaron la prueba: {100 * ok / len(RESULTADOS):.1f}%')
    sys.exit(0 if ok == len(RESULTADOS) else 1)


if __name__ == '__main__':
    main()
@prueba
def diagnostico_errp_p001():
    """estudios/diagnostico_errp_p001.py con datos sinteticos: los estratos del diseno y la permutacion que los respeta, el t de
    Welch y la prueba de permutacion con correccion por el maximo (sin senal no encuentra nada; con una senal en una sola columna
    la encuentra a ella sola), el AUC de validacion cruzada, el efecto minimo detectable, el reconocimiento de la ortesis simulada
    (y de una que no lo es), el corte de epocas (igual al de hardware.cortar_epoca) y un registro continuo con un ErrP inyectado
    150 ms tarde, que el barrido de desfases encuentra (y pierde cuando la epoca se corta muy lejos)."""
    import sys
    sys.path.insert(0, str(config.RAIZ / 'estudios'))
    import diagnostico_errp_p001 as dg
    import cerebro_sintetico as cs
    import hardware as hw
    from scipy.stats import ttest_ind
    rng = np.random.default_rng(3)

    def diseno(n_bloques):
        """El plan de Orquestador.calibrar_errp: por bloque de 20, 10 de cada cue y 3 errores por cue, mezclados."""
        y, d = [], []
        for _ in range(n_bloques):
            b = [(obj, i < 3) for obj in (0, 1) for i in range(10)]
            for k in rng.permutation(len(b)):
                obj, err = b[k]
                y.append(int(err))
                d.append(obj if not err else 1 - obj)
        return np.array(y), np.array(d)
    # estratos (bloque x cue) y permutacion dentro de ellos
    y, d = diseno(6)
    est = dg.estratos_del_diseno(y, d)
    assert len(np.unique(est)) == 12 and all((est == e).sum() == 10 and y[est == e].sum() == 3 for e in np.unique(est))
    yp = dg.permutar_en_estratos(y, est, rng)
    assert yp.sum() == y.sum() and (yp != y).any() and all(yp[est == e].sum() == 3 for e in np.unique(est))
    # efecto minimo detectable, d de Cohen, t de Welch (contra scipy) y su version para muchas etiquetas a la vez
    assert abs(dg.efecto_minimo_detectable(1.0, 50, 50) - 0.5603) < 1e-3
    F = rng.normal(size=(120, 5))
    assert np.allclose(dg.t_welch(F, y), ttest_ind(F[y == 1], F[y == 0], equal_var=False).statistic)
    Y = np.stack([dg.permutar_en_estratos(y, est, rng) for _ in range(4)])
    muchas = dg._t_welch_muchas(F, Y.astype(float))
    assert all(np.allclose(muchas[k], dg.t_welch(F, Y[k])) for k in range(4))
    G = rng.normal(size=(4000, 1))
    yg = np.arange(4000) % 2
    assert abs(dg.d_de_cohen(G + yg[:, None] * 1.0, yg)[0] - 1.0) < 0.1
    # prueba de permutacion con maximo: sin senal no pasa ninguna columna; con una senal de d = 1.5 pasa solo esa
    ruido = rng.normal(size=(120, 40))
    r0 = dg.prueba_permutacion(ruido, y, est, 400, rng)
    assert r0['p_corregido'].min() > 0.05 and r0['p_sin_corregir'].shape == (40,), r0['p_corregido'].min()
    senal = ruido.copy()
    senal[y == 1, 7] += 1.5
    r1 = dg.prueba_permutacion(senal, y, est, 400, rng)
    assert r1['p_corregido'][7] < 0.05 and (r1['p_corregido'] < 0.05).sum() == 1, r1['p_corregido'][7]
    assert r1['n_sobre_umbral'] >= 1 and r1['max_nula'].shape == (400,)
    # AUC de validacion cruzada: la senal de d = 1.5 da ~0.86 y el ruido ~0.5 (o un poco menos, como siempre en validacion cruzada)
    assert dg.auc_cv(senal[:, [7, 8, 9]], y) > 0.7 and abs(dg.auc_cv(ruido[:, :10], y) - 0.5) < 0.15
    # el LDA propio (mas rapido que el de sklearn) da los mismos puntajes, con su sesgo; y el AUC por rangos es el de sklearn
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    for ent, pru in list(StratifiedKFold(5, shuffle=True, random_state=0).split(senal[:, :12], y))[:2]:
        ref = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(senal[ent, :12], y[ent]).decision_function(senal[pru, :12])
        assert np.allclose(dg.puntajes_lda(senal[ent, :12], y[ent], senal[pru, :12]), ref, atol=1e-9)
    assert abs(dg.auc_de_puntajes(y, senal[:, 7]) - roc_auc_score(y, senal[:, 7])) < 1e-12
    # medias por ventana (rampa conocida: la media de la ventana que empieza a 100 ms es la del indice 75 a 99) y bootstrap
    rampa = np.tile(np.arange(250.0), (3, 2, 1))
    W, ini = dg.medias_por_ventana(rampa, 0.10, 0.30, 0.10)
    assert W.shape == (3, 2, 2) and np.allclose(W[:, :, 0], 87.0) and np.allclose(ini, [0.10, 0.20])
    Xb = rng.normal(size=(60, 2, 50))
    yb = (np.arange(60) % 3 == 0).astype(int)
    Xb[yb == 1] += 2.0
    obs, bajo, alto, dd = dg.bootstrap_onda_diferencia(Xb, yb, 200, rng)
    assert obs.shape == bajo.shape == (2, 50) and dd.shape == (200, 2, 50) and (bajo > 0).all() and (bajo < obs).all() and (obs < alto).all()
    # la ortesis simulada se reconoce por su latencia (y una que no depende de `seq` de esa forma, no)
    seqs = list(range(42, 172))
    lat = np.array([hw.latencia_mecanica_simulada(q, 0) for q in seqs]) + 0.008 + rng.normal(0, 0.003, len(seqs))
    r = dg.ortesis_parece_simulada(seqs, lat)
    assert r['simulada'] is True and r['semilla'] == 0 and r['r'] > 0.99 and abs(r['desfase_mediano_ms'] - 8.0) < 1.5, r
    assert dg.ortesis_parece_simulada(seqs, rng.uniform(0.03, 0.15, len(seqs)))['simulada'] is False
    assert dg.ortesis_parece_simulada(seqs, np.full(len(seqs), 0.1))['simulada'] is None
    ack, ini_ = dg.resumen_pasos([{'t': 1.0, 'marcador': 'paso_ack:7'}, {'t': 1.1, 'marcador': 'paso_inicio:7'}, {'t': 0.5, 'marcador': 'cue_cerrar'}])
    assert ack == {7: 1.0} and ini_ == {7: 1.1}
    # registro continuo: 8 canales con desfase de continua, sin huecos, y un ErrP del gemelo inyectado 150 ms despues del inicio
    fs = dg.FS
    yc, dc = diseno(5)
    estc = dg.estratos_del_diseno(yc, dc)
    t = 100.0 + np.arange(int(150 * fs)) / fs
    x = rng.normal(0, 40.0, size=(8, len(t))) + 2e5
    t0s = 105.0 + 1.3 * np.arange(len(yc))
    xi = dg.inyectar_errp(x, t, t0s + 0.15, yc == 1, 10.0, cs.W_ERRP)
    dentro = np.zeros(len(t), dtype=bool)
    for t0, e in zip(t0s + 0.15, yc):
        if e:
            dentro |= (t >= t0) & (t < t0 + 1.0)
    assert not (xi - x)[:, ~dentro].any() and (xi - x)[:, dentro].any()
    onda = dg.onda_diferencia_gemelo(10.0)[1]
    assert abs(np.abs(xi[2] - x[2]).max() - np.abs(onda).max()) < 0.05 * np.abs(onda).max()      # el canal de peso 1 recibe la onda entera
    tramos = dg.preparar_continuo(xi, t)
    assert len(tramos) == 1
    e_hw = hw.cortar_epoca(xi, t, t0s[3], fs)
    assert e_hw is not None and np.allclose(dg.cortar(tramos, t0s[3]), e_hw, atol=1e-9)           # el mismo corte que hardware.cortar_epoca
    assert not np.allclose(dg.cortar(tramos, t0s[3], linea_base=False), e_hw)
    sc = dg.escanear_desfases(tramos, t0s, yc, estc, dg.DESFASES_S, config.indices('errp'), 0, rng)
    i = int(np.argmax(sc['auc']))
    # las ventanas de 100 ms cubren 600 ms, asi que el maximo cae en una meseta ancha alrededor del retraso (no en un punto), y cortar la
    # epoca muy lejos de ahi (-300 o +500 ms) pierde la senal
    assert abs(sc['desfases_s'][i] - 0.15) <= 0.30 and max(sc['auc']) > 0.75, (sc['desfases_s'][i], sc['auc'])
    assert sc['auc'][0] < max(sc['auc']) - 0.15 and sc['auc'][-1] < max(sc['auc']) - 0.15, sc['auc']
    # curva de aprendizaje: n epocas hechas de instantes al azar del registro, con la onda inyectada en el 30 % (aqui sobre el ruido solo)
    cur = dg.curva_de_aprendizaje(x, t, 10.0, (60,), 1, rng)
    assert set(cur) == {60} and len(cur[60]) == 4 and cur[60][0] > 0.65 and cur[60][2] > 0.6, cur       # (media y sd con Fz/Cz/Pz, media y sd con 8 canales)
    # giro de la cabeza: la velocidad angular maxima de la ventana de la epoca
    imu = np.zeros((6, len(t)))
    imu[4, 5000] = 90.0
    g = dg.giro_maximo(imu, t, [t[5000] + 0.1, t[5000] + 3.0])
    assert g[0] == 90.0 and g[1] == 0.0
    return f"estratos, permutacion con maximo, ortesis simulada y barrido de desfases (ErrP inyectado a +150 ms: AUC maximo {max(sc['auc']):.2f} en {sc['desfases_s'][i] * 1e3:+.0f} ms)"


