"""Pruebas automaticas, sin hardware. Correlas antes de cada commit.

Uso:  python pruebas.py              pruebas rapidas (~1 min)
      python pruebas.py --completa   ademas el lazo real contra el cerebro sintetico (~5 min)
"""
import argparse
import csv
import subprocess
import sys
import time
import traceback

import numpy as np

import config

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
    assert config.COLUMNAS_CSV[-3:] == ['salud', 'excluido', 'alineacion']
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
    assert v.codigo() == 'VVVC' and v.escalon() == 1 and v.motivo_pausa() is None
    # al terminar de calentar publica su primer color real
    assert v.actualizar(0.0, detector={'fiabilidad': 1.0, 'congelado': False, 'epocas': minimo})         == [('detector', config.VERDE)]
    assert v.codigo() == 'VVVV' and v.escalon() == 1 and v.motivo_pausa() is None
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
    a = orquestador.argumentos(['sim', '--ciclo', '0', '--pasos_estatico', '60', '--pasos_adaptativo', '10',
                                '--sin_perturbacion'])
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
    a = orquestador.argumentos(['sim', '--ciclo', '0', '--pasos_estatico', '20',
                                '--pasos_adaptativo', '60', '--sin_perturbacion'])
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
        assert len(f['salud']) == 4 and set(f['salud']) <= set('VARC'), f
    # el detector calienta sus primeras 15 epocas validas sin emitir marcadores de salud
    assert [f['salud'][3] for f in filas[:15]] == ['C'] * 15 and filas[20]['salud'][3] != 'C'
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


# ------------------------------------------------------------ tablero
@prueba
def tablero_salud():
    """El tablero (sin pantalla) pinta los tres semaforos y PAUSA_SEGURA en rojo con el electrodo."""
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from pyqtgraph.Qt import QtWidgets
    import tablero
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    C = tablero.COLORES_SALUD
    t = tablero.Tablero()
    try:
        t.timer.stop()
        assert set(t.semaforos) == {'eeg', 'ortesis', 'detector'}
        assert C[config.CALENTANDO] in t.semaforos['detector'].styleSheet()      # gris al arrancar
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
    finally:
        t.close()
    return 'tres semaforos (gris al calentar), PAUSA SEGURA en rojo con el electrodo y la causa'


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
    leve = _sesion_caos(0, caos=1, nivel='leve')              # el nivel leve excluye mucho menos
    assert 0 < sum(leve.excluidos.values()) < sum(orq.excluidos.values()) / 3, (leve.excluidos, orq.excluidos)
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
    for k in range(10):
        o.mover(0.3 + 0.04 * k)
    media, sd = o.jitter()
    assert 3 < media < 20
    return (f'recentrado: {acc:.0%} tras mezclar canales; ErrP BA {det.ba:.2f}, rareza ok; '
            f'ACK {media:.1f}+-{sd:.1f} ms')


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
    orq = types.SimpleNamespace(salidas=Salidas())
    assert b.calibrar_errp(orq)
    assert len(b.detector.y_cal) == 60 and b.detector.ba > 0.9, (len(b.detector.y_cal), b.detector.ba)
    return f'usa las 60 epocas pedidas aunque el detector es casi perfecto (BA {b.detector.ba:.2f})'


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
    contador). Con perdidas de Bluetooth la ventana de MI sigue saliendo."""
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
                for _ in range(10):                # con ~7 % de perdidas casi todas las ventanas valen
                    ventana_mi = eeg.ventana(3.0, config.SALUD['mi_perdida_max'], config.SALUD['mi_hueco_max_s'])[0]
                    if ventana_mi is not None:
                        break
                    time.sleep(0.5)
                x, t = eeg._crudo(18.0)            # todo lo grabado: asi siempre hay algun hueco
                giro = eeg.movimiento(t[-1] - 10.0, t[-1])
                retraso = local_clock() - t[-1]
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


# Tres niveles: las rapidas no tocan la red ni esperan en tiempo real (reloj virtual o
# datos sinteticos); --lsl agrega las que levantan el gemelo o el puente y esperan en
# tiempo real (o que tardan mas de un minuto); --completa agrega las sesiones reales contra el gemelo.
RAPIDAS = ['contrato', 'vigilante', 'retroceso', 'agente_basico', 'p_hat_refleja_errp', 'agente_aprende',
           'agente_sin_sesgo', 'confianza_detector', 'maquina_estados', 'orquestador_sim', 'pausa_segura',
           'calibracion_repeticiones', 'calibracion_errp_fija', 'cp1_robusto', 'seleccion_canales_vistas',
           'inicio_movimiento', 'rechazo_por_cabeza', 'cierre_completo', 'deriva_reloj', 'plan_caos', 'caos_sim',
           'caos_agente_vs_sombra', 'tablero_salud', 'instantanea_estado', 'modelos_hardware',
           'detector_umbral_anidado', 'intervalo_por_ensayos', 'senal_valida', 'ortesis_sin_ack',
           'ortesis_serial_reconecta', 'registro_huecos', 'reloj_contador', 'puente_reconecta', 'cerebro_sintetico']
CON_LSL = ['detector_coadaptativo', 'reanudar', 'reconexion_eeg', 'silencio_sin_recrear', 'dos_flujos_eeg', 'entrada_unicorn',
           'verificar_unicorn', 'puente_hora_por_contador', 'gemelo_unicorn']
LAZO_REAL = ['lazo_real_sintetico', 'lazo_real_caos']


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
