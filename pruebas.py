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
    assert config.COLUMNAS_CSV[-2:] == ['salud', 'excluido']
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
    antes = set(config.RESULTADOS.glob('sesion_sim_*.csv'))
    orquestador.main(['sim', '--ciclo', '0', '--pasos_estatico', '20', '--pasos_adaptativo', '90'])
    nuevos = set(config.RESULTADOS.glob('sesion_sim_*.csv')) - antes
    assert len(nuevos) == 1
    filas = list(csv.DictReader(open(nuevos.pop())))
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
    for tipo in ('corte_eeg', 'rafaga_parpadeos', 'canal'):
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
        for tipo in ('corte_eeg', 'rafaga_parpadeos', 'canal'):
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
    return 'canales, ventana y epoca validadas por tiempo'


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
    """El flujo enmudece 2.5 s y sigue con la MISMA instancia (un tiron del dongle). El
    suavizado de marcas de LSL (dejitter) quedaria desfasado segundos; EntradaEEG renueva
    su entrada y las marcas vuelven a coincidir con el reloj en cuanto regresan los datos."""
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
                    pub['t'] += n / fs
                    out.push_chunk(rng.normal(0, 10, size=(n, 8)).tolist(), pub['t'])
            time.sleep(0.02)
    threading.Thread(target=publicar, daemon=True).start()
    eeg = hw.EntradaEEG(segundos=10.0, timeout=8.0, nombre=nombre)
    sensible = hw.EntradaEEG(segundos=10.0, timeout=8.0, nombre=nombre, renovar_s=0.3)
    import orquestador                              # el umbral se ajusta por linea de comandos
    assert orquestador.argumentos(['real']).renovar_eeg == config.SALUD['eeg_renovar_s']
    assert orquestador.argumentos(['real', '--renovar-eeg', '0.5']).renovar_eeg == 0.5
    retraso = lambda: local_clock() - eeg.ultimo_t()
    try:
        time.sleep(4.0)
        # una rafaga corta de paquetes perdidos (0.6 s) NO renueva con el umbral por defecto (1 s),
        # para no vaciar el buffer con cada tiron del dongle; con un umbral de 0.3 s si
        assert config.SALUD['eeg_renovar_s'] == 1.0
        antes = retraso()
        assert antes < 0.1, antes
        pub['callado'] = True
        time.sleep(0.6)
        pub['callado'] = False
        derivas = []
        for _ in range(15):
            time.sleep(0.1)
            derivas.append(abs(eeg.lecturas()['reloj_ms']))
        assert eeg.renovaciones == 0 and sensible.renovaciones == 1, (eeg.renovaciones, sensible.renovaciones)
        # sin renovar, las marcas quedan desfasadas un rato: el semaforo del reloj lo ve (y el agente no aprende)
        assert max(derivas) >= config.SALUD['reloj_rojo_ms'], max(derivas)
        sensible.cerrar()
        time.sleep(2.0)
        pub['callado'] = True
        t_corte = local_clock()
        time.sleep(2.5)
        pub['callado'] = False
        time.sleep(2.0)
        despues = retraso()
        assert abs(despues - antes) < 0.1, f'marcas desfasadas {1000 * (despues - antes):.0f} ms tras el silencio'
        assert eeg.reconexiones == 0 and eeg.renovaciones >= 1, (eeg.reconexiones, eeg.renovaciones)
        assert eeg.ventana(3.0) == (None, None)            # aun no hay 3 s limpios tras el hueco
        assert eeg.epoca(t_corte + 2.0) is None            # la epoca cruzaria el hueco
        time.sleep(2.5)
        x, t = eeg.ventana(3.0)
        assert x is not None and t[0] > t_corte + 2.0
        assert abs(eeg.lecturas()['reloj_ms']) < config.SALUD['reloj_rojo_ms']
    finally:
        pub['vivo'] = False
        eeg.cerrar()
    return f'tras 2.5 s de silencio las marcas coinciden con el reloj (diferencia {1000 * (despues - antes):+.0f} ms)'


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
                                '--espera', '0.3', '--ensayos_errp', '60', '--min_errp', '60',
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--completa', action='store_true')
    a = ap.parse_args()
    print('Pruebas ortesis-bci')
    for p in (contrato, vigilante, retroceso, agente_basico, agente_aprende, agente_sin_sesgo, confianza_detector,
              maquina_estados, orquestador_sim, pausa_segura, calibracion_repeticiones, deriva_reloj,
              plan_caos, caos_sim, caos_agente_vs_sombra, tablero_salud,
              instantanea_estado, reanudar,
              modelos_hardware, senal_valida,
              ortesis_sin_ack, ortesis_serial_reconecta, reconexion_eeg, silencio_sin_recrear, dos_flujos_eeg, registro_huecos,
              reloj_contador,
              puente_reconecta,
              cerebro_sintetico, gemelo_unicorn):
        p()
    if a.completa:
        lazo_real_sintetico()
        lazo_real_caos()
    ok = sum(RESULTADOS)
    print(f'\n{ok}/{len(RESULTADOS)} pruebas pasaron')
    sys.exit(0 if ok == len(RESULTADOS) else 1)


if __name__ == '__main__':
    main()
