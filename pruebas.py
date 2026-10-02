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
    malo = x.copy(); malo[2] = 5.0; malo[7] *= 40; malo[0, 10] = 200_000.0
    assert hw.revisar_canales(malo, fs) == {'FC1': 'saturado', 'C3': 'plano', 'Fz': 'ruidoso'}
    assert hw.revisar_canales(np.empty((8, 0)), fs) == {}                           # buffer vacio
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
    X, y = cs.sesion_errp(120, semilla=3)
    t = np.arange(X.shape[2]) / cs.FS + config.EPOCA_ERRP[0]
    dif = X[y == 1, config.CANALES_EEG.index('Cz')].mean(0) - X[y == 0, config.CANALES_EEG.index('Cz')].mean(0)
    pe, ne = dif[np.argmin(abs(t - 0.36))], dif[np.argmin(abs(t - 0.25))]
    det = hw.DetectorErrP().ajustar(X, y)
    assert 0.65 < ba_mi < 1.0 and pe > 2 and ne < 0 and det.ba > 0.75, (ba_mi, pe, ne, det.ba)
    return f'MI BA {ba_mi:.2f}; ErrP Pe {pe:+.1f} uV, Ne {ne:+.1f} uV; detector BA {det.ba:.2f} (espec {det.espec:.2f})'


@prueba
def lazo_real_sintetico():
    puente = subprocess.Popen([sys.executable, 'cerebro_sintetico.py'], cwd=config.RAIZ,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(2)
        r = subprocess.run([sys.executable, 'orquestador.py', 'real', '--ortesis-sim', '--forzar',
                            '--ensayos_mi', '24', '--min_mi', '24', '--duracion_mi', '2.5',
                            '--espera', '0.3', '--ensayos_errp', '60', '--min_errp', '60',
                            '--seg_revision', '3',
                            '--pasos_estatico', '5', '--pasos_adaptativo', '10'],
                           cwd=config.RAIZ, capture_output=True, text=True, timeout=400)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--completa', action='store_true')
    a = ap.parse_args()
    print('Pruebas ortesis-bci')
    for p in (contrato, vigilante, retroceso, agente_basico, agente_aprende, agente_sin_sesgo, confianza_detector,
              maquina_estados, orquestador_sim, pausa_segura, calibracion_repeticiones,
              modelos_hardware, senal_valida,
              ortesis_sin_ack, ortesis_serial_reconecta, reconexion_eeg, puente_reconecta,
              cerebro_sintetico):
        p()
    if a.completa:
        lazo_real_sintetico()
    ok = sum(RESULTADOS)
    print(f'\n{ok}/{len(RESULTADOS)} pruebas pasaron')
    sys.exit(0 if ok == len(RESULTADOS) else 1)


if __name__ == '__main__':
    main()
