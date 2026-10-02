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
    assert v.actualizar(0.0, eeg=bien_eeg, ortesis=bien_ort, reloj_ms=0.0,
                        detector={'fiabilidad': 1.0, 'congelado': False}) == []
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
    v.actualizar(14.0, reloj_ms=5.0, detector={'fiabilidad': 0.2, 'congelado': True})
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
    return 'cerebro sintetico + ortesis simulada: ' + ' | '.join(c.split(']')[0][1:] + (' GO' if 'GO' in c and 'NO GO' not in c else ' NO GO') for c in cps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--completa', action='store_true')
    a = ap.parse_args()
    print('Pruebas ortesis-bci')
    for p in (contrato, vigilante, retroceso, agente_basico, agente_aprende, agente_sin_sesgo, confianza_detector,
              maquina_estados, orquestador_sim, modelos_hardware, cerebro_sintetico):
        p()
    if a.completa:
        lazo_real_sintetico()
    ok = sum(RESULTADOS)
    print(f'\n{ok}/{len(RESULTADOS)} pruebas pasaron')
    sys.exit(0 if ok == len(RESULTADOS) else 1)


if __name__ == '__main__':
    main()
