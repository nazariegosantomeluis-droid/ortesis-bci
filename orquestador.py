"""Orquestador del lazo: maquina de estados, calibraciones, go/no go, lazo y registro.

Mismo codigo para simulacion y en vivo; solo cambia el backend.

Uso
  python orquestador.py sim --ciclo 0                  prueba rapida con piloto sintetico
  python orquestador.py sim                            a ritmo real (para ver el tablero)
  python orquestador.py sim --falla_detector           prueba el congelamiento
  python orquestador.py real --ortesis-sim --forzar    casco (o placa sintetica) sin ESP32
  python orquestador.py real --puerto COM4             todo real
  python orquestador.py real --puerto COM4 --saltar-calibracion   usa modelos guardados

Antes de 'real': puente_lsl.py corriendo y LabRecorder grabando.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime

import numpy as np
from pylsl import StreamOutlet, local_clock

import config
from agente_errp import AgenteErrP, ConfigAgente, ConfianzaDetector, sigmoide


def aviso(txt):
    print(txt, flush=True)


# ======================================================================
class MaquinaEstados:
    def __init__(self, salidas):
        self.salidas = salidas
        self.estado = config.ESTADOS[0]
        self.historial = [(local_clock(), self.estado)]
        salidas.marcador(config.m_bloque(self.estado))

    def ir_a(self, nuevo):
        if nuevo == self.estado:
            return
        if nuevo not in config.TRANSICIONES[self.estado]:
            raise ValueError(f'Transicion no permitida: {self.estado} -> {nuevo}')
        aviso(f'  [estado] {self.estado} -> {nuevo}')
        self.estado = nuevo
        self.historial.append((local_clock(), nuevo))
        self.salidas.marcador(config.m_bloque(nuevo))


class Salidas:
    """Flujos LSL que publica el orquestador: Marcadores, Paso y Estado (JSON)."""

    def __init__(self):
        self.marc = StreamOutlet(config.crear_info('Marcadores'))
        self.paso = StreamOutlet(config.crear_info('Paso'))
        self.est = StreamOutlet(config.crear_info('Estado'))

    def marcador(self, texto, t=None):
        self.marc.push_sample([texto], t if t is not None else local_clock())

    def publicar_paso(self, dec):
        self.paso.push_sample([dec.p_prima, float(dec.direccion), dec.delta])

    def estado(self, **datos):
        self.est.push_sample([json.dumps(datos, default=float)])


def checkpoint(salidas, n, ok, detalle, forzar, informativo=False):
    txt = f'[CP{n}] {detalle} -> {"GO" if ok else "NO GO"}'
    aviso(txt)
    salidas.estado(tipo='checkpoint', n=n, ok=bool(ok), texto=txt)
    if informativo:
        return ok
    if not ok and not forzar:
        return False
    if not ok:
        aviso(f'  (--forzar: se continua a pesar del NO GO en CP{n})')
    return True


# ======================================================================
class BackendSim:
    """Piloto sintetico del simulador."""

    def __init__(self, a):
        from simulador_lazo import PilotoSimulado, calibrar
        mk = lambda r: PilotoSimulado(sens=a.sens, espec=a.espec,
                                      semilla_sujeto=a.semilla, semilla_ruido=r)
        self.w0, self.c0 = calibrar(mk(10_000 + a.semilla))
        self.piloto = mk(a.semilla + 1)
        self.a, self.seq, self.t = a, 0, 0

    def preparar(self, orq):
        orq.fsm.ir_a('CAL_MI')
        orq.fsm.ir_a('CAL_ERRP')
        return {'w0': self.w0, 'c0': self.c0, 'sens': self.a.sens, 'espec': self.a.espec,
                'salida': 'binaria', 'p_error_cal': 0.3, 'umbral': 0.5}

    def cue(self, meta):
        pass

    def phi(self, meta):
        f = self.a.falla
        if f:
            self.piloto.detector_degradado(f[0] <= self.t < f[1])
        self.t += 1
        return self.piloto.rasgos(meta)

    def mover(self, fraccion):
        self.seq += 1
        return self.seq, local_clock(), 0.0

    def errp(self, seq, t_ack, erroneo, delta):
        return self.piloto.errp(erroneo, delta)

    def cerrar(self):
        pass


class BackendReal:
    """EEG por LSL + ortesis por USB (o simulada) + modelos de hardware.py."""

    def __init__(self, a):
        import hardware as hw
        self.hw, self.a = hw, a
        aviso('Conectando al flujo EEG...')
        self.eeg = hw.EntradaEEG()
        self.ortesis = hw.OrtesisSimulada() if a.ortesis_sim else hw.OrtesisSerial(a.puerto)
        self.angulo = 0.5

    # ---------------- checkpoint 1 ----------------
    def revisar(self, orq):
        import json
        filas_z, fuente = None, ''
        if config.IMPEDANCIAS_JSON.exists():
            dat = json.loads(config.IMPEDANCIAS_JSON.read_text())
            if time.time() - dat['t'] < 15 * 60:
                filas_z = dat['kohm']
                fuente = ' (simuladas)' if dat.get('simulada') else ''
        if filas_z:
            malos = [c for c, z in filas_z.items() if z > config.IMPEDANCIA_MAX_KOHM]
            ok_senal = not malos
            txt_senal = (f'impedancias{fuente} <= {config.IMPEDANCIA_MAX_KOHM:.0f} kOhm en '
                         f'{len(filas_z) - len(malos)}/{len(filas_z)}' + (f' (revisar {", ".join(malos)})' if malos else ''))
        else:
            aviso('Sin impedancias recientes (corre puente_lsl.py --impedancias). '
                  'Uso calidad de senal: quietos y con los ojos abiertos...')
            time.sleep(self.a.seg_revision)
            filas = self.eeg.calidad(self.a.seg_revision)
            for f in filas:
                aviso(f"  {f['canal']:>4}: {f['rms_uv']:6.1f} uV RMS | 60 Hz {f['red']:4.0%} | "
                      f"saturado {f['saturado']:4.0%} | {'bien' if f['ok'] else 'REVISAR'}")
            ok_senal = all(f['ok'] for f in filas)
            txt_senal = f'calidad de senal ok {sum(f["ok"] for f in filas)}/{len(filas)}'
        aviso('Midiendo latencia de la ortesis (20 pasos)...')
        for k in range(20):
            self.ortesis.mover(0.4 if k % 2 else 0.6)
            time.sleep(0.15)
        media, sd = self.ortesis.jitter()
        ok = ok_senal and sd <= config.LATENCIA_JITTER_MAX_MS
        return checkpoint(orq.salidas, 1, ok, f'{txt_senal}; latencia ACK {media:.1f} +- {sd:.1f} ms',
                          self.a.forzar)

    # ---------------- calibraciones secuenciales ----------------
    def _ventana_mi(self):
        x, _ = self.eeg.ventana(config.VENTANA_MI + 1.0)
        xf = self.hw.filtrar(x, config.BANDA_MI, self.eeg.fs)
        return xf[:, -int(config.VENTANA_MI * self.eeg.fs):]

    def _decidir_secuencial(self, y, pred, umbral, n, n_max, n_min, extra_ok=True):
        """GO si el limite inferior del IC 90% de la BA ya supera el umbral; NO GO si el
        superior ya quedo abajo; si no, seguir juntando ensayos."""
        lo, hi = self.hw.intervalo_ba(y, pred)
        aviso(f'    [{n} ensayos] BA {self.hw.exactitud_balanceada(y, pred):.2f}  IC90 [{lo:.2f}, {hi:.2f}]')
        if lo >= umbral and extra_ok:
            return 'go'
        # la exactitud sigue subiendo con mas ensayos (curva de aprendizaje): solo se
        # declara NO GO temprano con bastantes datos y el intervalo claramente abajo
        if hi < umbral - 0.05 and n >= 1.5 * n_min:
            return 'nogo'
        return 'go' if n >= n_max and lo >= umbral else ('fin' if n >= n_max else 'seguir')

    def calibrar_mi(self, orq):
        rng = np.random.default_rng()
        X, y = [], []
        for k in range(self.a.ensayos_mi):
            clase = int(rng.permutation([0, 1])[0]) if k % 2 == 0 else 1 - y[-1]   # pares balanceados
            aviso(f'[{k + 1}] preparate...')
            time.sleep(self.a.espera)
            orq.salidas.marcador(config.CUE_CERRAR if clase else config.CUE_RELAJA)
            orq.salidas.estado(tipo='cue', meta=1 if clase else -1)
            aviso('    >>> CERRAR: imagina que cierras la mano' if clase
                  else '    >>> RELAJA: imagina que abres y relajas la mano')
            time.sleep(self.a.duracion_mi)
            X.append(self._ventana_mi())
            y.append(clase)
            n = k + 1
            if n >= self.a.min_mi and n % 6 == 0 or n == self.a.ensayos_mi:
                self.decoder = self.hw.DecoderIM().ajustar(np.array(X), np.array(y))
                r = self._decidir_secuencial(np.array(y), self.decoder.pred_cv,
                                             config.MI_EXACTITUD_MIN, n, self.a.ensayos_mi, self.a.min_mi)
                if r != 'seguir':
                    break
        np.savez(config.RESULTADOS / f'calibracion_mi_{int(time.time())}.npz', X=np.array(X), y=np.array(y))
        self.hw.guardar(self.decoder, 'decoder_im.pkl')
        return checkpoint(orq.salidas, 2, self.decoder.ba >= config.MI_EXACTITUD_MIN,
                          f'MI: BA {self.decoder.ba:.2f} con {len(y)} ensayos (calibracion secuencial)',
                          self.a.forzar)

    def calibrar_errp(self, orq):
        rng = np.random.default_rng()

        def bloque():
            """20 ensayos: 10 cerrar y 10 abrir, con la misma tasa de error en cada direccion."""
            n_err = int(round(self.a.p_error * 10))
            b = [(obj, i < n_err) for obj in (0, 1) for i in range(10)]
            return [b[i] for i in rng.permutation(len(b))]

        plan, theta, X, y, n = [], 0.5, [], [], 0
        self.ortesis.mover(theta)
        while n < self.a.ensayos_errp:
            if not plan:
                plan = bloque()
            obj, err = plan.pop()
            n += 1
            aviso(f'[{n}] la ortesis debe {"CERRAR" if obj else "ABRIR"}: mirala')
            orq.salidas.estado(tipo='cue', meta=1 if obj else -1)
            orq.salidas.marcador(config.CUE_CERRAR if obj else config.CUE_RELAJA)
            time.sleep(self.a.espera)
            d = obj if not err else 1 - obj
            theta = float(np.clip(theta + (0.15 if d else -0.15), 0.1, 0.9))
            orq.salidas.paso.push_sample([float(d), 1.0 if d else -1.0, 0.15 if d else -0.15])
            seq, t_ack, _ = self.ortesis.mover(theta)
            orq.salidas.marcador(config.m_paso_ack(seq), t_ack)
            e = self.eeg.epoca(t_ack)
            if e is not None:
                X.append(e)
                y.append(int(err))
            if len(y) >= self.a.min_errp and n % 10 == 0 or n == self.a.ensayos_errp:
                self.detector = self.hw.DetectorErrP().ajustar(np.array(X), np.array(y))
                r = self._decidir_secuencial(np.array(y), self.detector.pred_cv, config.BA_MIN, n,
                                             self.a.ensayos_errp, self.a.min_errp,
                                             extra_ok=self.detector.espec >= config.ESPEC_MIN)
                if r != 'seguir':
                    break
        np.savez(config.RESULTADOS / f'calibracion_errp_{int(time.time())}.npz', X=np.array(X), y=np.array(y))
        self.hw.guardar(self.detector, 'detector_errp.pkl')
        d = self.detector
        ok = d.ba >= config.BA_MIN and d.espec >= config.ESPEC_MIN
        return checkpoint(orq.salidas, 3, ok,
                          f'ErrP: sens {d.sens:.2f}, espec {d.espec:.2f}, BA {d.ba:.2f} con {len(y)} epocas',
                          self.a.forzar)

    def preparar(self, orq):
        if not self.revisar(orq):
            return None
        if self.a.saltar_calibracion:
            self.decoder = self.hw.cargar('decoder_im.pkl')
            self.detector = self.hw.cargar('detector_errp.pkl')
            aviso(f'Modelos cargados: MI BA {self.decoder.ba:.2f}, '
                  f'ErrP sens {self.detector.sens:.2f} espec {self.detector.espec:.2f}')
        else:
            orq.fsm.ir_a('CAL_MI')
            if not self.calibrar_mi(orq):
                return None
            orq.fsm.ir_a('CAL_ERRP')
            if not self.calibrar_errp(orq):
                return None
        sens = float(np.clip(self.detector.sens, 0.51, 0.99))
        espec = float(np.clip(self.detector.espec, 0.51, 0.99))
        return {'w0': self.decoder.w0, 'c0': self.decoder.c0, 'sens': sens, 'espec': espec,
                'salida': 'calibrada', 'p_error_cal': self.detector.p_error_cal,
                'umbral': self.detector.umbral}

    # ---------------- lazo ----------------
    def cue(self, meta):
        aviso('    >>> CERRAR' if meta > 0 else '    >>> RELAJA')
        time.sleep(config.VENTANA_MI)            # que la ventana ya contenga imaginacion

    def phi(self, meta):
        return self.decoder.phi(self._ventana_mi())

    def mover(self, fraccion):
        return self.ortesis.mover(fraccion)

    def errp(self, seq, t_ack, erroneo, delta):
        e = self.eeg.epoca(t_ack)
        if e is None:
            return float('nan'), True
        return self.detector.p_error(e), self.detector.artefacto(e)

    def cerrar(self):
        self.ortesis.cerrar()
        self.eeg.cerrar()


# ======================================================================
class Orquestador:
    def __init__(self, backend, a):
        self.b, self.a = backend, a
        self.salidas = Salidas()
        time.sleep(0.5)                              # dar tiempo a que LabRecorder/tablero se conecten
        self.fsm = MaquinaEstados(self.salidas)
        self.angulo, self.filas, self.desplazamiento = 0.5, [], 0.0
        self.t_perturbacion = None
        config.RESULTADOS.mkdir(exist_ok=True)
        self.ruta_csv = config.RESULTADOS / datetime.now().strftime(f'sesion_{a.backend}_%Y%m%d_%H%M%S.csv')
        self.f_csv = open(self.ruta_csv, 'w', newline='')
        self.csv = csv.DictWriter(self.f_csv, fieldnames=config.COLUMNAS_CSV)
        self.csv.writeheader()

    def preparar(self):
        p = self.b.preparar(self)
        if p is None:
            return False
        cfg = ConfigAgente(modo=self.a.modo, sens=p['sens'], espec=p['espec'], eta_beta=self.a.eta,
                           salida_detector=p['salida'], p_error_calibracion=p['p_error_cal'],
                           usar_sesgo=not self.a.sin_sesgo)
        self.agente = AgenteErrP(p['w0'], p['c0'], cfg)
        self.umbral_errp = p['umbral']
        self.confianza = ConfianzaDetector(p['sens'], p['espec'])
        aviso(f"Agente '{self.a.modo}' listo (detector sens {p['sens']:.2f}, espec {p['espec']:.2f}, "
              f"salida {p['salida']}).")
        self.fsm.ir_a('LAZO_ESTATICO')
        return True

    # ---------------- un paso ----------------
    def paso(self, meta, aprender):
        t0 = time.perf_counter()
        dec = self.agente.decidir(self.b.phi(meta), self.desplazamiento)
        self.salidas.publicar_paso(dec)
        self.angulo = float(np.clip(self.angulo + dec.delta, 0, 1))
        seq, t_ack, lat = self.b.mover(self.angulo)
        self.salidas.marcador(config.m_paso_ack(seq), t_ack)   # estampado a la hora del ACK

        erroneo = dec.direccion != meta
        p_errp, art = self.b.errp(seq, t_ack, erroneo, dec.delta)
        detectado = bool(np.isfinite(p_errp) and p_errp > self.umbral_errp)
        fiab = self.confianza(erroneo, detectado, not art)
        sens_v, espec_v = self.confianza.vivo()
        info = self.agente.actualizar(p_errp, art, fiab if aprender else 0.0, sens_v, espec_v)

        if self.fsm.estado in ('LAZO_ADAPTATIVO', 'APRENDIZAJE_CONGELADO'):
            if self.confianza.congelado and self.fsm.estado == 'LAZO_ADAPTATIVO':
                self.fsm.ir_a('APRENDIZAJE_CONGELADO')
            elif not self.confianza.congelado and self.fsm.estado == 'APRENDIZAJE_CONGELADO':
                self.fsm.ir_a('LAZO_ADAPTATIVO')

        P_hat = info['P_hat']
        fila = {
            't_iso': datetime.now().isoformat(timespec='milliseconds'), 't_lsl': round(t_ack, 4),
            'seq': seq, 'estado': self.fsm.estado, 'meta': meta, 'angulo': round(self.angulo, 3),
            'p_prima': round(dec.p_prima, 4), 'direccion': dec.direccion, 'delta': round(dec.delta, 3),
            'P_hat': '' if not np.isfinite(P_hat) else round(P_hat, 4), 'artefacto': int(art),
            'fiabilidad': round(fiab, 3), 'beta': round(info['beta'], 4),
            'varianza_beta': round(info['varianza'], 4), 'sens_viva': round(sens_v, 3),
            'espec_viva': round(espec_v, 3), 'cambio': info['cambio'], 'explorando': int(dec.explorando),
            'error_verdadero': int(erroneo), 'error_sombra': int(dec.direccion_sombra != meta),
            'latencia_ack_ms': round(lat, 2),
        }
        self.csv.writerow(fila)
        self.f_csv.flush()
        self.filas.append(fila)
        self.salidas.estado(
            tipo='paso', paso=len(self.filas), estado=self.fsm.estado, meta=meta,
            angulo=self.angulo, p_crudo=float(sigmoide(dec.z)), b=self.agente.umbral_b,
            p_prima=dec.p_prima, P_hat=None if not np.isfinite(P_hat) else P_hat,
            error=int(erroneo), error_sombra=fila['error_sombra'], beta=info['beta'],
            sd_beta=float(np.sqrt(info['varianza'])), youden=self.confianza.youden,
            fiabilidad=fiab, congelado=self.confianza.congelado, cambio=info['cambio'], latencia_ms=lat,
            perturbado=self.desplazamiento != 0)

        espera = self.a.ciclo - (time.perf_counter() - t0)
        if espera > 0:
            time.sleep(espera)

    def bloque(self, n_pasos, aprender, perturbar_en=None):
        rng = np.random.default_rng(self.a.semilla + 7)
        orden = []
        for t in range(n_pasos):
            if t % config.PASOS_ENSAYO == 0:
                if not orden:
                    orden = list(rng.permutation([1, -1]))
                meta = int(orden.pop())
                self.salidas.marcador(config.CUE_CERRAR if meta > 0 else config.CUE_RELAJA)
                self.salidas.estado(tipo='cue', meta=meta)
                self.b.cue(meta)
            if perturbar_en is not None and t == perturbar_en:
                previo = self.fsm.estado
                self.fsm.ir_a('PERTURBACION')
                self.salidas.marcador(config.PERTURBACION_ON)
                self.desplazamiento = -config.PERTURBACION_LOGITS
                self.t_perturbacion = len(self.filas)
                self.beta_pre = self.agente.beta
                self.fsm.ir_a(previo)
            self.paso(meta, aprender)

    # ---------------- evaluacion ----------------
    def evaluar(self):
        if not self.filas:
            return
        e = np.array([f['error_verdadero'] for f in self.filas])
        s = np.array([f['error_sombra'] for f in self.filas])
        est = np.array([f['estado'] for f in self.filas])
        lat = np.array([f['latencia_ack_ms'] for f in self.filas])
        aviso('\n=== EVALUACION ===')
        for nombre in ('LAZO_ESTATICO', 'LAZO_ADAPTATIVO', 'APRENDIZAJE_CONGELADO'):
            m = est == nombre
            if m.any():
                aviso(f'  {nombre:22s} pasos={m.sum():4d}  error agente={e[m].mean():.2f}  '
                      f'error sombra={s[m].mean():.2f}')
        if self.a.backend == 'real':
            aviso(f'  latencia ACK: {lat.mean():.1f} +- {lat.std():.1f} ms')
        if self.t_perturbacion is not None:
            tp = self.t_perturbacion
            beta = np.array([f['beta'] for f in self.filas[tp:]])
            meta_beta = self.beta_pre + 0.7 * config.PERTURBACION_LOGITS
            idx = np.flatnonzero(beta >= meta_beta)
            if idx.size:
                n_rec = idx[0] + 1
                if self.a.backend == 'real':        # tiempo real medido con los ACK
                    seg = self.filas[tp + idx[0]]['t_lsl'] - self.filas[tp]['t_lsl']
                else:                               # simulacion: al ritmo nominal del lazo
                    seg = n_rec * config.CICLO_S
                k2 = tp + int(config.RECUPERACION_MAX_S / config.CICLO_S)
                checkpoint(self.salidas, 4, seg <= config.RECUPERACION_MAX_S,
                           f'recuperacion (beta al 70% de la perturbacion) en {n_rec} pasos '
                           f'= {seg:.0f} s; error ~2 min tras perturbar: agente {e[tp:k2].mean():.2f} '
                           f'vs sombra {s[tp:k2].mean():.2f}', False, informativo=True)
            else:
                checkpoint(self.salidas, 4, False, 'no se recupero dentro del bloque', False,
                           informativo=True)
        aviso(f'  beta final = {self.filas[-1]["beta"]}  |  cambios detectados = {self.agente.n_cambios}'
              f'  |  detector vivo: sens {self.confianza.sens:.2f}, espec {self.confianza.espec:.2f}')
        aviso(f'  CSV: {self.ruta_csv}')

    def cerrar(self):
        self.f_csv.close()
        self.b.cerrar()


# ======================================================================
def argumentos(argv=None):
    ap = argparse.ArgumentParser(description='Orquestador ortesis-bci')
    ap.add_argument('backend', choices=['sim', 'real'])
    ap.add_argument('--modo', choices=['bayes', 'fijo', 'estatico'], default='bayes')
    ap.add_argument('--eta', type=float, default=config.ETA_BETA, help="solo modo 'fijo'")
    ap.add_argument('--ciclo', type=float, default=None, help='s por paso (0 = sin esperas)')
    ap.add_argument('--pasos_estatico', type=int, default=None)
    ap.add_argument('--pasos_adaptativo', type=int, default=None)
    ap.add_argument('--sin_perturbacion', action='store_true')
    ap.add_argument('--semilla', type=int, default=0)
    ap.add_argument('--forzar', action='store_true', help='continua aunque un checkpoint de NO GO')
    ap.add_argument('--sin_sesgo', action='store_true',
                    help='apaga el detector de sesgo (si las metas no estan balanceadas)')
    # sim
    ap.add_argument('--sens', type=float, default=config.SENS)
    ap.add_argument('--espec', type=float, default=config.ESPEC)
    ap.add_argument('--falla_detector', action='store_true')
    # real
    ap.add_argument('--puerto', default=config.PUERTO_ORTESIS)
    ap.add_argument('--ortesis-sim', dest='ortesis_sim', action='store_true')
    ap.add_argument('--saltar-calibracion', dest='saltar_calibracion', action='store_true')
    ap.add_argument('--ensayos_mi', type=int, default=60, help='maximo; la calibracion para antes si ya decidio')
    ap.add_argument('--min_mi', type=int, default=24)
    ap.add_argument('--ensayos_errp', type=int, default=120, help='maximo')
    ap.add_argument('--min_errp', type=int, default=40)
    ap.add_argument('--duracion_mi', type=float, default=4.0)
    ap.add_argument('--espera', type=float, default=1.5)
    ap.add_argument('--p_error', type=float, default=0.3)
    ap.add_argument('--seg_revision', type=float, default=10.0)
    a = ap.parse_args(argv)
    sim = a.backend == 'sim'
    a.pasos_estatico = a.pasos_estatico or (60 if sim else 30)
    a.pasos_adaptativo = a.pasos_adaptativo or (300 if sim else 120)
    if a.ciclo is None:
        a.ciclo = config.CICLO_S if sim else 0.0   # en real el ciclo lo marcan cue + epoca
    a.falla = ((a.pasos_estatico + 150, a.pasos_estatico + 200) if a.falla_detector else None)
    return a


def main(argv=None):
    a = argumentos(argv)
    backend = BackendSim(a) if a.backend == 'sim' else BackendReal(a)
    orq = Orquestador(backend, a)
    try:
        if orq.preparar():
            aviso(f'Bloque LAZO_ESTATICO ({a.pasos_estatico} pasos)...')
            orq.bloque(a.pasos_estatico, aprender=False)
            orq.fsm.ir_a('LAZO_ADAPTATIVO')
            aviso(f'Bloque LAZO_ADAPTATIVO ({a.pasos_adaptativo} pasos)...')
            orq.bloque(a.pasos_adaptativo, aprender=True,
                       perturbar_en=None if a.sin_perturbacion else a.pasos_adaptativo // 3)
        else:
            aviso('Detenido por NO GO. Plan B: sesion grabada (puente_lsl.py --placa playback).')
    except KeyboardInterrupt:
        aviso('\nInterrumpido por el usuario.')
    finally:
        if 'EVALUACION' in config.TRANSICIONES[orq.fsm.estado]:
            orq.fsm.ir_a('EVALUACION')
        orq.evaluar()
        orq.cerrar()


if __name__ == '__main__':
    main()
