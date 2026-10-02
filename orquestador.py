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

Resiliencia: antes de cada paso el Vigilante (salud.py) revisa EEG, ortesis, reloj y
detector. Escalera de degradacion:
  1. todo bien: lazo normal
  2. detector poco fiable o reloj dudoso: el lazo sigue, el agente no aprende
  3. EEG perdido o canal despegado: PAUSA_SEGURA, la ortesis se abre despacio
  4. ortesis perdida: PAUSA_SEGURA sin poder moverla; se registra, se avisa y se reconecta
De la pausa se sale sola tras 3 s continuos de salud en VERDE, al estado previo.
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
from salud import Vigilante


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
        self.marcadores = []                       # copia local, para evaluar y probar

    def marcador(self, texto, t=None):
        self.marcadores.append(texto)
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
# Interfaz de un backend (la usan Orquestador.paso, revisar_salud y pausa_segura):
#   preparar(orq) -> dict | None     calibra y devuelve lo que necesita el agente
#   cue(meta)                        presenta la meta al piloto
#   phi(meta) -> rasgos | None       None = no hay ventana de EEG valida (no se decide)
#   mover(fraccion) -> (seq, t_ack | None, latencia_ms)      nunca lanza
#   errp(seq, t_ack, erroneo, delta) -> (p_errp, artefacto, excluido)
#   lecturas() -> {'eeg', 'ortesis', 'reloj_ms'}             para el Vigilante
#   posicion_segura() -> igual que mover(), despacio y a config.POSICION_SEGURA
#   reloj() -> s ; esperar(dt) ; fin_paso() ; cerrar()
class BackendSim:
    """Piloto sintetico del simulador.

    Lleva un reloj virtual (un ciclo por paso, lo que dure cada espera de la pausa),
    para que las fallas simuladas y la pausa segura se prueben en segundos y de
    forma determinista. `fallas` programa fallas a mano:
      {'corte_eeg': [(t0, dur)], 'canal': [(t0, dur, electrodo, motivo)],
       'rafaga_parpadeos': [(t0, dur)], 'ack_perdido': {seq}, 'pico_latencia': {seq: ms}}
    """

    EPOCA_S = config.EPOCA_ERRP[1]

    def __init__(self, a):
        from simulador_lazo import PilotoSimulado, calibrar
        mk = lambda r: PilotoSimulado(sens=a.sens, espec=a.espec,
                                      semilla_sujeto=a.semilla, semilla_ruido=r)
        self.w0, self.c0 = calibrar(mk(10_000 + a.semilla))
        self.piloto = mk(a.semilla + 1)
        self.a, self.seq, self.t = a, 0, 0
        self.t_virtual, self.fallas, self.caos = 0.0, {}, None
        self.acks_perdidos, self.ultima_latencia = 0, 0.0
        self.epocas_en_corte = []                  # seq de las epocas que tocaron un corte de EEG

    def preparar(self, orq):
        orq.fsm.ir_a('CAL_MI')
        orq.fsm.ir_a('CAL_ERRP')
        return {'w0': self.w0, 'c0': self.c0, 'sens': self.a.sens, 'espec': self.a.espec,
                'salida': 'binaria', 'p_error_cal': 0.3, 'umbral': 0.5}

    # ---------------- reloj virtual y fallas ----------------
    def reloj(self):
        return self.t_virtual

    def esperar(self, dt):
        self.t_virtual += dt
        if self.a.ciclo > 0:
            time.sleep(dt)

    def fin_paso(self):
        self.t_virtual += config.CICLO_S

    def _activa(self, tipo, t=None):
        """La falla por tiempo de ese tipo que esta activa en t (por defecto ahora), o None."""
        t = self.t_virtual if t is None else t
        for f in self.fallas.get(tipo, []):
            if f[0] <= t < f[0] + f[1]:
                return f
        return self.caos.activo(tipo, t) if self.caos else None

    def _por_paso(self, tipo, seq):
        if tipo == 'ack_perdido':
            return seq in self.fallas.get(tipo, ()) or bool(self.caos and self.caos.por_paso(tipo, seq))
        return self.fallas.get(tipo, {}).get(seq) or (self.caos.por_paso(tipo, seq) if self.caos else None)

    def lecturas(self):
        corte, canal = self._activa('corte_eeg'), self._activa('canal')
        eeg = {'edad_s': self.t_virtual - corte[0] if corte else 0.02,
               'tasa_hz': float(config.FLUJOS['EEG'][2]),
               'canales': {canal[2]: canal[3]} if canal and not corte else {}}
        return {'eeg': eeg, 'reloj_ms': 0.0,
                'ortesis': {'puerto_ok': True, 'acks_perdidos': self.acks_perdidos,
                            'latencia_ms': self.ultima_latencia}}

    # ---------------- lazo ----------------
    def cue(self, meta):
        pass

    def phi(self, meta):
        if self._activa('corte_eeg') or self._activa('canal'):
            return None                            # sin consumir aleatorios del piloto
        f = self.a.falla
        if f:
            self.piloto.detector_degradado(f[0] <= self.t < f[1])
        self.t += 1
        return self.piloto.rasgos(meta)

    def mover(self, fraccion):
        self.seq += 1
        if self._por_paso('ack_perdido', self.seq):
            self.acks_perdidos += 1
            return self.seq, None, float('nan')
        self.acks_perdidos = 0
        self.ultima_latencia = float(self._por_paso('pico_latencia', self.seq) or 0.0)
        return self.seq, local_clock(), self.ultima_latencia

    def posicion_segura(self):
        return self.mover(config.POSICION_SEGURA)

    def errp(self, seq, t_ack, erroneo, delta):
        p, art = self.piloto.errp(erroneo, delta)  # siempre: consumo fijo de aleatorios
        momentos = [self.t_virtual + d for d in (0.0, self.EPOCA_S / 2, self.EPOCA_S)]
        if any(self._activa('corte_eeg', t) for t in momentos):
            self.epocas_en_corte.append(seq)
            return float('nan'), True, 'epoca_invalida'
        if any(self._activa('canal', t) or self._activa('rafaga_parpadeos', t) for t in momentos):
            return p, True, ''                     # la epoca existe, pero es un artefacto
        return p, art, ''

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
        self.decoder = self.detector = None

    # ---------------- checkpoint 1 ----------------
    def revisar(self, orq):
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
            ok_senal = bool(filas) and all(f['ok'] for f in filas)
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
        """Ventana de MI filtrada, o None si el EEG no esta fresco y continuo."""
        x, _ = self.eeg.ventana(config.VENTANA_MI + 1.0)
        if x is None:
            return None
        xf = self.hw.filtrar(x, config.BANDA_MI, self.eeg.fs)
        return xf[:, -int(config.VENTANA_MI * self.eeg.fs):]

    def _ensayo_con_reintentos(self, tomar, que):
        """tomar() presenta el ensayo y devuelve su dato, o None si el EEG o la ortesis
        fallaron. Un ensayo afectado se repite como maximo config.CAL_REPETICIONES_MAX
        veces; despues se avisa y la calibracion sigue con el siguiente."""
        for k in range(config.CAL_REPETICIONES_MAX + 1):
            dato = tomar()
            if dato is not None:
                return dato
            if k < config.CAL_REPETICIONES_MAX:
                aviso(f'    {que} afectado por una falla: se repite ({k + 1}/{config.CAL_REPETICIONES_MAX})')
        aviso(f'    AVISO: {que} descartado tras {config.CAL_REPETICIONES_MAX} repeticiones; '
              f'la calibracion sigue')
        return None

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

    @staticmethod
    def _ajustable(y):
        """Hay ensayos de las dos clases para ajustar con validacion cruzada."""
        return len(y) >= 4 and np.bincount(y, minlength=2).min() >= 2

    def calibrar_mi(self, orq):
        rng = np.random.default_rng()
        X, y = [], []
        for k in range(self.a.ensayos_mi):
            clase = 1 - y[-1] if (k % 2 and y) else int(rng.permutation([0, 1])[0])   # pares balanceados

            def tomar():
                aviso(f'[{k + 1}] preparate...')
                time.sleep(self.a.espera)
                orq.salidas.marcador(config.CUE_CERRAR if clase else config.CUE_RELAJA)
                orq.salidas.estado(tipo='cue', meta=1 if clase else -1)
                aviso('    >>> CERRAR: imagina que cierras la mano' if clase
                      else '    >>> RELAJA: imagina que abres y relajas la mano')
                time.sleep(self.a.duracion_mi)
                return self._ventana_mi()
            v = self._ensayo_con_reintentos(tomar, 'ensayo de MI')
            if v is not None:
                X.append(v)
                y.append(clase)
            n = k + 1
            if (n >= self.a.min_mi and n % 6 == 0 or n == self.a.ensayos_mi) and self._ajustable(y):
                self.decoder = self.hw.DecoderIM().ajustar(np.array(X), np.array(y))
                r = self._decidir_secuencial(np.array(y), self.decoder.pred_cv,
                                             config.MI_EXACTITUD_MIN, n, self.a.ensayos_mi, self.a.min_mi)
                if r != 'seguir':
                    break
        if self.decoder is None:
            aviso('No se pudo calibrar MI: no quedaron ensayos validos.')
            return False
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

        plan, X, y, n = [], [], [], 0
        theta = [0.5]
        self.ortesis.mover(theta[0])
        while n < self.a.ensayos_errp:
            if not plan:
                plan = bloque()
            obj, err = plan.pop()
            n += 1

            def tomar():
                aviso(f'[{n}] la ortesis debe {"CERRAR" if obj else "ABRIR"}: mirala')
                orq.salidas.estado(tipo='cue', meta=1 if obj else -1)
                orq.salidas.marcador(config.CUE_CERRAR if obj else config.CUE_RELAJA)
                time.sleep(self.a.espera)
                d = obj if not err else 1 - obj
                theta[0] = float(np.clip(theta[0] + (0.15 if d else -0.15), 0.1, 0.9))
                orq.salidas.paso.push_sample([float(d), 1.0 if d else -1.0, 0.15 if d else -0.15])
                seq, t_ack, _ = self.ortesis.mover(theta[0])
                if t_ack is None:
                    return None                    # sin ACK no se sabe cuando empezo el movimiento
                orq.salidas.marcador(config.m_paso_ack(seq), t_ack)
                return self.eeg.epoca(t_ack)
            e = self._ensayo_con_reintentos(tomar, 'ensayo de ErrP')
            if e is not None:
                X.append(e)
                y.append(int(err))
            if (len(y) >= self.a.min_errp and n % 10 == 0 or n == self.a.ensayos_errp) and self._ajustable(y):
                self.detector = self.hw.DetectorErrP().ajustar(np.array(X), np.array(y))
                r = self._decidir_secuencial(np.array(y), self.detector.pred_cv, config.BA_MIN, n,
                                             self.a.ensayos_errp, self.a.min_errp,
                                             extra_ok=self.detector.espec >= config.ESPEC_MIN)
                if r != 'seguir':
                    break
        if self.detector is None:
            aviso('No se pudo calibrar el detector de ErrP: no quedaron epocas validas.')
            return False
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

    # ---------------- salud ----------------
    def reloj(self):
        return local_clock()

    def esperar(self, dt):
        time.sleep(dt)

    def fin_paso(self):
        pass

    def lecturas(self):
        eeg = self.eeg.lecturas()
        return {'eeg': eeg, 'reloj_ms': eeg['reloj_ms'], 'ortesis': self.ortesis.lecturas()}

    def posicion_segura(self):
        return self.ortesis.mover(config.POSICION_SEGURA, config.PAUSA_DURACION_MS)

    # ---------------- lazo ----------------
    def cue(self, meta):
        aviso('    >>> CERRAR' if meta > 0 else '    >>> RELAJA')
        time.sleep(config.VENTANA_MI)            # que la ventana ya contenga imaginacion

    def phi(self, meta):
        v = self._ventana_mi()
        return None if v is None else self.decoder.phi(v)

    def mover(self, fraccion):
        return self.ortesis.mover(fraccion)

    def errp(self, seq, t_ack, erroneo, delta):
        e = self.eeg.epoca(t_ack)
        if e is None:
            return float('nan'), True, 'epoca_invalida'
        return self.detector.p_error(e), self.detector.artefacto(e), ''

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
        self.vigilante = Vigilante()
        self.avisos_salud = []                       # lo que se dijo en consola sobre la salud
        self.excluidos, self.error_post = {}, None   # los llena evaluar()
        config.RESULTADOS.mkdir(exist_ok=True)
        base = datetime.now().strftime(f'sesion_{a.backend}_%Y%m%d_%H%M%S')
        self.ruta_csv, k = config.RESULTADOS / f'{base}.csv', 1
        while self.ruta_csv.exists():                # dos sesiones en el mismo segundo no se pisan
            k += 1
            self.ruta_csv = config.RESULTADOS / f'{base}_{k}.csv'
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

    # ---------------- salud y pausa segura ----------------
    def revisar_salud(self):
        """Lee a los subsistemas, publica cada cambio de semaforo (marcador, consola y
        flujo Estado) y devuelve el motivo de pausa: None, 'eeg', 'canal' u 'ortesis'."""
        l = self.b.lecturas()
        cambios = self.vigilante.actualizar(
            self.b.reloj(), eeg=l['eeg'], ortesis=l['ortesis'], reloj_ms=l['reloj_ms'],
            detector={'fiabilidad': self.confianza.fiabilidad_bruta, 'congelado': self.confianza.congelado,
                      'epocas': self.confianza.n_validas})
        for sub, color in cambios:
            self.salidas.marcador(config.m_salud(sub, color))
            detalle = self.vigilante.detalle[sub]
            txt = f'  [salud] {sub}: {color}' + (f' ({detalle})' if detalle else '')
            aviso(txt)
            self.avisos_salud.append(txt)
        if cambios:
            self.publicar_salud()
        return self.vigilante.motivo_pausa()

    def publicar_salud(self, motivo=None):
        self.salidas.estado(tipo='salud', estado=self.fsm.estado, colores=self.vigilante.colores,
                            detalle=self.vigilante.detalle, motivo=motivo, escalon=self.vigilante.escalon())

    def pausa_segura(self, motivo):
        """Protege al piloto y espera a que la salud vuelva. El agente no aprende aqui."""
        previo = self.fsm.estado
        self.fsm.ir_a('PAUSA_SEGURA')
        detalle = self.vigilante.detalle['ortesis' if motivo == 'ortesis' else 'eeg']
        txt = f'  [PAUSA SEGURA] motivo: {motivo} ({detalle}). El agente no aprende; la sesion sigue viva.'
        aviso(txt)
        self.avisos_salud.append(txt)
        seq = ''
        if motivo != 'ortesis':                      # con la ortesis perdida no hay a quien mover
            seq, _, _ = self.b.posicion_segura()
            self.angulo = config.POSICION_SEGURA
        self.registrar_fila(seq=seq, excluido=f'pausa:{motivo}')
        self.publicar_salud(motivo)
        t_aviso = t_sondeo = self.b.reloj()
        while True:
            self.b.esperar(0.1)
            self.revisar_salud()
            t = self.b.reloj()
            if self.vigilante.colores['ortesis'] != config.VERDE and t - t_sondeo >= 1.0:
                self.b.posicion_segura()             # sondeo: confirma que la ortesis responde
                self.angulo, t_sondeo = config.POSICION_SEGURA, t
            if self.vigilante.listo_para_reanudar(t):
                break
            if t - t_aviso >= 5.0:
                sigue = self.vigilante.motivo_pausa()
                donde = self.vigilante.detalle['ortesis' if sigue == 'ortesis' else 'eeg']
                aviso(f'  [PAUSA SEGURA] sigue: {sigue} ({donde})' if sigue
                      else '  [PAUSA SEGURA] salud recuperada: esperando VERDE continuo')
                self.publicar_salud(sigue or motivo)
                t_aviso = t
        aviso(f'  [PAUSA SEGURA] {config.SALUD["verde_para_reanudar_s"]:.0f} s de salud en VERDE: '
              f'se reanuda {previo}')
        self.fsm.ir_a(previo)
        self.publicar_salud()

    # ---------------- registro ----------------
    def registrar_fila(self, **campos):
        """Escribe una fila del CSV con el esquema del contrato (lo que falte queda vacio)."""
        fila = {c: '' for c in config.COLUMNAS_CSV}
        fila.update(t_iso=datetime.now().isoformat(timespec='milliseconds'),
                    t_lsl=round(local_clock(), 4), estado=self.fsm.estado,
                    angulo=round(self.angulo, 3), beta=round(self.agente.beta, 4),
                    varianza_beta=round(self.agente.var, 4), salud=self.vigilante.codigo())
        fila.update(campos)
        self.csv.writerow(fila)
        self.f_csv.flush()
        self.filas.append(fila)
        return fila

    # ---------------- un paso ----------------
    def paso(self, meta, aprender):
        """Un paso del lazo. Devuelve False, sin decidir ni mover, si no hay EEG valido."""
        t0 = time.perf_counter()
        phi = self.b.phi(meta)
        if phi is None:
            return False
        dec = self.agente.decidir(phi, self.desplazamiento)
        self.salidas.publicar_paso(dec)
        self.angulo = float(np.clip(self.angulo + dec.delta, 0, 1))
        seq, t_ack, lat = self.b.mover(self.angulo)
        erroneo = dec.direccion != meta

        if t_ack is None:                            # sin ACK no hay instante del movimiento: sin epoca
            p_errp, art, excluido, t_ack = float('nan'), True, 'sin_ack', local_clock()
        else:
            self.salidas.marcador(config.m_paso_ack(seq), t_ack)   # estampado a la hora del ACK
            p_errp, art, excluido = self.b.errp(seq, t_ack, erroneo, dec.delta)
        valido = not excluido
        detectado = bool(np.isfinite(p_errp) and p_errp > self.umbral_errp)
        fiab = self.confianza(erroneo, detectado, valido and not art)
        sens_v, espec_v = self.confianza.vivo()
        # escalon 2: con el reloj en ROJO la epoca puede estar desalineada; no se aprende de ella
        aprende = aprender and valido and self.vigilante.colores['reloj'] != config.ROJO
        info = self.agente.actualizar(p_errp, art or not valido, fiab if aprende else 0.0, sens_v, espec_v)

        if self.fsm.estado in ('LAZO_ADAPTATIVO', 'APRENDIZAJE_CONGELADO'):
            if self.confianza.congelado and self.fsm.estado == 'LAZO_ADAPTATIVO':
                self.fsm.ir_a('APRENDIZAJE_CONGELADO')
            elif not self.confianza.congelado and self.fsm.estado == 'APRENDIZAJE_CONGELADO':
                self.fsm.ir_a('LAZO_ADAPTATIVO')

        P_hat = info['P_hat']
        fila = self.registrar_fila(
            t_lsl=round(t_ack, 4), seq=seq, meta=meta, p_prima=round(dec.p_prima, 4),
            direccion=dec.direccion, delta=round(dec.delta, 3),
            P_hat='' if not np.isfinite(P_hat) else round(P_hat, 4), artefacto=int(art),
            fiabilidad=round(fiab, 3), beta=round(info['beta'], 4),
            varianza_beta=round(info['varianza'], 4), sens_viva=round(sens_v, 3),
            espec_viva=round(espec_v, 3), cambio=info['cambio'], explorando=int(dec.explorando),
            error_verdadero=int(erroneo), error_sombra=int(dec.direccion_sombra != meta),
            latencia_ack_ms='' if not np.isfinite(lat) else round(lat, 2), excluido=excluido)
        self.salidas.estado(
            tipo='paso', paso=len(self.filas), estado=self.fsm.estado, meta=meta,
            angulo=self.angulo, p_crudo=float(sigmoide(dec.z)), b=self.agente.umbral_b,
            p_prima=dec.p_prima, P_hat=None if not np.isfinite(P_hat) else P_hat,
            error=int(erroneo), error_sombra=fila['error_sombra'], beta=info['beta'],
            sd_beta=float(np.sqrt(info['varianza'])), youden=self.confianza.youden,
            fiabilidad=fiab, congelado=self.confianza.congelado, cambio=info['cambio'],
            latencia_ms=None if not np.isfinite(lat) else lat,
            perturbado=self.desplazamiento != 0, salud=self.vigilante.colores, excluido=excluido)

        self.b.fin_paso()
        espera = self.a.ciclo - (time.perf_counter() - t0)
        if espera > 0:
            time.sleep(espera)
        return True

    def bloque(self, n_pasos, aprender, perturbar_en=None):
        rng = np.random.default_rng(self.a.semilla + 7)
        orden = []
        for t in range(n_pasos):
            if t % config.PASOS_ENSAYO == 0:
                if not orden:
                    orden = list(rng.permutation([1, -1]))
                meta = int(orden.pop())
                self.presentar(meta)
            if perturbar_en is not None and t == perturbar_en:
                previo = self.fsm.estado
                self.fsm.ir_a('PERTURBACION')
                self.salidas.marcador(config.PERTURBACION_ON)
                self.desplazamiento = -config.PERTURBACION_LOGITS
                self.t_perturbacion = len(self.filas)
                self.beta_pre = self.agente.beta
                self.fsm.ir_a(previo)
            while True:                              # las pausas no consumen pasos del bloque
                motivo = self.revisar_salud()
                if motivo is None:
                    if self.paso(meta, aprender):
                        break
                    motivo = self.revisar_salud() or 'eeg'   # el EEG fallo justo al decidir
                self.pausa_segura(motivo)
                self.presentar(meta)                 # el ensayo se retoma con su cue

    def presentar(self, meta):
        self.salidas.marcador(config.CUE_CERRAR if meta > 0 else config.CUE_RELAJA)
        self.salidas.estado(tipo='cue', meta=meta)
        self.b.cue(meta)

    # ---------------- evaluacion ----------------
    def evaluar(self):
        if not self.filas:
            return
        self.excluidos = {}
        for f in self.filas:
            if f['excluido']:
                self.excluidos[f['excluido']] = self.excluidos.get(f['excluido'], 0) + 1
        validas = [f for f in self.filas if not f['excluido']]   # el analisis ignora los excluidos
        aviso('\n=== EVALUACION ===')
        desglose = ', '.join(f'{m} {n}' for m, n in self.excluidos.items())
        aviso(f'  excluidos del analisis: {sum(self.excluidos.values())} de {len(self.filas)} filas'
              + (f' ({desglose})' if desglose else ''))
        if not validas:
            aviso(f'  CSV: {self.ruta_csv}')
            return
        e = np.array([f['error_verdadero'] for f in validas])
        s = np.array([f['error_sombra'] for f in validas])
        est = np.array([f['estado'] for f in validas])
        lat = np.array([f['latencia_ack_ms'] for f in validas], dtype=float)
        for nombre in ('LAZO_ESTATICO', 'LAZO_ADAPTATIVO', 'APRENDIZAJE_CONGELADO'):
            m = est == nombre
            if m.any():
                aviso(f'  {nombre:22s} pasos={m.sum():4d}  error agente={e[m].mean():.2f}  '
                      f'error sombra={s[m].mean():.2f}')
        if self.a.backend == 'real':
            aviso(f'  latencia ACK: {lat.mean():.1f} +- {lat.std():.1f} ms')
        if self.t_perturbacion is not None:
            tp = sum(1 for f in self.filas[:self.t_perturbacion] if not f['excluido'])
            if tp < len(validas):
                k2 = tp + int(config.RECUPERACION_MAX_S / config.CICLO_S)
                self.error_post = {'agente': float(e[tp:k2].mean()), 'sombra': float(s[tp:k2].mean())}
                beta = np.array([f['beta'] for f in validas[tp:]])
                meta_beta = self.beta_pre + 0.7 * config.PERTURBACION_LOGITS
                idx = np.flatnonzero(beta >= meta_beta)
                if idx.size:
                    n_rec = idx[0] + 1
                    if self.a.backend == 'real':        # tiempo real medido con los ACK (incluye pausas)
                        seg = validas[tp + idx[0]]['t_lsl'] - validas[tp]['t_lsl']
                    else:                               # simulacion: al ritmo nominal del lazo
                        seg = n_rec * config.CICLO_S
                    checkpoint(self.salidas, 4, seg <= config.RECUPERACION_MAX_S,
                               f'recuperacion (beta al 70% de la perturbacion) en {n_rec} pasos '
                               f'= {seg:.0f} s; error ~2 min tras perturbar: agente '
                               f'{self.error_post["agente"]:.2f} vs sombra {self.error_post["sombra"]:.2f}',
                               False, informativo=True)
                else:
                    checkpoint(self.salidas, 4, False,
                               f'no se recupero dentro del bloque; error tras perturbar: agente '
                               f'{self.error_post["agente"]:.2f} vs sombra {self.error_post["sombra"]:.2f}',
                               False, informativo=True)
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


def correr(orq, a):
    """La sesion completa. Pase lo que pase (incluido Ctrl+C dentro de una pausa segura)
    termina en EVALUACION, con el resumen impreso y el CSV cerrado."""
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


def main(argv=None):
    a = argumentos(argv)
    backend = BackendSim(a) if a.backend == 'sim' else BackendReal(a)
    correr(Orquestador(backend, a), a)


if __name__ == '__main__':
    main()
