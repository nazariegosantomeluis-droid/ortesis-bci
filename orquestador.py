"""Orquestador del lazo: maquina de estados, calibraciones, go/no go, lazo y registro.

Mismo codigo para simulacion y en vivo; solo cambia el backend.

Uso
  python orquestador.py sim --ciclo 0                  prueba rapida con piloto sintetico
  python orquestador.py sim                            a ritmo real (para ver el tablero)
  python orquestador.py sim --falla_detector           prueba el congelamiento
  python orquestador.py sim --ciclo 0 --caos 1         caos estandar: cortes, ACK perdidos, canal despegado
  python orquestador.py real --ortesis-sim --forzar    casco (o placa sintetica) sin ESP32
  python orquestador.py real --puerto COM4             todo real
  python orquestador.py real --puerto COM4 --saltar-calibracion   usa modelos guardados

  python orquestador.py real --puerto COM4 --reanudar   continua la sesion tras un cierre inesperado

Antes de 'real': puente_lsl.py corriendo y LabRecorder grabando.

Persistencia: despues de cada paso se guarda una instantanea atomica de la sesion en
resultados/estado_sesion.json. --reanudar continua la misma sesion y el mismo CSV.

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
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from pylsl import StreamOutlet, local_clock

import config
from agente_errp import AgenteErrP, ConfigAgente, ConfianzaDetector, sigmoide
from caos import PlanCaos
from salud import Vigilante


def aviso(txt):
    print(txt, flush=True)


# ======================================================================
class MaquinaEstados:
    def __init__(self, salidas, estado=None):
        self.salidas = salidas
        self.estado = estado or config.ESTADOS[0]     # estado != None: sesion reanudada
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
_fallos_guardado = [0]


def guardar_instantanea(ruta, datos):
    """Escritura atomica: archivo temporal + os.replace, asi nunca queda un JSON a medias.
    Nunca lanza: si no se puede guardar se avisa y el lazo sigue (devuelve False)."""
    tmp = ruta.with_suffix('.tmp')
    try:
        with open(tmp, 'w') as f:
            json.dump(datos, f, default=float)
            f.flush()
            os.fsync(f.fileno())
        for intento in range(5):                 # en Windows otro proceso puede tener abierto el destino
            try:
                os.replace(tmp, ruta)
                return True
            except PermissionError:
                time.sleep(0.01 * (intento + 1))
    except OSError:
        pass
    _fallos_guardado[0] += 1
    if _fallos_guardado[0] % 20 == 1:            # avisa la primera vez y luego cada 20
        aviso(f'  AVISO: no se pudo guardar la instantanea de la sesion en {ruta}')
    return False


def cargar_instantanea():
    """La instantanea de la sesion a reanudar. Si no hay, sale con un mensaje claro."""
    try:
        inst = json.loads(config.ESTADO_SESION_JSON.read_text())
    except (OSError, ValueError):
        aviso(f'No hay sesion que reanudar: falta {config.ESTADO_SESION_JSON} (o esta danado).')
        sys.exit(2)
    if inst.get('terminada'):
        aviso('La sesion guardada ya termino: no hay nada que reanudar. Inicia una nueva sin --reanudar.')
        sys.exit(2)
    return inst


_ENTEROS = ('seq', 'meta', 'direccion', 'artefacto', 'explorando', 'error_verdadero', 'error_sombra')
_REALES = ('t_lsl', 'angulo', 'p_prima', 'delta', 'P_hat', 'fiabilidad', 'beta', 'varianza_beta',
           'sens_viva', 'espec_viva', 'latencia_ack_ms')


def recuperar_csv(ruta, n):
    """Deja el CSV con las primeras n filas (las que la instantanea conoce; si el proceso
    murio entre escribir una fila y guardar la instantanea, esa fila sobra y el paso se
    repite) y las devuelve con sus tipos, para que evaluar() las use."""
    with open(ruta, newline='') as f:
        crudas = list(csv.DictReader(f))[:n]
    if len(crudas) < n:
        aviso(f'  AVISO: el CSV tiene {len(crudas)} filas y la instantanea esperaba {n}; se continua.')
    with open(ruta, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=config.COLUMNAS_CSV)
        w.writeheader()
        w.writerows(crudas)
    filas = []
    for c in crudas:
        fila = dict(c)
        for k in _ENTEROS:
            fila[k] = int(c[k]) if c[k] != '' else ''
        for k in _REALES:
            fila[k] = float(c[k]) if c[k] != '' else ''
        filas.append(fila)
    return filas


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
#   instantanea() -> dict (JSON) ; restaurar(dict)           para --reanudar
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
        self.t_virtual, self.fallas = 0.0, {}
        self.caos = (PlanCaos(a.caos, config.CAOS[getattr(a, 'caos_nivel', 'estandar')])
                     if getattr(a, 'caos', None) is not None else None)
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

    def instantanea(self):
        return {'seq': self.seq, 't': self.t, 't_virtual': self.t_virtual,
                'acks_perdidos': self.acks_perdidos, 'ultima_latencia': self.ultima_latencia,
                'epocas_en_corte': list(self.epocas_en_corte),
                'rng_piloto': self.piloto.rng.bit_generator.state,
                'detector_piloto': [self.piloto.sens, self.piloto.espec]}

    def restaurar(self, d):
        self.seq, self.t, self.t_virtual = d['seq'], d['t'], d['t_virtual']
        self.acks_perdidos, self.ultima_latencia = d['acks_perdidos'], d['ultima_latencia']
        self.epocas_en_corte = list(d['epocas_en_corte'])
        self.piloto.rng.bit_generator.state = d['rng_piloto']
        self.piloto.sens, self.piloto.espec = d['detector_piloto']

    def cerrar(self):
        pass


class BackendReal:
    """EEG por LSL + ortesis por USB (o simulada) + modelos de hardware.py."""

    def __init__(self, a):
        import hardware as hw
        self.hw, self.a = hw, a
        aviso('Conectando al flujo EEG...')
        self.eeg = hw.EntradaEEG(fuente=getattr(a, 'fuente', 'puente'), nombre=getattr(a, 'eeg_nombre', None),
                                 tipo=getattr(a, 'eeg_tipo', None))
        # el caos solo afecta a la ortesis simulada y se activa con activar_caos()
        self.ortesis = hw.OrtesisSimulada() if a.ortesis_sim else hw.OrtesisSerial(a.puerto)
        self.activar_caos('calibracion')
        self.angulo = 0.5
        self.decoder = self.detector = None

    # ---------------- checkpoint 1 ----------------
    def revisar(self, orq):
        """CP1: el Unicorn no mide impedancias. Calidad de senal por canal (quietos, ojos
        abiertos) y latencia del ACK con metricas robustas (hardware.evaluar_latencias)."""
        aviso('Revisando la calidad de senal: quietos y con los ojos abiertos...')
        time.sleep(self.a.seg_revision)
        filas = self.eeg.calidad(self.a.seg_revision)
        for f in filas:
            aviso(f"  {f['canal']:>4}: {f['rms_uv']:6.1f} uV RMS | 60 Hz {f['red']:4.0%} | "
                  f"saturado {f['saturado']:4.0%} | {'bien' if f['ok'] else 'REVISAR'}")
        ok_senal = bool(filas) and all(f['ok'] for f in filas)
        malos = [f['canal'] for f in filas if not f['ok']]
        txt_senal = (f'calidad de senal ok {len(filas) - len(malos)}/{len(filas)}'
                     + (f' (revisar {", ".join(malos)})' if malos else ''))
        aviso(f'Midiendo latencia de la ortesis ({config.CP1_MOVIMIENTOS} movimientos)...')
        latencias = []
        for k in range(config.CP1_MOVIMIENTOS):
            _, t_ack, lat = self.ortesis.mover(0.4 if k % 2 else 0.6)
            latencias.append(lat if t_ack is not None else float('nan'))
            time.sleep(0.15)
        r = self.hw.evaluar_latencias(latencias)
        return checkpoint(orq.salidas, 1, ok_senal and r['ok'], f"{txt_senal}; {r['texto']}", self.a.forzar)

    def activar_caos(self, momento):
        """El caos de la ortesis simulada empieza en la calibracion o en el lazo (--caos-desde)."""
        if getattr(self.a, 'caos', None) is not None and momento == getattr(self.a, 'caos_desde', 'lazo') \
                and isinstance(self.ortesis, self.hw.OrtesisSimulada):
            self.ortesis.caos = PlanCaos(self.a.caos, config.CAOS[getattr(self.a, 'caos_nivel', 'estandar')])

    # ---------------- calibraciones secuenciales ----------------
    def _ventana_mi(self):
        """Ventana de MI filtrada, o None si el EEG no esta fresco y continuo."""
        u = config.SALUD                           # tolera perdidas de Bluetooth chicas
        x, _ = self.eeg.ventana(config.VENTANA_MI + 1.0, u['mi_perdida_max'], u['mi_hueco_max_s'])
        if x is None:
            return None
        xf = self.hw.filtrar(x, config.BANDA_MI, self.eeg.fs)
        return xf[:, -int(config.VENTANA_MI * self.eeg.fs):]

    def _ensayo_con_reintentos(self, tomar, que):
        """tomar() presenta el ensayo y devuelve su dato, o None si el EEG o la ortesis
        fallaron. Un ensayo afectado se repite como maximo config.CAL_REPETICIONES_MAX
        veces; despues se avisa y la calibracion sigue con el siguiente."""
        for k in range(config.CAL_REPETICIONES_MAX + 1):
            self.falla = 'sin dato'                # tomar() deja aqui el motivo si falla
            dato = tomar()
            if dato is not None:
                return dato
            if k < config.CAL_REPETICIONES_MAX:
                aviso(f'    {que} afectado por una falla ({self.falla}): se repite '
                      f'({k + 1}/{config.CAL_REPETICIONES_MAX})')
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

    def _canales_malos(self):
        """True (y deja el motivo en self.falla) si algun electrodo esta despegado ahora."""
        malos = self.eeg.lecturas()['canales']
        if malos:
            self.falla = '; '.join(f'{c} {m}' for c, m in malos.items())
        return bool(malos)

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
                if self._canales_malos():
                    return None
                self.falla = 'EEG sin ventana fresca y continua'
                return self._ventana_mi()
            v = self._ensayo_con_reintentos(tomar, 'ensayo de MI')
            if v is not None:
                X.append(v)
                y.append(clase)
            n = k + 1
            if (n >= self.a.min_mi and n % 6 == 0 or n == self.a.ensayos_mi) and self._ajustable(y):
                self.decoder = self.hw.DecoderIM().ajustar(np.array(X), np.array(y), config.candidatos('decoder'))
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
                          f'MI: BA {self.decoder.ba:.2f} con {len(y)} ensayos (calibracion secuencial; '
                          f'canales: {self.decoder.eleccion})',
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
                    self.falla = 'la ortesis no confirmo el movimiento'
                    return None                    # sin ACK no se sabe cuando empezo el movimiento
                orq.salidas.marcador(config.m_paso_ack(seq), t_ack)
                e = self.eeg.epoca(t_ack)
                if self._canales_malos():
                    return None
                self.falla = 'epoca incompleta o con un corte de EEG'
                return e
            e = self._ensayo_con_reintentos(tomar, 'ensayo de ErrP')
            if e is not None:
                X.append(e)
                y.append(int(err))
        # Todas las epocas pedidas, sin GO ni NO GO tempranos: con 40 a 60 epocas la parada
        # secuencial elegia estimados inflados por suerte (en el gemelo: 0.87 reportado contra
        # 0.69 real). El umbral se elige con validacion anidada (DetectorErrP.ajustar).
        if self._ajustable(y):
            self.detector = self.hw.DetectorErrP().ajustar(np.array(X), np.array(y), config.candidatos('detector'))
            aviso('    eleccion del detector (AUC de validacion cruzada): '
                  + ', '.join(f'{n} {v:.2f}' for n, v in self.detector.puntajes.items())
                  + f' -> {self.detector.eleccion}')
            lo, hi = self.hw.intervalo_ba(np.array(y), self.detector.pred_cv)
            aviso(f'    [{len(y)} epocas] BA {self.detector.ba:.2f}  IC90 [{lo:.2f}, {hi:.2f}]')
        if self.detector is None:
            aviso('No se pudo calibrar el detector de ErrP: no quedaron epocas validas.')
            return False
        np.savez(config.RESULTADOS / f'calibracion_errp_{int(time.time())}.npz', X=np.array(X), y=np.array(y))
        config.MODELOS.mkdir(exist_ok=True)
        np.savez(config.MODELOS / 'detector_errp_datos.npz', X=np.array(X), y=np.array(y))   # para co-adaptar
        self.hw.guardar(self.detector, 'detector_errp.pkl')
        d = self.detector
        ok = d.ba >= config.BA_MIN and d.espec >= config.ESPEC_MIN
        return checkpoint(orq.salidas, 3, ok,
                          f'ErrP: sens {d.sens:.2f}, espec {d.espec:.2f}, BA {d.ba:.2f} con {len(y)} epocas '
                          f'({d.eleccion})',
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
        self.preparar_coadaptacion(orq)
        sens = float(np.clip(self.detector.sens, 0.51, 0.99))
        espec = float(np.clip(self.detector.espec, 0.51, 0.99))
        return {'w0': self.decoder.w0, 'c0': self.decoder.c0, 'sens': sens, 'espec': espec,
                'salida': 'calibrada', 'p_error_cal': self.detector.p_error_cal,
                'umbral': self.detector.umbral}

    # ---------------- detector co-adaptativo ----------------
    def preparar_coadaptacion(self, orq):
        """Con los datos de calibracion, el detector sigue aprendiendo en el lazo."""
        ruta = config.MODELOS / 'detector_errp_datos.npz'
        self.coadapta = None
        if not ruta.exists():
            aviso('  (sin datos de calibracion del detector guardados: no se co-adapta)')
            return
        d = np.load(ruta)
        self.coadapta = self.hw.DetectorCoadaptativo(self.detector, d['X'], d['y'], config.COADAPTAR_CADA,
                                                     config.COADAPTAR_PRUEBA, al_cambiar=orq.detector_cambiado)

    # ---------------- persistencia ----------------
    def instantanea(self):
        return {'seq': self.ortesis.seq, 'M': np.asarray(self.decoder.M).tolist()}

    def restaurar(self, d):
        """Sesion reanudada: modelos guardados, el centro del recentrado donde iba y seq continuo."""
        self.decoder = self.hw.cargar('decoder_im.pkl')
        self.detector = self.hw.cargar('detector_errp.pkl')
        self.decoder.M = np.array(d['M'])
        self.ortesis.seq = d['seq']
        self.coadapta = None                      # se rearma en Orquestador.preparar

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
        time.sleep(config.VENTANA_MI + config.ESPERA_PRIMER_PASO_S)   # la ventana ya en estado estable

    def phi(self, meta):
        v = self._ventana_mi()
        return None if v is None else self.decoder.phi(v)

    def mover(self, fraccion):
        return self.ortesis.mover(fraccion)

    def errp(self, seq, t_ack, erroneo, delta):
        e = self.eeg.epoca(t_ack)
        if e is None:
            return float('nan'), True, 'epoca_invalida'
        p, art = self.detector.p_error(e), self.detector.artefacto(e)
        if getattr(self, 'coadapta', None) is not None:
            self.coadapta.observar(e, erroneo, p, art)   # puntuada antes de entrenar con ella
            self.detector = self.coadapta.actual
        return p, art, ''

    def cerrar(self):
        self.ortesis.cerrar()
        self.eeg.cerrar()
        if getattr(self, 'coadapta', None) is not None:
            self.coadapta.esperar()


# ======================================================================
class Orquestador:
    def __init__(self, backend, a, inst=None):
        """inst: instantanea de una sesion a reanudar (cargar_instantanea()), o None."""
        self.b, self.a, self.inst = backend, a, inst
        self.salidas = Salidas()
        time.sleep(0.5)                              # dar tiempo a que LabRecorder/tablero se conecten
        self.fsm = MaquinaEstados(self.salidas, inst['estado'] if inst else None)
        self.angulo, self.filas, self.desplazamiento = 0.5, [], 0.0
        self.t_perturbacion = self.beta_pre = None
        self.prog = None                             # progreso del bloque en curso (va en la instantanea)
        self.lista = False                           # True cuando ya hay agente (preparar() dio GO)
        self.vigilante = Vigilante()
        self.avisos_salud = []                       # lo que se dijo en consola sobre la salud
        self.excluidos, self.error_post = {}, None   # los llena evaluar()
        config.RESULTADOS.mkdir(exist_ok=True)
        if inst is not None:                         # mismo CSV, en modo anadir
            self.ruta_csv = Path(inst['ruta_csv'])
            self.filas = recuperar_csv(self.ruta_csv, inst['paso'])
            self.f_csv = open(self.ruta_csv, 'a', newline='')
            self.csv = csv.DictWriter(self.f_csv, fieldnames=config.COLUMNAS_CSV)
            return
        base = datetime.now().strftime(f'sesion_{a.backend}_%Y%m%d_%H%M%S')
        self.ruta_csv, k = config.RESULTADOS / f'{base}.csv', 1
        while self.ruta_csv.exists():                # dos sesiones en el mismo segundo no se pisan
            k += 1
            self.ruta_csv = config.RESULTADOS / f'{base}_{k}.csv'
        self.f_csv = open(self.ruta_csv, 'w', newline='')
        self.csv = csv.DictWriter(self.f_csv, fieldnames=config.COLUMNAS_CSV)
        self.csv.writeheader()

    def preparar(self):
        inst = self.inst
        if inst is not None:                         # sesion reanudada: nada de calibrar
            p = inst['preparacion']
            self.b.restaurar(inst['backend'])
        else:
            p = self.b.preparar(self)
            if p is None:
                return False
        self.preparacion = {k: (np.asarray(v, dtype=float).tolist() if k == 'w0' else v) for k, v in p.items()}
        cfg = ConfigAgente(modo=self.a.modo, sens=p['sens'], espec=p['espec'], eta_beta=self.a.eta,
                           salida_detector=p['salida'], p_error_calibracion=p['p_error_cal'],
                           usar_sesgo=not self.a.sin_sesgo)
        self.agente = AgenteErrP(p['w0'], p['c0'], cfg)
        self.umbral_errp = p['umbral']
        self.confianza = ConfianzaDetector(p['sens'], p['espec'])
        aviso(f"Agente '{self.a.modo}' listo (detector sens {p['sens']:.2f}, espec {p['espec']:.2f}, "
              f"salida {p['salida']}).")
        if hasattr(self.b, 'activar_caos'):
            self.b.activar_caos('lazo')
        self.lista = True
        if inst is None:
            self.fsm.ir_a('LAZO_ESTATICO')
            return True
        self.agente.desde_dict(inst['agente'])
        self.confianza.desde_dict(inst['confianza'])
        self.angulo, self.desplazamiento = inst['angulo'], inst['desplazamiento']
        self.t_perturbacion, self.beta_pre, self.prog = inst['t_perturbacion'], inst['beta_pre'], inst['prog']
        aviso(f"Sesion REANUDADA en el paso {inst['paso']} ({self.fsm.estado}), beta {self.agente.beta:+.3f}. "
              f"CSV: {self.ruta_csv}")
        return True

    def detector_cambiado(self, det):
        """El detector co-adaptativo cambio de modelo: umbral, agente y ConfianzaDetector se
        actualizan juntos para seguir siendo coherentes con el modelo vigente."""
        sens, espec = float(np.clip(det.sens, 0.51, 0.99)), float(np.clip(det.espec, 0.51, 0.99))
        self.umbral_errp = det.umbral
        self.agente.cfg.sens, self.agente.cfg.espec = sens, espec
        self.agente.cfg.p_error_calibracion = det.p_error_cal
        self.confianza.rebase(sens, espec)
        version = self.b.coadapta.version
        self.salidas.marcador(config.m_detector(version))
        aviso(f'  [detector] modelo v{version} (en sombra: sens {det.sens:.2f}, espec {det.espec:.2f})')

    # ---------------- persistencia ----------------
    def instantanea(self, terminada=False):
        """Todo lo necesario para continuar la sesion exactamente donde va."""
        return {'version': 1, 'terminada': terminada, 'args': dict(vars(self.a)),
                'paso': len(self.filas), 'ruta_csv': str(self.ruta_csv), 'estado': self.fsm.estado,
                'angulo': self.angulo, 'desplazamiento': self.desplazamiento,
                't_perturbacion': self.t_perturbacion, 'beta_pre': self.beta_pre, 'prog': self.prog,
                'agente': self.agente.a_dict(), 'confianza': self.confianza.a_dict(),
                'preparacion': self.preparacion, 'backend': self.b.instantanea()}

    def guardar(self, terminada=False):
        guardar_instantanea(config.ESTADO_SESION_JSON, self.instantanea(terminada))

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
        detalle = (self.vigilante.detalle['ortesis' if motivo == 'ortesis' else 'eeg']
                   or 'no hay ventana de EEG fresca y continua')
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
        # P_hat se calcula con la fiabilidad real del detector; el peso decide si se aprende
        info = self.agente.actualizar(p_errp, art or not valido, self.confianza.fiabilidad_bruta, sens_v, espec_v,
                                      peso=fiab if aprende else 0.0)

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

    def bloque(self, nombre, n_pasos, aprender, perturbar_en=None):
        """Un bloque de pasos. Su progreso vive en self.prog (y en la instantanea), asi una
        sesion reanudada lo retoma en el mismo paso, con la misma meta y el mismo orden."""
        p = self.prog
        if p is None or p['bloque'] != nombre:       # bloque nuevo (no reanudado)
            p = self.prog = {'bloque': nombre, 't': 0, 'meta': None, 'orden': [],
                             'rng': np.random.default_rng(self.a.semilla + 7).bit_generator.state}
        rng = np.random.default_rng()
        rng.bit_generator.state = p['rng']
        retomado = p['t'] % config.PASOS_ENSAYO != 0  # se reanuda a medio ensayo: repetir su cue
        for t in range(p['t'], n_pasos):
            if t % config.PASOS_ENSAYO == 0:
                if not p['orden']:
                    p['orden'] = [int(v) for v in rng.permutation([1, -1])]
                    p['rng'] = rng.bit_generator.state
                p['meta'] = p['orden'].pop()
                self.presentar(p['meta'])
            elif retomado:
                self.presentar(p['meta'])
            retomado, meta = False, p['meta']
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
                self.guardar()
                self.presentar(meta)                 # el ensayo se retoma con su cue
            p['t'] = t + 1
            self.guardar()                           # instantanea atomica despues de cada paso

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
        ic = lambda v: '[{:.2f}, {:.2f}]'.format(*self.hw_intervalo(v))
        aviso('  (intervalos del 90 %, remuestreando ensayos de 5 pasos)')
        for nombre in ('LAZO_ESTATICO', 'LAZO_ADAPTATIVO', 'APRENDIZAJE_CONGELADO'):
            m = est == nombre
            if m.any():
                aviso(f'  {nombre:22s} pasos={m.sum():4d}  error agente={e[m].mean():.2f} {ic(e[m])}  '
                      f'error sombra={s[m].mean():.2f} {ic(s[m])}')
        if self.a.backend == 'real':
            aviso(f'  latencia ACK: {lat.mean():.1f} +- {lat.std():.1f} ms')
        if self.t_perturbacion is not None:
            tp = sum(1 for f in self.filas[:self.t_perturbacion] if not f['excluido'])
            if tp < len(validas):
                k2 = tp + int(config.RECUPERACION_MAX_S / config.CICLO_S)
                self.error_post = {'agente': float(e[tp:k2].mean()), 'sombra': float(s[tp:k2].mean()),
                                   'ic_agente': self.hw_intervalo(e[tp:k2]), 'ic_sombra': self.hw_intervalo(s[tp:k2])}
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
                               f'{self.error_post["agente"]:.2f} {ic(e[tp:k2])} vs sombra '
                               f'{self.error_post["sombra"]:.2f} {ic(s[tp:k2])}',
                               False, informativo=True)
                else:
                    checkpoint(self.salidas, 4, False,
                               f'no se recupero dentro del bloque; error tras perturbar: agente '
                               f'{self.error_post["agente"]:.2f} vs sombra {self.error_post["sombra"]:.2f}',
                               False, informativo=True)
        co = getattr(self.b, 'coadapta', None)
        if co is not None:
            ba = co.ba_secuencial()
            aviso(f'  detector en vivo (cada epoca puntuada antes de entrenar con ella): BA '
                  + (f'{ba:.2f}' if ba is not None else 's/d') + f' en {len(co.historial)} epocas; '
                  f'{co.version - 1} cambios de modelo, {co.descartes} descartados')
        aviso(f'  beta final = {self.filas[-1]["beta"]}  |  cambios detectados = {self.agente.n_cambios}'
              f'  |  detector vivo: sens {self.confianza.sens:.2f}, espec {self.confianza.espec:.2f}')
        aviso(f'  CSV: {self.ruta_csv}')

    @staticmethod
    def hw_intervalo(errores):
        import hardware as hw                    # importa scipy/pyriemann: solo al evaluar
        return hw.intervalo_error(errores)

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
    ap.add_argument('--caos', type=int, default=None, metavar='SEMILLA',
                    help='inyecta el caos estandar (fallas reproducibles) en el simulador o en la ortesis simulada')
    ap.add_argument('--caos-desde', dest='caos_desde', choices=['lazo', 'calibracion'], default='lazo',
                    help='el caos empieza con el lazo (por defecto) o ya desde la calibracion')
    ap.add_argument('--caos-nivel', dest='caos_nivel', choices=sorted(config.CAOS), default='estandar',
                    help='estandar (una falla cada pocos segundos) o leve (una cada 2 a 3 minutos)')
    ap.add_argument('--reanudar', action='store_true',
                    help='continua la sesion guardada en resultados/estado_sesion.json (mismo CSV)')
    ap.add_argument('--sin_sesgo', action='store_true',
                    help='apaga el detector de sesgo (si las metas no estan balanceadas)')
    # sim
    ap.add_argument('--sens', type=float, default=config.SENS)
    ap.add_argument('--espec', type=float, default=config.ESPEC)
    ap.add_argument('--falla_detector', action='store_true')
    # real
    ap.add_argument('--puerto', default=config.PUERTO_ORTESIS)
    ap.add_argument('--ortesis-sim', dest='ortesis_sim', action='store_true')
    ap.add_argument('--fuente', choices=sorted(config.FUENTES_EEG), default='puente',
                    help='de donde viene el EEG: puente (puente_lsl.py o el gemelo) o unicornlsl (la app de g.tec)')
    ap.add_argument('--eeg-nombre', dest='eeg_nombre', default=None,
                    help='nombre del flujo LSL de EEG (en UnicornLSL, el que se escribio en la app o el numero de serie)')
    ap.add_argument('--eeg-tipo', dest='eeg_tipo', default=None,
                    help='tipo del flujo LSL de EEG, si se prefiere resolver por tipo')
    ap.add_argument('--saltar-calibracion', dest='saltar_calibracion', action='store_true')
    ap.add_argument('--ensayos_mi', type=int, default=60, help='maximo; la calibracion para antes si ya decidio')
    # minimo 36 (antes 24): en el gemelo, la BA reportada era optimista en +0.02; con 36, +0.00
    ap.add_argument('--min_mi', type=int, default=36)
    ap.add_argument('--ensayos_errp', type=int, default=120,
                    help='epocas de calibracion de ErrP; siempre se usan todas (sin parada temprana)')
    ap.add_argument('--duracion_mi', type=float, default=4.0)
    ap.add_argument('--espera', type=float, default=1.5)
    ap.add_argument('--p_error', type=float, default=0.3)
    ap.add_argument('--seg_revision', type=float, default=10.0)
    a = ap.parse_args(argv)
    sim = a.backend == 'sim'
    a.pasos_estatico = a.pasos_estatico or (60 if sim else 30)
    a.pasos_adaptativo = a.pasos_adaptativo or (300 if sim else 120)
    a.ciclo_cli = a.ciclo                      # lo que se escribio en la linea de comandos
    if a.ciclo is None:
        a.ciclo = config.CICLO_S if sim else 0.0   # en real el ciclo lo marcan cue + epoca
    a.falla = ((a.pasos_estatico + 150, a.pasos_estatico + 200) if a.falla_detector else None)
    return a


def correr(orq, a):
    """La sesion completa. Pase lo que pase (incluido Ctrl+C dentro de una pausa segura)
    termina en EVALUACION, con el resumen impreso y el CSV cerrado."""
    completa = False
    try:
        if orq.preparar():
            if orq.prog is None or orq.prog['bloque'] == 'estatico':
                aviso(f'Bloque LAZO_ESTATICO ({a.pasos_estatico} pasos)...')
                orq.bloque('estatico', a.pasos_estatico, aprender=False)
                orq.fsm.ir_a('LAZO_ADAPTATIVO')
            aviso(f'Bloque LAZO_ADAPTATIVO ({a.pasos_adaptativo} pasos)...')
            orq.bloque('adaptativo', a.pasos_adaptativo, aprender=True,
                       perturbar_en=None if a.sin_perturbacion else a.pasos_adaptativo // 3)
            completa = True
        else:
            aviso('Detenido por NO GO. Plan B: sesion grabada (puente_lsl.py --placa playback).')
    except KeyboardInterrupt:
        aviso('\nInterrumpido por el usuario.'
              + (' Para continuar esta sesion: el mismo comando con --reanudar.' if orq.lista else ''))
    finally:
        if 'EVALUACION' in config.TRANSICIONES[orq.fsm.estado]:
            orq.fsm.ir_a('EVALUACION')
        orq.evaluar()
        if completa:
            orq.guardar(terminada=True)              # una sesion completa ya no se reanuda
        orq.cerrar()


def argumentos_reanudados(inst, a):
    """Los argumentos de la sesion guardada. De la linea de comandos solo se toma lo que
    puede cambiar tras un cierre inesperado: --ciclo, --puerto y --ortesis-sim."""
    if inst['args']['backend'] != a.backend:
        aviso(f"La sesion guardada es '{inst['args']['backend']}', no '{a.backend}'.")
        sys.exit(2)
    r = argparse.Namespace(**inst['args'])
    r.reanudar, r.saltar_calibracion = True, True
    r.puerto, r.ortesis_sim = a.puerto, a.ortesis_sim or r.ortesis_sim
    if a.ciclo_cli is not None:
        r.ciclo = a.ciclo_cli
    return r


def main(argv=None):
    a = argumentos(argv)
    inst = None
    if a.reanudar:
        inst = cargar_instantanea()
        a = argumentos_reanudados(inst, a)
    backend = BackendSim(a) if a.backend == 'sim' else BackendReal(a)
    correr(Orquestador(backend, a, inst), a)


if __name__ == '__main__':
    main()
