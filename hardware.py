"""Entrada/salida real: EEG por LSL, ortesis por USB, decoder de MI y detector de ErrP.

Todo lo que toca hardware vive aqui, para que el orquestador sea el mismo en
simulacion y en vivo.
"""
from __future__ import annotations

import pickle
import threading
import time
from collections import deque

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

import config
from salud import Retroceso


# ============================ Senal ============================
def filtrar(x, banda, fs, red=config.RED_HZ):
    """Pasa-banda + notch de la red, fase cero (para ventanas y epocas ya grabadas)."""
    sos = butter(4, banda, btype='bandpass', fs=fs, output='sos')
    b, a = iirnotch(red, 30.0, fs)
    return sosfiltfilt(sos, sosfiltfilt(tf2sos(b, a), x, axis=-1), axis=-1)


def revisar_canales(x, fs, u=None):
    """{electrodo: motivo} de los canales que no sirven: 'saturado', 'plano' o 'ruidoso'.

    x: canales x muestras en uV, crudo. Con menos de 1 s no opina."""
    u = u or config.SALUD
    malos = {}
    if x.ndim != 2 or x.shape[1] < fs:
        return malos
    finito = np.nan_to_num(x)
    rms = filtrar(finito, (1.0, 40.0), fs).std(axis=1)
    for i in range(x.shape[0]):
        nombre = config.CANALES_EEG[i] if i < len(config.CANALES_EEG) else f'ch{i}'
        if np.abs(finito[i]).max() > u['canal_saturado_uv']:
            malos[nombre] = 'saturado'
        elif finito[i, -int(0.5 * fs):].std() < u['canal_plano_uv']:   # plano el ultimo medio segundo
            malos[nombre] = 'plano'
        elif rms[i] > u['canal_ruidoso_uv']:
            malos[nombre] = 'ruidoso'
    return malos


def _sin_huecos(t, u=None):
    """Las marcas de tiempo avanzan sin saltos (ni hacia adelante ni hacia atras)."""
    u = u or config.SALUD
    return len(t) < 2 or float(np.abs(np.diff(t)).max()) <= u['hueco_max_s']


def ventana_valida(x, t, fs, edad_s, segundos, u=None):
    """La ventana de los ultimos `segundos` es fresca, completa, continua y finita."""
    u = u or config.SALUD
    n = int(segundos * fs)
    if x.ndim != 2 or x.shape[1] < n or len(t) < n or edad_s > u['eeg_edad_rojo_s']:
        return False
    return bool(np.isfinite(x[:, -n:]).all() and _sin_huecos(t[-n:], u))


VENTANA_RELOJ = 1500      # lecturas de retraso que se recuerdan (~30 s)


def deriva_reloj(retrasos, u=None):
    """Deriva (ms) del retraso del EEG: mediana de las ultimas lecturas contra la mediana
    de la ventana larga. Detecta un CAMBIO de desfase; cuando el desfase nuevo ya es la
    mayoria de la ventana pasa a ser lo normal, asi el reloj nunca queda en ROJO para
    siempre. Con pocas lecturas (arranque, reconexion) vale 0."""
    u = u or config.SALUD
    r = np.asarray(retrasos, dtype=float)[-VENTANA_RELOJ:]
    if r.size < u['reloj_lecturas_base']:
        return 0.0
    return float(np.median(r[-25:]) - np.median(r)) * 1000


def cortar_epoca(x, t, t0, fs, antes=-config.EPOCA_ERRP[0], despues=config.EPOCA_ERRP[1],
                 banda=config.BANDA_ERRP):
    """Epoca filtrada y con linea base alrededor de t0, o None si no es de fiar.

    La epoca se corta por TIEMPO: si el tramo [t0 - antes, t0 + despues] tiene un
    hueco, valores no finitos o le faltan muestras, no hay epoca. El filtro de fase
    cero se aplica solo al tramo continuo que la contiene, para que un corte
    cercano no contamine la epoca."""
    if x.ndim != 2 or x.shape[1] < fs:
        return None
    n = int(round((antes + despues) * fs))
    i0 = int(np.searchsorted(t, t0 - antes))
    if i0 + n > x.shape[1]:
        return None
    tramo = t[i0:i0 + n]
    if abs(tramo[0] - (t0 - antes)) > 2.0 / fs or not _sin_huecos(tramo):
        return None
    saltos = np.flatnonzero(np.abs(np.diff(t)) > config.SALUD['hueco_max_s']) + 1
    j0 = int(saltos[saltos <= i0].max()) if (saltos <= i0).any() else 0
    j1 = int(saltos[saltos >= i0 + n].min()) if (saltos >= i0 + n).any() else x.shape[1]
    seg = x[:, j0:j1]
    if not np.isfinite(seg).all():
        return None
    e = filtrar(seg, banda, fs)[:, i0 - j0:i0 - j0 + n]
    return e - e[:, :int(antes * fs)].mean(axis=1, keepdims=True)


class EntradaEEG:
    """Lee el flujo 'EEG' de LSL en un hilo y guarda los ultimos segundos.

    Las marcas de tiempo quedan en el reloj local de LSL (clocksync + dejitter),
    el mismo reloj con el que se estampa el ACK de la ortesis. Asi la epoca del
    ErrP se corta con la misma base de tiempo que el movimiento.

    Resiliencia: la entrada se crea con recover=False para ENTERARSE de que el
    flujo se perdio. Si se pierde o enmudece, se vuelve a resolver con retroceso
    exponencial. Un flujo que vuelve es una instancia nueva y su desfase de reloj
    puede ser otro: al reconectar se vacia el buffer (ninguna ventana ni epoca
    mezcla los dos lados del hueco) y se reinicia la linea base del reloj.

    Si hay VARIOS flujos con el mismo nombre en la red (un gemelo olvidado en otra
    terminal, por ejemplo) se avisa, se usa el mas reciente y nunca se salta a otro
    mientras el propio siga publicado, aunque enmudezca un rato. Si el propio muere,
    solo se acepta un flujo creado DESPUES que el: uno mas viejo ya fue descartado al
    arrancar y no es la fuente que volvio.

    Silencios sin perder el flujo (un tiron del dongle): el suavizado de marcas de LSL
    (dejitter) supone muestreo regular y, tras un hueco, deja las marcas desfasadas
    segundos durante ~10 s (medido: 2.7 s tras un hueco de 3 s). Por eso, en cuanto el
    flujo calla mas de `renovar_s` (config.SALUD['eeg_renovar_s'], 1 s por defecto), la
    entrada se RENUEVA: una entrada nueva al mismo flujo empieza con el suavizado limpio
    y el buffer vacio. Un hueco mas corto no renueva (el dongle del Cyton pierde paquetes
    en rafagas cortas y vaciar el buffer cada vez seria peor); de ese desfase menor se
    encarga el semaforo del reloj, que suspende el aprendizaje.
    """

    def __init__(self, segundos=30.0, timeout=10.0, nombre='EEG', renovar_s=None):
        self.nombre, self._segundos = nombre, segundos
        self._renovar_s = config.SALUD['eeg_renovar_s'] if renovar_s is None else renovar_s
        self._uid, self._creado, self.reconexiones = None, -np.inf, 0
        self._info, self.renovaciones, self._renovada = None, 0, False
        self._llegadas = deque(maxlen=2000)       # (instante de llegada, muestras): tasa real
        self.inlet = self._resolver(timeout)
        if self.inlet is None:
            raise RuntimeError(f"No encontre el flujo '{nombre}'. ¿Esta corriendo puente_lsl.py?")
        n = int(segundos * self.fs)
        self._x, self._t = deque(maxlen=n), deque(maxlen=n)
        self._lag = deque(maxlen=VENTANA_RELOJ)   # retraso (reloj local - ultima marca) por lectura
        self._t_llegada = time.monotonic()        # reloj de pared: no depende del desfase del flujo
        self._retroceso, self._proximo_intento = Retroceso(), 0.0
        self._lock = threading.Lock()
        self._vivo = True
        self._hilo = threading.Thread(target=self._leer, daemon=True)
        self._hilo.start()

    def _resolver(self, timeout=None):
        """Entrada nueva al flujo, o None si no aparece o si la instancia actual sigue
        publicada. timeout=None: una sola busqueda de 1 s (reconexion)."""
        from pylsl import StreamInlet, resolve_byprop, proc_clocksync, proc_dejitter
        if timeout is not None and not resolve_byprop('name', self.nombre, timeout=timeout):
            return None                           # al arrancar: esperar a que aparezca alguno
        s = resolve_byprop('name', self.nombre, minimum=16, timeout=1.0)   # 1 s completo: verlos TODOS
        if not s or self._uid in {x.uid() for x in s}:
            return None                           # el flujo propio sigue vivo: no se cambia por otro
        elegido = max(s, key=lambda x: x.created_at())
        if elegido.created_at() <= self._creado:
            return None                           # solo quedan flujos mas viejos que el propio: esperar
        if len(s) > 1:
            print(f"  AVISO: hay {len(s)} flujos '{self.nombre}' en la red ("
                  + ', '.join(f'{x.hostname()} {x.uid()[:8]}' for x in s)
                  + f'). Uso el mas reciente ({elegido.uid()[:8]}); cierra los demas.', flush=True)
        self._uid, self._creado, self._info = elegido.uid(), elegido.created_at(), elegido
        self.fs = float(elegido.nominal_srate())
        return self._abrir()

    def _abrir(self):
        from pylsl import StreamInlet, proc_clocksync, proc_dejitter
        return StreamInlet(self._info, max_buflen=int(self._segundos) + 5,
                           processing_flags=proc_clocksync | proc_dejitter, recover=False)

    def _renovar(self):
        """Entrada nueva al MISMO flujo: suavizado de marcas limpio y buffer vacio."""
        try:
            nueva = self._abrir()
        except Exception:
            return
        with self._lock:
            self.inlet = nueva
            self._x.clear()
            self._t.clear()
            self._reiniciar_reloj()
        self.renovaciones += 1
        print(f"  [eeg] silencio en '{self.nombre}': entrada renovada ({self.renovaciones})", flush=True)

    def _reiniciar_reloj(self):
        self._lag.clear()

    def _reconectar(self):
        """Un intento de volver a resolver el flujo; los intentos se espacian con retroceso."""
        if time.monotonic() < self._proximo_intento:
            return
        nueva = self._resolver()
        if nueva is None:
            self._proximo_intento = time.monotonic() + self._retroceso.siguiente()
            return
        with self._lock:
            self.inlet = nueva
            self._x.clear()
            self._t.clear()
            self._reiniciar_reloj()
        self.reconexiones += 1
        self._retroceso.reiniciar()
        self._proximo_intento = 0.0
        print(f"  [eeg] flujo '{self.nombre}' recuperado (reconexion {self.reconexiones})", flush=True)

    def _leer(self):
        from pylsl import local_clock
        rojo = config.SALUD['eeg_edad_rojo_s']
        while self._vivo:
            try:
                datos, ts = self.inlet.pull_chunk(timeout=0.05)
            except Exception:                     # flujo perdido: murio el puente o el gemelo
                self._reconectar()
                time.sleep(0.05)
                continue
            if ts:
                ahora = time.monotonic()
                self._renovada = False
                with self._lock:
                    self._x.extend(datos)
                    self._t.extend(ts)
                    self._lag.append(local_clock() - ts[-1])
                    self._llegadas.append((ahora, len(ts)))
                    self._t_llegada = ahora
            elif time.monotonic() - self._t_llegada > self._renovar_s and not self._renovada:
                self._renovar()                   # un hueco descuadra el suavizado: entrada limpia
                self._renovada = True
            elif time.monotonic() - self._t_llegada > rojo:
                self._reconectar()                # mudo: quiza es otra instancia la que publica ahora

    def edad(self):
        """Segundos desde que llego la ultima muestra."""
        return time.monotonic() - self._t_llegada

    def ultimo_t(self):
        with self._lock:
            return self._t[-1] if self._t else -np.inf

    def _crudo(self, segundos):
        """(x canales x muestras en uV, t) de los ultimos `segundos`, sin validar."""
        n = int(segundos * self.fs)
        with self._lock:
            x = np.array(list(self._x)[-n:], dtype=float).T
            t = np.array(list(self._t)[-n:])
        return x, t

    def ventana(self, segundos):
        """(x, t) de los ultimos `segundos`, o (None, None) si no es fresca, completa y continua."""
        x, t = self._crudo(segundos)
        if not ventana_valida(x, t, self.fs, self.edad(), segundos):
            return None, None
        return x, t

    def lecturas(self):
        """Lo que el Vigilante necesita: edad, tasa real, canales malos y deriva del reloj (ms)."""
        u = config.SALUD
        edad = self.edad()
        x, t = self._crudo(u['ventana_canales_s'])
        with self._lock:
            lag = np.array(self._lag)
            desde = time.monotonic() - u['ventana_canales_s']
            llegadas = sum(n for cuando, n in self._llegadas if cuando > desde)
        if t.size == 0:
            return {'edad_s': edad, 'tasa_hz': 0.0, 'canales': {}, 'reloj_ms': 0.0}
        fresco = edad <= u['eeg_edad_rojo_s']
        return {'edad_s': edad,
                'tasa_hz': llegadas / u['ventana_canales_s'],     # muestras que de verdad llegaron
                'canales': revisar_canales(x, self.fs) if fresco else {},
                'reloj_ms': deriva_reloj(lag)}

    def esperar_hasta(self, t_lsl, timeout=2.0):
        t_fin = time.time() + timeout
        while self.ultimo_t() < t_lsl and time.time() < t_fin:
            time.sleep(0.01)
        return self.ultimo_t() >= t_lsl

    def epoca(self, t0, antes=-config.EPOCA_ERRP[0], despues=config.EPOCA_ERRP[1],
              banda=config.BANDA_ERRP):
        """Epoca filtrada y con linea base alrededor de t0 (reloj LSL); None si faltan datos
        o si el tramo cruza un corte."""
        if not self.esperar_hasta(t0 + despues):
            return None
        x, t = self._crudo(4.0)
        return cortar_epoca(x, t, t0, self.fs, antes, despues, banda)

    def calidad(self, segundos=10.0):
        """Por canal: uV RMS (1-40 Hz), fraccion de potencia en 60 Hz y saturacion."""
        x, _ = self._crudo(segundos)
        if x.size == 0:
            return []
        xf = filtrar(x, (1.0, 40.0), self.fs)
        f = np.fft.rfftfreq(x.shape[1], 1 / self.fs)
        X = (x - x.mean(1, keepdims=True)) * np.hanning(x.shape[1])
        P = np.abs(np.fft.rfft(X, axis=1)) ** 2
        red = P[:, (f > config.RED_HZ - 2) & (f < config.RED_HZ + 2)].sum(1) / \
              (P[:, (f > 1) & (f < 100)].sum(1) + 1e-12)
        sat = (np.abs(x) > config.SALUD['canal_saturado_uv']).mean(1)
        filas = []
        for i, (r, z, s) in enumerate(zip(xf.std(1), red, sat)):
            nombre = config.CANALES_EEG[i] if i < len(config.CANALES_EEG) else f'ch{i}'
            filas.append({'canal': nombre, 'rms_uv': float(r), 'red': float(z),
                          'saturado': float(s), 'ok': bool(3 < r < 60 and z < 0.5 and s == 0)})
        return filas

    def cerrar(self):
        self._vivo = False


# ============================ Ortesis ============================
class _OrtesisBase:
    """Interfaz comun. mover() bloquea hasta el ACK y devuelve (seq, t_ack_lsl, latencia_ms).

    mover() NUNCA lanza: si el ACK no llega devuelve (seq, None, nan) y lo cuenta.
    seq no se reinicia nunca, ni al reabrir el puerto: el ESP32 solo devuelve el
    seq que recibio, asi que la PC es la unica duena de la numeracion."""

    def __init__(self):
        self.seq = 0
        self.latencias_ms = deque(maxlen=200)
        self.telemetria = {}
        self.acks_perdidos = 0            # consecutivos
        self.puerto_ok = True
        self.ultima_latencia = 0.0

    def jitter(self):
        lat = np.array(self.latencias_ms)
        if lat.size < 2:
            return float('nan'), float('nan')
        return float(lat.mean()), float(lat.std())

    def lecturas(self):
        """Lo que el Vigilante necesita de la ortesis."""
        return {'puerto_ok': self.puerto_ok, 'acks_perdidos': self.acks_perdidos,
                'latencia_ms': self.ultima_latencia}

    def _con_ack(self, seq, t_envio, t_ack):
        lat = (t_ack - t_envio) * 1000
        self.latencias_ms.append(lat)
        self.acks_perdidos, self.ultima_latencia = 0, lat
        return seq, t_ack, lat

    def _sin_ack(self, seq):
        self.acks_perdidos += 1
        return seq, None, float('nan')

    @staticmethod
    def _a_firmware(fraccion):
        return int(round(np.clip(fraccion, 0.0, 1.0) * 1000))


class OrtesisSerial(_OrtesisBase):
    """ESP32 por USB. Protocolo en config.py (M / A / T).

    Si el puerto se cae (ESP32 reiniciado, cable suelto) el hilo lector lo reabre
    con retroceso exponencial; mientras tanto mover() devuelve "sin ACK" de
    inmediato. `abrir` permite inyectar un puerto de mentira en las pruebas."""

    def __init__(self, puerto=config.PUERTO_ORTESIS, baudios=config.BAUDIOS, timeout_ack=0.3, abrir=None):
        super().__init__()
        from pylsl import local_clock
        self._clock = local_clock
        if abrir is None:
            import serial
            abrir = lambda: serial.Serial(puerto, baudios, timeout=0.01)
        self._abrir = abrir
        self.ser = abrir()
        self.timeout_ack = timeout_ack
        self._acks = {}
        self._cv = threading.Condition()
        self._vivo = True
        threading.Thread(target=self._leer, daemon=True).start()
        time.sleep(0.5)

    def _reabrir(self, retroceso):
        try:
            self.ser.close()
        except Exception:
            pass
        time.sleep(retroceso.siguiente())
        try:
            self.ser = self._abrir()
        except Exception:
            return
        self.puerto_ok = True
        retroceso.reiniciar()
        print('  [ortesis] puerto recuperado', flush=True)

    def _linea(self, linea, t):
        partes = linea.decode(errors='ignore').strip().split(',')
        if partes[0] == 'A' and len(partes) >= 2:
            with self._cv:
                self._acks[int(partes[1])] = t
                self._cv.notify_all()
        elif partes[0] == 'T' and len(partes) >= 4:
            self.telemetria = {'t_us': int(partes[1]), 'angulo': int(partes[2]),
                               'fsr': int(partes[3])}

    def _leer(self):
        buf, retroceso = b'', Retroceso()
        while self._vivo:
            if not self.puerto_ok:
                self._reabrir(retroceso)
                continue
            try:
                buf += self.ser.read(256)
            except Exception:                     # puerto caido
                self.puerto_ok, buf = False, b''
                continue
            while b'\n' in buf:
                linea, buf = buf.split(b'\n', 1)
                t = self._clock()
                try:
                    self._linea(linea, t)
                except ValueError:                # linea corrupta: se ignora
                    pass

    def mover(self, fraccion, dur_ms=config.DURACION_PASO_MS):
        self.seq += 1
        seq = self.seq
        if not self.puerto_ok:
            return self._sin_ack(seq)
        with self._cv:
            self._acks.clear()                    # ACK atrasados de pasos anteriores
        t_envio = self._clock()
        try:
            self.ser.write(f'M,{seq},{self._a_firmware(fraccion)},{dur_ms}\n'.encode())
        except Exception:
            self.puerto_ok = False
            return self._sin_ack(seq)
        with self._cv:
            self._cv.wait_for(lambda: seq in self._acks, timeout=self.timeout_ack)
            t_ack = self._acks.pop(seq, None)
        if t_ack is None:
            return self._sin_ack(seq)
        return self._con_ack(seq, t_envio, t_ack)

    def cerrar(self):
        self.mover(0.0)
        self._vivo = False
        try:
            self.ser.close()
        except Exception:
            pass


class OrtesisSimulada(_OrtesisBase):
    """Misma interfaz, sin ESP32. Latencia y jitter configurables para probar el lazo.

    Con `caos` (un PlanCaos) pierde ACK y mete picos de latencia de forma reproducible."""

    def __init__(self, latencia_ms=8.0, jitter_ms=1.5, semilla=0, caos=None, timeout_ack=0.3):
        super().__init__()
        from pylsl import local_clock
        self._clock = local_clock
        self.lat, self.jit = latencia_ms, jitter_ms
        self.rng = np.random.default_rng(semilla)
        self.caos, self.timeout_ack = caos, timeout_ack
        self.angulo = 0.0

    def mover(self, fraccion, dur_ms=config.DURACION_PASO_MS):
        self.seq += 1
        t_envio = self._clock()
        latencia = max(0.0, self.rng.normal(self.lat, self.jit))
        if self.caos is not None:
            if self.caos.por_paso('ack_perdido', self.seq):
                time.sleep(self.timeout_ack)
                return self._sin_ack(self.seq)
            latencia = self.caos.por_paso('pico_latencia', self.seq) or latencia
        time.sleep(latencia / 1000)
        t_ack = self._clock()
        self.angulo = float(np.clip(fraccion, 0, 1))
        self.telemetria = {'angulo': self._a_firmware(self.angulo), 'fsr': 0}
        return self._con_ack(self.seq, t_envio, t_ack)

    def cerrar(self):
        pass


# ============================ Modelos ============================
class DecoderIM:
    """Covarianzas (OAS) -> recentrado riemanniano -> espacio tangente -> regresion logistica.

    Recentrado: las covarianzas se "blanquean" con la media riemanniana de la
    sesion, asi la sesion siempre queda centrada en la identidad. En vivo esa
    media se sigue actualizando con cada ventana SIN etiquetas (paso geodesico),
    de modo que si cambia la impedancia de un electrodo o el piloto se relaja,
    el decoder se re-centra solo. Su logit lineal (w0 . phi + c0) es lo que
    corrige el agente.
    """

    def __init__(self, paso_recentrado=0.02):
        self.paso = paso_recentrado

    @staticmethod
    def _blanquear(C, M_isqrt):
        return M_isqrt @ C @ M_isqrt

    def ajustar(self, X, y):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from pyriemann.utils.base import invsqrtm
        from pyriemann.utils.mean import mean_riemann
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        self.cov = Covariances('oas')
        C = self.cov.fit_transform(X)
        self.M = mean_riemann(C)
        Mi = invsqrtm(self.M)
        Cw = np.array([self._blanquear(c, Mi) for c in C])
        self.ts = TangentSpace(metric='riemann').fit(Cw)
        Z = self.ts.transform(Cw)
        k = int(min(5, np.bincount(y).min()))
        self.pred_cv = cross_val_predict(LogisticRegression(max_iter=2000), Z, y,
                                         cv=StratifiedKFold(max(k, 2), shuffle=True, random_state=0))
        self.y_cal = y
        self.ba = exactitud_balanceada(y, self.pred_cv)
        clf = LogisticRegression(max_iter=2000).fit(Z, y)
        self.w0, self.c0 = clf.coef_[0].copy(), float(clf.intercept_[0])
        return self

    def phi(self, x, actualizar_centro=True):
        """Rasgos de una ventana (canales x muestras), o None si la ventana no es finita."""
        from pyriemann.utils.base import invsqrtm
        from pyriemann.utils.geodesic import geodesic_riemann
        if not np.isfinite(x).all():
            return None                      # ventana corrupta: ni rasgos ni recentrado
        C = self.cov.transform(x[None])[0]
        if actualizar_centro and self.paso > 0:
            self.M = geodesic_riemann(self.M, C, self.paso)
        return self.ts.transform(self._blanquear(C, invsqrtm(self.M))[None])[0]


def _medias_por_ventana(X, fs=250, inicio=0.15, fin=0.65, ancho=0.05, pre=-config.EPOCA_ERRP[0]):
    """Vista temporal del ErrP: promedio por canal en ventanas de 50 ms entre 150 y 650 ms."""
    cortes = [(int((pre + t) * fs), int((pre + t + ancho) * fs)) for t in np.arange(inicio, fin, ancho)]
    return np.concatenate([X[:, :, i:j].mean(axis=2) for i, j in cortes], axis=1)


class DetectorErrP:
    """Detector de ErrP de dos vistas, fusionadas y calibradas.

    - Vista temporal: medias por ventana de 50 ms + LDA con encogimiento
      automatico (Ledoit-Wolf). Robusta con pocos ensayos.
    - Vista geometrica: covarianzas aumentadas con los prototipos de "error" y
      "correcto" en el espacio tangente de Riemann + regresion logistica.
      Captura la relacion espacial entre electrodos.
    - Fusion suave de ambas y calibracion de Platt: la salida es una
      probabilidad real, que el agente usa completa.
    - Umbral de Neyman-Pearson: el que maximiza la exactitud balanceada con
      especificidad >= 0.90 (no 0.5, que con clases desbalanceadas hunde la
      sensibilidad). Solo afecta el "detecto / no detecto"; el agente usa la
      probabilidad completa.
    - Detector de rareza: distancia riemanniana a la media de calibracion.

    En el cerebro sintetico con 60 epocas: temporal 0.81, geometrica 0.81,
    fusion 0.85 de exactitud balanceada.
    """

    def __init__(self, umbral_amplitud_uv=100.0, z_rareza=3.5):
        self.umbral_amp, self.z_rareza = umbral_amplitud_uv, z_rareza
        self.umbral = 0.5

    def _pipe(self):
        from pyriemann.estimation import ERPCovariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer
        temporal = make_pipeline(FunctionTransformer(_medias_por_ventana),
                                 LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        geometrica = make_pipeline(ERPCovariances(estimator='oas'), TangentSpace(metric='riemann'),
                                   LogisticRegression(class_weight='balanced', max_iter=2000))
        fusion = VotingClassifier([('temporal', temporal), ('geometrica', geometrica)], voting='soft')
        return CalibratedClassifierCV(fusion, method='sigmoid', cv=3)

    def ajustar(self, X, y):
        from pyriemann.estimation import Covariances
        from pyriemann.utils.distance import distance_riemann
        from pyriemann.utils.mean import mean_riemann
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        k = int(min(4, np.bincount(y).min()))
        proba = cross_val_predict(self._pipe(), X, y, method='predict_proba',
                                  cv=StratifiedKFold(max(k, 2), shuffle=True, random_state=0))[:, 1]
        self.p_error_cal = float(y.mean())
        self.umbral = self._umbral_neyman_pearson(proba, y)
        self.pred_cv, self.y_cal = (proba > self.umbral).astype(int), y
        self.sens = float((self.pred_cv[y == 1] == 1).mean())
        self.espec = float((self.pred_cv[y == 0] == 0).mean())
        self.ba = 0.5 * (self.sens + self.espec)
        self.pipe = self._pipe().fit(X, y)
        self._cov = Covariances('oas')
        C = self._cov.fit_transform(X)
        self._M = mean_riemann(C)
        d = np.array([distance_riemann(c, self._M) for c in C])
        self._d_mu = float(np.median(d))
        self._d_sd = float(1.4826 * np.median(np.abs(d - np.median(d))) + 1e-9)
        return self

    def _umbral_neyman_pearson(self, proba, y, espec_min=config.ESPEC_MIN):
        """Umbral que maximiza la exactitud balanceada SUJETO a especificidad >= espec_min
        (las falsas alarmas son lo que mas dana al agente). Si ninguno la cumple, usa la
        tasa base de errores."""
        mejor, umbral = -1.0, float(y.mean())
        for u in np.unique(proba):
            pred = proba > u
            esp = (~pred[y == 0]).mean()
            if esp >= espec_min:
                ba = 0.5 * (pred[y == 1].mean() + esp)
                if ba > mejor:
                    mejor, umbral = ba, float(u)
        return umbral

    def artefacto(self, e):
        from pyriemann.utils.distance import distance_riemann
        if np.ptp(e, axis=1).max() > self.umbral_amp:
            return True
        d = distance_riemann(self._cov.transform(e[None])[0], self._M)
        return bool((d - self._d_mu) / self._d_sd > self.z_rareza)

    def p_error(self, e):
        return float(self.pipe.predict_proba(e[None])[0, 1])


def exactitud_balanceada(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return float(0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean()))


def intervalo_ba(y, pred, nivel=0.90, n_boot=1000, semilla=0):
    """Intervalo bootstrap de la exactitud balanceada (para calibracion secuencial)."""
    rng = np.random.default_rng(semilla)
    y, pred = np.asarray(y), np.asarray(pred)
    i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    bas = []
    for _ in range(n_boot):
        a = rng.choice(i1, len(i1)); b = rng.choice(i0, len(i0))
        bas.append(0.5 * ((pred[a] == 1).mean() + (pred[b] == 0).mean()))
    q = (1 - nivel) / 2
    return float(np.quantile(bas, q)), float(np.quantile(bas, 1 - q))


def guardar(obj, nombre):
    config.MODELOS.mkdir(exist_ok=True)
    with open(config.MODELOS / nombre, 'wb') as f:
        pickle.dump(obj, f)


def cargar(nombre):
    ruta = config.MODELOS / nombre
    if not ruta.exists():
        raise FileNotFoundError(f'No existe {ruta}: corre primero la calibracion.')
    with open(ruta, 'rb') as f:
        return pickle.load(f)
