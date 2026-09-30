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


# ============================ Senal ============================
def filtrar(x, banda, fs, red=config.RED_HZ):
    """Pasa-banda + notch de la red, fase cero (para ventanas y epocas ya grabadas)."""
    sos = butter(4, banda, btype='bandpass', fs=fs, output='sos')
    b, a = iirnotch(red, 30.0, fs)
    return sosfiltfilt(sos, sosfiltfilt(tf2sos(b, a), x, axis=-1), axis=-1)


class EntradaEEG:
    """Lee el flujo 'EEG' de LSL en un hilo y guarda los ultimos segundos.

    Las marcas de tiempo quedan en el reloj local de LSL (clocksync + dejitter),
    el mismo reloj con el que se estampa el ACK de la ortesis. Asi la epoca del
    ErrP se corta con la misma base de tiempo que el movimiento.
    """

    def __init__(self, segundos=30.0, timeout=10.0):
        from pylsl import StreamInlet, resolve_byprop, proc_clocksync, proc_dejitter
        s = resolve_byprop('name', 'EEG', timeout=timeout)
        if not s:
            raise RuntimeError("No encontre el flujo 'EEG'. ¿Esta corriendo puente_lsl.py?")
        self.inlet = StreamInlet(s[0], max_buflen=int(segundos) + 5,
                                 processing_flags=proc_clocksync | proc_dejitter)
        self.fs = float(s[0].nominal_srate())
        n = int(segundos * self.fs)
        self._x, self._t = deque(maxlen=n), deque(maxlen=n)
        self._lock = threading.Lock()
        self._vivo = True
        self._hilo = threading.Thread(target=self._leer, daemon=True)
        self._hilo.start()

    def _leer(self):
        while self._vivo:
            datos, ts = self.inlet.pull_chunk(timeout=0.05)
            if ts:
                with self._lock:
                    self._x.extend(datos)
                    self._t.extend(ts)

    def ultimo_t(self):
        with self._lock:
            return self._t[-1] if self._t else -np.inf

    def ventana(self, segundos):
        """(x canales x muestras en uV, t) de los ultimos `segundos`."""
        n = int(segundos * self.fs)
        with self._lock:
            x = np.array(list(self._x)[-n:], dtype=float).T
            t = np.array(list(self._t)[-n:])
        return x, t

    def esperar_hasta(self, t_lsl, timeout=2.0):
        t_fin = time.time() + timeout
        while self.ultimo_t() < t_lsl and time.time() < t_fin:
            time.sleep(0.01)
        return self.ultimo_t() >= t_lsl

    def epoca(self, t0, antes=-config.EPOCA_ERRP[0], despues=config.EPOCA_ERRP[1],
              banda=config.BANDA_ERRP):
        """Epoca filtrada y con linea base alrededor de t0 (reloj LSL); None si faltan datos."""
        if not self.esperar_hasta(t0 + despues):
            return None
        x, t = self.ventana(4.0)
        if x.size == 0 or x.shape[1] < self.fs:
            return None
        xf = filtrar(x, banda, self.fs)
        i0 = int(np.searchsorted(t, t0 - antes))
        n = int(round((antes + despues) * self.fs))
        if i0 < 0 or i0 + n > xf.shape[1]:
            return None
        e = xf[:, i0:i0 + n]
        return e - e[:, :int(antes * self.fs)].mean(axis=1, keepdims=True)

    def calidad(self, segundos=10.0):
        """Por canal: uV RMS (1-40 Hz), fraccion de potencia en 60 Hz y saturacion."""
        x, _ = self.ventana(segundos)
        if x.size == 0:
            return []
        xf = filtrar(x, (1.0, 40.0), self.fs)
        f = np.fft.rfftfreq(x.shape[1], 1 / self.fs)
        X = (x - x.mean(1, keepdims=True)) * np.hanning(x.shape[1])
        P = np.abs(np.fft.rfft(X, axis=1)) ** 2
        red = P[:, (f > config.RED_HZ - 2) & (f < config.RED_HZ + 2)].sum(1) / \
              (P[:, (f > 1) & (f < 100)].sum(1) + 1e-12)
        sat = (np.abs(x) > 180_000).mean(1)
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
    """Interfaz comun. mover() bloquea hasta el ACK y devuelve (seq, t_ack_lsl, latencia_ms)."""

    def __init__(self):
        self.seq = 0
        self.latencias_ms = deque(maxlen=200)
        self.telemetria = {}

    def jitter(self):
        lat = np.array(self.latencias_ms)
        if lat.size < 2:
            return float('nan'), float('nan')
        return float(lat.mean()), float(lat.std())

    @staticmethod
    def _a_firmware(fraccion):
        return int(round(np.clip(fraccion, 0.0, 1.0) * 1000))


class OrtesisSerial(_OrtesisBase):
    """ESP32 por USB. Protocolo en config.py (M / A / T)."""

    def __init__(self, puerto=config.PUERTO_ORTESIS, baudios=config.BAUDIOS, timeout_ack=0.3):
        super().__init__()
        import serial
        from pylsl import local_clock
        self._clock = local_clock
        self.ser = serial.Serial(puerto, baudios, timeout=0.01)
        self.timeout_ack = timeout_ack
        self._acks = {}
        self._cv = threading.Condition()
        self._vivo = True
        threading.Thread(target=self._leer, daemon=True).start()
        time.sleep(0.5)

    def _leer(self):
        buf = b''
        while self._vivo:
            buf += self.ser.read(256)
            while b'\n' in buf:
                linea, buf = buf.split(b'\n', 1)
                t = self._clock()
                partes = linea.decode(errors='ignore').strip().split(',')
                if partes[0] == 'A' and len(partes) >= 2:
                    with self._cv:
                        self._acks[int(partes[1])] = t
                        self._cv.notify_all()
                elif partes[0] == 'T' and len(partes) >= 4:
                    self.telemetria = {'t_us': int(partes[1]), 'angulo': int(partes[2]),
                                       'fsr': int(partes[3])}

    def mover(self, fraccion, dur_ms=config.DURACION_PASO_MS):
        self.seq += 1
        seq = self.seq
        t_envio = self._clock()
        self.ser.write(f'M,{seq},{self._a_firmware(fraccion)},{dur_ms}\n'.encode())
        with self._cv:
            self._cv.wait_for(lambda: seq in self._acks, timeout=self.timeout_ack)
            t_ack = self._acks.pop(seq, None)
        if t_ack is None:
            raise TimeoutError(f'La ortesis no respondio ACK al paso {seq}')
        lat = (t_ack - t_envio) * 1000
        self.latencias_ms.append(lat)
        return seq, t_ack, lat

    def cerrar(self):
        try:
            self.mover(0.0)
        except Exception:
            pass
        self._vivo = False
        self.ser.close()


class OrtesisSimulada(_OrtesisBase):
    """Misma interfaz, sin ESP32. Latencia y jitter configurables para probar el lazo."""

    def __init__(self, latencia_ms=8.0, jitter_ms=1.5, semilla=0):
        super().__init__()
        from pylsl import local_clock
        self._clock = local_clock
        self.lat, self.jit = latencia_ms, jitter_ms
        self.rng = np.random.default_rng(semilla)
        self.angulo = 0.0

    def mover(self, fraccion, dur_ms=config.DURACION_PASO_MS):
        self.seq += 1
        t_envio = self._clock()
        time.sleep(max(0.0, self.rng.normal(self.lat, self.jit)) / 1000)
        t_ack = self._clock()
        self.angulo = float(np.clip(fraccion, 0, 1))
        lat = (t_ack - t_envio) * 1000
        self.latencias_ms.append(lat)
        self.telemetria = {'angulo': self._a_firmware(self.angulo), 'fsr': 0}
        return self.seq, t_ack, lat

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
        from pyriemann.utils.base import invsqrtm
        from pyriemann.utils.geodesic import geodesic_riemann
        C = self.cov.transform(x[None])[0]
        if actualizar_centro and self.paso > 0:
            self.M = geodesic_riemann(self.M, C, self.paso)
        return self.ts.transform(self._blanquear(C, invsqrtm(self.M))[None])[0]


class DetectorErrP:
    """Prototipos de ERP en geometria de Riemann, con probabilidades calibradas.

    - Cada epoca se aumenta con los promedios (prototipos) de "error" y
      "correcto" y se describe con su covarianza aumentada: captura forma de
      onda Y relacion espacial en una sola matriz.
    - Espacio tangente + regresion logistica, calibrada (Platt) con validacion
      cruzada: la salida es una probabilidad de verdad, que el agente usa
      completa en su P_hat (salida_detector='calibrada').
    - Detector de rareza: distancia riemanniana de la epoca a la media de
      calibracion. Epocas fuera de distribucion (movimiento, electrodo suelto)
      se marcan como artefacto aunque no rebasen un umbral de amplitud.
    """

    def __init__(self, umbral_amplitud_uv=100.0, z_rareza=3.5):
        self.umbral_amp, self.z_rareza = umbral_amplitud_uv, z_rareza

    def _pipe(self):
        from pyriemann.estimation import ERPCovariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        clf = CalibratedClassifierCV(LogisticRegression(class_weight='balanced', max_iter=2000),
                                     method='sigmoid', cv=3)
        return make_pipeline(ERPCovariances(estimator='oas'), TangentSpace(metric='riemann'), clf)

    def ajustar(self, X, y):
        from pyriemann.estimation import Covariances
        from pyriemann.utils.distance import distance_riemann
        from pyriemann.utils.mean import mean_riemann
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        k = int(min(4, np.bincount(y).min()))
        proba = cross_val_predict(self._pipe(), X, y, method='predict_proba',
                                  cv=StratifiedKFold(max(k, 2), shuffle=True, random_state=0))[:, 1]
        self.pred_cv, self.y_cal = (proba > 0.5).astype(int), y
        self.sens = float((self.pred_cv[y == 1] == 1).mean())
        self.espec = float((self.pred_cv[y == 0] == 0).mean())
        self.ba = 0.5 * (self.sens + self.espec)
        self.p_error_cal = float(y.mean())
        self.pipe = self._pipe().fit(X, y)
        self._cov = Covariances('oas')
        C = self._cov.fit_transform(X)
        self._M = mean_riemann(C)
        d = np.array([distance_riemann(c, self._M) for c in C])
        self._d_mu, self._d_sd = float(np.median(d)), float(1.4826 * np.median(np.abs(d - np.median(d))) + 1e-9)
        return self

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
