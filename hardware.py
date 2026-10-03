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
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch

import config
from salud import RelojContador, Retroceso


# ============================ Senal ============================
def filtrar(x, banda, fs, red=config.RED_HZ):
    """Pasa-banda + notch de la red, fase cero (para ventanas y epocas ya grabadas)."""
    sos = butter(4, banda, btype='bandpass', fs=fs, output='sos')
    b, a = iirnotch(red, 30.0, fs)
    return sosfiltfilt(sos, sosfiltfilt(tf2sos(b, a), x, axis=-1), axis=-1)


def potencia_alfa(x, fs, canales=config.PAPELES['alfa'], banda=config.BANDA_ALFA):
    """Potencia alfa media (uV^2/Hz) de los canales occipitales (semaforo PILOTO), con Welch
    de 2 s. x: canales x muestras, crudo. None con menos de 4 s."""
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] < 4 * fs:
        return None
    f, P = welch(x[config.indices(canales)], fs=fs, nperseg=int(2 * fs))
    return float(P[:, (f >= banda[0]) & (f <= banda[1])].mean())


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


def _sin_huecos(t, hueco_max_s=None):
    """Las marcas de tiempo avanzan sin saltos (ni hacia adelante ni hacia atras)."""
    hueco = config.SALUD['hueco_max_s'] if hueco_max_s is None else hueco_max_s
    return len(t) < 2 or float(np.abs(np.diff(t)).max()) <= hueco


def rejilla(x, t, fs):
    """Lleva muestras con hora propia (con alguna perdida) a una rejilla uniforme de fs,
    por interpolacion lineal. Devuelve (x canales x n, t)."""
    n = int(round((t[-1] - t[0]) * fs)) + 1
    tu = t[0] + np.arange(n) / fs
    return np.vstack([np.interp(tu, t, fila) for fila in x]), tu


def ventana_valida(x, t, fs, edad_s, segundos, perdida_max=0.0, hueco_max_s=None, u=None):
    """La ventana de los ultimos `segundos` (por HORA, no por numero de muestras) es fresca,
    finita y esta completa: como mucho falta la fraccion `perdida_max` de sus muestras y
    ningun hueco pasa de `hueco_max_s`. Por defecto es estricta; la ventana de imaginacion
    motora tolera perdidas de Bluetooth chicas (config.SALUD mi_perdida_max, mi_hueco_max_s)."""
    u = u or config.SALUD
    hueco = u['hueco_max_s'] if hueco_max_s is None else hueco_max_s
    if x.ndim != 2 or len(t) < 2 or x.shape[1] != len(t) or edad_s > u['eeg_edad_rojo_s']:
        return False
    ini = t[-1] - segundos + 1.0 / fs
    dentro = t >= ini - 1e-9
    if t[0] > ini + hueco or t[dentro][0] > ini + hueco:   # el buffer no llega tan atras
        return False
    falta = 1.0 - dentro.sum() / (segundos * fs)
    return bool(falta <= perdida_max + 1.0 / (segundos * fs)
                and _sin_huecos(t[dentro], hueco) and np.isfinite(x[:, dentro]).all())


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

    La epoca se corta por TIEMPO. Se toma el tramo sin huecos (ninguno mayor que
    hueco_max_s) que contiene [t0 - antes, t0 + despues]; si no existe, o tiene valores
    no finitos, no hay epoca. Las perdidas chicas dentro del tramo se interpolan a una
    rejilla uniforme, asi la epoca queda alineada con t0 aunque falten muestras. El filtro
    de fase cero se aplica solo a ese tramo, para que un corte cercano no la contamine."""
    if x.ndim != 2 or x.shape[1] < fs:
        return None
    ini, fin = t0 - antes, t0 + despues
    saltos = np.flatnonzero(np.abs(np.diff(t)) > config.SALUD['hueco_max_s']) + 1
    bordes = np.r_[0, saltos, len(t)]
    for a, b in zip(bordes[:-1], bordes[1:]):
        if t[a] <= ini + 1.0 / fs and t[b - 1] >= fin - 1.0 / fs:
            break
    else:
        return None
    if not np.isfinite(x[:, a:b]).all():
        return None
    xu, tu = rejilla(x[:, a:b], t[a:b], fs)
    n = int(round((antes + despues) * fs))
    i0 = int(round((ini - tu[0]) * fs))
    if xu.shape[1] < fs or i0 < 0 or i0 + n > xu.shape[1]:
        return None
    e = filtrar(xu, banda, fs)[:, i0:i0 + n]
    return e - e[:, :int(antes * fs)].mean(axis=1, keepdims=True)


class EntradaEEG:
    """Lee el EEG (y la IMU) de LSL en un hilo y guarda los ultimos segundos.

    fuente (config.FUENTES_EEG) dice de donde viene y en que canal esta cada cosa:
      'puente'      flujo del contrato, de puente_lsl.py o del gemelo: 8 canales de EEG ya
                    estampados con la hora reconstruida por contador; la IMU va en el flujo
                    'IMU', que se lee aparte si existe.
      'unicornlsl'  la app UnicornLSL de g.tec: un flujo de 17 canales estampado a la
                    llegada. Aqui se separan EEG, IMU, bateria y validez, y la hora de cada
                    muestra se reconstruye con el contador (salud.RelojContador).
    El flujo se resuelve por `nombre` si lo hay y, si no, por `tipo` (los dos se pueden
    cambiar al construir: orquestador.py --eeg-nombre / --eeg-tipo).

    La hora de cada muestra queda en el reloj local de LSL, el mismo con el que se estampa
    el ACK de la ortesis: asi la epoca del ErrP se corta con la base de tiempo del
    movimiento. NO se usa el suavizado de marcas de LSL (dejitter): la hora ya viene bien
    de la fuente, y el suavizado desfasa las marcas tras cualquier hueco (medido: 2.7 s tras
    un hueco de 3 s). Una perdida de Bluetooth queda como un hueco en la hora.

    Resiliencia: la entrada se crea con recover=False para ENTERARSE de que el flujo se
    perdio. Si se pierde o enmudece, se vuelve a resolver con retroceso exponencial. Un
    flujo que vuelve es una instancia nueva: al reconectar se vacia el buffer (ninguna
    ventana ni epoca mezcla los dos lados del hueco) y se reinicia el reloj.

    Si hay VARIOS flujos iguales en la red (un gemelo olvidado en otra terminal) se avisa,
    se usa el mas reciente y nunca se salta a otro mientras el propio siga publicado. Si el
    propio muere, solo se acepta uno creado DESPUES que el.
    """

    def __init__(self, segundos=30.0, timeout=10.0, nombre=None, fuente='puente', tipo=None):
        self.fuente = dict(config.FUENTES_EEG[fuente], id=fuente)
        self.nombre = nombre or self.fuente['nombre']
        self.tipo = tipo or self.fuente['tipo']
        self._clave = ('name', self.nombre) if self.nombre else ('type', self.tipo)
        self.etiqueta = f'{self._clave[0]}={self._clave[1]}'
        self._segundos = segundos
        self._uid, self._creado, self.reconexiones = None, -np.inf, 0
        self._info, self._reloj, self._imu_inlet = None, None, None
        self.bateria, self.invalidas = None, 0   # solo si la fuente los trae (unicornlsl)
        self._llegadas = deque(maxlen=2000)       # (instante de llegada, muestras): tasa real
        self.inlet = self._resolver(timeout, estricto=True)
        if self.inlet is None:
            raise RuntimeError(f"No encontre el flujo de EEG ({self.etiqueta}). "
                               f"¿Esta corriendo puente_lsl.py, el gemelo o la app UnicornLSL?")
        n = int(segundos * self.fs)
        self._x, self._t = deque(maxlen=n), deque(maxlen=n)
        self._imu_x, self._imu_t = deque(maxlen=n), deque(maxlen=n)
        self._lag = deque(maxlen=VENTANA_RELOJ)   # retraso (reloj local - ultima marca) por lectura
        self._t_llegada = time.monotonic()        # reloj de pared: no depende del desfase del flujo
        self._retroceso, self._proximo_intento = Retroceso(), 0.0
        self._lock = threading.Lock()
        self._vivo = True
        self._hilo = threading.Thread(target=self._leer, daemon=True)
        self._hilo.start()

    # ---------------- conexion ----------------
    def _resolver(self, timeout=None, estricto=False):
        """Entrada nueva al flujo, o None si no aparece o si la instancia actual sigue
        publicada. timeout=None: una sola busqueda de 1 s (reconexion)."""
        from pylsl import resolve_byprop
        clave, valor = self._clave
        if timeout is not None and not resolve_byprop(clave, valor, timeout=timeout):
            return None                           # al arrancar: esperar a que aparezca alguno
        s = resolve_byprop(clave, valor, minimum=16, timeout=1.0)          # 1 s completo: verlos TODOS
        if not s or self._uid in {x.uid() for x in s}:
            return None                           # el flujo propio sigue vivo: no se cambia por otro
        elegido = max(s, key=lambda x: x.created_at())
        if elegido.created_at() <= self._creado:
            return None                           # solo quedan flujos mas viejos que el propio: esperar
        if elegido.channel_count() != self.fuente['canales']:
            if estricto:
                raise RuntimeError(
                    f"El flujo {self.etiqueta} tiene {elegido.channel_count()} canales y la fuente "
                    f"'{self.fuente['id']}' espera {self.fuente['canales']}. Revisa --fuente (en la app "
                    f"UnicornLSL, el flujo combinado y no el dividido).")
            return None
        if len(s) > 1:
            print(f"  AVISO: hay {len(s)} flujos {self.etiqueta} en la red ("
                  + ', '.join(f'{x.hostname()} {x.uid()[:8]}' for x in s)
                  + f'). Uso el mas reciente ({elegido.uid()[:8]}); cierra los demas.', flush=True)
        self._uid, self._creado, self._info = elegido.uid(), elegido.created_at(), elegido
        self.fs = float(elegido.nominal_srate())
        if self.fuente['contador'] is not None:   # hora por contador: reloj nuevo para la instancia nueva
            self._reloj = RelojContador(self.fs)
        self._imu_inlet = self._abrir_imu()
        return self._abrir(elegido)

    def _abrir(self, info):
        from pylsl import StreamInlet, proc_clocksync
        return StreamInlet(info, max_buflen=int(self._segundos) + 5,
                           processing_flags=proc_clocksync, recover=False)

    def _abrir_imu(self):
        """La IMU del puente va en su propio flujo; si no esta, se sigue sin ella."""
        from pylsl import resolve_byprop
        if self.fuente['imu'] is not None or self.fuente['id'] != 'puente' or self.nombre != self.fuente['nombre']:
            return None
        s = resolve_byprop('name', 'IMU', timeout=1.0)
        if not s:
            return None
        try:                                      # abrirla ya: la lectura posterior no espera
            imu = self._abrir(max(s, key=lambda x: x.created_at()))
            imu.open_stream(timeout=3.0)
            return imu
        except Exception:
            return None

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
            for cola in (self._x, self._t, self._imu_x, self._imu_t, self._lag):
                cola.clear()
        self.reconexiones += 1
        self._retroceso.reiniciar()
        self._proximo_intento = 0.0
        print(f'  [eeg] flujo {self.etiqueta} recuperado (reconexion {self.reconexiones})', flush=True)

    # ---------------- lectura ----------------
    def _leer(self):
        from pylsl import local_clock
        rojo, f = config.SALUD['eeg_edad_rojo_s'], self.fuente
        while self._vivo:
            try:
                datos, ts = self.inlet.pull_chunk(timeout=0.05)
            except Exception:                     # flujo perdido: murio el puente o el gemelo
                self._reconectar()
                time.sleep(0.05)
                continue
            if ts:
                ahora = time.monotonic()
                datos = np.asarray(datos, dtype=float)
                horas = np.asarray(ts)
                if self._reloj is not None:       # la marca de LSL es la llegada: la hora sale del contador
                    horas = self._reloj.estampar(datos[:, f['contador']], ts[-1])
                with self._lock:
                    self._x.extend(datos[:, f['eeg']].tolist())
                    self._t.extend(horas.tolist())
                    if f['imu'] is not None:
                        self._imu_x.extend(datos[:, f['imu']].tolist())
                        self._imu_t.extend(horas.tolist())
                    if f['bateria'] is not None:
                        self.bateria = float(datos[-1, f['bateria']])
                    if f['validez'] is not None:
                        self.invalidas += int((datos[:, f['validez']] != 1).sum())
                    self._lag.append(local_clock() - horas[-1])
                    self._llegadas.append((ahora, len(ts)))
                    self._t_llegada = ahora
            elif time.monotonic() - self._t_llegada > rojo:
                self._reconectar()                # mudo: quiza es otra instancia la que publica ahora
            if self._imu_inlet is not None:
                try:
                    imu, ti = self._imu_inlet.pull_chunk(timeout=0.0)
                except Exception:
                    self._imu_inlet, imu, ti = None, [], []
                if ti:
                    with self._lock:
                        self._imu_x.extend(imu)
                        self._imu_t.extend(ti)

    def edad(self):
        """Segundos desde que llego la ultima muestra."""
        return time.monotonic() - self._t_llegada

    def ultimo_t(self):
        with self._lock:
            return self._t[-1] if self._t else -np.inf

    def _crudo(self, segundos):
        """(x canales x muestras en uV, t) de las ultimas segundos * fs muestras, sin validar."""
        n = int(segundos * self.fs)
        with self._lock:
            x = np.array(list(self._x)[-n:], dtype=float).T
            t = np.array(list(self._t)[-n:])
        return x, t

    def ventana(self, segundos, perdida_max=0.0, hueco_max_s=None):
        """(x, t) de los ultimos `segundos` en una rejilla uniforme, o (None, None) si la
        ventana no es fresca y completa. Estricta por defecto; ver ventana_valida()."""
        x, t = self._crudo(segundos + 0.5)
        if not ventana_valida(x, t, self.fs, self.edad(), segundos, perdida_max, hueco_max_s):
            return None, None
        dentro = t >= t[-1] - segundos + 0.5 / self.fs
        xu, tu = rejilla(x[:, dentro], t[dentro], self.fs)
        n = int(segundos * self.fs)
        if xu.shape[1] < n:                       # la primera muestra de la ventana se perdio: completar
            xu = np.hstack([np.repeat(xu[:, :1], n - xu.shape[1], axis=1), xu])
            tu = np.r_[tu[0] - np.arange(n - len(tu), 0, -1) / self.fs, tu]
        return xu[:, -n:], tu[-n:]

    def movimiento(self, t0, t1):
        """Velocidad angular maxima de la cabeza (grados/s, giroscopio) entre t0 y t1, o
        None si no hay IMU en ese tramo."""
        with self._lock:
            t = np.array(self._imu_t)
            if t.size == 0:
                return None
            sel = (t >= t0) & (t <= t1)
            if not sel.any():
                return None
            giro = np.array(self._imu_x, dtype=float)[sel, 3:6]
        return float(np.abs(giro).max())

    def lecturas(self):
        """Lo que el Vigilante necesita: edad, tasa real, canales malos, deriva del reloj (ms) y
        alfa occipital (semaforo PILOTO)."""
        u = config.SALUD
        edad = self.edad()
        xa, t = self._crudo(max(u['ventana_canales_s'], u['piloto_ventana_s']))
        x = xa[:, -int(u['ventana_canales_s'] * self.fs):] if xa.size else xa
        with self._lock:
            lag = np.array(self._lag)
            desde = time.monotonic() - u['ventana_canales_s']
            llegadas = sum(n for cuando, n in self._llegadas if cuando > desde)
        if t.size == 0:
            return {'edad_s': edad, 'tasa_hz': 0.0, 'canales': {}, 'reloj_ms': 0.0, 'alfa': None}
        fresco = edad <= u['eeg_edad_rojo_s']
        return {'edad_s': edad,
                'tasa_hz': llegadas / u['ventana_canales_s'],     # muestras que de verdad llegaron
                'canales': revisar_canales(x, self.fs) if fresco else {},
                'reloj_ms': deriva_reloj(lag),
                'alfa': potencia_alfa(xa, self.fs) if fresco else None}

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
def latencia_mecanica_simulada(seq, semilla=0):
    """Segundos entre el ACK y el inicio real del movimiento en la ortesis simulada. Es una
    funcion de (semilla, seq) para que el gemelo ancle el ErrP al mismo instante."""
    lo, hi = config.LATENCIA_MECANICA_SIM_MS
    return float(np.random.default_rng([semilla, 7, int(seq)]).uniform(lo, hi)) / 1000.0


def inicio_por_telemetria(tele, t_us_ack, umbral=config.UMBRAL_INICIO_ANGULO):
    """t_us del inicio real del movimiento: el primer instante tras el ACK en que el angulo se
    aleja del previo mas que `umbral`, interpolando entre muestras. tele: (n, 2) de (t_us,
    angulo) ordenado. None si no se ve el inicio."""
    tele = np.asarray(tele, dtype=float).reshape(-1, 2)
    antes = tele[tele[:, 0] <= t_us_ack]
    if not len(antes):
        return None
    base, previo = antes[-1, 1], antes[-1]
    for t, a in tele[tele[:, 0] > t_us_ack]:
        if abs(a - base) > umbral:
            d0, d1 = abs(previo[1] - base), abs(a - base)
            frac = (umbral - d0) / (d1 - d0) if d1 > d0 else 1.0
            return float(previo[0] + np.clip(frac, 0, 1) * (t - previo[0]))
        previo = (t, a)
    return None


class RelojEsp32:
    """Convierte el t_us del ESP32 al reloj de la PC con los pares (t_us del ACK, hora local de
    llegada del ACK): ajuste lineal, que absorbe la deriva entre los dos relojes."""

    def __init__(self, n=50):
        self.pares = deque(maxlen=n)

    def agregar(self, t_us, t_local):
        self.pares.append((float(t_us), float(t_local)))

    def a_local(self, t_us):
        if not self.pares:
            return None
        p = np.array(self.pares)
        if len(p) < 3 or p[-1, 0] - p[0, 0] < 1e6:          # pocos pares: solo el desfase
            return float(t_us / 1e6 + np.median(p[:, 1] - p[:, 0] / 1e6))
        # el minimo retardo de llegada es el bueno: se ajusta y se desplaza al borde inferior
        m, b = np.polyfit(p[:, 0] / 1e6, p[:, 1], 1)
        b += np.min(p[:, 1] - (m * p[:, 0] / 1e6 + b))
        return float(m * t_us / 1e6 + b)


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
        self._tele = deque(maxlen=3000)   # telemetria (t_us del ESP32, angulo)
        self._acks_us = {}                # seq -> t_us del ACK
        self._reloj_esp = RelojEsp32()
        self.latencias_mecanicas = deque(maxlen=100)   # s entre el ACK y el inicio real

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

    def inicio_movimiento(self, seq, t_ack, espera_s=0.4):
        """(hora local del inicio real del movimiento, como se obtuvo). Con telemetria: el
        primer cambio del angulo tras el ACK. Si no hay telemetria o no se ve el inicio: el ACK
        mas la latencia mecanica media medida ('ack+latencia') o el ACK solo ('ack')."""
        t_us_ack = self._acks_us.get(seq)
        if t_us_ack is not None:
            t_fin = time.time() + espera_s        # que llegue telemetria de despues del inicio
            while time.time() < t_fin and not (self._tele and self._tele[-1][0] > t_us_ack + 300_000):
                time.sleep(0.01)
            t_us0 = inicio_por_telemetria(list(self._tele), t_us_ack) if self._tele else None
            if t_us0 is not None:
                t0 = self._reloj_esp.a_local(t_us0)
                if t0 is not None and 0.0 <= t0 - t_ack <= 0.5:
                    self.latencias_mecanicas.append(t0 - t_ack)
                    return t0, 'telemetria'
        if self.latencias_mecanicas:
            return t_ack + float(np.mean(self.latencias_mecanicas)), 'ack+latencia'
        return t_ack, 'ack'


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
            if len(partes) >= 3:                  # A,<seq>,<t_us>: par para el reloj del ESP32
                self._acks_us[int(partes[1])] = int(partes[2])
                self._reloj_esp.agregar(int(partes[2]), t)
        elif partes[0] == 'T' and len(partes) >= 4:
            self.telemetria = {'t_us': int(partes[1]), 'angulo': int(partes[2]),
                               'fsr': int(partes[3])}
            self._tele.append((int(partes[1]), int(partes[2])))

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

    Con `caos` (un PlanCaos) pierde ACK y mete picos de latencia de forma reproducible.
    Simula tambien la mecanica: el movimiento empieza entre 30 y 150 ms despues del ACK
    (latencia_mecanica_simulada) y emite telemetria a 50 Hz con un reloj de ESP32 propio,
    desfasado y con deriva, como el real."""

    def __init__(self, latencia_ms=8.0, jitter_ms=1.5, semilla=0, caos=None, timeout_ack=0.3, telemetria=True):
        super().__init__()
        from pylsl import local_clock
        self._clock = local_clock
        self.lat, self.jit = latencia_ms, jitter_ms
        self.rng = np.random.default_rng(semilla)
        self.semilla = semilla
        self.caos, self.timeout_ack = caos, timeout_ack
        self.angulo = 0.0
        self.con_telemetria = telemetria
        self._esp0 = local_clock()

    def _t_us(self, t_local):
        """Reloj del ESP32 simulado: otro origen y 30 ppm de deriva."""
        return (t_local - self._esp0) * 1e6 * (1 + 30e-6) + 123_456.0

    def _telemetria(self, t_ack, desde, hacia, dur_ms):
        """Muestras (t_us, angulo) a 50 Hz alrededor del movimiento."""
        t0 = t_ack + latencia_mecanica_simulada(self.seq, self.semilla)
        for t in np.arange(t_ack - 0.1, t0 + dur_ms / 1000 + 0.1, 1 / config.TELEMETRIA_HZ):
            avance = np.clip((t - t0) / (dur_ms / 1000), 0, 1)
            angulo = desde + avance * (hacia - desde) + self.rng.normal(0, 1.0)
            self._tele.append((self._t_us(t + self.rng.uniform(0, 0.002)), angulo))

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
        desde = self._a_firmware(self.angulo)
        self.angulo = float(np.clip(fraccion, 0, 1))
        self.telemetria = {'angulo': self._a_firmware(self.angulo), 'fsr': 0}
        if self.con_telemetria:
            self._acks_us[self.seq] = self._t_us(t_ack)
            self._reloj_esp.agregar(self._t_us(t_ack), t_ack + self.rng.uniform(0, 0.003))
            self._telemetria(t_ack, desde, self._a_firmware(self.angulo), dur_ms)
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

    Canales: ajustar() puede recibir candidatos ({'nombre': indices}) y elige por
    validacion cruzada; la BA que reporta es anidada (la eleccion se repite dentro de
    cada pliegue), asi elegir no la infla. La eleccion queda en self.eleccion.
    """

    def __init__(self, paso_recentrado=0.02, canales=None):
        self.paso, self.canales = paso_recentrado, canales

    @staticmethod
    def _blanquear(C, M_isqrt):
        return M_isqrt @ C @ M_isqrt

    def _tangente(self, X):
        """Rasgos sin etiquetas (covarianza, recentrado, espacio tangente) de todo X."""
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from pyriemann.utils.base import invsqrtm
        from pyriemann.utils.mean import mean_riemann
        cov = Covariances('oas')
        C = cov.fit_transform(X)
        M = mean_riemann(C)
        Mi = invsqrtm(M)
        Cw = np.array([self._blanquear(c, Mi) for c in C])
        ts = TangentSpace(metric='riemann').fit(Cw)
        return cov, M, ts, ts.transform(Cw)

    def ajustar(self, X, y, candidatos=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        todos = list(range(X.shape[1]))
        if not candidatos:
            candidatos = {'fijos': self.canales if self.canales is not None else todos}
        k = int(min(5, np.bincount(y).min()))
        pliegues = StratifiedKFold(max(k, 2), shuffle=True, random_state=0)
        Z = {n: self._tangente(X[:, c])[3] for n, c in candidatos.items()}
        lr = lambda: LogisticRegression(max_iter=2000)
        ba_cv = lambda Zc, yc, cv: exactitud_balanceada(yc, cross_val_predict(lr(), Zc, yc, cv=cv))
        nombres = list(candidatos)                # en empate gana el primero (el de menos canales)
        if len(nombres) > 1:                      # eleccion repetida dentro de cada pliegue: BA sin sesgo
            self.pred_cv = np.zeros(len(y), dtype=int)
            for ent, pru in pliegues.split(Z[nombres[0]], y):
                k_in = int(min(4, np.bincount(y[ent]).min()))
                interno = StratifiedKFold(max(k_in, 2), shuffle=True, random_state=1)
                mejor = max(nombres, key=lambda n: ba_cv(Z[n][ent], y[ent], interno))
                self.pred_cv[pru] = lr().fit(Z[mejor][ent], y[ent]).predict(Z[mejor][pru])
            self.puntajes = {n: ba_cv(Z[n], y, pliegues) for n in nombres}
            self.eleccion = max(nombres, key=lambda n: self.puntajes[n])
        else:
            self.eleccion = nombres[0]
            self.pred_cv = cross_val_predict(lr(), Z[self.eleccion], y, cv=pliegues)
            self.puntajes = {self.eleccion: exactitud_balanceada(y, self.pred_cv)}
        self.canales = list(candidatos[self.eleccion])
        self.cov, self.M, self.ts, Zf = self._tangente(X[:, self.canales])
        self.y_cal = y
        self.ba = exactitud_balanceada(y, self.pred_cv)
        clf = lr().fit(Zf, y)
        self.w0, self.c0 = clf.coef_[0].copy(), float(clf.intercept_[0])
        return self

    def phi(self, x, actualizar_centro=True):
        """Rasgos de una ventana (canales x muestras), o None si la ventana no es finita."""
        from pyriemann.utils.base import invsqrtm
        from pyriemann.utils.geodesic import geodesic_riemann
        if not np.isfinite(x).all():
            return None                      # ventana corrupta: ni rasgos ni recentrado
        canales = getattr(self, 'canales', None)
        if canales is not None:
            x = x[canales]
        C = self.cov.transform(x[None])[0]
        if actualizar_centro and self.paso > 0:
            self.M = geodesic_riemann(self.M, C, self.paso)
        return self.ts.transform(self._blanquear(C, invsqrtm(self.M))[None])[0]


def _medias_por_ventana(X, fs=250, inicio=0.15, fin=0.65, ancho=0.05, pre=-config.EPOCA_ERRP[0]):
    """Vista temporal del ErrP: promedio por canal en ventanas de 50 ms entre 150 y 650 ms."""
    cortes = [(int((pre + t) * fs), int((pre + t + ancho) * fs)) for t in np.arange(inicio, fin, ancho)]
    return np.concatenate([X[:, :, i:j].mean(axis=2) for i, j in cortes], axis=1)


def _subconjunto(X, canales):
    """Los canales elegidos de cada epoca (funcion de modulo: el modelo se puede guardar)."""
    return X[:, canales, :]


def _potencia_theta(X, fs=250, pre=-config.EPOCA_ERRP[0]):
    """Vista theta: log de la potencia en 4-8 Hz en Fz y Cz entre 200 y 600 ms tras el
    movimiento. Tras un error aumenta la theta frontal de la linea media, aunque no tenga fase
    fija (el promedio de epocas no la ve)."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, (4.0, 8.0), btype='bandpass', fs=fs, output='sos')
    xf = sosfiltfilt(sos, X[:, config.indices(['Fz', 'Cz']), :], axis=-1)
    tramo = xf[:, :, int((pre + 0.2) * fs):int((pre + 0.6) * fs)]
    return np.log(np.mean(tramo ** 2, axis=2) + 1e-6)


VISTAS_ERRP = {'dos': ('temporal', 'geometrica'), 'tres': ('temporal', 'geometrica', 'theta')}


class DetectorErrP:
    """Detector de ErrP de dos o tres vistas, fusionadas y calibradas.

    - Vista temporal: medias por ventana de 50 ms + LDA con encogimiento
      automatico (Ledoit-Wolf). Robusta con pocos ensayos.
    - Vista geometrica: covarianzas aumentadas con los prototipos de "error" y
      "correcto" en el espacio tangente de Riemann + regresion logistica.
      Captura la relacion espacial entre electrodos.
    - Vista theta (opcional): potencia 4-8 Hz en Fz y Cz entre 200 y 600 ms.
    - Fusion suave y calibracion de Platt: la salida es una probabilidad real, que
      el agente usa completa.
    - Umbral de Neyman-Pearson: el que maximiza la exactitud balanceada con
      especificidad >= 0.90, elegido con validacion ANIDADA (dentro de cada
      pliegue): asi la BA del CP3 no esta inflada.
    - Canales y vistas: ajustar() puede recibir candidatos ({'nombre': (canales,
      'dos' | 'tres')}) y elige por AUC de validacion cruzada, tambien dentro de
      cada pliegue para reportar. La eleccion queda en self.eleccion.
    - Detector de rareza: distancia riemanniana a la media de calibracion.
    """

    def __init__(self, umbral_amplitud_uv=100.0, z_rareza=3.5, canales=None, vistas='dos'):
        self.umbral_amp, self.z_rareza = umbral_amplitud_uv, z_rareza
        self.canales, self.vistas = canales, vistas
        self.umbral = 0.5

    def _pipe(self, canales=None, vistas=None):
        from pyriemann.estimation import ERPCovariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler
        canales = canales if canales is not None else (self.canales if self.canales is not None
                                                       else list(range(len(config.CANALES_EEG))))
        sub = lambda: FunctionTransformer(_subconjunto, kw_args={'canales': list(canales)})
        piezas = {
            'temporal': make_pipeline(sub(), FunctionTransformer(_medias_por_ventana),
                                      LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
            'geometrica': make_pipeline(sub(), ERPCovariances(estimator='oas'), TangentSpace(metric='riemann'),
                                        LogisticRegression(class_weight='balanced', max_iter=2000)),
            'theta': make_pipeline(FunctionTransformer(_potencia_theta), StandardScaler(),
                                   LogisticRegression(class_weight='balanced', max_iter=2000)),
        }
        fusion = VotingClassifier([(v, piezas[v]) for v in VISTAS_ERRP[vistas or self.vistas]], voting='soft')
        return CalibratedClassifierCV(fusion, method='sigmoid', cv=3)

    def ajustar(self, X, y, candidatos=None, evaluar=True):
        """evaluar=False: sin la validacion anidada que estima sens / espec / BA (para re-entrenar
        en el lazo, donde el modelo nuevo se evalua en sombra con epocas que no vio)."""
        from pyriemann.estimation import Covariances
        from pyriemann.utils.distance import distance_riemann
        from pyriemann.utils.mean import mean_riemann
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        if not candidatos:
            candidatos = {'fijo': (self.canales if self.canales is not None
                                   else list(range(X.shape[1])), self.vistas)}
        nombres = list(candidatos)                # en empate gana el primero (el mas simple)
        k = int(min(4, np.bincount(y).min()))
        pliegues = StratifiedKFold(max(k, 2), shuffle=True, random_state=0)

        def probas(n, Xs, ys, cv):
            return cross_val_predict(self._pipe(*candidatos[n]), Xs, ys, method='predict_proba', cv=cv)[:, 1]
        # Validacion ANIDADA: en cada pliegue se eligen la configuracion (por AUC) y el umbral
        # con validacion cruzada del entrenamiento, y se aplican a la prueba. Elegir sobre los
        # mismos puntajes que se evaluan inflaba la BA reportada (maldicion del ganador).
        self.pred_cv = np.zeros(len(y), dtype=int)
        for ent, pru in (pliegues.split(X, y) if evaluar else []):
            k_in = int(min(3, np.bincount(y[ent]).min()))
            interno = StratifiedKFold(max(k_in, 2), shuffle=True, random_state=1)
            p_ent = {n: probas(n, X[ent], y[ent], interno) for n in nombres}
            mejor = max(nombres, key=lambda n: roc_auc_score(y[ent], p_ent[n]))
            u = self._umbral_neyman_pearson(p_ent[mejor], y[ent])
            p_pru = self._pipe(*candidatos[mejor]).fit(X[ent], y[ent]).predict_proba(X[pru])[:, 1]
            self.pred_cv[pru] = (p_pru > u).astype(int)
        # configuracion y umbral del modelo final, con validacion cruzada de todo
        p_todo = {n: probas(n, X, y, pliegues) for n in nombres}
        self.puntajes = {n: float(roc_auc_score(y, p_todo[n])) for n in nombres}
        self.eleccion = max(nombres, key=lambda n: self.puntajes[n])
        self.canales, self.vistas = list(candidatos[self.eleccion][0]), candidatos[self.eleccion][1]
        self.p_error_cal = float(y.mean())
        self.umbral = self._umbral_neyman_pearson(p_todo[self.eleccion], y)
        self.y_cal = y
        if evaluar:
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


class DetectorCoadaptativo:
    """El detector de ErrP sigue aprendiendo en el lazo, con las epocas que el piloto genera en
    el contexto real (en bloques con senal se sabe que pasos fueron erroneos).

    - Evaluacion secuencial honesta: cada epoca se puntua con el modelo vigente ANTES de usarse
      para entrenar. ba_secuencial() es la BA en vivo.
    - Cada `cada` epocas nuevas (sin artefacto) se re-entrena en otro hilo con calibracion + lazo,
      con la misma configuracion (canales y vistas) que eligio la calibracion.
    - El modelo nuevo no se evalua con las epocas con que se entreno: se prueba EN SOMBRA con las
      `prueba` epocas siguientes, y solo reemplaza al vigente si su BA no es peor. Si empeora, se
      descarta.
    - El cambio es atomico y ocurre en el hilo del lazo (dentro de observar), asi que no hay
      carreras con quien usa self.actual. al_cambiar(nuevo) avisa para actualizar el umbral, el
      agente y el ConfianzaDetector de forma coherente.
    """

    def __init__(self, detector, X_cal, y_cal, cada=20, prueba=20, al_cambiar=None):
        self.actual, self.cada, self.prueba, self.al_cambiar = detector, cada, prueba, al_cambiar
        self._X, self._y = [np.asarray(x) for x in X_cal], [int(v) for v in y_cal]
        self.historial = []                      # (detecto el vigente, era error), en orden
        self.version, self.descartes, self.nuevas = 1, 0, 0
        self.candidato, self._sombra = None, []  # (detecta candidato, detecta vigente, error)
        self._hilo, self._listo = None, None

    def _entrenar(self, X, y):
        return DetectorErrP(canales=self.actual.canales, vistas=self.actual.vistas).ajustar(
            np.array(X), np.array(y), evaluar=False)

    def _en_hilo(self, X, y):
        try:
            self._listo = self._entrenar(X, y)
        except Exception as e:                   # un re-entrenamiento fallido no tumba el lazo
            print(f'  [detector] no se pudo re-entrenar: {e}', flush=True)
            self._listo = None

    def observar(self, e, erroneo, p, artefacto=False):
        """e: epoca del paso; p: puntaje que le dio el modelo vigente (ya calculado)."""
        erroneo = int(bool(erroneo))
        if artefacto or e is None:
            return
        self.historial.append((int(p > self.actual.umbral), erroneo))
        if self._hilo is not None and not self._hilo.is_alive():   # termino de entrenar: a la sombra
            self._hilo, self.candidato, self._listo, self._sombra = None, self._listo, None, []
        if self.candidato is not None:
            c = int(self.candidato.p_error(e) > self.candidato.umbral)
            self._sombra.append((c, self.historial[-1][0], erroneo))
            if len(self._sombra) >= self.prueba:
                self._decidir()
        self._X.append(np.asarray(e)); self._y.append(erroneo)
        self.nuevas += 1
        if self.nuevas >= self.cada and self._hilo is None and self.candidato is None:
            self.nuevas = 0
            self._hilo = threading.Thread(target=self._en_hilo, args=(list(self._X), list(self._y)), daemon=True)
            self._hilo.start()

    def _decidir(self):
        s = np.array(self._sombra)
        ba = lambda col: _ba_o_none(s[:, 2], s[:, col])
        nuevo, vigente = ba(0), ba(1)
        if nuevo is not None and vigente is not None and nuevo >= vigente:
            errores, aciertos = s[s[:, 2] == 1, 0], s[s[:, 2] == 0, 0]
            # sens y espec del nuevo, medidas en sombra con epocas que no vio
            self.candidato.sens = float(errores.mean()) if len(errores) else self.actual.sens
            self.candidato.espec = float(1 - aciertos.mean()) if len(aciertos) else self.actual.espec
            self.candidato.ba = 0.5 * (self.candidato.sens + self.candidato.espec)
            self.actual, self.version = self.candidato, self.version + 1
            if self.al_cambiar:
                self.al_cambiar(self.actual)
        else:
            self.descartes += 1
        self.candidato, self._sombra = None, []

    def ba_secuencial(self, desde=0, hasta=None):
        """BA en vivo (cada epoca puntuada antes de entrenar con ella), o None sin las dos clases."""
        h = np.array(self.historial[desde:hasta]).reshape(-1, 2)
        return _ba_o_none(h[:, 1], h[:, 0])

    def esperar(self):
        """Espera a que termine el re-entrenamiento en curso (para pruebas y al cerrar)."""
        if self._hilo is not None:
            self._hilo.join()


def _ba_o_none(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if not ((y == 1).any() and (y == 0).any()):
        return None
    return exactitud_balanceada(y, pred)


def exactitud_balanceada(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return float(0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean()))


def evaluar_latencias(latencias_ms):
    """CP1 de la ortesis: latencias del ACK (nan = ACK perdido) contra tres umbrales robustos
    de config. Devuelve {'ok', 'mad_ms', 'p95_ms', 'perdidos', 'texto'}."""
    lat = np.asarray(latencias_ms, dtype=float)
    vivas = lat[np.isfinite(lat)]
    perdidos = 1.0 - vivas.size / max(lat.size, 1)
    if vivas.size < 2:
        return {'ok': False, 'mad_ms': float('nan'), 'p95_ms': float('nan'), 'perdidos': perdidos,
                'texto': f'sin ACK suficientes ({vivas.size} de {lat.size})'}
    mediana = float(np.median(vivas))
    mad = float(np.median(np.abs(vivas - mediana)))
    p95 = float(np.quantile(vivas, 0.95))
    ok = mad <= config.CP1_MAD_MAX_MS and p95 <= config.CP1_P95_MAX_MS and perdidos <= config.CP1_ACK_PERDIDOS_MAX
    texto = (f'latencia ACK mediana {mediana:.1f} ms, MAD {mad:.1f} (max {config.CP1_MAD_MAX_MS:.0f}), '
             f'p95 {p95:.1f} (max {config.CP1_P95_MAX_MS:.0f}), ACK perdidos {100 * perdidos:.0f} % '
             f'(max {100 * config.CP1_ACK_PERDIDOS_MAX:.0f})')
    return {'ok': ok, 'mad_ms': mad, 'p95_ms': p95, 'perdidos': perdidos, 'texto': texto}


def intervalo_error(errores, nivel=0.90, tam_ensayo=config.PASOS_ENSAYO, n_boot=2000, semilla=0):
    """Intervalo bootstrap de una tasa de error del lazo, remuestreando ENSAYOS (bloques de
    `tam_ensayo` pasos con la misma meta) y no pasos sueltos: los errores de un ensayo estan
    correlacionados y remuestrear pasos daria un intervalo demasiado estrecho."""
    e = np.asarray(errores, dtype=float)
    if e.size == 0:
        return 0.0, 1.0
    bloques = [e[i:i + tam_ensayo] for i in range(0, e.size, tam_ensayo)]
    rng = np.random.default_rng(semilla)
    tasas = []
    for _ in range(n_boot):
        elegidos = rng.integers(0, len(bloques), len(bloques))
        tasas.append(np.concatenate([bloques[i] for i in elegidos]).mean())
    q = (1 - nivel) / 2
    return float(np.quantile(tasas, q)), float(np.quantile(tasas, 1 - q))


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
