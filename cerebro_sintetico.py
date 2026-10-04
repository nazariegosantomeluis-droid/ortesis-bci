"""Cerebro sintetico: un gemelo digital del piloto con un g.tec Unicorn Hybrid Black puesto.
Publica por LSL lo mismo que el casco real y REACCIONA al lazo como lo haria una persona.

Sustituye al casco cuando no lo hay. A diferencia de la placa sintetica de BrainFlow
(ruido puro), este cerebro:

  - escucha las senales del orquestador (cue_cerrar / cue_relaja) e "imagina":
    desincronizacion (ERD) de los ritmos mu (8-13 Hz) y beta (18-26 Hz) sobre
    C3 al imaginar cerrar la mano derecha, con transicion suave;
  - mira la ortesis: escucha cada paso (flujo 'Paso') y su ACK (paso_ack:<seq>,
    estampado a la hora real del movimiento). Todo movimiento visto produce una
    respuesta visual occipital (P1 ~110 ms, N1 ~170 ms en PO7/Oz/PO8); si ademas
    contradice la intencion, un ErrP fronto-central (Ne ~250 ms, Pe ~350 ms,
    N ~500 ms en Fz/Cz/Pz) con jitter de latencia y amplitud;
  - tiene fondo 1/f, alfa occipital, conduccion de volumen, ruido de 60 Hz, parpadeos
    en Fz y, opcional, fatiga: el ERD se debilita y el alfa sube con el tiempo;
  - mueve la cabeza de vez en cuando: el giroscopio y el acelerometro lo registran y
    el EEG se ensucia a la vez (para probar el rechazo de artefactos por IMU);
  - lleva el contador de muestras del casco y puede perder muestras como el
    Bluetooth: el contador sigue y queda un hueco.

Formatos de salida (--formato):
  puente      lo que publica puente_lsl.py: flujo 'EEG' (8 canales) y flujo 'IMU' (6),
              con la hora de cada muestra ya reconstruida; una perdida es un hueco de hora.
  unicornlsl  lo que publica la app UnicornLSL de g.tec: UN flujo de tipo 'Data' con 17
              canales sin etiquetas (EEG 8, acelerometro 3, giroscopio 3, bateria,
              contador, validez), muestra por muestra y estampado a la llegada.

Uso
  python cerebro_sintetico.py                       piloto "bueno"
  python cerebro_sintetico.py --erd 0.15 --errp 4   piloto dificil
  python cerebro_sintetico.py --fatiga 0.3          se cansa durante la sesion
  python cerebro_sintetico.py --perdidas-bt 20      20 perdidas de Bluetooth por minuto
  python cerebro_sintetico.py --formato unicornlsl  como la app UnicornLSL
  python cerebro_sintetico.py --banco               evalua decoder y detector en segundos
  python cerebro_sintetico.py --caos 1              caos estandar: cortes de EEG, parpadeos, canal despegado
"""
import argparse
import threading
import time

import numpy as np
from pylsl import StreamInfo, StreamInlet, StreamOutlet, local_clock, resolve_byprop
from scipy.signal import butter, sosfilt

import config
from caos import PlanCaos

CH = config.CANALES_EEG                      # Fz C3 Cz C4 Pz PO7 Oz PO8 (Unicorn Hybrid Black)
FS = config.FLUJOS['EEG'][2]
IDX = {c: i for i, c in enumerate(CH)}
SERIE = 'UN-2099.01.01'                      # numero de serie de mentira (formato de g.tec)

# pesos espaciales (0-1) de cada fuente sobre los 8 electrodos
#                     Fz    C3    Cz    C4    Pz    PO7   Oz    PO8
W_MU_IZQ = np.array([0.10, 1.00, 0.40, 0.15, 0.25, 0.15, 0.05, 0.05])    # corteza motora izquierda
W_MU_DER = np.array([0.10, 0.15, 0.40, 1.00, 0.25, 0.05, 0.05, 0.15])
W_ERRP = np.array([0.90, 0.40, 1.00, 0.40, 0.60, 0.10, 0.05, 0.10])      # fronto-central (Fz, Cz, Pz)
W_VISUAL = np.array([0.05, 0.05, 0.10, 0.05, 0.30, 0.90, 1.00, 0.90])    # occipital (PO7, Oz, PO8)
W_ALFA = np.array([0.10, 0.25, 0.25, 0.25, 0.60, 0.90, 1.00, 0.90])      # occipital
W_PARPADEO = np.array([1.00, 0.10, 0.20, 0.10, 0.05, 0.00, 0.00, 0.00])  # frontal
W_THETA = np.array([1.00, 0.25, 0.80, 0.25, 0.40, 0.05, 0.05, 0.05])     # theta frontal de la linea media

# Tarea 2 (embodiment): con --embodiment e, la respuesta visual a un movimiento PROPIO se atenua
# a (1 - ATENUACION_MAX * e); la de un movimiento ajeno no. El efecto lo programamos nosotros:
# el gemelo solo sirve para verificar que el IIC lo recupera, no para afirmar que existe.
ATENUACION_MAX = 0.5


def plantilla_errp(error, amp_uv, rng):
    """ERP de retroalimentacion (1 s). Error: Ne/Pe/N tardia. Correcto: P300 chico."""
    t = np.arange(int(FS)) / FS
    lat = rng.normal(0, 0.03)
    a = amp_uv * rng.uniform(0.7, 1.3)
    g = lambda mu, sd: np.exp(-((t - mu - lat) ** 2) / (2 * sd ** 2))
    visual = 2.0 * g(0.32, 0.06)
    if not error:
        return visual
    return visual + a * (-0.7 * g(0.25, 0.035) + 1.0 * g(0.36, 0.05) - 0.5 * g(0.50, 0.06))


def plantilla_visual(amp_uv, rng):
    """Respuesta visual occipital a un movimiento visto (1 s): P1 ~110 ms y N1 ~170 ms."""
    t = np.arange(int(FS)) / FS
    lat = rng.normal(0, 0.01)
    a = amp_uv * rng.uniform(0.8, 1.2)
    g = lambda mu, sd: np.exp(-((t - mu - lat) ** 2) / (2 * sd ** 2))
    return a * (0.4 * g(0.11, 0.02) - 1.0 * g(0.17, 0.025))


def plantilla_theta(amp_uv, rng):
    """Estallido theta (~6 Hz) tras un error, maximo hacia 400 ms y SIN fase fija: sube la
    potencia en 4-8 Hz pero el promedio de epocas casi no lo ve. La literatura reporta un
    aumento de la theta frontal de la linea media tras los errores; aqui solo sirve para
    verificar que la vista theta del detector funciona (el efecto lo programamos nosotros)."""
    t = np.arange(int(FS)) / FS
    f, fase = rng.uniform(5.0, 7.0), rng.uniform(0, 2 * np.pi)
    return amp_uv * np.exp(-((t - 0.4) ** 2) / (2 * 0.1 ** 2)) * np.sin(2 * np.pi * f * t + fase)


class Cerebro:
    def __init__(self, a):
        self.a = a
        self.rng = np.random.default_rng(a.semilla)
        self.meta, self.dir_paso = 0, 0          # 0 = sin imaginar
        self.gan_izq = self.gan_der = 1.0        # ganancia mu/beta (1 = reposo)
        self.eventos = []                        # (t0, plantilla, pesos espaciales)
        self.t0_sesion = local_clock()
        self.n_err = self.n_ok = 0
        self.caos = (PlanCaos(a.caos, config.CAOS[getattr(a, 'caos_nivel', 'estandar')])
                     if getattr(a, 'caos', None) is not None else None)
        # el reloj del caos arranca con la senal del lazo (marcador bloque:LAZO_ESTATICO), o ya
        # desde el arranque con --caos-desde calibracion
        self.t0_caos = self.t0_sesion if getattr(a, 'caos_desde', 'lazo') == 'calibracion' else None
        self._canal_caos = None                  # para avisar una vez por falla
        self.lock = threading.Lock()
        n = len(CH)
        self.sos_mu = butter(4, (9, 12), btype='bandpass', fs=FS, output='sos')
        self.sos_beta = butter(4, (18, 26), btype='bandpass', fs=FS, output='sos')
        self.sos_alfa = butter(4, (8.5, 10.5), btype='bandpass', fs=FS, output='sos')
        self.norma = {'mu': np.sqrt(2 * 3 / FS), 'beta': np.sqrt(2 * 8 / FS), 'alfa': np.sqrt(2 * 2 / FS)}
        self.z = {k: [np.zeros((s.shape[0], 2)) for _ in range(m)] for k, s, m in
                  (('mu', self.sos_mu, 2), ('beta', self.sos_beta, 2), ('alfa', self.sos_alfa, 1))}
        self.rosa = np.zeros(n)
        # conduccion de volumen: mezcla leve entre electrodos
        self.mezcla = np.eye(n) + 0.08 * np.abs(self.rng.normal(size=(n, n)))
        # banco offline, sesion > 0: otra sesion del MISMO piloto. Misma anatomia (mezcla), otro ruido
        # y el casco recolocado (ganancia por electrodo entre 0.8 y 1.25). Ese cambio lo programamos
        # nosotros: sirve para probar la memoria entre sesiones, no para decir cuanto cambia una persona
        self.ganancia = np.ones(n)
        if getattr(a, 'sesion', 0):
            self.ganancia = np.random.default_rng([a.semilla, 31, a.sesion]).uniform(0.8, 1.25, n)
            self.rng = np.random.default_rng([a.semilla, 32, a.sesion])
        self.fase60 = 0.0
        # respuesta visual, cabeza, IMU y Bluetooth: generador aparte, para que el resto del EEG
        # sea la misma realizacion con o sin ellos (comparaciones limpias en el banco)
        self.rng_cuerpo = np.random.default_rng([a.semilla, 99])
        self.cabeza = None                       # movimiento en curso: (t0, dur, eje, grados/s, uV, pesos)
        self.imu = np.zeros((0, 6))              # IMU de las muestras de la ultima llamada a generar()
        self.contador = 0                        # contador de muestras del casco
        self._fin_perdida = -np.inf

    # ------------------------------------------------------------ escucha al lazo
    def escuchar(self):
        entradas = {}
        while True:
            for nombre in ('Marcadores', 'Paso'):
                if nombre not in entradas:
                    s = resolve_byprop('name', nombre, timeout=0.5)
                    if s:
                        entradas[nombre] = StreamInlet(s[0], max_buflen=30)
                        print(f'  [cerebro] escuchando {nombre}')
            for nombre, inl in list(entradas.items()):
                datos, ts = inl.pull_chunk(timeout=0.0)
                for m, t in zip(datos, ts):
                    t += inl.time_correction()
                    if nombre == 'Paso':
                        self.dir_paso = 1 if m[1] > 0 else -1
                    else:
                        self._marcador(m[0], t)
            time.sleep(0.005)

    def t_caos(self, t):
        """Segundos del reloj del caos en t, o None si el caos todavia no empieza."""
        return None if self.t0_caos is None else t - self.t0_caos

    def _marcador(self, txt, t):
        if txt == config.m_bloque('LAZO_ESTATICO') and self.t0_caos is None:
            self.t0_caos = t
        if txt == config.CUE_CERRAR:
            self.meta = 1
        elif txt == config.CUE_RELAJA:
            self.meta = -1
        elif txt.startswith('paso_ack:') and self.meta != 0 and self.dir_paso != 0:
            error = self.dir_paso != self.meta
            self.n_err += error
            self.n_ok += not error
            # el cerebro reacciona cuando VE moverse la ortesis, no cuando llega el ACK: la ortesis
            # simulada empieza a moverse entre 30 y 150 ms despues (misma funcion y semilla)
            if getattr(self.a, 'latencia_mecanica', True):
                import hardware as hw
                t += hw.latencia_mecanica_simulada(int(txt.split(':')[1]), getattr(self.a, 'semilla_ortesis', 0))
            self.movimiento(t, error)
        elif txt.startswith('paso_ajeno:') and self.meta != 0:
            # movimiento ajeno, anunciado y siempre hacia la meta: sin error y sin atenuacion
            if getattr(self.a, 'latencia_mecanica', True):
                import hardware as hw
                t += hw.latencia_mecanica_simulada(int(txt.split(':')[1]), getattr(self.a, 'semilla_ortesis', 0))
            self.movimiento(t, False, propio=False)

    def movimiento(self, t, error, propio=True):
        """El piloto ve moverse la ortesis en t: respuesta visual occipital siempre (atenuada
        segun el embodiment si el movimiento es propio) y, si el movimiento contradice su
        intencion, ErrP fronto-central."""
        n1 = self.a.n1 * (1 - ATENUACION_MAX * getattr(self.a, 'embodiment', 0.0) if propio else 1.0)
        with self.lock:
            self.eventos.append((t, plantilla_errp(error, self.a.errp, self.rng), W_ERRP))
            self.eventos.append((t, plantilla_visual(n1, self.rng_cuerpo), W_VISUAL))
            if error and getattr(self.a, 'theta', 0.0) > 0:
                self.eventos.append((t, plantilla_theta(self.a.theta, self.rng_cuerpo), W_THETA))

    # ------------------------------------------------------------ genera EEG
    def _banda(self, clave, sos, k, n):
        """Ruido de banda angosta con varianza unitaria (estado del filtro entre bloques)."""
        y, self.z[clave][k] = sosfilt(sos, self.rng.normal(size=n), zi=self.z[clave][k])
        return y / self.norma[clave]

    def generar(self, ts):
        """EEG (muestras x 8, en uV) de los instantes ts. Deja en self.imu la IMU de esas
        mismas muestras (muestras x 6: acelerometro en g, giroscopio en grados/s)."""
        n, a = len(ts), self.a
        horas = (ts[-1] - self.t0_sesion) / 3600
        fatiga = min(1.0, a.fatiga * horas * 4)              # llega a "a.fatiga" en ~15 min
        erd = a.erd * (1 - 0.6 * fatiga)
        # ERD: imaginar cerrar la mano derecha baja mu/beta sobre C3 (y un poco C4)
        obj_izq = 1 - erd if self.meta == 1 else (1 + 0.1 * erd if self.meta == -1 else 1.0)
        obj_der = 1 - 0.3 * erd if self.meta == 1 else 1.0
        tau = np.exp(-1 / (0.4 * FS))                        # transicion de ~0.4 s
        g_izq = np.empty(n); g_der = np.empty(n)
        for i in range(n):
            self.gan_izq = tau * self.gan_izq + (1 - tau) * obj_izq
            self.gan_der = tau * self.gan_der + (1 - tau) * obj_der
            g_izq[i], g_der[i] = self.gan_izq, self.gan_der
        x = np.zeros((len(CH), n))
        for k, (w, g) in enumerate(((W_MU_IZQ, g_izq), (W_MU_DER, g_der))):
            ritmo = 7.0 * self._banda('mu', self.sos_mu, k, n) + 4.0 * self._banda('beta', self.sos_beta, k, n)
            x += np.outer(w, ritmo * g)
        x += np.outer(W_ALFA, (6.0 + 6.0 * fatiga) * self._banda('alfa', self.sos_alfa, 0, n))
        # fondo 1/f aproximado (AR(1)) + blanco
        for i in range(n):
            self.rosa = 0.97 * self.rosa + self.rng.normal(0, 1.2, len(CH))
            x[:, i] += self.rosa
        x += self.rng.normal(0, 2.0, x.shape)
        # red electrica
        fase = self.fase60 + 2 * np.pi * 60 * np.arange(1, n + 1) / FS
        x += 1.5 * np.sin(fase)[None, :]
        self.fase60 = fase[-1] % (2 * np.pi)
        # ErrP y respuestas visuales programadas
        with self.lock:
            vivos = []
            for t0, pl, w in self.eventos:
                k = np.round((ts - t0) * FS).astype(int)
                ok = (k >= 0) & (k < len(pl))
                if ok.any():
                    x[:, ok] += np.outer(w, pl[k[ok]])
                if ts[-1] - t0 < 1.2:
                    vivos.append((t0, pl, w))
            self.eventos = vivos
        # parpadeos (con caos: rafagas)
        t_caos = self.t_caos(ts[-1])
        caos = self.caos if t_caos is not None else None
        tasa = a.parpadeos
        if caos and caos.activo('rafaga_parpadeos', t_caos):
            tasa = self.caos.tasas['rafaga_parpadeos']['por_segundo']
        if self.rng.random() < tasa * n / FS:
            i0 = self.rng.integers(0, n)
            dur = int(0.3 * FS)
            forma = 120 * np.sin(np.linspace(0, np.pi, dur))[: n - i0]
            x[:, i0:i0 + len(forma)] += np.outer(W_PARPADEO, forma)
        y = self.ganancia[:, None] * (self.mezcla @ x)
        y += self._cabeza_e_imu(ts)                          # el movimiento de cabeza ensucia el EEG
        # caos: un canal se despega (plano, o ruido grande con mucha red electrica)
        canal = caos.activo('canal', t_caos) if caos else None
        if canal:
            y[IDX[canal[2]]] = (0.0 if canal[3] == 'plano'
                                else 300.0 * self.rng.normal(size=n) + 200.0 * np.sin(fase))
        if canal != self._canal_caos:
            self._canal_caos = canal
            if canal:
                print(f'  [caos] canal {canal[2]} despegado ({canal[3]}) {canal[1]:.1f} s', flush=True)
        self.contador += n
        return y.T

    # ------------------------------------------------------------ cabeza e IMU
    def _cabeza_e_imu(self, ts):
        """Movimientos de cabeza ocasionales. Devuelve el artefacto de EEG (8 x n) y deja la
        IMU en self.imu: gravedad (~1 g en z) mas el movimiento en el giroscopio."""
        n, r = len(ts), self.rng_cuerpo
        tasa = getattr(self.a, 'cabeza', 0.0)
        if self.cabeza is None and tasa > 0 and r.random() < tasa * n / FS:
            self.cabeza = (ts[0], r.uniform(0.5, 1.5), int(r.integers(3)), r.uniform(40, 120),
                           r.uniform(60, 150), r.uniform(0.5, 1.0, len(CH)))
        forma = np.zeros(n)
        if self.cabeza is not None:
            t0, dur = self.cabeza[0], self.cabeza[1]
            fase = (ts - t0) / dur
            forma = np.where((fase >= 0) & (fase <= 1), np.sin(np.pi * np.clip(fase, 0, 1)), 0.0)
        imu = np.zeros((n, 6))
        imu[:, :3] = [0.0, 0.0, 1.0] + r.normal(0, 0.005, (n, 3))          # gravedad
        imu[:, 3:] = r.normal(0, 0.5, (n, 3))                               # giroscopio en reposo
        artefacto = np.zeros((len(CH), n))
        if self.cabeza is not None:
            _, _, eje, vel, uv, pesos = self.cabeza
            imu[:, 3 + eje] += vel * forma
            imu[:, (eje + 1) % 3] += 0.2 * forma                            # la gravedad cambia de eje
            artefacto = np.outer(pesos, uv * forma) + r.normal(0, 10.0, (len(CH), n)) * (forma > 0)
            if ts[-1] > self.cabeza[0] + self.cabeza[1]:
                self.cabeza = None
        self.imu = imu
        return artefacto

    def entregadas(self, ts):
        """Mascara de las muestras que SI llegan por Bluetooth (False = perdida; el contador
        del casco sigue contando). --perdidas-bt da las perdidas por minuto; el caos agrega las
        suyas (perdida_bt)."""
        r, por_min = self.rng_cuerpo, getattr(self.a, 'perdidas_bt', 0.0)
        if por_min > 0 and ts[-1] > self._fin_perdida and r.random() < por_min / 60 * len(ts) / FS:
            self._fin_perdida = ts[0] + r.uniform(0.02, 0.2)
        llega = ts > self._fin_perdida
        t_caos = self.t_caos(ts[0])
        if self.caos is not None and t_caos is not None:
            llega &= np.array([not self.caos.activo('perdida_bt', t_caos + (t - ts[0])) for t in ts])
        return llega


# ======================================================================
class Salida:
    """Publica por LSL en uno de los dos formatos (ver el encabezado del modulo)."""

    def __init__(self, a):
        self.formato, self.serie = a.formato, a.nombre_lsl
        self.offset = np.random.default_rng([a.semilla, 7]).uniform(-1, 1, len(CH)) * a.offset_dc
        self.abrir()

    def abrir(self):
        if self.formato == 'puente':
            self.eeg = StreamOutlet(config.crear_info('EEG'), chunk_size=10)
            self.imu = StreamOutlet(config.crear_info('IMU'), chunk_size=10)
        else:                                    # igual que UnicornLSL: tipo 'Data', sin etiquetas
            f = config.FUENTES_EEG['unicornlsl']
            self.datos = StreamOutlet(StreamInfo(self.serie, f['tipo'], f['canales'], FS, 'float32', self.serie))

    def cerrar(self):
        self.eeg = self.imu = self.datos = None

    def publicar(self, eeg, imu, ts, contadores):
        """eeg (n x 8), imu (n x 6), hora y contador de cada muestra entregada."""
        if len(ts) == 0:
            return
        eeg = eeg + self.offset                  # el EEG crudo del casco trae un offset de continua
        if self.formato == 'puente':
            self.eeg.push_chunk(eeg.tolist(), list(ts))      # hora por muestra: un hueco es un hueco
            self.imu.push_chunk(imu.tolist(), list(ts))
        else:
            f = config.FUENTES_EEG['unicornlsl']
            fila = np.zeros((len(ts), f['canales']), dtype=np.float32)
            fila[:, f['eeg']], fila[:, f['imu']] = eeg, imu
            fila[:, f['bateria']], fila[:, f['contador']], fila[:, f['validez']] = 87.0, contadores, 1.0
            for muestra in fila.tolist():
                self.datos.push_sample(muestra)  # sin hora propia: LSL estampa la llegada


# ======================================================================
# Banco de pruebas offline: genera sesiones completas sin LSL y en segundos,
# para comparar decoders/detectores o elegir parametros antes del domingo.
def _args(**k):
    a = argparse.Namespace(erd=0.25, errp=6.0, n1=4.0, theta=3.0, fatiga=0.0, parpadeos=0.15, cabeza=0.0,
                           perdidas_bt=0.0, semilla=0, caos=None, caos_desde='calibracion', embodiment=0.5)
    a.__dict__.update(k)
    return a


def _bloque(cer, t, seg):
    ts = t + np.arange(1, int(seg * FS) + 1) / FS
    return cer.generar(ts).T, ts[-1]


def sesion_mi(n=40, **k):
    """Ventanas de imaginacion motora (n, canales, 2 s) ya filtradas, y etiquetas (1 = cerrar)."""
    import hardware as hw
    cer = Cerebro(_args(**k))
    rng = np.random.default_rng(cer.a.semilla + 1)
    t, X, y = 0.0, [], []
    for _ in range(n):
        clase = int(rng.integers(0, 2))
        cer.meta = 1 if clase else -1
        sig, t = _bloque(cer, t, 3.0)
        X.append(hw.filtrar(sig, config.BANDA_MI, FS)[:, -int(config.VENTANA_MI * FS):])
        y.append(clase)
        cer.meta = 0
        _, t = _bloque(cer, t, 1.0)
    return np.array(X), np.array(y)


def sesion_errp(n=100, p_error=0.3, **k):
    """Epocas de ErrP (n, canales, 1 s) alrededor del movimiento, y etiquetas (1 = error)."""
    import hardware as hw
    cer = Cerebro(_args(**k))
    rng = np.random.default_rng(cer.a.semilla + 2)
    t, X, y = 0.0, [], []
    antes = -config.EPOCA_ERRP[0]
    for _ in range(n):
        err = bool(rng.random() < p_error)
        cer.meta = 1
        pre, t = _bloque(cer, t, 1.5)
        cer.movimiento(t + 1 / FS, err)
        post, t = _bloque(cer, t, 1.2)
        xf = hw.filtrar(np.hstack([pre, post]), config.BANDA_ERRP, FS)
        i0 = pre.shape[1] - int(antes * FS) + 1
        e = xf[:, i0:i0 + int(FS)]
        X.append(e - e[:, :int(antes * FS)].mean(axis=1, keepdims=True))
        y.append(int(err))
    return np.array(X), np.array(y)


def sesion_sham(n=40, sigue=False, **k):
    """Bloque sham offline (B1): el piloto reposa (meta 0) y la ortesis se mueve sola, en una direccion
    al azar (n/2 cerrar, n/2 abrir). Devuelve ventanas de MI (n, canales, 2 s) de 0.5 s antes a 1.5 s
    despues del inicio del movimiento, y la direccion (1 = cerrar). sigue=True es el control
    positivo: el piloto imagina lo que hace la ortesis, asi que p(t) SI debe seguirla."""
    import hardware as hw
    cer = Cerebro(_args(**k))
    rng = np.random.default_rng([cer.a.semilla, 5])           # flujo propio: etiquetas independientes del EEG
    dirs = rng.permutation(np.array([1, 0] * (n // 2)))
    t, X = 0.0, []
    for d in dirs:
        cer.meta = (1 if d else -1) if sigue else 0
        pre, t = _bloque(cer, t, 0.5)
        cer.movimiento(t + 1 / FS, False, propio=False)       # la ortesis se mueve sola
        post, t = _bloque(cer, t, 1.5)
        X.append(hw.filtrar(np.hstack([pre, post]), config.BANDA_MI, FS)[:, -int(config.VENTANA_MI * FS):])
        cer.meta = 0
        _, t = _bloque(cer, t, 1.0)
    return np.array(X), dirs


def sesion_embodiment(n=300, p_error=0.2, **k):
    """Epocas (n, canales, 1 s) de los movimientos de un lazo con movimientos ajenos (Tarea 2):
    uno de cada config.AJENOS_CADA, anunciado y hacia la meta. Devuelve (X, ajeno, correcto)."""
    import hardware as hw
    cer = Cerebro(_args(**k))
    rng = np.random.default_rng(cer.a.semilla + 3)
    t, X, aj, ok = 0.0, [], [], []
    antes = -config.EPOCA_ERRP[0]
    for i in range(n):
        ajeno = i % config.AJENOS_CADA == 1
        err = not ajeno and bool(rng.random() < p_error)
        cer.meta = 1
        pre, t = _bloque(cer, t, 1.5)
        cer.movimiento(t + 1 / FS, err, propio=not ajeno)
        post, t = _bloque(cer, t, 1.2)
        xf = hw.filtrar(np.hstack([pre, post]), config.BANDA_ERRP, FS)
        i0 = pre.shape[1] - int(antes * FS) + 1
        e = xf[:, i0:i0 + int(FS)]
        X.append(e - e[:, :int(antes * FS)].mean(axis=1, keepdims=True))
        aj.append(ajeno)
        ok.append(not err)
    return np.array(X), np.array(aj), np.array(ok)


def banco(a):
    import hardware as hw
    print(f'Banco offline: ERD {a.erd}, ErrP {a.errp} uV, fatiga {a.fatiga}')
    # como la calibracion real: canales y vistas elegidos por validacion anidada
    X, y = sesion_mi(40, erd=a.erd, fatiga=a.fatiga, semilla=a.semilla)
    dec = hw.DecoderIM().ajustar(X, y, config.candidatos('decoder'))
    print(f'  decoder MI (40 ensayos): BA {dec.ba:.2f} ({dec.eleccion})')
    X, y = sesion_errp(120, erd=a.erd, errp=a.errp, theta=a.theta, semilla=a.semilla)
    d = hw.DetectorErrP().ajustar(X, y, config.candidatos('detector'))
    print(f'  detector ErrP (120 epocas): sens {d.sens:.2f}, espec {d.espec:.2f}, BA {d.ba:.2f} ({d.eleccion})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--erd', type=float, default=0.25, help='profundidad del ERD (0-1); mas = piloto mas facil')
    ap.add_argument('--errp', type=float, default=6.0, help='amplitud del ErrP en uV')
    ap.add_argument('--n1', type=float, default=4.0, help='amplitud de la N1 visual occipital en uV')
    ap.add_argument('--theta', type=float, default=3.0, help='amplitud del estallido theta tras un error, en uV')
    ap.add_argument('--embodiment', type=float, default=0.5,
                    help='0-1: cuanto atenua el piloto la N1 de sus propios movimientos (Tarea 2)')
    ap.add_argument('--fatiga', type=float, default=0.0, help='0 = nunca se cansa, 1 = mucho')
    ap.add_argument('--parpadeos', type=float, default=0.15, help='parpadeos por segundo')
    ap.add_argument('--cabeza', type=float, default=0.02, help='movimientos de cabeza por segundo')
    ap.add_argument('--perdidas-bt', dest='perdidas_bt', type=float, default=0.0,
                    help='perdidas de Bluetooth por minuto (de 20 a 200 ms cada una)')
    ap.add_argument('--offset-dc', dest='offset_dc', type=float, default=15000.0,
                    help='offset de continua maximo por canal, en uV (el EEG crudo del casco lo trae)')
    ap.add_argument('--formato', choices=['puente', 'unicornlsl'], default='puente',
                    help="puente: flujos 'EEG' e 'IMU' del contrato; unicornlsl: un flujo 'Data' de 17 canales")
    ap.add_argument('--nombre-lsl', dest='nombre_lsl', default=SERIE,
                    help='nombre del flujo en formato unicornlsl (la app usa el numero de serie)')
    ap.add_argument('--sin-latencia-mecanica', dest='latencia_mecanica', action='store_false',
                    help='el ErrP se ancla al ACK y no al inicio real del movimiento de la ortesis simulada')
    ap.add_argument('--semilla', type=int, default=0)
    ap.add_argument('--banco', action='store_true', help='evalua decoder y detector offline y sale')
    ap.add_argument('--caos', type=int, default=None, metavar='SEMILLA',
                    help='inyecta caos: cortes de EEG, rafagas de parpadeos, canal despegado')
    ap.add_argument('--caos-desde', dest='caos_desde', choices=['lazo', 'calibracion'], default='lazo',
                    help='el caos empieza al llegar la senal del lazo (bloque:LAZO_ESTATICO) o desde el arranque')
    ap.add_argument('--caos-nivel', dest='caos_nivel', choices=sorted(config.CAOS), default='estandar',
                    help='estandar o leve (una falla cada 2 a 3 minutos)')
    a = ap.parse_args()
    if a.banco:
        return banco(a)

    cer = Cerebro(a)
    salida = Salida(a)
    threading.Thread(target=cer.escuchar, daemon=True).start()
    config.RESULTADOS.mkdir(exist_ok=True)
    print(f'Cerebro sintetico ({a.formato}): {len(CH)} canales de EEG e IMU a {FS} Hz; ERD {a.erd}, '
          f'ErrP {a.errp} uV, fatiga {a.fatiga}. Ctrl+C para parar.', flush=True)
    t_ult, t_rep = local_clock(), time.time()
    corte_previo, perdidas = None, 0
    try:
        while True:
            ahora = local_clock()
            t_caos = cer.t_caos(ahora)
            corte = cer.caos.activo('corte_eeg', t_caos) if cer.caos and t_caos is not None else None
            if corte:                              # caos: el EEG deja de llegar
                if corte_previo is None:
                    recrear = corte[1] >= cer.caos.tasas['corte_eeg']['recrear_desde_s']
                    print(f'  [caos] corte de EEG {corte[1]:.1f} s'
                          + (' (el flujo se destruye y se vuelve a crear)' if recrear else ''), flush=True)
                    if recrear:
                        salida.cerrar()            # como apagar y encender el casco
                corte_previo, t_ult = corte, ahora  # al volver no se rellena el hueco
                time.sleep(0.02)
                continue
            if corte_previo is not None:
                recreado, corte_previo = corte_previo[1] >= cer.caos.tasas['corte_eeg']['recrear_desde_s'], None
                if recreado:
                    salida.abrir()
            n = int((ahora - t_ult) * FS)
            if n > 0:
                ts = t_ult + np.arange(1, n + 1) / FS
                contadores = cer.contador + np.arange(1, n + 1)
                eeg = cer.generar(ts)
                llega = cer.entregadas(ts)
                perdidas += int((~llega).sum())
                salida.publicar(eeg[llega], cer.imu[llega], ts[llega], contadores[llega])
                t_ult = ts[-1]
            if time.time() - t_rep > 10:
                print(f'  imaginando: {["relaja", "nada", "cerrar"][cer.meta + 1]:6s} | '
                      f'movimientos vistos: {cer.n_ok} correctos, {cer.n_err} erroneos'
                      + (f' | muestras perdidas por Bluetooth: {perdidas}' if perdidas else ''), flush=True)
                t_rep = time.time()
            time.sleep(0.02)
    except KeyboardInterrupt:
        print('Deteniendo...')


if __name__ == '__main__':
    main()
