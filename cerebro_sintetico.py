"""Cerebro sintetico: un gemelo digital del piloto que publica el flujo 'EEG' por LSL
y REACCIONA al lazo como lo haria una persona.

Sustituye a puente_lsl.py cuando no hay casco. A diferencia de la placa sintetica
de BrainFlow (ruido puro), este cerebro:

  - escucha las senales del orquestador (cue_cerrar / cue_relaja) e "imagina":
    desincronizacion (ERD) de los ritmos mu (8-13 Hz) y beta (18-26 Hz) sobre
    C3 al imaginar cerrar la mano derecha, con transicion suave;
  - mira la ortesis: escucha cada paso (flujo 'Paso') y su ACK (paso_ack:<seq>,
    estampado a la hora real del movimiento); si el movimiento contradice la
    intencion, genera un ErrP fronto-central (Ne ~250 ms, Pe ~350 ms, N ~500 ms)
    con jitter de latencia y amplitud; si es correcto, solo una respuesta visual
    pequena (lo que hace dificil, y realista, la deteccion);
  - tiene fondo 1/f, alfa, conduccion de volumen, ruido de 60 Hz, parpadeos en
    Fz (para probar el detector de rareza) y, opcional, fatiga: el ERD se
    debilita y el alfa sube con el tiempo (para probar recentrado y agente).

Con esto el camino REAL completo (decoder de Riemann, detector de ErrP,
calibracion secuencial, ConfianzaDetector, agente) se valida con verdad conocida.

Uso
  python cerebro_sintetico.py                       piloto "bueno"
  python cerebro_sintetico.py --erd 0.15 --errp 4   piloto dificil
  python cerebro_sintetico.py --fatiga 0.3          se cansa durante la sesion
  python cerebro_sintetico.py --banco               evalua decoder y detector en segundos
"""
import argparse
import json
import threading
import time

import numpy as np
from pylsl import StreamInlet, StreamOutlet, local_clock, resolve_byprop
from scipy.signal import butter, sosfilt

import config

CH = config.CANALES_EEG                      # FC1 FC2 C3 C4 CP1 CP2 Cz Fz
FS = config.FLUJOS['EEG'][2]
IDX = {c: i for i, c in enumerate(CH)}

# pesos espaciales (0-1) de cada fuente sobre los 8 electrodos
W_MU_IZQ = np.array([0.4, 0.1, 1.0, 0.15, 0.5, 0.1, 0.3, 0.05])    # corteza motora izquierda
W_MU_DER = np.array([0.1, 0.4, 0.15, 1.0, 0.1, 0.5, 0.3, 0.05])
W_ERRP = np.array([0.8, 0.8, 0.4, 0.4, 0.3, 0.3, 1.0, 0.9])         # fronto-central
W_ALFA = np.array([0.2, 0.2, 0.4, 0.4, 0.9, 0.9, 0.3, 0.1])         # posterior
W_PARPADEO = np.array([0.3, 0.3, 0.05, 0.05, 0.0, 0.0, 0.15, 1.0])


def plantilla_errp(error, amp_uv, rng):
    """ERP de retroalimentacion (1 s). Error: Ne/Pe/N tardia. Correcto: P300 visual chico."""
    t = np.arange(int(FS)) / FS
    lat = rng.normal(0, 0.03)
    a = amp_uv * rng.uniform(0.7, 1.3)
    g = lambda mu, sd: np.exp(-((t - mu - lat) ** 2) / (2 * sd ** 2))
    visual = 2.0 * g(0.32, 0.06)
    if not error:
        return visual
    return visual + a * (-0.7 * g(0.25, 0.035) + 1.0 * g(0.36, 0.05) - 0.5 * g(0.50, 0.06))


class Cerebro:
    def __init__(self, a):
        self.a = a
        self.rng = np.random.default_rng(a.semilla)
        self.meta, self.dir_paso = 0, 0          # 0 = sin imaginar
        self.gan_izq = self.gan_der = 1.0        # ganancia mu/beta (1 = reposo)
        self.eventos = []                        # (t0, plantilla)
        self.t0_sesion = local_clock()
        self.n_err = self.n_ok = 0
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
        self.fase60 = 0.0

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

    def _marcador(self, txt, t):
        if txt == config.CUE_CERRAR:
            self.meta = 1
        elif txt == config.CUE_RELAJA:
            self.meta = -1
        elif txt.startswith('paso_ack:') and self.meta != 0 and self.dir_paso != 0:
            error = self.dir_paso != self.meta
            self.n_err += error
            self.n_ok += not error
            with self.lock:
                self.eventos.append((t, plantilla_errp(error, self.a.errp, self.rng)))

    # ------------------------------------------------------------ genera EEG
    def _banda(self, clave, sos, k, n):
        """Ruido de banda angosta con varianza unitaria (estado del filtro entre bloques)."""
        y, self.z[clave][k] = sosfilt(sos, self.rng.normal(size=n), zi=self.z[clave][k])
        return y / self.norma[clave]

    def generar(self, ts):
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
            for t0, pl in self.eventos:
                k = np.round((ts - t0) * FS).astype(int)
                ok = (k >= 0) & (k < len(pl))
                if ok.any():
                    x[:, ok] += np.outer(W_ERRP, pl[k[ok]])
                if ts[-1] - t0 < 1.2:
                    vivos.append((t0, pl))
            self.eventos = vivos
        # parpadeos
        if self.rng.random() < a.parpadeos * n / FS:
            i0 = self.rng.integers(0, n)
            dur = int(0.3 * FS)
            forma = 120 * np.sin(np.linspace(0, np.pi, dur))[: n - i0]
            x[:, i0:i0 + len(forma)] += np.outer(W_PARPADEO, forma)
        return (self.mezcla @ x).T


# ======================================================================
# Banco de pruebas offline: genera sesiones completas sin LSL y en segundos,
# para comparar decoders/detectores o elegir parametros antes del domingo.
def _args(**k):
    a = argparse.Namespace(erd=0.25, errp=6.0, fatiga=0.0, parpadeos=0.15, semilla=0)
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
        cer.eventos.append((t + 1 / FS, plantilla_errp(err, cer.a.errp, cer.rng)))
        post, t = _bloque(cer, t, 1.2)
        xf = hw.filtrar(np.hstack([pre, post]), config.BANDA_ERRP, FS)
        i0 = pre.shape[1] - int(antes * FS) + 1
        e = xf[:, i0:i0 + int(FS)]
        X.append(e - e[:, :int(antes * FS)].mean(axis=1, keepdims=True))
        y.append(int(err))
    return np.array(X), np.array(y)


def banco(a):
    import hardware as hw
    print(f'Banco offline: ERD {a.erd}, ErrP {a.errp} uV, fatiga {a.fatiga}')
    X, y = sesion_mi(40, erd=a.erd, fatiga=a.fatiga, semilla=a.semilla)
    print(f'  decoder MI (40 ensayos): BA {hw.DecoderIM().ajustar(X, y).ba:.2f}')
    X, y = sesion_errp(120, erd=a.erd, errp=a.errp, semilla=a.semilla)
    d = hw.DetectorErrP().ajustar(X, y)
    print(f'  detector ErrP (120 epocas): sens {d.sens:.2f}, espec {d.espec:.2f}, BA {d.ba:.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--erd', type=float, default=0.25, help='profundidad del ERD (0-1); mas = piloto mas facil')
    ap.add_argument('--errp', type=float, default=6.0, help='amplitud del ErrP en uV')
    ap.add_argument('--fatiga', type=float, default=0.0, help='0 = nunca se cansa, 1 = mucho')
    ap.add_argument('--parpadeos', type=float, default=0.15, help='parpadeos por segundo')
    ap.add_argument('--semilla', type=int, default=0)
    ap.add_argument('--banco', action='store_true', help='evalua decoder y detector offline y sale')
    a = ap.parse_args()
    if a.banco:
        return banco(a)

    cer = Cerebro(a)
    outlet = StreamOutlet(config.crear_info('EEG'), chunk_size=10)
    threading.Thread(target=cer.escuchar, daemon=True).start()
    config.RESULTADOS.mkdir(exist_ok=True)
    rng = np.random.default_rng(a.semilla)
    config.IMPEDANCIAS_JSON.write_text(json.dumps(
        {'t': time.time(), 'simulada': True,
         'kohm': {c: float(rng.uniform(4, 14)) for c in CH}}, indent=1))
    print(f'Cerebro sintetico publicando EEG ({len(CH)} canales, {FS} Hz): ERD {a.erd}, '
          f'ErrP {a.errp} uV, fatiga {a.fatiga}. Ctrl+C para parar.')
    t_ult, t_rep = local_clock(), time.time()
    try:
        while True:
            ahora = local_clock()
            n = int((ahora - t_ult) * FS)
            if n > 0:
                ts = t_ult + np.arange(1, n + 1) / FS
                outlet.push_chunk(cer.generar(ts).tolist(), ts[-1])
                t_ult = ts[-1]
            if time.time() - t_rep > 10:
                print(f'  imaginando: {["relaja", "nada", "cerrar"][cer.meta + 1]:6s} | '
                      f'movimientos vistos: {cer.n_ok} correctos, {cer.n_err} erroneos')
                t_rep = time.time()
            time.sleep(0.02)
    except KeyboardInterrupt:
        print('Deteniendo...')


if __name__ == '__main__':
    main()
