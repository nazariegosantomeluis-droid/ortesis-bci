"""Por que el agente aprende lento en el gemelo con el montaje del Unicorn (3 de octubre).

Lazo completo del gemelo SIN LSL, con la aritmetica de Orquestador.paso: calibracion de MI y
de ErrP como en el camino real (candidatos de config, 120 epocas, umbral anidado), bloque
estatico (30 pasos), bloque adaptativo (120) y perturbacion de 2.4 logits en el paso 40. Sin
movimientos ajenos ni detector co-adaptativo, para aislar al agente.

Cada sujeto (semilla) se calibra una vez. Despues el MISMO lazo (mismo ruido de fondo, mismas
metas) se corre con cada variante del agente:

  calibrada   el modo real de hoy: P_hat con la probabilidad de Platt del detector
  binaria     el modo del simulador: detecto / no detecto, con sens y espec vivas
  ...         ablaciones y correcciones candidatas (ver VARIANTES)

Mide: (1) beta y el error tras la perturbacion en cada variante; (2) la distribucion de P_hat y
del LLR por paso sobre las mismas epocas; (3) el diagrama de confiabilidad del detector
calibrado en epocas nuevas con la misma tasa de error que la calibracion.

SOLO es el gemelo: dice por que el agente es lento ahi, no como sera con una persona.

Uso: python estudios/agente_lento.py [sujetos] [repeticiones] [actual | ayer | debil]
     python estudios/agente_lento.py informe      tabla pareada de los regimenes ya corridos,
                                                  la correccion candidata en el simulador y la figura
"""
import copy
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import config
import hardware as hw
import cerebro_sintetico as cs
from agente_errp import AgenteErrP, ConfigAgente, ConfianzaDetector, logit, sigmoide
from brecha_mi import Mundo

FS = cs.FS
CACHE = config.RESULTADOS / 'agente_lento'
FIGURA = config.RAIZ / 'docs' / 'figuras' / 'agente_lento.png'
FIGURA_CONTROL = config.RAIZ / 'docs' / 'figuras' / 'control_negativo.png'     # la de la presentacion
PASOS_EST, PASOS_ADA, PERTURBA_EN = 30, 120, 40          # los del modo real
VENTANA = int(config.RECUPERACION_MAX_S / config.CICLO_S)  # los "2 min" de evaluar(): 57 pasos
META_BETA = 0.7 * config.PERTURBACION_LOGITS               # recuperacion del CP4

# Regimenes del gemelo y del detector. 'actual': lo que hace hoy la calibracion real (elige
# canales y vistas; el gemelo tiene theta tras el error). 'ayer': el de las corridas lentas del
# 2 de octubre (gemelo sin theta, detector de 8 canales y dos vistas). 'debil': ErrP mas chico.
# 'nulo': sin ErrP (control negativo).
REGIMENES = {
    'actual': {'errp': 6.0, 'theta': 3.0, 'candidatos': None},
    'ayer':   {'errp': 6.0, 'theta': 0.0, 'candidatos': {'8 canales, dos vistas': (list(range(8)), 'dos')}},
    'debil':  {'errp': 4.0, 'theta': 0.0, 'candidatos': None},
    # CONTROL NEGATIVO: el gemelo no produce ErrP. Un agente que aprende del ErrP NO debe
    # recuperarse aqui; si se recupera, lo mueve otra cosa (el supuesto de metas balanceadas).
    'nulo':   {'errp': 0.0, 'theta': 0.0, 'candidatos': None},
}
REGIMEN = 'actual'
# Donde se corta la epoca del ErrP (lo usa estudios/respaldo_ack.py): 'inicio' = el inicio real del
# movimiento (telemetria), 'ack+latencia' = el ACK mas la latencia mecanica media, 'ack' = el ACK.
ALINEACION = 'inicio'
MARGEN_S = 0.05                 # lo que se espera tras el fin de la epoca antes de cortarla
# Los topes del recorrido de la ortesis (lo usa estudios/paso_sin_movimiento.py): cada ensayo empieza
# en el punto medio y un paso que no cabe en el recorrido no se ve. None: no se modelan (todo paso
# es un movimiento visible; asi se midio este estudio). 'ciego': el gemelo no reacciona a un paso
# que no se ve, pero el lazo lee su epoca y aprende de ella (el orquestador de antes, con una
# persona). 'ignorar': el lazo no lee esa epoca ni aprende de ella (config.IGNORAR_SIN_MOVIMIENTO).
TOPES = None
LATENCIA_MEDIA_S = float(np.mean(config.LATENCIA_MECANICA_SIM_MS)) / 1000


# ------------------------------------------------------------ el gemelo, sin LSL
class Mundo2(Mundo):
    def __init__(self, semilla):
        super().__init__(semilla)
        self.cer.a.errp, self.cer.a.theta = REGIMENES[REGIMEN]['errp'], REGIMENES[REGIMEN]['theta']

    def epoca(self, t0):
        """La epoca del movimiento que empezo en t0, cortada como EntradaEEG.epoca."""
        n = int(4 * FS)
        x = self.x[:, -n:]
        t = self.t - (x.shape[1] - 1 - np.arange(x.shape[1])) / FS
        return hw.cortar_epoca(x, t, t0, FS)


def mover(m, erroneo, rng, visible=True):
    """La ortesis se mueve. El gemelo lo ve tras la latencia mecanica y la epoca se corta en el
    inicio real del movimiento (como con la telemetria del ESP32). visible=False: la ortesis ya
    estaba en el tope y no se movio; el gemelo no ve nada, pero la epoca se corta igual."""
    t_ack = m.t
    t0 = t_ack + rng.uniform(*config.LATENCIA_MECANICA_SIM_MS) / 1000
    if visible:
        m.cer.movimiento(t0, bool(erroneo))
    m.avanzar(t0 - m.t + config.EPOCA_ERRP[1] + MARGEN_S)
    return m.epoca({'inicio': t0, 'ack+latencia': t_ack + LATENCIA_MEDIA_S, 'ack': t_ack}[ALINEACION])


def epocas_mi(m, rng, n=40, espera=1.5, duracion=4.0):
    X, y = [], []
    for k in range(n):
        clase = 1 - y[-1] if (k % 2 and y) else int(rng.integers(2))
        m.avanzar(espera)
        m.cer.meta = 1 if clase else -1
        m.avanzar(duracion)
        X.append(m.ventana_mi()); y.append(clase)
    return np.array(X), np.array(y)


def epocas_errp(m, rng, n=120, p_error=0.3, espera=1.5):
    """Como BackendReal.calibrar_errp: bloques de 20 con la misma tasa de error por direccion."""
    X, y, plan = [], [], []
    while len(y) < n:
        if not plan:
            b = [(obj, i < int(round(p_error * 10))) for obj in (0, 1) for i in range(10)]
            plan = [b[i] for i in rng.permutation(len(b))]
        obj, err = plan.pop()
        m.cer.meta = 1 if obj else -1
        m.avanzar(espera)
        e = mover(m, err, rng)
        if e is not None:
            X.append(e); y.append(int(err))
    return np.array(X), np.array(y)


def preparar(semilla, salida=print):
    """Decoder y detector de un sujeto del gemelo, calibrados como en el camino real, y las
    probabilidades del detector en 400 epocas NUEVAS con la tasa de error de la calibracion."""
    nombre = f'sujeto_{semilla}' + ('' if REGIMEN == 'actual' else f'_{REGIMEN}')
    ruta = CACHE / (nombre + ('' if ALINEACION == 'inicio' else f'_corte_{ALINEACION}') + '.pkl')
    if ruta.exists():
        return pickle.loads(ruta.read_bytes())
    m = Mundo2(semilla)
    X, y = epocas_mi(m, np.random.default_rng(semilla + 100))
    dec = hw.DecoderIM().ajustar(X, y, config.candidatos('decoder'))
    Xe, ye = epocas_errp(m, np.random.default_rng(semilla + 300))
    det = hw.DetectorErrP().ajustar(Xe, ye, REGIMENES[REGIMEN]['candidatos'] or config.candidatos('detector'))
    Xn, yn = epocas_errp(m, np.random.default_rng(semilla + 400), n=400)
    mod = {'dec': dec, 'det': det, 'X_cal': Xe, 'y_cal': ye, 'y_nuevas': yn, 'X_nuevas': Xn,
           'p_nuevas': np.array([det.p_error(e) for e in Xn])}
    CACHE.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(pickle.dumps(mod))
    salida(f'  sujeto {semilla} [{REGIMEN}]: MI BA {dec.ba:.2f} ({dec.eleccion}); ErrP sens {det.sens:.2f} espec {det.espec:.2f} '
           f'BA {det.ba:.2f} ({det.eleccion}), umbral {det.umbral:.2f}')
    return mod


# ------------------------------------------------------------ el agente y sus variantes
class Agente(AgenteErrP):
    """El agente de siempre con ganchos para las ablaciones (agente_errp.py no se toca)."""
    umbral_cambio = None        # None: el chequeo predictivo cuenta p_errp > 0.5, como hoy
    sin_fiabilidad = False      # True: P_hat no escala el LLR calibrado por la fiabilidad
    transformar = None          # p_errp -> p_errp' antes de usarla (recalibracion candidata)
    prior_al_cambio = None      # p. ej. 0.5: al detectar un cambio, el prior de error vuelve ahi
    prior_prediccion = None     # igual, pero solo si el cambio lo detecto el chequeo predictivo (ErrP)
    prior_momentos = None       # alfa: prior = tasa de detecciones corregida por sens y espec vivas

    def prob_error(self, p_errp, sens=None, espec=None, fiabilidad=1.0):
        if self.transformar is not None and self.cfg.salida_detector == 'calibrada':
            p_errp = self.transformar(p_errp)
        return super().prob_error(p_errp, sens, espec, 1.0 if self.sin_fiabilidad else fiabilidad)

    def actualizar(self, p_errp, artefacto=False, fiabilidad=1.0, sens=None, espec=None, peso=None):
        self._p_errp = p_errp
        info = super().actualizar(p_errp, artefacto, fiabilidad, sens, espec, peso)
        if info['cambio'] and self.prior_al_cambio is not None:
            self.prior = self.prior_al_cambio
        if info['cambio'] == 'prediccion' and self.prior_prediccion is not None:
            self.prior = max(self.prior, self.prior_prediccion)
        if self.prior_momentos and np.isfinite(info['P_hat']) and sens is not None:
            # tasa de detecciones = sens * pi + (1 - espec) * (1 - pi)  ->  pi
            self._tasa = getattr(self, '_tasa', self.prior * sens + (1 - self.prior) * (1 - espec))
            self._tasa += self.prior_momentos * (float(p_errp > self.umbral_momentos) - self._tasa)
            self.prior = float(np.clip((self._tasa - (1 - espec)) / (sens + espec - 1), 0.05, 0.95))
        return info

    def _detectar_cambio(self, dec, detectado, sens, espec):
        if self.umbral_cambio is not None:
            detectado = self._p_errp > self.umbral_cambio
        return super()._detectar_cambio(dec, detectado, sens, espec)


def lazo(mod, semilla_lazo, salida='calibrada', umbral_cambio=False, sin_fiabilidad=False, transformar=None,
         prior_al_cambio=None, prior_momentos=None, alfa_prior=None, coadaptar=False, prior_prediccion=None,
         evidencia_nula=False):
    """Una sesion del lazo (estatico + adaptativo con perturbacion). Devuelve una fila por paso.
    coadaptar: el detector se re-entrena en el lazo como en BackendReal (DetectorCoadaptativo); aqui
    el re-entrenamiento termina antes del paso siguiente (en vivo tarda unos pasos).
    evidencia_nula: CONTROL NEGATIVO. El agente recibe siempre la tasa base como salida del detector
    (LLR = 0: el ErrP no le dice nada), pero la compuerta de confianza sigue abierta porque se
    alimenta del detector real. Un agente que aprende del ErrP no debe recuperarse asi."""
    dec, det = copy.deepcopy(mod['dec']), mod['det']
    m = Mundo2(semilla_lazo)
    m.avanzar(4.0)
    rng = np.random.default_rng(semilla_lazo + 7)       # metas y latencias: iguales en cada variante
    sens, espec = (float(np.clip(v, 0.51, 0.99)) for v in (det.sens, det.espec))
    cfg = ConfigAgente(modo='bayes', sens=sens, espec=espec, salida_detector=salida,
                       p_error_calibracion=det.p_error_cal,
                       umbral_detector=det.umbral if salida == 'binaria' else 0.5)
    if alfa_prior is not None:
        cfg.alfa_prior = alfa_prior
    ag = Agente(dec.w0, dec.c0, cfg)
    ag.umbral_cambio = det.umbral if umbral_cambio else None
    ag.sin_fiabilidad = sin_fiabilidad
    ag.transformar = transformar(mod) if transformar else None
    ag.prior_al_cambio, ag.prior_momentos, ag.umbral_momentos = prior_al_cambio, prior_momentos, det.umbral
    ag.prior_prediccion = prior_prediccion
    conf = ConfianzaDetector(sens, espec)
    umbral, co = [det.umbral], None
    if coadaptar:
        def al_cambiar(nuevo):                           # lo que hace Orquestador.detector_cambiado
            s_, e_ = (float(np.clip(v, 0.51, 0.99)) for v in (nuevo.sens, nuevo.espec))
            umbral[0] = nuevo.umbral
            ag.cfg.sens, ag.cfg.espec, ag.cfg.p_error_calibracion = s_, e_, nuevo.p_error_cal
            conf.rebase(s_, e_)
        co = hw.DetectorCoadaptativo(det, mod['X_cal'], mod['y_cal'], config.COADAPTAR_CADA,
                                     config.COADAPTAR_PRUEBA, al_cambiar=al_cambiar)
    filas, despl, angulo = [], 0.0, config.PUNTO_MEDIO
    for bloque, n, aprender in (('estatico', PASOS_EST, False), ('adaptativo', PASOS_ADA, True)):
        orden = []
        for t in range(n):
            if t % config.PASOS_ENSAYO == 0:
                if not orden:
                    orden = [int(v) for v in rng.permutation([1, -1])]
                meta = orden.pop()
                m.cer.meta = meta
                angulo = config.PUNTO_MEDIO               # Orquestador.iniciar_ensayo: centrado
                m.avanzar(config.VENTANA_MI + config.ESPERA_PRIMER_PASO_S)
            if bloque == 'adaptativo' and t == PERTURBA_EN:
                despl = -config.PERTURBACION_LOGITS
            beta_antes = ag.beta
            d = ag.decidir(dec.phi(m.ventana_mi()), despl)
            erroneo = d.direccion != meta
            antes, angulo = angulo, float(np.clip(angulo + d.delta, 0, 1))
            quieto = TOPES is not None and abs(angulo - antes) < config.PASO_VISIBLE - 1e-9
            if quieto and TOPES == 'ignorar':            # como Orquestador.paso: ni epoca ni aprendizaje
                rng.uniform(*config.LATENCIA_MECANICA_SIM_MS)
                m.avanzar(config.EPOCA_ERRP[1])
                p, art = float('nan'), True
            else:
                e = mover(m, erroneo, rng, visible=not quieto)
                p, art = det.p_error(e), det.artefacto(e)
                if co is not None:                       # como BackendReal.errp
                    co.observar(e, erroneo, p, art)
                    co.esperar()
                    det = co.actual
            detectado = bool(p > umbral[0])
            fiab = conf(erroneo, detectado, not art)     # igual que Orquestador.paso
            sv, ev = conf.vivo()
            fb = conf.fiabilidad_bruta
            prior = ag.prior
            info = ag.actualizar(det.p_error_cal if evidencia_nula else p, art, fb, sv, ev,
                                 peso=fiab if aprender else 0.0)
            filas.append({'bloque': bloque, 't': t, 'post': despl != 0, 'meta': meta, 'z': d.z,
                          'p_prima': d.p_prima, 'erroneo': bool(erroneo), 'sombra': d.direccion_sombra != meta,
                          'p_errp': p, 'detectado': bool(detectado), 'art': bool(art), 'P_hat': info['P_hat'],
                          'quieto': bool(quieto), 'angulo': angulo, 'congelado': bool(conf.congelado),
                          'beta_antes': beta_antes, 'beta': ag.beta, 'var': ag.var, 'peso': fiab, 'fb': fb,
                          'sens': sv, 'espec': ev, 'prior': prior, 'cambio': info['cambio'],
                          'version': co.version if co is not None else 1,
                          'llr_cal': logit(p) - logit(det.p_error_cal),
                          'llr_bin': float(np.log(sv / (1 - ev)) if detectado else np.log((1 - sv) / ev))})
    return filas


def resumen(filas):
    """Lo que paso tras la perturbacion en una sesion."""
    post = [f for f in filas if f['post']]
    b0 = post[0]['beta_antes']
    db = np.array([f['beta'] - b0 for f in post])
    rec = np.flatnonzero(db >= META_BETA)
    v = post[:VENTANA]
    vistos = [f for f in v if np.isfinite(f['p_errp'])]        # sin los pasos que no se leyeron (TOPES)
    return {'db20': db[19], 'db57': db[VENTANA - 1], 'pasos70': int(rec[0]) + 1 if rec.size else None,
            'err': np.mean([f['erroneo'] for f in v]), 'sombra': np.mean([f['sombra'] for f in v]),
            'cambios': sum(f['cambio'] != '' for f in post), 'db': db,
            'ba': 0.5 * (np.mean([f['detectado'] for f in vistos if f['erroneo']] or [np.nan])
                         + 1 - np.mean([f['detectado'] for f in vistos if not f['erroneo']] or [np.nan]))}


# ------------------------------------------------------------ recalibraciones candidatas
def platt_en_logit(mod):
    """Platt sobre el LOGIT de la salida de hoy, ajustado con validacion cruzada en las epocas
    de calibracion (las probabilidades de cada epoca vienen de un modelo que no la vio)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    if 'p_cv' not in mod:
        det = mod['det']
        cv = StratifiedKFold(4, shuffle=True, random_state=0)
        mod['p_cv'] = cross_val_predict(det._pipe(), mod['X_cal'], mod['y_cal'], method='predict_proba', cv=cv)[:, 1]
    x = np.array([logit(p) for p in mod['p_cv']])[:, None]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(x, mod['y_cal'])
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    return lambda p: float(sigmoide(a * logit(p) + b))


BASE = 'calibrada (hoy)'
VARIANTES = {
    BASE: {},
    'binaria (como el simulador)': {'salida': 'binaria'},
    'calibrada + chequeo predictivo con el umbral del detector': {'umbral_cambio': True},
    'calibrada + sin escalar el LLR por la fiabilidad': {'sin_fiabilidad': True},
    'calibrada + Platt en logit': {'transformar': platt_en_logit},
    'calibrada + prior rapido (alfa 0.10)': {'alfa_prior': 0.10},
    'calibrada + prior a 0.5 al detectar un cambio': {'prior_al_cambio': 0.5},
    'calibrada + prior por momentos (alfa 0.10)': {'prior_momentos': 0.10},
    'binaria + prior a 0.5 al detectar un cambio': {'salida': 'binaria', 'prior_al_cambio': 0.5},
    'calibrada + prior a 0.5 solo si el cambio es por prediccion': {'prior_prediccion': 0.5},
    'calibrada + lo mismo, con el umbral del detector en el chequeo': {'prior_prediccion': 0.5, 'umbral_cambio': True},
    'CONTROL sin evidencia del ErrP: calibrada (hoy)': {'evidencia_nula': True},
    'CONTROL sin evidencia del ErrP: prior a 0.5 al detectar un cambio': {'evidencia_nula': True, 'prior_al_cambio': 0.5},
    'CONTROL sin evidencia del ErrP: prior a 0.5 solo por prediccion': {'evidencia_nula': True, 'prior_prediccion': 0.5},
}


def confiabilidad(p, y, n=8):
    """Por cuantiles de la probabilidad predicha: (media predicha, fraccion observada, n).
    Y la pendiente de calibracion: regresion logistica de y sobre logit(p); 1 = calibrado,
    > 1 = las probabilidades estan comprimidas (poco confiadas)."""
    from sklearn.linear_model import LogisticRegression
    cortes = np.quantile(p, np.linspace(0, 1, n + 1))
    idx = np.clip(np.searchsorted(cortes, p, side='right') - 1, 0, n - 1)
    tabla = [(p[idx == k].mean(), y[idx == k].mean(), int((idx == k).sum())) for k in range(n) if (idx == k).any()]
    x = np.array([logit(v) for v in p])[:, None]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(x, y)
    return tabla, float(lr.coef_[0, 0]), float(lr.intercept_[0])


def correr(sujetos=4, repeticiones=4, salida=print):
    res = {v: [] for v in VARIANTES}
    filas = {v: [] for v in VARIANTES}
    mods = []
    for s in range(sujetos):
        mod = preparar(s, salida)
        mods.append(mod)
        for r in range(repeticiones):
            for v, kw in VARIANTES.items():
                f = lazo(mod, 1000 + 100 * s + r, **kw)
                res[v].append(resumen(f)); filas[v] += [dict(x, sujeto=s, rep=r) for x in f]
        salida(f'  sujeto {s}: ' + '; '.join(
            f"{v.split(' (')[0][:22]} db20 {np.mean([x['db20'] for x in res[v][-repeticiones:]]):+.2f}"
            for v in list(VARIANTES)[:2]))
    return res, filas, mods


def sesiones(filas, variante):
    """Por sesion (sujeto, repeticion) de una variante: lo que paso tras la perturbacion."""
    claves = sorted({(f['sujeto'], f['rep']) for f in filas[variante]})
    out = []
    for s_, r_ in claves:
        f = [x for x in filas[variante] if x['sujeto'] == s_ and x['rep'] == r_]
        post = [x for x in f if x['post']]
        db = np.array([x['beta'] - post[0]['beta_antes'] for x in post])
        rec = np.flatnonzero(db[:VENTANA] >= META_BETA)
        out.append({'db': db, 'db20': db[19], 'rec': int(rec[0]) + 1 if rec.size else None,
                    'err': np.mean([x['erroneo'] for x in post[:VENTANA]]),
                    'sombra': np.mean([x['sombra'] for x in post[:VENTANA]]),
                    'antes': np.mean([x['erroneo'] for x in f if x['bloque'] == 'adaptativo' and not x['post']]),
                    'tarde': np.mean([x['erroneo'] for x in post[VENTANA:]])})
    return out


def errores_post(filas, variante=BASE):
    """Los errores reales (sin artefacto) de los 57 pasos tras perturbar, con su evidencia."""
    return [f for f in filas[variante] if f['post'] and f['t'] < PERTURBA_EN + VENTANA
            and f['erroneo'] and not f['art']]


def informe(regimenes=('actual', 'ayer', 'debil'), salida=print):
    """Tabla pareada por regimen (con lo que guardo main) y la figura."""
    datos = {}
    for reg in regimenes:
        ruta = CACHE / f'corrida_{reg}.pkl'
        if ruta.exists():
            datos[reg] = pickle.loads(ruta.read_bytes())['filas']
    for reg, filas in datos.items():
        base = sesiones(filas, 'calibrada (hoy)')
        n = len(base)
        salida(f"\nRegimen '{reg}': {n} sesiones por variante; diferencia pareada contra 'calibrada (hoy)' (media +- EE)")
        for v in filas:
            x = sesiones(filas, v)
            de = np.array([a['err'] - b['err'] for a, b in zip(x, base)])
            dd = np.array([a['db20'] - b['db20'] for a, b in zip(x, base)])
            rec = [a['rec'] for a in x if a['rec'] is not None]
            salida(f"  {v:58s} error 2 min {np.mean([a['err'] for a in x]):.3f} ({de.mean():+.3f} +- "
                   f"{de.std(ddof=1) / np.sqrt(n):.3f}) | d beta 20 {np.mean([a['db20'] for a in x]):+.2f} "
                   f"({dd.mean():+.2f} +- {dd.std(ddof=1) / np.sqrt(n):.2f}) | recuperan en {VENTANA} pasos "
                   f"{len(rec)}/{n} (mediana {np.median(rec):.0f}) | error antes {np.mean([a['antes'] for a in x]):.2f}, "
                   f"despues de los 2 min {np.mean([a['tarde'] for a in x]):.2f}")
        f = errores_post(filas)
        ph = np.array([x['P_hat'] for x in f])
        pb = np.array([float(sigmoide(logit(x['prior']) + x['llr_bin'])) for x in f])
        p05 = np.array([float(sigmoide(x['fb'] * x['llr_cal'])) for x in f])
        salida(f"  errores reales tras perturbar (n={len(f)}), mismas epocas: prior del agente "
               f"{np.mean([x['prior'] for x in f]):.2f} -> P_hat pasa de 0.5 solo si el LLR supera "
               f"{np.mean([-logit(x['prior']) for x in f]):.2f}")
        salida(f"     P_hat > 0.5: calibrada {np.mean(ph > 0.5):.0%} (media {ph.mean():.2f}) | binaria "
               f"{np.mean(pb > 0.5):.0%} (media {pb.mean():.2f}) | calibrada con prior 0.5: "
               f"{np.mean(p05 > 0.5):.0%} (media {p05.mean():.2f})")
    if datos:
        salida(f'Figura: {graficar(datos)}')
        if all(SIN_ERRP_HOY in f for f in datos.values()):
            salida(f'Figura: {graficar_control(datos)}')
    return datos


def simulador(n=30, salida=print):
    """Las correcciones candidatas del prior en simulador_lazo.py (salida binaria), con el CONTROL
    NEGATIVO: con un detector sin informacion el agente NO debe recuperarse. Si se recupera, lo
    mueve el supuesto de metas balanceadas (el detector de sesgo), no el ErrP."""
    import simulador_lazo as sl
    variantes = (('antes', {}),
                 ('A: prior a 0.5 con cualquier cambio', {'prior_al_cambio': 0.5}),
                 ('B: prior a 0.5 solo si el cambio es por prediccion', {'prior_prediccion': 0.5}),
                 ('C: prior por momentos (alfa 0.10)', {'prior_momentos': 0.10}),
                 ('D: prior mas rapido (alfa 0.10)', {'alfa_prior': 0.10}))

    def una(semilla, sens, espec, alfa_prior=None, **ganchos):
        mk = lambda r: sl.PilotoSimulado(sens=sens, espec=espec, semilla_sujeto=semilla, semilla_ruido=r)
        w0, c0 = sl.calibrar(mk(10_000 + semilla))
        cfg = ConfigAgente(modo='bayes', sens=sens, espec=espec)
        if alfa_prior:
            cfg.alfa_prior = alfa_prior
        ag = Agente(w0, c0, cfg)
        ag.umbral_momentos = 0.5
        for k, v in ganchos.items():
            setattr(ag, k, v)
        r = sl.simular(ag, mk(semilla + 1), ConfianzaDetector(sens, espec), 600, 300, w_ref=w0)
        rec = np.flatnonzero(r['beta'][300:] >= r['beta'][299] + META_BETA)
        return np.r_[sl.metricas(r, 300)[:3], (rec[0] + 1) * config.CICLO_S if rec.size else np.nan]

    salida(f'\nSimulador ({n} sujetos, salida binaria): correcciones candidatas del prior y control negativo')
    for nombre, sens, espec in (('detector de referencia (sens 0.70, espec 0.90)', 0.70, 0.90),
                                ('detector debil (sens 0.40, espec 0.90)', 0.40, 0.90),
                                ('CONTROL: detector sin informacion (sens 0.12, espec 0.90)', 0.12, 0.90)):
        salida(f'  {nombre}')
        base = None
        for et, kw in variantes:
            M = np.array([una(s_, sens, espec, **kw) for s_ in range(n)])
            base = M if base is None else base
            d = M[:, 1] - base[:, 1]
            rec = int(np.isfinite(M[:, 3]).sum())
            salida(f'    {et:52s} error antes {M[:, 0].mean():.3f} | 2 min {M[:, 1].mean():.3f} ({d.mean():+.3f} +- '
                   f'{d.std(ddof=1) / np.sqrt(n):.3f}) | despues {M[:, 2].mean():.3f} | recuperan {rec:2d}/{n}'
                   + (f' en {np.nanmean(M[:, 3]):.0f} s' if rec else ''))


NOMBRES_REGIMEN = {'actual': 'detector actual\n(BA viva 0.82)', 'ayer': 'detector de ayer\n(0.74)',
                   'debil': 'detector d\u00e9bil\n(0.68)'}
A_CUALQUIERA = 'calibrada + prior a 0.5 al detectar un cambio'
B_PREDICCION = 'calibrada + prior a 0.5 solo si el cambio es por prediccion'
SIN_ERRP_HOY = 'CONTROL sin evidencia del ErrP: calibrada (hoy)'
SIN_ERRP_A = 'CONTROL sin evidencia del ErrP: prior a 0.5 al detectar un cambio'


def graficar(datos, ruta=FIGURA):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    azul, naranja, verde, morado = '#2a78d6', '#eb6834', '#2e9e6b', '#8a56c2'
    tinta, tinta2, rejilla = '#0b0b0b', '#52514e', '#e4e3df'
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.9))
    fig.patch.set_facecolor('#fcfcfb')

    def betas(a, reg, variantes, titulo):
        for v, color, nombre, *estilo in variantes:
            if v not in datos[reg]:
                continue
            db = np.array([x['db'][:VENTANA + 20] for x in sesiones(datos[reg], v)])
            x = np.arange(1, db.shape[1] + 1)
            m, ee = db.mean(0), db.std(0, ddof=1) / np.sqrt(len(db))
            a.fill_between(x, m - ee, m + ee, color=color, alpha=0.15, lw=0)
            a.plot(x, m, color=color, lw=2, label=nombre, ls=estilo[0] if estilo else '-')
        a.axhline(META_BETA, color=tinta2, ls='--', lw=1)
        a.text(VENTANA + 20, META_BETA + 0.05, 'recuperaci\u00f3n del CP4', ha='right', fontsize=8, color=tinta2)
        a.axvline(VENTANA, color=rejilla, lw=6, zorder=0)
        a.set_ylim(-0.6, 2.9)
        a.set_title(titulo, color=tinta)
        a.set_xlabel('pasos tras la perturbaci\u00f3n', color=tinta2)
        a.set_ylabel('cambio de \u03b2 (logits)', color=tinta2)
        a.legend(loc='lower right', frameon=False, fontsize=8)

    variantes = ((BASE, azul, 'calibrada (hoy)'), ('binaria (como el simulador)', naranja, 'binaria (como el simulador)'),
                 (A_CUALQUIERA, verde, 'prior a 0.5 con cualquier cambio'),
                 (B_PREDICCION, morado, 'prior a 0.5 solo si el cambio viene del ErrP'))
    control = ((BASE, azul, 'hoy, con el ErrP'), (SIN_ERRP_HOY, azul, 'hoy, SIN la evidencia del ErrP', '--'),
               (A_CUALQUIERA, verde, 'prior a 0.5 con cualquier cambio, con el ErrP'),
               (SIN_ERRP_A, verde, 'prior a 0.5 con cualquier cambio, SIN el ErrP', '--'))
    if 'debil' in datos:
        betas(ax[0], 'debil', variantes, '\u03b2 tras la perturbaci\u00f3n (detector d\u00e9bil)')
        betas(ax[1], 'debil', control, 'Control negativo: el mismo lazo sin la evidencia del ErrP')

    # Panel 3: P_hat de los errores reales, sobre las mismas epocas
    marcas, k = [], 0.0
    for reg_ in (r for r in ('actual', 'ayer', 'debil') if r in datos):
        f = errores_post(datos[reg_])
        grupos = ((azul, [x['P_hat'] for x in f]),
                  (naranja, [float(sigmoide(logit(x['prior']) + x['llr_bin'])) for x in f]),
                  (verde, [float(sigmoide(x['fb'] * x['llr_cal'])) for x in f]))
        for color, v in grupos:
            ax[2].boxplot([v], positions=[k], widths=0.7, patch_artist=True, showfliers=False,
                          medianprops={'color': tinta}, whiskerprops={'color': tinta2}, capprops={'color': tinta2},
                          boxprops={'facecolor': color, 'alpha': 0.55, 'edgecolor': tinta2})
            ax[2].text(k, 1.03, f'{100 * np.mean(np.array(v) > 0.5):.0f} %', ha='center', fontsize=8, color=tinta2)
            k += 1
        marcas.append((k - 2, reg_))
        k += 0.8
    ax[2].axhline(0.5, color=tinta2, ls='--', lw=1)
    ax[2].set_xticks([p_ for p_, _ in marcas])
    ax[2].set_xticklabels([NOMBRES_REGIMEN.get(r_, r_) for _, r_ in marcas])
    ax[2].set_ylim(-0.32, 1.1)
    ax[2].set_yticks(np.arange(0, 1.01, 0.2))
    ax[2].set_title('P_hat de los errores reales (arriba: cu\u00e1ntos pasan de 0.5)', color=tinta)
    ax[2].set_ylabel('P_hat', color=tinta2)
    for nombre, color in (('calibrada, con el prior del agente', azul), ('binaria, con el prior del agente', naranja),
                          ('calibrada, con prior 0.5', verde)):
        ax[2].bar([0], [0], color=color, alpha=0.55, label=nombre)
    ax[2].legend(loc='lower left', frameon=False, fontsize=8)

    # Panel 4: confiabilidad del detector calibrado
    for reg_, color, nombre in (('actual', azul, 'detector actual'), ('ayer', naranja, 'detector de ayer')):
        patron = 'sujeto_?.pkl' if reg_ == 'actual' else f'sujeto_?_{reg_}.pkl'
        mods = [pickle.loads(r.read_bytes()) for r in sorted(CACHE.glob(patron))]
        if not mods:
            continue
        p_ = np.concatenate([m['p_nuevas'] for m in mods])
        y_ = np.concatenate([m['y_nuevas'] for m in mods])
        tabla, pendiente, _ = confiabilidad(p_, y_)
        ax[3].plot([a for a, _, _ in tabla], [b for _, b, _ in tabla], color=color, lw=2, marker='o', ms=6,
                   label=f'{nombre}: pendiente {pendiente:.2f}')
    ax[3].plot([0, 1], [0, 1], color=tinta2, ls='--', lw=1)
    ax[3].set_xlim(0, 1)
    ax[3].set_ylim(0, 1)
    ax[3].set_title('Confiabilidad del detector calibrado', color=tinta)
    ax[3].set_xlabel('probabilidad de error predicha', color=tinta2)
    ax[3].set_ylabel('fracci\u00f3n de errores observada', color=tinta2)
    ax[3].legend(loc='upper left', frameon=False, fontsize=8)

    for a in ax:
        a.set_facecolor('#fcfcfb')
        a.grid(axis='y', color=rejilla, lw=0.8)
        a.set_axisbelow(True)
        for lado in ('top', 'right'):
            a.spines[lado].set_visible(False)
        for lado in ('left', 'bottom'):
            a.spines[lado].set_color(tinta2)
        a.tick_params(colors=tinta2)
    fig.text(0.01, 0.01, 'Gemelo digital sin LSL, 4 sujetos x 4 lazos por variante (no son datos de una persona). '
             'Perturbaci\u00f3n de 2.4 logits; franja gris: los 57 pasos (2 min) del CP4. Control negativo: el agente recibe '
             'LLR = 0 con la compuerta de confianza abierta; si aprende del ErrP, no debe recuperarse.\nPendiente de calibraci\u00f3n: 1 = calibrado; mayor que 1 = '
             'probabilidades comprimidas hacia la tasa base (1600 \u00e9pocas nuevas por detector).',
             fontsize=8, color=tinta2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return ruta


def ba_viva(filas, variante=BASE):
    """BA del detector en los pasos con movimiento de los 2 min tras perturbar (todas las sesiones)."""
    v = [f for f in filas[variante] if f['post'] and f['t'] < PERTURBA_EN + VENTANA
         and not f.get('quieto') and np.isfinite(f['p_errp'])]
    return 0.5 * (np.mean([f['detectado'] for f in v if f['erroneo']])
                  + 1 - np.mean([f['detectado'] for f in v if not f['erroneo']]))


def graficar_control(datos, ruta=FIGURA_CONTROL, nota=''):
    """La figura del control negativo para la presentacion: el agente de hoy con la evidencia
    del ErrP y sin ella (mismas sesiones), con tres calidades de detector."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    azul, naranja = '#2a78d6', '#eb6834'
    tinta, tinta2, rejilla = '#0b0b0b', '#52514e', '#e4e3df'
    regs = [r for r in ('actual', 'ayer', 'debil') if r in datos]
    nombres = {r: f'detector {n}\n(BA {ba_viva(datos[r]):.2f})'
               for r, n in (('actual', 'fuerte'), ('ayer', 'medio'), ('debil', 'd\u00e9bil')) if r in datos}
    con = {r: sesiones(datos[r], BASE) for r in regs}
    sin = {r: sesiones(datos[r], SIN_ERRP_HOY) for r in regs}
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.0), gridspec_kw={'width_ratios': [1.3, 1, 1]})
    fig.patch.set_facecolor('#fcfcfb')

    # Panel 1: beta tras la perturbacion, todas las sesiones juntas
    for d, color, estilo, nombre in ((con, azul, '-', 'con el ErrP'), (sin, naranja, '--', 'sin la evidencia del ErrP')):
        db = np.array([x['db'][:VENTANA + 20] for r in regs for x in d[r]])
        x = np.arange(1, db.shape[1] + 1)
        m, ee = db.mean(0), db.std(0, ddof=1) / np.sqrt(len(db))
        ax[0].fill_between(x, m - ee, m + ee, color=color, alpha=0.18, lw=0)
        ax[0].plot(x, m, color=color, lw=2.5, ls=estilo, label=f'{nombre} ({len(db)} sesiones)')
    ax[0].axhline(META_BETA, color=tinta2, ls='--', lw=1)
    ax[0].text(VENTANA + 20, META_BETA + 0.05, 'recuperaci\u00f3n (CP4)', ha='right', fontsize=9, color=tinta2)
    ax[0].axhline(config.PERTURBACION_LOGITS, color=tinta2, ls=':', lw=1)
    ax[0].text(1, config.PERTURBACION_LOGITS + 0.05, 'correcci\u00f3n ideal', ha='left', fontsize=9, color=tinta2)
    ax[0].axvline(VENTANA, color=rejilla, lw=6, zorder=0)
    ax[0].set_ylim(-0.1, 2.8)
    ax[0].set_title('Lo que aprende el agente tras la perturbaci\u00f3n', color=tinta)
    ax[0].set_xlabel('pasos tras la perturbaci\u00f3n', color=tinta2)
    ax[0].set_ylabel('correcci\u00f3n \u03b2 aprendida (logits)', color=tinta2)
    ax[0].legend(loc='lower right', frameon=False, fontsize=10)

    # Paneles 2 y 3: sesiones recuperadas y error, por detector
    x = np.arange(len(regs))
    ancho = 0.36
    for k, (d, color, nombre) in enumerate(((con, azul, 'con el ErrP'), (sin, naranja, 'sin la evidencia del ErrP'))):
        rec = [sum(a['rec'] is not None for a in d[r]) for r in regs]
        n = [len(d[r]) for r in regs]
        pos = x + (k - 0.5) * ancho
        ax[1].bar(pos, rec, ancho * 0.92, color=color, label=nombre)
        for px, v, nn in zip(pos, rec, n):
            ax[1].text(px, v + 0.3, f'{v}/{nn}', ha='center', fontsize=11, color=tinta, fontweight='bold')
        err = np.array([[a['err'] for a in d[r]] for r in regs])
        ax[2].bar(pos, err.mean(1), ancho * 0.92, color=color, label=nombre,
                  yerr=err.std(1, ddof=1) / np.sqrt(err.shape[1]), capsize=4, error_kw={'ecolor': tinta2, 'lw': 1.2})
        for px, v in zip(pos, err.mean(1)):
            ax[2].text(px, 0.02, f'{v:.2f}', ha='center', fontsize=10, color='white', fontweight='bold')
    sombra = np.mean([a['sombra'] for r in regs for a in con[r]])
    ax[2].axhline(sombra, color=tinta2, ls='--', lw=1.2)
    ax[2].text(x[-1] + 0.5, sombra + 0.008, f'decoder sin aprender ({sombra:.2f})', ha='right', fontsize=9, color=tinta2)
    ax[1].set_ylim(0, 18.5)
    ax[1].set_yticks([0, 4, 8, 12, 16])
    ax[1].set_title('Sesiones que se recuperan en 2 min (de 16)', color=tinta)
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=10)
    ax[2].set_ylim(0, 0.56)
    ax[2].set_title('Error en los 2 min tras la perturbaci\u00f3n', color=tinta)
    ax[2].set_ylabel('fracci\u00f3n de pasos err\u00f3neos', color=tinta2)
    for a in ax[1:]:
        a.set_xticks(x)
        a.set_xticklabels([nombres[r] for r in regs])
    for a in ax:
        a.set_facecolor('#fcfcfb')
        a.grid(axis='y', color=rejilla, lw=0.8)
        a.set_axisbelow(True)
        for lado in ('top', 'right'):
            a.spines[lado].set_visible(False)
        for lado in ('left', 'bottom'):
            a.spines[lado].set_color(tinta2)
        a.tick_params(colors=tinta2)
    fig.suptitle('Control negativo: sin la evidencia del ErrP el agente no se recupera', color=tinta,
                 fontsize=15, fontweight='bold', x=0.01, ha='left')
    fig.text(0.01, 0.01, 'Gemelo digital sin LSL: 3 detectores x 4 sujetos x 4 lazos (no son datos de una persona). Control: las '
             'mismas sesiones, pero el agente recibe siempre la tasa base como salida del detector (LLR = 0);\nla compuerta '
             'de confianza sigue abierta. Perturbaci\u00f3n de 2.4 logits; franja gris: los 57 pasos (2 min) del CP4; '
             'recuperaci\u00f3n = \u03b2 al 70 % de la perturbaci\u00f3n. Barras de error: 1 error est\u00e1ndar.'
             + ('\n' + nota if nota else ''),
             fontsize=8.5, color=tinta2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.12 if nota else 0.09, 1, 0.93))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return ruta


def main():
    global REGIMEN
    if len(sys.argv) > 1 and sys.argv[1] == 'informe':     # con lo ya corrido: tabla pareada, simulador y figura
        informe()
        return simulador()
    sujetos = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    REGIMEN = sys.argv[3] if len(sys.argv) > 3 else 'actual'
    salida = lambda *a: print(*a, flush=True)
    r = REGIMENES[REGIMEN]
    salida(f"Regimen '{REGIMEN}': ErrP del gemelo {r['errp']} uV, theta {r['theta']} uV, detector "
           + ('elegido por la calibracion' if r['candidatos'] is None else ', '.join(r['candidatos'])))
    res, filas, mods = correr(sujetos, reps, salida)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f'corrida_{REGIMEN}.pkl').write_bytes(pickle.dumps({'res': res, 'filas': filas}))
    n = sujetos * reps
    print(f'\n{sujetos} sujetos del gemelo x {reps} lazos = {n} sesiones por variante '
          f'(perturbacion de {config.PERTURBACION_LOGITS} logits; meta del CP4: beta +{META_BETA:.2f})')
    print(f"{'variante':58s} {'d beta 20':>10s} {'d beta 57':>10s} {'recupera':>9s} {'pasos':>6s} "
          f"{'agente':>7s} {'sombra':>7s} {'cambios':>8s} {'BA viva':>8s}")
    for v, rr in res.items():
        rec = [x['pasos70'] for x in rr if x['pasos70'] is not None]
        m = lambda k: np.nanmean([x[k] for x in rr])
        print(f"{v:58s} {m('db20'):+10.2f} {m('db57'):+10.2f} {len(rec):6d}/{n:<2d} "
              f"{(np.median(rec) if rec else float('nan')):6.0f} {m('err'):7.2f} {m('sombra'):7.2f} "
              f"{m('cambios'):8.1f} {m('ba'):8.2f}")
    hoy = filas['calibrada (hoy)']
    en = lambda k: np.mean([f['prior'] for f in hoy if f['bloque'] == 'adaptativo' and f['t'] == PERTURBA_EN + k])
    print(f'prior de error del agente (variante de hoy): al perturbar {en(0):.2f}, a los 20 pasos {en(20):.2f}, '
          f'a los 57 {en(57):.2f}; tasa real de error en esos 57 pasos: '
          f'{np.mean([x["err"] for x in res["calibrada (hoy)"]]):.2f}')

    # (2) P_hat y LLR tras la perturbacion, en las epocas de la variante de hoy
    post = [f for f in filas['calibrada (hoy)'] if f['post'] and f['t'] < PERTURBA_EN + VENTANA and not f['art']]
    print(f'\nEvidencia por paso en los {VENTANA} pasos tras perturbar (variante de hoy, {len(post)} epocas sin artefacto):')
    for nombre, sel in (('errores', True), ('aciertos', False)):
        g = [f for f in post if f['erroneo'] == sel]
        q = lambda k: '{:+.2f} [{:+.2f}, {:+.2f}]'.format(*np.quantile([f[k] for f in g], [0.5, 0.1, 0.9]))
        ap = [f['fb'] * f['llr_cal'] for f in g]
        print(f'  {nombre:8s} n={len(g):4d}  p_errp {q("p_errp")}  P_hat {q("P_hat")}  LLR calibrado {q("llr_cal")}  '
              f'(x fiabilidad: {np.median(ap):+.2f})  LLR binario {q("llr_bin")}')
    bin_post = [f for f in filas['binaria (como el simulador)'] if f['post'] and f['t'] < PERTURBA_EN + VENTANA and not f['art']]
    for nombre, sel in (('errores', True), ('aciertos', False)):
        g = [f['P_hat'] for f in bin_post if f['erroneo'] == sel]
        print(f'  P_hat en la variante binaria, {nombre}: mediana {np.median(g):.2f}, media {np.mean(g):.2f}')

    # (3) confiabilidad del detector calibrado en epocas nuevas
    p = np.concatenate([m['p_nuevas'] for m in mods]); y = np.concatenate([m['y_nuevas'] for m in mods])
    tabla, pendiente, corte = confiabilidad(p, y)
    print(f'\nConfiabilidad del detector calibrado ({len(p)} epocas nuevas, tasa de error {y.mean():.2f}):')
    print('  predicha -> observada (n): ' + '  '.join(f'{a:.2f} -> {b:.2f} ({c})' for a, b, c in tabla))
    print(f'  rango de p_errp (percentiles 1 y 99): {np.quantile(p, 0.01):.2f} a {np.quantile(p, 0.99):.2f}; '
          f'pendiente de calibracion {pendiente:.2f} (1 = calibrado, > 1 = comprimido)')
    for m_, s in zip(mods, range(sujetos)):
        _, pend, _ = confiabilidad(m_['p_nuevas'], m_['y_nuevas'])
        d = m_['det']
        det_n = m_['p_nuevas'] > d.umbral
        print(f'  sujeto {s}: calibracion sens {d.sens:.2f} espec {d.espec:.2f} | en epocas nuevas sens '
              f'{det_n[m_["y_nuevas"] == 1].mean():.2f} espec {1 - det_n[m_["y_nuevas"] == 0].mean():.2f} | '
              f'umbral {d.umbral:.2f} | pendiente {pend:.2f} | p_errp maxima {m_["p_nuevas"].max():.2f}')


if __name__ == '__main__':
    main()
