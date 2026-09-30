"""Simulador del lazo MI -> ortesis -> ErrP -> agente, sin casco ni persona.

Modela: rasgos de imaginacion motora con ruido, un detector de ErrP imperfecto
(sensibilidad/especificidad), artefactos, la perturbacion de la demo, fallas del
detector y un detalle que casi nadie simula: si el paso es tan chico que no se
ve, el cerebro no genera ErrP.

Uso
  python simulador_lazo.py                       figura + tabla con 20 sujetos
  python simulador_lazo.py --sens 0.72 --espec 0.91
  python simulador_lazo.py --falla_detector      prueba el congelamiento
  python simulador_lazo.py --barrido             elige eta para el modo 'fijo'
"""
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression

import config
from agente_errp import AgenteErrP, ConfigAgente, ConfianzaDetector


class PilotoSimulado:
    def __init__(self, dim=4, separacion=1.0, sens=config.SENS, espec=config.ESPEC,
                 p_artefacto=0.05, paso_visible=config.PASO_VISIBLE, deriva=0.0,
                 semilla_sujeto=0, semilla_ruido=1):
        u = np.random.default_rng(semilla_sujeto).normal(size=dim)
        self.u = u / np.linalg.norm(u)
        self.rng = np.random.default_rng(semilla_ruido)
        self.dim, self.sep = dim, separacion
        self.sens0, self.espec0 = sens, espec
        self.sens, self.espec = sens, espec
        self.p_art, self.paso_visible, self.deriva = p_artefacto, paso_visible, deriva
        self.desplazamiento = np.zeros(dim)

    def rasgos(self, y):
        """y = +1 cerrar, -1 relajar."""
        if self.deriva:
            self.desplazamiento += self.deriva * self.rng.normal(size=self.dim)
        return y * self.sep * self.u + self.desplazamiento + self.rng.normal(size=self.dim)

    def errp(self, erroneo, delta):
        """(p_errp, artefacto). Consume siempre los mismos numeros aleatorios."""
        r = self.rng.random(3)
        b = self.rng.beta(2, 2)
        if r[0] < self.p_art:
            return float(r[2]), True
        visible = abs(delta) >= self.paso_visible
        p_det = self.sens if (erroneo and visible) else 1 - self.espec
        detecta = r[1] < p_det
        return float(0.5 + 0.5 * b if detecta else 0.5 * b), False

    def perturbar(self, w, logits):
        w = np.asarray(w, float)
        self.desplazamiento = self.desplazamiento - logits * w / np.dot(w, w)

    def detector_degradado(self, si):
        self.sens, self.espec = (0.45, 0.60) if si else (self.sens0, self.espec0)


def calibrar(piloto, n=80):
    """Simula la calibracion de imaginacion motora: devuelve (w0, c0)."""
    y = np.repeat([1, -1], n // 2)
    X = np.array([piloto.rasgos(v) for v in y])
    clf = LogisticRegression().fit(X, (y > 0).astype(int))
    return clf.coef_[0], float(clf.intercept_[0])


def simular(agente, piloto, confianza=None, pasos=600, t_perturb=300,
            logits=config.PERTURBACION_LOGITS, falla=None, w_ref=None):
    campos = ['error', 'error_sombra', 'angulo', 'meta', 'beta', 'varianza', 'P_hat',
              'fiab', 'youden', 'cambio', 'explorando']
    reg = {k: [] for k in campos}
    angulo, orden, y = 0.5, [], 1
    for t in range(pasos):
        if t % config.PASOS_ENSAYO == 0:
            if not orden:
                orden = list(piloto.rng.permutation([1, -1]))
            y = int(orden.pop())
        if t == t_perturb:
            piloto.perturbar(w_ref if w_ref is not None else agente.w0, logits)
        if falla:
            piloto.detector_degradado(falla[0] <= t < falla[1])
        dec = agente.decidir(piloto.rasgos(y))
        angulo = float(np.clip(angulo + dec.delta, 0, 1))
        erroneo = dec.direccion != y
        p_errp, art = piloto.errp(erroneo, dec.delta)
        if confianza is not None:
            fiab = confianza(erroneo, p_errp > 0.5, not art)
            sens, espec = confianza.vivo()
            yd = confianza.youden
        else:
            fiab, sens, espec, yd = 1.0, None, None, np.nan
        info = agente.actualizar(p_errp, art, fiab, sens, espec)
        for k, v in (('error', erroneo), ('error_sombra', dec.direccion_sombra != y),
                     ('angulo', angulo), ('meta', 1.0 if y > 0 else 0.0),
                     ('beta', info['beta']), ('varianza', info['varianza']),
                     ('P_hat', info['P_hat']), ('fiab', fiab), ('youden', yd),
                     ('cambio', {'': 0, 'sesgo': 1, 'prediccion': 2}[info['cambio']]),
                     ('explorando', dec.explorando)):
            reg[k].append(float(v))
    return {k: np.array(v) for k, v in reg.items()}


def correr(modo, semilla, sens=config.SENS, espec=config.ESPEC, eta=config.ETA_BETA,
           pasos=600, falla=None, deriva=0.0, usar_sesgo=True, confianza_viva=True, **cfg_extra):
    mk = lambda r: PilotoSimulado(sens=sens, espec=espec, deriva=deriva,
                                  semilla_sujeto=semilla, semilla_ruido=r)
    w0, c0 = calibrar(mk(10_000 + semilla))
    cfg = ConfigAgente(modo=modo, sens=sens, espec=espec, eta_beta=eta,
                       usar_sesgo=usar_sesgo, **cfg_extra)
    ag = AgenteErrP(w0, c0, cfg)
    conf = ConfianzaDetector(sens, espec) if (modo != 'estatico' and confianza_viva) else None
    return simular(ag, mk(semilla + 1), conf, pasos, pasos // 2, falla=falla, w_ref=w0)


def metricas(r, t_p):
    """[error antes, error primeros 2 min, error despues, error de seguimiento |cierre - meta|]"""
    k = t_p + int(round(config.RECUPERACION_MAX_S / config.CICLO_S))
    e = r['error']
    return np.array([e[:t_p].mean(), e[t_p:k].mean(), e[k:].mean(),
                     np.abs(r['angulo'] - r['meta']).mean()])


def tabla(modos, semillas, **kw):
    print(f"\n{'modo':10s} {'antes':>14s} {'primeros 2 min':>16s} {'despues':>14s} {'seguimiento':>15s}"
          f"   ({len(semillas)} sujetos)")
    res = {}
    for m in modos:
        M = np.array([metricas(correr(m, s, **kw), kw.get('pasos', 600) // 2) for s in semillas])
        res[m] = M
        mu, sd = M.mean(0), M.std(0)
        print(f"{m:10s} " + '  '.join(f'{a:.3f} +- {b:.3f}' for a, b in zip(mu, sd)))
    return res


def graficar(resultados, t_p, falla=None, ruta=None):
    import matplotlib.pyplot as plt
    movil = lambda x, w=30: np.convolve(x, np.ones(w) / w, mode='valid')
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for nombre, r in resultados.items():
        ax[0].plot(np.arange(len(movil(r['error']))) + 30, movil(r['error']), label=nombre)
        ax[1].plot(r['angulo'], lw=0.8, label=nombre)
    ax[1].plot(next(iter(resultados.values()))['meta'], 'k--', lw=0.5, label='meta')
    r = resultados.get('bayes')
    if r is not None:
        x = np.arange(len(r['beta']))
        sd = np.sqrt(r['varianza'])
        ax[2].plot(r['beta'], label='beta')
        ax[2].fill_between(x, r['beta'] - 2 * sd, r['beta'] + 2 * sd, alpha=0.2, label='+- 2 sd')
        ax[2].axhline(config.PERTURBACION_LOGITS, color='g', ls=':', lw=0.8, label='beta ideal')
        for t, tipo in zip(np.flatnonzero(r['cambio']), r['cambio'][r['cambio'] > 0]):
            ax[2].axvline(t, color='purple' if tipo == 1 else 'brown', lw=0.7, alpha=0.7)
        exp = r['explorando'] > 0
        ax[2].scatter(x[exp], np.full(exp.sum(), -1.5), s=4, color='orange', label='paso informativo')
        ax[3].plot(r['youden'], label='Youden vivo del detector')
        ax[3].plot(r['fiab'], label='fiabilidad (0 = congelado)', alpha=0.8)
        ax[3].set_ylim(-0.1, 1.1)
    for a in ax:
        a.axvline(t_p, color='r', ls=':')
        if falla:
            a.axvspan(*falla, color='orange', alpha=0.15)
        a.legend(loc='upper left', fontsize=7, ncol=4)
    ax[0].set_ylabel('error (movil 30)')
    ax[1].set_ylabel('cierre (0-1)')
    ax[2].set_ylabel('beta')
    ax[3].set_ylabel('detector')
    ax[3].set_xlabel('paso   (morado: cambio por sesgo, cafe: por prediccion)')
    fig.tight_layout()
    config.RESULTADOS.mkdir(exist_ok=True)
    fig.savefig(ruta or config.RESULTADOS / 'simulacion.png', dpi=120)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sens', type=float, default=config.SENS)
    ap.add_argument('--espec', type=float, default=config.ESPEC)
    ap.add_argument('--eta', type=float, default=config.ETA_BETA)
    ap.add_argument('--semilla', type=int, default=0)
    ap.add_argument('--semillas', type=int, default=20)
    ap.add_argument('--pasos', type=int, default=600)
    ap.add_argument('--deriva', type=float, default=0.0, help='deriva lenta de los rasgos por paso')
    ap.add_argument('--falla_detector', action='store_true')
    ap.add_argument('--barrido', action='store_true')
    ap.add_argument('--sin_sesgo', action='store_true',
                    help='apaga el detector de sesgo (uso con metas no balanceadas)')
    ap.add_argument('--sin_grafica', action='store_true')
    a = ap.parse_args()
    t_p = a.pasos // 2
    falla = (t_p + 120, t_p + 180) if a.falla_detector else None
    kw = dict(sens=a.sens, espec=a.espec, pasos=a.pasos, falla=falla, deriva=a.deriva,
              usar_sesgo=not a.sin_sesgo)

    if a.barrido:
        print('Barrido de eta (modo fijo) contra el modo bayes:')
        for eta in (0.1, 0.2, 0.3, 0.5, 0.8):
            M = np.array([metricas(correr('fijo', s, eta=eta, **kw), t_p) for s in range(a.semillas)])
            print(f'  eta={eta:.1f}: ' + '  '.join(f'{v:.3f}' for v in M.mean(0)))
        tabla(['bayes'], range(a.semillas), **kw)
        return

    tabla(['estatico', 'fijo', 'bayes'], range(a.semillas), eta=a.eta, **kw)
    if not a.sin_grafica:
        import matplotlib.pyplot as plt
        res = {m: correr(m, a.semilla, eta=a.eta, **kw) for m in ('estatico', 'fijo', 'bayes')}
        graficar(res, t_p, falla)
        print(f"\nbeta final (bayes): {res['bayes']['beta'][-1]:.2f}  (ideal ~{config.PERTURBACION_LOGITS})")
        print(f'Figura: {config.RESULTADOS / "simulacion.png"}')
        plt.show()


if __name__ == '__main__':
    main()
