"""Curva de robustez: como se recupera el agente segun la exactitud balanceada (BA)
del detector de ErrP. Solo el simulador de rasgos (simulador_lazo.py), sin LSL.

Para cada BA se fija la especificidad en 0.90 y la sensibilidad en 2*BA - 0.90.
Modos: 'bayes' (el agente) y 'estatico' (sin aprender, la referencia). 600 pasos,
perturbacion de config.PERTURBACION_LOGITS a la mitad (como lo hace `correr`).

Por sujeto:
- error en los primeros 2 min tras perturbar y error despues (`metricas`, indices 1 y 2);
- tiempo de recuperacion: pasos con la perturbacion activa hasta que beta alcanza el
  70 % de la perturbacion, en segundos (config.CICLO_S). Si no llega antes del fin de
  la sesion, el sujeto cuenta como no recuperado y queda fuera de la media y la mediana.

Uso (desde la raiz):  python estudios/curva_robustez.py [semillas]   (por defecto 30)
Figura: docs/figuras/curva_robustez.png. Son cifras del simulador, no de una persona.
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))                  # la raiz del repositorio
import numpy as np
import config
from simulador_lazo import correr, metricas

BAS = (0.65, 0.70, 0.75, 0.80, 0.85)
ESPEC_FIJA = 0.90
FRACCION = 0.70                                # beta debe llegar al 70 % de la perturbacion
PASOS = 600
FIGURA = RAIZ / 'docs' / 'figuras' / 'curva_robustez.png'


def sens_de_ba(ba, espec=ESPEC_FIJA):
    """BA = (sens + espec) / 2  ->  sens = 2*BA - espec."""
    return round(2 * ba - espec, 4)


def tiempo_recuperacion(beta, t_p, fraccion=FRACCION):
    """Segundos desde la perturbacion hasta que beta >= fraccion * perturbacion.

    beta[t_p] ya es el valor tras el primer paso con la perturbacion, asi que el
    indice i cuenta como i + 1 pasos. None si no se recupera en el resto de la sesion."""
    meta = fraccion * config.PERTURBACION_LOGITS
    llego = np.flatnonzero(beta[t_p:] >= meta)
    return None if llego.size == 0 else (llego[0] + 1) * config.CICLO_S


def medir(semillas):
    """{ba: {'sens', 'bayes': M, 'estatico': M, 'rec': [s o None]}} con M (sujetos x 4)."""
    t_p = PASOS // 2
    res = {}
    for ba in BAS:
        sens = sens_de_ba(ba)
        if sens + ESPEC_FIJA <= 1:
            print(f'BA {ba:.2f}: sens {sens:.2f} + espec {ESPEC_FIJA:.2f} <= 1, '
                  'ConfianzaDetector no la acepta; se omite')
            continue
        fila = {'sens': sens, 'rec': []}
        for modo in ('bayes', 'estatico'):
            M = []
            for s in semillas:
                r = correr(modo, s, sens=sens, espec=ESPEC_FIJA, pasos=PASOS)
                M.append(metricas(r, t_p))
                if modo == 'bayes':
                    fila['rec'].append(tiempo_recuperacion(r['beta'], t_p))
            fila[modo] = np.array(M)
        res[ba] = fila
        print(f'  BA {ba:.2f} lista', flush=True)
    return res


def imprimir(res, n):
    print(f'\nCurva de robustez (simulador, {n} sujetos; espec fija {ESPEC_FIJA:.2f}, '
          f'perturbacion {config.PERTURBACION_LOGITS} logits, recuperacion = beta al '
          f'{FRACCION:.0%})')
    print(f"{'BA':>5s} {'sens':>5s} | {'err 2 min bayes':>16s} {'err 2 min estat':>16s} | "
          f"{'despues bayes':>15s} {'despues estat':>15s} | {'t rec media':>15s} "
          f"{'mediana':>8s} {'no rec':>7s}")
    pm = lambda x: f'{x.mean():.3f} +- {x.std():.3f}'
    for ba, f in res.items():
        rec = np.array([t for t in f['rec'] if t is not None])
        no_rec = sum(t is None for t in f['rec'])
        t_med = f'{rec.mean():6.1f} +- {rec.std():5.1f}' if rec.size else '     -'
        t_mdn = f'{np.median(rec):8.1f}' if rec.size else '       -'
        print(f'{ba:5.2f} {f["sens"]:5.2f} | {pm(f["bayes"][:, 1]):>16s} '
              f'{pm(f["estatico"][:, 1]):>16s} | {pm(f["bayes"][:, 2]):>15s} '
              f'{pm(f["estatico"][:, 2]):>15s} | {t_med:>15s} {t_mdn} {no_rec:>4d}/{n}')
    print('tiempos en s; media y sd solo entre los recuperados; +- = 1 sd entre sujetos')


def graficar(res, n, ruta=FIGURA):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    azul, naranja = '#2a78d6', '#eb6834'          # bayes, estatico
    tinta, tinta2, rejilla = '#0b0b0b', '#52514e', '#e4e3df'
    bas = np.array(list(res))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_facecolor('#fcfcfb')

    # Panel 1: error en los 2 min tras perturbar
    for modo, color, dx, nombre in (('bayes', azul, -0.004, 'agente (bayes)'),
                                    ('estatico', naranja, 0.004, 'est\u00e1tico (sin aprender)')):
        y = np.array([res[b][modo][:, 1].mean() for b in bas])
        sd = np.array([res[b][modo][:, 1].std() for b in bas])
        ax[0].errorbar(bas + dx, y, yerr=sd, color=color, lw=2, marker='o', ms=7,
                       capsize=4, elinewidth=1.2, label=nombre)
    ax[0].set_title('Error en los 2 min tras la perturbaci\u00f3n', color=tinta)
    ax[0].set_ylabel('fracci\u00f3n de pasos err\u00f3neos', color=tinta2)
    ax[0].set_ylim(bottom=0)
    ax[0].legend(loc='lower left', frameon=False, fontsize=9)

    # Panel 2: tiempo de recuperacion del agente
    rng = np.random.default_rng(0)
    for b in bas:
        rec = np.array([t for t in res[b]['rec'] if t is not None])
        no_rec = sum(t is None for t in res[b]['rec'])
        if rec.size:
            ax[1].scatter(b + rng.uniform(-0.008, 0.008, rec.size), rec, s=14,
                          color=azul, alpha=0.30, linewidths=0, zorder=2)
            ax[1].errorbar(b, rec.mean(), yerr=rec.std(), color=azul, marker='o', ms=8,
                           capsize=5, lw=0, elinewidth=1.5, zorder=3)
            ax[1].scatter(b, np.median(rec), marker='_', s=260, color=tinta, zorder=4)
        if no_rec:
            ax[1].annotate(f'{no_rec}/{n} no\nrecuperados', (b, 1.0), xycoords=('data', 'axes fraction'),
                           ha='center', va='top', fontsize=8, color=tinta2)
    ax[1].axhline(config.RECUPERACION_MAX_S, color=tinta2, ls='--', lw=1)
    ax[1].text(bas[-1] + 0.028, config.RECUPERACION_MAX_S, '2 min (CP4)', ha='right',
               va='bottom', fontsize=8, color=tinta2)
    ax[1].scatter([], [], color=azul, alpha=0.3, s=14, label='un sujeto')
    ax[1].errorbar([], [], yerr=[], color=azul, marker='o', lw=0, elinewidth=1.5, capsize=5,
                   label='media \u00b1 1 sd')
    ax[1].scatter([], [], marker='_', s=260, color=tinta, label='mediana')
    ax[1].legend(loc='center right', frameon=False, fontsize=9)
    ax[1].set_title(f'Tiempo de recuperaci\u00f3n del agente (\u03b2 al {FRACCION:.0%})',
                    color=tinta)
    ax[1].set_ylabel('segundos tras la perturbaci\u00f3n', color=tinta2)
    ax[1].set_ylim(bottom=0)

    for a in ax:
        a.set_facecolor('#fcfcfb')
        a.set_xlabel('exactitud balanceada (BA) del detector de ErrP', color=tinta2)
        a.set_xticks(bas)
        a.set_xlim(bas[0] - 0.03, bas[-1] + 0.03)
        a.axvline(config.BA_MIN, color=rejilla, lw=6, zorder=0)
        a.grid(axis='y', color=rejilla, lw=0.8)
        a.set_axisbelow(True)
        for lado in ('top', 'right'):
            a.spines[lado].set_visible(False)
        for lado in ('left', 'bottom'):
            a.spines[lado].set_color(tinta2)
        a.tick_params(colors=tinta2)
    fig.text(0.01, 0.01, f'Simulador, {n} sujetos (no son datos de una persona). '
             f'Especificidad fija {ESPEC_FIJA:.2f}, sensibilidad = 2\u00b7BA \u2212 '
             f'{ESPEC_FIJA:.2f}; perturbaci\u00f3n de {config.PERTURBACION_LOGITS} logits '
             f'a la mitad de {PASOS} pasos de {config.CICLO_S} s.\nBarras: \u00b11 sd entre '
             f'sujetos. Franja gris: BA m\u00ednima del CP3 ({config.BA_MIN}).',
             fontsize=8, color=tinta2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return ruta


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    res = medir(range(n))
    imprimir(res, n)
    print(f'\nFigura: {graficar(res, n)}')


if __name__ == '__main__':
    main()
