"""Analisis de potencia del IIC (Tarea 2, EXPLORATORIO) con las sesiones del gemelo que ya se
simularon: estudios/datos/iic_gemelo.csv (176 sesiones de estudios/embodiment_gemelo.py y de
la medicion del sesgo). No genera EEG nuevo.

Modelo: el IIC de una sesion es su d verdadera mas ruido, con d = kappa * embodiment y
desviacion sigma = sqrt(c / ajenos), el error de una d de Cohen cuando los propios correctos
crecen en proporcion a los ajenos (c teorico = 1 + ajenos / propios). kappa y c se ajustan a
las sesiones.

Preguntas:
  1. Cuantos movimientos ajenos hacen falta para que el intervalo del 90 % mida +-0.2.
  2. Cuantos, para que el Spearman entre embodiment (0.2 / 0.5 / 0.8) e IIC de un estudio como
     el nuestro (16 sujetos por nivel) salga significativo (una cola, alfa 0.05).

SOLO vale para el gemelo: el tamano del efecto (kappa) lo programamos nosotros
(cerebro_sintetico.ATENUACION_MAX). En una persona se desconoce.

Uso: python estudios/potencia_iic.py
"""
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
import numpy as np
from scipy.stats import norm, rankdata, t as t_student

import config

DATOS = Path(__file__).resolve().parent / 'datos' / 'iic_gemelo.csv'
FIGURA = config.RAIZ / 'docs' / 'figuras' / 'potencia_iic.png'
NIVELES = np.array([0.2, 0.5, 0.8])
Z90 = norm.ppf(0.95)                       # intervalo del 90 %
MEDIO_ANCHO = 0.2
ALFA, POTENCIA = 0.05, 0.80
# en el banco: 1 de cada 10 es ajeno y el 80 % de los propios son correctos
PROPIOS_POR_AJENO = (config.AJENOS_CADA - 1) * 0.8
C_TEORICO = 1 + 1 / PROPIOS_POR_AJENO


def cargar():
    with open(DATOS, newline='') as f:
        filas = list(csv.DictReader(f))
    col = lambda k, tipo=float: np.array([tipo(x[k]) for x in filas])
    return {'lote': col('lote', str), 'sujeto': col('sujeto', int), 'ajenos': col('ajenos'),
            'emb': col('embodiment'), 'iic': col('iic')}


def ajustar(d):
    """kappa (pendiente por el origen, ponderada por los ajenos) y c (varianza * ajenos)."""
    w = d['ajenos']
    kappa = float(np.sum(w * d['emb'] * d['iic']) / np.sum(w * d['emb'] ** 2))
    res = d['iic'] - kappa * d['emb']
    return kappa, float(np.mean(w * res ** 2)), res


def spearman(x, Y):
    """Spearman de x contra cada fila de Y, y su p de una cola (rho > 0) por la aproximacion t."""
    rx = rankdata(x)
    RY = rankdata(Y, axis=1)
    rx = (rx - rx.mean()) / rx.std()
    RY = (RY - RY.mean(1, keepdims=True)) / RY.std(1, keepdims=True)
    rho = (RY * rx).mean(1)
    n = len(x)
    return rho, t_student.sf(rho * np.sqrt((n - 2) / np.maximum(1e-12, 1 - rho ** 2)), n - 2)


def estudios_simulados(kappa, c, ajenos, sujetos, n=4000, semilla=0):
    """n estudios de `sujetos` sesiones por nivel: (rho, p) de cada uno y fraccion de sujetos
    con los tres niveles en orden."""
    rng = np.random.default_rng(semilla)
    x = np.repeat(NIVELES, sujetos)
    Y = kappa * x + rng.normal(0, np.sqrt(c / ajenos), (n, x.size))
    rho, p = spearman(x, Y)
    Y3 = Y.reshape(n, 3, sujetos)
    return rho, p, float(np.mean((Y3[:, 0] < Y3[:, 1]) & (Y3[:, 1] < Y3[:, 2])))


def observado(d, ajenos):
    """Spearman medido en el estudio de 16 sujetos con ese numero de ajenos, e intervalo del
    90 % remuestreando sujetos."""
    m = (d['ajenos'] == ajenos) & (d['lote'] == 'embodiment_gemelo.py') & (d['emb'] > 0)
    suj = np.unique(d['sujeto'][m])
    tabla = np.array([[d['iic'][m & (d['sujeto'] == s) & (d['emb'] == e)][0] for e in NIVELES] for s in suj])
    x = np.repeat(NIVELES, len(suj))
    rho = spearman(x, tabla.T.reshape(1, -1))[0][0]
    rng = np.random.default_rng(0)
    boot = [spearman(x, tabla[rng.integers(0, len(suj), len(suj))].T.reshape(1, -1))[0][0] for _ in range(2000)]
    return float(rho), float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))


def analizar(sujetos=16):
    d = cargar()
    kappa, c, res = ajustar(d)
    rng = np.random.default_rng(1)
    idx = [rng.integers(0, len(res), len(res)) for _ in range(2000)]
    kappas = [ajustar({k: v[i] for k, v in d.items()})[0] for i in idx]
    r = {'n': len(res), 'kappa': kappa, 'kappa_ic': tuple(np.quantile(kappas, [0.05, 0.95])), 'c': c,
         'sd': {int(a): (float(res[d['ajenos'] == a].std()), float(np.sqrt(C_TEORICO / a)))
                for a in np.unique(d['ajenos'])},
         'ajenos_intervalo': c * (Z90 / MEDIO_ANCHO) ** 2,
         'ajenos_intervalo_teorico': C_TEORICO * (Z90 / MEDIO_ANCHO) ** 2,
         'observado': {a: observado(d, a) for a in (12, 30)}, 'sujetos': sujetos}
    n_cola = 3 * sujetos
    t_crit = t_student.ppf(1 - ALFA, n_cola - 2)
    r['rho_critico'] = float(t_crit / np.sqrt(n_cola - 2 + t_crit ** 2))
    rejilla = np.unique(np.round(np.geomspace(6, 600, 60)).astype(int))
    curvas = {}
    for s in (sujetos, 2 * sujetos):
        sim = [estudios_simulados(kappa, c, a, s) for a in rejilla]
        curvas[s] = {'potencia': np.array([np.mean(p < ALFA) for _, p, _ in sim]),
                     'rho': np.array([np.quantile(rho, [0.05, 0.5, 0.95]) for rho, _, _ in sim]),
                     'orden': np.array([o for _, _, o in sim])}
    r.update(rejilla=rejilla, curvas=curvas)
    primero = lambda y, nivel: int(rejilla[np.argmax(y >= nivel)]) if np.any(y >= nivel) else None
    r['ajenos_potencia'] = {s: (primero(curvas[s]['potencia'], 0.5), primero(curvas[s]['potencia'], POTENCIA))
                            for s in curvas}
    return r


def graficar(r, ruta=FIGURA):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    azul, naranja = '#2a78d6', '#eb6834'
    tinta, tinta2, rejilla = '#0b0b0b', '#52514e', '#e4e3df'
    x, s = r['rejilla'], r['sujetos']
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_facecolor('#fcfcfb')

    # Panel 1: medio ancho del intervalo del 90 %
    ax[0].plot(x, Z90 * np.sqrt(r['c'] / x), color=azul, lw=2, label='modelo ajustado a las sesiones')
    for a, (sd, _) in r['sd'].items():
        ax[0].scatter(a, Z90 * sd, s=60, color=tinta, zorder=3)
    ax[0].scatter([], [], s=60, color=tinta, label='medido (1.64 × desviación entre sesiones)')
    n02 = r['ajenos_intervalo']
    ax[0].axhline(MEDIO_ANCHO, color=tinta2, ls='--', lw=1)
    ax[0].axvline(n02, color=naranja, lw=1.5)
    ax[0].text(n02 * 1.08, 0.62, f'±{MEDIO_ANCHO} con {n02:.0f} ajenos\n(~{n02 * config.AJENOS_CADA:.0f} pasos del lazo)',
               color=naranja, fontsize=9, va='top')
    ax[0].axvline(12, color=rejilla, lw=6, zorder=0)
    ax[0].set_title('Medio ancho del intervalo del 90 % del IIC', color=tinta)
    ax[0].set_ylabel('medio ancho (unidades de d)', color=tinta2)
    ax[0].set_ylim(0, 0.8)
    ax[0].legend(loc='upper right', frameon=False, fontsize=9)

    # Panel 2: Spearman esperado y potencia
    c = r['curvas'][s]
    ax[1].fill_between(x, c['rho'][:, 0], c['rho'][:, 2], color=azul, alpha=0.15, lw=0)
    ax[1].plot(x, c['rho'][:, 1], color=azul, lw=2, label=f'Spearman esperado ({s} sujetos por nivel; banda: 90 %)')
    ax[1].axhline(r['rho_critico'], color=tinta2, ls='--', lw=1)
    ax[1].text(x[-1], r['rho_critico'] + 0.015, f'significativo (una cola, {ALFA})', ha='right',
               fontsize=8, color=tinta2)
    for a, (rho, lo, hi) in r['observado'].items():
        ax[1].errorbar(a, rho, yerr=[[rho - lo], [hi - rho]], color=tinta, marker='o', ms=7, capsize=4,
                       elinewidth=1.2, zorder=3)
    ax[1].errorbar([], [], yerr=[], color=tinta, marker='o', ms=7, capsize=4, lw=0, elinewidth=1.2,
                   label='medido en el gemelo (intervalo del 90 %)')
    n50, n80 = r['ajenos_potencia'][s]
    for n, txt, ha in ((n50, 'potencia 50 %', 'right'), (n80, f'potencia {100 * POTENCIA:.0f} %', 'left')):
        if n:                                   # marcas sobre el eje: no tapan los puntos medidos
            ax[1].scatter(n, -0.4, marker='^', s=110, color=naranja, clip_on=False, zorder=5)
            ax[1].text(n * (0.9 if ha == 'right' else 1.12), -0.33, f'{txt}:\n{n} ajenos', color=naranja,
                       fontsize=9, va='center', ha=ha)
    ax[1].axvline(12, color=rejilla, lw=6, zorder=0)
    ax[1].set_title('Spearman entre embodiment (0.2 / 0.5 / 0.8) e IIC', color=tinta)
    ax[1].set_ylabel('correlación de Spearman', color=tinta2)
    ax[1].set_ylim(-0.4, 1.0)
    ax[1].legend(loc='upper left', frameon=False, fontsize=9)

    for a in ax:
        a.set_facecolor('#fcfcfb')
        a.set_xscale('log')
        a.set_xticks([6, 12, 30, 60, 120, 300, 600])
        a.set_xticklabels(['6', '12', '30', '60', '120', '300', '600'])
        a.minorticks_off()
        a.set_xlabel('movimientos ajenos por sesión (1 de cada 10 pasos del lazo)', color=tinta2)
        a.grid(axis='y', color=rejilla, lw=0.8)
        a.set_axisbelow(True)
        for lado in ('top', 'right'):
            a.spines[lado].set_visible(False)
        for lado in ('left', 'bottom'):
            a.spines[lado].set_color(tinta2)
        a.tick_params(colors=tinta2)
    fig.text(0.01, 0.01, f'Gemelo digital, {r["n"]} sesiones ya simuladas (no son datos de una persona; el tamaño '
             f'del efecto lo programamos nosotros: d = {r["kappa"]:.2f} × embodiment).\nModelo: IIC = d + ruido '
             f'de desviación √({r["c"]:.2f} / ajenos). Franja gris: los 12 ajenos de la demo '
             f'({12 * config.AJENOS_CADA} pasos adaptativos).', fontsize=8, color=tinta2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return ruta


def main():
    r = analizar()
    s = r['sujetos']
    print(f'{r["n"]} sesiones del gemelo ya simuladas ({DATOS.name})')
    print(f'  d = kappa * embodiment, kappa = {r["kappa"]:.2f} [{r["kappa_ic"][0]:.2f}, {r["kappa_ic"][1]:.2f}] (90 %)')
    print(f'  varianza del IIC = c / ajenos, c = {r["c"]:.2f} (teorico {C_TEORICO:.2f})')
    for a, (sd, teo) in r['sd'].items():
        print(f'     {a:3d} ajenos: desviacion medida {sd:.3f}, teorica {teo:.3f} -> intervalo del 90 %: +-{Z90 * sd:.2f}')
    print(f'1) intervalo de +-{MEDIO_ANCHO}: {r["ajenos_intervalo"]:.0f} ajenos (con c teorico, '
          f'{r["ajenos_intervalo_teorico"]:.0f}) = ~{r["ajenos_intervalo"] * config.AJENOS_CADA:.0f} pasos del lazo adaptativo')
    print(f'2) Spearman significativo (una cola, alfa {ALFA}): rho critico {r["rho_critico"]:.2f} con {3 * s} sesiones')
    for a, (rho, lo, hi) in r['observado'].items():
        print(f'     medido con {a} ajenos: {rho:+.2f} [{lo:+.2f}, {hi:+.2f}]')
    for suj, (n50, n80) in r['ajenos_potencia'].items():
        c = r['curvas'][suj]
        i12, i30 = (int(np.argmin(np.abs(r['rejilla'] - a))) for a in (12, 30))
        print(f'     {suj} sujetos por nivel: potencia 50 % con {n50} ajenos y {100 * POTENCIA:.0f} % con {n80}; '
              f'con 12 ajenos {100 * c["potencia"][i12]:.0f} %, con 30 {100 * c["potencia"][i30]:.0f} %')
    c = r['curvas'][s]
    for a in (12, 30, 60, 120, 300):
        i = int(np.argmin(np.abs(r['rejilla'] - a)))
        print(f'     {r["rejilla"][i]:3d} ajenos: Spearman esperado {c["rho"][i, 1]:+.2f}; '
              f'un sujeto ordena bien los tres niveles el {100 * c["orden"][i]:.0f} % de las veces (azar: 17 %)')
    print(f'Figura: {graficar(r)}')


if __name__ == '__main__':
    main()
