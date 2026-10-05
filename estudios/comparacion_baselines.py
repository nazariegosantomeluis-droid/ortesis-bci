"""Comparacion final con baselines, para la presentacion (5 de octubre). SOLO el gemelo, no una persona.

Cuatro formas de usar el mismo decoder en el mismo lazo, mismas sesiones y mismas semillas (la sesion
k del sujeto s usa la semilla 1000 + 100 s + k en todos los metodos, asi que el piloto, las metas, el
detector y la perturbacion son identicos):

  estatico   el decoder calibrado sin corregir (modo 'estatico': el agente no aprende)
  eta fijo   el agente con eta constante (config.ETA_BETA = 0.3), sin varianza ni deteccion de cambio
  bayes      el agente de la demo: Kalman sobre beta, P_hat con la confiabilidad viva del detector
  sin ErrP   el control negativo: el mismo agente bayes, pero recibe siempre la tasa base como salida
             del detector (LLR = 0: el ErrP no le dice nada)

Lazo de estudios/agente_lento.py con los topes del recorrido y el modo 'ignorar' (el del orquestador
de hoy): bloque estatico de 30 pasos y bloque adaptativo de 120, con la perturbacion de 2.4 logits en
el paso 40 del adaptativo. Detector actual (calibracion real: elige canales y vistas); los otros dos
detectores del estudio del agente lento (de ayer y debil) van en la tabla. 4 sujetos x 4 lazos = 16
sesiones por metodo. Recuperacion: beta al 70 % de la perturbacion en los 57 pasos (2 min) del CP4.

Salidas (es y en): docs/figuras/comparacion_baselines_{es,en}.png y docs/comparacion_baselines_{es,en}.md.

Uso: python estudios/comparacion_baselines.py [sujetos] [repeticiones]
     python estudios/comparacion_baselines.py informe      tabla y figura con lo ya corrido
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')        # un proceso por tarea, un hilo cada uno
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import config
import agente_lento as al

CACHE = config.RESULTADOS / 'comparacion_baselines'
DOCS = config.RAIZ / 'docs'
REGIMENES = ('actual', 'ayer', 'debil')
METODOS = {                                  # clave: argumentos de al.lazo
    'estatico': {'cfg_agente': {'modo': 'estatico'}},
    'fijo': {'cfg_agente': {'modo': 'fijo'}},
    'bayes': {},
    'sin_errp': {'evidencia_nula': True},
}
COLORES = {'estatico': '#7f7f7f', 'fijo': '#2a9d5c', 'bayes': '#2a78d6', 'sin_errp': '#eb6834'}
TEXTO = {
    'es': {
        'metodo': {'estatico': 'Estático (sin aprender)', 'fijo': 'Eta fija (0.3)', 'bayes': 'Bayes (el nuestro)',
                   'sin_errp': 'Sin ErrP (control negativo)'},
        'titulo': 'Comparación con baselines: el agente bayes se recupera mejor con el ErrP, y sin él no se recupera',
        'p1': 'Lo que aprende el agente tras la perturbación', 'x1': 'pasos tras la perturbación',
        'y1': 'corrección β aprendida (logits)', 'meta': 'recuperación (CP4)', 'ideal': 'corrección ideal',
        'p2': 'Error en los 2 min tras la perturbación', 'y2': 'fracción de pasos erróneos',
        'p3': 'Sesiones que se recuperan en 2 min', 'de': 'de',
        'pie': 'Gemelo digital sin LSL, detector actual, {n} sesiones por método (4 sujetos × 4 lazos, mismas semillas), con los topes del '
               'recorrido. Perturbación de 2.4 logits; recuperación = β al 70 % de la perturbación en 2 min (57 pasos).\n'
               'Banda y barras de error: 1 error estándar entre sesiones. No son datos de una persona.',
        'cols': ['Método', 'Error antes de perturbar', 'Error en los 2 min tras perturbar', 'Se recuperan', 'Pasos hasta recuperar (mediana)'],
        'tit_tabla': 'Comparación con baselines (gemelo digital, {n} sesiones por método, mismas semillas)',
        'det': {'actual': 'detector fuerte', 'ayer': 'detector medio', 'debil': 'detector débil'},
        'nota': ['**Es el gemelo digital, no una persona.** Error = fracción de pasos en que la dirección elegida no fue la meta, media ± error estándar entre sesiones.',
                 'Recuperación: β llega al 70 % de la perturbación (2.4 logits) dentro de los 57 pasos (2 min) del CP4; los pasos son los del lazo, no segundos. «—» = ninguna sesión se recuperó.',
                 'Métodos: *estático* = el decoder calibrado sin corregir; *eta fija* = el agente con eta constante 0.3; *bayes* = el agente de la demo; '
                 '*sin ErrP* = el agente bayes recibiendo siempre la tasa base como salida del detector (LLR = 0), el control negativo.',
                 'Mismas semillas: la sesión k del sujeto s usa la semilla 1000 + 100 s + k en todos los métodos (mismo piloto, metas, detector y perturbación). '
                 'Reproducir: `python estudios/comparacion_baselines.py`.'],
        'otros': 'Con los otros dos detectores (error en los 2 min tras perturbar y sesiones que se recuperan):',
    },
    'en': {
        'metod': None,
        'metodo': {'estatico': 'Static (no learning)', 'fijo': 'Fixed eta (0.3)', 'bayes': 'Bayes (ours)',
                   'sin_errp': 'No ErrP (negative control)'},
        'titulo': 'Comparison with baselines: the Bayesian agent recovers best with the ErrP, and not without it',
        'p1': 'What the agent learns after the perturbation', 'x1': 'steps after the perturbation',
        'y1': 'learned correction β (logits)', 'meta': 'recovery (CP4)', 'ideal': 'ideal correction',
        'p2': 'Error in the 2 min after the perturbation', 'y2': 'fraction of wrong steps',
        'p3': 'Sessions that recover within 2 min', 'de': 'of',
        'pie': 'Digital twin without LSL, current detector, {n} sessions per method (4 subjects × 4 loops, same seeds), with the travel limits. '
               '2.4-logit perturbation; recovery = β reaches 70 % of the perturbation within 2 min (57 steps).\n'
               'Band and error bars: 1 standard error across sessions. Not data from a person.',
        'cols': ['Method', 'Error before the perturbation', 'Error in the 2 min after it', 'Recovered', 'Steps to recover (median)'],
        'tit_tabla': 'Comparison with baselines (digital twin, {n} sessions per method, same seeds)',
        'det': {'actual': 'strong detector', 'ayer': 'medium detector', 'debil': 'weak detector'},
        'nota': ['**This is the digital twin, not a person.** Error = fraction of steps where the chosen direction was not the goal, mean ± standard error across sessions.',
                 'Recovery: β reaches 70 % of the perturbation (2.4 logits) within the 57 steps (2 min) of CP4; steps are loop steps, not seconds. “—” = no session recovered.',
                 'Methods: *static* = the calibrated decoder with no correction; *fixed eta* = the agent with constant eta 0.3; *Bayes* = the demo agent; '
                 '*no ErrP* = the Bayesian agent always receiving the base rate as the detector output (LLR = 0), the negative control.',
                 'Same seeds: session k of subject s uses seed 1000 + 100 s + k for every method (same pilot, goals, detector and perturbation). '
                 'Reproduce: `python estudios/comparacion_baselines.py`.'],
        'otros': 'With the other two detectors (error in the 2 min after the perturbation and sessions that recover):',
    },
}


# ------------------------------------------------------------ correr
def una(tarea):
    """Un sujeto de un regimen: los cuatro metodos, las mismas sesiones."""
    reg, sujeto, reps = tarea
    al.REGIMEN, al.TOPES = reg, 'ignorar'
    mod = al.preparar(sujeto, salida=lambda *a: None)
    filas = {k: [] for k in METODOS}
    for r in range(reps):
        for k, kw in METODOS.items():
            filas[k] += [dict(x, sujeto=sujeto, rep=r) for x in al.lazo(mod, 1000 + 100 * sujeto + r, **kw)]
    return reg, filas


def correr(sujetos=4, reps=4):
    datos = {reg: {k: [] for k in METODOS} for reg in REGIMENES}
    tareas = [(reg, s, reps) for reg in REGIMENES for s in range(sujetos)]
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, len(tareas))) as ex:
        for reg, filas in ex.map(una, tareas):
            for k, f in filas.items():
                datos[reg][k] += f
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / 'corrida.pkl').write_bytes(pickle.dumps({'sujetos': sujetos, 'reps': reps, 'datos': datos}))
    return datos


# ------------------------------------------------------------ medir
def metricas(filas):
    """Por metodo (lista de filas de sus sesiones): cifras de la tabla y de la figura, una por sesion."""
    ses = al.sesiones({'m': filas}, 'm')
    rec = [x['rec'] for x in ses if x['rec'] is not None]
    err = np.array([x['err'] for x in ses])
    antes = np.array([x['antes'] for x in ses])
    ee = lambda v: float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return {'n': len(ses), 'antes': float(antes.mean()), 'antes_ee': ee(antes), 'err': float(err.mean()), 'err_ee': ee(err),
            'recuperan': len(rec), 'mediana': float(np.median(rec)) if rec else None,
            'sombra': float(np.mean([x['sombra'] for x in ses])),
            'db': np.array([x['db'][:al.VENTANA + 20] for x in ses])}


def tabla(m, idioma):
    """Tabla en Markdown (es o en) a partir de metricas() de los cuatro metodos."""
    t = TEXTO[idioma]
    n = next(iter(m.values()))['n']
    lineas = ['| ' + ' | '.join(t['cols']) + ' |', '|' + '---|' * len(t['cols'])]
    for k in METODOS:
        x = m[k]
        lineas.append(f"| {t['metodo'][k]} | {x['antes']:.3f} ± {x['antes_ee']:.3f} | {x['err']:.3f} ± {x['err_ee']:.3f} | "
                      f"{x['recuperan']}/{x['n']} | {'—' if x['mediana'] is None else format(x['mediana'], '.0f')} |")
    return lineas, n


def informe_md(datos, idioma, ruta=None):
    t = TEXTO[idioma]
    m = {reg: {k: metricas(datos[reg][k]) for k in METODOS} for reg in datos}
    lineas, n = tabla(m['actual'], idioma)
    md = [f"# {t['tit_tabla'].format(n=n)}", '', *lineas, '']
    md += [f'- {x}' for x in t['nota']]
    otros = [r for r in ('ayer', 'debil') if r in m]
    if otros:
        md += ['', t['otros'], '', '| | ' + ' | '.join(t['metodo'][k] for k in METODOS) + ' |', '|---|' + '---|' * len(METODOS)]
        for r in otros:
            md.append(f"| {t['det'][r]} | " + ' | '.join(f"{m[r][k]['err']:.3f}; {m[r][k]['recuperan']}/{m[r][k]['n']}" for k in METODOS) + ' |')
    ruta = ruta or DOCS / f'comparacion_baselines_{idioma}.md'
    ruta.write_text('\n'.join(md) + '\n', encoding='utf-8')
    return ruta, m


def figura(m, idioma, ruta=None):
    """La figura de la presentacion (es o en): beta aprendida, error y sesiones recuperadas por metodo."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = TEXTO[idioma]
    m = m['actual']
    tinta, tinta2, rejilla = '#0b0b0b', '#52514e', '#e4e3df'
    n = next(iter(m.values()))['n']
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={'width_ratios': [1.35, 1, 1]})
    fig.patch.set_facecolor('#fcfcfb')
    for k in METODOS:
        db = m[k]['db']
        x = np.arange(1, db.shape[1] + 1)
        med, ee = db.mean(0), db.std(0, ddof=1) / np.sqrt(len(db))
        ax[0].fill_between(x, med - ee, med + ee, color=COLORES[k], alpha=0.18, lw=0)
        ax[0].plot(x, med, color=COLORES[k], lw=2.5, ls='--' if k == 'sin_errp' else '-', label=t['metodo'][k])
    ax[0].axhline(al.META_BETA, color=tinta2, ls='--', lw=1)
    ax[0].text(al.VENTANA + 20, al.META_BETA + 0.05, t['meta'], ha='right', fontsize=9, color=tinta2)
    ax[0].axhline(config.PERTURBACION_LOGITS, color=tinta2, ls=':', lw=1)
    ax[0].text(1, config.PERTURBACION_LOGITS + 0.05, t['ideal'], ha='left', fontsize=9, color=tinta2)
    ax[0].axvline(al.VENTANA, color=rejilla, lw=6, zorder=0)
    ax[0].set_ylim(-0.15, 3.3)
    ax[0].set_title(t['p1'], color=tinta)
    ax[0].set_xlabel(t['x1'], color=tinta2)
    ax[0].set_ylabel(t['y1'], color=tinta2)
    ax[0].legend(loc='upper left', frameon=False, fontsize=9.5, ncol=2, columnspacing=1.2)
    x = np.arange(len(METODOS))
    for i, k in enumerate(METODOS):
        ax[1].bar(i, m[k]['err'], 0.7, color=COLORES[k], yerr=m[k]['err_ee'], capsize=4, error_kw={'ecolor': tinta2, 'lw': 1.2})
        ax[1].text(i, 0.02, f"{m[k]['err']:.2f}", ha='center', fontsize=11, color='white', fontweight='bold')
        ax[2].bar(i, m[k]['recuperan'], 0.7, color=COLORES[k])
        ax[2].text(i, m[k]['recuperan'] + 0.3, f"{m[k]['recuperan']}/{m[k]['n']}", ha='center', fontsize=11, color=tinta, fontweight='bold')
    ax[1].set_ylim(0, 0.62)
    ax[1].set_title(t['p2'], color=tinta)
    ax[1].set_ylabel(t['y2'], color=tinta2)
    ax[2].set_ylim(0, n + 2.5)
    ax[2].set_yticks(range(0, n + 1, 4))
    ax[2].set_title(t['p3'], color=tinta)
    for a in ax[1:]:
        a.set_xticks(x)
        a.set_xticklabels([t['metodo'][k].replace(' (', '\n(') for k in METODOS], fontsize=9)
    for a in ax:
        a.set_facecolor('#fcfcfb')
        a.grid(axis='y', color=rejilla, lw=0.8)
        a.set_axisbelow(True)
        for lado in ('top', 'right'):
            a.spines[lado].set_visible(False)
        for lado in ('left', 'bottom'):
            a.spines[lado].set_color(tinta2)
        a.tick_params(colors=tinta2)
    fig.suptitle(t['titulo'], color=tinta, fontsize=15, fontweight='bold', x=0.01, ha='left')
    fig.text(0.01, 0.01, t['pie'].format(n=n), fontsize=8.5, color=tinta2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    ruta = ruta or DOCS / 'figuras' / f'comparacion_baselines_{idioma}.png'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return ruta


def informe(salida=print):
    d = pickle.loads((CACHE / 'corrida.pkl').read_bytes())
    for idioma in ('es', 'en'):
        ruta, m = informe_md(d['datos'], idioma)
        salida(f'{ruta}\n{figura(m, idioma)}')
    lineas, n = tabla(m['actual'], 'es')
    salida('\n'.join(lineas))
    return m


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'informe':
        return informe()
    sujetos = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    correr(sujetos, reps)
    informe()


if __name__ == '__main__':
    main()
