"""Reporte del detector de ErrP al final de CAL_ERRP (5 de octubre): una figura con la curva de
confiabilidad, la curva ROC, el umbral elegido, la sensibilidad, la especificidad y los falsos positivos.

Todo sale de lo que ya calculo DetectorErrP.ajustar sobre las epocas de la calibracion, sin tocar el detector:
  - `p_cv`: la probabilidad de error de cada epoca con un modelo que NO la vio (validacion cruzada de la
    configuracion elegida). Con ella se dibujan la confiabilidad y la ROC.
  - `pred_cv`: las decisiones de la validacion ANIDADA (la configuracion y el umbral se eligen sin ver la
    epoca que se juzga). De ahi salen la sensibilidad, la especificidad y los falsos positivos del
    encabezado: son los honestos. El umbral que se usa en el lazo se eligio sobre `p_cv` completo, asi que
    el punto de operacion que se marca en la ROC (sobre `p_cv`) puede verse algo mejor.

Funciones puras (`calcular`) y la figura (`figura`); `desde_detector` las junta y nunca lanza: un fallo del
reporte no puede tumbar la calibracion. Apagable con orquestador.py --sin-reporte-detector.
"""
from __future__ import annotations

import numpy as np

import config

__all__ = ['calcular', 'figura', 'desde_detector']

TEXTO = {
    'es': {'titulo': 'Reporte del detector de ErrP ({n} épocas de calibración, {e} errores)',
           'conf': 'Curva de confiabilidad', 'x_conf': 'probabilidad de error predicha', 'y_conf': 'fracción real de errores',
           'perfecta': 'calibración perfecta', 'roc': 'Curva ROC', 'x_roc': 'falsos positivos (1 − especificidad)',
           'y_roc': 'sensibilidad', 'op': 'punto de operación', 'op_anidado': 'validación anidada (la del CP3)',
           'azar': 'azar', 'hist': 'Puntajes por época y umbral', 'x_hist': 'probabilidad de error predicha',
           'y_hist': 'épocas', 'aciertos': 'sin error (aciertos)', 'errores': 'con error', 'umbral': 'umbral elegido',
           'resumen': 'Umbral {u:.2f} · sensibilidad {s:.2f} · especificidad {e:.2f} · BA {b:.2f}\n'
                      'Falsos positivos {fp} de {neg} aciertos ({fpr:.0%}) · falsos negativos {fn} de {pos} errores · AUC {auc:.2f} · ECE {ece:.2f}',
           'pie': 'Sensibilidad, especificidad y falsos positivos: validación anidada (el umbral y la configuración se eligen sin ver la época que se juzga). '
                  'Confiabilidad y ROC: probabilidades de validación cruzada; el umbral del lazo se eligió sobre ellas, así que el punto de operación de la ROC puede verse algo mejor.\n'
                  'Detector elegido: {elec}. Si la fuente es el gemelo, no son datos de una persona.'},
    'en': {'titulo': 'ErrP detector report ({n} calibration epochs, {e} errors)',
           'conf': 'Reliability curve', 'x_conf': 'predicted error probability', 'y_conf': 'observed fraction of errors',
           'perfecta': 'perfect calibration', 'roc': 'ROC curve', 'x_roc': 'false positives (1 − specificity)',
           'y_roc': 'sensitivity', 'op': 'operating point', 'op_anidado': 'nested validation (the CP3 one)',
           'azar': 'chance', 'hist': 'Per-epoch scores and threshold', 'x_hist': 'predicted error probability',
           'y_hist': 'epochs', 'aciertos': 'no error (correct)', 'errores': 'error', 'umbral': 'chosen threshold',
           'resumen': 'Threshold {u:.2f} · sensitivity {s:.2f} · specificity {e:.2f} · BA {b:.2f}\n'
                      'False positives {fp} of {neg} correct steps ({fpr:.0%}) · false negatives {fn} of {pos} errors · AUC {auc:.2f} · ECE {ece:.2f}',
           'pie': 'Sensitivity, specificity and false positives: nested validation (the threshold and configuration are chosen without seeing the epoch being judged). '
                  'Reliability and ROC: cross-validated probabilities; the loop threshold was chosen on them, so the ROC operating point can look slightly better.\n'
                  'Chosen detector: {elec}. If the source is the digital twin, these are not data from a person.'},
}


def calcular(p_cv, y, umbral, pred_cv, bins=6):
    """Cifras del reporte. p_cv: probabilidad de error por epoca (fuera de muestra); y: 1 = error;
    pred_cv: decision por epoca de la validacion anidada (None: la de p_cv > umbral)."""
    from sklearn.metrics import roc_auc_score, roc_curve
    p, y = np.asarray(p_cv, dtype=float), np.asarray(y).astype(int)
    if len(p) != len(y) or len(y) == 0 or y.min() == y.max():
        raise ValueError('hacen falta epocas de las dos clases y una probabilidad por epoca')
    pred = (p > umbral).astype(int) if pred_cv is None else np.asarray(pred_cv).astype(int)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    fp, tn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 0)).sum())
    sens, espec = tp / pos, tn / neg
    fpr, tpr, _ = roc_curve(y, p)
    # confiabilidad: casillas con el mismo numero de epocas (con ~120 epocas, las uniformes quedan casi vacias)
    orden = np.argsort(p)
    cajas = [c for c in np.array_split(orden, bins) if len(c)]
    media_p = np.array([p[c].mean() for c in cajas])
    frac = np.array([y[c].mean() for c in cajas])
    n_caja = np.array([len(c) for c in cajas])
    ece = float(np.sum(n_caja * np.abs(media_p - frac)) / len(y))
    pred_u = p > umbral
    return {'n': len(y), 'pos': pos, 'neg': neg, 'umbral': float(umbral), 'sens': sens, 'espec': espec, 'ba': 0.5 * (sens + espec),
            'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 'fpr': fp / neg, 'auc': float(roc_auc_score(y, p)), 'ece': ece,
            'roc': (fpr, tpr), 'confiabilidad': (media_p, frac, n_caja),
            'punto_roc': (float(pred_u[y == 0].mean()), float(pred_u[y == 1].mean())), 'p': p, 'y': y}


def figura(r, ruta, idioma='es', eleccion=''):
    """La figura del reporte (es o en) a partir de calcular()."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = TEXTO[idioma]
    azul, naranja, rojo = '#2a78d6', '#eb6834', '#c0392b'
    tinta, tinta2, rejilla = '#0b0b0b', '#52514e', '#e4e3df'
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.6))
    fig.patch.set_facecolor('#fcfcfb')
    mp, fr, nc = r['confiabilidad']
    ax[0].plot([0, 1], [0, 1], color=tinta2, ls='--', lw=1, label=t['perfecta'])
    ax[0].plot(mp, fr, color=azul, lw=2.5, marker='o', ms=7)
    for x, y_, n in zip(mp, fr, nc):
        ax[0].annotate(f'n={n}', (x, y_), textcoords='offset points', xytext=(5, -13), fontsize=8, color=tinta2)
    ax[0].set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax[0].set_title(t['conf'], color=tinta)
    ax[0].set_xlabel(t['x_conf'], color=tinta2)
    ax[0].set_ylabel(t['y_conf'], color=tinta2)
    ax[0].legend(loc='upper left', frameon=False, fontsize=9)
    fpr, tpr = r['roc']
    ax[1].plot([0, 1], [0, 1], color=tinta2, ls='--', lw=1, label=t['azar'])
    ax[1].plot(fpr, tpr, color=azul, lw=2.5, label=f"ROC (AUC {r['auc']:.2f})")
    ax[1].scatter([r['punto_roc'][0]], [r['punto_roc'][1]], s=90, color=naranja, zorder=5, label=t['op'])
    ax[1].scatter([r['fpr']], [r['sens']], s=90, marker='D', color=rojo, zorder=5, label=t['op_anidado'])
    ax[1].axvline(1 - config.ESPEC_MIN, color=rejilla, lw=5, zorder=0)
    ax[1].set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax[1].set_title(t['roc'], color=tinta)
    ax[1].set_xlabel(t['x_roc'], color=tinta2)
    ax[1].set_ylabel(t['y_roc'], color=tinta2)
    ax[1].legend(loc='lower right', frameon=False, fontsize=9)
    corte = np.linspace(0, 1, 21)
    ax[2].hist(r['p'][r['y'] == 0], bins=corte, color=azul, alpha=0.65, label=t['aciertos'])
    ax[2].hist(r['p'][r['y'] == 1], bins=corte, color=naranja, alpha=0.65, label=t['errores'])
    ax[2].axvline(r['umbral'], color=rojo, lw=2.5, label=f"{t['umbral']} ({r['umbral']:.2f})")
    ax[2].set_title(t['hist'], color=tinta)
    ax[2].set_xlabel(t['x_hist'], color=tinta2)
    ax[2].set_ylabel(t['y_hist'], color=tinta2)
    ax[2].legend(loc='upper center', frameon=False, fontsize=9)
    for a in ax:
        a.set_facecolor('#fcfcfb')
        a.grid(color=rejilla, lw=0.8)
        a.set_axisbelow(True)
        for lado in ('top', 'right'):
            a.spines[lado].set_visible(False)
        a.tick_params(colors=tinta2)
    fig.suptitle(t['titulo'].format(n=r['n'], e=r['pos']), color=tinta, fontsize=15, fontweight='bold', x=0.01, ha='left')
    fig.text(0.01, 0.885, t['resumen'].format(u=r['umbral'], s=r['sens'], e=r['espec'], b=r['ba'], fp=r['fp'], neg=r['neg'], fpr=r['fpr'],
                                              fn=r['fn'], pos=r['pos'], auc=r['auc'], ece=r['ece']), fontsize=11.5, color=tinta, va='top', linespacing=1.6)
    import textwrap
    pie = '\n'.join(textwrap.fill(par, 215) for par in t['pie'].format(elec=eleccion or '-').split('\n'))
    fig.text(0.01, 0.01, pie, fontsize=8, color=tinta2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.84))
    from pathlib import Path
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return Path(ruta)


def desde_detector(det, ruta, idioma='es'):
    """Figura de un DetectorErrP recien ajustado. Devuelve (ruta, cifras) o (None, motivo): nunca lanza."""
    try:
        p, y = getattr(det, 'p_cv', None), getattr(det, 'y_cal', None)
        if p is None or y is None or len(p) != len(y):
            return None, 'el detector no guardo las probabilidades de validacion cruzada (¿viene de la memoria de otra sesion?)'
        r = calcular(p, y, det.umbral, getattr(det, 'pred_cv', None))
        return figura(r, ruta, idioma, getattr(det, 'eleccion', '')), r
    except Exception as e:                       # el reporte nunca tumba la calibracion
        return None, f'{type(e).__name__}: {e}'
