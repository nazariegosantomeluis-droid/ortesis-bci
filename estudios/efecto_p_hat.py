"""Efecto de calcular P_hat con la fiabilidad real del detector cuando el agente no
aprende (bloque estatico y aprendizaje congelado), con salida 'calibrada'.
Solo el simulador de rasgos (simulador_lazo.py), sin LSL ni el gemelo.

Variantes de la llamada a AgenteErrP.actualizar:
- vieja: fiabilidad 0 cuando no se aprende (estatico o congelado). Con salida
  'calibrada' P_hat queda igual al prior y el prior no se mueve.
- nueva: fiabilidad = confianza.fiabilidad_bruta (P_hat informativo) y
  peso = fiab si se aprende, si no 0 (como el orquestador desde 2178d8d). El prior
  se adapta hacia P_hat tambien en el bloque estatico.

Sesion por sujeto: bloque estatico de 60 pasos sin aprender, bloque adaptativo de
300 pasos con la perturbacion de config.PERTURBACION_LOGITS en su paso 100
(piloto.perturbar, como simular). Metas en ensayos de config.PASOS_ENSAYO pasos con
orden balanceado. Las dos variantes ven los mismos numeros aleatorios del piloto.

Dos maneras de entregar p_errp al agente (p_error_calibracion = 0.3 en ambas):
- cruda: la salida de PilotoSimulado.errp en [0, 1] tal cual. Es una aproximacion:
  no esta calibrada (sobrestima el error en pasos correctos), y con ella el prior
  sube en vez de bajar.
- exacta: el posterior verdadero del simulador a prior 0.3, que solo depende de si
  hubo deteccion: sigma(logit(0.3) + log LR), LR = sens/(1-espec) o (1-sens)/espec.
  Con salida calibrada de verdad el prior sigue la tasa real de error; es el caso
  de la sospecha (en el estatico el decoder casi no se equivoca y el prior baja).

Complementaria: los mismos bloques que el backend real del orquestador (estatico 30,
adaptativo 120, perturbacion en su paso 40).

Uso (desde la raiz):  python estudios/efecto_p_hat.py [semillas]   (por defecto 30)
Son cifras del simulador, no de una persona.
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))                  # la raiz del repositorio
import numpy as np
import config
from agente_errp import AgenteErrP, ConfigAgente, ConfianzaDetector, sigmoide, logit
from simulador_lazo import PilotoSimulado, calibrar

DETECTORES = ((0.70, 0.90), (0.55, 0.90))      # (sens, espec): por defecto y tipo Unicorn (BA ~0.73)
SALIDAS = ('cruda', 'exacta')
P_ERROR_CAL = 0.3
BLOQUES = {'demo sim (60 / 300, perturba en 100)': (60, 300, 100),
           'backend real (30 / 120, perturba en 40)': (30, 120, 40)}
VARIANTES = ('vieja', 'nueva')
PASOS_2MIN = int(config.RECUPERACION_MAX_S / config.CICLO_S)   # 57, como la EVALUACION
FRACCION = 0.7                                 # beta_pre + 0.7 * perturbacion (CP4)


def p_exacta(p, sens, espec):
    """Posterior verdadero del simulador a prior P_ERROR_CAL: solo importa si hubo deteccion."""
    llr = np.log(sens / (1 - espec)) if p > 0.5 else np.log((1 - sens) / espec)
    return float(sigmoide(logit(P_ERROR_CAL) + llr))


def sesion(semilla, sens, espec, variante, salida, pasos_est, pasos_ad, t_pert):
    """Corre una sesion y devuelve sus metricas (dict)."""
    mk = lambda r: PilotoSimulado(sens=sens, espec=espec, semilla_sujeto=semilla, semilla_ruido=r)
    w0, c0 = calibrar(mk(10_000 + semilla))
    cfg = ConfigAgente(modo='bayes', sens=sens, espec=espec, salida_detector='calibrada',
                       p_error_calibracion=P_ERROR_CAL)
    ag = AgenteErrP(w0, c0, cfg)
    conf = ConfianzaDetector(sens, espec)
    piloto = mk(semilla + 1)
    total, tp = pasos_est + pasos_ad, pasos_est + t_pert
    err, sombra, beta = np.zeros(total), np.zeros(total), np.zeros(total)
    congelado = np.zeros(total)
    orden, y = [], 1
    for t in range(total):
        if t % config.PASOS_ENSAYO == 0:
            if not orden:
                orden = list(piloto.rng.permutation([1, -1]))
            y = int(orden.pop())
        if t == pasos_est:
            prior_ini = ag.prior
        if t == tp:
            piloto.perturbar(w0, config.PERTURBACION_LOGITS)
            beta_pre, prior_pert = ag.beta, ag.prior
        aprender = t >= pasos_est
        dec = ag.decidir(piloto.rasgos(y))
        erroneo = dec.direccion != y
        p, art = piloto.errp(erroneo, dec.delta)
        fiab = conf(erroneo, p > 0.5, not art)
        sens_v, espec_v = conf.vivo()
        if salida == 'exacta':
            p = p_exacta(p, sens, espec)
        if variante == 'vieja':
            ag.actualizar(p, art, fiab if aprender else 0.0, sens_v, espec_v)
        else:
            ag.actualizar(p, art, conf.fiabilidad_bruta, sens_v, espec_v,
                          peso=fiab if aprender else 0.0)
        err[t], sombra[t], beta[t] = erroneo, dec.direccion_sombra != y, ag.beta
        congelado[t] = conf.congelado
    llego = np.flatnonzero(beta[tp:] >= beta_pre + FRACCION * config.PERTURBACION_LOGITS)
    return {'e2': err[tp:tp + PASOS_2MIN].mean(), 'pre': err[pasos_est:tp].mean(),
            'prior_ini': prior_ini, 'prior_pert': prior_pert,
            'rec': None if llego.size == 0 else int(llego[0] + 1),
            'sombra2': sombra[tp:tp + PASOS_2MIN].mean(),
            'cong': congelado[pasos_est:].mean()}


def medir(semillas, bloques):
    """{(sens, espec, salida): {variante: [metricas por sujeto]}}"""
    return {(s, e, sal): {v: [sesion(k, s, e, v, sal, *bloques) for k in semillas] for v in VARIANTES}
            for s, e in DETECTORES for sal in SALIDAS}


def imprimir(res, n, nombre):
    pm = lambda x: f'{np.mean(x):.3f} +- {np.std(x):.3f}'
    print(f'\nBloques {nombre}; simulador, {n} sujetos; salida_detector calibrada; '
          f'perturbacion {config.PERTURBACION_LOGITS} logits')
    print(f"{'detector':>13s} {'p_errp':>6s} {'variante':>8s} | {'err 2 min':>15s} {'err adapt pre':>15s} | "
          f"{'prior ini adapt':>15s} {'prior perturb':>15s} | {'pasos rec':>14s} {'mediana':>7s} "
          f"{'no rec':>6s} | {'congelado':>9s}")
    for (s, e, sal), por_var in res.items():
        for v, filas in por_var.items():
            col = lambda k: np.array([f[k] for f in filas], dtype=float)
            rec = np.array([f['rec'] for f in filas if f['rec'] is not None], dtype=float)
            no_rec = sum(f['rec'] is None for f in filas)
            t_rec = f'{rec.mean():5.1f} +- {rec.std():5.1f}' if rec.size else '     -'
            print(f'{f"s {s:.2f} e {e:.2f}":>13s} {sal:>6s} {v:>8s} | {pm(col("e2")):>15s} '
                  f'{pm(col("pre")):>15s} | {pm(col("prior_ini")):>15s} {pm(col("prior_pert")):>15s} | '
                  f'{t_rec:>14s} {np.median(rec) if rec.size else float("nan"):7.1f} '
                  f'{no_rec:>3d}/{n:<2d} | {col("cong").mean():9.3f}')
        # diferencia pareada nueva - vieja (mismos sujetos, mismos aleatorios)
        d = np.array([b['e2'] - a['e2'] for a, b in zip(por_var['vieja'], por_var['nueva'])])
        ic = 1.96 * d.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
        r = [(a['rec'], b['rec']) for a, b in zip(por_var['vieja'], por_var['nueva'])
             if a['rec'] is not None and b['rec'] is not None]
        dr = np.array([b - a for a, b in r], dtype=float)
        ic_r = 1.96 * dr.std(ddof=1) / np.sqrt(dr.size) if dr.size > 1 else float('nan')
        sombra = np.mean([f['sombra2'] for f in por_var['vieja']])
        print(f'{"":>20s} nueva - vieja: err 2 min {d.mean():+.4f} (IC95 +- {ic:.4f}); '
              f'pasos rec {dr.mean() if dr.size else float("nan"):+.1f} (IC95 +- {ic_r:.1f}, '
              f'{dr.size} pares); sombra 2 min {sombra:.3f}')
    print(f'err 2 min: {PASOS_2MIN} pasos tras perturbar; pasos rec: hasta beta >= beta_pre + '
          f'{FRACCION} * perturbacion (media +- sd entre recuperados dentro del bloque); '
          f'congelado: fraccion del bloque adaptativo; +- = 1 sd entre sujetos; IC95 pareado')


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    for nombre, bloques in BLOQUES.items():
        imprimir(medir(range(n), bloques), n, nombre)


if __name__ == '__main__':
    main()
