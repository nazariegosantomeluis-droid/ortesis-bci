"""Por que la ortesis casi nunca cierra completa, y que lo corrige (simulador, sin LSL).

Metricas por variante (30 sujetos del simulador, bloque adaptativo con perturbacion):
  - fraccion de ensayos "cerrar" que terminan con la ortesis >= 0.95 (cierre completo)
  - fraccion de ensayos "relaja" que terminan <= 0.05
  - error del agente en los 2 min tras perturbar (no debe empeorar)
Variantes: la actual; paso maximo mayor; y "ensayo desde el punto medio" (cada ensayo
empieza con la ortesis en 0.5)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
import numpy as np
import config
from agente_errp import AgenteErrP, ConfigAgente, ConfianzaDetector
from simulador_lazo import PilotoSimulado, calibrar


def sesion(semilla, ganancia=0.20, paso_max=0.20, centrar=False, pasos=600):
    mk = lambda r: PilotoSimulado(semilla_sujeto=semilla, semilla_ruido=r)
    w0, c0 = calibrar(mk(10_000 + semilla))
    piloto = mk(semilla + 1)
    ag = AgenteErrP(w0, c0, ConfigAgente(ganancia=ganancia, paso_max=paso_max))
    conf = ConfianzaDetector()
    angulo, orden, fin_cerrar, fin_relaja, err = 0.5, [], [], [], []
    t_p = pasos // 2
    for t in range(pasos):
        if t % config.PASOS_ENSAYO == 0:
            if t:
                (fin_cerrar if y > 0 else fin_relaja).append(angulo)
            if not orden:
                orden = list(piloto.rng.permutation([1, -1]))
            y = int(orden.pop())
            if centrar:
                angulo = 0.5
        if t == t_p:
            piloto.perturbar(w0, config.PERTURBACION_LOGITS)
        dec = ag.decidir(piloto.rasgos(y))
        angulo = float(np.clip(angulo + dec.delta, 0, 1))
        erroneo = dec.direccion != y
        p, art = piloto.errp(erroneo, dec.delta)
        fiab = conf(erroneo, p > 0.5, not art)
        ag.actualizar(p, art, fiab, *conf.vivo())
        err.append(erroneo)
    k = t_p + int(config.RECUPERACION_MAX_S / config.CICLO_S)
    return (np.mean(np.array(fin_cerrar) >= 0.95), np.mean(np.array(fin_relaja) <= 0.05),
            np.mean(err[t_p:k]), np.mean(err[:t_p]))


VARIANTES = [('actual (ganancia 0.20, paso max 0.20)', dict()),
             ('paso max 0.30, ganancia 0.30', dict(ganancia=0.30, paso_max=0.30)),
             ('paso max 0.35, ganancia 0.35', dict(ganancia=0.35, paso_max=0.35)),
             ('ensayo desde 0.5', dict(centrar=True)),
             ('ensayo desde 0.5 + paso max 0.30', dict(centrar=True, ganancia=0.30, paso_max=0.30))]

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f'{N} sujetos del simulador')
    for nombre, kw in VARIANTES:
        F = np.array([sesion(s, **kw) for s in range(N)])
        print(f'  {nombre:38s} cierra completo {F[:, 0].mean():.2f} | abre completo {F[:, 1].mean():.2f} | '
              f'error antes {F[:, 3].mean():.3f} | error 2 min tras perturbar {F[:, 2].mean():.3f}', flush=True)
