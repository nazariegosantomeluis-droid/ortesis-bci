"""Cuanto cuesta perder la telemetria del ESP32: el respaldo por ACK de la alineacion de la
epoca del ErrP, medido en lazo cerrado con el gemelo sin LSL (el banco de agente_lento.py).

La ortesis empieza a moverse entre 30 y 150 ms despues del ACK. Tres escenarios, con las mismas
sesiones (metas y ruido de fondo iguales):

  telemetria      la epoca se corta en el inicio real del movimiento, en calibracion y en lazo
  ack             no hay telemetria nunca: calibracion y lazo se cortan en el ACK
  respaldo        hay telemetria en la calibracion y se pierde en el lazo: el lazo corta en el ACK
                  mas la latencia mecanica media (lo que hace OrtesisSerial.inicio_movimiento)

Mide la BA del detector (anidada en calibracion y en vivo) y la recuperacion del agente.
SOLO verifica: la latencia mecanica y el ErrP los programamos nosotros en el gemelo.

Uso: python estudios/respaldo_ack.py [sujetos] [repeticiones] [topes]
     topes: 'ignorar' modela los topes del recorrido de la ortesis como el lazo de hoy (ver
     agente_lento.TOPES); sin el, todo paso es un movimiento visible (asi se midio primero).
"""
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # la raiz del repositorio
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import agente_lento as al

ESCENARIOS = {                      # nombre: (corte en la calibracion, corte en el lazo)
    'telemetria en calibracion y lazo': ('inicio', 'inicio'),
    'sin telemetria: todo en el ACK': ('ack', 'ack'),
    'se pierde en el lazo: ACK + latencia media': ('inicio', 'ack+latencia'),
}


def correr(sujetos=4, reps=4, salida=print):
    al.REGIMEN, al.MARGEN_S = 'actual', 0.12     # el mismo ritmo en los tres escenarios
    filas, modelos = {e: [] for e in ESCENARIOS}, {}
    for s in range(sujetos):
        for corte in ('inicio', 'ack'):
            al.ALINEACION = corte
            modelos[s, corte] = al.preparar(s, salida)
        for r in range(reps):
            for e, (cal, lazo) in ESCENARIOS.items():
                al.ALINEACION = lazo
                filas[e] += [dict(x, sujeto=s, rep=r) for x in al.lazo(modelos[s, cal], 1000 + 100 * s + r)]
        salida(f'  sujeto {s} listo')
    al.ALINEACION = 'inicio'
    return filas, modelos


def main():
    sujetos = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    al.TOPES = sys.argv[3] if len(sys.argv) > 3 else None
    filas, modelos = correr(sujetos, reps, lambda *a: print(*a, flush=True))
    al.CACHE.mkdir(parents=True, exist_ok=True)
    nombre = 'respaldo_ack.pkl' if al.TOPES is None else f'respaldo_ack_topes_{al.TOPES}.pkl'
    (al.CACHE / nombre).write_bytes(pickle.dumps({'filas': filas}))
    n = sujetos * reps
    ref = list(ESCENARIOS)[0]
    base = al.sesiones(filas, ref)
    print(f'\n{sujetos} sujetos del gemelo x {reps} lazos = {n} sesiones por escenario; diferencia pareada contra "{ref}"'
          + (f'; topes del recorrido: {al.TOPES}' if al.TOPES else '; sin topes (todo paso visible)'))
    for e, (cal, _) in ESCENARIOS.items():
        x = al.sesiones(filas, e)
        de = np.array([a['err'] - b['err'] for a, b in zip(x, base)])
        rec = [a['rec'] for a in x if a['rec'] is not None]
        F = [f for f in filas[e] if not f['art']]
        sens = np.mean([f['detectado'] for f in F if f['erroneo']])
        espec = 1 - np.mean([f['detectado'] for f in F if not f['erroneo']])
        ba_cal = np.mean([modelos[s, cal]['det'].ba for s in range(sujetos)])
        print(f"  {e:44s} detector: BA de calibracion {ba_cal:.2f}, en vivo {0.5 * (sens + espec):.2f} (sens {sens:.2f}, "
              f"espec {espec:.2f}) | agente: error 2 min {np.mean([a['err'] for a in x]):.3f} ({de.mean():+.3f} +- "
              f"{de.std(ddof=1) / np.sqrt(n):.3f}), sombra {np.mean([a['sombra'] for a in x]):.2f} | recuperan en "
              f"{al.VENTANA} pasos {len(rec)}/{n}" + (f' (mediana {np.median(rec):.0f})' if rec else '')
              + f" | congelado {np.mean([f['peso'] == 0 for f in filas[e] if f['post']]):.0%}")


if __name__ == '__main__':
    main()
