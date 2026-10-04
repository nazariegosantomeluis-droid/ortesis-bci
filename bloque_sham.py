"""Bloque sham (B1): con el piloto en REPOSO la ortesis se mueve al azar y p(t) del decoder de MI
no debe seguirla. Si la sigue, p(t) esta leyendo otra cosa que la intencion (ruido de los servos,
cables que se mueven, respuesta visual al movimiento) y el lazo "funcionaria" por el motivo equivocado.

Uso (despues de calibrar el decoder: existe modelos/decoder_im.pkl)
  python bloque_sham.py --puerto COM4                  casco real + ortesis por USB
  python bloque_sham.py --ortesis-sim                  ortesis simulada (EEG real o del gemelo)
  python bloque_sham.py --fuente unicornlsl            EEG desde la app UnicornLSL de g.tec

Instruccion al piloto: no imagines ningun movimiento; mira la ortesis con la mente en blanco o
cuenta hacia atras de 3 en 3. Quieto, sin apretar la mandibula.

Cada paso: la ortesis vuelve al centro (no cuenta), reposo, y se mueve ARRIBA o ABAJO de 0.5 al azar
(mitad cerrar, mitad abrir). La ventana de MI va de 0.5 s antes a 1.5 s despues del inicio del
movimiento. hardware.evaluar_sham compara p con la direccion y dice PASA / FALLA.
"""
import argparse
import csv
import time
from datetime import datetime

import numpy as np

import config

PASO_SHAM = 0.2             # el movimiento llega a 0.5 +- 0.2 del rango
REPOSO_S = 1.5              # entre el centrado y el movimiento: que su ruido no entre a la ventana
DESPUES_S = 1.5             # ventana: hasta 1.5 s despues del ACK


def correr(eeg, ortesis, decoder, pasos=config.SHAM_PASOS, semilla=None, reposo_s=REPOSO_S,
           despues_s=DESPUES_S, salida=print):
    """Corre el bloque. Devuelve (filas, resultado de hardware.evaluar_sham). Cada fila:
    paso, direccion (1 = cerrar), p (nan si el paso no sirve) y excluido (motivo o '')."""
    import hardware as hw
    rng = np.random.default_rng(semilla)
    direcciones = rng.permutation(np.array([1, 0] * (pasos // 2)))
    u = config.SALUD
    filas = []
    for k, d in enumerate(direcciones):
        ortesis.mover(0.5, config.CENTRADO_DURACION_MS)
        time.sleep(reposo_s + config.CENTRADO_DURACION_MS / 1000)
        salida(f'[{k + 1}/{len(direcciones)}] reposo: la ortesis va a {"CERRAR" if d else "ABRIR"} sola')
        _, t_ack, _ = ortesis.mover(0.5 + (PASO_SHAM if d else -PASO_SHAM))
        time.sleep(despues_s)
        fila = {'paso': k, 'direccion': int(d), 'p': float('nan'), 'excluido': ''}
        if t_ack is None:
            fila['excluido'] = 'sin_ack'
        else:
            x, _ = eeg.ventana(config.VENTANA_MI + 1.0, u['mi_perdida_max'], u['mi_hueco_max_s'])
            if x is None:
                fila['excluido'] = 'eeg_no_fresco'
            else:
                fin = eeg.ultimo_t()
                giro = eeg.movimiento(fin - config.VENTANA_MI, fin) if hasattr(eeg, 'movimiento') else None
                if giro is not None and giro > config.GIRO_ARTEFACTO_DPS:
                    fila['excluido'] = f'cabeza:{giro:.0f}dps'
                else:
                    xf = hw.filtrar(x, config.BANDA_MI, eeg.fs)[:, -int(config.VENTANA_MI * eeg.fs):]
                    phi = decoder.phi(xf, actualizar_centro=False)      # el sham no mueve el recentrado
                    if phi is not None:
                        fila['p'] = float(1 / (1 + np.exp(-(decoder.w0 @ phi + decoder.c0))))
                    else:
                        fila['excluido'] = 'ventana_no_finita'
        filas.append(fila)
    resultado = hw.evaluar_sham([f['p'] for f in filas], [f['direccion'] for f in filas])
    return filas, resultado


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--puerto', default=config.PUERTO_ORTESIS)
    ap.add_argument('--ortesis-sim', dest='ortesis_sim', action='store_true')
    ap.add_argument('--fuente', choices=sorted(config.FUENTES_EEG), default='puente')
    ap.add_argument('--eeg-nombre', dest='eeg_nombre', default=None)
    ap.add_argument('--pasos', type=int, default=config.SHAM_PASOS)
    ap.add_argument('--semilla', type=int, default=None)
    a = ap.parse_args()
    import hardware as hw
    decoder = hw.cargar('decoder_im.pkl')
    print(f'Decoder cargado (BA de calibracion {decoder.ba:.2f}, canales {decoder.eleccion}).')
    eeg = hw.EntradaEEG(fuente=a.fuente, nombre=a.eeg_nombre)
    ortesis = hw.OrtesisSimulada() if a.ortesis_sim else hw.OrtesisSerial(a.puerto)
    print('Piloto: REPOSO. No imagines ningun movimiento; mira la ortesis con la mente en blanco.')
    time.sleep(3.0)
    try:
        filas, r = correr(eeg, ortesis, decoder, a.pasos, a.semilla)
    finally:
        ortesis.cerrar()
        eeg.cerrar()
    config.RESULTADOS.mkdir(exist_ok=True)
    ruta = config.RESULTADOS / f'sham_{datetime.now():%Y%m%d_%H%M%S}.csv'
    with open(ruta, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['paso', 'direccion', 'p', 'excluido'])
        w.writeheader()
        w.writerows(filas)
    excl = sum(1 for f in filas if f['excluido'])
    print(r['texto'] + (f' ({excl} pasos excluidos)' if excl else ''))
    print(f'Registro guardado en {ruta}')
    return 0 if r['pasa'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
