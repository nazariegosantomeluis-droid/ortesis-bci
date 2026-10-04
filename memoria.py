"""Memoria entre sesiones: lo que una sesion deja para que la siguiente del MISMO piloto calibre
con menos ensayos (orquestador.py --guardar-memoria y --desde-sesion; apagado por defecto).

Que guarda (un archivo junto al CSV de la sesion, config.SUFIJO_MEMORIA):
  decoder    el DecoderIM de la sesion, tal como termino (con su centro recentrado en el lazo)
  mi         los ensayos de calibracion de MI como rasgos: covarianza de los 8 canales, recentrada
             con el centro de SU sesion y llevada al espacio tangente en la identidad. Es el formato
             del decoder pre-entrenado (DecoderIM.ajustar_desde): el recentrado alinea sesiones igual
             que alinea personas, asi que los rasgos de ayer sirven hoy sin saber como quedo el casco
  detector   el DetectorErrP vigente al terminar
  errp       sus epocas con etiqueta (calibracion y, si hubo co-adaptacion, las del lazo)
  agente     el estado del agente y de la confianza del detector, y beta_sin_perturbar

Como arranca la sesion siguiente (--desde-sesion):
  MI     config.MEMORIA_ENSAYOS_MI ensayos, de largo fijo. Su centro (sin usar las etiquetas)
         recentra el decoder; el clasificador se ajusta con los rasgos de la sesion previa mas los
         de hoy, que pesan config.MEMORIA_PESO_NUEVO veces. El CP2 es la BA de validacion cruzada
         sobre los ensayos de HOY.
  ErrP   config.MEMORIA_EPOCAS_ERRP epocas. El detector se ajusta con las epocas previas mas las
         de hoy, con los canales y vistas que eligio la sesion previa. El CP3 se mide SOLO con las
         epocas de hoy (detector_con_memoria): las previas entrenan, nunca se evaluan.
  agente beta arranca en beta_sin_perturbar: la que habia antes de la perturbacion de la demo.
         NUNCA la beta final, que incluye la correccion de una perturbacion artificial que hoy no
         esta. No cambia como aprende el agente, solo de donde parte.

Para una sesion ya corrida sin --guardar-memoria (la del domingo con v-demo):
  python memoria.py                              arma la memoria con lo que quedo en modelos/ y resultados/
  python memoria.py --csv resultados/sesion_real_<fecha>.csv
OJO: modelos/ guarda la ULTIMA calibracion, sea de quien sea. Hay que armar la memoria antes de que
otra calibracion (otro piloto, el gemelo, pruebas.py --completa) la pise; el programa dice de
cuando es cada pieza.
"""
import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

import config
import hardware as hw

VERSION = 1


def rasgos_mi(X):
    """Rasgos de unas ventanas de MI (n, 8, muestras): covarianza OAS de los 8 canales, recentrada
    con su propia media riemanniana y proyectada al espacio tangente en la identidad."""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import tangent_space
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.mean import mean_riemann
    C = Covariances('oas').fit_transform(np.asarray(X))
    Mi = invsqrtm(mean_riemann(C))
    return tangent_space(np.array([Mi @ c @ Mi for c in C]), np.eye(C.shape[1]))


def mi_de_sesion(X=None, y=None, previa=None):
    """El bloque 'mi' de la memoria: los ensayos de esta sesion (si calibro) apilados con los que
    traia de la sesion previa. Cada sesion va recentrada con su propio centro. None si no hay nada."""
    Z, etiquetas = ([], []) if previa is None else ([previa['Z']], [previa['y']])
    if X is not None and len(X):
        Z.append(rasgos_mi(X).astype(np.float32))
        etiquetas.append(np.asarray(y, dtype=int))
    if not Z:
        return None
    return {'Z': np.vstack(Z), 'y': np.concatenate(etiquetas), 'canales': list(config.CANALES_EEG),
            'nombre': 'memoria de sesion', 'banda': config.BANDA_MI, 'ventana_s': config.VENTANA_MI}


def ensayos_del_decoder(dec):
    """(X, y, archivo) de la calibracion de MI con que se ajusto un decoder: el calibracion_mi_*.npz mas
    nuevo de resultados/ cuyas etiquetas son las suyas. (None, None, None) si no esta."""
    for ruta in sorted(config.RESULTADOS.glob('calibracion_mi_*.npz'), reverse=True):
        d = np.load(ruta)
        if np.array_equal(d['y'], getattr(dec, 'y_cal', None)):
            return d['X'], d['y'], ruta.name
    return None, None, None


def construir(decoder, detector, mi, X_errp, y_errp, agente=None, confianza=None, beta_sin_perturbar=None, origen=''):
    return {'version': VERSION, 't': time.time(), 'origen': origen, 'decoder': decoder, 'mi': mi, 'detector': detector,
            'errp': {'X': np.asarray(X_errp, dtype=np.float32), 'y': np.asarray(y_errp, dtype=int)},
            'agente': dict(agente or {}, beta_sin_perturbar=beta_sin_perturbar), 'confianza': confianza}


def ruta_de(ruta):
    """El archivo de memoria de una sesion: se puede dar el propio archivo o el CSV de la sesion."""
    ruta = Path(ruta)
    return ruta if ruta.name.endswith(config.SUFIJO_MEMORIA) else ruta.with_name(ruta.stem + config.SUFIJO_MEMORIA)


def guardar(ruta, memoria):
    ruta = ruta_de(ruta)
    tmp = ruta.with_suffix('.tmp')
    tmp.write_bytes(pickle.dumps(memoria))
    os.replace(tmp, ruta)                                  # nunca queda una memoria a medias
    return ruta


def cargar(ruta):
    ruta = ruta_de(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f'No existe {ruta}. Se crea con orquestador.py real --guardar-memoria, o con '
                                f'python memoria.py para una sesion ya corrida.')
    m = pickle.loads(ruta.read_bytes())
    if m.get('version') != VERSION:
        raise ValueError(f"{ruta.name} es de la version {m.get('version')} de la memoria y este codigo usa la {VERSION}")
    mi = m['mi']
    if mi is None:
        raise ValueError(f'{ruta.name} no trae ensayos de MI: no hay con que arrancar el decoder')
    if mi['canales'] != list(config.CANALES_EEG) or tuple(mi['banda']) != tuple(config.BANDA_MI) or mi['ventana_s'] != config.VENTANA_MI:
        raise ValueError(f'{ruta.name} se grabo con otro montaje, otra banda u otra ventana de MI: no sirve con este contrato')
    return m


def texto(m):
    """Una linea con lo que trae la memoria y de cuando es."""
    h = (time.time() - m['t']) / 3600
    d, b = m['detector'], m['agente'].get('beta_sin_perturbar')
    return (f"Memoria de {m['origen'] or 'una sesion'} (guardada hace {h:.1f} h): {len(m['mi']['y'])} ensayos de MI, "
            f"{len(m['errp']['y'])} epocas de ErrP ({int(m['errp']['y'].sum())} errores), detector de entonces sens "
            f"{d.sens:.2f} espec {d.espec:.2f} ({d.eleccion}), beta sin perturbar "
            + ('sin dato' if b is None else f'{b:+.2f}')
            + '. OJO: solo sirve si es del MISMO piloto.')


def beta_inicial(m, beta_max):
    """La beta con que arranca el agente: la de antes de la perturbacion de la sesion previa, acotada.
    None si la memoria no la trae (el agente arranca en 0, como siempre)."""
    b = m['agente'].get('beta_sin_perturbar')
    return None if b is None else float(np.clip(b, -beta_max, beta_max))


def detector_con_memoria(previo, Xp, yp, X, y, direccion=None):
    """Detector de la sesion de hoy: epocas de la sesion previa (Xp, yp) mas las de hoy (X, y), con
    los canales y vistas que eligio la previa. sens, espec, BA y pred_cv son HONESTAS para hoy:
    validacion cruzada en la que solo las epocas de hoy caen en el pliegue de prueba; las previas
    siempre entrenan y nunca se evaluan. El umbral de cada pliegue se elige sin ver su prueba."""
    from sklearn.model_selection import StratifiedKFold
    Xp, X, yp, y = np.asarray(Xp, dtype=float), np.asarray(X, dtype=float), np.asarray(yp, dtype=int), np.asarray(y, dtype=int)

    def ajustar(i):
        return hw.DetectorErrP(canales=previo.canales, vistas=previo.vistas).ajustar(
            np.concatenate([Xp, X[i]]), np.r_[yp, y[i]], evaluar=False)
    pred = np.zeros(len(y), dtype=int)
    k = int(min(4, np.bincount(y, minlength=2).min()))
    for ent, pru in StratifiedKFold(max(k, 2), shuffle=True, random_state=0).split(X, y):
        d = ajustar(ent)
        pred[pru] = [int(d.p_error(e) > d.umbral) for e in X[pru]]
    det = ajustar(np.arange(len(y)))
    det.pred_cv, det.y_cal = pred, y                       # las de hoy: las que ve el CP3
    det.sens, det.espec = float((pred[y == 1] == 1).mean()), float((pred[y == 0] == 0).mean())
    det.ba = 0.5 * (det.sens + det.espec)
    det.eleccion = f"{getattr(previo, 'eleccion', 'fijo')} + memoria de {len(yp)} epocas"
    det.puntajes = {}
    if direccion is not None:
        det.por_direccion = hw.metricas_por_direccion(y, pred, direccion)
    return det


# ====================================================================== sesiones ya corridas
def desde_archivos(csv=None):
    """Arma la memoria de una sesion que no la guardo, con lo que dejo: modelos/decoder_im.pkl,
    modelos/detector_errp.pkl, modelos/detector_errp_datos.npz, el calibracion_mi_*.npz de
    resultados/ cuyas etiquetas son las del decoder, y el estado del agente de
    resultados/estado_sesion.json si esa instantanea es de la misma sesion. Devuelve (memoria, notas
    de donde salio cada pieza, CSV de la sesion)."""
    notas = []
    dec, det = hw.cargar('decoder_im.pkl'), hw.cargar('detector_errp.pkl')
    edad = lambda r: (time.time() - r.stat().st_mtime) / 3600
    notas.append(f"decoder y detector de modelos/, calibrados hace {edad(config.MODELOS / 'decoder_im.pkl'):.1f} h y "
                 f"{edad(config.MODELOS / 'detector_errp.pkl'):.1f} h")
    X_mi, y_mi, archivo = ensayos_del_decoder(dec)
    if archivo is None:
        raise FileNotFoundError('ningun resultados/calibracion_mi_*.npz tiene las etiquetas del decoder de modelos/: '
                                'no se puede armar la memoria de MI')
    mi = mi_de_sesion(X_mi, y_mi)
    notas.append(f'ensayos de MI de {archivo} ({len(y_mi)}; sus etiquetas son las del decoder)')
    e = np.load(config.MODELOS / 'detector_errp_datos.npz')
    agente = confianza = beta = None
    try:
        inst = json.loads(config.ESTADO_SESION_JSON.read_text())
    except (OSError, ValueError):
        inst = None
    if inst and (csv is None or Path(inst['ruta_csv']).name == Path(csv).name):
        csv = csv or inst['ruta_csv']
        agente, confianza = inst['agente'], inst['confianza']
        beta = None if inst.get('sham') else (inst['beta_pre'] if inst.get('beta_pre') is not None else agente['beta'])
        notas.append(f"estado del agente de {config.ESTADO_SESION_JSON.name} (sesion {Path(inst['ruta_csv']).name}"
                     + ('' if inst.get('terminada') else ', SIN terminar') + ')')
    else:
        notas.append('sin estado del agente: la instantanea guardada no es de esa sesion (el agente arrancara en 0)')
    if csv is None:
        raise FileNotFoundError('no hay instantanea de sesion: di cual con --csv resultados/sesion_real_<fecha>.csv')
    return construir(dec, det, mi, e['X'], e['y'], agente, confianza, beta, Path(csv).name), notas, Path(csv)


def main():
    ap = argparse.ArgumentParser(description='Arma la memoria de una sesion ya corrida (para --desde-sesion)')
    ap.add_argument('--csv', help='CSV de la sesion (por defecto, la de resultados/estado_sesion.json)')
    a = ap.parse_args()
    m, notas, csv = desde_archivos(a.csv)
    for n in notas:
        print('  ' + n)
    ruta = guardar(csv, m)
    print(texto(m))
    print(f'Guardada: {ruta}\nPara arrancar de ella: python orquestador.py real --puerto COM4 --desde-sesion {ruta}')


if __name__ == '__main__':
    main()
