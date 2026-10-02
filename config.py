"""Contrato de interfaces de ortesis-bci.

Todos los modulos importan de aqui. Si algo cambia, se cambia AQUI y se avisa
al equipo. Nadie define nombres de flujos, marcadores o columnas en otro lado.
"""
from pathlib import Path

# ============================ Flujos LSL ============================
# nombre: (tipo, canales, Hz [0 = irregular], formato, source_id, quien lo produce)
FLUJOS = {
    'EEG':        ('EEG',     8, 250, 'float32', 'cyton-01',  'puente_lsl.py'),
    'Intencion':  ('Control', 1, 16,  'float32', 'mi-01',     'decoder MI (B1)'),
    'Marcadores': ('Markers', 1, 0,   'string',  'orq-01',    'orquestador'),
    'Paso':       ('Control', 3, 0,   'float32', 'agente-01', 'orquestador'),
    'Error':      ('Control', 3, 0,   'float32', 'errp-01',   'detector ErrP (B2)'),
    'Estado':     ('Markers', 1, 0,   'string',  'estado-01', 'orquestador (JSON por paso, para el tablero)'),
}
CANALES_EEG   = ['FC1', 'FC2', 'C3', 'C4', 'CP1', 'CP2', 'Cz', 'Fz']
CANALES_PASO  = ['p_prima', 'direccion', 'delta']
CANALES_ERROR = ['p_errp', 'artefacto', 'youden']


def crear_info(nombre):
    """StreamInfo del contrato (pylsl se importa aqui para que config no lo exija)."""
    from pylsl import StreamInfo
    tipo, n, fs, fmt, sid, _ = FLUJOS[nombre]
    info = StreamInfo(nombre, tipo, n, fs, fmt, sid)
    if nombre == 'EEG':
        chns = info.desc().append_child('channels')
        for c in CANALES_EEG:
            ch = chns.append_child('channel')
            ch.append_child_value('label', c)
            ch.append_child_value('unit', 'microvolts')
    return info


# ============================ Marcadores ============================
CUE_CERRAR      = 'cue_cerrar'
CUE_RELAJA      = 'cue_relaja'
PERTURBACION_ON = 'perturbacion:on'
def m_paso_ack(seq): return f'paso_ack:{seq}'
def m_bloque(nombre): return f'bloque:{nombre}'
def m_salud(subsistema, color): return f'salud:{subsistema}:{color}'


# ============================ CSV (una fila por paso) ============================
COLUMNAS_CSV = ['t_iso', 't_lsl', 'seq', 'estado', 'meta', 'angulo', 'p_prima',
                'direccion', 'delta', 'P_hat', 'artefacto', 'fiabilidad', 'beta',
                'varianza_beta', 'sens_viva', 'espec_viva', 'cambio', 'explorando',
                'error_verdadero', 'error_sombra', 'latencia_ack_ms', 'salud', 'excluido']
# salud: una letra por subsistema (V/A/R) en el orden de SUBSISTEMAS.
# excluido: vacio = paso valido; si no, el motivo (el paso queda fuera del analisis).
MOTIVOS_EXCLUSION = ['pausa:eeg', 'pausa:canal', 'pausa:ortesis', 'sin_ack', 'epoca_invalida']


# ============================ Tiempos (s) ============================
CICLO_S     = 2.1
VENTANA_MI  = 2.0            # ventana de decision de imaginacion motora
EPOCA_ERRP  = (-0.2, 0.8)    # alrededor del ACK del paso
PASOS_ENSAYO = 5             # pasos por ensayo (misma meta)

# ============================ Senal ============================
RED_HZ     = 60.0            # Mexico
BANDA_MI   = (8.0, 30.0)
BANDA_ERRP = (1.0, 10.0)

# ============================ Agente ============================
ETA_BETA  = 0.3              # solo para el modo 'fijo' (linea base)
SENS      = 0.70             # por defecto, hasta que se calibre el detector
ESPEC     = 0.90
PASO_VISIBLE = 0.08          # fraccion del rango que el piloto percibe (medirlo con el piloto)
PASO_MAX  = 0.20

# ============================ Umbrales go / no go ============================
LATENCIA_JITTER_MAX_MS = 15.0   # checkpoint 1
IMPEDANCIA_MAX_KOHM = 20.0      # checkpoint 1 (Cyton con gel)
MI_EXACTITUD_MIN = 0.70         # checkpoint 2
BA_MIN           = 0.75         # checkpoint 3
ESPEC_MIN        = 0.90
RECUPERACION_MAX_S = 120.0      # checkpoint 4
PERTURBACION_LOGITS = 2.4

# ============================ Ortesis (USB serial) ============================
PUERTO_ORTESIS = 'COM4'
BAUDIOS        = 115200
DURACION_PASO_MS = 250
# PC -> ESP32:  "M,<seq>,<angulo 0-1000>,<duracion_ms>\n"
# ESP32 -> PC:  "A,<seq>,<t_us>\n"      ACK al aplicar el primer pulso
#               "T,<t_us>,<angulo>,<fsr>\n"   telemetria a 50 Hz

# ============================ Salud ============================
SUBSISTEMAS = ['eeg', 'ortesis', 'reloj', 'detector']
VERDE, AMARILLO, ROJO = 'VERDE', 'AMARILLO', 'ROJO'
SALUD = {
    'eeg_edad_amarillo_s': 0.3, 'eeg_edad_rojo_s': 1.0,     # edad de la ultima muestra
    'eeg_tasa_amarillo': 0.10, 'eeg_tasa_rojo': 0.25,       # desviacion relativa de la tasa real
    'hueco_max_s': 0.02,                                    # salto entre muestras que cuenta como corte
    'canal_plano_uv': 0.1, 'canal_saturado_uv': 180_000.0, 'canal_ruidoso_uv': 100.0,
    'ventana_canales_s': 2.0,
    'acks_amarillo': 1, 'acks_rojo': 3,                     # ACK perdidos consecutivos
    'latencia_pico_ms': 80.0,
    'reloj_amarillo_ms': 20.0, 'reloj_rojo_ms': 50.0,       # deriva del retraso contra su linea base
    'reloj_lecturas_base': 40,                              # lecturas para (re)medir la linea base
    'detector_amarillo': 0.7,                               # fiabilidad bajo este valor
    'verde_para_reanudar_s': 3.0,                           # VERDE continuo para salir de la pausa
}
POSICION_SEGURA   = 0.0          # abierta
PAUSA_DURACION_MS = 1500         # abrir despacio
RECONEXION_INICIAL_S, RECONEXION_MAX_S = 0.5, 8.0           # retroceso exponencial
CAL_REPETICIONES_MAX = 3         # repeticiones de un ensayo de calibracion afectado por una falla

# ============================ Maquina de estados ============================
ESTADOS = ['IMPEDANCIAS', 'CAL_MI', 'CAL_ERRP', 'LAZO_ESTATICO',
           'LAZO_ADAPTATIVO', 'APRENDIZAJE_CONGELADO', 'PERTURBACION',
           'PAUSA_SEGURA', 'EVALUACION']
TRANSICIONES = {
    'IMPEDANCIAS':           ['CAL_MI', 'LAZO_ESTATICO', 'EVALUACION'],
    'CAL_MI':                ['CAL_ERRP', 'EVALUACION'],
    'CAL_ERRP':              ['LAZO_ESTATICO', 'EVALUACION'],
    'LAZO_ESTATICO':         ['LAZO_ADAPTATIVO', 'PAUSA_SEGURA', 'EVALUACION'],
    'LAZO_ADAPTATIVO':       ['APRENDIZAJE_CONGELADO', 'PERTURBACION', 'PAUSA_SEGURA', 'EVALUACION'],
    'APRENDIZAJE_CONGELADO': ['LAZO_ADAPTATIVO', 'PERTURBACION', 'PAUSA_SEGURA', 'EVALUACION'],
    'PERTURBACION':          ['LAZO_ADAPTATIVO', 'APRENDIZAJE_CONGELADO', 'PAUSA_SEGURA', 'EVALUACION'],
    'PAUSA_SEGURA':          ['LAZO_ESTATICO', 'LAZO_ADAPTATIVO', 'APRENDIZAJE_CONGELADO', 'EVALUACION'],
    'EVALUACION':            [],
}

# ============================ Rutas ============================
RAIZ       = Path(__file__).resolve().parent
RESULTADOS = RAIZ / 'resultados'
MODELOS    = RAIZ / 'modelos'
IMPEDANCIAS_JSON = RESULTADOS / 'impedancias.json'
ESTADO_SESION_JSON = RESULTADOS / 'estado_sesion.json'
