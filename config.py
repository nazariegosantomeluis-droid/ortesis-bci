"""Contrato de interfaces de ortesis-bci.
Todos importan de aqui. Si algo cambia, se cambia AQUI y se avisa al equipo."""
from pathlib import Path
from pylsl import StreamInfo

# ---------- Flujos LSL ----------
# nombre: (tipo, canales, Hz [0 = irregular], formato, source_id, quien lo produce)
FLUJOS = {
    'EEG':        ('EEG',     8, 250, 'float32', 'cyton-01',  'P1 puente / B1'),
    'Intencion':  ('Control', 1, 16,  'float32', 'mi-01',     'B1 decoder MI'),
    'Marcadores': ('Markers', 1, 0,   'string',  'orq-01',    'P1 orquestador'),
    'Paso':       ('Control', 3, 0,   'float32', 'agente-01', 'P1 agente'),
    'Error':      ('Control', 3, 0,   'float32', 'errp-01',   'B2 detector+CUSUM'),
}
CANALES_PASO  = ['p_prima', 'direccion', 'dtheta']
CANALES_ERROR = ['p_errp', 'artefacto', 'cusum']   # CONFIRMAR con B2

def crear_info(nombre):
    tipo, n, fs, fmt, sid, _ = FLUJOS[nombre]
    return StreamInfo(nombre, tipo, n, fs, fmt, sid)

# ---------- Marcadores ----------
CUE_CERRAR      = 'cue_cerrar'
CUE_RELAJA      = 'cue_relaja'
PERTURBACION_ON = 'perturbacion:on'
def m_paso_ack(seq): return f'paso_ack:{seq}'
def m_bloque(nombre): return f'bloque:{nombre}'

# ---------- CSV (una fila por paso) ----------
COLUMNAS_CSV = ['t_iso', 'estado', 'meta', 'angulo', 'p_prima', 'direccion',
                'dtheta', 'P_hat', 'artefacto', 'fiabilidad', 'beta',
                'error_verdadero']

# ---------- Tiempos (s) ----------
CICLO_S     = 2.1
VENTANA_MI  = 1.0
EPOCA_ERRP  = (-0.2, 0.8)   # alrededor del paso_ack

# ---------- Umbrales y parametros ----------
ETA_BETA   = 0.3    # 0.5 si BA>=0.75, 0.8 si BA>=0.85
SENS       = 0.70   # por defecto hasta que B2 mida
ESPEC      = 0.90
CUSUM_CONGELAR   = 4.0   # nats
CUSUM_REANUDAR   = 1.0
MI_EXACTITUD_MIN = 0.70  # checkpoint 2
BA_MIN           = 0.75  # checkpoint 3
ESPEC_MIN        = 0.90
PERTURBACION_LOGITS = 2.4

# ---------- Ortesis (CONFIRMAR con P2) ----------
PUERTO_ORTESIS = 'COM4'
BAUDIOS        = 115200
# Protocolo: PC -> "M,<seq>,<ang>,<dur>\n"   ESP32 -> "ACK,<seq>\n"

# ---------- Maquina de estados ----------
ESTADOS = ['IMPEDANCIAS', 'CAL_MI', 'CAL_ERRP', 'LAZO_ESTATICO',
           'LAZO_ADAPTATIVO', 'APRENDIZAJE_CONGELADO', 'PERTURBACION',
           'EVALUACION']
TRANSICIONES = {   # estado: a cuales puede pasar
    'IMPEDANCIAS':           ['CAL_MI'],
    'CAL_MI':                ['CAL_ERRP'],
    'CAL_ERRP':              ['LAZO_ESTATICO'],
    'LAZO_ESTATICO':         ['LAZO_ADAPTATIVO', 'EVALUACION'],
    'LAZO_ADAPTATIVO':       ['APRENDIZAJE_CONGELADO', 'PERTURBACION', 'EVALUACION'],
    'APRENDIZAJE_CONGELADO': ['LAZO_ADAPTATIVO', 'EVALUACION'],
    'PERTURBACION':          ['LAZO_ADAPTATIVO', 'APRENDIZAJE_CONGELADO'],
    'EVALUACION':            [],
}

# ---------- Rutas ----------
RAIZ       = Path(__file__).parent
RESULTADOS = RAIZ / 'resultados'