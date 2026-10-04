"""Contrato de interfaces de ortesis-bci.

Todos los modulos importan de aqui. Si algo cambia, se cambia AQUI y se avisa
al equipo. Nadie define nombres de flujos, marcadores o columnas en otro lado.
"""
from pathlib import Path

# ============================ Flujos LSL ============================
# nombre: (tipo, canales, Hz [0 = irregular], formato, source_id, quien lo produce)
FLUJOS = {
    'EEG':        ('EEG',     8, 250, 'float32', 'unicorn-01', 'puente_lsl.py (o el gemelo)'),
    'IMU':        ('IMU',     6, 250, 'float32', 'unicorn-imu-01', 'puente_lsl.py (o el gemelo)'),
    'Intencion':  ('Control', 1, 16,  'float32', 'mi-01',     'decoder MI (B1)'),
    'Marcadores': ('Markers', 1, 0,   'string',  'orq-01',    'orquestador'),
    'Paso':       ('Control', 3, 0,   'float32', 'agente-01', 'orquestador'),
    'Error':      ('Control', 3, 0,   'float32', 'errp-01',   'detector ErrP (B2)'),
    'Estado':     ('Markers', 1, 0,   'string',  'estado-01', 'orquestador (JSON por paso, para el tablero)'),
}
# Montaje del g.tec Unicorn Hybrid Black, en el orden en que lo entrega el casco (BrainFlow
# UNICORN_BOARD y la API de g.tec): 8 EEG a 250 Hz por Bluetooth, mas IMU, bateria,
# contador de muestras e indicador de validez.
CANALES_EEG   = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
CANALES_IMU   = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']    # g y grados/s
# Papel de cada sensor. La calibracion decide, por validacion cruzada, si cada modelo usa
# solo los canales de su papel o los 8 (no se fija aqui ni con el gemelo: seria circular).
PAPELES = {
    'mi':     ['C3', 'Cz', 'C4'],        # imaginacion motora (ERD mu/beta)
    'errp':   ['Fz', 'Cz', 'Pz'],        # potencial de error
    'visual': ['PO7', 'Oz', 'PO8'],      # respuesta visual al movimiento (Tarea 2)
    'alfa':   ['PO7', 'Oz', 'PO8'],      # alfa occipital: semaforo PILOTO (somnolencia / atencion)
}


def candidatos(modelo):
    """Configuraciones entre las que elige la calibracion real por validacion cruzada (anidada
    para reportar): los canales del papel contra los 8 y, en el detector, dos contra tres vistas."""
    todos = list(range(len(CANALES_EEG)))
    if modelo == 'decoder':
        return {'C3/Cz/C4': indices('mi'), '8 canales': todos}
    return {f'{nc}, {nv} vistas': (c, nv) for nc, c in (('Fz/Cz/Pz', indices('errp')), ('8 canales', todos))
            for nv in ('dos', 'tres')}


def indices(canales):
    """Posiciones en CANALES_EEG de un papel ('mi', 'errp'...) o de una lista de electrodos."""
    return [CANALES_EEG.index(c) for c in (PAPELES[canales] if isinstance(canales, str) else canales)]


# De donde puede venir el EEG y en que canal del flujo esta cada cosa.
#   puente:     flujo del contrato (puente_lsl.py o el gemelo). 8 canales de EEG ya estampados
#               con la hora reconstruida por contador; la IMU va en el flujo 'IMU'.
#   unicornlsl: la app UnicornLSL de g.tec (respaldo). Un solo flujo de tipo 'Data' con 17
#               canales SIN etiquetas, estampado a la llegada; su nombre es el que se escriba
#               en la app (por defecto, el numero de serie). El orden es el de la API de g.tec;
#               verificar_unicorn.py lo comprueba con el casco real.
FUENTES_EEG = {
    'puente':     {'nombre': 'EEG', 'tipo': None, 'canales': 8, 'eeg': list(range(8)), 'imu': None,
                   'bateria': None, 'contador': None, 'validez': None},
    'unicornlsl': {'nombre': None, 'tipo': 'Data', 'canales': 17, 'eeg': list(range(8)),
                   'imu': list(range(8, 14)), 'bateria': 14, 'contador': 15, 'validez': 16},
}
CANALES_PASO  = ['p_prima', 'direccion', 'delta']
CANALES_ERROR = ['p_errp', 'artefacto', 'youden']


def crear_info(nombre):
    """StreamInfo del contrato (pylsl se importa aqui para que config no lo exija)."""
    from pylsl import StreamInfo
    tipo, n, fs, fmt, sid, _ = FLUJOS[nombre]
    info = StreamInfo(nombre, tipo, n, fs, fmt, sid)
    etiquetas = {'EEG': [(c, 'microvolts') for c in CANALES_EEG],
                 'IMU': [(c, 'g' if c.startswith('acc') else 'deg/s') for c in CANALES_IMU]}
    if nombre in etiquetas:
        chns = info.desc().append_child('channels')
        for c, unidad in etiquetas[nombre]:
            ch = chns.append_child('channel')
            ch.append_child_value('label', c)
            ch.append_child_value('unit', unidad)
    return info


# ============================ Marcadores ============================
CUE_CERRAR      = 'cue_cerrar'
CUE_RELAJA      = 'cue_relaja'
PERTURBACION_ON = 'perturbacion:on'
CENTRADO        = 'centrado'       # la ortesis vuelve al punto medio antes del cue (no es un paso)
AVISO_AJENO     = 'aviso_ajeno'    # la pantalla anuncia un movimiento ajeno (Tarea 2)
def m_paso_ajeno(seq): return f'paso_ajeno:{seq}'             # ACK de un movimiento ajeno
def m_paso_ack(seq): return f'paso_ack:{seq}'
def m_paso_quieto(seq): return f'paso_quieto:{seq}'           # ACK de un paso que no movio la ortesis (tope)
def m_bloque(nombre): return f'bloque:{nombre}'
def m_paso_inicio(seq): return f'paso_inicio:{seq}'           # inicio real del movimiento (telemetria)
def m_salud(subsistema, color): return f'salud:{subsistema}:{color}'
def m_detector(version): return f'detector:v{version}'      # cambio de modelo del detector co-adaptativo


# ============================ CSV (una fila por paso) ============================
COLUMNAS_CSV = ['t_iso', 't_lsl', 'seq', 'estado', 'bloque', 'meta', 'angulo', 'p_prima',
                'direccion', 'delta', 'P_hat', 'artefacto', 'fiabilidad', 'beta',
                'varianza_beta', 'sens_viva', 'espec_viva', 'cambio', 'explorando',
                'error_verdadero', 'error_sombra', 'latencia_ack_ms', 'salud', 'excluido', 'alineacion',
                'ajeno', 'n1_uv', 'iic']
# bloque: vacio, o 'real' / 'sham' en los dos bloques del control causal (--sham).
# alineacion: a que se alineo la epoca del ErrP: 'telemetria' (inicio real del movimiento),
# 'ack+latencia' (ACK mas la latencia mecanica media medida) o 'ack'. SIN_MOVIMIENTO: la ortesis
# ya estaba en el tope y no se movio; no hay epoca (ver IGNORAR_SIN_MOVIMIENTO).
SIN_MOVIMIENTO = 'sin_movimiento'
# salud: una letra por subsistema (V/A/R, o C = detector calentando) en el orden de SUBSISTEMAS.
# excluido: vacio = paso valido; si no, el motivo (el paso queda fuera del analisis).
# ajeno: 1 = movimiento ajeno (Tarea 2); n1_uv: amplitud de la N1 visual del paso (vacio si la
# epoca no sirve); iic: indice de integracion corporal acumulado hasta ese paso (EXPLORATORIO).
MOTIVOS_EXCLUSION = ['pausa:eeg', 'pausa:canal', 'pausa:ortesis', 'sin_ack', 'epoca_invalida', 'ajeno']


# ============================ Tiempos (s) ============================
CICLO_S     = 2.1
VENTANA_MI  = 2.0            # ventana de decision de imaginacion motora
# espera extra tras la senal antes del primer paso de cada ensayo: la ventana del primer paso
# ya no empieza con la transicion mental (gemelo, 8 sujetos: el error del primer paso baja de
# 0.22 a 0.15, ~80 % de lo que se gana esperando 2 s)
ESPERA_PRIMER_PASO_S = 1.0
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
# la ortesis casi nunca cerraba completa (simulador: 27 % de los ensayos de cerrar). Con cada
# ensayo desde el punto medio y paso maximo 0.30: 68 % cierra y 75 % abre, mismo error
PASO_MAX  = 0.30
GANANCIA_PASO = 0.30            # paso = ganancia * |2 p' - 1|, entre PASO_VISIBLE y PASO_MAX
CENTRAR_ENSAYO = True           # cada ensayo empieza con la ortesis en el punto medio
PUNTO_MEDIO = 0.5
CENTRADO_DURACION_MS = 400      # termina antes de la ventana de MI del primer paso (cue + 1 s)
# Un paso que no mueve la ortesis de forma visible (ya estaba en el tope, o le faltaba menos de
# PASO_VISIBLE) no tiene nada que ver: no hay ErrP que leer. Con True, ese paso cuenta como
# decision en el analisis, pero el agente no aprende de el y no cuenta como deteccion fallida
# para la confianza del detector. False: lo de antes (toda decision se lee como un movimiento).
IGNORAR_SIN_MOVIMIENTO = True

# ============================ Embodiment (Tarea 2, EXPLORATORIO) ============================
# En el lazo adaptativo, uno de cada AJENOS_CADA pasos es un movimiento ajeno: la pantalla lo
# anuncia (AUTOMATICO), la ortesis se mueve sola hacia la meta y el agente no aprende de el.
# Va en el 2o paso de uno de cada dos ensayos: tras el centrado y un paso propio siempre cabe.
AJENOS_CADA = 10
PASO_AJENO = 0.15
AVISO_AJENO_S = 1.0             # el aviso en pantalla antes del movimiento ajeno
# firma principal: atenuacion sensorial de la N1 visual (literatura: N1 occipital ~150-200 ms)
CANALES_N1 = ['PO7', 'Oz', 'PO8']
VENTANA_N1 = (0.14, 0.20)       # s desde el inicio del movimiento
IIC_MIN_EPOCAS = (20, 5)        # (propias correctas, ajenas) para estimar el IIC
CUESTIONARIO = ['Senti la ortesis como parte de mi mano.',                      # propiedad
                'Senti que yo causaba los movimientos de la ortesis.',          # agencia
                'Pude mover la ortesis hacia donde queria.']                    # control

# ============================ Umbrales go / no go ============================
# checkpoint 1: el Unicorn no mide impedancias; se revisa la calidad de senal por canal y la
# latencia del ACK con metricas robustas (un pico aislado no tumba el CP1, un jitter tipico si)
CP1_MOVIMIENTOS = 40
CP1_MAD_MAX_MS = 15.0          # desviacion absoluta mediana de la latencia
CP1_P95_MAX_MS = 60.0          # percentil 95 de la latencia
CP1_ACK_PERDIDOS_MAX = 0.10     # fraccion de movimientos sin ACK
IMPEDANCIA_MAX_KOHM = 20.0      # solo para puente_lsl.py --placa cyton --impedancias
MI_EXACTITUD_MIN = 0.70         # checkpoint 2
BA_MIN           = 0.75         # checkpoint 3
ESPEC_MIN        = 0.90
RECUPERACION_MAX_S = 120.0      # checkpoint 4
PERTURBACION_LOGITS = 2.4

# ============================ Control causal con sham (--sham) ============================
# Dos bloques adaptativos del mismo largo, 'real' y 'sham', en orden al azar; cada uno arranca con
# el agente reiniciado (beta, varianza y prior) y recibe su perturbacion en el mismo paso. En el
# sham el agente aprende igual de rapido (fiabilidad fija en la calibrada, sin congelar) de una
# senal que no dice nada del error de cada paso (agente_errp.SenalSham). La idea de un bloque sham
# dentro de la sesion es de jusren (rama b1-b2-sham-errp); esta implementacion es otra.
BLOQUES_SHAM = ('real', 'sham')         # columna 'bloque' del CSV y marcadores bloque:real / bloque:sham
SHAM_ERRP_PASOS = 60                    # por bloque: los dos caben en ~4 min a CICLO_S
SHAM_ERRP_PERTURBAR_EN = 10             # paso del bloque en que entra su perturbacion (tras 2 ensayos)
# Que recibe el agente en el bloque sham (medido en estudios/sham_gemelo.py):
#   nula       sin evidencia del ErrP: la tasa base de la calibracion (LLR = 0)
#   recientes  los p_errp del mismo bloque, permutados entre los ultimos SHAM_ERRP_MEMORIA pasos.
#              OJO: conserva la TASA de ErrP, y con las decisiones cargadas a un lado tras la
#              perturbacion la tasa sola ya dice hacia donde corregir (gemelo: se recupera 11 de 16)
SHAM_ERRP_FUENTES = ('nula', 'recientes')
SHAM_ERRP_FUENTE = 'nula'
SHAM_ERRP_MEMORIA = 8

# ============================ Controles de especificidad (ideas de jusren) ============================
# Dos controles que propuso jusren en la rama b1-b2-sham-errp; aqui con implementacion propia.
# 1. ErrP por direccion (siempre, al calibrar): si el detector da mas falsas alarmas cuando la
#    ortesis cierra que cuando abre (o al reves), el agente corregiria hacia un lado sin que el
#    piloto haya visto un error. Se decide con la prueba exacta de Fisher; solo avisa.
ERRP_DIRECCION_ALFA = 0.05
# 2. Control de reposo (--control-reposo): el piloto no imagina nada y la ortesis se mueve sola;
#    p(t) del decoder no debe seguirla (si la sigue, lee los servos, los cables o la respuesta
#    visual, no la intencion). Prueba de Mann-Whitney de p contra la direccion; solo avisa.
CONTROL_REPOSO = 'control_reposo'       # marcador del inicio del control
REPOSO_MOVIMIENTOS = 40                 # mitad a cerrar, mitad a abrir
# la ventana de MI termina este tiempo despues del movimiento: lo que pasa en el lazo, donde la
# ventana de un paso alcanza al movimiento del paso anterior (mediana medida entre pasos: 0.87 s)
REPOSO_DESPUES_S = 0.9
REPOSO_ALFA = 0.05
REPOSO_MIN_POR_DIRECCION = 8            # con menos movimientos validos por direccion no se decide

# ============================ Ortesis (USB serial) ============================
PUERTO_ORTESIS = 'COM4'
BAUDIOS        = 115200
DURACION_PASO_MS = 250
# inicio real del movimiento (telemetria T del ESP32): el angulo se aleja del previo mas que esto
UMBRAL_INICIO_ANGULO = 5          # en unidades del firmware (0-1000)
# movimiento de cabeza (giroscopio del Unicorn): por encima de esto, la epoca o la ventana es
# artefacto (el agente no aprende de ese paso, el decoder no se recentra); sin pausa
GIRO_ARTEFACTO_DPS = 20.0
TELEMETRIA_HZ = 50
LATENCIA_MECANICA_SIM_MS = (30.0, 150.0)   # la de la ortesis simulada (y la que usa el gemelo)
# PC -> ESP32:  "M,<seq>,<angulo 0-1000>,<duracion_ms>\n"
# ESP32 -> PC:  "A,<seq>,<t_us>\n"      ACK al aplicar el primer pulso
#               "T,<t_us>,<angulo>,<fsr>\n"   telemetria a 50 Hz

# ============================ Salud ============================
# piloto: alfa occipital contra su linea base (somnolencia, ojos cerrados o desconexion de la
# tarea). Solo avisa: no pausa, no excluye pasos y no cuenta en la escalera de degradacion.
SUBSISTEMAS = ['eeg', 'ortesis', 'reloj', 'detector', 'piloto']
VERDE, AMARILLO, ROJO = 'VERDE', 'AMARILLO', 'ROJO'
CALENTANDO = 'CALENTANDO'       # detector y piloto: aun no hay con que juzgarlos (gris en el tablero)
BANDA_ALFA = (8.0, 13.0)
SALUD = {
    'eeg_edad_amarillo_s': 0.3, 'eeg_edad_rojo_s': 1.0,     # edad de la ultima muestra
    'eeg_tasa_amarillo': 0.10, 'eeg_tasa_rojo': 0.25,       # desviacion relativa de la tasa real
    'hueco_max_s': 0.02,                                    # salto entre muestras que cuenta como corte
    # la ventana de imaginacion motora tolera perdidas de Bluetooth chicas (se interpolan):
    'mi_perdida_max': 0.10, 'mi_hueco_max_s': 0.25,         # fraccion de muestras y hueco mas largo
    # saturado: el Unicorn mide +-750 mV (dato de g.tec, por confirmar con verificar_unicorn.py)
    'canal_plano_uv': 0.1, 'canal_saturado_uv': 700_000.0, 'canal_ruidoso_uv': 100.0,
    'ventana_canales_s': 2.0,
    'acks_amarillo': 1, 'acks_rojo': 3,                     # ACK perdidos consecutivos
    'latencia_pico_ms': 80.0,
    'reloj_amarillo_ms': 20.0, 'reloj_rojo_ms': 50.0,       # deriva del retraso contra su linea base
    'reloj_lecturas_base': 40,                              # lecturas para (re)medir la linea base
    'detector_amarillo': 0.7,                               # fiabilidad bajo este valor
    'detector_epocas_min': 15,                              # epocas validas antes de opinar del detector
    # piloto: potencia alfa de PO7/Oz/PO8 en los ultimos piloto_ventana_s, dividida entre la
    # mediana de sus primeros piloto_base_s. Umbrales sin validar en personas: solo avisan
    'piloto_base_s': 60.0, 'piloto_ventana_s': 20.0, 'piloto_amarillo': 1.5, 'piloto_rojo': 2.5,
    'verde_para_reanudar_s': 3.0,                           # VERDE continuo para salir de la pausa
}
POSICION_SEGURA   = 0.0          # abierta
PAUSA_DURACION_MS = 1500         # abrir despacio
RECONEXION_INICIAL_S, RECONEXION_MAX_S = 0.5, 8.0           # retroceso exponencial
CAL_REPETICIONES_MAX = 3         # repeticiones de un ensayo de calibracion afectado por una falla
COADAPTAR_CADA = 20             # epocas nuevas del lazo entre re-entrenamientos del detector
COADAPTAR_PRUEBA = 30           # epocas con que se prueba en sombra un modelo nuevo (20 era muy ruidoso)

# ============================ Caos ============================
# "Caos estandar": fallas que inyecta --caos <semilla> (caos.py). cada_s = separacion media.
CAOS_ESTANDAR = {
    'corte_eeg':        {'cada_s': 45.0,  'duracion_s': (1.0, 5.0), 'recrear_desde_s': 3.0},
    'rafaga_parpadeos': {'cada_s': 40.0,  'duracion_s': (2.0, 4.0), 'por_segundo': 3.0},
    'canal':            {'cada_s': 120.0, 'duracion_s': (4.0, 10.0)},   # un canal se despega
    'ack_perdido':      {'p': 0.03},                                    # por paso
    'pico_latencia':    {'p': 0.05, 'ms': (80.0, 300.0)},               # por paso
    'perdida_bt':       {'cada_s': 20.0,  'duracion_s': (0.02, 0.2)},   # muestras que no llegan por Bluetooth
}
# "Caos leve": mismas fallas, del orden de una cada 2 a 3 minutos sumando todos los tipos.
CAOS_LEVE = {
    'corte_eeg':        {'cada_s': 900.0, 'duracion_s': (1.0, 5.0), 'recrear_desde_s': 3.0},
    'rafaga_parpadeos': {'cada_s': 900.0, 'duracion_s': (2.0, 4.0), 'por_segundo': 3.0},
    'canal':            {'cada_s': 1200.0, 'duracion_s': (4.0, 10.0)},
    'perdida_bt':       {'cada_s': 450.0, 'duracion_s': (0.02, 0.2)},
    'ack_perdido':      {'p': 0.002},
    'pico_latencia':    {'p': 0.003, 'ms': (80.0, 300.0)},
}
CAOS = {'estandar': CAOS_ESTANDAR, 'leve': CAOS_LEVE}                   # --caos-nivel

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
# junto al CSV de cada sesion: todo lo que se publico en el flujo Estado, una linea JSON por
# evento con su hora ({'t': ..., 'evento': {...}}). Lo usa repetir_sesion.py (plan B)
SUFIJO_ESTADO = '_estado.jsonl'
# modelos calibrados hace mas que esto: aviso al cargarlos (pueden ser de otro piloto o del gemelo)
MODELOS_EDAD_AVISO_H = 6.0

# ============================ IA (copiloto, co-investigador y narrador) ============================
# Reglas en ia.py. Todo esto esta apagado por defecto y nada corre dentro del lazo de control.
ARCHIVO_ENV = RAIZ / '.env'             # ANTHROPIC_API_KEY=...  (en .gitignore)
IA_MODELO = 'claude-opus-5-5'           # modelo vigente al 3 de octubre de 2026 (guia oficial de la API)
IA_ESFUERZO = 'medium'                  # output_config.effort: low | medium | high | xhigh | max
IA_MAX_TOKENS = 16000
IA_TIEMPO_MAX_S = 60.0                  # por peticion; el narrador usa uno mucho mas corto
IA_MAX_VUELTAS = 8                      # peticiones de herramientas por pregunta
IA_MAX_LISTA = 200                      # una lista de numeros mas larga no se envia: seria una senal cruda
COPILOTO_MAX_FILAS = 40                 # filas del CSV que devuelve la herramienta pasos() de una vez
PASO_CHICO = 0.15                       # |delta| menor que esto es un paso "chico" (pasos sin ErrP por tamano)
# Propuestas (co-investigador entre bloques y proxima sesion del copiloto): esquema fijo en
# ia.ESQUEMA_PROPUESTA. Un parametro que no esta aqui no se puede proponer, y un valor fuera de su
# rango se rechaza antes de mostrarselo a nadie. Nada se aplica sin la aprobacion de una persona.
ACCIONES_PROPUESTA = ('continuar', 'pausa', 'ajustar_parametro', 'recalibrar')
PARAMETROS_PROPUESTA = {
    'paso_visible': (0.05, 0.12),       # ConfigAgente.paso_visible: el paso minimo que se le muestra al piloto
    'paso_max':     (0.15, 0.35),       # ConfigAgente.paso_max: nunca mas de un tercio del recorrido por paso
    'ganancia':     (0.15, 0.45),       # ConfigAgente.ganancia
    'ajenos_cada':  (0, 20),            # movimientos ajenos de la Tarea 2 (0 = ninguno)
    'pausa_s':      (30, 300),          # solo para la accion 'pausa': descanso con la ortesis abierta
}
PROPUESTA_MAX_JUSTIFICACION = 600
# co-investigador entre bloques (orquestador.py --coinvestigador)
COINVESTIGADOR_API_S = 20.0             # lo mas que se espera a la API antes de usar las reglas
COINVESTIGADOR_ESPERA_S = 60.0          # lo mas que se espera la decision del operador; sin decision, nada cambia
SUFIJO_PROPUESTAS = '_propuestas.jsonl'  # junto al CSV: cada propuesta, su decision y su efecto
# umbrales de las reglas deterministas (ia.propuesta_por_reglas), las que se usan sin API
REGLAS_PROPUESTA = {
    'pasos_min': 30,                    # con menos pasos validos no se propone nada
    'excluidos_max': 0.25, 'pausa_s': 60,
    'ba_recalibrar': 0.60, 'congelado_max': 0.5,
    'errores_min': 8, 'dif_sin_errp': 0.25, 'paso_visible_incremento': 0.02,
}
