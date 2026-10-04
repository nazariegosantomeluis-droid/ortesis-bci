"""Lo comun a toda la IA del proyecto (copiloto, co-investigador y narrador).

Reglas (TAREAS.md, "Bloque final"):
  - La llave va en ANTHROPIC_API_KEY, leida de un .env que esta en .gitignore.
  - A la API solo van metricas agregadas y anonimas: nunca EEG crudo ni nombres (sanear()).
  - Nada de esto corre dentro del lazo de control: el orquestador no importa este modulo en un paso.
  - Sin llave, sin conexion o sin el paquete `anthropic`, cliente() devuelve None y quien llama
    usa reglas o plantillas deterministas.
  - Ninguna propuesta se aplica sola: validar_propuesta() la revisa contra los rangos seguros de
    config y despues la aprueba o la rechaza una persona.

Las pruebas usan una API simulada: cualquier objeto con .messages.create(**kw) que devuelva
bloques con .type, .text, .name, .input e .id, y .stop_reason.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import config


class SinRespuesta(Exception):
    """La API no dio una respuesta util (declino, se corto o dio demasiadas vueltas)."""


def cargar_env(ruta=None):
    """Pone en el entorno las variables del .env (CLAVE=valor por linea), sin pisar las que ya estan."""
    ruta = Path(ruta or config.ARCHIVO_ENV)
    if not ruta.exists():
        return
    for linea in ruta.read_text(encoding='utf-8').splitlines():
        clave, igual, valor = linea.strip().partition('=')
        if igual and clave and not clave.startswith('#'):
            os.environ.setdefault(clave.strip(), valor.strip().strip('"\''))


def cliente(tiempo_max_s=config.IA_TIEMPO_MAX_S):
    """El cliente de la API, o None si no hay llave o no esta instalado `anthropic`."""
    cargar_env()
    if not os.environ.get('ANTHROPIC_API_KEY'):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic(timeout=tiempo_max_s, max_retries=1)


def sanear(dato):
    """Lo unico que sale hacia la API pasa por aqui. Quita las carpetas de las rutas (llevan el
    nombre de usuario de la maquina) y se niega a enviar una lista larga de numeros: seria una
    senal cruda, no una metrica agregada."""
    if isinstance(dato, dict):
        return {str(k): sanear(v) for k, v in dato.items()}
    if isinstance(dato, (list, tuple)):
        if len(dato) > config.IA_MAX_LISTA and all(isinstance(v, (int, float)) for v in dato):
            raise ValueError(f'no se envia a la API una lista de {len(dato)} numeros: parece una senal cruda')
        return [sanear(v) for v in dato]
    if isinstance(dato, Path):
        return dato.name
    if isinstance(dato, str) and ('\\' in dato or '/' in dato) and Path(dato).is_absolute():
        return Path(dato).name
    if isinstance(dato, float):
        return round(dato, 4) if dato == dato else None          # nan -> null
    if dato is None or isinstance(dato, (str, int, bool)):
        return dato
    if hasattr(dato, 'tolist'):                                    # numpy
        return sanear(dato.tolist())
    return str(dato)


def a_json(dato):
    return json.dumps(sanear(dato), ensure_ascii=False)


def texto(respuesta):
    return ''.join(b.text for b in respuesta.content if b.type == 'text').strip()


def conversar(cli, sistema, pregunta, herramientas, ejecutar, max_vueltas=config.IA_MAX_VUELTAS):
    """Una pregunta respondida con herramientas (lazo manual de la API de mensajes).
    ejecutar(nombre, entrada) -> dato; lo que devuelve se sanea antes de enviarlo.
    Devuelve (texto de la respuesta, [(nombre, entrada, resultado)] en el orden en que se usaron)."""
    mensajes, usadas = [{'role': 'user', 'content': pregunta}], []
    for _ in range(max_vueltas):
        r = cli.messages.create(model=config.IA_MODELO, max_tokens=config.IA_MAX_TOKENS, system=sistema,
                                tools=herramientas, messages=mensajes,
                                output_config={'effort': config.IA_ESFUERZO})
        if r.stop_reason != 'tool_use':
            if r.stop_reason != 'end_turn' or not texto(r):
                raise SinRespuesta(f'la API termino con {r.stop_reason}')
            return texto(r), usadas
        mensajes.append({'role': 'assistant', 'content': r.content})
        resultados = []
        for b in r.content:
            if b.type != 'tool_use':
                continue
            try:
                res, error = sanear(ejecutar(b.name, dict(b.input))), False
            except Exception as e:                                # la herramienta fallo: que el modelo lo sepa
                res, error = {'error': f'{type(e).__name__}: {e}'}, True
            usadas.append((b.name, dict(b.input), res))
            resultados.append({'type': 'tool_result', 'tool_use_id': b.id,
                               'content': json.dumps(res, ensure_ascii=False), 'is_error': error})
        mensajes.append({'role': 'user', 'content': resultados})    # todos los resultados en un solo mensaje
    raise SinRespuesta(f'la API pidio herramientas {max_vueltas} veces sin responder')


def pedir_json(cli, sistema, datos, esquema, esfuerzo=config.IA_ESFUERZO):
    """Una respuesta en JSON con esquema fijo (salida estructurada). datos se sanea antes de enviarse."""
    r = cli.messages.create(model=config.IA_MODELO, max_tokens=config.IA_MAX_TOKENS, system=sistema,
                            messages=[{'role': 'user', 'content': a_json(datos)}],
                            output_config={'effort': esfuerzo, 'format': {'type': 'json_schema', 'schema': esquema}})
    if r.stop_reason != 'end_turn':
        raise SinRespuesta(f'la API termino con {r.stop_reason}')
    try:
        return json.loads(texto(r))
    except ValueError as e:
        raise SinRespuesta(f'la API no devolvio JSON: {e}') from e


# ====================================================================== propuestas
# Esquema fijo de una propuesta (co-investigador entre bloques y proxima sesion del copiloto).
ESQUEMA_PROPUESTA = {
    'type': 'object',
    'properties': {
        'accion': {'type': 'string', 'enum': list(config.ACCIONES_PROPUESTA)},
        'parametro': {'type': ['string', 'null'], 'enum': list(config.PARAMETROS_PROPUESTA) + [None]},
        'valor': {'type': ['number', 'null']},
        'justificacion': {'type': 'string'},
    },
    'required': ['accion', 'parametro', 'valor', 'justificacion'],
    'additionalProperties': False,
}


def validar_propuesta(p):
    """(True, '') si la propuesta cumple el esquema y su valor cae en el rango seguro de config;
    si no, (False, motivo). Una propuesta invalida no se le muestra al operador como aprobable."""
    if not isinstance(p, dict) or set(p) != set(ESQUEMA_PROPUESTA['required']):
        return False, 'no tiene exactamente los campos accion, parametro, valor y justificacion'
    if p['accion'] not in config.ACCIONES_PROPUESTA:
        return False, f"accion desconocida: {p['accion']!r}"
    if not isinstance(p['justificacion'], str) or not p['justificacion'].strip():
        return False, 'falta la justificacion'
    if len(p['justificacion']) > config.PROPUESTA_MAX_JUSTIFICACION:
        return False, 'la justificacion es demasiado larga'
    param = {'pausa': 'pausa_s', 'ajustar_parametro': p['parametro']}.get(p['accion'])
    if param is None:                                             # continuar o recalibrar: sin parametro
        if p['accion'] == 'ajustar_parametro':
            return False, 'falta el parametro'
        if p['parametro'] is not None or p['valor'] is not None:
            return False, f"'{p['accion']}' no lleva parametro ni valor"
        return True, ''
    if p['accion'] == 'pausa' and p['parametro'] not in (None, 'pausa_s'):
        return False, "'pausa' solo admite el parametro pausa_s"
    if param not in config.PARAMETROS_PROPUESTA or (p['accion'] == 'ajustar_parametro' and param == 'pausa_s'):
        return False, f'parametro no ajustable: {param!r}'
    lo, hi = config.PARAMETROS_PROPUESTA[param]
    v = p['valor']
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        return False, 'el valor no es un numero'
    if not lo <= v <= hi:
        return False, f'{param} = {v} esta fuera del rango seguro [{lo}, {hi}]'
    return True, ''


def propuesta_por_reglas(r):
    """Reglas deterministas equivalentes al co-investigador, para cuando no hay API. r es el resumen
    agregado de un bloque o de una sesion (copiloto.resumen_para_propuesta). La primera regla que
    se cumple decide; todas citan el dato que usaron."""
    u = config.REGLAS_PROPUESTA
    def p(accion, justificacion, parametro=None, valor=None):
        return {'accion': accion, 'parametro': parametro, 'valor': valor, 'justificacion': justificacion}
    n = r.get('pasos') or 0
    if n < u['pasos_min']:
        return p('continuar', f'solo hay {n} pasos validos: no alcanza para proponer un cambio')
    exc = r.get('fraccion_excluidos')
    if exc is not None and exc > u['excluidos_max']:
        return p('pausa', f"{exc:.0%} de las filas quedaron excluidas ({r.get('excluidos')}): revisar el casco y la "
                          f"ortesis antes de seguir", 'pausa_s', u['pausa_s'])
    alfa = r.get('alfa_rel')
    if alfa is not None and alfa >= config.SALUD['piloto_amarillo']:
        return p('pausa', f'alfa occipital x{alfa:.1f} de su linea base: posible fatiga o somnolencia',
                 'pausa_s', u['pausa_s'])
    ba, fiab = r.get('ba_viva'), r.get('fiabilidad_media')
    if (ba is not None and ba < u['ba_recalibrar']) or (r.get('fraccion_congelado') or 0) > u['congelado_max']:
        return p('recalibrar', f"el detector de ErrP dejo de informar (BA viva {ba if ba is None else round(ba, 2)}, "
                               f"aprendizaje congelado {r.get('fraccion_congelado') or 0:.0%} del bloque)")
    chicos = (r.get('sin_errp_por_paso') or {}).get('chico')
    grandes = (r.get('sin_errp_por_paso') or {}).get('grande')
    visible = r.get('paso_visible', config.PASO_VISIBLE)
    if chicos and grandes and chicos['errores'] >= u['errores_min'] and grandes['errores'] >= u['errores_min'] \
            and chicos['sin_errp'] - grandes['sin_errp'] > u['dif_sin_errp']:
        nuevo = round(min(visible + u['paso_visible_incremento'], config.PARAMETROS_PROPUESTA['paso_visible'][1]), 2)
        if nuevo > visible:
            return p('ajustar_parametro',
                     f"los errores con pasos chicos pasan sin ErrP {chicos['sin_errp']:.0%} de las veces contra "
                     f"{grandes['sin_errp']:.0%} con pasos grandes: quiza el piloto no los ve", 'paso_visible', nuevo)
    return p('continuar', f"sin motivo para cambiar: error del agente {r.get('error_agente')}, de la sombra "
                          f"{r.get('error_sombra')}, BA viva {ba if ba is None else round(ba, 2)}, fiabilidad media "
                          f"{fiab if fiab is None else round(fiab, 2)}")


SISTEMA_PROPUESTA = (
    'Eres co-investigador de una sesion de interfaz cerebro-computadora: una ortesis de mano controlada por '
    'imaginacion motora, con un agente que se corrige con el potencial de error (ErrP). Recibes un resumen '
    'agregado y anonimo en JSON. Propon UNA accion para lo que sigue, basada solo en esos datos: continuar, '
    'pausa (parametro pausa_s, segundos de descanso), ajustar_parametro (uno solo, con su valor nuevo) o '
    'recalibrar. Rangos seguros de los parametros: ' + json.dumps(config.PARAMETROS_PROPUESTA) + '. '
    'Si los datos no justifican un cambio, propon continuar. La justificacion es breve (una o dos frases), '
    'en espanol, y cita las cifras que usaste. Una persona aprobara o rechazara la propuesta: no se aplica sola.')


def proponer(resumen, cli=None):
    """La propuesta para lo que sigue: de la API si hay cliente y su respuesta es valida; si no, de
    las reglas. Devuelve {'propuesta', 'origen': 'api' | 'reglas', 'valida', 'motivo', 'rechazada_api'}.
    Nunca lanza y nunca aplica nada."""
    rechazada = None
    if cli is not None:
        try:
            cand = pedir_json(cli, SISTEMA_PROPUESTA, resumen, ESQUEMA_PROPUESTA)
            ok, motivo = validar_propuesta(cand)
            if ok:
                return {'propuesta': cand, 'origen': 'api', 'valida': True, 'motivo': '', 'rechazada_api': None}
            rechazada = {'propuesta': cand, 'motivo': motivo}
        except Exception as e:                    # sin conexion, tiempo agotado, respuesta inservible...
            rechazada = {'propuesta': None, 'motivo': f'{type(e).__name__}: {e}'}
    cand = propuesta_por_reglas(resumen)
    ok, motivo = validar_propuesta(cand)
    return {'propuesta': cand, 'origen': 'reglas', 'valida': ok, 'motivo': motivo, 'rechazada_api': rechazada}
