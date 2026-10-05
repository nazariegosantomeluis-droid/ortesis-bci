"""Prueba de humo de la API real: la primera llamada de verdad (hasta hoy todo se probo con la API simulada).

    python ia_humo.py            # necesita ANTHROPIC_API_KEY en el entorno o en .env (esta en .gitignore)

Hace tres peticiones chicas, una por cada forma en que el proyecto usa la API, con datos SINTETICOS
(no son de una persona): una pregunta con herramientas (`ia.conversar`, como el copiloto), una
salida estructurada (`ia.pedir_json` con `ESQUEMA_PROPUESTA`, como el co-investigador) y una frase
del narrador. Dice cual funciono, cuanto tardo y por que fallo la que fallo. Solo lee: no toca
sesiones ni modelos. Cuesta unos centavos."""
from __future__ import annotations

import sys
import time

import config
import ia
import narrador

# Resumen de un bloque inventado, con las llaves que lee ia.propuesta_por_reglas.
RESUMEN_SINTETICO = {
    'pasos': 60, 'excluidos': 4, 'fraccion_excluidos': 0.06, 'alfa_rel': 1.1, 'ba_viva': 0.78,
    'fiabilidad_media': 0.7, 'fraccion_congelado': 0.0, 'paso_visible': config.PASO_VISIBLE,
    'error_agente': 0.24, 'error_sombra': 0.41,
    'sin_errp_por_paso': {'chico': {'errores': 12, 'sin_errp': 0.5}, 'grande': {'errores': 14, 'sin_errp': 0.2}},
}
HECHO_SINTETICO = {'evento': 'checkpoint', 'n': 3, 'ok': True}
HERRAMIENTA_SUMA = [{'name': 'sumar', 'description': 'Suma dos numeros.',
                     'input_schema': {'type': 'object', 'properties': {'a': {'type': 'number'}, 'b': {'type': 'number'}},
                                      'required': ['a', 'b'], 'additionalProperties': False}}]


def humo(cli, salida=print):
    """Corre las tres peticiones. Devuelve [{'prueba', 'ok', 'segundos', 'detalle'}]; nunca lanza."""
    resultados = []

    def una(nombre, fn):
        t = time.time()
        try:
            ok, detalle = fn()
        except Exception as e:
            ok, detalle = False, f'{type(e).__name__}: {str(e)[:300]}'
        resultados.append({'prueba': nombre, 'ok': bool(ok), 'segundos': round(time.time() - t, 1), 'detalle': detalle})
        salida(f"  {'OK   ' if ok else 'FALLA'} {nombre:22s} {resultados[-1]['segundos']:5.1f} s  {detalle}")

    def herramientas():
        txt, usadas = ia.conversar(cli, 'Responde en una frase y usa la herramienta para sumar.',
                                   'Cuanto es 17.5 mas 24.5?', HERRAMIENTA_SUMA,
                                   lambda nombre, e: {'resultado': e['a'] + e['b']})
        return bool(usadas) and '42' in txt, f'herramientas usadas: {[u[0] for u in usadas]}; respuesta: {txt[:80]!r}'

    def estructurada():
        cand = ia.pedir_json(cli, ia.SISTEMA_PROPUESTA, RESUMEN_SINTETICO, ia.ESQUEMA_PROPUESTA)
        ok, motivo = ia.validar_propuesta(cand)
        return ok, (f"propuesta {cand['accion']}" if ok else f'la API respondio pero no es valida: {motivo}: {cand}')

    def frase():
        n = narrador.Narrador('es', cli)
        txt, origen = n.frase(HECHO_SINTETICO)
        return origen == 'api', f'origen {origen}: {txt!r}' + ('' if origen == 'api' else ' (cayo a la plantilla: ver por que con la API a mano)')

    una('preguntas con herramientas', herramientas)
    una('salida estructurada', estructurada)
    una('frase del narrador', frase)
    return resultados


def main():
    cli = ia.cliente()
    if cli is None:
        print('No hay cliente: falta ANTHROPIC_API_KEY (ponla en .env como ANTHROPIC_API_KEY=...) '
              'o el paquete anthropic (pip install -r requirements.txt).')
        return 2
    print(f'Prueba de humo con {config.IA_MODELO} (esfuerzo {config.IA_ESFUERZO}); datos sinteticos.')
    r = humo(cli)
    print(f"{sum(x['ok'] for x in r)}/{len(r)} peticiones bien.")
    return 0 if all(x['ok'] for x in r) else 1


if __name__ == '__main__':
    sys.exit(main())
