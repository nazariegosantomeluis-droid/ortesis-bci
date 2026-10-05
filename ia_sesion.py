"""IA sobre una sesion real (5 de octubre): corre el copiloto, el co-investigador y el informe clinico sobre una sesion
grabada y AUDITA que lo que dicen cite datos que existen. No cambia nada de la sesion ni aplica ninguna propuesta.

  1. El copiloto responde 6 preguntas reales (recuperacion, agente contra sombra, congelamiento, fatiga, excluidos,
     comparacion con la sesion anterior). Con llave en .env responde Claude con las herramientas; sin llave o sin
     conexion, las plantillas (el informe lo dice por pregunta: `origen`).
  2. El co-investigador propone (ia.proponer) sobre el resumen agregado de la sesion: se valida contra los rangos seguros
     de config.PARAMETROS_PROPUESTA, se compara con las reglas deterministas y se revisan las cifras de su justificacion.
  3. El informe clinico (copiloto.informe): 4 Markdown, figuras y la propuesta para la proxima sesion, pendiente de una persona.

AUDITORIA de cifras (`auditar_texto`): cada numero de una respuesta debe existir en lo que las herramientas devuelven para
esa sesion (resumen, metricas por bloque, eventos, comparacion, filas del CSV y los umbrales de config), con la tolerancia
del redondeo con que se escribio; y todo «paso N» debe ser un paso que existe. Un numero que no se encuentra NO es
necesariamente falso (puede ser una resta o un porcentaje derivado): se lista para que una persona lo revise. Y uno que
se encuentra NO esta necesariamente bien atribuido: «verificada» solo dice que la cifra existe en los datos de la sesion
(con la API real, en 256 numeros, un entero chico como el «42 pasos» de una resta coincide por azar, y un porcentaje
entero como «35 %» de 0.353 no, porque los enteros se comparan exactos).

Uso:  python ia_sesion.py --sesion resultados/sesion_real_<fecha>.csv [--sin-ia] [--salida informe.md]
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np

import config
import copiloto
import ia

PREGUNTAS = ('¿Cuánto tardó el agente en recuperarse tras la perturbación?',
             '¿El agente le ganó a la sombra y con qué certeza?',
             '¿Por qué se congeló el aprendizaje, y cuándo?',
             '¿Hubo señales de fatiga o somnolencia del piloto?',
             '¿Cuántos pasos se excluyeron y por qué?',
             '¿Cómo se compara esta sesión con la anterior?')
# numeros que una respuesta puede decir sin sacarlos de la sesion: umbrales y criterios del proyecto
CONSTANTES = (0.5, 0.7, 0.75, 0.9, 0.05, 0.1, 90, 70, 100, 5, 10, 2.4, 1.68, 2, 3, 4, 0, 1, 20, 40, 60, 120)


def _plano(x, salida):
    """Todos los numeros de una estructura de herramientas (dicts, listas, tuplas)."""
    if isinstance(x, bool) or x is None:
        return
    if isinstance(x, (int, float, np.integer, np.floating)):
        if np.isfinite(x):
            salida.add(float(x))
    elif isinstance(x, dict):
        for v in x.values():
            _plano(v, salida)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _plano(v, salida)
    elif isinstance(x, str):
        for n in re.findall(r'-?\d+(?:\.\d+)?', x):
            salida.add(float(n))


def universo(s):
    """Los numeros reales de la sesion: lo que devuelven las herramientas agregadas del copiloto (resumen, metricas por bloque,
    eventos, comparacion). Las filas del CSV NO entran: con cientos de valores de tres decimales casi cualquier cifra inventada
    coincidiria con alguno; lo que una respuesta cite de un paso suelto se verifica con lo que devolvio la herramienta `pasos`
    (con la API, `usadas`)."""
    u = set(float(c) for c in CONSTANTES)
    paquetes = [s.resumen_sesion(), s.eventos_de()]
    for b in copiloto.BLOQUES:
        for m in copiloto.METRICAS:
            try:
                paquetes.append(s.metrica(m, b))
            except Exception:
                pass
    try:
        paquetes.append(s.comparar_sesiones())
    except Exception:
        pass
    for p in paquetes:
        _plano(p, u)
    return u


def _decimales(txt):
    return len(txt.split('.')[1]) if '.' in txt else 0


MENOS_UNICODE = chr(0x2212)          # U+2212, el «−» que escribe Claude en vez del '-' ASCII (se ven igual: por eso va por codigo)


def normalizar_cifras(txt):
    """Deja las cifras de un texto en la forma que lee `auditar_texto`: el signo menos de Unicode pasa a '-' y la coma decimal
    del espanol (0,3227; 71,4 s) a punto. Visto con la API real el 5 de octubre: Claude contesta en espanol con coma, y sin esto
    «0,3227» se leia como 0 y 3227 (un 0 que siempre «se verifica» y un 3227 que nunca). Una coma entre dos digitos es siempre
    decimal; las listas («pasos 29, 34») llevan espacio tras la coma. Un millar con coma en ingles («1,350») se leeria como 1.35."""
    return re.sub(r'(?<=\d),(?=\d)', '.', txt.replace(MENOS_UNICODE, '-'))


def auditar_texto(texto, s, u=None):
    """Revisa las cifras y los pasos que cita un texto. Devuelve {'cifras', 'verificadas', 'no_encontradas', 'pasos_citados',
    'pasos_inexistentes'}. Una cifra es 'verificada' si coincide con algun numero real (o su version en %) dentro del redondeo."""
    texto = normalizar_cifras(texto)
    u = u if u is not None else universo(s)
    arr = np.array(sorted(u)) if u else np.array([])
    pasos = [int(n) for n in re.findall(r'paso[s]?\s+(?:de la perturbaci[oó]n\s+)?(\d+)', texto, flags=re.I)]
    pasos += [int(a) for ab in re.findall(r'pasos?\s+(\d+)\s*(?:-|–|a)\s*(\d+)', texto, flags=re.I) for a in ab]
    inexistentes = sorted({n for n in pasos if not 1 <= n <= len(s.filas)})
    ver, falta = [], []
    for m in re.finditer(r'(?<![\w.])([+-]?\d+(?:\.\d+)?)\s*(%?)', texto):
        crudo, pct = m.group(1), m.group(2)
        # fechas, horas y nombres de archivo no son cifras de la sesion
        if re.match(r'\d{4}\d{2}', crudo) or (m.start() and texto[m.start() - 1] in '_:/'):
            continue
        v = float(crudo)
        tol = 0.5 * 10 ** -_decimales(crudo) if '.' in crudo else 1e-9        # un entero se dijo exacto; un decimal, con el redondeo con que se escribio
        candidatos = [v / 100.0, v] if pct else [v, v / 100.0 if abs(v) > 1 else v]
        ok = bool(arr.size) and any(np.abs(arr - c).min() <= (tol / 100.0 if pct and c == v / 100.0 else tol) + 1e-9 for c in candidatos)
        (ver if ok else falta).append(crudo + pct)
    return {'cifras': len(ver) + len(falta), 'verificadas': ver, 'no_encontradas': falta,
            'pasos_citados': sorted(set(pasos)), 'pasos_inexistentes': inexistentes}


def auditar(ruta_csv, cli=None, anteriores=None):
    """Corre las tres cosas sobre una sesion y devuelve todo lo que hace falta para el informe."""
    s = copiloto.Sesion(Path(ruta_csv))
    u = universo(s)
    out = {'sesion': s.ruta.name, 'pasos': len(s.filas), 'api': cli is not None, 'preguntas': []}
    # una pregunta de congelamiento con el primer paso congelado de la sesion, si lo hubo
    cong = s.eventos_de(tipo='congelamiento')
    preguntas = list(PREGUNTAS)
    if isinstance(cong, list) and cong:
        preguntas[2] = f"¿Por qué se congeló el aprendizaje en el paso {cong[0]['paso']}?"
    for q in preguntas:
        txt, origen, usadas = copiloto.responder(s, q, cli)
        uq = set(u)
        _plano([x[2] for x in usadas], uq)                       # con la API, tambien lo que devolvieron las herramientas que uso
        out['preguntas'].append({'pregunta': q, 'respuesta': txt, 'origen': origen, 'herramientas': [x[0] for x in usadas],
                                 'auditoria': auditar_texto(txt, s, uq), 'sin_dato': copiloto.SIN_DATO in txt})
    resumen = copiloto.resumen_para_propuesta(s)
    # La propuesta (y el informe, que la lleva) se redacta sobre ESTE resumen, no sobre las herramientas: sus cifras (exactitud, fracciones)
    # tambien son reales. Visto con la API real: «exactitud 0.833» (= 1 - error) solo estaba en el resumen y salia como no encontrada.
    u_propuesta = set(u)
    _plano(resumen, u_propuesta)
    r = ia.proponer(resumen, cli)
    p = r['propuesta']
    reglas = ia.propuesta_por_reglas(resumen)
    ok_rangos, motivo = ia.validar_propuesta(p)
    # «coincide» compara la DECISION (accion, parametro y valor), no la redaccion: la justificacion de la API nunca es la de las reglas
    out['coinvestigador'] = {'origen': r['origen'], 'propuesta': p, 'valida': bool(ok_rangos), 'motivo': motivo,
                             'coincide_con_reglas': all(p.get(k) == reglas.get(k) for k in ('accion', 'parametro', 'valor')),
                             'reglas': reglas, 'resumen': resumen, 'rechazada_api': r.get('rechazada_api'),
                             'auditoria_justificacion': auditar_texto(str(p.get('justificacion', '')), s, u_propuesta)}
    informe = copiloto.informe(s, anteriores=anteriores, cli=cli)
    arch = []
    for f in informe['archivos']:
        a = {'archivo': Path(f).name}
        if str(f).endswith('.md'):
            a['auditoria'] = auditar_texto(Path(f).read_text(encoding='utf-8'), s, u_propuesta)
        arch.append(a)
    out['informe'] = {'archivos': arch, 'origen_texto': informe['origen_texto'],
                      'propuesta_proxima_sesion': informe['propuesta']['propuesta'],
                      'origen_propuesta': informe['propuesta']['origen'], 'rechazada_api': informe['propuesta'].get('rechazada_api')}
    return out


def _linea_auditoria(a):
    return (f"{len(a['verificadas'])} de {a['cifras']} cifras verificadas" + (f"; sin encontrar: {', '.join(a['no_encontradas'])}" if a['no_encontradas'] else '')
            + (f"; pasos que NO existen: {a['pasos_inexistentes']}" if a['pasos_inexistentes'] else '')
            + (f"; pasos citados: {a['pasos_citados']}" if a['pasos_citados'] else ''))


def _r3(v):
    return round(v, 3) if isinstance(v, float) else v


def _decision(p):
    """«continuar», o «ajustar_parametro paso_visible = 0.1»: la decision de una propuesta sin su justificacion."""
    return str(p.get('accion')) + (f" {p.get('parametro')} = {p.get('valor')}" if p.get('parametro') else '')


def _motivo_api(rechazada):
    """Por que la API no dio una propuesta aceptable (o '' si la dio), para que una caida a reglas no pase en silencio."""
    return '' if not rechazada else f" La API no dio una propuesta aceptable y se usaron las reglas: {str(rechazada.get('motivo'))[:300]}."


def a_markdown(o):
    n_api = sum(q['origen'] == 'api' for q in o['preguntas'])
    L = [f"# IA sobre la sesión `{o['sesion']}` ({o['pasos']} pasos)", '',
         f"- Copiloto y co-investigador: **{'con la API de Claude' if o['api'] else 'sin API (plantillas y reglas deterministas)'}**."
         + (f" Preguntas respondidas por la API: {n_api} de {len(o['preguntas'])}." if o['api'] else ''),
         '- Nada se aplicó ni se cambió: la propuesta del co-investigador y la de la próxima sesión siguen pendientes de una persona.',
         '- «Verificada» quiere decir que la cifra existe en los datos de la sesión; no prueba que esté bien atribuida (un entero chico puede coincidir por azar). '
         '«Sin encontrar» no es necesariamente un error (una resta, un porcentaje redondeado): se revisa a mano.', '',
         '## Copiloto: 6 preguntas', '']
    for i, q in enumerate(o['preguntas'], 1):
        L += [f"### {i}. {q['pregunta']}", '', f"*origen: {q['origen']}" + (f" · herramientas: {', '.join(q['herramientas'])}" if q['herramientas'] else '') + '*',
              '', f"> {q['respuesta']}", '', f"**Auditoría:** {_linea_auditoria(q['auditoria'])}"
              + (' · responde «no hay dato»' if q['sin_dato'] else ''), '']
    c = o['coinvestigador']
    p = c['propuesta']
    L += ['## Co-investigador', '', f"- Origen: {c['origen']}. Propuesta: **{p.get('accion')}**" + (f" `{p.get('parametro')}` = {p.get('valor')}" if p.get('parametro') else '')
          + f". Justificación: {p.get('justificacion')}" + _motivo_api(c.get('rechazada_api')),
          f"- ¿Válida? {'sí' if c['valida'] else 'NO: ' + str(c['motivo'])} (rangos seguros de `config.PARAMETROS_PROPUESTA`).",
          f"- ¿Coincide con la decisión de las reglas deterministas? {'sí' if c['coincide_con_reglas'] else 'no: las reglas proponían ' + _decision(c['reglas'])}.",
          f"- Cifras de su justificación: {_linea_auditoria(c['auditoria_justificacion'])}.",
          f"- Resumen que recibió: error del agente {c['resumen'].get('error_agente')}, sombra {c['resumen'].get('error_sombra')}, BA viva {c['resumen'].get('ba_viva')}, "
          f"fracción congelado {_r3(c['resumen'].get('fraccion_congelado'))}, excluidos {c['resumen'].get('excluidos')}.", '',
          '## Informe clínico', '', f"- Texto de interpretación: {o['informe']['origen_texto']}. Propuesta para la próxima sesión "
          f"({o['informe'].get('origen_propuesta', '?')}): `{o['informe']['propuesta_proxima_sesion']}`." + _motivo_api(o['informe'].get('rechazada_api')), '']
    for a in o['informe']['archivos']:
        L.append(f"- `{a['archivo']}`" + (f": {_linea_auditoria(a['auditoria'])}" if 'auditoria' in a else ''))
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description='Copiloto, co-investigador e informe clinico sobre una sesion, con auditoria de cifras')
    ap.add_argument('--sesion', required=True, help='resultados/sesion_real_<fecha>.csv')
    ap.add_argument('--sin-ia', dest='sin_ia', action='store_true', help='solo plantillas y reglas, aunque haya llave')
    ap.add_argument('--anteriores', nargs='*', help='CSV de las sesiones anteriores (por defecto, las 3 previas)')
    ap.add_argument('--salida')
    a = ap.parse_args(argv)
    cli = None if a.sin_ia else ia.cliente()
    o = auditar(a.sesion, cli, a.anteriores)
    md = a_markdown(o)
    ruta = Path(a.salida) if a.salida else Path(a.sesion).with_name(Path(a.sesion).stem + '_ia.md')
    ruta.write_text(md, encoding='utf-8')
    print(md)
    print(f'Informe: {ruta}')


if __name__ == '__main__':
    main()
