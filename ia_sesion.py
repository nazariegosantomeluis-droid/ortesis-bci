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
necesariamente falso (puede ser una resta o un porcentaje derivado): se lista para que una persona lo revise.

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


def auditar_texto(texto, s, u=None):
    """Revisa las cifras y los pasos que cita un texto. Devuelve {'cifras', 'verificadas', 'no_encontradas', 'pasos_citados',
    'pasos_inexistentes'}. Una cifra es 'verificada' si coincide con algun numero real (o su version en %) dentro del redondeo."""
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
    r = ia.proponer(resumen, cli)
    p = r['propuesta']
    reglas = ia.propuesta_por_reglas(resumen)
    ok_rangos, motivo = ia.validar_propuesta(p)
    out['coinvestigador'] = {'origen': r['origen'], 'propuesta': p, 'valida': bool(ok_rangos), 'motivo': motivo,
                             'coincide_con_reglas': p == reglas, 'reglas': reglas, 'resumen': resumen,
                             'rechazada_api': r.get('rechazada_api'),
                             'auditoria_justificacion': auditar_texto(str(p.get('justificacion', '')), s, u)}
    informe = copiloto.informe(s, anteriores=anteriores, cli=cli)
    arch = []
    for f in informe['archivos']:
        a = {'archivo': Path(f).name}
        if str(f).endswith('.md'):
            a['auditoria'] = auditar_texto(Path(f).read_text(encoding='utf-8'), s, u)
        arch.append(a)
    out['informe'] = {'archivos': arch, 'origen_texto': informe['origen_texto'],
                      'propuesta_proxima_sesion': informe['propuesta']['propuesta']}
    return out


def _linea_auditoria(a):
    return (f"{len(a['verificadas'])} de {a['cifras']} cifras verificadas" + (f"; sin encontrar: {', '.join(a['no_encontradas'])}" if a['no_encontradas'] else '')
            + (f"; pasos que NO existen: {a['pasos_inexistentes']}" if a['pasos_inexistentes'] else '')
            + (f"; pasos citados: {a['pasos_citados']}" if a['pasos_citados'] else ''))


def a_markdown(o):
    L = [f"# IA sobre la sesión `{o['sesion']}` ({o['pasos']} pasos)", '',
         f"- Copiloto y co-investigador: **{'con la API de Claude' if o['api'] else 'sin API (plantillas y reglas deterministas)'}**.",
         '- Nada se aplicó ni se cambió: la propuesta del co-investigador y la de la próxima sesión siguen pendientes de una persona.', '',
         '## Copiloto: 6 preguntas', '']
    for i, q in enumerate(o['preguntas'], 1):
        L += [f"### {i}. {q['pregunta']}", '', f"*origen: {q['origen']}*", '', f"> {q['respuesta']}", '', f"**Auditoría:** {_linea_auditoria(q['auditoria'])}"
              + (' · responde «no hay dato»' if q['sin_dato'] else ''), '']
    c = o['coinvestigador']
    p = c['propuesta']
    L += ['## Co-investigador', '', f"- Origen: {c['origen']}. Propuesta: **{p.get('accion')}**" + (f" `{p.get('parametro')}` = {p.get('valor')}" if p.get('parametro') else '')
          + f". Justificación: {p.get('justificacion')}",
          f"- ¿Válida? {'sí' if c['valida'] else 'NO: ' + str(c['motivo'])} (rangos seguros de `config.PARAMETROS_PROPUESTA`).",
          f"- ¿Coincide con las reglas deterministas? {'sí' if c['coincide_con_reglas'] else 'no: las reglas proponían ' + str(c['reglas'].get('accion'))}.",
          f"- Cifras de su justificación: {_linea_auditoria(c['auditoria_justificacion'])}.",
          f"- Resumen que recibió: error del agente {c['resumen'].get('error_agente')}, sombra {c['resumen'].get('error_sombra')}, BA viva {c['resumen'].get('ba_viva')}, "
          f"fracción congelado {c['resumen'].get('fraccion_congelado')}, excluidos {c['resumen'].get('excluidos')}.", '',
          '## Informe clínico', '', f"- Texto de interpretación: {o['informe']['origen_texto']}. Propuesta para la próxima sesión: `{o['informe']['propuesta_proxima_sesion']}`.", '']
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
