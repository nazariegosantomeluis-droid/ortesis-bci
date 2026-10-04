"""Copiloto clinico: preguntas en lenguaje natural sobre una sesion, respondidas solo con sus datos.

Las herramientas leen el CSV de la sesion y, si existe, su registro `_estado.jsonl` (eventos del
flujo Estado, checkpoints y marcadores). Con llave en ANTHROPIC_API_KEY (.env) responde Claude
usando esas herramientas; sin llave, sin conexion o si la API falla, responden plantillas
deterministas con las mismas herramientas. Nunca corre dentro del lazo de control.

Uso
  python copiloto.py --ultima "cuanto tardo en recuperarse tras la perturbacion?"
  python copiloto.py --sesion resultados/sesion_real_....csv            modo interactivo
  python copiloto.py --sesion <csv> --informe                           informe entre sesiones (4 Markdown)
  python copiloto.py --sesion <csv> --decidir aprobar | rechazar        la propuesta para la proxima sesion
  python copiloto.py --sesion <csv> --sin-ia "hubo senales de fatiga?"  solo plantillas

A la API solo van las cifras que devuelven las herramientas (ia.sanear): nunca EEG crudo ni rutas.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

import config
import ia

SIN_DATO = 'no hay dato'
TIPOS_EVENTO = ('pausa', 'congelamiento', 'cambio', 'perturbacion', 'checkpoint', 'semaforo')
METRICAS = ('error', 'recuperacion', 'ba_viva', 'fiabilidad', 'latencia_ack', 'excluidos', 'alfa', 'sin_errp_por_paso')
BLOQUES = ('todo', 'estatico', 'adaptativo', 'real', 'sham', 'tras_perturbacion')
_ESTADOS = {'estatico': ('LAZO_ESTATICO',), 'adaptativo': ('LAZO_ADAPTATIVO', 'APRENDIZAJE_CONGELADO')}


def _numero(v):
    for tipo in (int, float):
        try:
            return tipo(v)
        except ValueError:
            pass
    return v


def sin_dato(motivo):
    return {'dato': None, 'motivo': f'{SIN_DATO}: {motivo}'}


class Sesion:
    """Una sesion grabada. `paso` es el numero de fila del CSV (desde 1), el mismo del tablero."""

    def __init__(self, ruta):
        self.ruta = Path(ruta)
        with open(self.ruta, newline='') as f:
            self.filas = [dict({c: _numero(v) for c, v in fila.items()}, paso=k + 1)
                          for k, fila in enumerate(csv.DictReader(f))]
        self.real = '_real_' in self.ruta.name
        self.eventos, self.marcadores, self.por_paso = [], [], {}
        registro = self.ruta.with_name(self.ruta.stem + config.SUFIJO_ESTADO)
        if registro.exists():
            visto = 0                                        # ultimo paso publicado antes de cada linea
            for linea in registro.read_text(encoding='utf-8').splitlines():
                try:
                    d = json.loads(linea)
                except ValueError:
                    continue
                if 'marcador' in d:
                    self.marcadores.append((visto, d['marcador']))
                elif 'evento' in d:
                    e = d['evento']
                    if e.get('tipo') == 'paso':
                        visto = e['paso']
                        self.por_paso[visto] = e
                    self.eventos.append((visto, e))
        pasos = sorted(self.por_paso)
        self.perturbaciones = [p for a, p in zip([None] + pasos, pasos) if self.por_paso[p].get('perturbado')
                               and not (a and self.por_paso[a].get('perturbado'))]

    # ------------------------------------------------------------ seleccion
    def filas_de(self, bloque=None):
        """Las filas validas (no excluidas) de un bloque."""
        f = [x for x in self.filas if not x['excluido']]
        if bloque in (None, '', 'todo'):
            return f
        if bloque in _ESTADOS:
            return [x for x in f if x['estado'] in _ESTADOS[bloque]]
        if bloque in config.BLOQUES_SHAM:
            return [x for x in f if x.get('bloque') == bloque]
        if bloque == 'tras_perturbacion':
            return self._tras(self._perturbacion_principal())
        raise ValueError(f'bloque desconocido: {bloque!r}; usa uno de {BLOQUES}')

    def _perturbacion_principal(self):
        """La perturbacion del bloque real si la sesion fue --sham; si no, la unica."""
        for p in self.perturbaciones:
            if self.filas[p - 1].get('bloque', '') in ('', 'real'):
                return p
        return None

    def _tras(self, p):
        """Filas validas desde la perturbacion p hasta el fin de su bloque (como mucho ~2 min)."""
        if p is None:
            return []
        b = self.filas[p - 1].get('bloque', '')
        return [x for x in self.filas[p - 1:] if not x['excluido'] and x.get('bloque', '') == b][
            :int(config.RECUPERACION_MAX_S / config.CICLO_S)]

    @staticmethod
    def _rango(filas):
        return [filas[0]['paso'], filas[-1]['paso']] if filas else None

    # ------------------------------------------------------------ herramientas
    def resumen_sesion(self):
        validas = self.filas_de()
        if not validas:
            return sin_dato('la sesion no tiene pasos validos')
        bloques = {}
        for b in ('estatico', 'adaptativo') + config.BLOQUES_SHAM:
            if self.filas_de(b):
                bloques[b] = self.metrica('error', b)
        cong = self.eventos_de(tipo='congelamiento')
        r = {'sesion': self.ruta.name, 'backend': 'real' if self.real else 'sim', 'filas': len(self.filas),
             'pasos_validos': len(validas), 'duracion_s': round(self.filas[-1]['t_lsl'] - self.filas[0]['t_lsl'], 1),
             'bloques': bloques, 'recuperacion': self.metrica('recuperacion'),
             'excluidos': self.metrica('excluidos'), 'ba_viva': self.metrica('ba_viva'),
             'sin_movimiento': sum(x.get('alineacion') == config.SIN_MOVIMIENTO for x in validas),
             'congelamientos': cong, 'cambios_detectados': self.eventos_de(tipo='cambio'),
             'checkpoints': [e['detalle'] for e in self.eventos_de(tipo='checkpoint')],
             'cambios_de_modelo_del_detector': sum(m.startswith('detector:v') for _, m in self.marcadores)}
        sham = [e for _, e in self.eventos if e.get('tipo') == 'sham']
        if sham:
            r['control_causal_sham'] = {k: v for k, v in sham[-1].items() if k != 'tipo'}
        return r

    def eventos_de(self, desde=None, hasta=None, tipo=None):
        """Pausas, congelamientos, cambios detectados, perturbaciones, checkpoints y cambios de semaforo."""
        if tipo not in (None, '') + TIPOS_EVENTO:
            raise ValueError(f'tipo desconocido: {tipo!r}; usa uno de {TIPOS_EVENTO}')
        ev, previa, inicio = [], None, None
        motivos = [(p, e) for p, e in self.eventos if e.get('tipo') == 'salud' and e.get('estado') == 'PAUSA_SEGURA']
        for f in self.filas:
            p = f['paso']
            if str(f['excluido']).startswith('pausa:'):
                d = next((e['detalle'].get('ortesis' if e.get('motivo') == 'ortesis' else 'eeg', '')
                          for q, e in motivos if q == p - 1), '')
                ev.append({'paso': p, 'tipo': 'pausa', 'detalle': f['excluido'] + (f' ({d})' if d else '')})
            if f['cambio']:
                ev.append({'paso': p, 'tipo': 'cambio', 'detalle': f"el agente detecto un cambio por {f['cambio']}"})
            congelado = f['estado'] == 'APRENDIZAJE_CONGELADO'
            if congelado and inicio is None:
                inicio = f
            if inicio is not None and (not congelado and not f['excluido'] or f is self.filas[-1]):
                fin = p if congelado else p - 1
                ev.append({'paso': inicio['paso'], 'tipo': 'congelamiento', 'hasta_paso': fin,
                           'detalle': f"aprendizaje congelado del paso {inicio['paso']} al {fin}: sens viva "
                                      f"{inicio['sens_viva']}, espec viva {inicio['espec_viva']} al congelarse"})
                inicio = None
            if previa is not None and f['salud'] != previa['salud'] and f['salud'] and previa['salud']:
                cambios = [f'{s} {a}->{b}' for s, a, b in zip(config.SUBSISTEMAS, previa['salud'], f['salud']) if a != b]
                ev.append({'paso': p, 'tipo': 'semaforo', 'detalle': ', '.join(cambios)})
            previa = f
        ev += [{'paso': p, 'tipo': 'perturbacion',
                'detalle': f'perturbacion de {config.PERTURBACION_LOGITS} logits'
                           + (f" (bloque {self.filas[p - 1]['bloque']})" if self.filas[p - 1].get('bloque') else '')}
               for p in self.perturbaciones]
        ev += [{'paso': p, 'tipo': 'checkpoint', 'detalle': e['texto']} for p, e in self.eventos
               if e.get('tipo') == 'checkpoint']
        lo, hi = desde or 0, hasta or len(self.filas)
        return sorted((e for e in ev if lo <= e['paso'] <= hi and tipo in (None, '', e['tipo'])),
                      key=lambda e: e['paso'])

    def metrica(self, nombre, bloque=None):
        if nombre not in METRICAS:
            raise ValueError(f'metrica desconocida: {nombre!r}; usa una de {METRICAS}')
        if nombre == 'recuperacion':
            return self._recuperacion(bloque)
        if nombre == 'excluidos':
            return self._excluidos(bloque)
        filas = self.filas_de(bloque)
        if not filas:
            return sin_dato(f"la sesion no tiene pasos validos en el bloque '{bloque or 'todo'}'")
        base = {'bloque': bloque or 'todo', 'pasos': self._rango(filas), 'n': len(filas)}
        return getattr(self, '_' + nombre)(filas, base)

    def _error(self, filas, base):
        import hardware as hw
        e = [x['error_verdadero'] for x in filas]
        s = [x['error_sombra'] for x in filas]
        lo, hi = hw.intervalo_diferencia(s, e)
        return dict(base, error_agente=float(np.mean(e)), ic90_agente=hw.intervalo_error(e),
                    error_sombra=float(np.mean(s)), ic90_sombra=hw.intervalo_error(s),
                    sombra_menos_agente=float(np.mean(s) - np.mean(e)), ic90_diferencia=(lo, hi),
                    agente_mejor_con_certeza=bool(lo > 0),
                    nota='intervalos del 90 % remuestreando ensayos de 5 pasos; la certeza pide que el de la '
                         'diferencia excluya el 0')

    def _recuperacion(self, bloque):
        if not self.perturbaciones:
            return sin_dato('no hay registro de una perturbacion en esta sesion'
                            + ('' if self.por_paso else ' (falta su archivo _estado.jsonl)'))
        out = []
        for p in self.perturbaciones:
            b = self.filas[p - 1].get('bloque', '')
            if bloque in config.BLOQUES_SHAM and b != bloque:
                continue
            previas = [x for x in self.filas[:p - 1] if not x['excluido'] and x.get('bloque', '') == b]
            beta_pre = previas[-1]['beta'] if previas else 0.0
            tras = [x for x in self.filas[p - 1:] if not x['excluido'] and x.get('bloque', '') == b]
            rec = next((k for k, x in enumerate(tras) if x['beta'] >= beta_pre + 0.7 * config.PERTURBACION_LOGITS), None)
            r = {'paso_perturbacion': p, 'bloque': b or 'adaptativo', 'beta_antes': beta_pre,
                 'criterio': 'beta sube 70 % de la perturbacion', 'recuperado': rec is not None,
                 'pasos_validos_tras_perturbar': len(tras)}
            if rec is not None:
                seg = tras[rec]['t_lsl'] - tras[0]['t_lsl'] if self.real else (rec + 1) * config.CICLO_S
                r.update(pasos=rec + 1, paso_recuperacion=tras[rec]['paso'], segundos=round(float(seg), 1),
                         segundos_son='medidos' if self.real else 'nominales (simulador)')
            out.append(r)
        return out or sin_dato(f"no hubo perturbacion en el bloque '{bloque}'")

    def _excluidos(self, bloque):
        filas = [x for x in self.filas if x['excluido']]
        if bloque in config.BLOQUES_SHAM:
            filas = [x for x in filas if x.get('bloque') == bloque]
        elif bloque in _ESTADOS:
            filas = [x for x in filas if x['estado'] in _ESTADOS[bloque]]
        motivos = {}
        for x in filas:
            motivos.setdefault(x['excluido'], []).append(x['paso'])
        return {'bloque': bloque or 'todo', 'excluidos': len(filas), 'de_filas': len(self.filas),
                'por_motivo': {m: {'n': len(p), 'pasos': p[:config.COPILOTO_MAX_FILAS]} for m, p in motivos.items()},
                'significado': 'pausa:* = fila de una pausa segura; sin_ack = la ortesis no confirmo el movimiento; '
                               'epoca_invalida = la epoca cruza un corte de EEG; ajeno = movimiento automatico de la Tarea 2'}

    def _ba_viva(self, filas, base):
        u = filas[-1]
        return dict(base, sens_viva=u['sens_viva'], espec_viva=u['espec_viva'],
                    ba_viva=round((u['sens_viva'] + u['espec_viva']) / 2, 3), en_el_paso=u['paso'],
                    nota='sensibilidad y especificidad del detector de ErrP estimadas en vivo (ConfianzaDetector)')

    def _fiabilidad(self, filas, base):
        f = [x['fiabilidad'] for x in filas if x['fiabilidad'] != '']
        if not f:
            return sin_dato('ningun paso del bloque tiene fiabilidad registrada')
        cong = [x['paso'] for x in filas if x['estado'] == 'APRENDIZAJE_CONGELADO']
        return dict(base, media=float(np.mean(f)), minima=float(np.min(f)), pasos_congelados=len(cong),
                    fraccion_congelado=len(cong) / len(filas),
                    nota='fiabilidad = Youden vivo / Youden de calibracion; bajo 0.5 el aprendizaje se congela (vale 0) '
                         'y se reanuda sobre 0.7')

    def _latencia_ack(self, filas, base):
        lat = [x['latencia_ack_ms'] for x in filas if x['latencia_ack_ms'] != '']
        if not lat:
            return sin_dato('ningun paso del bloque tiene latencia del ACK')
        return dict(base, media_ms=float(np.mean(lat)), desviacion_ms=float(np.std(lat)),
                    p95_ms=float(np.percentile(lat, 95)), maxima_ms=float(np.max(lat)),
                    sin_ack=sum(x['excluido'] == 'sin_ack' for x in self.filas))

    def _alfa(self, filas, base):
        a = [(x['paso'], self.por_paso[x['paso']].get('alfa')) for x in filas if x['paso'] in self.por_paso]
        a = [(p, v) for p, v in a if v is not None]
        if not a:
            return sin_dato('esta sesion no registro el alfa occipital (el simulador no lo tiene, y el casco necesita '
                            f"{config.SALUD['piloto_base_s']:.0f} s de linea base)")
        v, k = [x for _, x in a], max(1, len(a) // 3)
        return dict(base, pasos=[a[0][0], a[-1][0]], n=len(a), media=float(np.mean(v)), maxima=float(np.max(v)),
                    paso_del_maximo=a[int(np.argmax(v))][0], primer_tercio=float(np.mean(v[:k])),
                    ultimo_tercio=float(np.mean(v[-k:])),
                    avisos_del_semaforo_piloto=[e for e in self.eventos_de(tipo='semaforo') if 'piloto' in e['detalle']],
                    nota=f"alfa occipital dividida entre su linea base; el semaforo PILOTO avisa desde "
                         f"x{config.SALUD['piloto_amarillo']} (umbral sin validar en personas)")

    def _sin_errp_por_paso(self, filas, base):
        """De los pasos erroneos con epoca, cuantos pasaron sin ErrP detectado, segun el tamano del paso."""
        r, aprox = {}, False
        for nombre, cabe in (('chico', lambda d: d < config.PASO_CHICO), ('grande', lambda d: d >= config.PASO_CHICO)):
            det = []
            for x in filas:
                if x['error_verdadero'] != 1 or x['P_hat'] == '' or not cabe(abs(x['delta'])):
                    continue
                d = self.por_paso.get(x['paso'], {}).get('detectado')
                aprox |= d is None
                det.append(x['P_hat'] > 0.5 if d is None else d)
            r[nombre] = {'errores': len(det), 'sin_errp': float(1 - np.mean(det)) if det else None}
        return dict(base, **r, corte=config.PASO_CHICO,
                    nota='deteccion aproximada con P_hat > 0.5' if aprox else 'deteccion registrada por paso')

    def pasos(self, desde, hasta=None, columnas=None):
        hasta = desde if hasta is None else hasta
        columnas = columnas or ['estado', 'meta', 'direccion', 'delta', 'P_hat', 'fiabilidad', 'beta',
                                'sens_viva', 'espec_viva', 'error_verdadero', 'error_sombra', 'excluido']
        malas = [c for c in columnas if c not in config.COLUMNAS_CSV]
        if malas:
            raise ValueError(f'columnas desconocidas: {malas}; las del CSV son {config.COLUMNAS_CSV}')
        filas = [x for x in self.filas if desde <= x['paso'] <= hasta]
        if not filas:
            return sin_dato(f'la sesion tiene {len(self.filas)} pasos; no existe el rango {desde}-{hasta}')
        return {'filas': [{'paso': x['paso'], **{c: x.get(c, '') for c in columnas}} for x in filas[:config.COPILOTO_MAX_FILAS]],
                'truncado': len(filas) > config.COPILOTO_MAX_FILAS, 'limite': config.COPILOTO_MAX_FILAS}

    def anteriores(self, n=1):
        """Las n sesiones previas del mismo tipo (real o sim) en la misma carpeta, de la mas vieja a la mas nueva."""
        prefijo = self.ruta.name[:self.ruta.name.index('_', len('sesion_')) + 1]
        todas = sorted(p for p in self.ruta.parent.glob(prefijo + '*.csv') if p.name < self.ruta.name)
        return todas[-n:] if n else []

    def comparar_sesiones(self, rutas=None):
        """Esta sesion contra otras de la misma carpeta (por nombre de archivo; por defecto, la anterior)."""
        otras = [self.ruta.parent / Path(r).name for r in rutas] if rutas else self.anteriores(1)
        if not otras:
            return sin_dato('no hay una sesion anterior del mismo tipo en la carpeta de resultados')
        faltan = [p.name for p in otras if not p.exists()]
        if faltan:
            return sin_dato(f'no existen estas sesiones: {faltan}')
        return {'sesiones': [clave(Sesion(p)) for p in otras] + [clave(self)],
                'nota': 'la ultima de la lista es la sesion actual; se supone el mismo piloto: los archivos no lo dicen'}


def clave(s):
    """Las cifras de una sesion que se comparan entre sesiones."""
    post = s.filas_de('tras_perturbacion')
    rec = s.metrica('recuperacion')
    rec = next((r for r in rec if r['bloque'] != 'sham'), None) if isinstance(rec, list) else None
    todo = s.metrica('error') if s.filas_de() else {}
    r = {'sesion': s.ruta.name, 'pasos_validos': len(s.filas_de()),
         'error_agente': todo.get('error_agente'), 'error_sombra': todo.get('error_sombra'),
         'recuperado': rec and rec['recuperado'], 'pasos_para_recuperarse': rec and rec.get('pasos'),
         'excluidos': sum(1 for x in s.filas if x['excluido']),
         'pasos_congelados': sum(x['estado'] == 'APRENDIZAJE_CONGELADO' for x in s.filas)}
    if post:
        e = s._error(post, {})
        r.update(error_agente_tras_perturbar=e['error_agente'], error_sombra_tras_perturbar=e['error_sombra'])
    if s.filas_de():
        r['ba_viva'] = s.metrica('ba_viva')['ba_viva']
    return r


def resumen_para_propuesta(s, bloque=None):
    """El resumen agregado que recibe el co-investigador (o las reglas): un bloque o la sesion entera."""
    filas = s.filas_de(bloque)
    todas = [x for x in s.filas if bloque in (None, '', 'todo') or x in filas or
             (x['excluido'] and (x.get('bloque') == bloque or x['estado'] in _ESTADOS.get(bloque, ())))]
    r = {'bloque': bloque or 'todo', 'pasos': len(filas), 'paso_visible': config.PASO_VISIBLE,
         'excluidos': s._excluidos(bloque)['por_motivo'] and {m: d['n'] for m, d in s._excluidos(bloque)['por_motivo'].items()},
         'fraccion_excluidos': (len(todas) - len(filas)) / len(todas) if todas else None}
    if not filas:
        return r
    e, f, alfa = s._error(filas, {}), s._fiabilidad(filas, {}), s._alfa(filas, {})
    r.update(exactitud=1 - e['error_agente'], error_agente=round(e['error_agente'], 3),
             error_sombra=round(e['error_sombra'], 3), ic90_sombra_menos_agente=e['ic90_diferencia'],
             ba_viva=s._ba_viva(filas, {})['ba_viva'], fiabilidad_media=f.get('media'),
             fraccion_congelado=f.get('fraccion_congelado'), alfa_rel=alfa.get('ultimo_tercio'),
             sin_movimiento=sum(x.get('alineacion') == config.SIN_MOVIMIENTO for x in filas),
             sin_errp_por_paso={k: v for k, v in s._sin_errp_por_paso(filas, {}).items() if k in ('chico', 'grande')})
    return r


# ====================================================================== preguntas
def _esquema(props, requeridos=()):
    return {'type': 'object', 'properties': props, 'required': list(requeridos), 'additionalProperties': False}


_ENTERO = {'type': 'integer', 'minimum': 1}
HERRAMIENTAS = [
    {'name': 'resumen_sesion', 'input_schema': _esquema({}),
     'description': 'Resumen de la sesion: pasos, error del agente y de la sombra por bloque, recuperacion tras la '
                    'perturbacion, excluidos, BA viva del detector, congelamientos, cambios y checkpoints. Empieza por aqui.'},
    {'name': 'eventos', 'input_schema': _esquema({'desde': _ENTERO, 'hasta': _ENTERO,
                                                  'tipo': {'type': 'string', 'enum': list(TIPOS_EVENTO)}}),
     'description': 'Eventos de la sesion entre dos pasos (por defecto, todos): pausas seguras, congelamientos del '
                    'aprendizaje, cambios detectados por el agente, perturbaciones, checkpoints y cambios de semaforo.'},
    {'name': 'metrica', 'input_schema': _esquema({'nombre': {'type': 'string', 'enum': list(METRICAS)},
                                                  'bloque': {'type': 'string', 'enum': list(BLOQUES)}}, ['nombre']),
     'description': 'Una metrica de un bloque (por defecto, toda la sesion): error (agente y sombra con intervalos '
                    'y su diferencia), recuperacion, ba_viva, fiabilidad, latencia_ack, excluidos (por motivo), alfa '
                    '(alfa occipital: fatiga o somnolencia) y sin_errp_por_paso.'},
    {'name': 'pasos', 'input_schema': _esquema({'desde': _ENTERO, 'hasta': _ENTERO,
                                                'columnas': {'type': 'array', 'items': {'type': 'string',
                                                                                        'enum': config.COLUMNAS_CSV}}}, ['desde']),
     'description': f'Filas puntuales del CSV (como mucho {config.COPILOTO_MAX_FILAS} por llamada), para ver que paso '
                    'alrededor de un paso concreto.'},
    {'name': 'comparar_sesiones', 'input_schema': _esquema({'rutas': {'type': 'array', 'items': {'type': 'string'}}}),
     'description': 'Las cifras clave de esta sesion junto a las de otras (nombres de archivo de la carpeta de '
                    'resultados). Sin rutas: la sesion anterior del mismo tipo.'},
]
SISTEMA = (
    'Eres el copiloto clinico de una sesion de interfaz cerebro-computadora: una ortesis de mano controlada por '
    'imaginacion motora; un agente corrige al decoder con el potencial de error (ErrP); la "sombra" es el decoder '
    'sin corregir. Responde SOLO con lo que devuelvan las herramientas. Cita siempre los pasos y los valores que '
    f'usaste. Si una herramienta dice "{SIN_DATO}", o el dato no esta, dilo con esas palabras: "{SIN_DATO}". '
    'Nunca inventes ni estimes una cifra. No des diagnosticos medicos: una senal como el alfa occipital solo es un '
    'indicio. Responde breve, en el idioma de la pregunta.')


def ejecutar(s, nombre, entrada):
    if nombre == 'resumen_sesion':
        return s.resumen_sesion()
    if nombre == 'eventos':
        return s.eventos_de(entrada.get('desde'), entrada.get('hasta'), entrada.get('tipo'))
    if nombre == 'metrica':
        return s.metrica(entrada['nombre'], entrada.get('bloque'))
    if nombre == 'pasos':
        return s.pasos(entrada['desde'], entrada.get('hasta'), entrada.get('columnas'))
    if nombre == 'comparar_sesiones':
        return s.comparar_sesiones(entrada.get('rutas'))
    raise ValueError(f'herramienta desconocida: {nombre}')


def _plano(txt):
    return unicodedata.normalize('NFD', txt.lower()).encode('ascii', 'ignore').decode()


def _vacio(r):
    return isinstance(r, dict) and r.get('dato', 0) is None


def responder_reglas(s, pregunta):
    """Plantillas deterministas sobre las mismas herramientas, para cuando no hay API."""
    q = _plano(pregunta)
    n = next((int(v) for v in re.findall(r'\d+', q)), None)
    ic = lambda v: f'[{v[0]:.2f}, {v[1]:.2f}]'
    if 'congel' in q:
        cong = s.eventos_de(tipo='congelamiento')
        if n is None:
            return ('Congelamientos del aprendizaje: ' + '; '.join(e['detalle'] for e in cong)) if cong else \
                f'{SIN_DATO}: el aprendizaje no se congelo en esta sesion.'
        if not 1 <= n <= len(s.filas):
            return f'{SIN_DATO}: la sesion tiene {len(s.filas)} pasos; el paso {n} no existe.'
        f = s.filas[n - 1]
        e = next((e for e in cong if e['paso'] <= n <= e['hasta_paso']), None)
        if e is None:
            return (f"En el paso {n} el aprendizaje no estaba congelado (estado {f['estado']}, fiabilidad "
                    f"{'s/d' if f['fiabilidad'] == '' else f['fiabilidad']}). "
                    + ('Congelamientos de la sesion: ' + '; '.join(x['detalle'] for x in cong) if cong
                       else 'En esta sesion no se congelo nunca.'))
        prev = [x for x in s.filas[max(0, e['paso'] - 16):e['paso'] - 1] if not x['excluido'] and x['P_hat'] != '']
        det = lambda x: s.por_paso.get(x['paso'], {}).get('detectado', x['P_hat'] > 0.5)
        err, ok = [x for x in prev if x['error_verdadero']], [x for x in prev if not x['error_verdadero']]
        return (f"El paso {n} cae en un congelamiento: {e['detalle']}. El aprendizaje se congela cuando la fiabilidad "
                f"viva del detector (su Youden vivo entre el de calibracion) baja de 0.5, y se reanuda sobre 0.7. En los "
                f"{len(prev)} pasos con epoca anteriores al paso {e['paso']}, el detector marco {sum(map(det, err))} de "
                f"{len(err)} errores y dio {sum(map(det, ok))} falsas alarmas en {len(ok)} aciertos.")
    if 'recuper' in q or 'perturb' in q:
        r = s.metrica('recuperacion')
        if _vacio(r):
            return r['motivo']
        return ' '.join(
            f"Perturbacion en el paso {x['paso_perturbacion']} (bloque {x['bloque']}, beta antes {x['beta_antes']:+.2f}): "
            + (f"se recupero en {x['pasos']} pasos = {x['segundos']} s ({x['segundos_son']}), en el paso {x['paso_recuperacion']}."
               if x['recuperado'] else f"NO se recupero en los {x['pasos_validos_tras_perturbar']} pasos validos que quedaban.")
            for x in r)
    if 'sombra' in q or 'gano' in q or 'certeza' in q:
        partes = []
        for b in ('tras_perturbacion', 'adaptativo', 'todo'):
            r = s.metrica('error', b)
            if not _vacio(r):
                partes.append(f"{b} (pasos {r['pasos'][0]}-{r['pasos'][1]}): agente {r['error_agente']:.2f} "
                              f"{ic(r['ic90_agente'])}, sombra {r['error_sombra']:.2f} {ic(r['ic90_sombra'])}, sombra - agente "
                              f"{r['sombra_menos_agente']:+.2f} {ic(r['ic90_diferencia'])}: "
                              + ('el agente gano con certeza (el intervalo excluye el 0)' if r['agente_mejor_con_certeza']
                                 else 'la diferencia no excluye el 0: no hay certeza'))
        return 'Error del agente contra la sombra, con intervalos del 90 %. ' + '. '.join(partes) + '.' if partes \
            else f'{SIN_DATO}: la sesion no tiene pasos validos.'
    if 'fatiga' in q or 'alfa' in q or 'somnol' in q or 'cansa' in q:
        r = s.metrica('alfa')
        if _vacio(r):
            return r['motivo'] + ' Sin esa senal no se puede decir nada de fatiga.'
        av = r['avisos_del_semaforo_piloto']
        return (f"Alfa occipital contra su linea base (pasos {r['pasos'][0]}-{r['pasos'][1]}): media x{r['media']:.2f}, "
                f"primer tercio x{r['primer_tercio']:.2f}, ultimo tercio x{r['ultimo_tercio']:.2f}, maximo x{r['maxima']:.2f} "
                f"en el paso {r['paso_del_maximo']}. "
                + (f"El semaforo PILOTO aviso en los pasos {[e['paso'] for e in av]}. " if av
                   else 'El semaforo PILOTO no aviso. ') + 'Es un indicio, no un diagnostico: ' + r['nota'] + '.')
    if 'exclu' in q:
        r = s.metrica('excluidos')
        return (f"Se excluyeron {r['excluidos']} de {r['de_filas']} filas"
                + (': ' + '; '.join(f"{m}: {d['n']} (pasos {d['pasos']})" for m, d in r['por_motivo'].items()) if r['excluidos'] else '')
                + f". {r['significado']}.")
    if 'compar' in q or 'anterior' in q:
        r = s.comparar_sesiones()
        if _vacio(r):
            return r['motivo']
        campos = ('pasos_validos', 'error_agente', 'error_sombra', 'error_agente_tras_perturbar', 'pasos_para_recuperarse',
                  'ba_viva', 'excluidos', 'pasos_congelados')
        fmt = lambda v: 's/d' if v is None else f'{v:.2f}' if isinstance(v, float) else str(v)
        return ' | '.join(f"{x['sesion']}: " + ', '.join(f'{c} {fmt(x.get(c))}' for c in campos) for x in r['sesiones']) \
            + '. ' + r['nota'] + '.'
    r = s.resumen_sesion()
    if _vacio(r):
        return r['motivo']
    return ('Sin IA solo respondo sobre congelamientos, recuperacion, agente contra sombra, fatiga, excluidos y la '
            f"comparacion con la sesion anterior. Resumen: {r['pasos_validos']} pasos validos de {r['filas']}; "
            + '; '.join(f"{b}: agente {m['error_agente']:.2f}, sombra {m['error_sombra']:.2f}" for b, m in r['bloques'].items())
            + f"; BA viva del detector {r['ba_viva']['ba_viva']}; checkpoints: {r['checkpoints']}.")


def responder(s, pregunta, cli=None):
    """La respuesta y su origen ('api' o 'reglas'). Con cli = None, o si la API falla, plantillas."""
    if cli is not None:
        try:
            txt, usadas = ia.conversar(cli, SISTEMA, pregunta, HERRAMIENTAS, lambda n, e: ejecutar(s, n, e))
            return txt, 'api', usadas
        except Exception as e:                    # sin conexion, tiempo agotado, la API declino...
            aviso = f' (la IA no respondio: {type(e).__name__}; respuesta por plantilla)'
            return responder_reglas(s, pregunta) + aviso, 'reglas', []
    return responder_reglas(s, pregunta), 'reglas', []


# ====================================================================== propuestas: registro y decision
def ruta_propuestas(ruta_csv):
    ruta_csv = Path(ruta_csv)
    return ruta_csv.with_name(ruta_csv.stem + config.SUFIJO_PROPUESTAS)


def registrar(ruta_csv, **linea):
    """Agrega una linea al registro de propuestas de la sesion (propuesta, decision o efecto)."""
    with open(ruta_propuestas(ruta_csv), 'a', encoding='utf-8') as f:
        f.write(json.dumps(ia.sanear(dict(linea, t=time.strftime('%Y-%m-%dT%H:%M:%S'))), ensure_ascii=False) + '\n')


def leer_propuestas(ruta_csv):
    ruta = ruta_propuestas(ruta_csv)
    return [json.loads(l) for l in ruta.read_text(encoding='utf-8').splitlines() if l.strip()] if ruta.exists() else []


def pendiente(ruta_csv, momento=None):
    """La ultima propuesta valida que aun no tiene decision (de ese momento, si se da), o None."""
    lineas = leer_propuestas(ruta_csv)
    decididas = {l['id'] for l in lineas if l.get('registro') == 'decision'}
    for l in reversed(lineas):
        if l.get('registro') == 'propuesta' and l['id'] not in decididas and l['valida'] \
                and momento in (None, l['momento']):
            return l
    return None


def proponer_y_registrar(ruta_csv, resumen, momento, cli=None):
    r = ia.proponer(resumen, cli)
    ident = f"{momento}#{sum(l.get('registro') == 'propuesta' for l in leer_propuestas(ruta_csv)) + 1}"
    registrar(ruta_csv, registro='propuesta', id=ident, momento=momento, resumen=resumen, **r)
    return dict(r, id=ident, momento=momento)


def decidir(ruta_csv, ident, aprobar, por='operador', efecto=None):
    """La decision la toma una persona. efecto: lo que cambio al aplicarla (o por que no se aplico nada)."""
    registrar(ruta_csv, registro='decision', id=ident, decision='aprobada' if aprobar else 'rechazada', por=por)
    if efecto is not None:
        registrar(ruta_csv, registro='efecto', id=ident, efecto=efecto)


# ====================================================================== informe entre sesiones
# Los textos del informe llevan acentos a proposito: van a archivos Markdown, no a la consola.
_T = {
    'es': {'titulo_t': 'Informe de sesión para el terapeuta', 'titulo_p': 'Resumen de su sesión',
           'cifras': 'Cifras de la sesión', 'comp': 'Comparación con las sesiones anteriores', 'interp': 'Interpretación',
           'prop': 'Propuesta para la próxima sesión', 'fig': 'Figuras',
           'pend': 'Requiere la aprobación de una persona; no se aplica sola',
           'origen': {'api': 'generada por IA a partir de las cifras de arriba', 'reglas': 'por reglas deterministas'},
           'aviso': 'Mismo piloto supuesto: los archivos de sesión son anónimos. Cifras exploratorias de una sola sesión.',
           'sin_ant': 'No hay sesiones anteriores con las que comparar.', 'si': 'sí', 'no': 'no', 'sd': 's/d'},
    'en': {'titulo_t': 'Session report for the therapist', 'titulo_p': 'Your session summary',
           'cifras': 'Session figures', 'comp': 'Comparison with previous sessions', 'interp': 'Interpretation',
           'prop': 'Proposal for the next session', 'fig': 'Figures',
           'pend': 'Requires approval by a person; it is never applied automatically',
           'origen': {'api': 'AI-generated from the figures above', 'reglas': 'from deterministic rules'},
           'aviso': 'Same pilot assumed: session files are anonymous. Exploratory figures from a single session.',
           'sin_ant': 'There are no previous sessions to compare with.', 'si': 'yes', 'no': 'no', 'sd': 'n/a'},
}
_FILAS = [('pasos_validos', 'Pasos válidos', 'Valid steps'), ('error_agente', 'Error del agente', 'Agent error'),
          ('error_sombra', 'Error de la sombra (decoder sin corregir)', 'Shadow error (uncorrected decoder)'),
          ('error_agente_tras_perturbar', 'Error del agente tras la perturbación', 'Agent error after the perturbation'),
          ('error_sombra_tras_perturbar', 'Error de la sombra tras la perturbación', 'Shadow error after the perturbation'),
          ('recuperado', 'Se recuperó de la perturbación', 'Recovered from the perturbation'),
          ('pasos_para_recuperarse', 'Pasos para recuperarse', 'Steps to recover'),
          ('ba_viva', 'BA viva del detector de ErrP', 'Live BA of the ErrP detector'),
          ('excluidos', 'Filas excluidas', 'Excluded rows'), ('pasos_congelados', 'Pasos con el aprendizaje congelado',
                                                             'Steps with learning frozen')]
ESQUEMA_INTERPRETACION = {'type': 'object', 'additionalProperties': False,
                          'properties': {k: {'type': 'string'} for k in ('terapeuta_es', 'terapeuta_en', 'paciente_es', 'paciente_en')},
                          'required': ['terapeuta_es', 'terapeuta_en', 'paciente_es', 'paciente_en']}
SISTEMA_INFORME = (
    'Recibes las cifras agregadas y anonimas de una sesion de una ortesis de mano controlada por imaginacion motora, '
    'y las de sesiones anteriores. Escribe cuatro parrafos de interpretacion (3 a 5 frases cada uno) basados SOLO en '
    'esas cifras: para el terapeuta en espanol y en ingles (tecnico, con las cifras), y para el paciente en espanol y '
    'en ingles (lenguaje sencillo y amable, sin jerga ni promesas, sin diagnosticos). Si falta un dato, dilo; no inventes.')


def _fmt(v, t):
    return t['sd'] if v is None else (t['si'] if v else t['no']) if isinstance(v, bool) else f'{v:.2f}' if isinstance(v, float) else str(v)


def _interpretacion_plantilla(c, ant):
    ea, es = c.get('error_agente_tras_perturbar'), c.get('error_sombra_tras_perturbar')
    acierto = round(10 * (1 - c['error_agente'])) if c.get('error_agente') is not None else None
    previo = ant[-1].get('error_agente') if ant else None
    tend_es = '' if previo is None else (' Mejor que la sesión anterior.' if c['error_agente'] < previo - 0.02 else
                                         ' Parecido a la sesión anterior.' if abs(c['error_agente'] - previo) <= 0.02 else
                                         ' Algo por debajo de la sesión anterior; es normal que varíe de un día a otro.')
    tend_en = '' if previo is None else (' Better than the previous session.' if c['error_agente'] < previo - 0.02 else
                                         ' Similar to the previous session.' if abs(c['error_agente'] - previo) <= 0.02 else
                                         ' Slightly below the previous session; day-to-day variation is normal.')
    rec_es = ('no hubo perturbación registrada' if c['recuperado'] is None else
              f"el agente se recuperó en {c['pasos_para_recuperarse']} pasos" if c['recuperado'] else 'el agente no se recuperó dentro del bloque')
    rec_en = ('no perturbation was recorded' if c['recuperado'] is None else
              f"the agent recovered in {c['pasos_para_recuperarse']} steps" if c['recuperado'] else 'the agent did not recover within the block')
    post_es = '' if ea is None else f' Tras la perturbación, error del agente {ea:.2f} contra {es:.2f} de la sombra.'
    post_en = '' if ea is None else f' After the perturbation, agent error {ea:.2f} versus {es:.2f} for the shadow.'
    return {
        'terapeuta_es': f"Error del agente {_fmt(c.get('error_agente'), _T['es'])} contra {_fmt(c.get('error_sombra'), _T['es'])} de la sombra; {rec_es}.{post_es} BA viva del detector {_fmt(c.get('ba_viva'), _T['es'])}.{tend_es}",
        'terapeuta_en': f"Agent error {_fmt(c.get('error_agente'), _T['en'])} versus {_fmt(c.get('error_sombra'), _T['en'])} for the shadow; {rec_en}.{post_en} Live detector BA {_fmt(c.get('ba_viva'), _T['en'])}.{tend_en}",
        'paciente_es': ('Hoy no hubo suficientes movimientos para resumir la sesión.' if acierto is None else
                        f'Hoy la órtesis se movió hacia donde usted quería en unos {acierto} de cada 10 movimientos.{tend_es} '
                        'El sistema aprende de las señales de su cerebro cuando se equivoca, y ese aprendizaje sigue en cada sesión.'),
        'paciente_en': ('There were not enough movements today to summarise the session.' if acierto is None else
                        f'Today the orthosis moved the way you intended in about {acierto} out of 10 movements.{tend_en} '
                        'The system learns from your brain signals when it makes a mistake, and that learning continues every session.'),
    }


def _figuras(s, claves, base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    filas = s.filas_de()
    x = [f['paso'] for f in filas]
    k = np.ones(20) / 20
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    if len(filas) >= 20:
        ax[0].plot(x[19:], np.convolve([f['error_verdadero'] for f in filas], k, 'valid'), color='#ff7f0e', label='agente / agent')
        ax[0].plot(x[19:], np.convolve([f['error_sombra'] for f in filas], k, 'valid'), color='#7f7f7f', label='sombra / shadow')
    ax[1].plot(x, [f['beta'] for f in filas], color='#1f77b4', label='beta')
    for p in s.perturbaciones:
        for a in ax:
            a.axvline(p, color='r', ls=':', lw=1)
    ax[0].set_ylabel('error (20 pasos / steps)'); ax[1].set_ylabel('beta'); ax[1].set_xlabel('paso / step')
    ax[0].legend(loc='upper right', fontsize=8); ax[0].set_ylim(0, 0.8)
    fig.tight_layout()
    rutas = [base.with_name(base.name + '_fig_sesion.png')]
    fig.savefig(rutas[0], dpi=120); plt.close(fig)
    if len(claves) > 1:
        fig, a = plt.subplots(figsize=(8, 3))
        i = np.arange(len(claves))
        a.bar(i - 0.2, [c.get('error_agente') or 0 for c in claves], 0.4, color='#ff7f0e', label='agente / agent')
        a.bar(i + 0.2, [c.get('error_sombra') or 0 for c in claves], 0.4, color='#7f7f7f', label='sombra / shadow')
        a.set_xticks(i); a.set_xticklabels([c['sesion'][-19:-4] for c in claves], fontsize=7)
        a.set_ylabel('error'); a.legend(fontsize=8); fig.tight_layout()
        rutas.append(base.with_name(base.name + '_fig_comparacion.png'))
        fig.savefig(rutas[1], dpi=120); plt.close(fig)
    return rutas


def informe(s, anteriores=None, cli=None):
    """Informe entre sesiones: cuatro Markdown (terapeuta y paciente, espanol e ingles) con las figuras
    y una propuesta para la proxima sesion, que queda pendiente de la decision de una persona.
    Devuelve {'archivos', 'propuesta', 'origen_texto'}."""
    ant = [clave(Sesion(p)) for p in (anteriores if anteriores is not None else s.anteriores(3))]
    c = clave(s)
    base = s.ruta.with_name(s.ruta.stem)
    figs = _figuras(s, ant + [c], base)
    interp, origen = _interpretacion_plantilla(c, ant), 'reglas'
    if cli is not None:
        try:
            interp, origen = ia.pedir_json(cli, SISTEMA_INFORME, {'sesion': c, 'anteriores': ant}, ESQUEMA_INTERPRETACION), 'api'
        except Exception:
            pass                                                  # sin IA: queda la plantilla
    prop = proponer_y_registrar(s.ruta, resumen_para_propuesta(s), 'proxima_sesion', cli)
    archivos = []
    for publico in ('terapeuta', 'paciente'):
        for idioma, col in (('es', 1), ('en', 2)):
            t = _T[idioma]
            md = [f"# {t['titulo_t' if publico == 'terapeuta' else 'titulo_p']}", '', f'`{s.ruta.name}`', '',
                  f"## {t['interp']}", '', interp[f'{publico}_{idioma}'], '', f"*({t['origen'][origen]})*", '']
            if publico == 'terapeuta':
                md += [f"## {t['cifras']}", '', '| | |', '|---|---|']
                md += [f'| {f[col]} | {_fmt(c.get(f[0]), t)} |' for f in _FILAS]
                md += ['', f"## {t['comp']}", '']
                if ant:
                    md += ['| | ' + ' | '.join(x['sesion'][-19:-4] for x in ant + [c]) + ' |', '|---|' + '---|' * (len(ant) + 1)]
                    md += [f'| {f[col]} | ' + ' | '.join(_fmt(x.get(f[0]), t) for x in ant + [c]) + ' |' for f in _FILAS]
                else:
                    md.append(t['sin_ant'])
                p = prop['propuesta']
                md += ['', f"## {t['prop']}", '', '```json', json.dumps(p, ensure_ascii=False, indent=1), '```', '',
                       f"*{t['pend']} ({t['origen'][prop['origen']]}): `python copiloto.py --sesion {s.ruta.name} --decidir aprobar`*"]
            md += ['', f"## {t['fig']}", ''] + [f'![]({f.name})' for f in figs] + ['', f"*{t['aviso']}*", '']
            ruta = base.with_name(base.name + f'_informe_{publico}_{idioma}.md')
            ruta.write_text('\n'.join(md), encoding='utf-8')
            archivos.append(ruta)
    return {'archivos': archivos + figs, 'propuesta': prop, 'origen_texto': origen}


# ====================================================================== linea de comandos
def ultima(backend='real'):
    rutas = sorted(config.RESULTADOS.glob(f'sesion_{backend}_*.csv'))
    if not rutas:
        raise SystemExit(f'No hay sesiones {backend} en {config.RESULTADOS}.')
    return rutas[-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description='Copiloto clinico de ortesis-bci')
    ap.add_argument('pregunta', nargs='?')
    ap.add_argument('--sesion', help='CSV de la sesion')
    ap.add_argument('--ultima', action='store_true', help='la ultima sesion real grabada')
    ap.add_argument('--sin-ia', dest='sin_ia', action='store_true', help='solo plantillas, aunque haya llave')
    ap.add_argument('--informe', action='store_true', help='informe entre sesiones (Markdown) con propuesta')
    ap.add_argument('--anteriores', nargs='*', help='CSV de las sesiones anteriores (por defecto, las 3 previas)')
    ap.add_argument('--decidir', choices=['aprobar', 'rechazar'],
                    help='decide la propuesta pendiente de la sesion (entre bloques, o la de la proxima sesion)')
    a = ap.parse_args(argv)
    if not a.sesion and not a.ultima:
        ap.error('indica --sesion <csv> o --ultima')
    s = Sesion(a.sesion or ultima())
    cli = None if a.sin_ia else ia.cliente()
    if a.decidir:
        p = pendiente(s.ruta)
        if p is None:
            print('No hay una propuesta pendiente en esta sesion.')
            return 1
        # la de la proxima sesion solo se anota; el efecto de una entre bloques lo registra el orquestador
        decidir(s.ruta, p['id'], a.decidir == 'aprobar', efecto='queda anotada para la proxima sesion; nada cambio '
                'en el codigo ni en la configuracion' if p['momento'] == 'proxima_sesion' else None)
        print(f"Propuesta {p['id']} {'APROBADA' if a.decidir == 'aprobar' else 'RECHAZADA'}: {json.dumps(p['propuesta'], ensure_ascii=False)}")
        return 0
    if a.informe:
        r = informe(s, a.anteriores, cli)
        print(f"Informe ({'con IA' if r['origen_texto'] == 'api' else 'por plantillas'}):")
        for ruta in r['archivos']:
            print(f'  {ruta}')
        print(f"Propuesta para la proxima sesion ({r['propuesta']['origen']}, pendiente de aprobacion): "
              f"{json.dumps(r['propuesta']['propuesta'], ensure_ascii=False)}")
        return 0
    print(f"Sesion {s.ruta.name}: {len(s.filas)} filas. " + ('Responde Claude con herramientas.' if cli else 'Sin llave: responden plantillas.'))
    if a.pregunta:
        print(responder(s, a.pregunta, cli)[0])
        return 0
    while True:                                   # modo interactivo
        try:
            q = input('pregunta> ').strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if q in ('', 'salir', 'exit'):
            return 0
        print(responder(s, q, cli)[0])


if __name__ == '__main__':
    sys.exit(main())
