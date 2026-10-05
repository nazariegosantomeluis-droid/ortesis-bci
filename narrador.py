"""Narrador para el jurado: un proceso aparte que escucha el flujo 'Estado' y cuenta, en una frase
corta, lo que acaba de pasar: checkpoints, perturbacion, recuperacion, congelamiento del
aprendizaje, pausas seguras y el resultado del control causal.

Con llave (ANTHROPIC_API_KEY en .env) la frase la escribe Claude a partir SOLO de los datos del
evento; si la API tarda mas de NARRADOR_API_S, falla, o la frase trae un numero que no esta en el
evento, se usa una plantilla. Nunca bloquea el lazo: es otro proceso y solo lee el flujo.

Cada frase se imprime y se publica en el flujo LSL 'Narracion' (JSON); `tablero.py --narrador` la
muestra en una franja.

Uso:  python narrador.py                  espanol
      python narrador.py --idioma en      ingles
      python narrador.py --sin-ia         solo plantillas
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import config
import ia

# Las frases llevan acentos a proposito: son lo que lee el jurado (flujo LSL y tablero).
PLANTILLAS = {
    'es': {
        'checkpoint_ok': 'Punto de control {n} superado: {que}.',
        'checkpoint_no': 'Punto de control {n} no superado: {que}.',
        'perturbacion': 'Perturbamos el decoder a propósito: ahora se equivoca más. ¿Lo corregirá el agente con el potencial de error del piloto?',
        'recuperacion': 'El agente se recuperó de la perturbación en {pasos} pasos (unos {segundos} s), guiado solo por el potencial de error del piloto.',
        'congelamiento': 'El detector de ErrP dejó de ser fiable: el agente deja de aprender para no aprender mal.',
        'descongelado': 'El detector de ErrP vuelve a ser fiable: el agente aprende de nuevo.',
        'pausa': 'Pausa segura ({motivo}): la órtesis se abre y el agente no aprende hasta que todo vuelva a estar bien.',
        'reanudado': 'Todo en orden otra vez: el lazo continúa donde se quedó.',
        'sham_claro': 'Control causal: con el potencial de error del piloto el agente se recuperó; sin él, no.',
        'sham_dudoso': 'Control causal: en esta sesión los dos bloques no se separaron con claridad.',
        'que': {1: 'señal y órtesis', 2: 'imaginación motora', 3: 'detector de ErrP', 4: 'recuperación tras la perturbación'},
        'motivo': {'eeg': 'se perdió el EEG', 'canal': 'un electrodo se despegó', 'ortesis': 'la órtesis no responde'},
    },
    'en': {
        'checkpoint_ok': 'Checkpoint {n} passed: {que}.',
        'checkpoint_no': 'Checkpoint {n} not passed: {que}.',
        'perturbacion': 'We perturbed the decoder on purpose: it now makes more mistakes. Will the agent fix it using the pilot\'s error potential?',
        'recuperacion': 'The agent recovered from the perturbation in {pasos} steps (about {segundos} s), guided only by the pilot\'s error potential.',
        'congelamiento': 'The ErrP detector stopped being reliable: the agent stops learning rather than learn the wrong thing.',
        'descongelado': 'The ErrP detector is reliable again: the agent resumes learning.',
        'pausa': 'Safe pause ({motivo}): the orthosis opens and the agent does not learn until everything is fine again.',
        'reanudado': 'All good again: the loop continues where it left off.',
        'sham_claro': 'Causal control: with the pilot\'s error potential the agent recovered; without it, it did not.',
        'sham_dudoso': 'Causal control: in this session the two blocks did not separate clearly.',
        'que': {1: 'signal and orthosis', 2: 'motor imagery', 3: 'ErrP detector', 4: 'recovery after the perturbation'},
        'motivo': {'eeg': 'EEG lost', 'canal': 'an electrode came loose', 'ortesis': 'the orthosis is not responding'},
    },
}
# El nombre del idioma lleva acento a proposito. Con la API real (5 de octubre), con «espanol» y un prompt sin acentos, 3 de 7
# frases en espanol salieron sin acentos («ortesis», «recupero», «proposito») y es lo que lee el jurado: el modelo copia la
# ortografia del prompt. Por eso tambien se le pide la ortografia completa.
IDIOMAS_API = {'es': 'español', 'en': 'inglés'}
SISTEMA = (
    'Eres el narrador de una demostracion en vivo ante un jurado: una ortesis de mano controlada por imaginacion '
    'motora, con un agente que se corrige usando el potencial de error (ErrP) del cerebro del piloto. Recibes UN '
    'evento en JSON. Escribe UNA frase corta (maximo {n} caracteres) que explique a un publico no experto que acaba '
    'de pasar y por que importa. Usa solo los datos del evento: no agregues cifras que no esten ahi ni des '
    'diagnosticos. Sin comillas, sin emojis, sin preambulo. Escribe con la ortografia completa del idioma, acentos '
    'incluidos. Idioma: {idioma}.')


class Narrador:
    """Convierte los eventos del flujo Estado en hechos que vale la pena contar, y cada hecho en una frase."""

    def __init__(self, idioma='es', cli=None):
        if idioma not in PLANTILLAS:
            raise ValueError(f'idioma debe ser uno de {sorted(PLANTILLAS)}')
        self.idioma, self.cli = idioma, cli
        self.perturbado = self.congelado = self.en_pausa = False
        self.beta_previa, self.meta_beta, self.paso_perturbacion = 0.0, None, None

    def observar(self, e):
        """Los hechos (dicts con 'evento') que produce un evento del flujo Estado; casi siempre ninguno."""
        tipo, hechos = e.get('tipo'), []
        if tipo == 'checkpoint':
            hechos.append({'evento': 'checkpoint', 'n': e['n'], 'ok': bool(e['ok'])})
        elif tipo == 'salud' and e.get('estado') == 'PAUSA_SEGURA' and not self.en_pausa:
            self.en_pausa = True
            hechos.append({'evento': 'pausa', 'motivo': e.get('motivo') or 'eeg'})
        elif tipo == 'sham':
            # con nombres que dicen que es cada cifra: con «agente: 0.19» la API real narro un error (menor es mejor) como si fuera
            # un puntaje («llego a 0.19... solo a 0.33»)
            hechos.append({'evento': 'sham', 'solo_real': bool(e.get('solo_real')),
                           **{b: {'error_del_agente': e[b]['agente'], 'pasos_para_recuperarse': e[b]['pasos']}
                              for b in config.BLOQUES_SHAM if e.get(b)}})
        elif tipo == 'paso':
            if self.en_pausa:
                self.en_pausa = False
                hechos.append({'evento': 'reanudado', 'paso': e['paso']})
            ciego = e.get('bloque') is not None      # control causal: nada que delate cual bloque es el real
            if e['perturbado'] and not self.perturbado:
                self.meta_beta = self.beta_previa + 0.7 * config.PERTURBACION_LOGITS
                self.paso_perturbacion = e['paso']
                hechos.append({'evento': 'perturbacion', 'paso': e['paso'], 'logits': config.PERTURBACION_LOGITS})
            elif not e['perturbado']:
                self.meta_beta = None
            if self.meta_beta is not None and e['beta'] >= self.meta_beta:
                pasos = e['paso'] - self.paso_perturbacion + 1
                self.meta_beta = None
                if not ciego:
                    hechos.append({'evento': 'recuperacion', 'paso': e['paso'], 'pasos': pasos,
                                   'segundos': round(pasos * config.CICLO_S)})
            if e['congelado'] != self.congelado and not ciego:
                hechos.append({'evento': 'congelamiento' if e['congelado'] else 'descongelado', 'paso': e['paso'],
                               'youden_vivo': round(e['youden'], 2)})
            self.perturbado, self.congelado, self.beta_previa = e['perturbado'], e['congelado'], e['beta']
        return hechos

    def plantilla(self, h):
        t = PLANTILLAS[self.idioma]
        if h['evento'] == 'checkpoint':
            return t['checkpoint_ok' if h['ok'] else 'checkpoint_no'].format(n=h['n'], que=t['que'].get(h['n'], ''))
        if h['evento'] == 'pausa':
            return t['pausa'].format(motivo=t['motivo'].get(h['motivo'], h['motivo']))
        if h['evento'] == 'sham':
            return t['sham_claro' if h['solo_real'] else 'sham_dudoso']
        return t[h['evento']].format(**h)

    def frase(self, h, nacio=None):
        """(frase, 'api' | 'plantilla'). Nunca lanza. nacio: time.time() del hecho; si ya es viejo
        (la API iba atrasada) no se pregunta: una frase tardia confunde mas de lo que explica."""
        if self.cli is not None and (nacio is None or time.time() - nacio <= config.NARRADOR_VIGENCIA_S):
            try:
                r = self.cli.messages.create(
                    model=config.IA_MODELO, max_tokens=config.NARRADOR_MAX_TOKENS,
                    system=SISTEMA.format(n=config.NARRADOR_MAX_CARACTERES, idioma=IDIOMAS_API[self.idioma]),
                    messages=[{'role': 'user', 'content': ia.a_json(h)}], output_config={'effort': 'low'})
                txt = ' '.join(ia.texto(r).split())
                if r.stop_reason == 'end_turn' and frase_valida(txt, h):
                    return txt, 'api'
            except Exception:
                pass                                 # sin red, tiempo agotado, la API declino...: plantilla
        return self.plantilla(h), 'plantilla'


def frase_valida(txt, h):
    """Corta, de una linea, y sin cifras que no esten en el evento (redondeadas, o en porcentaje)."""
    if not txt or len(txt) > config.NARRADOR_MAX_CARACTERES or '\n' in txt:
        return False
    permitidos = set()

    def junta(v):
        if isinstance(v, dict):
            for x in v.values():
                junta(x)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            for x in (v, 100 * v):
                permitidos.update({f'{x:.0f}', f'{x:.1f}', f'{x:.2f}', f'{x:g}'})
    junta(h)
    return all(n.replace(',', '.').rstrip('.') in permitidos for n in re.findall(r'\d+(?:[.,]\d+)?', txt))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Narrador para el jurado')
    ap.add_argument('--idioma', choices=sorted(PLANTILLAS), default='es')
    ap.add_argument('--sin-ia', dest='sin_ia', action='store_true', help='solo plantillas, aunque haya llave')
    a = ap.parse_args(argv)
    from pylsl import StreamInlet, StreamOutlet, resolve_byprop
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')     # una consola sin acentos no tumba al narrador
    cli = None if a.sin_ia else ia.cliente(config.NARRADOR_API_S, reintentos=0)    # NARRADOR_API_S es un tope: sin reintento no se duplica
    narrador = Narrador(a.idioma, cli)
    salida = StreamOutlet(config.crear_info('Narracion'))
    print(f"Narrador ({a.idioma}, {'Claude con plantillas de respaldo' if cli else 'plantillas'}): esperando el flujo Estado...",
          flush=True)
    entrada = None
    hilo, cola = ThreadPoolExecutor(max_workers=1), deque()      # un hilo: las frases salen en orden
    try:
        while True:
            if entrada is None:
                flujos = resolve_byprop('name', 'Estado', timeout=2.0)
                if flujos:
                    entrada = StreamInlet(flujos[0], max_buflen=60)
                    print('Conectado al orquestador.', flush=True)
                continue
            m, _ = entrada.pull_sample(timeout=0.2)
            if m is not None:
                try:
                    for h in narrador.observar(json.loads(m[0])):
                        cola.append((h, hilo.submit(narrador.frase, h, time.time())))
                except (ValueError, KeyError, TypeError):
                    pass                             # un evento raro no detiene al narrador
            while cola and cola[0][1].done():
                h, f = cola.popleft()
                txt, origen = f.result()
                salida.push_sample([json.dumps({'texto': txt, 'evento': h['evento'], 'origen': origen}, ensure_ascii=False)])
                print(f'[{origen}] {txt}', flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        hilo.shutdown(wait=False, cancel_futures=True)


if __name__ == '__main__':
    main()
