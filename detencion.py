"""Lo que dice el tablero cuando el orquestador se detiene por un NO GO (CP1, CP2 o CP3).

Antes el tablero se quedaba con los paneles vacios y la unica pista era una linea chica de color. Ahora el
orquestador publica un evento de Estado (`tipo = 'detenida'`) con el checkpoint, el motivo y que hacer, y el
tablero lo muestra en grande. El texto de "que hacer" sale del arbol de decision de docs/DOMINGO.md (seccion 5);
si uno cambia, el otro tambien. Sin Qt ni hardware, para probarlo:

  evento(n, motivo, datos, puerto)    el evento de Estado
  pasos(n, datos, puerto)             que hacer, una frase por linea
  texto_consola(ev)                   lo mismo para la terminal del orquestador
  html(ev)                            lo que pone el tablero (escapado: el motivo viene de texto libre)

`datos` son los numeros que decidieron el NO GO, no texto: CP1 {senal_ok, canales_malos, sin_muestras, latencia_ok},
CP2 {ba, sin_ensayos}, CP3 {ba, espec, sin_epocas}. Con --forzar un NO GO no detiene la sesion y nada de esto se
publica; "no quedo ningun ensayo (o epoca) valido" si la detiene aun con --forzar, porque no hay modelo con que seguir.
El CP4 solo informa, nunca detiene.
"""
import html as _html

import config

def _f(x):
    return f'{x:.2f}'


def pasos(n, datos=None, puerto=None, simulada=False):
    """Que hacer despues de un NO GO en el CP n, en frases cortas (con el comando entre comillas cuando lo hay).
    `puerto` es el COM de la ortesis por USB, o None si va simulada o por Wi-Fi; `simulada`, que es --ortesis-sim."""
    d = datos or {}
    if n == 1:
        r = []
        if d.get('sin_muestras'):
            r += ['No llegaron muestras de EEG: revisa que el casco este encendido y enviando ("python ver_flujos.py" debe '
                  'mostrar un solo EEG con datos) y vuelve a lanzar el orquestador.']
        elif not d.get('senal_ok', True):
            malos = ', '.join(d.get('canales_malos') or []) or 'los electrodos marcados REVISAR'
            r += [f'Reacomoda {malos}, agrega gel y espera 1 minuto.',
                  'Vuelve a lanzar el orquestador.',
                  'Si sigue mal tras dos intentos, no fuerces: con un canal malo el lazo entra en pausa segura todo el '
                  'tiempo. Plan B: "python demo.py planb".']
        if not d.get('latencia_ok', True):
            if puerto:
                r += [f'Cambia el cable o el puerto USB y cierra lo que use el {puerto}.',
                      f'Reinicia el ESP32 y repite "python verificar_ortesis.py --puerto {puerto}".']
            elif simulada:
                r += ['La ortesis es la simulada y aun asi la latencia fallo: la laptop esta saturada. Cierra lo que no uses y repite.']
            else:
                r += ['Revisa la alimentacion y la conexion de la ortesis (Wi-Fi o cable) y repite la medicion.']
            if not simulada:
                r += ['Si la ortesis no responde, haz la demo sin ella: "python orquestador.py real --ortesis-sim". '
                      'El tablero muestra todo; dilo al jurado.']
        return r or ['Mira la consola del orquestador: dice que fallo en el CP1.']
    if n == 2:
        if d.get('sin_ensayos'):
            return ['No quedo ningun ensayo valido: se descartan por electrodo despegado, movimiento de cabeza o EEG sin '
                    'datos frescos (la consola dice cual).',
                    'Revisa los electrodos, pide al piloto que no mueva la cabeza durante la senal y repite.']
        return ['Repite una vez: pide al piloto imaginar la sensacion de cerrar la mano, sin moverla, y relajar de verdad '
                'en RELAJA. Vuelve a lanzar el orquestador.',
                'Si sigue por debajo de 0.70, cambia de piloto si hay otro.',
                'Sin tiempo ni otro piloto: el mismo comando con --forzar sigue con el decoder que haya. La ortesis se '
                'equivocara mas; el agente lo corrige solo si el CP3 sale bien.']
    if n == 3:
        if d.get('sin_epocas'):
            return ['No quedo ninguna epoca valida para el detector de ErrP (la consola dice por que).',
                    'Repite solo la calibracion de ErrP: el mismo comando con --solo-errp, con el piloto mirando la ortesis.']
        ba, espec = d.get('ba'), d.get('espec')
        if ba is not None and ba < config.DETENCION_CP3_BA_NO_INFORMA:
            return [f'El detector no informa (BA {_f(ba)}). Repite la calibracion de ErrP una vez: el mismo comando con '
                    '--solo-errp, con el piloto mirando la ortesis.',
                    'Si no mejora, plan B: "python demo.py planb" (repite una sesion real de hoy; dilo al jurado).']
        if ba is not None and ba >= config.BA_MIN and espec is not None and espec < config.DETENCION_CP3_ESPEC_FALSAS_ALARMAS:
            return [f'Demasiadas falsas alarmas (especificidad {_f(espec)}). Repite solo la calibracion de ErrP: el mismo '
                    'comando con --solo-errp.']
        return ['Dos caminos:',
                '1) Repetir solo la calibracion de ErrP (5 min): el mismo comando con --solo-errp, con el piloto mas atento '
                'a la ortesis.',
                '2) Seguir con este detector: el mismo comando con --saltar-calibracion. Ya no se evalua el CP3; el agente '
                'aprendera mas lento y el CP4 puede dar NO GO. Dilo tal cual.']
    return ['Mira la consola del orquestador y docs/DOMINGO.md, seccion 5.']


def evento(n, motivo, datos=None, puerto=None, simulada=False):
    """El evento de Estado de una parada. `motivo` es la linea del checkpoint ("ErrP: sens 0.83, espec 0.87...").
    Con n = None (la parada no dijo en que CP) no inventa uno: titulo sin CP y la referencia al arbol."""
    return {'tipo': 'detenida', 'n': None if n is None else int(n),
            'titulo': 'Sesion detenida' if n is None else f'Sesion detenida en CP{int(n)}', 'motivo': str(motivo),
            'que_hacer': pasos(n, datos, puerto, simulada)}


def texto_consola(ev):
    """Las mismas lineas para la terminal: titulo con el motivo y despues que hacer."""
    lineas = [f"{ev['titulo']}: {ev['motivo']}", 'Que hacer:']
    return '\n'.join(lineas + [f'  {l}' for l in ev['que_hacer']] + ['  (todo el arbol: docs/DOMINGO.md, seccion 5. Plan B: python demo.py planb)'])


def html(ev):
    """El texto grande del tablero. Todo lo que no es marca propia pasa por html.escape."""
    e = _html.escape
    pasos_html = ''.join(f"<div style='margin-top:8px'>{e(p)}</div>" for p in ev['que_hacer'])
    return (f"<div style='font-size:34px;font-weight:bold;color:white'>{e(ev['titulo'])}: {e(ev['motivo'])}</div>"
            f"<div style='font-size:20px;font-weight:bold;color:white;margin-top:14px'>Que hacer</div>"
            f"<div style='font-size:20px;color:white'>{pasos_html}</div>")
