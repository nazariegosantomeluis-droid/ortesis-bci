"""Estado del sistema: revision previa a la sesion y franja en vivo del tablero.

  python estado_sistema.py                   revision previa (unos 15 s; no mueve la ortesis)
  python estado_sistema.py --puerto COM4     ademas mide la latencia del ACK con 10 movimientos
  python estado_sistema.py --ortesis-sim     lo mismo con la ortesis simulada (para probar)
  python estado_sistema.py --sin-api         no llama a la API: solo mira que haya llave
  python tablero.py --estado-sistema         la misma revision, en vivo, en una franja del tablero

Revisa, y para cada falla dice la solucion en una linea:
  casco      bateria, validez, perdidas por contador y calidad por canal
  flujos     flujos LSL del contrato y su tasa real; dos fuentes de EEG a la vez; flujos repetidos
  procesos   puente, gemelo, orquestador, tablero, narrador y LabRecorder vivos
  ack        latencia del ACK de la ortesis (los mismos umbrales del CP1)
  laptop     con cargador, suspension por inactividad desactivada, tapa
  disco      espacio libre donde se graba
  env, api   llave en .env y una llamada minima a la API
  git        rama, commit y cambios sin commit del codigo en uso

Cada revision es una funcion pura (revisar_*) sobre lo que entrega un lector (leer_* / listar_*):
asi se prueba sin casco, sin ortesis y sin red. Nada de esto corre dentro del lazo de control: el
tablero lo llama en otro hilo y el orquestador no lo importa.

Bateria y validez: la app UnicornLSL las trae en su flujo; el flujo 'EEG' del puente no, asi que
puente_lsl.py --estado las deja en config.ESTADO_PUENTE_JSON. Las perdidas se cuentan igual con las
dos fuentes: la hora de cada muestra sale del contador del casco, y una perdida es un hueco de hora.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque

import numpy as np

import config

OK, AVISO, FALLA = 'OK', 'AVISO', 'FALLA'
U = config.ESTADO_SISTEMA
SIN_VENTANA = getattr(subprocess, 'CREATE_NO_WINDOW', 0)     # que no parpadee una consola desde el tablero


def res(clave, estado, texto, solucion=''):
    return {'clave': clave, 'estado': estado, 'texto': texto, 'solucion': '' if estado == OK else solucion}


def _correr(*cmd):
    """Salida de un comando del sistema, o None si no existe, falla o tarda."""
    try:
        r = subprocess.run(cmd, cwd=config.RAIZ, capture_output=True, text=True, errors='replace', timeout=10,
                           creationflags=SIN_VENTANA)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


# ====================================================================== flujos LSL
def listar_flujos(espera=1.0):
    from pylsl import resolve_streams
    return [{'nombre': s.name(), 'tipo': s.type(), 'canales': s.channel_count(), 'hz': s.nominal_srate(),
             'host': s.hostname(), 'info': s} for s in resolve_streams(wait_time=espera)]


def es_fuente_eeg(f):
    """El flujo 'EEG' del contrato (puente o gemelo) o el de la app UnicornLSL."""
    return f['tipo'] in (config.FLUJOS['EEG'][0], config.FUENTES_EEG['unicornlsl']['tipo'])


def medir_tasa(info, segundos=1.5):
    """Muestras por segundo que de verdad llegan de un flujo, o None si no se pudo abrir."""
    from pylsl import StreamInlet
    try:
        entrada = StreamInlet(info, max_buflen=5)
        entrada.open_stream(timeout=2.0)
        entrada.pull_chunk(timeout=0.2)                      # lo que hubiera en cola no cuenta
        t0, n = time.monotonic(), 0
        while time.monotonic() - t0 < segundos:
            n += len(entrada.pull_chunk(timeout=0.1)[1])
        return n / (time.monotonic() - t0)
    except Exception:
        return None


def revisar_flujos(flujos, tasas=None):
    """flujos: los de listar_flujos(). tasas: {nombre: (Hz medidos | None, Hz nominales)}."""
    r, fuentes = [], [f for f in flujos if es_fuente_eeg(f)]
    lista = ', '.join(f"{f['nombre']}" + (f" {f['hz']:g} Hz" if f['hz'] else '') for f in flujos) or 'ninguno'
    if not fuentes:
        r.append(res('flujos', FALLA, f'no hay flujo de EEG en la red (flujos: {lista})',
                     'lanza python puente_lsl.py --placa unicorn (o abre UnicornLSL, conecta el casco y pulsa Start)'))
    else:
        r.append(res('flujos', OK, lista))
        varias = len(fuentes) > 1
        r.append(res('fuentes_eeg', FALLA if varias else OK,
                     f"{len(fuentes)} fuentes de EEG a la vez: " + ', '.join(f"{f['nombre']} en {f['host']}" for f in fuentes)
                     if varias else f"una sola fuente de EEG ({fuentes[0]['nombre']})",
                     'cierra la que sobra (un gemelo o un puente olvidado en otra terminal); python ver_flujos.py --solo_lista las lista'))
    nombres = [f['nombre'] for f in flujos if f['nombre'] in config.FLUJOS and not es_fuente_eeg(f)]
    repetidos = sorted({n for n in nombres if nombres.count(n) > 1})
    if repetidos:
        r.append(res('flujos_repetidos', FALLA, 'hay mas de un flujo ' + ', '.join(repetidos),
                     'dos orquestadores (o pruebas.py) publican a la vez: cierra el que sobra'))
    if tasas:
        deficit = lambda v: 1.0 if v[0] is None else max(0.0, v[1] - v[0]) / v[1]
        peor = max(deficit(v) for v in tasas.values())
        s = config.SALUD
        r.append(res('tasas', FALLA if peor >= s['eeg_tasa_rojo'] else AVISO if peor >= s['eeg_tasa_amarillo'] else OK,
                     ', '.join(f"{n} " + ('sin datos' if v[0] is None else f'{v[0]:.0f}') + f' de {v[1]:g} Hz'
                               for n, v in tasas.items()),
                     'llegan menos muestras de las debidas: acerca el dongle al casco, sin obstaculos, y aleja telefonos y otros Bluetooth'))
    return r


# ====================================================================== casco
def leer_estado_puente():
    """Lo que dejo puente_lsl.py --estado, o None si no hay o ya es viejo."""
    try:
        d = json.loads(config.ESTADO_PUENTE_JSON.read_text())
    except (OSError, ValueError):
        return None
    return d if time.time() - d.get('t', 0) <= U['puente_vigencia_s'] else None


def fraccion_perdida(t, fs):
    """Fraccion de muestras perdidas en un tramo, por los huecos de su hora (la hora de cada
    muestra sale del contador del casco: una perdida de Bluetooth es un hueco)."""
    if len(t) < 2:
        return 0.0
    faltan = np.round(np.diff(t) * fs) - 1
    faltan = float(faltan[faltan > 0].sum())
    return faltan / (faltan + len(t))


def leer_casco(eeg, segundos_abierto):
    """Lecturas del casco con una hardware.EntradaEEG ya abierta."""
    x, t = eeg._crudo(U['ventana_s'])
    con_datos = x.ndim == 2 and x.shape[1] >= 2 * eeg.fs
    l = {'bateria': eeg.bateria, 'validas': None, 'edad_s': eeg.edad(), 'tasa_hz': eeg.lecturas()['tasa_hz'],
         'perdidas': fraccion_perdida(t, eeg.fs), 'calidad': eeg.calidad(U['ventana_s']) if con_datos else []}
    if eeg.fuente['validez'] is not None:                    # UnicornLSL: van en el propio flujo
        l['validas'] = 1.0 - eeg.invalidas / max(segundos_abierto * eeg.fs, 1.0)
    else:
        p = leer_estado_puente()
        if p:
            l['bateria'] = p.get('bateria')
            if p.get('invalidas') is not None:
                l['validas'] = 1.0 - p['invalidas'] / max(p.get('muestras') or 0, 1)
    return l


def revisar_casco(l):
    """l: {'bateria' (% | None), 'validas' (fraccion | None), 'perdidas' (fraccion), 'calidad'
    (filas de EntradaEEG.calidad), 'edad_s'}."""
    r = []
    b = l['bateria']
    if b is None:
        r.append(res('bateria', AVISO, 'sin dato de la bateria del casco (el flujo EEG del puente no la lleva)',
                     'lanza el puente con --estado: python puente_lsl.py --placa unicorn --estado'))
    else:
        r.append(res('bateria', FALLA if b < U['bateria_falla'] else AVISO if b < U['bateria_aviso'] else OK,
                     f'bateria del casco {b:.0f} %', 'carga el casco por USB antes de la sesion (no transmite mientras carga)'))
    if l['validas'] is not None:
        r.append(res('validez', OK if l['validas'] >= U['validez_min'] else AVISO,
                     f"{100 * l['validas']:.1f} % de las muestras marcadas como validas",
                     'el casco marca muestras no validas: revisa que este bien puesto y cerca de su dongle'))
    if l['edad_s'] > config.SALUD['eeg_edad_rojo_s']:
        r.append(res('perdidas', FALLA, f"no llegan muestras hace {l['edad_s']:.0f} s",
                     'revisa que el puente siga vivo y el casco encendido y emparejado con SU dongle'))
    else:
        p = l['perdidas']
        r.append(res('perdidas', FALLA if p >= U['perdidas_falla'] else AVISO if p >= U['perdidas_aviso'] else OK,
                     f"{100 * p:.2f} % de muestras perdidas por contador en los ultimos {U['ventana_s']:.0f} s",
                     'acerca el dongle al casco (cable de extension USB), sin obstaculos, y aleja telefonos y otros Bluetooth'))
    malos = [f for f in l['calidad'] if not f['ok']]
    if not l['calidad']:
        r.append(res('canales', AVISO, 'todavia no hay EEG suficiente para la calidad por canal', 'espera unos segundos'))
    else:
        r.append(res('canales', FALLA if malos else OK,
                     'revisar ' + ', '.join(f"{f['canal']} ({f['rms_uv']:.0f} uV RMS, 60 Hz {f['red']:.0%})" for f in malos)
                     if malos else f"{len(l['calidad'])} canales bien ("
                     + ', '.join(f"{f['canal']} {f['rms_uv']:.0f}" for f in l['calidad']) + ' uV RMS)',
                     'reacomoda esos electrodos, agrega gel o solucion y espera un minuto'))
    return r


# ====================================================================== procesos
def listar_procesos():
    """[(pid, pid del padre, linea de comando)] de todos los procesos, o None si no se pudo preguntar."""
    # ponytail: lanza un PowerShell por consulta (~0.5 s); con psutil seria gratis, pero es otra dependencia
    if os.name == 'nt':
        txt = _correr('powershell', '-NoProfile', '-Command',
                      'Get-CimInstance Win32_Process | Where-Object { $_.CommandLine } | '
                      'ForEach-Object { "$($_.ProcessId) $($_.ParentProcessId) $($_.CommandLine)" }')
    else:
        txt = _correr('ps', '-eo', 'pid=,ppid=,args=')
    if txt is None:
        return None
    procesos = []
    for linea in txt.splitlines():
        partes = linea.split(None, 2)
        if len(partes) == 3 and partes[0].isdigit() and partes[1].isdigit():
            procesos.append((int(partes[0]), int(partes[1]), partes[2]))
    return procesos


def vivos(procesos, nombre):
    """PID de los procesos que corren `nombre`. Un guion .py cuenta solo si es el guion que corre un
    Python (no una terminal que lo menciona), y una vez: el python.exe de un .venv de Windows es un
    lanzador que arranca al interprete de verdad con los mismos argumentos; se queda el hijo."""
    if not nombre.endswith('.py'):
        return [pid for pid, _, cmd in procesos if nombre.lower() in cmd.lower()]
    patron = re.compile(r'^"?[^"]*python[\w.]*"?\s+(-\S+\s+)*"?([^"\s]*[\\/])?' + re.escape(nombre) + r'("|\s|$)', re.I)
    pids = {pid: padre for pid, padre, cmd in procesos if patron.search(cmd)}
    return [pid for pid in pids if pid not in pids.values()]


def revisar_procesos(procesos):
    """procesos: los de listar_procesos(), o None."""
    if procesos is None:
        return [res('procesos', AVISO, 'no se pudo listar los procesos', 'revisa a mano las terminales abiertas')]
    v = {n: vivos(procesos, n) for n in config.PROCESOS}
    lista = ', '.join(f"{n} ({', '.join(map(str, p))})" for n, p in v.items() if p) or 'ninguno del proyecto'
    fuentes = v['puente_lsl.py'] + v['cerebro_sintetico.py']
    if len(fuentes) > 1:
        return [res('procesos', FALLA, f'hay {len(fuentes)} fuentes de EEG vivas (puente o gemelo): {lista}',
                    f"cierra la que sobra con su PID (taskkill /PID {fuentes[0]} /F); nunca taskkill /IM python.exe")]
    if len(v['orquestador.py']) > 1:
        return [res('procesos', FALLA, f'hay dos orquestadores vivos: {lista}',
                    f"cierra el que sobra con su PID (taskkill /PID {v['orquestador.py'][0]} /F)")]
    if not v['LabRecorder']:
        return [res('procesos', AVISO, f'vivos: {lista}; LabRecorder no esta abierto',
                    'abre LabRecorder, selecciona todos los flujos y graba: sin el no queda el XDF de la sesion')]
    return [res('procesos', OK, f'vivos: {lista}')]


# ====================================================================== ortesis
def medir_ack(ortesis, movimientos=U['ack_movimientos']):
    """Latencias del ACK (nan = perdido) de unos movimientos cortos, como el CP1."""
    latencias = []
    for k in range(movimientos):
        _, t_ack, lat = ortesis.mover(0.4 if k % 2 else 0.6)
        latencias.append(lat if t_ack is not None else float('nan'))
        time.sleep(0.15)
    return latencias


def revisar_ack(latencias):
    if len(latencias) < U['ack_movimientos']:
        return [res('ack', AVISO, f'latencia del ACK sin medir ({len(latencias)} movimientos)',
                    'antes de la sesion: python estado_sistema.py --puerto COM4; en el lazo se mide sola')]
    import hardware as hw
    e = hw.evaluar_latencias(latencias)
    return [res('ack', OK if e['ok'] else FALLA, e['texto'],
                'cambia el cable o el puerto USB, cierra lo que use el COM (monitor serie de Arduino) y reinicia el ESP32')]


# ====================================================================== laptop y disco
def indices_powercfg(texto):
    """(con cargador, con bateria) de la salida de `powercfg /query`: los dos ultimos valores
    hexadecimales. Asi no depende del idioma de Windows."""
    h = re.findall(r'0x[0-9a-fA-F]{8}', texto or '')
    return (int(h[-2], 16), int(h[-1], 16)) if len(h) >= 2 else None


def leer_energia():
    """Lo que no cambia durante la sesion: suspension por inactividad (s) y accion al cerrar la tapa."""
    if os.name != 'nt':
        return {'suspension': None, 'tapa': None}
    consulta = lambda sub, ajuste: indices_powercfg(_correr('powercfg', '/query', 'SCHEME_CURRENT', sub, ajuste))
    return {'suspension': consulta('SUB_SLEEP', 'STANDBYIDLE'), 'tapa': consulta('SUB_BUTTONS', 'LIDACTION')}


def leer_cargador():
    """(con cargador: True | False | None, % de bateria de la laptop | None)."""
    if os.name != 'nt':
        return None, None
    import ctypes

    class Estado(ctypes.Structure):
        _fields_ = [('ac', ctypes.c_ubyte), ('bandera', ctypes.c_ubyte), ('porcentaje', ctypes.c_ubyte),
                    ('ahorro', ctypes.c_ubyte), ('vida', ctypes.c_ulong), ('vida_total', ctypes.c_ulong)]
    e = Estado()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(e)):
        return None, None
    return {0: False, 1: True}.get(e.ac), (e.porcentaje if e.porcentaje <= 100 else None)


def revisar_laptop(cargador, bateria, energia):
    """cargador y bateria: de leer_cargador(); energia: de leer_energia()."""
    r = []
    if cargador is None:
        r.append(res('cargador', AVISO, 'no se pudo leer si la laptop tiene el cargador', 'revisalo a mano'))
    else:
        r.append(res('cargador', OK if cargador else FALLA,
                     'laptop con cargador' if cargador else 'la laptop esta con bateria'
                     + (f' ({bateria} %)' if bateria is not None else ''), 'conecta el cargador de la laptop'))
    i = 1 if cargador is False else 0                        # el ajuste que rige ahora
    s = energia.get('suspension')
    if s is None:
        r.append(res('suspension', AVISO, 'no se pudo leer la suspension por inactividad (powercfg)',
                     'revisala a mano: Configuracion > Sistema > Energia > Pantalla y suspension'))
    else:
        r.append(res('suspension', OK if s[i] == 0 else FALLA,
                     'suspension por inactividad desactivada' if s[i] == 0
                     else f'la laptop se suspende tras {s[i] / 60:.0f} min de inactividad',
                     f"powercfg /change standby-timeout-{'dc' if i else 'ac'} 0"))
    t = energia.get('tapa')
    if t is not None:
        r.append(res('tapa', OK if t[i] == 0 else AVISO,
                     'cerrar la tapa no hace nada' if t[i] == 0 else 'cerrar la tapa suspende la laptop',
                     'no cierres la tapa durante la sesion'))
    return r


def revisar_disco(libre_gb):
    return [res('disco', FALLA if libre_gb < U['disco_falla_gb'] else AVISO if libre_gb < U['disco_aviso_gb'] else OK,
                f'{libre_gb:.1f} GB libres donde se graba', 'libera espacio: borra XDF y sesiones viejas de resultados/')]


# ====================================================================== .env y API
SOLUCION_API = {
    'AuthenticationError': 'la llave no es valida: copia otra vez ANTHROPIC_API_KEY en .env',
    'PermissionDeniedError': 'la llave no tiene permiso para ese modelo: revisa la cuenta en console.anthropic.com',
    'NotFoundError': f'el modelo {config.IA_MODELO} no existe para esta llave: revisa config.IA_MODELO',
    'RateLimitError': 'limite de uso alcanzado: espera un minuto o revisa el plan de la cuenta',
    'APIConnectionError': 'no hay conexion a internet: revisa la red (sin ella la IA usa plantillas y reglas)',
    'APITimeoutError': 'la API tarda demasiado: revisa la red (sin ella la IA usa plantillas y reglas)',
}


def revisar_api(cli=None, llamar=True):
    """La llave y, con llamar=True, una llamada minima a la API. cli: un cliente ya hecho (pruebas).
    Nunca imprime la llave. Sin llave la IA no es una falla: usa plantillas y reglas."""
    import ia
    if cli is None:
        ia.cargar_env()
        if not os.environ.get('ANTHROPIC_API_KEY'):
            falta = 'no existe .env' if not config.ARCHIVO_ENV.exists() else '.env no tiene ANTHROPIC_API_KEY'
            return [res('env', AVISO, f'{falta}: copiloto, co-investigador y narrador usan plantillas y reglas',
                        f'crea {config.ARCHIVO_ENV.name} en la carpeta del proyecto con la linea ANTHROPIC_API_KEY=...')]
        cli = ia.cliente(U['api_s'])
        if cli is None:
            return [res('env', AVISO, 'hay llave, pero falta el paquete anthropic', 'pip install -r requirements.txt')]
    r = [res('env', OK, 'hay llave de API')]
    if not llamar:
        return r
    t0 = time.time()
    try:
        cli.messages.create(model=config.IA_MODELO, max_tokens=U['api_max_tokens'],
                            messages=[{'role': 'user', 'content': 'Responde solo: ok'}],
                            output_config={'effort': 'low'})
    except Exception as e:
        return r + [res('api', FALLA, f'la llamada a la API fallo: {type(e).__name__}: {str(e)[:120]}',
                        SOLUCION_API.get(type(e).__name__, 'revisa la llave, la red y config.IA_MODELO; sin API la IA usa plantillas y reglas'))]
    return r + [res('api', OK, f'{config.IA_MODELO} respondio en {time.time() - t0:.1f} s')]


# ====================================================================== git
def leer_git():
    git = lambda *a: (_correr('git', *a) or '').strip()
    return {'rama': git('rev-parse', '--abbrev-ref', 'HEAD'), 'version': git('describe', '--tags', '--always'),
            'sucios': len(git('status', '--porcelain', '--untracked-files=no').splitlines()),
            'atras': int(git('rev-list', '--count', 'HEAD..@{u}') or 0)}


def revisar_git(g):
    if not g['version']:
        return [res('git', AVISO, 'no se pudo leer la version de git', 'corre desde la carpeta del repositorio, con git instalado')]
    txt = f"rama {g['rama']}, version {g['version']}"
    if g['sucios']:
        return [res('git', AVISO, f"{txt}, con {g['sucios']} archivos modificados sin commit",
                    'el codigo en uso no es el de ningun commit: git status, y luego git stash o git commit')]
    if g['atras']:
        return [res('git', AVISO, f"{txt}, {g['atras']} commits detras de su rama remota",
                    'git pull, si Luis ya aprobo lo nuevo')]
    return [res('git', OK, txt)]


# ====================================================================== todo junto
class Monitor:
    """Una revision completa por ciclo(). Mantiene abierta la entrada de EEG entre ciclos y guarda
    lo que no cambia durante la sesion (ajustes de energia y version de git). latencias: las del ACK
    (ms; nan = perdido); en el tablero las llena cada paso del flujo Estado."""

    def __init__(self, latencias=None):
        self.eeg, self._t_eeg, self._fijo = None, None, None
        self.latencias = latencias if latencias is not None else deque(maxlen=config.CP1_MOVIMIENTOS)

    def _entrada(self, flujos):
        """La entrada de EEG de la fuente que haya (se abre una vez), o None."""
        # ponytail: si a media sesion se cambia de puente a UnicornLSL hay que reabrir el tablero
        if self.eeg is None:
            app = config.FUENTES_EEG['unicornlsl']
            if any(f['nombre'] == config.FUENTES_EEG['puente']['nombre'] for f in flujos):
                fuente, nombre = 'puente', None
            else:
                nombre = next((f['nombre'] for f in flujos if f['tipo'] == app['tipo'] and f['canales'] == app['canales']), None)
                if nombre is None:
                    return None
                fuente = 'unicornlsl'
            import hardware as hw
            try:
                self.eeg = hw.EntradaEEG(segundos=2 * U['ventana_s'], timeout=2.0, fuente=fuente, nombre=nombre)
            except RuntimeError:
                return None
            self._t_eeg = time.monotonic()
        return self.eeg

    def ciclo(self, espera_s=0.0, api=False):
        """Lista de resultados. espera_s: segundos que junta EEG antes de mirarlo (revision previa).
        api=True hace la llamada minima a la API; en vivo solo se mira que haya llave."""
        flujos = listar_flujos()
        eeg = self._entrada(flujos)
        time.sleep(espera_s)
        casco, tasas = [], {}
        if eeg is not None:
            l = leer_casco(eeg, time.monotonic() - self._t_eeg)
            casco, tasas[eeg.nombre or eeg.tipo] = revisar_casco(l), (l['tasa_hz'], eeg.fs)
        for f in flujos:
            if f['hz'] > 0 and not es_fuente_eeg(f):
                tasas[f['nombre']] = (medir_tasa(f['info']), f['hz'])
        if self._fijo is None:
            self._fijo = {'energia': leer_energia(), 'git': revisar_git(leer_git())}
        base = config.RESULTADOS if config.RESULTADOS.exists() else config.RAIZ
        return (casco + revisar_flujos(flujos, tasas) + revisar_procesos(listar_procesos())
                + revisar_ack(list(self.latencias)) + revisar_laptop(*leer_cargador(), self._fijo['energia'])
                + revisar_disco(shutil.disk_usage(base).free / 1e9) + revisar_api(llamar=api) + self._fijo['git'])

    def cerrar(self):
        if self.eeg is not None:
            self.eeg.cerrar()


def texto(resultados):
    """Una linea por revision; las que no estan bien llevan debajo su solucion."""
    lineas = []
    for r in resultados:
        lineas.append(f"  {r['estado']:6s} {r['clave']}: {r['texto']}")
        if r['solucion']:
            lineas.append(f"         -> {r['solucion']}")
    return '\n'.join(lineas)


def previa(puerto=None, ortesis_sim=False, api=True, salida=print):
    """La revision previa completa. Devuelve los resultados."""
    monitor, falla_puerto = Monitor(), []
    if puerto or ortesis_sim:
        import hardware as hw
        try:
            ortesis = hw.OrtesisSimulada() if ortesis_sim else hw.OrtesisSerial(puerto)
        except Exception as e:                               # puerto ocupado o inexistente
            falla_puerto = [res('ack', FALLA, f'no se pudo abrir {puerto}: {type(e).__name__}: {str(e)[:100]}',
                                'cierra el monitor serie de Arduino y mira el numero del puerto en el Administrador de dispositivos')]
        else:
            salida(f"Midiendo la latencia del ACK ({U['ack_movimientos']} movimientos)...")
            try:
                monitor.latencias.extend(medir_ack(ortesis))
            finally:
                ortesis.cerrar()
    salida(f"Revisando el sistema (unos {U['ventana_s'] + 8:.0f} s)...")
    try:
        resultados = monitor.ciclo(espera_s=U['ventana_s'] + 1.0, api=api)
    finally:
        monitor.cerrar()
    if falla_puerto:
        resultados = [falla_puerto[0] if r['clave'] == 'ack' else r for r in resultados]
    salida(texto(resultados))
    n = {e: sum(r['estado'] == e for r in resultados) for e in (FALLA, AVISO)}
    salida(f"\n{n[FALLA]} fallas y {n[AVISO]} avisos" + ('' if n[FALLA] else ': nada impide empezar'))
    return resultados


def main():
    ap = argparse.ArgumentParser(description='Revision previa del sistema, con la solucion de cada falla')
    ap.add_argument('--puerto', help='mide la latencia del ACK de la ortesis en este puerto (la mueve 10 veces)')
    ap.add_argument('--ortesis-sim', dest='ortesis_sim', action='store_true', help='lo mismo con la ortesis simulada')
    ap.add_argument('--sin-api', dest='sin_api', action='store_true', help='no llama a la API: solo mira que haya llave')
    a = ap.parse_args()
    resultados = previa(a.puerto, a.ortesis_sim, api=not a.sin_api)
    sys.exit(1 if any(r['estado'] == FALLA for r in resultados) else 0)


if __name__ == '__main__':
    main()
