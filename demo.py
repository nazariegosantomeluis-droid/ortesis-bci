"""Modo demo automatico: revisa los requisitos, lanza los procesos en orden y deja una bitacora.

Es un atajo para la seccion 4 de docs/DOMINGO.md. No cambia el orquestador: los comandos
manuales de esa guia siguen valiendo si algo falla aqui.

  python demo.py preflight                      revisa y dice OK / AVISO / FALLA, con que hacer
  python demo.py preflight --limpiar-modelos    ademas mueve los modelos viejos a modelos/_anteriores/
  python demo.py lanzar                         preflight + puente + tablero + orquestador (plan casco)
  python demo.py lanzar --plan unicornlsl       fuente = app UnicornLSL (no lanza el puente)
  python demo.py lanzar --plan gemelo           sin casco: el gemelo digital como fuente
  python demo.py lanzar -- --sham --preentrenado    lo que va tras `--` se pasa tal cual al orquestador
  python demo.py planb                          tablero + repeticion de la ultima sesion real

Lo que NO hace: no agrega --forzar ni --saltar-calibracion por su cuenta, no esconde un
checkpoint en NO GO y no contesta el cuestionario. Una FALLA del preflight detiene `lanzar`
salvo que se pida --ignorar-fallas (queda anotado en la bitacora).

Los procesos de fondo (puente, tablero, narrador) escriben en resultados/logs_demo/<fecha>/ y
se cierran al terminar por su PID, nunca por nombre. La bitacora queda en resultados/demo_<fecha>.json.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import config

OK, AVISO, FALLA = 'OK', 'AVISO', 'FALLA'
PLANES = ('casco', 'unicornlsl', 'gemelo')
EDAD_VERIFICACION_H = 4.0         # una verificacion del casco o de la ortesis mas vieja que esto ya no vale
DISCO_MIN_MB = 500
ESPERA_FLUJO_S = {'casco': 45.0, 'gemelo': 20.0}     # cuanto esperar a que aparezca el flujo EEG
STREAMS_DEL_ORQUESTADOR = ('Marcadores', 'Paso', 'Estado')


class ErrorDemo(Exception):
    """Algo impide seguir con la demo; el mensaje dice que hacer."""


def rev(clave, estado, texto, que_hacer=''):
    return {'clave': clave, 'estado': estado, 'texto': texto, 'que_hacer': que_hacer}


# ====================================================================== comprobaciones
def _git(*args, timeout=15):
    try:
        r = subprocess.run(['git', *args], cwd=config.RAIZ, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def revisar_codigo(git=_git):
    """Version del codigo, cambios sin guardar y si origin/main trae algo que no tienes."""
    rc, desc = git('describe', '--tags', '--always')
    if rc != 0:
        return [rev('codigo', AVISO, 'no pude leer la version con git', 'Confirma con Luis que codigo se usa hoy.')]
    out = [rev('version', OK if desc == 'v-demo' else AVISO,
               f'version {desc}' + ('' if desc == 'v-demo' else ' (no es exactamente la etiqueta v-demo)'),
               '' if desc == 'v-demo' else 'Confirma con Luis que esta es la version de la demo.')]
    rc, sucio = git('status', '--porcelain', '-uno')
    n = len([l for l in sucio.splitlines() if l.strip()]) if rc == 0 else 0
    out.append(rev('cambios', AVISO if n else OK, f'{n} archivo(s) modificados sin guardar' if n else 'sin cambios sin guardar',
                   'Se usaria codigo que no esta en git: guardalo o descartalo.' if n else ''))
    rc, _ = git('fetch', '--quiet', 'origin', 'main')
    if rc != 0:
        out.append(rev('origin', AVISO, 'sin red o sin acceso a origin: no pude comparar con origin/main',
                       'Normal si no hay internet; la regla del proyecto pide revisar antes de una demo.'))
    else:
        rc, n_nuevos = git('rev-list', '--count', 'HEAD..origin/main')
        nuevos = int(n_nuevos) if rc == 0 and n_nuevos.isdigit() else 0
        out.append(rev('origin', AVISO if nuevos else OK,
                       f'origin/main trae {nuevos} commit(s) que no tienes' if nuevos else 'al dia con origin/main',
                       'Pregunta a Luis antes de seguir (solo el aprueba cambios a main).' if nuevos else ''))
    return out


def revisar_dependencias(plan, ortesis_sim, sin_tablero, importar=importlib.import_module):
    """Importa de verdad lo que hace falta (pylsl carga liblsl; brainflow, su biblioteca nativa)."""
    requeridas = ['numpy', 'scipy', 'sklearn', 'pylsl', 'pyriemann']
    if plan == 'casco':
        requeridas.append('brainflow')
    if not ortesis_sim:
        requeridas.append('serial')
    if not sin_tablero:
        requeridas.append('pyqtgraph')
    faltan = []
    for nombre in requeridas:
        try:
            importar(nombre)
        except Exception as e:                     # ImportError, pero tambien OSError de una DLL
            faltan.append(f'{nombre} ({type(e).__name__})')
    if faltan:
        return [rev('dependencias', FALLA, 'no se pueden importar: ' + ', '.join(faltan),
                    'pip install -r requirements.txt, con el entorno activado (source .venv/Scripts/activate).')]
    return [rev('dependencias', OK, 'importan: ' + ', '.join(requeridas))]


def _modelos_viejos(carpeta):
    return [p for p in Path(carpeta).glob('*')
            if p.is_file() and p.suffix in ('.pkl', '.npz') and p.name != config.DECODER_PREENTRENADO]


def revisar_modelos(carpeta=None, ahora=time.time):
    carpeta = carpeta or config.MODELOS
    viejos = _modelos_viejos(carpeta) if Path(carpeta).exists() else []
    if not viejos:
        return [rev('modelos', OK, 'sin modelos viejos en modelos/')]
    horas = (ahora() - max(p.stat().st_mtime for p in viejos)) / 3600
    return [rev('modelos', AVISO, f'{len(viejos)} archivo(s) de calibraciones anteriores; el mas reciente, de hace {horas:.1f} h',
                'Si no son de este piloto: python demo.py preflight --limpiar-modelos (los mueve a modelos/_anteriores/; '
                'el decoder preentrenado se queda). --saltar-calibracion los cargaria.')]


def limpiar_modelos(carpeta=None, estado_sesion=None, ahora=time.time):
    """Mueve (no borra) los modelos y el estado de sesion a modelos/_anteriores/<fecha>/.
    Devuelve (carpeta destino o None, nombres movidos). El decoder preentrenado se queda."""
    carpeta = Path(carpeta or config.MODELOS)
    estado = Path(estado_sesion or config.ESTADO_SESION_JSON)
    archivos = _modelos_viejos(carpeta) if carpeta.exists() else []
    if estado.exists():
        archivos.append(estado)
    if not archivos:
        return None, []
    destino = carpeta / '_anteriores' / time.strftime('%Y%m%d_%H%M%S', time.localtime(ahora()))
    destino.mkdir(parents=True, exist_ok=True)
    for p in archivos:
        shutil.move(str(p), str(destino / p.name))
    return destino, [p.name for p in archivos]


def _resolver_lsl(espera=2.0):
    from pylsl import resolve_streams
    return [{'name': s.name(), 'type': s.type(), 'host': s.hostname()} for s in resolve_streams(wait_time=espera)]


def revisar_flujos(plan, resolver=_resolver_lsl):
    """Antes de lanzar: ningun flujo EEG ni del orquestador en la red. Con la app UnicornLSL, debe verse su flujo Data."""
    try:
        flujos = resolver()
    except Exception as e:
        return [rev('flujos', FALLA, f'no pude mirar la red LSL: {type(e).__name__}: {e}', 'Revisa la instalacion de pylsl.')]
    eeg = [f for f in flujos if f['name'] == 'EEG']
    orq = [f['name'] for f in flujos if f['name'] in STREAMS_DEL_ORQUESTADOR]
    out = []
    if eeg:
        quien = ', '.join(sorted({f['host'] for f in eeg}))
        out.append(rev('flujo_eeg', FALLA, f'ya hay {len(eeg)} flujo(s) EEG en la red (host: {quien})',
                       'Cierra el gemelo o el puente que quedo abierto (otra terminal u otra maquina) y repite. '
                       'Debe haber un solo flujo EEG; python ver_flujos.py --solo_lista lo muestra.'))
    else:
        out.append(rev('flujo_eeg', OK, 'la red LSL no tiene flujos EEG (lo lanzara este programa)' if plan != 'unicornlsl'
                       else 'la red LSL no tiene flujos EEG'))
    if orq:
        out.append(rev('flujos_orquestador', FALLA, 'ya existen flujos del orquestador: ' + ', '.join(sorted(set(orq))),
                       'Hay otra sesion o python pruebas.py corriendo. Cierrala antes de empezar.'))
    if plan == 'unicornlsl':
        datos = [f for f in flujos if f['type'] == 'Data']
        out.append(rev('app_unicornlsl', OK if datos else FALLA,
                       f"flujo de la app: {datos[0]['name']}" if datos else 'no veo el flujo Data de la app UnicornLSL',
                       '' if datos else 'Abre UnicornLSL, conecta el casco con su dongle y pulsa Start.'))
    return out


def _puertos():
    from serial.tools import list_ports
    return [p.device for p in list_ports.comports()]


def revisar_puerto(puerto, ortesis_sim, listar=_puertos):
    if ortesis_sim:
        return [rev('puerto', OK, 'ortesis simulada: no hace falta puerto')]
    try:
        hay = listar()
    except Exception as e:
        return [rev('puerto', AVISO, f'no pude listar los puertos: {type(e).__name__}: {e}')]
    if puerto.lower() in [p.lower() for p in hay]:
        return [rev('puerto', OK, f'{puerto} existe')]
    return [rev('puerto', FALLA, f"{puerto} no esta; puertos vistos: {', '.join(hay) or 'ninguno'}",
                'Mira el numero en el Administrador de dispositivos y pasalo con --puerto. Cierra el monitor serie de Arduino.')]


def _leer_json(ruta):
    try:
        return json.loads(Path(ruta).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def revisar_verificaciones(plan, ortesis_sim, resultados=None, ahora=time.time):
    """Lo que dejaron verificar_unicorn.py y verificar_ortesis.py hoy (no los corre: son guiados)."""
    resultados = Path(resultados or config.RESULTADOS)
    out = []
    if plan != 'gemelo':
        d = _leer_json(resultados / 'verificacion_unicorn.json')
        fuente = 'brainflow' if plan == 'casco' else 'lsl'
        cmd = 'python verificar_unicorn.py ' + ('brainflow' if plan == 'casco' else 'lsl')
        if d is None:
            out.append(rev('verif_casco', AVISO, 'no hay verificacion del casco', f'Corre {cmd} con el casco puesto.'))
        elif (ahora() - d.get('t', 0)) / 3600 > EDAD_VERIFICACION_H:
            out.append(rev('verif_casco', AVISO, f"la verificacion del casco es de hace {(ahora() - d.get('t', 0)) / 3600:.1f} h",
                           f'Repitela: {cmd}.'))
        else:
            from verificar_unicorn import CRITICAS
            res = d.get('por_fuente', {}).get(fuente)
            malas = sorted({r['clave'] for r in res or [] if r['clave'] in CRITICAS and r['estado'] != 'OK'}) if res else None
            if res is None:
                out.append(rev('verif_casco', AVISO, f'la verificacion de hoy no probo la fuente {fuente}', f'Corre {cmd}.'))
            elif malas:
                out.append(rev('verif_casco', FALLA, f'la verificacion de {fuente} falla en: {", ".join(malas)}',
                               'No calibres con esa fuente (docs/DOMINGO.md, seccion 2).'))
            else:
                out.append(rev('verif_casco', OK, f'la verificacion de hoy aprueba la fuente {fuente}'))
    if not ortesis_sim:
        d = _leer_json(resultados / 'verificacion_ortesis.json')
        if d is None or d.get('simulada'):
            out.append(rev('verif_ortesis', AVISO, 'no hay verificacion de la ortesis real',
                           'Corre python verificar_ortesis.py --puerto COM4 (30 movimientos).'))
        elif (ahora() - d.get('t', 0)) / 3600 > EDAD_VERIFICACION_H:
            out.append(rev('verif_ortesis', AVISO, f"la verificacion de la ortesis es de hace {(ahora() - d.get('t', 0)) / 3600:.1f} h",
                           'Repitela: python verificar_ortesis.py --puerto COM4.'))
        else:
            v = str(d.get('veredicto', ''))
            estado = FALLA if v.startswith('FALLA') else AVISO if v.startswith('AVISO') else OK
            out.append(rev('verif_ortesis', estado, v[:110] or 'sin veredicto',
                           'Mira la tabla de la seccion 3 de docs/DOMINGO.md.' if estado != OK else ''))
    return out


def revisar_llave():
    import ia
    ia.cargar_env()
    hay = bool(os.environ.get('ANTHROPIC_API_KEY'))
    try:
        importlib.import_module('anthropic')
        paquete = True
    except Exception:
        paquete = False
    if hay and paquete:
        return [rev('api', OK, 'hay llave y paquete anthropic (la IA no se ha probado nunca con la API real)')]
    return [rev('api', AVISO, 'sin ' + ('llave' if not hay else 'paquete anthropic') + ': narrador, copiloto y co-investigador usaran plantillas y reglas',
                'Es el plan B y esta probado. Si quieres la API: ANTHROPIC_API_KEY en .env y pip install anthropic.')]


def revisar_plan_b(resultados=None, ahora=time.time):
    resultados = Path(resultados or config.RESULTADOS)
    hay = sorted(resultados.glob('sesion_real_*' + config.SUFIJO_ESTADO), key=lambda p: p.stat().st_mtime) if resultados.exists() else []
    if not hay:
        return [rev('plan_b', AVISO, 'no hay ninguna sesion real grabada para repetir',
                    'Tras el primer ensayo bueno guarda resultados/sesion_real_*_estado.jsonl (es el plan B).')]
    return [rev('plan_b', OK, f'{len(hay)} sesion(es) para repetir; la ultima: {hay[-1].name}')]


def revisar_disco(ruta=None, uso=shutil.disk_usage):
    ruta = Path(ruta or config.RESULTADOS)
    ruta.mkdir(exist_ok=True)
    libre_mb = uso(ruta).free / 1e6
    return [rev('disco', OK if libre_mb >= DISCO_MIN_MB else FALLA, f'{libre_mb:.0f} MB libres para los resultados',
                '' if libre_mb >= DISCO_MIN_MB else 'Libera espacio: la sesion y el EEG crudo ocupan decenas de MB.')]


def preflight(a):
    """Todas las comprobaciones, en el orden en que importan. Devuelve la lista de revisiones."""
    out = revisar_codigo()
    out += revisar_dependencias(a.plan, a.ortesis_sim, a.sin_tablero)
    out += revisar_modelos()
    out += revisar_flujos(a.plan)
    out += revisar_puerto(a.puerto, a.ortesis_sim)
    out += revisar_verificaciones(a.plan, a.ortesis_sim)
    out += revisar_llave()
    out += revisar_plan_b()
    out += revisar_disco()
    return out


RECORDATORIOS = (
    'La laptop en la corriente y sin suspension (powercfg /change standby-timeout-ac 0); no cierres la tapa.',
    'Casco con su dongle (no con el Bluetooth de la laptop); la Unicorn Suite cerrada.',
    'LabRecorder no lo abre este programa: abrelo y selecciona todos los flujos (los del orquestador aparecen al arrancar).',
    'Un ensayo bueno se guarda: sesion_real_*.csv, *_estado.jsonl, *_cuestionario.json, sesion_unicorn_*.csv y el XDF de LabRecorder.',
)


def imprimir_revisiones(revs, salida=print):
    for r in revs:
        salida(f"  [{r['estado']:5s}] {r['clave']}: {r['texto']}")
        if r['que_hacer'] and r['estado'] != OK:
            salida(f"            -> {r['que_hacer']}")
    n = {e: sum(r['estado'] == e for r in revs) for e in (OK, AVISO, FALLA)}
    salida(f"  {n[OK]} OK, {n[AVISO]} aviso(s), {n[FALLA]} falla(s)")
    return n[FALLA] == 0


# ====================================================================== comandos
def comandos(plan, a, extras=(), ahora=time.time):
    """Los comandos de cada proceso como (script, argumentos). Funcion pura: no lanza nada."""
    if plan not in PLANES:
        raise ValueError(f'plan debe ser uno de {PLANES}')
    cmd = {}
    if plan == 'casco':
        puente = ['--placa', 'unicorn']
        if a.serie:
            puente += ['--serie', a.serie]
        puente += ['--grabar', str(config.RESULTADOS / ('sesion_unicorn_' + time.strftime('%Y%m%d_%H%M%S', time.localtime(ahora())) + '.csv'))]
        cmd['fuente'] = ('puente_lsl.py', puente)
    elif plan == 'gemelo':
        cmd['fuente'] = ('cerebro_sintetico.py', [])
    if not a.sin_tablero:
        t = (['--copiloto'] if a.copiloto else []) + (['--narrador'] if a.narrador else []) + (['--flechas'] if a.flechas else [])
        cmd['tablero'] = ('tablero.py', t)
    if a.narrador:
        cmd['narrador'] = ('narrador.py', ['--idioma', a.idioma])
    orq = ['real'] + (['--ortesis-sim'] if a.ortesis_sim else ['--puerto', a.puerto])
    if plan == 'unicornlsl':
        orq += ['--fuente', 'unicornlsl'] + (['--eeg-nombre', a.eeg_nombre] if a.eeg_nombre else [])
    cmd['orquestador'] = ('orquestador.py', orq + list(extras))
    return cmd


def argv_de(script, args):
    """El comando real: pasa por `demo.py hijo`, que instala el cierre limpio y corre el script."""
    return [sys.executable, str(config.RAIZ / 'demo.py'), 'hijo', script, *args]


def hijo(argv):
    """Corre un script del proyecto como si fuera `python script args`, con Ctrl+Break (Windows) o
    SIGINT convertidos en KeyboardInterrupt para que el puente suelte el casco al cerrarse."""
    # un proceso lanzado en segundo plano desde un shell puede heredar SIGINT ignorado: se restituye
    signal.signal(signal.SIGINT, signal.default_int_handler)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    import runpy
    script = str(config.RAIZ / argv[0])
    sys.argv = [script, *argv[1:]]
    runpy.run_path(script, run_name='__main__')


# ====================================================================== procesos
class Procesos:
    """Los procesos de fondo de la demo: cada uno con su log, y se cierran por PID."""

    def __init__(self, carpeta_logs, popen=subprocess.Popen):
        self.carpeta, self.popen, self.hijos = Path(carpeta_logs), popen, {}

    def lanzar(self, nombre, script, args):
        self.carpeta.mkdir(parents=True, exist_ok=True)
        log = open(self.carpeta / f'{nombre}.log', 'w', encoding='utf-8')
        flags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)      # Windows: permite Ctrl+Break por hijo
        p = self.popen(argv_de(script, args), cwd=config.RAIZ, stdout=log, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL, env=dict(os.environ, PYTHONUNBUFFERED='1'), creationflags=flags)
        self.hijos[nombre] = (p, log)
        return p

    def vivo(self, nombre):
        return nombre in self.hijos and self.hijos[nombre][0].poll() is None

    def cola(self, nombre, lineas=12):
        try:
            texto = (self.carpeta / f'{nombre}.log').read_text(encoding='utf-8', errors='replace')
        except OSError:
            return ''
        utiles = [l for l in texto.splitlines() if 'INFO' not in l]
        return '\n'.join(utiles[-lineas:])

    def cerrar_todos(self, espera_s=8.0):
        """Cierre limpio (SIGINT o Ctrl+Break), luego terminate y, si hace falta, kill. Por PID."""
        cerrados = {}
        for nombre, (p, log) in reversed(list(self.hijos.items())):
            if p.poll() is None:
                try:
                    p.send_signal(getattr(signal, 'CTRL_BREAK_EVENT', signal.SIGINT))
                    p.wait(timeout=espera_s)
                    cerrados[nombre] = 'limpio'
                except Exception:
                    try:
                        p.terminate()
                        p.wait(timeout=5)
                        cerrados[nombre] = 'terminate'
                    except Exception:
                        p.kill()
                        cerrados[nombre] = 'kill'
            else:
                cerrados[nombre] = f'ya habia salido ({p.returncode})'
            log.close()
        self.hijos = {}
        return cerrados


def esperar_flujo(nombre, resolver, procesos, proceso, timeout_s, dormir=time.sleep, reloj=time.time):
    """Espera a que aparezca un flujo LSL; si el proceso que lo publica muere antes, lo dice con la cola de su log."""
    t_fin = reloj() + timeout_s
    while reloj() < t_fin:
        if not procesos.vivo(proceso):
            raise ErrorDemo(f'{proceso} se cerro antes de publicar el flujo {nombre}. Ultimas lineas:\n' + procesos.cola(proceso))
        if any(f['name'] == nombre for f in resolver()):
            return True
        dormir(1.0)
    raise ErrorDemo(f'no aparecio el flujo {nombre} en {timeout_s:.0f} s. Ultimas lineas de {proceso}:\n' + procesos.cola(proceso))


# ====================================================================== bitacora
def listar_resultados(carpeta=None):
    carpeta = Path(carpeta or config.RESULTADOS)
    return sorted(p.name for p in carpeta.glob('*') if p.is_file()) if carpeta.exists() else []


def escribir_bitacora(datos, ahora=time.time, carpeta=None):
    carpeta = Path(carpeta or config.RESULTADOS)
    carpeta.mkdir(exist_ok=True)
    base = 'demo_' + time.strftime('%Y%m%d_%H%M%S', time.localtime(ahora()))
    ruta, k = carpeta / (base + '.json'), 1
    while ruta.exists():                          # dos bitacoras en el mismo segundo: no se pisan
        k += 1
        ruta = carpeta / f'{base}_{k}.json'
    ruta.write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding='utf-8')
    return ruta


# ====================================================================== lanzar
def lanzar(a, extras=(), salida=print, procesos=None, resolver=_resolver_lsl, correr_foreground=None,
           hacer_preflight=None, resultados=None, dormir=time.sleep):
    """Preflight, procesos de fondo, orquestador en primer plano, cierre y bitacora. Devuelve el codigo de salida.
    Los argumentos con valor por omision existen para probarlo sin red, sin hardware y sin esperar."""
    t0 = time.time()
    resultados = Path(resultados or config.RESULTADOS)
    salida(f'Modo demo: plan {a.plan}, ' + ('ortesis simulada' if a.ortesis_sim else f'ortesis en {a.puerto}'))
    if a.limpiar_modelos:
        destino, movidos = limpiar_modelos()
        salida(f'  modelos viejos movidos a {destino}: {", ".join(movidos)}' if movidos else '  no habia modelos que mover')
    revs = (hacer_preflight or preflight)(a)
    sin_fallas = imprimir_revisiones(revs, salida)
    bitacora = {'inicio': t0, 'plan': a.plan, 'puerto': None if a.ortesis_sim else a.puerto, 'extras_orquestador': list(extras),
                'preflight': revs, 'ignoro_fallas': bool(a.ignorar_fallas and not sin_fallas)}
    if not sin_fallas and not a.ignorar_fallas:
        salida('Hay fallas: arreglalas y repite, o usa --ignorar-fallas bajo tu responsabilidad (queda en la bitacora).')
        bitacora['salida'] = 'detenido por el preflight'
        salida(f'Bitacora: {escribir_bitacora(bitacora, carpeta=resultados)}')
        return 2
    for r in RECORDATORIOS:
        salida('  * ' + r)
    if a.plan == 'gemelo':
        salida('  * La fuente es el gemelo digital, no una persona: presentalo asi.')
    cmd = comandos(a.plan, a, extras)
    bitacora['comandos'] = {k: [v[0], *v[1]] for k, v in cmd.items()}
    carpeta = resultados / 'logs_demo' / time.strftime('%Y%m%d_%H%M%S', time.localtime(t0))
    procesos = procesos or Procesos(carpeta)
    bitacora['logs'] = str(procesos.carpeta)
    antes = set(listar_resultados(resultados))
    codigo, cierre = 1, {}
    try:
        if 'fuente' in cmd:
            nombre, args = cmd['fuente']
            salida(f'  lanzando {nombre} (log en {procesos.carpeta})')
            procesos.lanzar('fuente', nombre, args)
            esperar_flujo('EEG', resolver, procesos, 'fuente', ESPERA_FLUJO_S[a.plan], dormir=dormir)
            salida('  flujo EEG listo')
        for clave in ('narrador', 'tablero'):
            if clave in cmd:
                procesos.lanzar(clave, *cmd[clave])
                dormir(2.0)
                salida(f'  {clave} lanzado' if procesos.vivo(clave) else
                       f'  AVISO: {clave} se cerro al arrancar; sigo sin el. Ultimas lineas:\n' + procesos.cola(clave))
        salida('  orquestador en esta terminal (Ctrl+C para detener)\n')
        correr_foreground = correr_foreground or _correr_en_primer_plano
        codigo = correr_foreground(argv_de(*cmd['orquestador']))
    except ErrorDemo as e:
        salida(f'ERROR: {e}')
        bitacora['error'] = str(e)
        codigo = 3
    except KeyboardInterrupt:
        salida('Detenido por el operador.')
        codigo = 130
    finally:
        cierre = procesos.cerrar_todos()
        bitacora.update(fin=time.time(), codigo_orquestador=codigo, cierre_de_procesos=cierre,
                        archivos_nuevos=sorted(set(listar_resultados(resultados)) - antes))
        salida(f'Bitacora: {escribir_bitacora(bitacora, carpeta=resultados)}')
    return codigo


def _correr_en_primer_plano(argv):
    p = subprocess.Popen(argv, cwd=config.RAIZ)
    try:
        return p.wait()
    except KeyboardInterrupt:                    # el Ctrl+C tambien le llega al orquestador: que cierre solo
        try:
            return p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.terminate()
            raise


def planb(a, salida=print, resultados=None, dormir=time.sleep, correr_foreground=None, procesos=None):
    """Tablero + repeticion de la ultima sesion real (docs/DOMINGO.md, seccion 6)."""
    hay = revisar_plan_b(resultados)[0]
    if hay['estado'] != OK:
        salida(f"No hay plan B: {hay['texto']}.")
        return 1
    carpeta = Path(resultados or config.RESULTADOS) / 'logs_demo' / time.strftime('%Y%m%d_%H%M%S')
    procesos, codigo = procesos or Procesos(carpeta), 1
    try:
        procesos.lanzar('tablero', 'tablero.py', [])
        dormir(2.0)
        args = ['--ultima', '--velocidad', str(a.velocidad)] + ([] if a.ortesis_sim else ['--puerto', a.puerto])
        salida('Repitiendo la ultima sesion real en el tablero. Es una repeticion: nada se decide en vivo, y hay que decirlo.')
        codigo = (correr_foreground or _correr_en_primer_plano)(argv_de('repetir_sesion.py', args))
    except KeyboardInterrupt:
        codigo = 130
    finally:
        procesos.cerrar_todos()
    return codigo


# ====================================================================== linea de comandos
def argumentos(argv=None):
    # sin abreviaturas: lo que va tras `--` es del orquestador y no debe confundirse con una bandera de aqui
    ap = argparse.ArgumentParser(description='Modo demo automatico de ortesis-bci', allow_abbrev=False)
    sub = ap.add_subparsers(dest='orden', required=True)

    def comunes(p):
        p.add_argument('--plan', choices=PLANES, default='casco',
                       help='casco: puente de BrainFlow; unicornlsl: app UnicornLSL; gemelo: sin casco')
        p.add_argument('--puerto', default=config.PUERTO_ORTESIS)
        p.add_argument('--ortesis-sim', dest='ortesis_sim', action='store_true', help='ortesis simulada')
        p.add_argument('--serie', default='', help='numero de serie del Unicorn (solo con varios cascos cerca)')
        p.add_argument('--eeg-nombre', dest='eeg_nombre', default='', help='nombre del flujo de la app UnicornLSL')
        p.add_argument('--sin-tablero', dest='sin_tablero', action='store_true')
        p.add_argument('--limpiar-modelos', dest='limpiar_modelos', action='store_true',
                       help='mueve modelos/ y estado_sesion.json a modelos/_anteriores/')
    p = sub.add_parser('preflight', help='revisa los requisitos y sale', allow_abbrev=False)
    comunes(p)
    p = sub.add_parser('lanzar', help='preflight y sesion completa', allow_abbrev=False)
    comunes(p)
    p.add_argument('--copiloto', action='store_true')
    p.add_argument('--narrador', action='store_true')
    p.add_argument('--flechas', action='store_true')
    p.add_argument('--idioma', choices=('es', 'en'), default='es')
    p.add_argument('--ignorar-fallas', dest='ignorar_fallas', action='store_true', help='seguir aunque el preflight falle')
    p = sub.add_parser('planb', help='tablero + repeticion de la ultima sesion real', allow_abbrev=False)
    p.add_argument('--velocidad', type=float, default=2.0)
    p.add_argument('--puerto', default=config.PUERTO_ORTESIS)
    p.add_argument('--ortesis-sim', dest='ortesis_sim', action='store_true')
    p = sub.add_parser('hijo', help='uso interno: corre un script del proyecto con cierre limpio')
    p.add_argument('resto', nargs=argparse.REMAINDER)
    a, resto = ap.parse_known_args(argv)
    if a.orden == 'lanzar':
        a.extras = [x for x in resto if x != '--']
    elif resto:
        ap.error('argumentos no reconocidos: ' + ' '.join(resto))
    return a


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'hijo':
        return hijo(argv[1:])
    a = argumentos(argv)
    if a.orden == 'preflight':
        if a.limpiar_modelos:
            destino, movidos = limpiar_modelos()
            print(f'Modelos movidos a {destino}: {", ".join(movidos)}' if movidos else 'No habia modelos que mover.')
        ok = imprimir_revisiones(preflight(a))
        for r in RECORDATORIOS:
            print('  * ' + r)
        return 0 if ok else 2
    if a.orden == 'lanzar':
        return lanzar(a, a.extras)
    if a.orden == 'planb':
        return planb(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
