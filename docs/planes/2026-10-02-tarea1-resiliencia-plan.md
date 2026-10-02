# Tarea 1 — Resiliencia: plan de implementación

> **Para quien lo ejecute:** seguir tarea por tarea con superpowers:executing-plans
> (o subagent-driven-development). Los pasos usan casillas (`- [ ]`).

**Meta:** que el lazo detecte fallas de EEG, órtesis, reloj y detector, se proteja en
`PAUSA_SEGURA`, se recupere solo y no pierda la sesión.

**Arquitectura:** un `Vigilante` puro (`salud.py`) recibe lecturas de los backends y
asigna semáforos; el orquestador lo consulta antes de cada paso y decide la pausa.
`hardware.py` deja de lanzar excepciones y valida la señal por tiempo. `caos.py`
genera fallas reproducibles que consumen el gemelo, `OrtesisSimulada` y `BackendSim`.

**Tecnología:** Python 3.11, numpy, scipy, pylsl, pyserial, pyriemann, pyqtgraph.

**Especificación:** `docs/especificaciones/2026-10-02-tarea1-resiliencia-diseno.md`

## Restricciones globales

- Español en nombres, comentarios y mensajes; **sin acentos dentro del código**.
- El contrato vive en `config.py`: ningún nombre de marcador, columna, estado o
  umbral se define en otro archivo.
- `python pruebas.py` pasa antes de cada commit. Cada función nueva lleva su prueba
  en `pruebas.py` con el decorador `@prueba`, sin hardware, y se agrega a la lista
  de `main()`.
- Código propio; ninguna mejora se declara sin medirla; cifras del gemelo o del
  simulador etiquetadas como tales.
- Un commit por tarea. Entorno: `source .venv/Scripts/activate`.
- Para correr una sola prueba: `python -c "import pruebas; pruebas.<nombre>()"`.

## Foco de revisión

Condiciones que la especificación implica y que ninguna prueba cubría; cada una
tiene su prueba en la tarea indicada.

1. **Semáforo de órtesis atascado en AMARILLO durante la pausa** (el último paso
   tuvo un pico de latencia): la pausa debe salir igual → sondeo en la Tarea 3,
   caso en `caos_sim`.
2. **Buffer de EEG vacío al arrancar** (ninguna muestra todavía): `lecturas()` y
   `ventana()` no lanzan → `senal_valida`, Tarea 2.
3. **Falla al escribir la instantánea** (antivirus bloquea `os.replace` en
   Windows): avisa y el lazo sigue → `reanudar`, Tarea 6.
4. **`Ctrl+C` dentro de la pausa:** llega a `EVALUACION` y cierra el CSV →
   `caos_sim`, Tarea 3.
5. **`--reanudar` sin instantánea o con sesión terminada:** mensaje claro y salida
   sin traza → `reanudar`, Tarea 6.

## Archivos

| Archivo | Cambio |
|---|---|
| `config.py` | estado, transiciones, marcador, columnas, `SALUD`, constantes, `CAOS_ESTANDAR` |
| `salud.py` (nuevo) | `Vigilante`, `Retroceso` |
| `caos.py` (nuevo) | `PlanCaos` |
| `hardware.py` | funciones puras de validación, `EntradaEEG`, órtesis, `DecoderIM.phi` |
| `orquestador.py` | pausa, escalera, exclusiones, backends, persistencia, `--caos`, `--reanudar` |
| `cerebro_sintetico.py` | `--caos` |
| `puente_lsl.py` | reconexión de la placa |
| `tablero.py` | semáforos, `PAUSA_SEGURA`, `tipo='salud'` |
| `agente_errp.py` | `a_dict()` / `desde_dict()` en agente y `ConfianzaDetector` |
| `pruebas.py`, `README.md`, `CLAUDE.md` | pruebas y documentación |

---

### Tarea 1: Contrato y `Vigilante`

**Archivos:** modificar `config.py`, `pruebas.py`; crear `salud.py`.

**Produce:**
- `config.SUBSISTEMAS`, `config.VERDE/AMARILLO/ROJO`, `config.m_salud(sub, color)`,
  `config.SALUD`, `config.POSICION_SEGURA`, `config.PAUSA_DURACION_MS`,
  `config.RECONEXION_INICIAL_S`, `config.RECONEXION_MAX_S`,
  `config.CAL_REPETICIONES_MAX`, `config.ESTADO_SESION_JSON`,
  `config.MOTIVOS_EXCLUSION`.
- `salud.Vigilante().actualizar(t, eeg=None, ortesis=None, reloj_ms=None, detector=None) -> list[tuple[str, str]]`
- `Vigilante.colores: dict`, `.detalle: dict`, `.codigo() -> str`, `.escalon() -> int`,
  `.motivo_pausa() -> str | None`, `.listo_para_reanudar(t) -> bool`
- `salud.Retroceso(inicial, maximo).siguiente() -> float`, `.reiniciar()`

- [ ] **Paso 1: pruebas que fallan** (en `pruebas.py`)

```python
@prueba
def vigilante():
    from salud import Vigilante
    V, A, R = config.VERDE, config.AMARILLO, config.ROJO
    bien_eeg = {'edad_s': 0.02, 'tasa_hz': 250.0, 'canales': {}}
    bien_ort = {'puerto_ok': True, 'acks_perdidos': 0, 'latencia_ms': 8.0}
    v = Vigilante()
    assert v.actualizar(0.0, eeg=bien_eeg, ortesis=bien_ort, reloj_ms=0.0,
                        detector={'fiabilidad': 1.0, 'congelado': False}) == []
    assert v.codigo() == 'VVVV' and v.escalon() == 1 and v.motivo_pausa() is None
    # corte de EEG: rojo inmediato, con marcador de cambio
    assert v.actualizar(1.0, eeg={**bien_eeg, 'edad_s': 1.4}) == [('eeg', R)]
    assert v.motivo_pausa() == 'eeg' and v.escalon() == 3
    # vuelve: hacen falta 3 s continuos en verde
    v.actualizar(2.0, eeg=bien_eeg)
    assert not v.listo_para_reanudar(4.9) and v.listo_para_reanudar(5.0)
    # canal despegado: dice cual y por que
    v.actualizar(6.0, eeg={**bien_eeg, 'canales': {'C3': 'plano', 'Fz': 'ruidoso'}})
    assert v.motivo_pausa() == 'canal' and 'C3 plano' in v.detalle['eeg'] and 'Fz ruidoso' in v.detalle['eeg']
    v.actualizar(7.0, eeg=bien_eeg)
    # ortesis: 1 ACK perdido amarillo, 3 rojo, puerto caido rojo
    v.actualizar(8.0, ortesis={**bien_ort, 'acks_perdidos': 1})
    assert v.colores['ortesis'] == A and v.motivo_pausa() is None
    v.actualizar(9.0, ortesis={**bien_ort, 'acks_perdidos': 3})
    assert v.motivo_pausa() == 'ortesis' and v.escalon() == 4
    v.actualizar(10.0, ortesis={**bien_ort, 'latencia_ms': 150.0})
    assert v.colores['ortesis'] == A
    v.actualizar(11.0, ortesis={**bien_ort, 'puerto_ok': False})
    assert v.colores['ortesis'] == R
    v.actualizar(12.0, ortesis=bien_ort)
    # reloj rojo y detector congelado: escalon 2, sin pausa
    v.actualizar(13.0, reloj_ms=60.0)
    assert v.colores['reloj'] == R and v.escalon() == 2 and v.motivo_pausa() is None
    assert not v.listo_para_reanudar(99.0)            # el reloj en rojo bloquea la salida
    v.actualizar(14.0, reloj_ms=5.0, detector={'fiabilidad': 0.2, 'congelado': True})
    assert v.colores['detector'] == R and v.escalon() == 2
    assert v.listo_para_reanudar(17.0)                # el detector no la bloquea
    return 'semaforos, detalle del electrodo, verde continuo y escalones'


@prueba
def retroceso():
    from salud import Retroceso
    r = Retroceso(0.5, 8.0)
    assert [r.siguiente() for _ in range(6)] == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]
    r.reiniciar()
    assert r.siguiente() == 0.5
    return '0.5, 1, 2, 4, 8, 8 y reinicio'
```

Ampliar `contrato()`:

```python
    assert 'PAUSA_SEGURA' in config.ESTADOS
    for e in ('LAZO_ESTATICO', 'LAZO_ADAPTATIVO', 'APRENDIZAJE_CONGELADO', 'PERTURBACION'):
        assert 'PAUSA_SEGURA' in config.TRANSICIONES[e], e
    assert set(config.TRANSICIONES['PAUSA_SEGURA']) == {
        'LAZO_ESTATICO', 'LAZO_ADAPTATIVO', 'APRENDIZAJE_CONGELADO', 'EVALUACION'}
    assert config.m_salud('eeg', config.ROJO) == 'salud:eeg:ROJO'
    assert config.COLUMNAS_CSV[-2:] == ['salud', 'excluido']
```

- [ ] **Paso 2:** `python -c "import pruebas; pruebas.vigilante(); pruebas.retroceso(); pruebas.contrato()"` → tres FALLA.

- [ ] **Paso 3: `config.py`**

```python
# ============================ Salud ============================
SUBSISTEMAS = ['eeg', 'ortesis', 'reloj', 'detector']
VERDE, AMARILLO, ROJO = 'VERDE', 'AMARILLO', 'ROJO'
def m_salud(subsistema, color): return f'salud:{subsistema}:{color}'
SALUD = {
    'eeg_edad_amarillo_s': 0.3, 'eeg_edad_rojo_s': 1.0,
    'eeg_tasa_amarillo': 0.10, 'eeg_tasa_rojo': 0.25,    # desviacion relativa de la tasa
    'canal_plano_uv': 0.1, 'canal_saturado_uv': 180_000.0, 'canal_ruidoso_uv': 100.0,
    'ventana_canales_s': 2.0,
    'acks_amarillo': 1, 'acks_rojo': 3, 'latencia_pico_ms': 80.0,
    'reloj_amarillo_ms': 20.0, 'reloj_rojo_ms': 50.0,
    'detector_amarillo': 0.7,                            # fiabilidad bajo este valor
    'verde_para_reanudar_s': 3.0,
}
POSICION_SEGURA   = 0.0          # abierta
PAUSA_DURACION_MS = 1500         # abrir despacio
RECONEXION_INICIAL_S, RECONEXION_MAX_S = 0.5, 8.0
CAL_REPETICIONES_MAX = 3
MOTIVOS_EXCLUSION = ['pausa:eeg', 'pausa:canal', 'pausa:ortesis', 'sin_ack', 'epoca_invalida']
```

Además: `COLUMNAS_CSV += ['salud', 'excluido']`; `'PAUSA_SEGURA'` en `ESTADOS`
(antes de `EVALUACION`) y en las cuatro listas de `TRANSICIONES`;
`'PAUSA_SEGURA': ['LAZO_ESTATICO', 'LAZO_ADAPTATIVO', 'APRENDIZAJE_CONGELADO', 'EVALUACION']`;
`ESTADO_SESION_JSON = RESULTADOS / 'estado_sesion.json'`.

- [ ] **Paso 4: `salud.py`**

```python
"""Salud del lazo: un semaforo por subsistema y el retroceso de las reconexiones.

Vigilante no toca hardware ni LSL: recibe lecturas y un tiempo, y decide colores.
Asi se prueba completo sin casco.
"""
import config

V, A, R = config.VERDE, config.AMARILLO, config.ROJO


class Retroceso:
    """Esperas de reconexion: inicial, x2, x2... hasta el maximo."""

    def __init__(self, inicial=config.RECONEXION_INICIAL_S, maximo=config.RECONEXION_MAX_S):
        self.inicial, self.maximo = inicial, maximo
        self.reiniciar()

    def reiniciar(self):
        self._espera = self.inicial

    def siguiente(self):
        e = self._espera
        self._espera = min(self.maximo, 2 * e)
        return e


class Vigilante:
    def __init__(self, umbrales=None):
        self.u = umbrales or config.SALUD
        self.colores = {s: V for s in config.SUBSISTEMAS}
        self.detalle = {s: '' for s in config.SUBSISTEMAS}
        self._canal_malo = False
        self._t_verde = None

    def actualizar(self, t, eeg=None, ortesis=None, reloj_ms=None, detector=None):
        """Devuelve los cambios de color [(subsistema, color)] para publicarlos como marcador."""
        previos = dict(self.colores)
        if eeg is not None:
            self._eeg(eeg)
        if ortesis is not None:
            self._ortesis(ortesis)
        if reloj_ms is not None:
            self._poner('reloj', self._nivel(abs(reloj_ms), 'reloj_amarillo_ms', 'reloj_rojo_ms'),
                        f'deriva {reloj_ms:+.0f} ms')
        if detector is not None:
            c = R if detector['congelado'] else (A if detector['fiabilidad'] < self.u['detector_amarillo'] else V)
            self._poner('detector', c, f"fiabilidad {detector['fiabilidad']:.2f}")
        bien = self.colores['eeg'] == V and self.colores['ortesis'] == V and self.colores['reloj'] != R
        if not bien:
            self._t_verde = None
        elif self._t_verde is None:
            self._t_verde = t
        return [(s, c) for s, c in self.colores.items() if c != previos[s]]

    def _nivel(self, x, k_amarillo, k_rojo):
        return R if x >= self.u[k_rojo] else (A if x >= self.u[k_amarillo] else V)

    def _poner(self, sub, color, texto):
        self.colores[sub] = color
        self.detalle[sub] = '' if color == V else texto

    def _eeg(self, e):
        self._canal_malo = False
        if e['canales']:
            self._canal_malo = True
            return self._poner('eeg', R, '; '.join(f'{c} {m}' for c, m in e['canales'].items()))
        c_edad = self._nivel(e['edad_s'], 'eeg_edad_amarillo_s', 'eeg_edad_rojo_s')
        nominal = config.FLUJOS['EEG'][2]
        c_tasa = self._nivel(abs(e['tasa_hz'] - nominal) / nominal, 'eeg_tasa_amarillo', 'eeg_tasa_rojo')
        if c_edad != V:
            return self._poner('eeg', c_edad, f"sin muestras hace {e['edad_s']:.1f} s")
        self._poner('eeg', c_tasa, f"tasa {e['tasa_hz']:.0f} Hz")

    def _ortesis(self, o):
        if not o['puerto_ok']:
            return self._poner('ortesis', R, 'puerto caido')
        if o['acks_perdidos'] >= self.u['acks_rojo']:
            return self._poner('ortesis', R, f"{o['acks_perdidos']} ACK perdidos seguidos")
        if o['acks_perdidos'] >= self.u['acks_amarillo']:
            return self._poner('ortesis', A, 'ACK perdido')
        if o['latencia_ms'] >= self.u['latencia_pico_ms']:
            return self._poner('ortesis', A, f"latencia {o['latencia_ms']:.0f} ms")
        self._poner('ortesis', V, '')

    def codigo(self):
        return ''.join(self.colores[s][0] for s in config.SUBSISTEMAS)

    def motivo_pausa(self):
        if self.colores['eeg'] == R:
            return 'canal' if self._canal_malo else 'eeg'
        return 'ortesis' if self.colores['ortesis'] == R else None

    def escalon(self):
        """1 todo bien, 2 no aprende, 3 EEG perdido, 4 ortesis perdida."""
        if self.colores['ortesis'] == R:
            return 4
        if self.colores['eeg'] == R:
            return 3
        return 2 if R in (self.colores['reloj'], self.colores['detector']) else 1

    def listo_para_reanudar(self, t):
        return self._t_verde is not None and t - self._t_verde >= self.u['verde_para_reanudar_s']
```

- [ ] **Paso 5:** el orquestador escribe las columnas nuevas vacías (`'salud': '', 'excluido': ''`
  en `fila`) para que `orquestador_sim` siga pasando. Agregar `vigilante` y
  `retroceso` a la lista de `main()` de `pruebas.py`.
- [ ] **Paso 6:** `python pruebas.py` → 11/11.
- [ ] **Paso 7:** `git add config.py salud.py orquestador.py pruebas.py && git commit -m "Contrato de salud, estado PAUSA_SEGURA y Vigilante"`

---

### Tarea 2: Blindaje de `hardware.py` y reconexión

**Archivos:** modificar `hardware.py`, `puente_lsl.py`, `pruebas.py`.

**Consume:** `salud.Retroceso`, `config.SALUD`.

**Produce:**
- `hardware.revisar_canales(x, fs) -> dict[str, str]` (`x`: canales × muestras, uV)
- `hardware.ventana_valida(x, t, fs, ahora, segundos) -> bool`
- `hardware.cortar_epoca(x, t, t0, fs, antes, despues, banda) -> np.ndarray | None`
- `EntradaEEG.lecturas() -> {'edad_s', 'tasa_hz', 'canales', 'reloj_ms'}`
- `EntradaEEG.ventana(segundos)` → `(x, t)` o `(None, None)`; `epoca(t0)` → época o `None`
- `_OrtesisBase.lecturas() -> {'puerto_ok', 'acks_perdidos', 'latencia_ms'}`
- `mover(fraccion, dur_ms)` → `(seq, t_ack | None, latencia_ms | nan)`; **nunca lanza**
- `OrtesisSimulada(..., caos=None)` (el `PlanCaos` llega en la Tarea 4)

- [ ] **Paso 1: pruebas que fallan**

```python
@prueba
def senal_valida():
    import hardware as hw
    fs, rng = 250, np.random.default_rng(0)
    x = rng.normal(0, 10, size=(8, 1000)); t = 100 + np.arange(1000) / fs
    assert hw.revisar_canales(x, fs) == {}
    malo = x.copy(); malo[2] = 5.0; malo[7] *= 40; malo[0, 10] = 200_000.0
    assert hw.revisar_canales(malo, fs) == {'FC1': 'saturado', 'C3': 'plano', 'Fz': 'ruidoso'}
    assert hw.ventana_valida(x, t, fs, ahora=t[-1] + 0.05, segundos=3.0)
    assert not hw.ventana_valida(x, t, fs, ahora=t[-1] + 2.0, segundos=3.0)        # rancia
    assert not hw.ventana_valida(x[:, :300], t[:300], fs, t[299] + 0.05, 3.0)      # incompleta
    assert not hw.ventana_valida(np.empty((0,)), np.empty(0), fs, 0.0, 3.0)        # buffer vacio
    con_nan = x.copy(); con_nan[1, 500] = np.nan
    assert not hw.ventana_valida(con_nan, t, fs, t[-1] + 0.05, 3.0)
    t0 = t[600]
    e = hw.cortar_epoca(x, t, t0, fs)
    assert e is not None and e.shape == (8, 250)
    t_hueco = t.copy(); t_hueco[620:] += 1.5                                        # corte dentro de la epoca
    assert hw.cortar_epoca(x, t_hueco, t0, fs) is None
    assert hw.cortar_epoca(con_nan, t, t[450], fs) is None
    assert hw.cortar_epoca(x, t, t[-10], fs) is None                                # faltan muestras
    return 'canales, ventana y epoca validadas por tiempo'


@prueba
def ortesis_sin_ack():
    import hardware as hw

    class PlanFijo:                       # pierde los ACK 2, 3 y 4
        def por_paso(self, tipo, seq):
            return (2 <= seq <= 4) if tipo == 'ack_perdido' else None
    o = hw.OrtesisSimulada(caos=PlanFijo())
    r = [o.mover(0.5) for _ in range(5)]
    assert [x[0] for x in r] == [1, 2, 3, 4, 5]                # seq nunca se reinicia
    assert r[0][1] is not None and r[1][1] is None and np.isnan(r[1][2])
    assert r[4][1] is not None and o.lecturas()['acks_perdidos'] == 0
    o2 = hw.OrtesisSimulada(caos=PlanFijo())
    for _ in range(4):
        o2.mover(0.5)
    assert o2.lecturas() == {'puerto_ok': True, 'acks_perdidos': 3, 'latencia_ms': o2.lecturas()['latencia_ms']}
    return 'mover() no lanza; cuenta ACK perdidos y conserva seq'
```

- [ ] **Paso 2:** correr ambas → FALLA.

- [ ] **Paso 3: funciones puras** (en `hardware.py`, sección Señal)

```python
def revisar_canales(x, fs, u=None):
    """{electrodo: motivo} de los canales que no sirven: 'saturado', 'plano' o 'ruidoso'."""
    u = u or config.SALUD
    malos = {}
    if x.ndim != 2 or x.shape[1] < fs:
        return malos
    finito = np.nan_to_num(x)
    rms = filtrar(finito, (1.0, 40.0), fs).std(axis=1)
    for i in range(x.shape[0]):
        nombre = config.CANALES_EEG[i] if i < len(config.CANALES_EEG) else f'ch{i}'
        if np.abs(finito[i]).max() > u['canal_saturado_uv']:
            malos[nombre] = 'saturado'
        elif finito[i].std() < u['canal_plano_uv']:
            malos[nombre] = 'plano'
        elif rms[i] > u['canal_ruidoso_uv']:
            malos[nombre] = 'ruidoso'
    return malos


def _sin_huecos(t, fs):
    return len(t) < 2 or float(np.diff(t).max()) <= 3.0 / fs


def ventana_valida(x, t, fs, ahora, segundos, u=None):
    """Fresca, completa, sin huecos de tiempo y finita."""
    u = u or config.SALUD
    n = int(segundos * fs)
    if x.ndim != 2 or x.shape[1] < n or len(t) < n:
        return False
    if ahora - t[-1] > u['eeg_edad_rojo_s']:
        return False
    return bool(np.isfinite(x[:, -n:]).all() and _sin_huecos(t[-n:], fs))


def cortar_epoca(x, t, t0, fs, antes=-config.EPOCA_ERRP[0], despues=config.EPOCA_ERRP[1],
                 banda=config.BANDA_ERRP):
    """Epoca filtrada con linea base, o None si el tramo tiene huecos, NaN o le faltan muestras."""
    if x.ndim != 2 or x.shape[1] < fs:
        return None
    n = int(round((antes + despues) * fs))
    i0 = int(np.searchsorted(t, t0 - antes))
    if i0 + n > x.shape[1]:
        return None
    tramo = t[i0:i0 + n]
    if abs(tramo[0] - (t0 - antes)) > 2.0 / fs or not _sin_huecos(tramo, fs):
        return None
    # el filtro de fase cero necesita contexto limpio: 0.5 s antes, si existe y es continuo
    j0 = max(0, i0 - int(0.5 * fs))
    if not _sin_huecos(t[j0:i0 + n], fs):
        j0 = i0
    seg = x[:, j0:i0 + n]
    if not np.isfinite(seg).all():
        return None
    e = filtrar(seg, banda, fs)[:, i0 - j0:]
    return e - e[:, :int(antes * fs)].mean(axis=1, keepdims=True)
```

Nota para quien implemente: la `epoca()` actual filtra una ventana de 4 s. Si con el
contexto de 0.5 s el banco (`python cerebro_sintetico.py --banco` y la prueba
`cerebro_sintetico`) empeora, usar todo el contexto continuo disponible hasta 3 s
antes de la época y volver a medir. No aceptar una BA menor que la actual.

- [ ] **Paso 4: `EntradaEEG`**
  - `__init__`: mover la resolución a `_conectar()`; guardar `self._retroceso = Retroceso()`,
    `self._lag = deque(maxlen=500)`, `self._lag_base = None`, `self.reconexiones = 0`.
  - `_leer`: envolver `pull_chunk` en `try/except Exception` (un flujo perdido no
    mata el hilo). Con datos: guardar el retraso `local_clock() - ts[-1]` y reiniciar
    el retroceso. Sin datos por más de `eeg_edad_rojo_s`: esperar
    `self._retroceso.siguiente()`, volver a resolver `'EEG'` (timeout 1 s) y, si
    aparece, reemplazar `self.inlet`; contar en `self.reconexiones` y avisar por
    consola `[eeg] flujo recuperado`.
  - `lecturas()`:

```python
    def lecturas(self):
        from pylsl import local_clock
        ahora = local_clock()
        u = config.SALUD
        x, t = self._crudo(u['ventana_canales_s'])
        if t.size == 0:
            return {'edad_s': float('inf'), 'tasa_hz': 0.0, 'canales': {}, 'reloj_ms': 0.0}
        recientes = t[t > ahora - u['ventana_canales_s']]
        fresco = ahora - t[-1] <= u['eeg_edad_rojo_s']
        lag = np.array(self._lag)
        if self._lag_base is None and lag.size >= 200:
            self._lag_base = float(np.median(lag))
        deriva = 0.0 if self._lag_base is None else (float(np.median(lag[-50:])) - self._lag_base) * 1000
        return {'edad_s': float(ahora - t[-1]),
                'tasa_hz': recientes.size / u['ventana_canales_s'],
                'canales': revisar_canales(x, self.fs) if fresco else {},
                'reloj_ms': deriva}
```

  - `_crudo(segundos)` es la `ventana()` actual; `ventana(segundos)` devuelve
    `(None, None)` si `ventana_valida` falla; `epoca(t0)` usa `cortar_epoca` sobre
    `_crudo(4.0)` tras `esperar_hasta`.
  - `calidad()` usa `_crudo`.

- [ ] **Paso 4b: línea base del reloj tras reconectar (ajuste de Luis).** La entrada
  se crea con `recover=False`; `pull_chunk` lanza al perderse el flujo y se
  reconecta con `Retroceso`. En cada reconexión, y al volver los datos tras un
  silencio de más de `eeg_edad_rojo_s`, se vacían `_x`, `_t` y `_lag` y
  `_lag_base = None` (deriva 0 hasta volver a medirla con 40 lecturas).
  `EntradaEEG(nombre='EEG')` acepta el nombre del flujo para poder probarla sin
  chocar con un gemelo en marcha. Prueba `reconexion_eeg`: publica un flujo de
  prueba por LSL, lo destruye 1.5 s y lo recrea estampando con +0.25 s de desfase;
  verifica `reconexiones == 1`, que un `Vigilante` alimentado con `lecturas()` pasa
  por ROJO y llega a `listo_para_reanudar`, que el reloj no queda en ROJO, que
  `ventana()` es `None` hasta tener datos nuevos continuos y que `epoca(t0)` es
  `None` para un `t0` dentro del hueco.

- [ ] **Paso 5: órtesis**
  - `_OrtesisBase.__init__`: `self.acks_perdidos = 0`, `self.puerto_ok = True`,
    `self.ultima_latencia = 0.0`; `lecturas()` devuelve los tres.
  - `OrtesisSerial._leer`: `try/except` alrededor del `read` y del análisis de cada
    línea. Si `read` lanza `serial.SerialException` u `OSError`: `puerto_ok = False`,
    cerrar, y reintentar `serial.Serial(...)` con `Retroceso` hasta abrir; al abrir,
    `puerto_ok = True` y aviso `[ortesis] puerto recuperado`. `self.seq` no se toca.
  - `OrtesisSerial.mover`: `try/except` en `write` (falla → `puerto_ok = False`);
    sin ACK: `self.acks_perdidos += 1; return seq, None, float('nan')`; con ACK:
    `self.acks_perdidos = 0`.
  - `OrtesisSimulada(latencia_ms, jitter_ms, semilla, caos=None)`: en `mover`,
    `if caos and caos.por_paso('ack_perdido', seq)` → espera `0.3 s`·0 (no dormir en
    pruebas), cuenta y devuelve `(seq, None, nan)`; `pico = caos.por_paso('pico_latencia', seq)`
    → usa `pico` ms como latencia.
  - `DecoderIM.phi`: actualizar el centro solo si
    `np.isfinite(C).all() and np.linalg.cond(C) < 1e8`.

- [ ] **Paso 6: consumidores** — `BackendReal._ventana_mi` devuelve `None` si la
  ventana no es válida; `BackendReal.mover` y `calibrar_errp` toleran `t_ack is None`
  (por ahora: saltar el ensayo; el paso excluido llega en la Tarea 3).

- [ ] **Paso 7: `puente_lsl.py`** — en el lazo principal, guardar `t_dato`; si
  `time.time() - t_dato > 2.0`: `stop_stream`/`release_session` dentro de
  `try/except`, luego repetir `prepare_session` + `start_stream` con `Retroceso`
  (aviso por intento) hasta que funcione; el `outlet` no se recrea. Comentario en el
  código y en el README: no probado con casco real.

- [ ] **Paso 8:** `python pruebas.py` y `python cerebro_sintetico.py --banco`; la BA
  del detector no baja respecto a la línea base (0.77 en la prueba `cerebro_sintetico`).
- [ ] **Paso 9:** `git commit -m "Blindaje de hardware: senal validada por tiempo, ortesis sin excepciones, reconexion con retroceso"`

---

### Tarea 3: `PAUSA_SEGURA` y escalera de degradación

**Archivos:** modificar `orquestador.py`, `simulador_lazo.py` (nada salvo que haga
falta), `pruebas.py`.

**Consume:** `Vigilante`, `lecturas()` de EEG y órtesis, `mover` sin excepción.

**Produce (interfaz de backend, en `BackendSim` y `BackendReal`):**
- `lecturas() -> {'eeg': dict, 'ortesis': dict, 'reloj_ms': float}`
- `posicion_segura() -> (seq, t_ack | None, lat)` (mueve a `POSICION_SEGURA` en `PAUSA_DURACION_MS`)
- `esperar(dt)`, `reloj() -> float`
- `phi(meta)` devuelve `None` si la ventana no es válida
- `errp(...)` devuelve `(nan, True, 'epoca_invalida')` o `(p, art, '')`
- `Orquestador.revisar_salud() -> str | None`, `Orquestador.pausa_segura(motivo)`

- [ ] **Paso 1: prueba que falla.** Un backend de prueba con fallas programadas a mano
  (el caos por semilla llega en la Tarea 4):

```python
@prueba
def pausa_segura():
    import orquestador
    a = orquestador.argumentos(['sim', '--ciclo', '0', '--pasos_estatico', '20',
                                '--pasos_adaptativo', '60', '--sin_perturbacion'])
    b = orquestador.BackendSim(a)
    # corte de EEG de 4 s en t=60 s, canal C3 plano 5 s en t=100 s, 3 ACK perdidos desde seq 70
    b.fallas = {'corte_eeg': [(60.0, 4.0)], 'canal': [(100.0, 5.0, 'C3', 'plano')],
                'ack_perdido': {70, 71, 72}, 'pico_latencia': {10: 150.0}}
    orq = orquestador.Orquestador(b, a)
    orquestador.correr(orq, a)
    filas = orq.filas
    pausas = [f for f in filas if f['estado'] == 'PAUSA_SEGURA']
    assert [f['excluido'] for f in pausas] == ['pausa:eeg', 'pausa:canal', 'pausa:ortesis'], pausas
    lazo = [f for f in filas if f['estado'] != 'PAUSA_SEGURA']
    assert len(lazo) == 80                                   # las pausas no consumen pasos
    # el agente no aprende en pausa ni en pasos excluidos: beta igual a la fila anterior
    for i, f in enumerate(filas):
        if f['excluido'] and i:
            assert f['beta'] == filas[i - 1]['beta'], (i, f)
    assert {f['excluido'] for f in lazo} <= {'', 'sin_ack', 'epoca_invalida'}
    assert orq.fsm.estado == 'EVALUACION'
    estados = [e for _, e in orq.fsm.historial]
    i = estados.index('PAUSA_SEGURA')
    assert estados[i - 1] == estados[i + 1]                  # regresa al estado previo
    assert any(m.startswith('salud:eeg:ROJO') for m in orq.salidas.marcadores)
    assert 'C3 plano' in ' '.join(orq.avisos_salud)
    return f'{len(pausas)} pausas (eeg, canal, ortesis); el agente no aprende en ellas'
```

- [ ] **Paso 2:** correr → FALLA.

- [ ] **Paso 3: refactor mínimo de `main`.** Extraer el cuerpo del `try` a
  `correr(orq, a)` (incluye el `finally` con `EVALUACION`, `evaluar` y `cerrar`);
  `main` construye y llama. `Salidas.marcador` guarda además el texto en
  `self.marcadores` (lista) para las pruebas.

- [ ] **Paso 4: `BackendSim` con reloj virtual y fallas.**

```python
    # en __init__
    self.t_virtual, self.fallas, self.caos = 0.0, {}, None
    self.acks_perdidos, self.ultima_latencia = 0, 0.0

    def reloj(self):
        return self.t_virtual

    def esperar(self, dt):
        self.t_virtual += dt
        if self.a.ciclo > 0:
            time.sleep(dt)

    def _activa(self, tipo):
        """Falla por tiempo activa ahora: tupla de la falla o None."""
        for f in self.fallas.get(tipo, []):
            if f[0] <= self.t_virtual < f[0] + f[1]:
                return f
        return self.caos.activo(tipo, self.t_virtual) if self.caos else None

    def lecturas(self):
        corte, canal = self._activa('corte_eeg'), self._activa('canal')
        eeg = {'edad_s': self.t_virtual - corte[0] if corte else 0.02, 'tasa_hz': 250.0,
               'canales': {canal[2]: canal[3]} if canal else {}}
        return {'eeg': eeg, 'reloj_ms': 0.0,
                'ortesis': {'puerto_ok': True, 'acks_perdidos': self.acks_perdidos,
                            'latencia_ms': self.ultima_latencia}}
```

  - `phi(meta)`: devuelve `None` si `_activa('corte_eeg')` o `_activa('canal')`
    (**sin** consumir números aleatorios del piloto).
  - `mover(fraccion)`: `seq += 1`; si `seq` está en `fallas['ack_perdido']` (o el
    caos lo dice): `acks_perdidos += 1`, devuelve `(seq, None, nan)`; si no,
    `acks_perdidos = 0`, latencia `fallas['pico_latencia'].get(seq, 0.0)`.
  - `errp(...)`: llama **siempre** a `self.piloto.errp(...)` (consumo fijo de
    aleatorios) y devuelve `(nan, True, 'epoca_invalida')` si un corte está activo
    en `[t_virtual, t_virtual + 0.8]`; registra esos `seq` en
    `self.epocas_en_corte` para la prueba de la Tarea 4.
  - `posicion_segura()`: `return self.mover(config.POSICION_SEGURA)`.
  - `Orquestador.paso` llama `self.b.esperar(config.CICLO_S)`-equivalente: en `sim`
    el reloj virtual avanza `CICLO_S` por paso; la espera real sigue siendo `a.ciclo`.

- [ ] **Paso 5: `BackendReal`.** `lecturas()` combina `self.eeg.lecturas()` (separa
  `reloj_ms`) y `self.ortesis.lecturas()`; `reloj = local_clock`; `esperar = time.sleep`;
  `posicion_segura()` → `self.ortesis.mover(config.POSICION_SEGURA, config.PAUSA_DURACION_MS)`;
  `phi` devuelve `None` si `_ventana_mi()` es `None`; `errp` devuelve el tercer
  elemento `'epoca_invalida'` cuando `epoca()` es `None`.

- [ ] **Paso 6: orquestador.**

```python
    def revisar_salud(self):
        """Lee a los subsistemas, publica los cambios de semaforo y dice si hay que pausar."""
        l = self.b.lecturas()
        cambios = self.vigilante.actualizar(
            self.b.reloj(), eeg=l['eeg'], ortesis=l['ortesis'], reloj_ms=l['reloj_ms'],
            detector={'fiabilidad': self.confianza.fiabilidad_bruta, 'congelado': self.confianza.congelado})
        for sub, color in cambios:
            self.salidas.marcador(config.m_salud(sub, color))
            txt = f'  [salud] {sub}: {color}' + (f' ({self.vigilante.detalle[sub]})' if self.vigilante.detalle[sub] else '')
            aviso(txt)
            self.avisos_salud.append(txt)
        if cambios:
            self.publicar_salud()
        return self.vigilante.motivo_pausa()

    def publicar_salud(self, motivo=None):
        self.salidas.estado(tipo='salud', estado=self.fsm.estado, colores=self.vigilante.colores,
                            detalle=self.vigilante.detalle, motivo=motivo)

    def pausa_segura(self, motivo):
        previo = self.fsm.estado
        self.fsm.ir_a('PAUSA_SEGURA')
        detalle = self.vigilante.detalle['ortesis' if motivo == 'ortesis' else 'eeg']
        aviso(f'  [PAUSA SEGURA] motivo: {motivo} ({detalle}). El agente no aprende; la sesion sigue viva.')
        seq = ''
        if motivo != 'ortesis':
            seq, _, _ = self.b.posicion_segura()
            self.angulo = config.POSICION_SEGURA
        self.registrar_fila(seq=seq, excluido=f'pausa:{motivo}')
        self.publicar_salud(motivo)
        t_aviso = t_sondeo = self.b.reloj()
        while True:
            self.b.esperar(0.1)
            self.revisar_salud()
            t = self.b.reloj()
            if self.vigilante.colores['ortesis'] != config.VERDE and t - t_sondeo >= 1.0:
                self.b.posicion_segura()                  # sondeo: confirma que la ortesis volvio
                self.angulo, t_sondeo = config.POSICION_SEGURA, t
            if self.vigilante.listo_para_reanudar(t):
                break
            if t - t_aviso >= 5.0:
                m = self.vigilante.motivo_pausa() or 'esperando 3 s en verde'
                aviso(f'  [PAUSA SEGURA] sigue: {m} {self.vigilante.detalle}')
                t_aviso = t
        aviso(f'  [PAUSA SEGURA] salud en VERDE {config.SALUD["verde_para_reanudar_s"]:.0f} s: se reanuda {previo}')
        self.fsm.ir_a(previo)
        self.publicar_salud()
```

  - `registrar_fila(**campos)`: arma la fila con todas las columnas del contrato
    (vacías por defecto), `t_iso`, `t_lsl=self.b.reloj()`, `estado`, `angulo`,
    `beta`/`varianza_beta` actuales del agente, `salud=self.vigilante.codigo()`;
    escribe, hace `flush` y la agrega a `self.filas`. `paso()` la usa también.
  - `bloque()`: antes de cada paso

```python
            while True:
                motivo = self.revisar_salud()
                if motivo is None:
                    r = self.paso(meta, aprender)
                    if r != 'sin_ventana':
                        break
                    motivo = self.revisar_salud() or 'eeg'
                self.pausa_segura(motivo)
                self.salidas.marcador(config.CUE_CERRAR if meta > 0 else config.CUE_RELAJA)
                self.salidas.estado(tipo='cue', meta=meta)
                self.b.cue(meta)                              # el ensayo se retoma con su cue
```

  - `paso()`:
    - `phi = self.b.phi(meta)`; si es `None` → `return 'sin_ventana'` (no se decide).
    - tras `mover`: si `t_ack is None` → `excluido = 'sin_ack'`, no se publica
      `paso_ack`, `p_errp, art = nan, True`, `t_ack = self.b.reloj()` para la fila.
    - si no: `p_errp, art, exc = self.b.errp(...)`; `excluido = exc`.
    - `valido = not excluido`; `fiab = self.confianza(erroneo, detectado, valido and not art)`.
    - `aprende = aprender and valido and self.vigilante.colores['reloj'] != config.ROJO`;
      `self.agente.actualizar(p_errp, art or not valido, fiab if aprende else 0.0, ...)`.
    - fila con `salud` y `excluido`; el JSON `tipo='paso'` lleva `salud=self.vigilante.colores`
      y `excluido`.
    - las transiciones a `APRENDIZAJE_CONGELADO` quedan como están.
  - `__init__`: `self.vigilante = Vigilante()`, `self.avisos_salud = []`.
  - `evaluar()`: trabajar sobre `validas = [f for f in self.filas if not f['excluido']]`;
    los índices de la perturbación se recalculan sobre `validas` (guardar
    `self.seq_perturbacion` y buscarlo). Línea nueva:
    `excluidos: N (pausa:eeg a, pausa:canal b, pausa:ortesis c, sin_ack d, epoca_invalida e)`.
    Guardar el desglose en `self.excluidos` (dict) para las pruebas.
  - `correr()`: `KeyboardInterrupt` dentro de la pausa pasa por el mismo `finally`
    (`PAUSA_SEGURA → EVALUACION` está permitida).

- [ ] **Paso 7: calibraciones** (`BackendReal`). Función auxiliar:

```python
    def _ensayo_con_reintentos(self, tomar, que):
        """tomar() devuelve el dato o None si el EEG o la ortesis fallaron. Maximo
        config.CAL_REPETICIONES_MAX repeticiones; despues se sigue y se avisa."""
        for k in range(config.CAL_REPETICIONES_MAX + 1):
            dato = tomar()
            if dato is not None:
                return dato
            if k < config.CAL_REPETICIONES_MAX:
                aviso(f'    {que} afectado por una falla: se repite ({k + 1}/{config.CAL_REPETICIONES_MAX})')
        aviso(f'    AVISO: {que} descartado tras {config.CAL_REPETICIONES_MAX} repeticiones; se sigue')
        return None
```

  `calibrar_mi` y `calibrar_errp` meten el cue, la espera y la toma del dato dentro
  de `tomar()`; un `None` final no agrega ensayo. Prueba:

```python
@prueba
def calibracion_repeticiones():
    import orquestador
    b = orquestador.BackendReal.__new__(orquestador.BackendReal)   # sin hardware
    intentos = []
    assert b._ensayo_con_reintentos(lambda: intentos.append(1), 'ensayo') is None
    assert len(intentos) == 1 + config.CAL_REPETICIONES_MAX
    cuenta = iter([None, None, 'dato'])
    assert b._ensayo_con_reintentos(lambda: next(cuenta), 'ensayo') == 'dato'
    return f'maximo {config.CAL_REPETICIONES_MAX} repeticiones y aviso'
```

- [ ] **Paso 8: caso de `Ctrl+C` en pausa** (Foco de revisión 4), dentro de
  `pausa_segura` de pruebas: backend con un corte de EEG infinito cuyo `esperar`
  lanza `KeyboardInterrupt` al tercer tic; `correr()` termina con
  `orq.fsm.estado == 'EVALUACION'` y `orq.f_csv.closed`.
- [ ] **Paso 9:** `python pruebas.py` → todas; `python simulador_lazo.py --sin_grafica`
  da los mismos números de referencia (0.325 / 0.243 / 0.211): el agente no cambió.
- [ ] **Paso 10:** `git commit -m "PAUSA_SEGURA y escalera de degradacion en el orquestador"`

---

### Tarea 4: Modo caos, aceptación y medición

**Archivos:** crear `caos.py`; modificar `config.py`, `cerebro_sintetico.py`,
`orquestador.py`, `pruebas.py`.

**Produce:**
- `config.CAOS_ESTANDAR`
- `caos.PlanCaos(semilla, tasas=None)`
  - `.activo(tipo, t) -> tuple | None` — `('corte_eeg')`: `(t0, dur)`;
    `'rafaga_parpadeos'`: `(t0, dur)`; `'canal'`: `(t0, dur, electrodo, modo)`
  - `.por_paso(tipo, seq)` — `'ack_perdido'` → `bool`; `'pico_latencia'` → `float | None` (ms)
- `--caos <semilla>` en `orquestador.py` y `cerebro_sintetico.py`

- [ ] **Paso 1: `config.CAOS_ESTANDAR`**

```python
CAOS_ESTANDAR = {
    'corte_eeg':        {'cada_s': 45.0,  'duracion_s': (1.0, 5.0), 'recrear_desde_s': 3.0},
    'rafaga_parpadeos': {'cada_s': 40.0,  'duracion_s': (2.0, 4.0), 'por_segundo': 3.0},
    'canal':            {'cada_s': 120.0, 'duracion_s': (4.0, 10.0)},
    'ack_perdido':      {'p': 0.03},
    'pico_latencia':    {'p': 0.05, 'ms': (80.0, 300.0)},
}
```

- [ ] **Paso 2: pruebas que fallan**

```python
@prueba
def plan_caos():
    from caos import PlanCaos
    a, b, c = PlanCaos(7), PlanCaos(7), PlanCaos(8)
    ts = np.arange(0, 600, 0.1)
    for tipo in ('corte_eeg', 'rafaga_parpadeos', 'canal'):
        ea = [a.activo(tipo, t) for t in ts]
        assert ea == [b.activo(tipo, t) for t in ts[::-1]][::-1], tipo      # no depende del orden
        ev = {e for e in ea if e}
        assert 2 <= len(ev) <= 40, (tipo, len(ev))
        lo, hi = config.CAOS_ESTANDAR[tipo]['duracion_s']
        assert all(lo <= e[1] <= hi for e in ev), tipo
    assert {e[3] for e in (a.activo('canal', t) for t in ts) if e} <= {'plano', 'ruidoso'}
    perdidos = [a.por_paso('ack_perdido', s) for s in range(2000)]
    assert perdidos == [b.por_paso('ack_perdido', s) for s in range(2000)]
    assert 0.015 < np.mean(perdidos) < 0.05
    picos = [p for p in (a.por_paso('pico_latencia', s) for s in range(2000)) if p]
    assert all(80 <= p <= 300 for p in picos) and 0.03 < len(picos) / 2000 < 0.08
    assert [c.activo('corte_eeg', t) for t in ts] != [a.activo('corte_eeg', t) for t in ts]
    return 'reproducible por semilla y dentro de los rangos'


def _sesion_caos(semilla, caos):
    import orquestador
    argv = ['sim', '--ciclo', '0', '--semilla', str(semilla), '--pasos_estatico', '60',
            '--pasos_adaptativo', '300'] + (['--caos', str(caos)] if caos is not None else [])
    a = orquestador.argumentos(argv)
    orq = orquestador.Orquestador(orquestador.BackendSim(a), a)
    orquestador.correr(orq, a)
    return orq


@prueba
def caos_sim():
    orq = _sesion_caos(0, caos=1)
    assert orq.fsm.estado == 'EVALUACION'
    por_seq = {f['seq']: f for f in orq.filas if f['seq'] != ''}
    assert orq.b.epocas_en_corte, 'el caos estandar debe producir al menos una epoca en un corte'
    for seq in orq.b.epocas_en_corte:                       # ninguna llega al agente
        assert por_seq[seq]['excluido'] == 'epoca_invalida' and por_seq[seq]['P_hat'] == '', seq
    for i, f in enumerate(orq.filas[1:], 1):                # no aprende en pausa ni excluidos
        if f['excluido']:
            assert f['beta'] == orq.filas[i - 1]['beta'], i
    assert sum(1 for f in orq.filas if f['estado'] != 'PAUSA_SEGURA') == 360
    return f'termina sin excepcion; excluidos {orq.excluidos}'


@prueba
def caos_agente_vs_sombra():
    """Exploratorio, sobre el simulador: con el caos estandar el agente sigue ganandole a la sombra."""
    ag, so, exc = [], [], {}
    for s in range(12):
        orq = _sesion_caos(s, caos=100 + s)
        ag.append(orq.error_post['agente']); so.append(orq.error_post['sombra'])
        for k, v in orq.excluidos.items():
            exc[k] = exc.get(k, 0) + v
    assert np.mean(ag) < np.mean(so) - 0.05, (np.mean(ag), np.mean(so))
    return (f'simulador, 12 sujetos, tras perturbar: agente {np.mean(ag):.2f} vs sombra {np.mean(so):.2f}; '
            f'excluidos por motivo {exc}')
```

  `evaluar()` guarda `self.error_post = {'agente': ..., 'sombra': ...}` (los mismos
  números que imprime el CP4, sobre filas válidas).

- [ ] **Paso 3: `caos.py`**

```python
"""Ingenieria del caos para el lazo: fallas reproducibles por semilla.

Las fallas por tiempo viven en una linea de tiempo por tipo (independiente de
cuando se consulte); las fallas por paso dependen solo de (semilla, tipo, seq).
"""
import numpy as np

import config

_TIPOS = ['corte_eeg', 'rafaga_parpadeos', 'canal', 'ack_perdido', 'pico_latencia']


class PlanCaos:
    def __init__(self, semilla, tasas=None):
        self.semilla, self.tasas = int(semilla), tasas or config.CAOS_ESTANDAR
        self._lineas = {}

    def _rng(self, tipo, *extra):
        return np.random.default_rng([self.semilla, _TIPOS.index(tipo), *extra])

    def _linea(self, tipo, hasta):
        """Eventos (t0, dur, ...) del tipo; se extiende por tramos sin cambiar lo ya generado."""
        l = self._lineas.setdefault(tipo, {'rng': self._rng(tipo), 't': 10.0, 'ev': []})
        c = self.tasas[tipo]
        while l['t'] <= hasta:
            t0 = l['t'] + l['rng'].exponential(c['cada_s'])
            dur = float(l['rng'].uniform(*c['duracion_s']))
            ev = (float(t0), dur)
            if tipo == 'canal':
                ev += (config.CANALES_EEG[int(l['rng'].integers(len(config.CANALES_EEG)))],
                       ['plano', 'ruidoso'][int(l['rng'].integers(2))])
            l['ev'].append(ev)
            l['t'] = t0 + dur + 5.0              # al menos 5 s entre fallas del mismo tipo
        return l['ev']

    def activo(self, tipo, t):
        for ev in self._linea(tipo, t):
            if ev[0] <= t < ev[0] + ev[1]:
                return ev
        return None

    def por_paso(self, tipo, seq):
        r, c = self._rng(tipo, int(seq)), self.tasas[tipo]
        if tipo == 'ack_perdido':
            return bool(r.random() < c['p'])
        return float(r.uniform(*c['ms'])) if r.random() < c['p'] else None
```

- [ ] **Paso 4: consumidores.**
  - `orquestador.argumentos`: `--caos` (`type=int`, por defecto `None`).
    `BackendSim`: `self.caos = PlanCaos(a.caos) if a.caos is not None else None`;
    en `errp`, con `rafaga_parpadeos` activa fuerza `art = True`. `BackendReal`:
    `OrtesisSimulada(caos=PlanCaos(a.caos))` si hay `--caos` y `--ortesis-sim`.
  - `cerebro_sintetico.py --caos <semilla>`, con tiempo `t` relativo al arranque:
    - `corte_eeg` activo: no publica; al terminar hace `t_ult = local_clock()` (no
      rellena el hueco); si `dur >= recrear_desde_s`, destruye y recrea el `outlet`.
    - `rafaga_parpadeos` activa: `Cerebro.generar` usa `por_segundo` en lugar de
      `a.parpadeos`.
    - `canal` activo: tras la mezcla, ese canal se reemplaza por `0.0` (plano) o por
      `300 * rng.normal` + 60 Hz de 200 uV (ruidoso).
    - imprime cada falla al empezar: `[caos] corte de EEG 3.2 s`.
- [ ] **Paso 5:** `python pruebas.py`. Si `caos_agente_vs_sombra` no cumple el margen,
  **no** se relaja el umbral sin entender por qué: revisar qué motivo de exclusión
  domina y reportarlo a Luis.
- [ ] **Paso 6: corrida real contra el gemelo** (prueba `lazo_real_caos`, solo con
  `--completa`, después de `lazo_real_sintetico` para reutilizar sus modelos):
  `cerebro_sintetico.py --caos 1` + `orquestador.py real --ortesis-sim --forzar
  --saltar-calibracion --caos 1 --pasos_estatico 30 --pasos_adaptativo 120`.
  Afirma: código de salida 0, `EVALUACION` y la línea `excluidos:` en la salida, y
  al menos una `PAUSA SEGURA`. Devuelve la línea del CP4 y la de excluidos.
- [ ] **Paso 7:** `python pruebas.py --completa`; anotar los números (agente contra
  sombra y excluidos por motivo, del simulador y del gemelo) para el README.
- [ ] **Paso 8:** `git commit -m "Modo caos reproducible, pruebas de aceptacion y medicion"`

---

### Tarea 5: Tablero

**Archivos:** modificar `tablero.py`.

**Consume:** JSON `tipo='salud'` (`estado`, `colores`, `detalle`, `motivo`) y el campo
`salud` de `tipo='paso'`.

- [ ] **Paso 1:** en la cabecera, tres `QLabel` (`EEG`, `ORTESIS`, `DETECTOR`) con
  fondo según color: `{'VERDE': '#2ca02c', 'AMARILLO': '#e6b800', 'ROJO': '#d62728'}`,
  texto blanco, `border-radius:8px; padding:4px 10px`. Método
  `_semaforos(colores, detalle)`; el `toolTip` de cada uno es su detalle.
- [ ] **Paso 2:** `_procesar`: con `tipo == 'salud'` actualiza semáforos; si
  `e['estado'] == 'PAUSA_SEGURA'`, `lbl_estado` en rojo con
  `PAUSA SEGURA · <motivo> · <detalle>` (por ejemplo `PAUSA SEGURA · canal · C3 plano`).
  Con `tipo == 'paso'` llama a `_semaforos(e['salud'], {})` si el campo existe
  (compatibilidad con sesiones grabadas sin él).
- [ ] **Paso 3:** verificación manual: `python tablero.py --captura resultados/tablero_caos.png --segundos 40 &`
  y `python orquestador.py sim --caos 1`; abrir la imagen y confirmar semáforos y
  pausa en rojo.
- [ ] **Paso 4:** `python pruebas.py`; `git commit -m "Tablero: semaforos de salud y PAUSA_SEGURA en rojo"`

---

### Tarea 6: Persistencia y `--reanudar` (última por prioridad)

**Archivos:** modificar `agente_errp.py`, `orquestador.py`, `hardware.py`, `pruebas.py`.

**Produce:**
- `AgenteErrP.a_dict() -> dict`, `.desde_dict(d)`; lo mismo en `ConfianzaDetector`
- `orquestador.guardar_instantanea(ruta, datos) -> bool`
- `Orquestador.instantanea() -> dict`, `Orquestador.restaurar(d)`
- `BackendSim.instantanea()/restaurar(d)`, `BackendReal.instantanea()/restaurar(d)`
- `--reanudar`

- [ ] **Paso 1: prueba que falla**

```python
@prueba
def reanudar():
    import json, orquestador
    base = ['sim', '--ciclo', '0', '--semilla', '4', '--pasos_estatico', '20', '--pasos_adaptativo', '90']
    # referencia: sesion sin interrumpir
    a = orquestador.argumentos(base)
    ref = orquestador.Orquestador(orquestador.BackendSim(a), a)
    orquestador.correr(ref, a)
    # proceso matado a mitad de sesion
    p = subprocess.Popen([sys.executable, 'orquestador.py'] + base[:1] + ['--ciclo', '0.05'] + base[3:],
                         cwd=config.RAIZ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(4.0)
    p.kill(); p.wait()
    inst = json.loads(config.ESTADO_SESION_JSON.read_text())
    assert 10 < inst['paso'] < 110 and not inst['terminada']
    r = subprocess.run([sys.executable, 'orquestador.py', 'sim', '--reanudar', '--ciclo', '0'],
                       cwd=config.RAIZ, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    filas = list(csv.DictReader(open(inst['ruta_csv'])))
    assert len(filas) == 110
    assert float(filas[inst['paso'] - 1]['beta']) == round(inst['agente']['beta'], 4)
    quitar = lambda f: {k: v for k, v in f.items() if k not in ('t_iso', 't_lsl')}
    assert [quitar(f) for f in filas] == [quitar({k: str(v) for k, v in f.items()}) for f in ref.filas]
    # sesion ya terminada o sin instantanea: mensaje claro, sin traza
    r2 = subprocess.run([sys.executable, 'orquestador.py', 'sim', '--reanudar'],
                        cwd=config.RAIZ, capture_output=True, text=True, timeout=60)
    assert r2.returncode != 0 and 'Traceback' not in r2.stderr and 'ya termino' in (r2.stdout + r2.stderr)
    # un fallo al guardar no tumba el lazo
    assert orquestador.guardar_instantanea(config.RESULTADOS / 'no_existe' / 'x.json', {}) is False
    return f"matado en el paso {inst['paso']}; reanudado identico a la sesion sin interrumpir"
```

- [ ] **Paso 2:** correr → FALLA.
- [ ] **Paso 3: `agente_errp.py`**

```python
    # AgenteErrP
    def a_dict(self):
        return {'beta': self.beta, 'var': self.var, 'prior': self.prior, 'sesgo': self.sesgo,
                'desde_cambio': self._desde_cambio, 'cusum_pred': self.cusum_pred,
                'n_cambios': dict(self.n_cambios)}

    def desde_dict(self, d):
        self.beta, self.var, self.prior, self.sesgo = d['beta'], d['var'], d['prior'], d['sesgo']
        self._desde_cambio, self.cusum_pred = d['desde_cambio'], d['cusum_pred']
        self.n_cambios, self._pendiente = dict(d['n_cambios']), None

    # ConfianzaDetector
    def a_dict(self):
        return {k: getattr(self, k) for k in ('err_det', 'err_tot', 'ok_nodet', 'ok_tot', 'congelado')}

    def desde_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)
```

- [ ] **Paso 4: escritura atómica**

```python
def guardar_instantanea(ruta, datos):
    """Archivo temporal + os.replace. Nunca lanza: un fallo al guardar no tumba el lazo."""
    tmp = ruta.with_suffix('.tmp')
    try:
        with open(tmp, 'w') as f:
            json.dump(datos, f, default=float)
            f.flush()
            os.fsync(f.fileno())
        for intento in range(5):                 # en Windows otro proceso puede tener el archivo
            try:
                os.replace(tmp, ruta)
                return True
            except PermissionError:
                time.sleep(0.01 * (intento + 1))
    except OSError:
        pass
    aviso(f'  AVISO: no se pudo guardar la instantanea en {ruta}')
    return False
```

- [ ] **Paso 5: `Orquestador`.**
  - `bloque()` pasa a guardar su progreso en `self.prog = {'bloque', 't', 'meta',
    'orden', 'rng'}` (`rng` = `rng.bit_generator.state`) para poder retomarlo en
    medio de un ensayo.
  - `instantanea()`: `version: 1`, `terminada`, `args` (los de `argumentos()` que
    definen la sesión), `paso = len(self.filas)`, `ruta_csv`, `estado` y
    `estado_previo`, `angulo`, `desplazamiento`, `seq_perturbacion`, `beta_pre`,
    `prog`, `agente`, `confianza`, `umbral_errp`, `preparacion` (el dict `p` con
    `w0` y `c0` como listas), `backend = self.b.instantanea()`.
  - Se guarda al final de cada `paso()`, al salir de cada pausa y, con
    `terminada=True`, al final de `correr()`.
  - `BackendSim.instantanea()`: `seq`, `t`, `t_virtual`, `acks_perdidos`, estado del
    `rng` del piloto, `piloto.desplazamiento`, `sens`/`espec` del piloto.
  - `BackendReal.instantanea()`: `seq` de la órtesis y `M` del decoder
    (`self.decoder.M.tolist()`); `restaurar` carga los `.pkl`, repone `M` y `seq`.
  - `--reanudar`: lee la instantánea; si no existe o `terminada`, imprime
    `No hay sesion que reanudar` / `La sesion ya termino` y sale con código 2. Si
    existe: reconstruye `a` desde `args` (los de línea de comandos `--ciclo` y
    `--puerto` mandan), abre el CSV, lo trunca a `paso` filas, lo reabre en modo
    añadir, recarga `self.filas`, construye agente y confianza desde `preparacion`
    y aplica `desde_dict`; la máquina de estados arranca en `estado` (parámetro
    nuevo `MaquinaEstados(salidas, estado=None)`; si era `PAUSA_SEGURA`, en
    `estado_previo`). `correr()` salta lo ya hecho según `prog`.
  - La perturbación del simulador vive en `piloto.desplazamiento` y en
    `self.desplazamiento`: ambas se restauran.
- [ ] **Paso 6:** `python pruebas.py`; `git commit -m "Persistencia atomica de la sesion y --reanudar"`

---

### Tarea 7: Documentación

**Archivos:** `README.md`, `CLAUDE.md`, `TAREAS.md`.

- [ ] **Paso 1: README.** Sección nueva "Resiliencia" con: tabla de semáforos, la
  escalera de cuatro escalones, `PAUSA_SEGURA`, `--caos`, `--reanudar`, tabla de
  motivos de exclusión, y los números medidos en la Tarea 4 **etiquetados como
  simulador o gemelo**. Nota explícita: reconexión del Cyton no probada con casco
  real. Actualizar `9/9`, la tabla de archivos (`salud.py`, `caos.py`) y los
  comandos del día de la demo.
- [ ] **Paso 2: `CLAUDE.md`.** Filas de `salud.py` y `caos.py` en Arquitectura;
  comandos con `--caos` y `--reanudar`; línea nueva en "Resultados de referencia"
  con el caos estándar.
- [ ] **Paso 3: `TAREAS.md`.** Marcar la Tarea 1 como hecha con la fecha y lo que
  quedó pendiente, si algo quedó.
- [ ] **Paso 4:** `python pruebas.py --completa`; `git commit -m "Documentacion de resiliencia y resultados del caos"`

---

## Cobertura de la especificación

| Sección de la especificación | Tarea |
|---|---|
| Contrato, `Vigilante`, `Retroceso` | 1 |
| Blindaje de hardware, reconexión, `puente_lsl.py` | 2 |
| Pausa segura, escalera, exclusiones, calibración con 3 repeticiones | 3 |
| Caos, aceptación, reporte de excluidos por motivo | 4 |
| Tablero con electrodo y motivo | 3 (consola) y 5 |
| Persistencia y `--reanudar` | 6 |
| Documentación | 7 |
