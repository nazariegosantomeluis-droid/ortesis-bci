# Tarea 1 — Resiliencia: el lazo que no se cae (especificación)

Fecha: 2 de octubre de 2026. Aprobada por Luis en conversación el mismo día.
Demo con casco real: domingo 4 de octubre de 2026.

## Objetivo

Si se desconecta el dongle del Cyton, se reinicia el ESP32, se congela LSL o llega
una época corrupta a media demo, el sistema lo detecta, se protege, se recupera solo
y no pierde la sesión.

## Qué truena hoy (leído en el código)

| Falla | Dónde | Efecto actual |
|---|---|---|
| ACK perdido | `OrtesisSerial.mover` | `TimeoutError`; `main` solo atrapa `KeyboardInterrupt` |
| Puerto caído o línea corrupta | `OrtesisSerial._leer` | el hilo lector muere en silencio (`ser.read`, `int(partes[1])`) |
| Corte de EEG | `EntradaEEG.ventana` | devuelve datos viejos; el decoder decide con una ventana rancia |
| Época con hueco | `EntradaEEG.epoca` | corta por número de muestras, no por tiempo; la época corrupta llega al detector |
| Canal plano o NaN | `DecoderIM.phi` | el recentrado puede envenenar `M` de forma permanente |
| Dongle desconectado | `puente_lsl.py` | la placa no se reconecta; el flujo LSL queda mudo |

## Decisiones aprobadas

1. **Órtesis perdida** → `PAUSA_SEGURA` con motivo `ortesis`: no se puede mover, se
   registra, se avisa y se reconecta sola. No se sigue decodificando (sin movimiento
   no hay ErrP).
2. **Un canal despegado** → EEG en ROJO y pausa. El tablero y la consola dicen **qué
   electrodo falla y por qué** (plano, saturado o ruidoso). Se reanuda solo al volver
   a VERDE.
3. **`puente_lsl.py` reconecta la placa** con retroceso exponencial. Queda etiquetado
   como **no probado con casco real** hasta el domingo.
4. **Calibración:** un ensayo afectado se repite como máximo 3 veces; después se
   sigue con el siguiente ensayo y se avisa.
5. **Medición del caos:** se reporta cuántos pasos se excluyeron y por qué motivo.
6. **Prioridad por tiempo:** contrato y `Vigilante`, blindaje de hardware, pausa
   segura y caos van primero; la persistencia puede quedar al final.

## Componentes

### 1. Contrato (`config.py`)

- `SUBSISTEMAS = ['eeg', 'ortesis', 'reloj', 'detector']`; colores `VERDE`,
  `AMARILLO`, `ROJO`.
- Marcador `m_salud(subsistema, color)` → `salud:<subsistema>:<color>`.
- Estado `PAUSA_SEGURA`. Se llega desde `LAZO_ESTATICO`, `LAZO_ADAPTATIVO`,
  `APRENDIZAJE_CONGELADO` y `PERTURBACION`; sale hacia los tres estados de lazo y
  hacia `EVALUACION`.
- CSV: dos columnas nuevas al final.
  - `salud`: cuatro letras en el orden de `SUBSISTEMAS` (por ejemplo `VVAV`).
  - `excluido`: vacío si el paso es válido; si no, el motivo. Valores:
    `pausa:eeg`, `pausa:canal`, `pausa:ortesis`, `sin_ack`, `epoca_invalida`.
- Umbrales en `SALUD` (valores iniciales; se ajustan con la medición del caos):

| Clave | Valor | Significado |
|---|---|---|
| `eeg_edad_amarillo_s` / `eeg_edad_rojo_s` | 0.3 / 1.0 | edad de la última muestra |
| `eeg_tasa_amarillo` / `eeg_tasa_rojo` | 0.10 / 0.25 | desviación relativa de la tasa real |
| `canal_plano_uv` | 0.1 | desviación estándar mínima en la ventana |
| `canal_saturado_uv` | 180 000 | valor absoluto crudo |
| `canal_ruidoso_uv` | 100 | RMS de 1 a 40 Hz |
| `ventana_canales_s` | 2.0 | ventana para revisar canales |
| `acks_amarillo` / `acks_rojo` | 1 / 3 | ACK perdidos consecutivos |
| `latencia_pico_ms` | 80 | pico de latencia (AMARILLO) |
| `reloj_amarillo_ms` / `reloj_rojo_ms` | 20 / 50 | deriva del retraso contra su línea base |
| `verde_para_reanudar_s` | 3.0 | VERDE continuo para salir de la pausa |

- Otras constantes: `POSICION_SEGURA = 0.0`, `PAUSA_DURACION_MS = 1500`,
  `RECONEXION_INICIAL_S = 0.5`, `RECONEXION_MAX_S = 8.0`,
  `CAL_REPETICIONES_MAX = 3`, `ESTADO_SESION_JSON`, `CAOS_ESTANDAR`.

### 2. `salud.py`

- **`Vigilante`**: clase pura, sin hardware ni LSL. Recibe lecturas y un tiempo `t`:

  ```
  actualizar(t, eeg=None, ortesis=None, reloj_ms=None, detector=None) -> [(subsistema, color), ...]
  ```

  - `eeg = {'edad_s', 'tasa_hz', 'canales': {electrodo: motivo}}`
  - `ortesis = {'puerto_ok', 'acks_perdidos', 'latencia_ms'}`
  - `detector = {'fiabilidad', 'congelado'}`

  Expone `colores`, `detalle` (texto por subsistema, por ejemplo
  `"C3 plano; Fz ruidoso"`), `codigo()` (las cuatro letras), `escalon()` (1 a 4),
  `motivo_pausa()` (`None`, `'eeg'`, `'canal'` u `'ortesis'`) y
  `listo_para_reanudar(t)`.

  Reglas:
  - ROJO entra de inmediato.
  - Para salir de la pausa se piden 3 s continuos con EEG y órtesis en VERDE y el
    reloj fuera de ROJO. El detector no bloquea la salida: durante la pausa no hay
    pasos con los que pueda recuperarse.
  - Reloj en ROJO no pausa: el agente no aprende de ese paso.
  - Detector: ROJO si está congelado, AMARILLO si la fiabilidad está bajo el umbral
    de reanudación, VERDE en otro caso.
  - Detector en `CALENTANDO` (ajuste de Luis, gris en el tablero): mientras el
    `ConfianzaDetector` tenga menos de 15 épocas válidas, el semáforo no opina ni
    emite cambios ni marcadores. Solo afecta al semáforo: la lógica de
    congelamiento del aprendizaje no cambia.
- **`Retroceso(inicial, maximo)`**: esperas 0.5, 1, 2, 4, 8, 8... con `reiniciar()`.
  Lo usan `EntradaEEG`, `OrtesisSerial` y `puente_lsl.py`.

**Qué mide "reloj" (con honestidad):** el retraso entre el reloj local de LSL y la
marca de la última muestra, comparado con su mediana de línea base. Mezcla deriva
de reloj con retardo de transporte; no los separa. Un aumento sostenido de
cualquiera de los dos vuelve sospechosa la base de tiempo de la época, y por eso se
deja de aprender.

**Línea base del reloj tras una reconexión (ajuste de Luis):** cuando el EEG se
reconecta, el flujo es una instancia nueva y su desfase de reloj puede ser otro.
`EntradaEEG` crea la entrada con `recover=False` para enterarse de la pérdida, y en
cada reconexión (y tras cualquier silencio de más de `eeg_edad_rojo_s`) vacía el
buffer y reinicia la línea base del retraso. Mientras la línea base se vuelve a
medir, la deriva vale 0. Así el reloj no queda en ROJO para siempre, la pausa puede
terminar y ninguna ventana ni época mezcla muestras de los dos lados del hueco.

### 3. Blindaje de `hardware.py`

- Funciones puras y probables sin LSL:
  - `revisar_canales(x, fs)` → `{electrodo: 'plano' | 'saturado' | 'ruidoso'}`.
  - `ventana_valida(x, t, fs, ahora, segundos)` → fresca, completa, sin huecos, finita.
  - `cortar_epoca(x, t, t0, fs, ...)` → época o `None` si el tramo tiene un hueco
    de tiempo, valores no finitos o le faltan muestras.
- `EntradaEEG`:
  - si pasan más de `eeg_edad_rojo_s` sin datos, vuelve a resolver el flujo con
    `Retroceso`;
  - `lecturas()` entrega edad, tasa real, canales malos y retraso del reloj;
  - `ventana()` y `epoca()` devuelven `None` cuando los datos no son válidos.
- `OrtesisSerial`:
  - `mover()` no lanza: sin ACK devuelve `(seq, None, nan)` y cuenta el ACK perdido;
  - el lector tolera líneas corruptas;
  - si el puerto cae, lo reabre con `Retroceso` y conserva `seq` (el ESP32 solo hace
    eco del `seq` recibido, así que basta con no reiniciarlo en la PC);
  - `lecturas()` entrega `puerto_ok`, ACK perdidos consecutivos y última latencia.
- `OrtesisSimulada`: misma interfaz; con caos pierde ACK y mete picos de latencia.
- `DecoderIM.phi`: no actualiza el centro con una covarianza no finita o mal
  condicionada.

### 4. Pausa segura y escalera (`orquestador.py`)

Escalera de degradación:

| Escalón | Condición | Comportamiento |
|---|---|---|
| 1 | Todo en VERDE | lazo normal |
| 2 | Detector poco fiable o reloj en ROJO | sigue el lazo, el agente no aprende |
| 3 | EEG perdido o canal despegado | `PAUSA_SEGURA`, órtesis a posición segura |
| 4 | Órtesis perdida | `PAUSA_SEGURA` sin mover: registro y aviso |

- `bloque()` revisa la salud antes de cada paso. Con motivo de pausa:
  1. guarda el estado previo y va a `PAUSA_SEGURA`;
  2. si la órtesis responde, la abre despacio (`POSICION_SEGURA`, `PAUSA_DURACION_MS`);
  3. escribe una fila con `estado=PAUSA_SEGURA` y `excluido=pausa:<motivo>`;
  4. avisa en consola y publica `tipo='salud'` con motivo y detalle (el electrodo y
     por qué);
  5. revisa la salud cada 0.1 s; con motivo `ortesis` manda un movimiento de sondeo
     por segundo a la posición segura;
  6. tras 3 s continuos en VERDE regresa al estado previo y repite el cue del ensayo.
- Las pausas no consumen pasos del bloque. No hay límite de tiempo: se avisa cada
  5 s y `Ctrl+C` lleva a `EVALUACION`.
- Dentro de `paso()`:
  - sin ACK → `excluido=sin_ack`, no se publica `paso_ack`, el agente no aprende;
  - época inválida → `excluido=epoca_invalida`, el agente no aprende;
  - ventana de MI inválida → no se decide: se entra a la pausa.
- `evaluar()` ignora las filas excluidas y reporta el total de excluidos por motivo.
- Calibraciones: el ensayo afectado se repite hasta `CAL_REPETICIONES_MAX` veces;
  después se sigue y se avisa. No hay estado nuevo.
- Interfaz nueva de los backends: `lecturas()`, `posicion_segura()`, `esperar(dt)`,
  `reloj()`, `instantanea()` y `restaurar(d)`. `BackendSim` usa un reloj virtual
  (un ciclo por paso, 0.1 s por tic de pausa) para que las pruebas sean rápidas y
  deterministas.

### 5. Persistencia y `--reanudar`

- Después de cada paso: instantánea atómica en `resultados/estado_sesion.json`
  (archivo temporal, `flush`, `fsync`, `os.replace` con reintentos; un fallo al
  guardar avisa pero nunca tumba el lazo).
- Contenido: versión; argumentos de la sesión; agente (`beta`, varianza, prior,
  sesgo, CUSUM, refractario, contadores de cambios); `ConfianzaDetector`
  (contadores y `congelado`); estado de la máquina y estado previo; progreso del
  bloque (bloque, paso dentro del bloque, meta, orden restante, generador);
  perturbación (`desplazamiento`, `t_perturbacion`, `beta_pre`); ángulo; `seq`;
  número de paso; ruta del CSV; `w0`, `c0` y parámetros del detector; centro `M`
  del decoder; estado del backend; `terminada`.
- `--reanudar`: carga la instantánea, reabre el mismo CSV en modo añadir (si tiene
  una fila de más, lo trunca al paso de la instantánea), restaura todo y continúa.
  En `real` carga los modelos guardados y no recalibra. Si la sesión ya terminó o
  no hay instantánea, sale con un mensaje claro.
- En `sim` se guarda también el estado de los generadores aleatorios: una sesión
  matada y reanudada debe ser idéntica a una sin interrumpir.

### 6. Modo caos

- `caos.py` → `PlanCaos(semilla, tasas=config.CAOS_ESTANDAR)`:
  - fallas por tiempo (`corte_eeg`, `rafaga_parpadeos`, `canal_despegado`):
    `activo(tipo, t)` sobre una línea de tiempo generada por tipo;
  - fallas por paso (`ack_perdido`, `pico_latencia`): `por_paso(tipo, seq)`,
    determinista por (semilla, tipo, `seq`), sin depender del orden de llamadas.
- Caos estándar (valores iniciales):

| Falla | Frecuencia | Parámetros |
|---|---|---|
| Corte de EEG | uno cada ~45 s | 1 a 5 s; los de 3 s o más recrean el flujo |
| ACK perdido | 3 % de los pasos | — |
| Pico de latencia | 5 % de los pasos | 80 a 300 ms |
| Ráfaga de parpadeos | una cada ~40 s | 2 a 4 s, 3 parpadeos por segundo |
| Canal despegado | uno cada ~120 s | 4 a 10 s; plano o ruidoso |

- Consumidores: `cerebro_sintetico.py --caos <semilla>`, `OrtesisSimulada` y
  `BackendSim` (`orquestador.py ... --caos <semilla>`).

### 7. Tablero

- Tres semáforos en la cabecera: EEG, órtesis y detector.
- `PAUSA_SEGURA` en rojo, con el motivo y el detalle (electrodo y causa).
- Atiende el mensaje `tipo='salud'` para actualizarse durante la pausa.

### 8. `puente_lsl.py`

Si la placa deja de entregar datos más de 2 s: libera la sesión y la vuelve a
preparar con `Retroceso`, conservando el flujo LSL. No probado con casco real.

## Pruebas (todas en `pruebas.py`, sin hardware)

| Prueba | Qué verifica |
|---|---|
| `contrato` (ampliada) | `PAUSA_SEGURA`, transiciones, `m_salud`, columnas nuevas |
| `vigilante` | colores por subsistema, detalle del electrodo, VERDE continuo, escalones |
| `retroceso` | secuencia de esperas y reinicio |
| `senal_valida` | `revisar_canales`, `ventana_valida`, `cortar_epoca` con huecos y NaN |
| `ortesis_sin_ack` | `mover()` no lanza y `seq` continúa |
| `reconexion_eeg` | el flujo se recrea con otro desfase de reloj: se reconecta, el reloj no queda en ROJO, se sale de la pausa y ninguna ventana ni época cruza el hueco |
| `plan_caos` | reproducible por semilla y dentro de los rangos |
| `caos_sim` | sesión con caos sin excepción; ninguna época de un corte llega al agente; `beta` no cambia en pausa ni en pasos excluidos |
| `caos_agente_vs_sombra` | varias semillas; error tras perturbar agente < sombra; reporta excluidos por motivo |
| `calibracion_repeticiones` | máximo 3 repeticiones y aviso |
| `reanudar` | proceso matado a mitad; `beta` y paso continúan; en `sim`, CSV idéntico |
| `lazo_real_caos` (`--completa`) | gemelo con `--caos`: termina sin excepción y reporta números |

Las cifras del gemelo y del simulador se reportan como tales, nunca como datos de
una persona.

## Orden de entrega

1. Contrato y `Vigilante`.
2. Blindaje de `hardware.py`, reconexión y `puente_lsl.py`.
3. `PAUSA_SEGURA` y escalera.
4. Caos, pruebas de aceptación y medición.
5. Tablero.
6. Persistencia y `--reanudar`.
7. README y resultados de referencia en `CLAUDE.md`.

## Fuera de alcance

- Fallas durante `IMPEDANCIAS`.
- Resincronizar el ángulo con la telemetría `T` del ESP32 (cada `M` lleva el ángulo
  absoluto, así que el estado se corrige solo en el siguiente paso).
- Seguir operando con menos de 8 canales.

## Cambios durante la implementación

Decididos al medir; cada uno tiene su prueba en `pruebas.py`.

- **Frescura y tasa del EEG por llegada real.** La edad se mide con el reloj de pared de
  la última llegada y la tasa con las muestras que de verdad llegaron (solo cuenta el
  déficit). Las marcas de tiempo suavizadas no sirven para medir ninguna de las dos.
- **Varios flujos `EEG` en la red.** Se usa el más reciente y se avisa; no se salta a
  otro mientras el propio siga publicado; si el propio muere, solo se acepta uno creado
  después (`dos_flujos_eeg`). Se encontró porque había un gemelo olvidado en otra terminal.
- **Renovación de la entrada tras un silencio.** El suavizado de marcas de LSL (dejitter)
  deja las marcas atrasadas tras un hueco (2.7 s tras un hueco de 3 s). En cuanto el flujo
  calla 0.3 s se abre una entrada nueva al mismo flujo (`silencio_sin_recrear`).
- **Deriva del reloj contra una mediana móvil** de ~30 s en lugar de una línea base fija:
  detecta un cambio de desfase y lo absorbe, así el reloj nunca queda en ROJO para siempre
  (`deriva_reloj`). Sustituye al reinicio de línea base como mecanismo principal; el
  reinicio al reconectar se conserva.
- **Detector en `CALENTANDO`** hasta 15 épocas válidas.
- **Canal plano** evaluado sobre el último medio segundo (detección en 0.5 s).
- **Exclusiones.** Un canal despegado o una ráfaga de parpadeos a media época cuenta
  como artefacto, no como paso excluido; solo un corte de EEG da `epoca_invalida`.
- **`Ctrl+C` no termina la sesión:** sigue siendo reanudable; solo una sesión completa
  queda marcada como terminada.
- **Pruebas añadidas fuera del plan:** `ortesis_serial_reconecta`, `puente_reconecta`,
  `dos_flujos_eeg`, `silencio_sin_recrear`, `deriva_reloj`, `instantanea_estado`,
  `tablero_salud`.
