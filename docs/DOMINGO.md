# Domingo 4 de octubre: guion de la demo

Todo lo de aquí se probó con el gemelo digital; **nada se ha probado con el casco real**. Si algo no coincide con lo que ves, confía en lo que dice la pantalla y usa el árbol de decisión de la sección 5.

Los comandos se corren en Git Bash desde la carpeta del proyecto, con el entorno activado:

```bash
cd ~/Downloads/ortesis-bci
source .venv/Scripts/activate
```

## 1. Antes de salir de casa (10 minutos)

1. **Que la laptop no se duerma.** Conéctala a la corriente y corre esto una vez (PowerShell o CMD):
   ```
   powercfg /change standby-timeout-ac 0
   powercfg /change monitor-timeout-ac 0
   ```
   No cierres la tapa durante la demo: la suspensión congela la sesión.
2. **Código de la demo.** `git checkout main && git pull` y comprueba que estás en la etiqueta `v-demo` (`git describe --tags`).
3. **Humo.** `python pruebas.py` debe terminar con todas las pruebas en verde (unos 3 minutos).
4. **Quita los modelos viejos.** En `modelos/` quedan los del gemelo de las pruebas. Quítalos para que nadie los cargue por error, **sin llevarte `decoder_preentrenado.pkl`** (no está en el repositorio y regenerarlo exige bajar unos 350 MB de PhysioNet):
   ```bash
   python demo.py preflight --limpiar-modelos      # los mueve a modelos/_anteriores/ y deja el preentrenado
   ```
   A mano, el equivalente que borra de verdad es `find modelos -maxdepth 1 \( -name '*.pkl' -o -name '*.npz' \) ! -name decoder_preentrenado.pkl -delete` y `rm -f resultados/estado_sesion.json`. Un `rm -f modelos/*.pkl` a secas **sí** borra el preentrenado.
5. **Lleva:** el casco con **su dongle**, la órtesis con su cable USB, gel o solución para los electrodos, y cargadores.

## 2. Al llegar: verificar el casco (5 minutos)

Empareja el casco con su dongle (no con el Bluetooth de la laptop) y cierra la Unicorn Suite: solo una aplicación puede conectarse al casco a la vez.

```bash
python verificar_unicorn.py brainflow
```

Sigue la guía de unos 50 segundos (quieto, parpadear, ojos cerrados, mover la cabeza). Al final imprime un veredicto, que también queda en `resultados/verificacion_unicorn.json`:

| Veredicto | Qué hacer |
|---|---|
| `usa BrainFlow (fuente principal)` | Sigue con la sección 4 tal cual. |
| `NO uses todavia ninguna...` con `brainflow` | Abre la app UnicornLSL, conecta el casco, pulsa Start y corre `python verificar_unicorn.py lsl`. |
| `usa la app UnicornLSL (respaldo)` | En la sección 4 no lances el puente; el orquestador lleva `--fuente unicornlsl`. |
| Falla con las dos fuentes | Lee qué comprobación falla (orden de canales, unidades, contador). No sigas con el casco: ve al plan B (sección 6) y avisa a Luis. |

Si la comprobación dice que el orden de los canales o las unidades no son los esperados, **no calibres**: el contrato de `config.py` supone Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 en microvolts.

**Prueba de toques (descarta canales intercambiados).** Los parpadeos y el alfa solo *sugieren* el orden de los canales, y un electrodo malo los confunde. Los toques lo comprueban de frente: con quien lleva el casco quieto, otra persona toca cada electrodo 3 s con la yema del dedo (golpecitos seguidos) cuando la pantalla lo pide, y el pico debe salir en ese canal:

```bash
python verificar_unicorn.py brainflow --solo-toques     # ~45 s: un reposo corto y los 8 toques
python verificar_unicorn.py brainflow --toques          # la guía completa y, al final, los toques
```

Imprime una matriz (fila = electrodo tocado, columna = canal que respondió, en veces su reposo): lo correcto es el máximo de cada fila en la diagonal. La izquierda y la derecha son las de quien lleva el casco (C3 es la izquierda). Si dos electrodos responden cada uno donde toca el otro (por ejemplo C3 y C4), dice `electrodos intercambiados` y el veredicto pasa a `NO uses`: hay que corregir el orden de los canales en `puente_lsl.plan_placa` o `config.FUENTES_EEG`. Un electrodo que no se toca, o que responde poco, queda en AVISO y no se da por bueno ni por malo.

## 3. Verificar la órtesis (2 minutos)

```bash
python verificar_ortesis.py --puerto COM4
```

Mueve la órtesis 30 veces y mide cuánto tarda en empezar a moverse tras el ACK.

| Resultado | Qué significa | Qué hacer |
|---|---|---|
| `OK: latencia mecanica mediana ...` | Hay telemetría: la época del ErrP se alinea al movimiento real. | Sigue. |
| `AVISO: el inicio se vio en menos del 80 %` | Telemetría intermitente: se usará el ACK más la latencia media. | Sigue. Cuesta algo (en el gemelo se recuperan 12 de 16 sesiones en lugar de las 16), pero el lazo funciona. |
| `FALLA: no se vio el inicio del movimiento`, y la órtesis **sí se movió** | El firmware no manda la telemetría `T`. | Sigue si no hay tiempo de corregirlo: la época se corta en el ACK. El detector será más débil (en el gemelo, BA 0.72 en lugar de 0.86, y se recuperan 10 de 16 sesiones). |
| `FALLA`, y la órtesis **no se movió** | No hay ACK: otro programa usa el COM, o el número cambió. | Cierra el monitor serie de Arduino; mira el número en el Administrador de dispositivos y cámbialo en el comando. |

## 4. La sesión

Antes de empezar, cierra cualquier gemelo o puente que haya quedado abierto en otra terminal: debe haber **un solo flujo `EEG`** en la red. `python ver_flujos.py` lo muestra.

```bash
# terminal 1: casco -> LSL, y graba el EEG crudo
python puente_lsl.py --placa unicorn --grabar resultados/sesion_unicorn.csv

# LabRecorder: selecciona todos los flujos (EEG, IMU, Marcadores, Paso, Estado) y graba el XDF

# terminal 2: tablero
python tablero.py

# terminal 3: la sesión
python orquestador.py real --puerto COM4
```

**Mano virtual** (segunda pantalla, o en lugar de la órtesis para calibrar el ErrP): `python orquestador.py real --ortesis-sim --mano-virtual` y, en otra terminal, `python mano_virtual.py --pantalla 1` (o `demo.py lanzar ... --mano-virtual --pantalla 1`). Ábrela antes con `python mano_virtual.py --demo` para comprobar que la pantalla y la ventana funcionan; Esc la cierra. Si la órtesis es la real, la mano solo repite lo que se le ordena.

Con la app UnicornLSL como fuente, no lances el puente y usa `python orquestador.py real --puerto COM4 --fuente unicornlsl` (agrega `--eeg-nombre <nombre>` si hay más de un flujo de tipo `Data`).

### Atajo: `python demo.py`

Hace lo de arriba en el orden correcto, con una revisión previa y una bitácora. No cambia el orquestador: si algo falla aquí, los comandos manuales de arriba siguen valiendo.

```bash
python demo.py preflight --puerto COM4                  # solo revisa: OK / AVISO / FALLA, con qué hacer
python demo.py lanzar --puerto COM4                     # revisa, y lanza puente + tablero + orquestador
python demo.py lanzar --plan unicornlsl --puerto COM4   # la fuente es la app UnicornLSL: no lanza el puente
python demo.py lanzar --plan gemelo --ortesis-sim       # sin casco: el gemelo digital como fuente
python demo.py lanzar --ortesis-udp                     # la órtesis por Wi-Fi (ESP32 en 192.168.4.1), en lugar de --puerto
python demo.py lanzar --plan gemelo --ortesis-udp 127.0.0.1   # sin la placa: lanza también ortesis_udp_sim.py (la ESP32 simulada)
python demo.py lanzar --puerto COM4 -- --sham --preentrenado   # lo que va tras `--` pasa tal cual al orquestador
python demo.py planb --puerto COM4                      # plan B 1: tablero + repetición de la última sesión real
```

`--ortesis-udp [IP]` (y `--udp-puerto`) hace lo mismo que en el orquestador y es excluyente con `--ortesis-sim`: no pide pyserial ni la verificación USB, y el preflight avisa a dónde apunta (UDP no se puede comprobar sin mandar una orden; con el plan `gemelo` y una IP de esta laptop, `demo.py` lanza `ortesis_udp_sim.py` antes que el orquestador; con otro plan, el aviso recuerda lanzarlo a mano). `planb --ortesis-udp` repite los ángulos por Wi-Fi.

También acepta `--narrador`, `--copiloto`, `--flechas`, `--idioma en`, `--serie <número>` (varios cascos cerca), `--eeg-nombre` y `--sin-tablero`.

| El preflight revisa | AVISO o FALLA si |
|---|---|
| Código | no estás en la etiqueta `v-demo` (AVISO), hay cambios sin guardar (AVISO) o `origin/main` trae commits que no tienes (AVISO: pregunta a Luis; solo él aprueba cambios a `main`) |
| Dependencias | falta algo que se importa de verdad: numpy, scipy, scikit-learn, pylsl, pyriemann y, según el plan, brainflow, pyserial y pyqtgraph (FALLA) |
| Modelos | quedan calibraciones de otra persona o del gemelo (AVISO; `--limpiar-modelos` las mueve) |
| Flujos LSL | ya hay un flujo `EEG`, o `Marcadores`/`Paso`/`Estado` (FALLA); con `--plan unicornlsl`, no se ve el flujo `Data` de la app (FALLA) |
| Órtesis | el puerto no existe (FALLA) |
| Verificaciones de hoy | `verificar_unicorn.py` o `verificar_ortesis.py` no se corrieron, son de hace más de 4 h o probaron otra fuente (AVISO); una comprobación crítica en rojo (FALLA). **No las corre: son guiadas.** |
| IA y plan B | sin llave de la API (AVISO: narrador, copiloto y co-investigador usan plantillas); ninguna sesión real grabada para repetir (AVISO) |
| Disco | menos de 500 MB libres (FALLA) |

- Una FALLA detiene `lanzar` (código de salida 2). `--ignorar-fallas` sigue de todos modos, bajo tu responsabilidad, y queda escrito en la bitácora.
- Los procesos de fondo escriben en `resultados/logs_demo/<fecha>/` (`fuente.log`, `tablero.log`, `narrador.log`). La bitácora, con el preflight, los comandos exactos, los extras, cómo se cerró cada proceso y los archivos nuevos, queda en `resultados/demo_<fecha>.json`.
- Ctrl+C en la terminal del orquestador cierra todo, y cierra cada proceso por su PID (nunca por nombre). Primero se les pide que terminen, para que el puente suelte el casco.
- El código de salida de `lanzar` es el del orquestador. **No esconde un checkpoint en NO GO, no agrega `--forzar` ni `--saltar-calibracion` por su cuenta y no contesta el cuestionario.** Si hace falta forzar, se pide a propósito: `python demo.py lanzar -- --forzar`.
- El EEG crudo queda en `resultados/sesion_unicorn_<fecha>.csv` (con fecha), no en `sesion_unicorn.csv`: guarda ese. **LabRecorder no lo abre este programa**: ábrelo tú y selecciona todos los flujos; `Marcadores`, `Paso` y `Estado` aparecen cuando arranca el orquestador.
- **Sin probar en Windows.** Se probó en Linux contra el gemelo. Quedan sin comprobar con la laptop de la demo: el listado de puertos COM, el cierre con Ctrl+Break y que el puente suelte el casco al cerrarse. Antes de depender de él, corre en esa laptop `python demo.py preflight` y una vez `python demo.py lanzar --plan gemelo --ortesis-sim`.

**Qué va a pasar** (unos 15 minutos):

| Fase | Duración | Qué hace el piloto |
|---|---|---|
| CP1: señal y latencia | 1 min | Quieto, ojos abiertos. Luego la órtesis se mueve sola 40 veces. |
| Calibración de MI (CP2) | 3 a 6 min | Con `CERRAR`, imagina cerrar la mano derecha (sin moverla); con `RELAJA`, descansa. Las dos palabras se ven idénticas (mismo color y tamaño). Con `--cue-audio` suenan dos tonos: sube = CERRAR, baja = RELAJA; con `--cue-sin-visual` la pantalla solo muestra un `+`. El decoder usa solo C3, Cz y C4 (`--decoder-canales auto` para volver a elegir). |
| Calibración de ErrP (CP3) | 5 min | Mira la órtesis. La consola dice hacia dónde debe moverse; a veces se equivoca a propósito. |
| Lazo estático (30 pasos) | 1 min | Igual que en MI: imagina o relaja según la señal. La órtesis ya obedece. |
| Lazo adaptativo (120 pasos) | 4 a 5 min | Igual. En el paso 40 llega la perturbación. Cuando el tablero diga `AUTOMATICO`, la órtesis se mueve sola: solo obsérvala. |
| Evaluación (CP4) y cuestionario | 1 min | Tres afirmaciones, de 1 a 7, en la terminal 3. |

Cada ensayo empieza con la órtesis volviendo al punto medio. Pídele al piloto que no mueva la cabeza y que parpadee entre ensayos.

Cuando la órtesis ya llegó al tope (cerrada o abierta del todo), los pasos que quedan del ensayo no la mueven. Es normal: el sistema no aprende de esos pasos, porque sin movimiento no hay nada que el piloto pueda juzgar. EVALUACION dice cuántos fueron (`pasos sin movimiento`). En la calibración de ErrP la órtesis vuelve sola al centro de vez en cuando, para que cada ensayo tenga un movimiento que se vea.

**Tras un ensayo bueno, guarda** de `resultados/`: `sesion_real_<fecha>.csv`, `sesion_real_<fecha>_estado.jsonl`, `sesion_real_<fecha>_cuestionario.json`, `sesion_unicorn.csv` y el XDF de LabRecorder. El `_estado.jsonl` es el plan B: con él se puede repetir la sesión en el tablero.

## 5. Árbol de decisión por checkpoint

Un checkpoint en NO GO detiene la sesión (salvo el CP4, que solo informa). Los modelos ya calibrados quedan guardados en `modelos/`.

### CP1: señal por canal y latencia de la órtesis

El mensaje dice qué falló.

- **Un canal `plano`, `saturado` o `ruidoso`.**
  1. Reacomoda ese electrodo, agrega gel y espera un minuto.
  2. Vuelve a lanzar el orquestador.
  3. Si sigue mal tras dos intentos, no fuerces: con un canal malo el lazo entra en pausa segura todo el tiempo. Ve al plan B.
- **No encuentra el flujo `EEG`.** Revisa que el puente siga vivo en la terminal 1 y que ninguna otra aplicación esté conectada al casco. `python ver_flujos.py` debe mostrar un solo `EEG`.
- **Latencia del ACK** (MAD > 15 ms, p95 > 60 ms o más de 10 % sin ACK).
  1. Cambia el cable o el puerto USB y cierra lo que use el COM.
  2. Reinicia el ESP32 y repite `python verificar_ortesis.py --puerto COM4`.
  3. Si la órtesis no responde, haz la demo sin ella: `python orquestador.py real --ortesis-sim`. El tablero muestra todo; dilo al jurado.

### CP2: imaginación motora, BA ≥ 0.70

- **BA entre 0.60 y 0.70.** Repite una vez: recuérdale al piloto que imagine la sensación de cerrar la mano, sin moverla, y que relaje de verdad en `RELAJA`. Vuelve a lanzar el orquestador.
- **Sigue por debajo de 0.70.** Cambia de piloto si hay otro disponible.
- **No hay tiempo ni otro piloto.** `python orquestador.py real --puerto COM4 --forzar` sigue adelante con el decoder que haya. La órtesis se equivocará más; el agente lo corrige solo si el CP3 sale bien.

### CP3: detector de ErrP, BA ≥ 0.75 y especificidad ≥ 0.90

Es el riesgo principal: el Unicorn tiene 3 electrodos fronto-centrales. La curva de robustez (`docs/figuras/curva_robustez.png`) dice qué esperar según la BA.

- **BA entre 0.65 y 0.75, o especificidad entre 0.85 y 0.90.** Dos caminos:
  1. Repetir solo la calibración de ErrP (5 minutos), con el piloto más atento a la órtesis:
     ```bash
     python orquestador.py real --puerto COM4 --solo-errp
     ```
  2. Seguir con ese detector:
     ```bash
     python orquestador.py real --puerto COM4 --saltar-calibracion
     ```
     Con `--saltar-calibracion` no se vuelve a evaluar el CP3. El agente aprenderá más lento y el CP4 puede dar NO GO: en el gemelo, con un detector de BA 0.67 a 0.73 se recuperan en 2 minutos 10 u 11 sesiones de 16 (con uno de 0.83, las 16). Si el detector deja de informar, el aprendizaje se congela solo. Dilo tal cual: es lo que muestra la curva de robustez.

  En la corrida de referencia contra el gemelo pasó justo esto: BA 0.85 con especificidad 0.87 (NO GO), y con ese detector el agente se recuperó en 50 s.
- **BA por debajo de 0.65.** El detector no informa. Repite la calibración de ErrP una vez; si no mejora, ve al plan B.
- **BA ≥ 0.75 pero especificidad < 0.85.** Demasiadas falsas alarmas: repite con `--solo-errp`.

### CP4: recuperación en 120 s o menos

Se calcula al final y no detiene nada.

- **NO GO con un detector débil** (mira en EVALUACION `detector vivo: sens ..., espec ...`). Es lo esperado según la curva de robustez. Muestra el error del agente contra el de la sombra en el bloque completo y el control negativo (`docs/figuras/control_negativo.png`).
- **NO GO con un detector bueno.** Si hay tiempo, corre otro lazo con los mismos modelos (6 minutos): `python orquestador.py real --puerto COM4 --saltar-calibracion`.
- No inventes una recuperación que no ocurrió: el CSV queda como salió.

## 6. Plan B

En este orden:

1. **Repetir una sesión real de hoy.** Si ya hubo un ensayo bueno con el piloto, se repite en el tablero, y la órtesis hace los mismos movimientos:
   ```bash
   python tablero.py                                              # terminal 1
   python repetir_sesion.py --ultima --velocidad 2 --puerto COM4  # terminal 2
   ```
   `--ultima` toma la sesión real más reciente; también se le puede dar un archivo `resultados/sesion_real_<fecha>_estado.jsonl`. Empieza en el lazo y conserva los checkpoints. Es una repetición: nada se decide en vivo, y hay que decirlo. Atajo: `python demo.py planb --puerto COM4` (si no hay sesión grabada, lo dice y no abre nada).
2. **El gemelo digital en vivo**, sin casco. El lazo completo corre de verdad, pero el cerebro es sintético:
   ```bash
   python cerebro_sintetico.py                 # terminal 1, en lugar del puente
   python tablero.py                           # terminal 2
   python orquestador.py real --puerto COM4    # terminal 3 (o --ortesis-sim sin la órtesis)
   ```
   Tarda lo mismo que una sesión real. Hay que presentarlo como gemelo, no como una persona. Atajo: `python demo.py lanzar --plan gemelo --puerto COM4`.
3. **La señal grabada.** `python puente_lsl.py --placa playback --archivo resultados/sesion_unicorn.csv` (o el `sesion_unicorn_<fecha>.csv` que dejó `demo.py`) vuelve a publicar el EEG crudo. Sirve para mostrar la señal y los semáforos; no sirve para el lazo, porque lo grabado no responde a las señales nuevas.

## 7. Si algo se cae a media sesión

| Qué pasa | Qué hacer |
|---|---|
| El tablero dice `PAUSA SEGURA` | Nada: el sistema abre la órtesis, espera y sigue solo cuando la señal vuelve 3 s. El tablero dice qué semáforo falló y por qué. |
| La pausa no termina | Arregla lo que indica: `EEG` (puente o dongle), un electrodo (reacomódalo) u `ORTESIS` (cable). No hay que reiniciar nada. |
| Se cerró el orquestador | El mismo comando con `--reanudar`. Sigue la misma sesión y el mismo CSV, sin recalibrar. |
| Se cerró el puente | Vuelve a lanzarlo en la terminal 1; el orquestador se reconecta solo. |
| Se cerró el tablero | `python tablero.py` otra vez; no afecta a la sesión. |
| El semáforo `PILOTO` se pone amarillo o rojo | Solo avisa: el alfa occipital subió (cansancio u ojos cerrados). Dale un descanso al piloto entre bloques. |
| En consola: `[detector] fallo la co-adaptacion` | Nada: el lazo sigue con el detector vigente. Para apagar el re-entrenamiento en la siguiente sesión: `--sin-coadaptativo`. |

## 8. Qué decir, y qué no

- **El resultado central es el control negativo.** En el gemelo, sin la evidencia del ErrP el agente se recupera en 1 de cada 16 sesiones; con ella, en 10 a 16 de 16, según la calidad del detector (`docs/figuras/control_negativo.png`). El agente aprende de la señal de error del cerebro, no del protocolo.
- Un tercio de los pasos no mueve la órtesis (ya está en el tope) y de esos el agente no aprende. Lo decimos tal cual: sin movimiento no hay nada que el cerebro pueda juzgar.
- Las cifras del README y de las figuras son **del simulador o del gemelo**, no de una persona. Las únicas cifras de una persona son las de la sesión de hoy.
- El IIC es **exploratorio**: con unos 12 movimientos ajenos por sesión su intervalo mide ±0.5 (`docs/figuras/potencia_iic.png`).
- La velocidad del agente depende de la calidad del detector (`docs/figuras/curva_robustez.png`; es del simulador rápido, donde todo paso informa, así que el gemelo y una persona son más lentos que esa curva). Si el detector de hoy sale débil, el agente será lento, y eso es lo que el sistema debe hacer.
