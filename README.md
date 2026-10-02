# ortesis-bci

Órtesis de mano controlada por imaginación motora, con un agente que se corrige solo usando el **potencial de error (ErrP)** del cerebro como recompensa.

## Hardware

El casco de la demo es un **g.tec Unicorn Hybrid Black**: 8 canales de EEG a 250 Hz por Bluetooth, más acelerómetro y giroscopio de 3 ejes, batería, contador de muestras e indicador de validez. El contrato (`config.py`) usa su montaje y le da un papel a cada sensor:

| Sensores | Papel |
|---|---|
| C3, Cz, C4 | Imaginación motora (ERD mu/beta) |
| Fz, Cz, Pz | Potencial de error (ErrP) |
| PO7, Oz, PO8 | Respuesta visual al movimiento (Tarea 2) y alfa occipital (semáforo PILOTO) |
| Acelerómetro y giroscopio | Rechazo de artefactos por movimiento de cabeza |
| Contador de muestras | Hora de cada muestra y pérdidas de Bluetooth |

La calibración real decidirá, por validación cruzada, si cada modelo usa los canales de su papel o los 8. El gemelo ya se comporta como un Unicorn (montaje, respuesta visual occipital, IMU, contador y pérdidas de Bluetooth) y puede publicar en el formato del puente o en el de la app UnicornLSL. **En curso:** la selección de canales en la calibración, el CP1 sin impedancias y el rechazo por movimiento de cabeza todavía no están implementados (ver `TAREAS.md`); las secciones de abajo que hablan del Cyton quedan como estaban hasta entonces.

El orquestador puede leer el EEG de dos fuentes (`--fuente`): `puente` (por defecto: `puente_lsl.py` o el gemelo; flujos `EEG` e `IMU`) o `unicornlsl` (la app de g.tec: un flujo de tipo `Data` con 17 canales, que se resuelve por tipo o con `--eeg-nombre <nombre o número de serie>`). Solo una aplicación puede conectarse al casco a la vez.

## Instalación

```bash
python -m venv .venv
source .venv/Scripts/activate        # Git Bash en Windows (en CMD: .venv\Scripts\activate)
pip install -r requirements.txt
python pruebas.py                    # debe decir 33/33 pruebas pasaron
```

## Archivos

| Archivo | Qué hace |
|---|---|
| `config.py` | **El contrato**: flujos LSL, marcadores, columnas del CSV, umbrales, protocolo del ESP32, estados. Nadie define estas cosas en otro lado. |
| `agente_errp.py` | Agente bayesiano + `ConfianzaDetector` (confiabilidad viva del detector de ErrP). |
| `simulador_lazo.py` | Piloto sintético para probar y comparar agentes sin casco. |
| `orquestador.py` | Máquina de estados, calibraciones, checkpoints go/no go, lazo, CSV. Backends `sim` y `real`. |
| `hardware.py` | EEG por LSL, órtesis por USB (o simulada), decoder de MI con recentrado y detector de ErrP calibrado. |
| `puente_lsl.py` | BrainFlow → LSL: Unicorn (`--placa unicorn --serie <num>`), placa sintética, playback o Cyton. Publica `EEG` e `IMU` con la hora de cada muestra reconstruida por contador, y registra los huecos de Bluetooth. |
| `verificar_unicorn.py` | Con el casco puesto: comprueba orden de canales, unidades, contador, IMU, batería y validez, y dice qué fuente usar. |
| `cerebro_sintetico.py` | **Gemelo digital del piloto**: publica EEG por LSL que *reacciona* al lazo (ERD al imaginar, ErrP cuando la órtesis se equivoca). Reemplaza al casco para ensayar. |
| `salud.py` | `Vigilante`: semáforo VERDE / AMARILLO / ROJO por subsistema (EEG, órtesis, reloj, detector) y el retroceso de las reconexiones. |
| `caos.py` | `PlanCaos`: fallas reproducibles por semilla (ingeniería del caos aplicada al lazo). |
| `tablero.py` | Tablero en vivo de 5 paneles, con tres semáforos de salud en la cabecera. |
| `ver_flujos.py` | Diagnóstico: qué flujos LSL hay en la red y qué publican. |
| `pruebas.py` | Pruebas automáticas sin hardware. |

## Qué tiene de nuevo

**Agente (`agente_errp.py`)**

- **Kalman sobre `beta`.** La corrección del decoder tiene media y varianza; la varianza es la tasa de aprendizaje, así que no hay `eta` que elegir.
- **P_hat bayesiano con la confiabilidad *viva* del detector.** Con salida calibrada usa la probabilidad completa del ErrP y corrige el cambio de prior entre calibración y lazo.
- **Dos detectores de cambio** que re-inflan la varianza. El ErrP decide *qué* aprender; los detectores deciden *cuándo* aprender rápido.
  - *Sesgo de decisiones*: es rápido, y asume metas balanceadas.
  - *Chequeo predictivo*: el agente predice cuántos ErrP debería provocar según su confianza; si aparecen más, está seguro y equivocado. Funciona aunque las metas no estén balanceadas.
- **`ConfianzaDetector`.** Estima en vivo la sensibilidad y especificidad del detector de ErrP con posteriores Beta con olvido. El aprendizaje se escala con el índice de Youden y se congela si el detector deja de informar.

**Señal (`hardware.py`, `puente_lsl.py`)**

- **Recentrado riemanniano no supervisado.** El decoder de MI se re-centra solo con cada ventana. En la prueba, tras mezclar canales pasa de 50 % a 98 % de exactitud sin recalibrar.
- **Detector de ErrP de dos vistas fusionadas** (temporal con LDA encogido + geométrica de Riemann), con probabilidades calibradas, umbral de Neyman-Pearson (especificidad ≥ 0.90) y detector de rareza para épocas fuera de distribución.
- **Calibración honesta.** MI se detiene sola cuando el intervalo de confianza ya decide (mínimo 36 ensayos); el detector de ErrP usa siempre 120 épocas con el umbral elegido por validación anidada (ver el hallazgo de la maldición del ganador).
- **Hora por contador.** La hora de cada muestra se reconstruye con el contador del casco, sin el jitter de llegada por Bluetooth; una pérdida queda como un hueco visible.

Resultados en simulación (30 sujetos, perturbación de 2.4 logits, `python simulador_lazo.py --semillas 30`; medidos en Windows 11 con Python 3.11 el 2 de octubre de 2026):

| Agente | Error antes | Primeros 2 min | Después |
|---|---|---|---|
| Estático (sin aprender) | 0.165 | 0.325 | 0.316 |
| `eta` fijo 0.3 | 0.170 | 0.246 | 0.178 |
| **Bayes (el nuestro)** | **0.169** | **0.214** | **0.175** |

Con el detector de ErrP degradado a propósito, el aprendizaje baja a menos del 10 % y se congela la mayor parte de la falla. Los congelamientos en falso son de 1 %.

## Gemelo digital del piloto

`cerebro_sintetico.py` sustituye al casco (un Unicorn Hybrid Black). Escucha las señales y los pasos del orquestador y responde como una persona: desincroniza mu/beta sobre C3 al imaginar cerrar, genera una respuesta visual occipital (N1 ≈ 170 ms en PO7/Oz/PO8) ante cada movimiento de la órtesis y un ErrP fronto-central (Ne ≈ 250 ms, Pe ≈ 350 ms en Fz/Cz/Pz) cuando va al lado contrario, parpadea, mueve la cabeza de vez en cuando (el giroscopio lo registra y el EEG se ensucia), pierde muestras por Bluetooth si se le pide y, opcionalmente, se cansa. Con él se valida el camino **real** completo con verdad conocida.

Corrida completa (`orquestador.py real --ortesis-sim` contra el cerebro sintético): CP1, CP2 (MI BA 0.83), CP3 (ErrP BA 0.90) y CP4 en **GO**. Tras la perturbación, el agente tuvo un error de **0.14**; el decoder sin aprender, de 0.47.

```bash
python cerebro_sintetico.py --banco          # decoder y detector offline, en segundos
python cerebro_sintetico.py                  # terminal 1 (en lugar de puente_lsl.py)
python cerebro_sintetico.py --erd 0.15 --errp 4 --fatiga 0.5   # piloto difícil
python cerebro_sintetico.py --perdidas-bt 20                   # 20 pérdidas de Bluetooth por minuto
python cerebro_sintetico.py --formato unicornlsl               # publica como la app UnicornLSL de g.tec
```

## Hallazgo: la calibración secuencial inflaba la exactitud (maldición del ganador)

**El síntoma.** En una sesión contra el gemelo, el detector de ErrP se calibró con especificidad 0.96 y en el lazo vivía en 0.72; el decoder de MI se calibró con BA 0.88 y el bloque estático tuvo 0.37 de error.

**Cómo lo encontró el gemelo.** Con ablaciones y sin LSL (`estudios/`) se descartó una a una cada sospecha: el recentrado en línea (±0.01 de error), las respuestas cerebrales a cada movimiento dentro de la ventana de MI (±0.01), el traslape de épocas con un paso cada 0.87 s (especificidad 0.95 contra 0.96 con 2.5 s) y una carrera entre mensajes en el gemelo (0 de 220 movimientos mal juzgados). Lo que sí apareció fue medir lo reportado en calibración contra lo real **en épocas nuevas** (16 sujetos del gemelo, montaje del Unicorn):

| Calibración del detector de ErrP | Sujetos | Reportado: sens / espec / BA | Real en épocas nuevas |
|---|---|---|---|
| Secuencial: GO temprano (40 a 60 épocas) | 2 de 16 | 0.84 / 0.90 / **0.87** | 0.56 / 0.81 / **0.69** |
| Secuencial: NO GO temprano (60 épocas) | 11 de 16 | BA 0.57 | con 120 épocas habría sido 0.72 |
| **Corregida: 120 épocas fijas y umbral anidado** | 16 de 16 | 0.51 / 0.89 / **0.70** | 0.56 / 0.89 / **0.73** |

La parada secuencial revisaba cada 10 épocas y daba GO en cuanto el estimado salía alto: con pocos datos eso solo ocurre por suerte, y además el umbral de Neyman-Pearson se elegía sobre los mismos puntajes que se evaluaban. Es la maldición del ganador: se reporta el máximo de varios estimados ruidosos. En MI el efecto era menor (+0.02 de error con un mínimo de 24 ensayos; +0.00 con 36).

**La corrección.** El detector de ErrP se calibra siempre con las 120 épocas, sin GO ni NO GO tempranos, y el umbral se elige dentro de cada pliegue (validación anidada): lo reportado queda a −0.02 ± 0.04 de lo real. MI exige al menos 36 ensayos. Además, el primer paso de cada ensayo fallaba más (0.22 contra 0.18) porque su ventana empezaba con la transición mental; esperar 1 s más tras la señal lo baja a 0.15. El resto del 0.37 era ruido: un bloque estático de 30 pasos tiene desviación de 0.08 (en vivo, con 150 pasos: BA 0.83 en calibración y 0.20 de error).

**El costo honesto.** Con el montaje del Unicorn (3 electrodos fronto-centrales) y el ErrP del gemelo, la BA real del detector ronda 0.73 y el CP3 (0.75) dio GO en 1 de 16 sujetos. Antes "pasaba" por el sesgo. Son cifras del gemelo, no de una persona.

## Resiliencia: el lazo que no se cae

Si se desconecta el dongle, se reinicia el ESP32, se congela LSL o llega una época corrupta, el sistema lo detecta, se protege, se recupera solo y no pierde la sesión.

**Semáforos (`salud.py`).** Antes de cada paso, el `Vigilante` revisa cuatro subsistemas. Cada cambio de color sale en consola, en el flujo `Estado` y como marcador `salud:<subsistema>:<color>`. Los umbrales están en `config.SALUD`.

| Subsistema | Qué mide | AMARILLO | ROJO |
|---|---|---|---|
| EEG | edad de la última muestra, muestras que de verdad llegan, canales | 0.3 s sin muestras; 10 % menos muestras | 1 s sin muestras; 25 % menos; un canal plano, saturado o ruidoso |
| Órtesis | ACK perdidos, latencia, puerto | 1 ACK perdido; latencia ≥ 80 ms | 3 ACK perdidos seguidos; puerto caído |
| Reloj | deriva del retraso del EEG contra su línea base | 20 ms | 50 ms |
| Detector | fiabilidad viva del `ConfianzaDetector` | fiabilidad < 0.7 | aprendizaje congelado |

El detector empieza en `CALENTANDO` (gris en el tablero) hasta tener 15 épocas válidas: antes de eso su fiabilidad es ruido y no emite cambios.

**Escalera de degradación**, de mejor a peor:

| Escalón | Condición | Qué hace el sistema |
|---|---|---|
| 1 | Todo en VERDE | Lazo normal. |
| 2 | Detector poco fiable o reloj en ROJO | El lazo sigue; el agente no aprende. |
| 3 | EEG perdido o canal despegado | `PAUSA_SEGURA`: la órtesis se abre despacio; consola y tablero dicen qué electrodo falla y por qué. |
| 4 | Órtesis perdida | `PAUSA_SEGURA` sin poder moverla: se registra, se avisa y se reconecta sola. |

De la pausa se sale sola tras 3 s continuos de EEG y órtesis en VERDE; se regresa al estado previo y se repite el cue del ensayo. Las pausas no consumen pasos del bloque.

**Qué queda fuera del análisis.** La columna `excluido` del CSV lleva el motivo; `EVALUACION` ignora esas filas y reporta cuántas fueron.

| Motivo | Significado |
|---|---|
| `pausa:eeg`, `pausa:canal`, `pausa:ortesis` | Fila de una pausa segura. |
| `sin_ack` | La órtesis no confirmó el movimiento: no hay instante desde el cual cortar la época. |
| `epoca_invalida` | La época cruza un corte de EEG o le faltan muestras. Las épocas se cortan por tiempo, no por número de muestras. |

**Reconexión automática**, con retroceso exponencial (0.5 s, 1, 2, 4, 8, 8...):

- `EntradaEEG` vuelve a resolver el flujo si se pierde; al reconectar vacía el buffer y reinicia el reloj. Si hay varios flujos iguales en la red (un gemelo olvidado en otra terminal), avisa, usa el más reciente y no salta a otro mientras el suyo siga publicado. La hora de cada muestra viene de la fuente, reconstruida con el contador del casco, y **no** se usa el suavizado de marcas de LSL: una pérdida de Bluetooth queda como un hueco en la hora, y tras un silencio las marcas siguen alineadas sin tener que vaciar el buffer. La ventana de imaginación motora tolera pérdidas chicas (hasta 10 % de las muestras y huecos de hasta 0.25 s, interpolados); la época de ErrP no cruza huecos de más de 20 ms.
- `OrtesisSerial` reabre el puerto y conserva `seq`. `mover()` nunca lanza una excepción.
- `puente_lsl.py` vuelve a preparar la placa si deja de entregar datos 2 s. **Probado solo con la placa sintética de BrainFlow; no probado con el Cyton real.**
- `puente_lsl.py` estampa cada muestra con la hora reconstruida por el contador de la placa e imprime cada 30 s un **registro de huecos**: ráfagas de muestras perdidas, su duración media y máxima, silencios de llegada y frecuencia por minuto. Sirve para medir el Bluetooth con el casco real.

**Calibración.** Un ensayo afectado por una falla se repite como máximo 3 veces (la consola dice por qué); después se avisa y la calibración sigue.

**Persistencia.** Después de cada paso se guarda una instantánea atómica en `resultados/estado_sesion.json`. Tras un cierre inesperado (o un `Ctrl+C`), el mismo comando con `--reanudar` continúa la misma sesión y el mismo CSV. En `real` carga los modelos guardados y no recalibra.

**Modo caos.** `--caos <semilla>` inyecta el "caos estándar" (`config.CAOS_ESTANDAR`), reproducible por semilla:

| Falla | Frecuencia | Detalle |
|---|---|---|
| Corte de EEG | uno cada ~45 s | 1 a 5 s; los de 3 s o más destruyen y recrean el flujo |
| ACK perdido | 3 % de los pasos | — |
| Pico de latencia | 5 % de los pasos | 80 a 300 ms |
| Ráfaga de parpadeos | una cada ~40 s | 2 a 4 s |
| Canal despegado | uno cada ~120 s | 4 a 10 s, plano o ruidoso |

```bash
python orquestador.py sim --ciclo 0 --caos 1                    # simulador con caos, en segundos
python cerebro_sintetico.py --caos 1                            # terminal 1: gemelo con caos
python orquestador.py real --ortesis-sim --forzar --caos 1      # terminal 3 (--forzar: el caos tumba el CP1)
python orquestador.py sim --ciclo 0 --caos 1 --caos-nivel leve  # caos leve: una falla cada 2 a 3 minutos
python orquestador.py real --puerto COM4 --reanudar             # continuar tras un cierre inesperado
```

**Resultados con caos (exploratorios; simulador y gemelo, no una persona).** Error en los ~2 min tras la perturbación, sobre pasos no excluidos. "Caos leve" (`--caos-nivel leve`) es una falla cada 2 a 3 minutos; el estándar, una cada pocos segundos.

| Medición | Agente | Sombra | Excluidos y pausas |
|---|---|---|---|
| Simulador, 30 sujetos, sin caos | 0.235 | 0.323 | ninguno; agente por debajo en 26 de 30 sujetos |
| Simulador, 30 sujetos, caos leve | 0.232 | 0.326 | 105 de 10 867 filas: `pausa:eeg` 39, `pausa:canal` 28, `epoca_invalida` 20, `sin_ack` 18; agente por debajo en 27 de 30 |
| Simulador, 30 sujetos, caos estándar | 0.222 | 0.333 | 1109 de 11 425 filas: `pausa:eeg` 443, `sin_ack` 332, `pausa:canal` 181, `epoca_invalida` 152, `pausa:ortesis` 1; agente por debajo en 29 de 30 |
| Gemelo, sin caos | 0.26 | 0.46 | 0 de 150 filas; ninguna pausa; CP1 a CP4 en GO |
| Gemelo, caos leve (semilla 2) | 0.37 | 0.47 | 2 de 151 filas: `pausa:eeg` 1, `epoca_invalida` 1; 1 pausa, reanudada sola; CP4 en GO (22 s) |
| Gemelo, caos estándar (semilla 1) | 0.37 | 0.47 | 20 de 160 filas: `pausa:eeg` 7, `sin_ack` 6, `epoca_invalida` 4, `pausa:canal` 3; 10 pausas, todas reanudadas solas; CP4 en GO (24 s) |

Las tres filas del gemelo usan los mismos modelos (una sola calibración) ; son del montaje anterior y de antes de la hora por contador; es **una corrida por condición**, así que no es concluyente. La semilla del caos leve se eligió para que cayera al menos un corte de EEG dentro de los ~5 minutos de sesión. Antes, con el umbral en 0.3 s, tres corridas con caos estándar dieron 0.44/0.49, 0.30/0.53 y 0.46/0.47 (agente/sombra), con modelos distintos entre ellas.

En el simulador el agente sigue claramente por debajo de la sombra con caos leve y estándar. En el gemelo quedó por debajo en todas las corridas, por márgenes entre 0.01 y 0.23. El agente no se modificó para estas mediciones. Todas las sesiones terminaron sin excepción y todas las pausas se reanudaron solas.

**Lo que enseñó el caos (medido, no supuesto):**

- Tras un silencio sin perder el flujo, el suavizado de marcas de tiempo de LSL (dejitter) deja las marcas atrasadas: 2.7 s tras un hueco de 3 s, y tarda más de 10 s en converger. Eso desalineaba las épocas de ErrP y congelaba el aprendizaje. Primero se resolvió renovando la entrada tras cada silencio; ahora la hora de cada muestra se reconstruye con el contador del casco y el suavizado no se usa, así que el problema desaparece de raíz (prueba `silencio_sin_recrear`: 5 ms de diferencia tras 2.5 s de silencio, sin vaciar el buffer).
- La deriva del reloj se mide contra una mediana móvil de ~30 s: detecta un cambio de desfase y lo absorbe, así el semáforo del reloj nunca queda en ROJO para siempre.
- Una sesión sobrevivió a 37 minutos de suspensión del equipo (tapa cerrada) y se reanudó sola al despertar. Aun así: **no cierres la tapa durante la demo**.

## Probar sin hardware

```bash
python simulador_lazo.py                        # tabla + figura
python simulador_lazo.py --falla_detector       # congelamiento
python simulador_lazo.py --sin_sesgo            # metas no balanceadas
python orquestador.py sim --ciclo 0             # orquestador completo, rápido
python tablero.py  &  python orquestador.py sim # ver el tablero en vivo
```

## El día de la demo (en este orden)

**0. Verificar el casco (una vez, menos de 5 minutos).** El Unicorn llega el mismo día, así que hay cosas que solo se pudieron probar con el gemelo. Con el casco puesto y emparejado con **su dongle** (no con el Bluetooth de la laptop):

```bash
python verificar_unicorn.py brainflow            # con la Unicorn Suite cerrada
python verificar_unicorn.py lsl                  # solo si lo anterior falla: con la app UnicornLSL abierta y en Start
```

Guía de unos 50 segundos (quieto, parpadear, ojos cerrados, mover la cabeza). Comprueba el orden de los canales, las unidades del EEG, que el contador avance de 1 en 1, que el acelerómetro mida ~1 g, la batería y la validez, y termina con un **veredicto** que dice qué comando usar. Solo una aplicación puede conectarse al casco a la vez.

**1. Sesión**, con BrainFlow como fuente principal:

```bash
# terminal 1: casco -> LSL (y graba crudo para el plan B)
python puente_lsl.py --placa unicorn --grabar resultados/sesion_unicorn.csv
# LabRecorder: seleccionar todos los flujos y grabar XDF
# terminal 2
python tablero.py
# terminal 3
python orquestador.py real --puerto COM4
```

Con la app UnicornLSL como fuente de respaldo no se usa el puente: `python orquestador.py real --puerto COM4 --fuente unicornlsl` (agrega `--eeg-nombre <nombre>` si hay más de un flujo de tipo `Data`).

Con modelos ya calibrados: `--saltar-calibracion`. Sin ESP32: `--ortesis-sim`.
**Plan B** (checkpoint 3 falla): `python puente_lsl.py --placa playback --archivo resultados/sesion_unicorn.csv` y correr el orquestador igual.

El puente imprime cada 30 s el registro de huecos de Bluetooth y la batería.

**Antes de empezar:** cierra cualquier `cerebro_sintetico.py` o `puente_lsl.py` que haya quedado abierto en otra terminal. Debe haber un solo flujo `EEG` en la red; `python ver_flujos.py` lo muestra.

**Si algo se cae a media demo:** el sistema entra solo en `PAUSA_SEGURA` y sale solo. Si se cerró el orquestador, vuelve a lanzarlo con `--reanudar`.

## Checkpoints (los decide P1)

| CP | Dónde | Criterio | Si falla |
|---|---|---|---|
| 1 | IMPEDANCIAS | Calidad de señal por canal (el Unicorn no mide impedancias) y latencia del ACK en 40 movimientos: MAD ≤ 15 ms, p95 ≤ 60 ms, ACK perdidos ≤ 10 % | Más gel / revisar firmware antes de seguir |
| 2 | CAL_MI | Exactitud balanceada MI ≥ 0.70 (secuencial, mínimo 36 ensayos) | Cambiar piloto o mano vs pies |
| 3 | CAL_ERRP | BA ErrP ≥ 0.75 y especificidad ≥ 0.90, con 120 épocas fijas y umbral anidado | Plan B: sesión grabada |
| 4 | EVALUACION | Recuperación ≤ 120 s | Congelar y usar la ruta de 24 h |

`--forzar` continúa aunque un checkpoint diga NO GO (solo para pruebas).

## Protocolo del ESP32 (para P2)

```
PC -> ESP32   M,<seq>,<angulo 0-1000>,<duracion_ms>\n     0 = abierta, 1000 = cerrada
ESP32 -> PC   A,<seq>,<t_us>\n     ACK al aplicar el primer pulso (marca el inicio del ErrP)
ESP32 -> PC   T,<t_us>,<angulo>,<fsr>\n     telemetría a 50 Hz
```
USB serial a 115200 baudios. El orquestador espera el ACK máximo 300 ms por paso y mide la latencia de cada uno. Si el ACK no llega, el paso queda excluido y el lazo sigue; con tres seguidos entra en `PAUSA_SEGURA`. El ESP32 solo debe devolver el `seq` que recibió: tras reiniciarse no necesita recordar nada.
