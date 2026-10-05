# ortesis-bci

Órtesis de mano controlada por imaginación motora, con un agente que se corrige solo usando el **potencial de error (ErrP)** del cerebro como recompensa.

## Tres escalas de aprendizaje

El sistema aprende en tres escalas de tiempo, y en las tres la última palabra sobre lo que cambia la tiene una regla explícita o una persona:

| Escala | Quién aprende | Cada cuánto | De qué aprende | Estado |
|---|---|---|---|---|
| Rápida | El **agente bayesiano** (`agente_errp.py`) | cada paso (~2 s) | Del ErrP del piloto: corrige el sesgo del decoder | En la demo |
| Media | El **detector co-adaptativo** (`hardware.DetectorCoadaptativo`) | cada ~20 pasos | De las épocas del propio lazo: se re-entrena y se prueba en sombra antes de entrar | En la demo |
| Lenta | **Claude como co-investigador** (`ia.py`) | entre bloques y entre sesiones | De un resumen agregado del bloque: propone continuar, pausar, ajustar un parámetro o recalibrar | Apagado por defecto |

La escala lenta nunca toca el lazo de control: recibe solo métricas agregadas y anónimas (nunca EEG crudo), su propuesta se valida contra rangos seguros en `config.py` y **nada cambia sin que una persona la apruebe**. Sin llave o sin conexión, las mismas decisiones salen de reglas deterministas.

## Hardware

El casco de la demo es un **g.tec Unicorn Hybrid Black**: 8 canales de EEG a 250 Hz por Bluetooth, más acelerómetro y giroscopio de 3 ejes, batería, contador de muestras e indicador de validez. El contrato (`config.py`) usa su montaje y le da un papel a cada sensor:

| Sensores | Papel |
|---|---|
| C3, Cz, C4 | Imaginación motora (ERD mu/beta) |
| Fz, Cz, Pz | Potencial de error (ErrP) |
| PO7, Oz, PO8 | Respuesta visual al movimiento (Tarea 2) y alfa occipital (semáforo PILOTO) |
| Acelerómetro y giroscopio | Rechazo de artefactos por movimiento de cabeza |
| Contador de muestras | Hora de cada muestra y pérdidas de Bluetooth |

La calibración real decidirá, por validación cruzada, si cada modelo usa los canales de su papel o los 8. El gemelo ya se comporta como un Unicorn (montaje, respuesta visual occipital, IMU, contador y pérdidas de Bluetooth) y puede publicar en el formato del puente o en el de la app UnicornLSL. La selección de canales en la calibración, el CP1 sin impedancias (calidad de señal por canal y latencia del ACK) y el rechazo por movimiento de cabeza ya están implementados; **nada se ha probado todavía con el casco** (`verificar_unicorn.py` es lo primero el domingo).

El orquestador puede leer el EEG de dos fuentes (`--fuente`): `puente` (por defecto: `puente_lsl.py` o el gemelo; flujos `EEG` e `IMU`) o `unicornlsl` (la app de g.tec: un flujo de tipo `Data` con 17 canales, que se resuelve por tipo o con `--eeg-nombre <nombre o número de serie>`). Solo una aplicación puede conectarse al casco a la vez.

## Instalación

```bash
python -m venv .venv
source .venv/Scripts/activate        # Git Bash en Windows (en CMD: .venv\Scripts\activate)
pip install -r requirements.txt
python pruebas.py                    # debe decir 54/54 pruebas pasaron
```

## Archivos

| Archivo | Qué hace |
|---|---|
| `config.py` | **El contrato**: flujos LSL, marcadores, columnas del CSV, umbrales, protocolo del ESP32, estados. Nadie define estas cosas en otro lado. |
| `agente_errp.py` | Agente bayesiano + `ConfianzaDetector` (confiabilidad viva del detector de ErrP) + `SenalSham` (lo que recibe el agente en el bloque sham del control causal). |
| `simulador_lazo.py` | Piloto sintético para probar y comparar agentes sin casco. |
| `orquestador.py` | Máquina de estados, calibraciones, checkpoints go/no go, lazo, CSV. Backends `sim` y `real`. Con `--sham`, control causal: bloque real contra bloque sham. |
| `hardware.py` | EEG por LSL, órtesis por USB (o simulada), decoder de MI con recentrado y detector de ErrP calibrado. |
| `puente_lsl.py` | BrainFlow → LSL: Unicorn (`--placa unicorn --serie <num>`), placa sintética, playback o Cyton. Publica `EEG` e `IMU` con la hora de cada muestra reconstruida por contador, y registra los huecos de Bluetooth. |
| `verificar_unicorn.py` | Con el casco puesto: comprueba orden de canales, unidades, contador, IMU, batería y validez, y dice qué fuente usar. |
| `bloque_sham.py` | Bloque sham (B1): con el piloto en reposo la órtesis se mueve al azar y `p(t)` del decoder no debe seguirla (`hardware.evaluar_sham`). |
| `cerebro_sintetico.py` | **Gemelo digital del piloto**: publica EEG por LSL que *reacciona* al lazo (ERD al imaginar, ErrP cuando la órtesis se equivoca, N1 visual atenuada según `--embodiment`). Reemplaza al casco para ensayar. |
| `embodiment.py` | Tarea 2 (exploratorio): N1 visual de cada movimiento e índice de integración corporal (IIC) con intervalo y tendencia. |
| `estudios/` | Mediciones offline con el gemelo y el simulador que respaldan cada decisión (ver su README). |
| `salud.py` | `Vigilante`: semáforo VERDE / AMARILLO / ROJO por subsistema (EEG, órtesis, reloj, detector y piloto, que solo avisa) y el retroceso de las reconexiones. |
| `caos.py` | `PlanCaos`: fallas reproducibles por semilla (ingeniería del caos aplicada al lazo). |
| `repetir_sesion.py` | Plan B: repite en el tablero, y si se quiere en la órtesis, una sesión grabada (`resultados/sesion_..._estado.jsonl`). |
| `ortesis_udp_sim.py` | Firmware 1.2 de la ESP32 simulado en la laptop (UDP): prueba `OrtesisUDP` y el orquestador sin la placa. |
| `verificar_ortesis.py` | Mide con la telemetría del ESP32 cuánto tarda la órtesis en empezar a moverse tras el ACK. |
| `ia.py` | Lo común a toda la IA: llave desde `.env`, lo que puede salir hacia la API (`sanear`), preguntas con herramientas, y el esquema, la validación y las reglas deterministas de las propuestas. Nada de esto corre dentro del lazo. |
| `copiloto.py` | Copiloto clínico: preguntas sobre una sesión respondidas con herramientas sobre su CSV, e informe entre sesiones. |
| `narrador.py` | Narrador para el jurado: proceso aparte que escucha `Estado` y publica una frase por evento relevante en el flujo `Narracion` (Claude, con plantillas de respaldo). |
| `tablero.py` | Tablero en vivo de 5 paneles, con cuatro semáforos en la cabecera (EEG, órtesis, detector y piloto), el aviso AUTOMATICO de los movimientos ajenos y una línea con el IIC. Con `--flechas`, un sexto panel con la contribución de cada ErrP al cambio de `beta`. |
| `ver_flujos.py` | Diagnóstico: qué flujos LSL hay en la red y qué publican. |
| `memoria.py` | Memoria entre sesiones del mismo piloto: lo que deja una sesión (`--guardar-memoria`) para que la siguiente calibre más corto (`--desde-sesion`). |
| `estado_sistema.py` | Revisión previa del sistema (casco, flujos LSL, procesos, ACK, laptop, disco, API y git) y la franja en vivo de `tablero.py --estado-sistema`; cada falla con su solución en una línea. |
| `pruebas.py` | Pruebas automáticas sin hardware. |

## Qué tiene de nuevo

**Agente (`agente_errp.py`)**

- **Kalman sobre `beta`.** La corrección del decoder tiene media y varianza; la varianza es la tasa de aprendizaje, así que no hay `eta` que elegir.
- **P_hat bayesiano con la confiabilidad *viva* del detector.** Con salida calibrada usa la probabilidad completa del ErrP y corrige el cambio de prior entre calibración y lazo.
- **Dos detectores de cambio** que re-inflan la varianza. El ErrP decide *qué* aprender; los detectores deciden *cuándo* aprender rápido.
  - *Sesgo de decisiones*: es rápido, y asume metas balanceadas.
  - *Chequeo predictivo*: el agente predice cuántos ErrP debería provocar según su confianza; si aparecen más, está seguro y equivocado. Funciona aunque las metas no estén balanceadas.
- **`ConfianzaDetector`.** Estima en vivo la sensibilidad y especificidad del detector de ErrP con posteriores Beta con olvido. El aprendizaje se escala con el índice de Youden y se congela si el detector deja de informar.
- **Un paso que no mueve la órtesis no enseña.** Con la órtesis ya en el tope, el paso siguiente no se ve, así que no hay ErrP que leer. Ese paso cuenta como decisión en el análisis, pero el agente no aprende de él y no cuenta como detección fallida para la confianza del detector (columna `alineacion = sin_movimiento`, marcador `paso_quieto:<seq>`, interruptor `config.IGNORAR_SIN_MOVIMIENTO`). En la calibración de ErrP la órtesis vuelve al centro cuando el movimiento no cabe en el recorrido: cada época tiene un movimiento real.

**Señal (`hardware.py`, `puente_lsl.py`)**

- **Recentrado riemanniano no supervisado.** El decoder de MI se re-centra solo con cada ventana. En la prueba, tras mezclar canales pasa de 50 % a 98 % de exactitud sin recalibrar.
- **Detector de ErrP de dos o tres vistas fusionadas** (temporal con LDA encogido, geométrica de Riemann y, opcional, potencia theta en Fz y Cz), con probabilidades calibradas, umbral de Neyman-Pearson (especificidad ≥ 0.90) y detector de rareza para épocas fuera de distribución. La calibración elige por validación cruzada anidada los canales (los de su papel o los 8) y las vistas, y registra la elección.
- **Época alineada al movimiento real.** La época del ErrP se corta donde la telemetría del ESP32 dice que la órtesis empezó a moverse. Sin telemetría usa el ACK más la latencia mecánica media medida; si tampoco la hay, el ACK. La columna `alineacion` del CSV dice cuál se usó en cada paso. Perder la telemetría cuesta, pero el lazo sigue: en el gemelo (`python estudios/respaldo_ack.py 4 4 ignorar`, 16 sesiones por escenario, con los topes del recorrido) la BA viva del detector baja de 0.86 a 0.72 si nunca hubo telemetría y a 0.81 si se pierde en el lazo; el error del agente en los 2 min tras perturbar pasa de 0.33 a 0.40 y a 0.34, y se recuperan 10 y 12 de 16 sesiones en lugar de las 16.
- **Detector co-adaptativo** (encendido por defecto; `--sin-coadaptativo` lo apaga). El detector se re-entrena en otro hilo cada 20 épocas del lazo, y el modelo nuevo solo reemplaza al vigente si en una prueba en sombra de 30 épocas no es peor. Ningún fallo del re-entrenamiento detiene el lazo: queda registrado en consola y en EVALUACION, y el lazo sigue con el modelo vigente; con tres fallos seguidos se apaga solo. En el gemelo (16 sesiones por detector, con los topes del recorrido) no empeora el error del agente: −0.005 ± 0.004 con el detector fuerte, −0.024 ± 0.012 con el medio y −0.008 ± 0.006 con el débil; sesión por sesión va de −0.14 a +0.05.
- **Calibración honesta.** MI se detiene sola cuando el intervalo de confianza ya decide (mínimo 36 ensayos); el detector de ErrP usa siempre 120 épocas con el umbral elegido por validación anidada (ver el hallazgo de la maldición del ganador).
- **Hora por contador.** La hora de cada muestra se reconstruye con el contador del casco, sin el jitter de llegada por Bluetooth; una pérdida queda como un hueco visible.

Resultados en simulación (30 sujetos, perturbación de 2.4 logits, `python simulador_lazo.py --semillas 30`; medidos en Windows 11 con Python 3.11 y repetidos el 3 de octubre de 2026 con la configuración final):

| Agente | Error antes | Primeros 2 min | Después |
|---|---|---|---|
| Estático (sin aprender) | 0.165 | 0.325 | 0.316 |
| `eta` fijo 0.3 | 0.170 | 0.246 | 0.178 |
| **Bayes (el nuestro)** | **0.169** | **0.214** | **0.175** |

Con el detector de ErrP degradado a propósito, el aprendizaje baja a menos del 10 % y se congela la mayor parte de la falla. Los congelamientos en falso son de 1 %.

**Curva de robustez** (simulador, 30 sujetos, `estudios/curva_robustez.py`): cuánto aguanta el agente un detector de ErrP peor. Con BA ≥ 0.75 la recuperación ya es casi plana; por debajo se alarga rápido. El decoder estático se queda en 0.325 de error en todos los casos.

| BA del detector | Error del agente, 2 min tras perturbar | Recuperación |
|---|---|---|
| 0.65 | 0.259 | 162 s |
| 0.70 | 0.236 | 88 s |
| 0.75 | 0.219 | 65 s |
| 0.80 | 0.214 | 63 s |
| 0.85 | 0.214 | 56 s |

![Curva de robustez: error y tiempo de recuperación según la BA del detector](docs/figuras/curva_robustez.png)

## Gemelo digital del piloto

`cerebro_sintetico.py` sustituye al casco (un Unicorn Hybrid Black). Escucha las señales y los pasos del orquestador y responde como una persona: desincroniza mu/beta sobre C3 al imaginar cerrar, genera una respuesta visual occipital (N1 ≈ 170 ms en PO7/Oz/PO8) ante cada movimiento de la órtesis y un ErrP fronto-central (Ne ≈ 250 ms, Pe ≈ 350 ms en Fz/Cz/Pz) cuando va al lado contrario, parpadea, mueve la cabeza de vez en cuando (el giroscopio lo registra y el EEG se ensucia), pierde muestras por Bluetooth si se le pide y, opcionalmente, se cansa. Con él se valida el camino **real** completo con verdad conocida.

Corrida completa con la configuración final (`orquestador.py real --ortesis-sim` contra el cerebro sintético, una corrida, 3 de octubre): CP1 y CP2 en **GO** (MI BA 0.88 con 42 ensayos). El CP3 dio **NO GO** por especificidad (ErrP: sensibilidad 0.83, especificidad 0.87, BA 0.85 con 120 épocas; el criterio pide especificidad ≥ 0.90) y el lazo se corrió con ese detector. CP4 en **GO**: el agente recuperó en 28 pasos (50 s) y en los 2 min tras la perturbación quedó en 0.39 contra 0.54 de la sombra. 30 de los 138 pasos no movieron la órtesis. Es una sola corrida; la referencia con 16 sesiones por detector está en la sección del control negativo.

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
| **Configuración final: además elige canales y vistas** | 16 de 16 | 0.72 / 0.91 / **0.81** | 0.72 / 0.90 / **0.81** |

La parada secuencial revisaba cada 10 épocas y daba GO en cuanto el estimado salía alto: con pocos datos eso solo ocurre por suerte, y además el umbral de Neyman-Pearson se elegía sobre los mismos puntajes que se evaluaban. Es la maldición del ganador: se reporta el máximo de varios estimados ruidosos. En MI el efecto era menor (+0.02 de error con un mínimo de 24 ensayos; +0.00 con 36).

**La corrección.** El detector de ErrP se calibra siempre con las 120 épocas, sin GO ni NO GO tempranos, y el umbral se elige dentro de cada pliegue (validación anidada): lo reportado queda a −0.02 ± 0.04 de lo real. MI exige al menos 36 ensayos. Además, el primer paso de cada ensayo fallaba más (0.22 contra 0.18) porque su ventana empezaba con la transición mental; esperar 1 s más tras la señal lo baja a 0.15. El resto del 0.37 era ruido: un bloque estático de 30 pasos tiene desviación de 0.08 (en vivo, con 150 pasos: BA 0.83 en calibración y 0.20 de error).

**El costo honesto.** Con el montaje del Unicorn (3 electrodos fronto-centrales), el ErrP del gemelo y el detector fijo de 8 canales y dos vistas, la BA real rondaba 0.73 y el CP3 (0.75) daba GO en 1 de 16 sujetos. Antes "pasaba" por el sesgo. Con la configuración final (la calibración elige canales y vistas, y el gemelo produce theta tras el error) la BA real es 0.81, lo reportado coincide con lo real (+0.00 ± 0.04) y el CP3 da GO en 8 de 16. Parte de esa mejora viene de la theta, que programamos nosotros. Son cifras del gemelo, no de una persona.

## Control negativo: el agente aprende del ErrP, y por eso su velocidad depende del detector

**El síntoma.** En tres corridas contra el gemelo, el agente tardó de 56 a 80 pasos en recuperarse de la perturbación y en dos de ellas empató con el decoder sin aprender; en el simulador tarda unos 20.

**Lo que se midió** (`estudios/agente_lento.py`: el lazo completo del gemelo sin LSL, con la misma aritmética del orquestador; 4 sujetos × 4 lazos por variante y el mismo ruido en todas). Error del agente en los 2 minutos tras perturbar (el decoder sin aprender queda en 0.49):

| Detector (BA viva) | Salida calibrada (la de hoy) | Salida binaria | Prior a 0.5 al detectar un cambio (descartada) |
|---|---|---|---|
| El actual (0.82) | 0.280 | 0.282 | 0.246 |
| El de aquellas corridas (0.74) | 0.294 | 0.312 | 0.257 |
| Uno débil (0.68) | 0.316 | 0.360 | 0.274 |

- **No era la salida calibrada.** La binaria no es más rápida, y con detector débil es peor (+0.044 ± 0.018). La calibración de Platt sí comprimía las probabilidades del detector de aquellas corridas (pendiente de calibración 1.31 contra 1.05 del actual), pero corregir eso apenas mejora (−0.016 ± 0.012).
- **Era la evidencia.** El agente calcula la probabilidad de haberse equivocado combinando su prior de error (~0.23) con lo que dice el detector. Con ese prior, un error solo lo convence (probabilidad mayor que 0.5) si el detector es rotundo: pasa con el 65 % de los errores con el detector actual, con el 44 % con el de aquellas corridas y con el 24 % con el débil.
- **La corrección tentadora se descartó.** Subir el prior a 0.5 cuando se detecta un cambio es la variante más rápida de la tabla, pero **falla el control negativo**: si al agente se le quita la evidencia del ErrP (recibe siempre la tasa base), se recupera igual, 16 de 16 sesiones. Lo que lo mueve ahí es el detector de sesgo, que supone metas balanceadas, y no el cerebro del piloto. En el simulador, con un detector sin información, pasa lo mismo: 30 de 30 sujetos.
- **El agente de hoy pasa ese control.** Sin la evidencia del ErrP casi no se recupera: 4 de 16 sesiones contra 13 de 16 con el detector débil, y su error sube +0.121 ± 0.019 (medido con todo paso visible; con los topes del recorrido, más abajo, es 1 de 16 contra 10 a 16). Por eso no se cambió: su velocidad depende de la calidad del detector, como muestra la curva de robustez, y eso es justo lo que afirma el proyecto.

![Agente lento: β tras la perturbación, control negativo, P_hat de los errores y confiabilidad del detector](docs/figuras/agente_lento.png)

Son cifras del gemelo, no de una persona. Parte de la mejora del detector actual viene de la actividad theta tras el error, que programamos nosotros.

### Los pasos que no mueven la órtesis (3 de octubre, tarde)

**El hueco.** Cada ensayo empieza en el punto medio y da 5 pasos de hasta 0.30 del recorrido. Con un decoder seguro la órtesis llega al tope en el segundo paso y los siguientes no la mueven; tras la perturbación pasa lo mismo hacia el lado equivocado. En las sesiones contra el gemelo eso era del 14 al 28 % de los pasos. Una persona no ve nada en esos pasos y no produce ErrP, pero el gemelo reaccionaba al marcador del paso aunque la órtesis no se moviera. Por eso el estudio de arriba (y todas las corridas anteriores) eran optimistas, y el lazo tenía un defecto que el gemelo tapaba: leía esos pasos como "sin ErrP", lo que refuerza la decisión y cuenta como detección fallida.

**Lo que se midió** (`estudios/paso_sin_movimiento.py`: el mismo lazo sin LSL, con los topes del recorrido y un gemelo que no reacciona a lo que no se mueve; 16 sesiones por celda). Error del agente en los 2 minutos tras perturbar y sesiones que se recuperan en ese plazo (el decoder sin aprender queda en 0.49):

| Detector (BA viva) | Todo paso se ve (estudio de arriba) | Con topes, el lazo lee esos pasos (antes) | Con topes, el lazo los ignora (**hoy**) | Hoy, sin la evidencia del ErrP |
|---|---|---|---|---|
| Fuerte (0.83) | 0.280, 16 de 16 | 0.409, 9 de 16; congelado 21 % | **0.319, 16 de 16**; congelado 0 % | 0.463, 1 de 16 |
| Medio (0.73) | 0.294, 15 de 16 | 0.376, 11 de 16; congelado 13 % | **0.389, 10 de 16**; congelado 5 % | 0.467, 1 de 16 |
| Débil (0.67) | 0.316, 13 de 16 | 0.391, 7 de 16; congelado 26 % | **0.400, 11 de 16**; congelado 5 % | 0.469, 1 de 16 |

- **Un tercio de los pasos no informa** (31 a 36 % tras la perturbación), y eso cuesta: con el detector fuerte la recuperación pasa de 19 a 28 pasos, y con los otros dos se recuperan 10 u 11 sesiones de 16 en lugar de 13 a 15.
- **La corrección ayuda donde el detector es bueno.** Con el detector fuerte, leer esos pasos congelaba el aprendizaje 21 % del tiempo y solo se recuperaban 9 sesiones de 16; ignorándolos se recuperan las 16 y el error baja de 0.41 a 0.32. Con detector medio o débil el error no cambia (+0.01, dentro del error estándar de ±0.02); lo que cambia es que el aprendizaje ya casi no se congela en falso.
- **El control negativo se sostiene, y queda más limpio:** sin la evidencia del ErrP se recupera 1 sesión de 16 con cualquiera de los tres detectores; con ella, 10 a 16 de 16.

![Control negativo con la configuración final](docs/figuras/control_negativo.png)

Sigue siendo el gemelo, no una persona. El simulador rápido (`simulador_lazo.py`, la curva de robustez) no modela los topes: ahí todo paso informa, y por eso recupera antes.

## Control causal en vivo: bloque real contra bloque sham (`--sham`)

El control negativo de arriba se midió fuera de línea. `--sham` lo lleva a la sesión, frente al jurado. En lugar del bloque adaptativo corren **dos bloques del mismo largo, uno real y uno sham, en orden al azar**:

- Cada bloque (80 pasos) arranca con el agente reiniciado (beta, varianza y prior) y sin perturbación, y recibe la suya (2.4 logits) en el mismo paso (el 10, tras dos ensayos).
- En el bloque sham el agente aprende igual de rápido (fiabilidad fija en la calibrada, sin congelar), pero **no recibe la evidencia del ErrP**: recibe la tasa base de la calibración, que no dice nada del paso. El `ConfianzaDetector` sigue midiendo al detector, así que la BA viva se puede comparar entre bloques.
- **Ciego simple:** el piloto no sabe cuál bloque es cuál. La consola y el tablero dicen «A» y «B»; el tablero muestra cuál es el real solo cuando el operador pulsa *Revelar bloques*.
- El orden queda en el CSV (columna `bloque`) y en los marcadores `bloque:real` y `bloque:sham`. `EVALUACION` y el tablero comparan los bloques lado a lado: error tras perturbar con intervalo del 90 %, tiempo de recuperación y si se recuperó. El CP4 es el del bloque real.

```bash
python orquestador.py real --puerto COM4 --sham      # 80 pasos por bloque: unos 5.6 minutos los dos
python estudios/sham_gemelo.py                       # el criterio de aceptación en el gemelo (~2 min)
```

### Qué recibe el agente en el sham: tres candidatos, uno sirve

Medido en el gemelo sin LSL y con los topes del recorrido (`estudios/sham_gemelo.py`; 4 sujetos × 4 sesiones, 80 pasos por bloque, perturbación en el paso 10, con los tres detectores del control negativo). **Es el gemelo, no una persona.**

| Detector | Qué recibe el agente en el sham | Real se recupera | Sham se recupera | Error tras perturbar, real / sham | Sham − real (IC 90 %) |
|---|---|---|---|---|---|
| actual | **sin evidencia del ErrP** (`nula`, por defecto) | **16/16** | **1/16** | 0.296 / 0.471 | **+0.175 [+0.154, +0.196]** |
| actual | `p_errp` del bloque, permutados (`recientes`) | 16/16 | 15/16 | 0.306 / 0.356 | +0.050 [+0.029, +0.072] |
| actual | «sham ciego»: `p_errp` de la calibración (`calibracion`) | 16/16 | 7/16 | 0.300 / 0.426 | +0.126 [+0.095, +0.156] |
| de ayer | sin evidencia del ErrP | 15/16 | 0/16 | 0.333 / 0.458 | +0.125 [+0.092, +0.160] |
| de ayer | permutados | 15/16 | 12/16 | 0.322 / 0.347 | +0.025 [−0.017, +0.068] |
| de ayer | sham ciego | 16/16 | 5/16 | 0.318 / 0.426 | +0.108 [+0.062, +0.153] |
| débil | sin evidencia del ErrP | 11/16 | 1/16 | 0.393 / 0.454 | +0.062 [+0.030, +0.093] |
| débil | permutados | 10/16 | 10/16 | 0.385 / 0.389 | +0.004 [−0.033, +0.046] |
| débil | sham ciego | 11/16 | 2/16 | 0.385 / 0.429 | +0.044 [+0.008, +0.080] |

El criterio de aceptación (real ≥ 12 de 16, sham ≤ 3 de 16 y una diferencia de error cuyo intervalo excluye el 0) **se cumple con el sham sin evidencia y los detectores actual y de ayer**. Con el detector débil el propio bloque real se queda en 11 de 16. Con bloques de 60 pasos las cifras eran casi las mismas (real 16/16, sham 0/16, +0.136 [+0.111, +0.161] con el detector actual).

### Hallazgo: un sham que conserva la tasa de ErrP no es un sham

El primer diseño era el más elegante: darle al agente los `p_errp` del mismo bloque **permutados** entre los pasos recientes. Misma distribución, ninguna relación con el error de cada paso. Con él, el bloque sham se recupera en 15 de 16 sesiones. No es un fallo de la permutación: es que **la permutación no quita la información que el agente usa**.

1. **El agente corrige una sola cosa:** `beta`, el sesgo del decoder. En cada paso mueve `beta` según `g = q − p'`, donde `p'` es lo seguro que estaba de «cerrar» y `q` lo que cree tras el ErrP. Si decidió «abrir» y el ErrP dice «error», `q` sube y `beta` va hacia «cerrar»; si decidió «cerrar» y hay ErrP, al revés.
2. **Con decisiones repartidas entre los dos lados, el orden importa.** Un ErrP asignado al paso equivocado empuja al lado equivocado la mitad de las veces, y permutar los `p_errp` los cancela entre sí.
3. **Tras la perturbación las decisiones no están repartidas.** El decoder perturbado dice «abrir» casi siempre, con mucha seguridad. Entonces *todos* los ErrP, caigan en el paso que caigan, empujan a `beta` hacia el mismo lado: el correcto. Permutarlos no cambia la suma.
4. **Lo que queda es la tasa.** Con las metas balanceadas, la mitad de esos «abrir» son errores: la tasa de ErrP sube de ~20 % a ~50 %. Un agente bayesiano lee eso bien: «estoy muy seguro y me equivoco la mitad de las veces; mi sesgo está mal». Esa lectura no necesita saber *en qué paso* estuvo cada error.

Por eso lo que el agente aprende del ErrP es, sobre todo, **cuántos hay**; la alineación paso a paso aporta menos (0.05 de error entre el bloque real y el permutado). Y por eso un sham que conserve la tasa de ErrP del lazo deja pasar la información.

El **«sham ciego»** lo confirma por el otro lado. Recibe `p_errp` sacados al azar de los de la calibración: su distribución, a su tasa de error (30 %), sin ninguna relación con el lazo actual. No sabe nada de la perturbación y aun así se recupera en 7 de 16 (5 y 2 de 16 con los otros detectores): con las decisiones cargadas a un lado, **cualquier** flujo de detecciones empuja a `beta` hacia el otro, y uno con la tasa de un 30 % de errores empuja bastante. Solo el sham **sin evidencia** (LLR = 0: `P_hat` se queda en el prior) deja al agente quieto, con 0 a 1 de 16.

La conclusión para el control: el contraste limpio no es «ErrP ordenados contra desordenados», sino **«con la evidencia del ErrP contra sin ella»**. El sham permutado sigue disponible para mostrarlo (`--sham-fuente recientes`); el ciego se descartó para el lazo y queda en el estudio.

### Prior de error por paso (encendido por defecto desde el 4 de octubre, ε = 0.10): la frontera del sham ciego

Hipótesis (4 de octubre): el sham ciego se recupera 7 de 16 porque, con el agente muy seguro y equivocado, `P_hat` promedia el **prior global** de error y el gradiente empuja `beta` sin información real. Se probó dar a cada paso como prior el error que el propio agente predice, `1 − max(p', 1 − p')`, con un piso ε (`ConfigAgente.prior_por_paso` y `piso_prior`; ε = `None` es la tasa global, el prior global vivo del agente). **Por decisión de Luis (4 de octubre) es el defecto: `config.PRIOR_POR_PASO = True`, `PISO_PRIOR_PASO = 0.10`**, con la condición de que la mediana de pasos hasta recuperarse no empeorara más de 25 % (medido abajo). Se apaga con `ConfigAgente(prior_por_paso=False)`.

Medido en el gemelo sin LSL (`python estudios/prior_por_paso.py`: 4 sujetos × 4 sesiones, 80 pasos por bloque, perturbación en el paso 10, detector actual). **Es el gemelo, no una persona.**

| ε | Real se recupera | Sham ciego se recupera | Sham − real, ciego (IC 90 %) | Control negativo: sin ErrP, recupera en 2 min |
|---|---|---|---|---|
| sin la bandera (prior global) | 16/16 | 7/16 | +0.126 [+0.095, +0.156] | 3/16 |
| 0 | 16/16 | 1/16 | +0.170 [+0.140, +0.196] | 0/16 |
| 0.05 | 16/16 | 2/16 | +0.183 [+0.154, +0.210] | 0/16 |
| **0.10** | 16/16 | 2/16 | +0.158 [+0.127, +0.191] | 0/16 |
| 0.15 | 16/16 | 4/16 | +0.152 [+0.124, +0.179] | 0/16 |
| 0.20 | 16/16 | 7/16 | +0.140 [+0.100, +0.179] | 8/16 |
| tasa global | 16/16 | 11/16 | +0.109 [+0.074, +0.145] | 16/16 |

- **La frontera está entre ε = 0.10 y 0.15:** con ε ≤ 0.10 se cumple el criterio de `TAREAS.md` también con el sham ciego (real ≥ 12, sham ≤ 3, intervalo sin el 0) y el control negativo da 0 de 16. Con 0.15 el ciego llega a 4 de 16 (justo fuera). Desde 0.20 el piso vuelve a mezclar el prior global y el control negativo falla (8 de 16, y 16 de 16 con la tasa global): **la tasa global como piso está descartada.**
- **El control negativo es obligatorio** (`CLAUDE.md`): se midió en el lazo de `estudios/agente_lento.py` (todo paso visible, sin topes), con el agente recibiendo siempre la tasa base (LLR = 0). Sin la bandera dio 3 de 16 (la referencia del README decía 4 de 16: varía entre corridas). Con la bandera y ε ≤ 0.15, 0 de 16. Con el sham `nula` del orquestador (el de la demo) el sham se recupera 0 de 16 con ε ≤ 0.20.
- **Costo con detectores peores** (ε entre 0 y 0.10): con el detector de ayer el real se recupera 12 a 15 de 16 (sin la bandera, 15 de 16) y el sham ciego 0 a 1 de 16 (antes 5); con el débil el real baja de 11 a 9–10 de 16 con ε ≤ 0.05 y queda en 10 a 12 de 16 con 0.10 (ese detector ya no cumplía el criterio). Con ε = 0.10 el bloque real queda a una sesión o menos de la base en los tres detectores. Las diferencias de 1 a 2 sesiones sobre 16 no se distinguen del azar.
- **Qué dice esto del mecanismo:** consistente con la hipótesis, pero no la prueba. Con ε bajo, un ErrP en una decisión muy segura del agente se lee como probable falsa alarma y empuja poco; solo un ErrP que sigue llegando con la perturbación (el real) la contrarresta. Queda sin medir si el costo de ε bajo aparece con una persona (el ErrP real puede ser más ruidoso que el del gemelo).
- El sham ciego (`calibracion`) sigue solo en `estudios/sham_gemelo.py`; el del orquestador sigue siendo `nula`.

#### Velocidad de recuperación con el prior por paso (ε = 0.10) contra sin bandera

Bloque real, 16 sesiones de 80 pasos, detector actual, gemelo (`python estudios/prior_por_paso.py velocidad`). Pasos hasta que `beta` llega al 70 % de la perturbación. Son las mismas 16 sesiones medidas en dos corridas (con el sham `calibracion` y con el `nula`; solo cambia el orden de los bloques y el azar del sham).

| Corrida | Recuperan (sin bandera → con ε 0.10) | Mediana de pasos | Percentil 90 | Cociente de medianas (IC 90 %) |
|---|---|---|---|---|
| con sham ciego | 16/16 → 16/16 | 24.0 → 29.0 | 37.0 → 45.5 | ×1.21 [0.94, 1.49] |
| con sham `nula` | 16/16 → 16/16 | 28.0 → 27.5 | 40.5 → 49.0 | ×0.98 [0.72, 1.38] |
| las dos juntas (32) | | 25.5 → 29.0 | 38.0 → 49.1 | ×1.14 [0.89, 1.35] |

La mediana no empeora más del 25 % en ninguna de las tres lecturas (el peor punto es +21 %), pero **el intervalo llega hasta ×1.49: con 16 sesiones no se descarta una pérdida mayor**, y el percentil 90 sí sube (de 38 a 49 pasos, unos 25 s a ciclo nominal). En el simulador de rasgos (`simulador_lazo.py`, 30 sujetos, salida binaria) el error en los primeros 2 min tras perturbar pasa de 0.214 a 0.221 (±0.05): no se distingue.

#### Tabla del control causal con el defecto nuevo (ε = 0.10)

Gemelo sin LSL, 4 sujetos × 4 sesiones, 80 pasos por bloque, perturbación en el paso 10. «Sin ErrP» es el control negativo de `agente_lento.py` (todo paso visible, sin topes). **Gemelo, no una persona.**

| Detector | Real se recupera | Sham `nula` (el de la demo) | Sham ciego | Sin ErrP (control negativo) |
|---|---|---|---|---|
| actual | 16/16 | 0/16 (+0.184 [+0.155, +0.212]) | 2/16 (+0.158 [+0.127, +0.191]) | 0/16 (error 0.471; sombra 0.487) |
| de ayer | 15/16 | 0/16 (+0.164 [+0.138, +0.190]) | 0/16 (+0.124 [+0.087, +0.160]) | no medido |
| débil | 12/16 con `nula`, 10/16 con ciego | 0/16 (+0.078 [+0.054, +0.101]) | 1/16 (+0.057 [+0.035, +0.079]) | no medido |

Entre paréntesis, sham − real (error tras perturbar). Antes del defecto nuevo, con el detector actual: real 16/16, `nula` 1/16, ciego 7/16, sin ErrP 3/16 (tabla de arriba). No se midió el sham permutado (`--sham-fuente recientes`) con el defecto nuevo, y las cifras de referencia de `CLAUDE.md` (corrida completa contra el gemelo, caos, `paso_sin_movimiento`) son anteriores a este cambio y no se repitieron.

### Lo que una sola sesión puede mostrar

Tras la perturbación quedan 70 pasos por bloque: la diferencia de error de una sesión tiene un intervalo de ±0.3 y casi nunca excluye el 0. Lo que se ve en vivo es si beta se recuperó en un bloque y no en el otro. En el simulador rápido el contraste de error es menor (la perturbación sube el error de la sombra a 0.30, no a 0.50), y ahí solo se comprueba la recuperación: 8 de 12 en el real contra 2 de 12 en el sham (`pruebas.py`, `orquestador_sham`).

**En vivo contra el gemelo, por LSL (4 sesiones reales completas con `--ortesis-sim`; pocas, y una sola calibración sirvió para tres de ellas).** El sham no se recuperó en ninguna. El bloque real se recuperó en 3 de 4:

| Sesión | Pasos por bloque | Detector (CP3) | Bloque real | Bloque sham |
|---|---|---|---|---|
| 1 | 60 | BA 0.89, GO | **no se recuperó** en los 50 pasos tras perturbar (beta llegó a 1.44 de 1.86) | no se recuperó |
| 2 | 80 | BA 0.72, NO GO (forzado) | se recuperó en 25 pasos (35 s) | no se recuperó |
| 3 | 80 | el de la sesión 2 | se recuperó en 46 pasos (69 s) | no se recuperó |
| 4 | 80 | el de la sesión 2 | se recuperó en 35 pasos (50 s) | no se recuperó |

En la sesión 1 solo 15 de los 50 pasos tras perturbar tuvieron una época útil: 27 no movieron la órtesis (con un decoder muy seguro llega al tope en dos pasos) y 8 fueron artefacto. Con 60 pasos por bloque el margen en vivo era justo (la recuperación de la sesión 3, a los 46 pasos, habría entrado por cuatro pasos). **Por eso el valor por defecto es ahora 80 pasos por bloque** (decisión de Luis para la final): quedan 70 pasos tras perturbar y los dos bloques duran unos 5.6 minutos a 2.1 s por paso.

**Crédito.** La idea de un bloque sham dentro de la sesión es de jusren. Su sham (`bloque_sham.py`, más abajo) es otro control: con el piloto en reposo la órtesis se mueve sola y `p(t)` del decoder no debe seguirla. La implementación de `--sham` es distinta y propia.

## Copiloto clínico (`copiloto.py`)

Preguntas en lenguaje natural sobre una sesión, respondidas **solo con sus datos**. El copiloto no ve el EEG: consulta cinco herramientas que leen el CSV de la sesión y su registro `_estado.jsonl` (eventos del flujo `Estado`, checkpoints y marcadores):

| Herramienta | Qué devuelve |
|---|---|
| `resumen_sesion()` | Pasos, error del agente y de la sombra por bloque, recuperación, excluidos, BA viva, congelamientos, cambios y checkpoints. |
| `eventos(desde, hasta, tipo)` | Pausas (con su causa), congelamientos, cambios detectados, perturbaciones, checkpoints y cambios de semáforo. |
| `metrica(nombre, bloque)` | Error con intervalos y la diferencia sombra − agente, tiempo de recuperación, BA viva, fiabilidad, latencia del ACK, excluidos por motivo, alfa occipital y errores sin ErrP según el tamaño del paso. |
| `pasos(desde, hasta, columnas)` | Filas puntuales del CSV, 40 como máximo por llamada. |
| `comparar_sesiones(rutas)` | Las cifras clave de esta sesión junto a las de otras (por defecto, la anterior). |

Toda respuesta cita los pasos y los valores que usó. Si el dato no existe, la herramienta lo dice («no hay dato: …») y el copiloto lo repite; nunca estima.

```bash
python copiloto.py --ultima "¿cuánto tardó en recuperarse tras la perturbación?"
python copiloto.py --sesion resultados/sesion_real_....csv          # modo interactivo
python tablero.py --copiloto                                        # caja de preguntas en el tablero
python copiloto.py --ultima --informe                               # informe entre sesiones
python copiloto.py --ultima --decidir aprobar                       # la propuesta la decide una persona
```

**Con y sin IA.** Con la llave en un archivo `.env` (`ANTHROPIC_API_KEY=...`; está en `.gitignore`) responde Claude (`claude-opus-5-5`) llamando a esas herramientas. Sin llave, sin conexión, o si la API falla o declina, responden plantillas deterministas sobre las mismas herramientas, y la respuesta lo dice. Las seis preguntas de prueba se responden de las dos formas: por qué se congeló el aprendizaje en el paso N, cuánto tardó en recuperarse, si el agente le ganó a la sombra y con qué certeza, si hubo señales de fatiga, cuántos pasos se excluyeron y por qué, y cómo se compara con la sesión anterior.

**Informe entre sesiones.** `--informe` compara la sesión con las anteriores (las tres previas del mismo tipo, o `--anteriores`) y escribe cuatro Markdown junto al CSV: para el terapeuta (cifras, tabla comparativa y propuesta) y para el paciente (lenguaje sencillo), en español y en inglés, con dos figuras. Las cifras siempre salen del código; con IA, Claude escribe solo el párrafo de interpretación. El informe incluye una **propuesta para la próxima sesión** en JSON, con el mismo esquema y los mismos rangos seguros que el co-investigador; queda pendiente en `sesion_..._propuestas.jsonl` hasta que una persona la apruebe o la rechace.

**Qué sale hacia la API.** Solo lo que devuelven las herramientas, y todo pasa por `ia.sanear`: quita las carpetas de las rutas (llevan el nombre de usuario de la máquina) y se niega a enviar una lista de más de 200 números, que sería una señal cruda. Los archivos de sesión no tienen nombres de personas.

**Sin probar con la API real:** el 3 de octubre no había llave en la máquina. Todo está probado con una API simulada (`pruebas.py`: `copiloto_herramientas` y `copiloto_api_simulada`); la primera llamada real hay que hacerla antes de encenderlo en la final.

## Co-investigador entre bloques (`--coinvestigador`)

La escala lenta del aprendizaje. Al terminar cada bloque, el orquestador:

1. **Resume el bloque** en cifras agregadas (`copiloto.resumen_para_propuesta`): exactitud, error contra la sombra con el intervalo de la diferencia, BA viva, fiabilidad media y fracción del bloque con el aprendizaje congelado, excluidos por motivo, alfa occipital, pasos sin movimiento y errores que pasaron sin ErrP según el tamaño del paso.
2. **Pide una propuesta** a Claude con esquema fijo (`ia.ESQUEMA_PROPUESTA`): una acción (`continuar`, `pausa`, `ajustar_parametro` o `recalibrar`), un parámetro y un valor si aplica, y una justificación breve. Sin llave, sin conexión o si la API tarda más de 20 s, propone un conjunto de reglas deterministas (`ia.propuesta_por_reglas`) con el mismo esquema.
3. **La valida** contra los rangos seguros de `config.PARAMETROS_PROPUESTA`. Una propuesta de la API que no cumple el esquema o se sale del rango se descarta (queda registrada con el motivo) y deciden las reglas.
4. **Se la muestra al operador** en el tablero, con los botones **Aprobar** y **Rechazar** (también vale `python copiloto.py --sesion <csv> --decidir aprobar` desde otra terminal). Espera la decisión hasta 60 s.
5. **Aplica el efecto solo si se aprobó**, y registra propuesta, decisión y efecto en `sesion_..._propuestas.jsonl`.

| Parámetro ajustable | Rango seguro | Qué es |
|---|---|---|
| `paso_visible` | 0.05 a 0.12 | El paso mínimo que se le muestra al piloto. |
| `paso_max` | 0.15 a 0.35 | El paso máximo: nunca más de un tercio del recorrido. |
| `ganancia` | 0.15 a 0.45 | Cuánto crece el paso con la confianza del decoder. |
| `ajenos_cada` | 0 a 20 | Movimientos ajenos de la Tarea 2 (0 = ninguno). |
| `pausa_s` (solo en `pausa`) | 30 a 300 s | Descanso con la órtesis abierta. |

**Lo que no puede pasar.** Nada cambia sin una aprobación: rechazada, o sin decisión a tiempo, la sesión sigue igual. No hay parámetros del aprendizaje del agente (prior, varianza, umbrales de cambio) en la lista. Entre los dos bloques del control causal no se ajusta nada aunque se apruebe (cambiaría las condiciones de la comparación), y ahí el resumen y el tablero solo dicen «bloque A» o «bloque B». `recalibrar` aprobado termina la sesión en orden y dice cómo relanzarla (`--solo-errp`). Tras el último bloque la propuesta se registra sin esperar decisión. La consulta ocurre entre bloques: dentro de un paso del lazo no corre nada de la IA.

```bash
python orquestador.py real --puerto COM4 --coinvestigador
```

Las reglas deterministas, en orden: pocos pasos válidos (< 30) → continuar; más de 25 % de filas excluidas → pausa para revisar el casco y la órtesis; alfa occipital ≥ 1.5 veces su línea base → pausa; BA viva < 0.60 o más de la mitad del bloque congelado → recalibrar; errores con pasos chicos que pasan sin ErrP 25 puntos más que con pasos grandes → subir `paso_visible` 0.02; si no, continuar. Los umbrales están en `config.REGLAS_PROPUESTA` y **no están validados con personas**.

Probado en el simulador con la API simulada (`pruebas.py`, `coinvestigador_entre_bloques`): aprobada, `paso_visible` pasa de 0.08 a 0.10 y el lazo lo usa desde el bloque siguiente; rechazada o sin decisión, nada cambia; una propuesta de `paso_max = 0.95` se descarta. **Sin probar con la API real.**

## El cerebro enseñando a la máquina: flechas de ErrP (`tablero.py --flechas`)

```bash
python tablero.py --flechas           # apagado por defecto
```

Un sexto panel dibuja **una flecha por paso**: su dirección es hacia dónde movió ese paso a `beta` (arriba = hacia CERRAR, abajo = hacia RELAJAR) y su largo, cuánto. Violeta cuando el detector marcó ese paso como ErrP, gris cuando no (la ausencia de ErrP también enseña, menos). Sobre el panel, una línea cuenta el último paso en palabras. Tras la perturbación se ve la ráfaga de flechas violetas hacia el mismo lado que va corrigiendo el agente.

- **Es lo que el agente hizo, no una interpretación:** la flecha es la diferencia de `beta` entre dos pasos consecutivos del flujo `Estado` (el contrato no cambia). Un paso sin época útil, o con el aprendizaje congelado, no dibuja flecha.
- **El cambio de `beta` junta todo lo que el agente usó de ese paso** (ErrP, prior y confianza del detector); no mide la amplitud del ErrP en microvoltios.
- **Respeta el ciego del control causal:** en una sesión `--sham` el panel queda en blanco hasta pulsar *Revelar bloques*, porque el tamaño de las flechas (grandes en el bloque real, casi nulas en el sham `nula`) delataría cuál es cuál.
- Probado sin pantalla en `pruebas.py` (`tablero_flechas`). **Todavía no se ha visto con una sesión en vivo.**

## Narrador para el jurado (`narrador.py`)

Un proceso aparte que escucha el flujo `Estado` y, cuando pasa algo que vale la pena contar, publica **una frase corta** en español o en inglés: checkpoints, la perturbación, la recuperación, el congelamiento del aprendizaje (y cuando se reanuda), las pausas seguras con su motivo (y cuando terminan) y el resultado del control causal.

```bash
python narrador.py --idioma en        # terminal aparte; imprime cada frase y la publica en el flujo LSL `Narracion`
python tablero.py --narrador          # franja con la última frase
```

- **Nunca bloquea el lazo:** es otro proceso y solo lee; el orquestador no sabe que existe.
- **Basado solo en los datos del evento.** Con llave, Claude recibe el evento (un puñado de cifras) y escribe la frase. Se descarta, y se usa la plantilla, si la API tarda más de 6 s, falla o declina, si la frase pasa de 220 caracteres, o **si trae una cifra que no está en el evento**. Un evento con más de 8 s de atraso ya no se le pregunta a la API: una frase tardía confunde.
- **Sin llave o sin conexión**, plantillas en los dos idiomas (`--sin-ia` las fuerza).
- **Respeta el ciego del control causal:** durante los bloques A y B no cuenta recuperaciones ni congelamientos, que delatarían cuál es el real; al final cuenta el resultado.

Probado con la API simulada (`pruebas.py`, `narrador_jurado`) y, con plantillas, contra una sesión del simulador en vivo por LSL. **Sin probar con la API real.**

## Transferencia desde PhysioNet: arrancar la calibración con un decoder pre-entrenado (`--preentrenado`)

¿Cuántos ensayos de calibración de imaginación motora ahorra empezar con un decoder entrenado con otras personas? Se midió con la **EEG Motor Movement/Imagery Database** (Schalk et al. 2004; PhysioNet, vía `mne.datasets.eegbci`): «mano derecha imaginada» contra «reposo» (corridas 4, 8 y 12), con los 8 canales del Unicorn remuestreados a 250 Hz y la misma ventana y banda del lazo. El decoder es el del proyecto (covarianzas → recentrado riemanniano → espacio tangente → regresión logística). El recentrado es lo que permite transferir: cada persona queda centrada en la identidad, así que un clasificador ajustado con otras se le aplica tal cual; la persona nueva solo aporta su centro, que sale de EEG **sin etiquetas** (su minuto de reposo, o sus propios ensayos).

Dejando una persona fuera cada vez (40 personas), calibrando con los primeros *n* ensayos de sus corridas 4 y 8 y probando siempre en su corrida 12 (`python estudios/transferencia_physionet.py`; media ± error estándar):

| Ensayos propios | Desde cero (hoy) | Pre-entrenado, solo recentrado | Pre-entrenado + ensayos propios | Diferencia con desde cero |
|---|---|---|---|---|
| 0 | — | **0.623 ± 0.021** | — | — |
| 4 | 0.610 ± 0.024 | 0.635 ± 0.021 | 0.655 ± 0.021 | +0.044 ± 0.023 |
| 8 | 0.646 ± 0.024 | 0.658 ± 0.021 | 0.689 ± 0.022 | +0.042 ± 0.023 |
| 12 | 0.666 ± 0.026 | 0.658 ± 0.023 | **0.704 ± 0.020** | +0.037 ± 0.021 |
| 16 | 0.678 ± 0.027 | 0.662 ± 0.022 | 0.706 ± 0.022 | +0.028 ± 0.023 |
| 20 | 0.678 ± 0.025 | 0.661 ± 0.022 | 0.703 ± 0.022 | +0.025 ± 0.020 |
| 24 | 0.697 ± 0.025 | 0.667 ± 0.023 | 0.707 ± 0.024 | +0.010 ± 0.020 |
| 28 | 0.706 ± 0.023 | 0.663 ± 0.024 | 0.716 ± 0.026 | +0.009 ± 0.020 |

![Transferencia desde PhysioNet](docs/figuras/transferencia_physionet.png)

**Lo que dice, sin adornos.** Funciona, pero ayuda poco. Sin un solo ensayo propio el decoder pre-entrenado ya decide con BA 0.62, lo que desde cero pide 8 ensayos. Con 12 ensayos propios llega a 0.70, lo que desde cero pide 28: **ahorra unos 16 ensayos, cerca de minuto y medio de calibración**. La ventaja es de unos 0.04 de BA con 4 a 12 ensayos (menos de dos errores estándar en cada punto) y desaparece hacia los 28. No es un atajo para saltarse la calibración: es un mejor punto de partida cuando hay pocos ensayos.

**Lo que no dice.** Son personas reales, pero no el piloto, ni el Unicorn (ahí son electrodos de gel de un equipo de 64 canales), ni la tarea exacta del proyecto (allí el «reposo» es el descanso entre ensayos, no «imagina que abres y relajas»). En el gemelo el decoder de PhysioNet, sin ensayos propios, da BA 0.79 en 8 sujetos (0.70 a 0.86): solo dice que el patrón aprendido de personas es el que programamos en el gemelo.

**Cómo se usa.** Apagado por defecto. El modelo no va en el repositorio (`modelos/` está en `.gitignore`): se genera una vez, con conexión.

```bash
python estudios/transferencia_physionet.py           # descarga 40 personas (~350 MB, lento) y mide
python estudios/transferencia_physionet.py modelo    # guarda modelos/decoder_preentrenado.pkl
python orquestador.py real --puerto COM4 --preentrenado
```

Con `--preentrenado`, la calibración de MI ajusta el clasificador con los ensayos de las otras personas más los del piloto (cada uno pesa como 20 de los otros) y usa los 8 canales, sin elegir entre C3/Cz/C4 y los 8. La BA que reporta el CP2 sigue siendo de validación cruzada sobre los ensayos del piloto, y el mínimo de 36 ensayos no cambia: bajarlo (`--min_mi`) acorta la calibración, pero el estimado con pocos ensayos vuelve a ser optimista (ver «la calibración secuencial inflaba la exactitud»).

## Memoria entre sesiones: la segunda sesión del mismo piloto calibra más corto (`--desde-sesion`)

Apagado por defecto. Una sesión puede dejar junto a su CSV lo que aprendió, y la siguiente **del mismo piloto** arranca de ahí en lugar de empezar de cero:

```bash
python orquestador.py real --puerto COM4 --guardar-memoria     # al terminar: resultados/sesion_real_<fecha>_memoria.pkl
python memoria.py                                              # lo mismo para una sesión ya corrida sin la bandera
python orquestador.py real --puerto COM4 --desde-sesion resultados/sesion_real_<fecha>.csv
```

| Pieza | Qué se guarda | Cómo arranca la sesión siguiente |
|---|---|---|
| Decoder de MI | Los ensayos de calibración como rasgos recentrados con el centro de **su** sesión (el formato del decoder pre-entrenado) | 12 ensayos en lugar de 36 a 60 (`config.MEMORIA_ENSAYOS_MI`), de largo fijo. Su centro, que no usa las etiquetas, recentra el decoder; el clasificador se ajusta con los rasgos previos más los de hoy, que pesan el triple. El lazo sigue recentrando con cada ventana, como siempre. |
| Detector de ErrP | El detector y sus épocas con etiqueta (calibración y, con co-adaptación, las del lazo) | 40 épocas en lugar de 120 (`config.MEMORIA_EPOCAS_ERRP`). Se ajusta con las previas más las de hoy, con los canales y vistas que eligió la sesión previa. |
| Agente | Su estado completo y la `beta` de **antes** de la perturbación | `beta` arranca en la de antes de la perturbación. Nunca en la final: esa incluye la corrección de una perturbación artificial de 2.4 logits que la sesión siguiente no tiene. Cómo aprende el agente no cambia. |

**Los checkpoints no se heredan.** El CP2 es la BA de validación cruzada sobre los 12 ensayos de hoy, y el CP3 se mide solo con las 40 épocas de hoy: las de la sesión previa entrenan, pero nunca caen en un pliegue de prueba (`memoria.detector_con_memoria`). Con tan pocos ensayos el intervalo es ancho, y se imprime. Como el largo es fijo y no hay parada temprana, el estimado no se infla por suerte.

`python memoria.py` arma la memoria de una sesión que no la guardó (la del domingo, corrida con `v-demo`) con lo que quedó en `modelos/` y `resultados/`. Ojo: `modelos/` guarda la última calibración, sea de quien sea; hay que armarla antes de que otra calibración la pise.

### Cuántos ensayos ahorra (`estudios/memoria_sesiones.py`)

Validación honesta, igual con todas las fuentes: la sesión previa solo aporta la memoria; de la sesión nueva, los primeros *n* ensayos calibran y los **últimos** prueban, y los de prueba nunca ajustan nada (ni el centro, ni el clasificador, ni la elección de canales, ni el umbral). El peso y los largos se fijaron antes de medir.

**Con los datos reales del piloto: pendiente.** En esta máquina no había ninguna calibración real cuando se escribió esto (4 de octubre, mediodía). En cuanto haya dos calibraciones del mismo piloto:

```bash
python estudios/memoria_sesiones.py lista      # las calibraciones guardadas, con su hora
python estudios/memoria_sesiones.py reales resultados/calibracion_mi_<previa>.npz resultados/calibracion_mi_<nueva>.npz \
       resultados/calibracion_errp_<previa>.npz resultados/calibracion_errp_<nueva>.npz
```

Una pareja de sesiones es una sola medición: el programa la reporta con el intervalo de sus ensayos de prueba y la etiqueta como exploratoria.

**Con personas reales que no son el piloto** (EEGMMIDB, las 40 del estudio de transferencia; memoria = corrida 4, calibración = primeros *n* ensayos de la corrida 8, BA en la corrida 12; media ± error estándar). Las tres corridas son del mismo día y sin quitarse el gorro, así que mide si la memoria ayuda entre corridas, no entre días:

| Ensayos de hoy | Desde cero | Con memoria | Decoder previo sin recalibrar | Memoria − cero (pareada) |
|---|---|---|---|---|
| 0 | — | **0.689 ± 0.024** | 0.672 ± 0.027 | — |
| 8 | 0.646 ± 0.027 | 0.694 ± 0.026 | 0.672 ± 0.027 | +0.047 ± 0.016 |
| 12 | 0.689 ± 0.025 | 0.715 ± 0.026 | 0.672 ± 0.027 | +0.026 ± 0.015 |

Lectura: sin ningún ensayo de hoy, la memoria (recentrada con 12 s de EEG sin etiquetas) da lo mismo que calibrar desde cero con 12; es decir, **ahorra al menos 12 ensayos, que es lo más que estos datos dejan medir** (cada corrida tiene unos 14 ensayos). Con los mismos ensayos, la ventaja es de 0.03 a 0.05, y con 12 no llega a dos errores estándar. No dice si 12 ensayos con memoria alcanzan a una calibración completa de 36.

**Con el gemelo** (4 sujetos; solo verificación: «otra sesión» es otro ruido y otra ganancia por electrodo, un cambio que programamos nosotros y que el recentrado deshace por construcción): MI con 12 ensayos de hoy, 0.70 ± 0.10 desde cero contra 0.88 ± 0.03 con memoria (desde cero con 36: 0.84 ± 0.04). ErrP con 40 épocas de hoy, 0.68 ± 0.05 contra 0.80 ± 0.02 (desde cero con 120: 0.74 ± 0.05), y **el CP3 con memoria habría reportado 0.80 contra 0.80 real**: no infla (con 80 y 120 épocas reporta 0.77 y 0.80 contra 0.81 real: se queda corto, no largo). Por LSL contra el gemelo, la prueba `lazo_real_memoria` corre las dos sesiones de punta a punta.

**El estado del agente no aporta nada medible en el gemelo.** En las 48 sesiones de `estudios/paso_sin_movimiento.py`, la `beta` de antes de la perturbación es ruido alrededor de cero (media −0.05 a −0.12 según el detector, desviación 0.14 a 0.20; dentro de un mismo sujeto varía tanto como entre sujetos) y antes de perturbar el agente y la sombra se equivocan igual (0.16 contra 0.15). Se hereda porque es barato y porque una persona podría tener un sesgo estable que el gemelo no tiene; con dos sesiones reales se puede ver.

**Sin probar con una persona.** Y una memoria que no es del mismo piloto estorba: en una corrida contra el gemelo con una memoria de épocas ajenas, el CP3 dio BA 0.58 con las 40 de hoy (NO GO, como debe).

![Memoria entre sesiones](docs/figuras/memoria_sesiones.png)

## Estado del sistema: revisión previa y franja en vivo (`estado_sistema.py`)

Apagado por defecto y fuera del lazo de control. Un solo comando revisa, antes de empezar, lo que suele fallar a media demo, y dice qué hacer con cada falla en una línea:

```bash
python estado_sistema.py                 # revisión previa (unos 15 s; no mueve la órtesis)
python estado_sistema.py --puerto COM4   # además mide la latencia del ACK con 10 movimientos
python estado_sistema.py --sin-api       # no llama a la API: solo mira que haya llave
python tablero.py --estado-sistema       # la misma revisión cada 10 s, en una franja del tablero
python puente_lsl.py --placa unicorn --estado   # el puente deja la batería y la validez del casco para la revisión
```

| Revisión | Qué mira | Falla si |
|---|---|---|
| `bateria`, `validez`, `perdidas`, `canales` | El casco: batería, muestras válidas, muestras perdidas por contador en los últimos 5 s y calidad por canal (la del CP1) | batería < 15 % (aviso < 30 %), pérdidas ≥ 5 % (aviso ≥ 1 %), un canal plano, saturado o ruidoso |
| `flujos`, `fuentes_eeg`, `flujos_repetidos`, `tasas` | Los flujos LSL de la red y las muestras por segundo que de verdad llegan | no hay EEG, **hay dos fuentes de EEG a la vez**, hay dos `Estado` / `Marcadores` / `Paso`, o llega menos del 75 % de la tasa nominal |
| `procesos` | Puente, gemelo, orquestador, tablero, narrador y LabRecorder vivos, con su PID | puente y gemelo a la vez, o dos orquestadores (aviso si LabRecorder no está abierto) |
| `ack` | Latencia del ACK con los umbrales del CP1; en el tablero, con los últimos 40 pasos del lazo | MAD > 15 ms, p95 > 60 ms o más de 10 % sin ACK |
| `cargador`, `suspension`, `tapa` | La laptop: con cargador y sin suspensión por inactividad | está con batería o se suspende sola |
| `disco` | Espacio libre donde se graba | < 1 GB (aviso < 5 GB) |
| `env`, `api` | La llave en `.env` y una llamada mínima a la API | hay llave pero la llamada falla (sin llave solo avisa: la IA usa plantillas y reglas) |
| `git` | Rama, versión (`git describe`) y cambios sin commit | solo avisa |

El programa termina con código 1 si hay alguna falla. La batería y la validez del casco van en el flujo de la app UnicornLSL, pero no en el flujo `EEG` del puente: con `--estado`, `puente_lsl.py` las escribe cada 5 s en `resultados/estado_puente.json`. Las pérdidas se cuentan igual con las dos fuentes, por los huecos de la hora de cada muestra (que sale del contador del casco).

**Probado sin hardware:** con el gemelo (en los dos formatos), con la placa sintética de BrainFlow, con la órtesis simulada y con la API simulada. Con el Unicorn, con la órtesis real y con una llamada real a la API, no.

## Resiliencia: el lazo que no se cae

Si se desconecta el dongle, se reinicia el ESP32, se congela LSL o llega una época corrupta, el sistema lo detecta, se protege, se recupera solo y no pierde la sesión.

**Semáforos (`salud.py`).** Antes de cada paso, el `Vigilante` revisa cinco subsistemas. Cada cambio de color sale en consola, en el flujo `Estado` y como marcador `salud:<subsistema>:<color>`. Los umbrales están en `config.SALUD`.

| Subsistema | Qué mide | AMARILLO | ROJO |
|---|---|---|---|
| EEG | edad de la última muestra, muestras que de verdad llegan, canales | 0.3 s sin muestras; 10 % menos muestras | 1 s sin muestras; 25 % menos; un canal plano, saturado o ruidoso |
| Órtesis | ACK perdidos, latencia, puerto | 1 ACK perdido; latencia ≥ 80 ms | 3 ACK perdidos seguidos; puerto caído |
| Reloj | deriva del retraso del EEG contra su línea base | 20 ms | 50 ms |
| Detector | fiabilidad viva del `ConfianzaDetector` | fiabilidad < 0.7 | aprendizaje congelado |
| Piloto (solo avisa) | alfa occipital (8–13 Hz en PO7/Oz/PO8, últimos 20 s) contra su línea base del primer minuto del lazo | ×1.5 | ×2.5 |

El detector empieza en `CALENTANDO` (gris en el tablero) hasta tener 15 épocas válidas: antes de eso su fiabilidad es ruido y no emite cambios. El piloto también, mientras mide su línea base. **El semáforo PILOTO solo avisa**: el alfa occipital sube con la somnolencia, los ojos cerrados o la desconexión de la tarea, pero no pausa, no excluye pasos y no cuenta en la escalera de degradación. Sus umbrales no están validados en personas. En el gemelo, con `--fatiga 1`, el alfa llega a ×2.8 a los 15 minutos (ROJO); sin fatiga se queda alrededor de ×1.

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

**Resultados con caos (exploratorios; simulador y gemelo, no una persona; medidos el 3 de octubre con la configuración final).** Error en los ~2 min tras la perturbación, sobre pasos no excluidos. Los movimientos ajenos de la Tarea 2 cuentan como excluidos (`ajeno`): no son decisiones del agente. "Caos leve" (`--caos-nivel leve`) es una falla cada 2 a 3 minutos; el estándar, una cada pocos segundos.

| Medición | Agente | Sombra | Excluidos y pausas |
|---|---|---|---|
| Simulador, 30 sujetos, sin caos | 0.225 | 0.296 | 900 de 10 800 filas, todas movimientos ajenos; agente por debajo en 28 de 30 sujetos |
| Simulador, 30 sujetos, caos leve | 0.220 | 0.289 | 985 de 10 847 filas: `ajeno` 900, `pausa:eeg` 26, `epoca_invalida` 23, `pausa:canal` 21, `sin_ack` 15; agente por debajo en 28 de 30 |
| Simulador, 30 sujetos, caos estándar | 0.223 | 0.296 | 2168 de 11 417 filas: `ajeno` 900, `pausa:eeg` 436, `epoca_invalida` 350, `sin_ack` 301, `pausa:canal` 180, `pausa:ortesis` 1; agente por debajo en 27 de 30 |
| Gemelo (montaje Unicorn), sin caos | 0.39 | 0.54 | 12 de 150 filas, todas movimientos ajenos; ninguna pausa; CP1 y CP2 en GO, CP3 en NO GO por especificidad (0.87) y CP4 en GO: recuperación en 28 pasos = 50 s; 30 pasos sin movimiento |
| Gemelo (montaje Unicorn), caos leve (semilla 2) | 0.23 | 0.53 | 13 de 151 filas: `ajeno` 12, `pausa:eeg` 1; 1 pausa, reanudada sola; recuperación en 22 pasos = 45 s; 38 pasos sin movimiento |
| Gemelo (montaje Unicorn), caos estándar (semilla 1) | 0.32 | 0.49 | 38 de 160 filas: `ajeno` 12, `epoca_invalida` 11, `pausa:eeg` 7, `sin_ack` 5, `pausa:canal` 3; 10 pausas, todas reanudadas solas; recuperación en 15 pasos = 44 s; 23 pasos sin movimiento |

Las tres filas del gemelo usan los mismos modelos (una sola calibración, la del CP3 en NO GO por especificidad; por eso el lazo se corrió con `--forzar`) y son **una corrida por condición**: los intervalos del 90 % de esos 2 minutos son muy anchos (por ejemplo [0.23, 0.56]). En el bloque adaptativo completo el agente quedó por debajo de la sombra en las tres: 0.35 contra 0.41, 0.19 contra 0.38 y 0.31 contra 0.45. Por la mañana, antes de corregir los pasos sin movimiento (el gemelo reaccionaba a todos los pasos), las mismas tres condiciones dieron 0.33/0.54, 0.51/0.54 (sin recuperarse) y 0.21/0.44; la de caos leve, repetida dos veces con los mismos modelos, sí se recuperó (0.23/0.54 y 0.28/0.54). Una sola corrida dice poco.

En el simulador el agente sigue claramente por debajo de la sombra con caos leve y estándar. En el gemelo, con la configuración final, las tres corridas se recuperaron en 44 a 50 s y el agente quedó por debajo de la sombra en los 2 minutos tras perturbar. El empate con la sombra de las corridas del 2 de octubre venía del detector de entonces (8 canales y dos vistas, BA viva ~0.74; sección del control negativo). El aprendizaje del agente no se modificó. Todas las sesiones terminaron sin excepción y todas las pausas se reanudaron solas.

**Lo que enseñó el caos (medido, no supuesto):**

- Tras un silencio sin perder el flujo, el suavizado de marcas de tiempo de LSL (dejitter) deja las marcas atrasadas: 2.7 s tras un hueco de 3 s, y tarda más de 10 s en converger. Eso desalineaba las épocas de ErrP y congelaba el aprendizaje. Primero se resolvió renovando la entrada tras cada silencio; ahora la hora de cada muestra se reconstruye con el contador del casco y el suavizado no se usa, así que el problema desaparece de raíz (prueba `silencio_sin_recrear`: 5 ms de diferencia tras 2.5 s de silencio, sin vaciar el buffer).
- La deriva del reloj se mide contra una mediana móvil de ~30 s: detecta un cambio de desfase y lo absorbe, así el semáforo del reloj nunca queda en ROJO para siempre.
- Una sesión sobrevivió a 37 minutos de suspensión del equipo (tapa cerrada) y se reanudó sola al despertar. Aun así: **no cierres la tapa durante la demo**.

## Embodiment: índice de integración corporal (Tarea 2 mínima, exploratorio)

**La idea.** Si el cerebro predice las consecuencias de sus propios movimientos, responde menos a lo que él mismo causó (atenuación sensorial). Con la órtesis: la respuesta visual temprana (N1 occipital) a un movimiento propio debería ser menor que a uno ajeno. **Es una métrica exploratoria, no validada clínicamente.** De las tres firmas que pedía la Tarea 2 solo está esta; la selectividad del error y la resonancia motora quedaron fuera de la versión mínima.

**Protocolo.** En el lazo adaptativo, 1 de cada 10 pasos es un movimiento ajeno: el tablero muestra **AUTOMATICO** durante 1 s (marcador `aviso_ajeno`) y la órtesis se mueve sola 0.15 hacia la meta (marcador `paso_ajeno:<seq>`). Va siempre hacia la meta para que el contraste no se mezcle con la respuesta al error. El agente, la confianza del detector y el detector co-adaptativo no aprenden de esos pasos; en el CSV llevan `ajeno = 1` y `excluido = ajeno`. `--ajenos-cada 0` los apaga.

**El índice.** IIC = d de Cohen entre la N1 de los movimientos ajenos y la de los propios correctos (media de PO7/Oz/PO8 entre 140 y 200 ms tras el inicio del movimiento), con intervalo bootstrap del 90 % y la tendencia entre las dos mitades de la sesión. Positivo = los movimientos propios se atenúan. Sale en las columnas `n1_uv` e `iic` del CSV, en el flujo `Estado`, en una línea del tablero y en EVALUACION. Al terminar una sesión `real`, un cuestionario de tres afirmaciones (propiedad, agencia y control; de 1 a 7) se guarda en `resultados/sesion_..._cuestionario.json` junto con el IIC (`--sin-cuestionario` lo salta).

**Lo que dice el gemelo** (`estudios/embodiment_gemelo.py`, 16 sujetos, sesiones independientes). El gemelo atenúa la N1 propia a (1 − 0.5 × embodiment): el efecto lo programamos nosotros, así que esto solo verifica el estimador.

| Ajenos por sesión | IIC con embodiment 0 / 0.2 / 0.5 / 0.8 | Intervalo incluye 0 con embodiment 0 | Spearman, todas las sesiones juntas | Orden perfecto por sujeto |
|---|---|---|---|---|
| 12 (la demo: 120 pasos adaptativos) | +0.12 / +0.19 / +0.16 / +0.13 | 88 % | −0.12 [−0.31, +0.07] | 2 de 16 |
| 30 (300 pasos adaptativos) | +0.12 / +0.09 / +0.19 / +0.21 | 100 % | +0.24 [−0.03, +0.50] | 4 de 16 |

**El criterio de aceptación no se cumple.** El IIC ordena los niveles apenas con 30 ajenos y nada con 12. El estimador no tiene sesgo: con embodiment 0, 64 sesiones de 30 ajenos dan +0.01 (el +0.12 de la tabla es azar de esas 16; otras 48 dieron −0.02). Lo que falta es potencia. En el gemelo la N1 mide ~3.5 µV contra ~4.5 µV de ruido por época; con embodiment 0.8 el efecto verdadero es d ≈ 0.3, y con 12 ajenos el intervalo de una sola sesión mide ±0.5. **En la demo el IIC solo podrá mostrar una atenuación grande (d ≳ 0.6); con una sesión no distingue niveles cercanos de embodiment.** Por eso se presenta como exploratorio, con su intervalo (decisión de Luis, 3 de octubre).

**Análisis de potencia** (`estudios/potencia_iic.py`, con las 176 sesiones del gemelo ya simuladas en `estudios/datos/iic_gemelo.csv`; no genera EEG nuevo). El IIC de una sesión se comporta como su d verdadera más ruido de desviación √(1.14 / ajenos). El modelo reproduce lo medido (0.34 con 12 ajenos y 0.18 con 30) y coincide con la teoría del error de una d de Cohen. En el gemelo, d = 0.32 × embodiment (intervalo del 90 %: 0.27 a 0.37).

![Potencia del IIC: ancho del intervalo y Spearman según el número de movimientos ajenos](docs/figuras/potencia_iic.png)

- **Intervalo de ±0.2:** hacen falta **77 movimientos ajenos**, unos 770 pasos del lazo adaptativo con 1 ajeno de cada 10 (la demo tiene 120). Con 12 ajenos el intervalo mide ±0.55 y con 30, ±0.30.
- **Spearman significativo** (una cola, α = 0.05; con 16 sujetos por nivel el valor crítico es 0.24): con los 12 ajenos de la demo la potencia es de 52 % y con **29 ajenos, de 80 %**. Nuestro estudio de 30 ajenos quedó justo en el borde (+0.24). Con 32 sujetos por nivel bastarían 13 ajenos.
- **Significativo no es alto.** Un Spearman esperado de 0.5 pide unos 60 ajenos y uno de 0.8, unos 300. Un solo sujeto ordena bien los tres niveles el 27 % de las veces con 12 ajenos y el 33 % con 30 (por azar, 17 %).

Todo esto vale para el gemelo: el tamaño del efecto lo programamos nosotros y en una persona se desconoce. Lo que sí se traslada es la escala: el ancho del intervalo baja con la raíz del número de ajenos.

## Bloque sham y especificidad por dirección (B1 y B2)

**Sham.** Con el piloto en reposo, la órtesis vuelve al centro y se mueve sola a 0.5 ± 0.2 (mitad cerrar, mitad abrir, 40 pasos); la ventana de MI va de 0.5 s antes a 1.5 s después del movimiento. `hardware.evaluar_sham` mide la AUC de `p` contra la dirección con un intervalo bootstrap del 95 %: PASA si el intervalo incluye 0.5. Si FALLA, `p(t)` está leyendo los servos, los cables o la respuesta visual, no la intención. Con 40 pasos el intervalo mide ~±0.2: detecta que `p` siga a la órtesis con claridad, no un seguimiento leve. Se corre después del CP3 y no cambia la máquina de estados:

```bash
python bloque_sham.py --puerto COM4        # o --ortesis-sim; sale con código 1 si FALLA
```

Medido en el gemelo (EXPLORATORIO; `pruebas.py`, prueba `bloque_sham`): piloto en reposo, AUC 0.34 a 0.65 en 12 sesiones, ninguna con falsa alarma; piloto que imagina lo que hace la órtesis, AUC 0.73 a 0.95 en 6 de 6, todas detectadas. Con etiquetas independientes del EEG, el criterio da falsa alarma en 5.5 % de 600 simulaciones, como corresponde a un intervalo al 95 %. Falta probarlo con el casco: el gemelo no tiene ruido de servos.

**Especificidad por dirección.** `calibrar_errp` pasa la dirección en que se movió la órtesis a `DetectorErrP.ajustar(direccion=...)`, que desglosa las predicciones de su validación anidada en `detector.por_direccion` (sensibilidad, especificidad y BA de cerrar y de abrir, y `dif_espec`). La consola la imprime tras la BA. Si la diferencia de especificidad pasa de `config.ESPEC_DIF_MAX` (0.05) **solo avisa**, no da NO GO: con 120 épocas quedan ~40 aciertos por dirección y el azar solo produce una diferencia de ~0.06. El umbral sigue siendo uno solo para las dos direcciones; umbrales por dirección en el agente quedan como decisión pendiente.

**Lo que se les agregó (4 de octubre).** Dos cosas, sin duplicar lo de jusren:

- *Prueba de Fisher por dirección.* Tras la línea de arriba, la calibración imprime el p-valor de la prueba exacta de Fisher de que las falsas alarmas no dependen de la dirección (`hardware.fisher_por_direccion`); avisa si p < 0.05. Con unos 40 aciertos por dirección una diferencia de 0.05 es azar, y la prueba dice si lo visto es más que eso. Con datos sintéticos avisa en 191 de 200 calibraciones cuando las falsas alarmas son 30 % a un lado y 2 % al otro, y en 10 de 200 cuando son iguales (`pruebas.py`, `controles_especificidad`).
- *El bloque sham dentro del orquestador.* `python orquestador.py real --puerto COM4 --control-reposo` corre `bloque_sham.correr` tras calibrar (también con `--saltar-calibracion`), con el EEG y la órtesis de la sesión, sin detener y volver a arrancar el orquestador.

**Ojo con los nombres:** `bloque_sham.py` y `--control-reposo` son este control (la órtesis sola con el piloto en reposo); `orquestador.py --sham` es el control causal del agente, real contra sham.

## Probar sin hardware

```bash
python simulador_lazo.py                        # tabla + figura
python simulador_lazo.py --falla_detector       # congelamiento
python simulador_lazo.py --sin_sesgo            # metas no balanceadas
python orquestador.py sim --ciclo 0             # orquestador completo, rápido
python tablero.py  &  python orquestador.py sim # ver el tablero en vivo
```

## El día de la demo (en este orden)

**El guion completo está en [docs/DOMINGO.md](docs/DOMINGO.md):** los comandos en orden, qué hacer con cada veredicto de `verificar_unicorn.py`, el árbol de decisión de cada checkpoint y el plan B. Lo de abajo es el resumen.

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

Con modelos ya calibrados: `--saltar-calibracion`. Sin ESP32: `--ortesis-sim`. Para que el detector no cambie durante el lazo: `--sin-coadaptativo`.
Si el CP3 da NO GO, `--solo-errp` repite solo la calibración de ErrP con el decoder de MI ya calibrado. Al cargar modelos guardados, el orquestador dice hace cuánto se calibraron y avisa si tienen más de 6 horas (en `modelos/` pueden quedar los del gemelo o los de otro piloto).

**Plan B.** Cada sesión deja, junto a su CSV, `resultados/sesion_..._estado.jsonl` con todo lo que publicó al tablero. `python repetir_sesion.py --ultima --velocidad 2 --puerto COM4` repite la última sesión real en el tablero, y la órtesis hace los mismos movimientos, sin casco ni calibración. Es una repetición y hay que decirlo. La alternativa es el gemelo en vivo (`cerebro_sintetico.py` en lugar del puente). Reproducir el EEG crudo (`puente_lsl.py --placa playback --archivo resultados/sesion_unicorn.csv`) sirve para mostrar la señal, pero no para el lazo: lo grabado no responde a las señales nuevas.

El puente imprime cada 30 s el registro de huecos de Bluetooth y la batería.

**Bloque sham (opcional, ~3 min, lo decide P1):** el orquestador no tiene un estado para él y comparte el casco y la órtesis, así que se corre entre dos arranques. Deja que el orquestador llegue al CP3, detenlo con Ctrl+C al entrar a `LAZO_ESTATICO` (los modelos ya están en `modelos/`), corre `python bloque_sham.py --puerto COM4` con el piloto en reposo y sigue con `python orquestador.py real --puerto COM4 --saltar-calibracion`.

**Antes de empezar:** cierra cualquier `cerebro_sintetico.py` o `puente_lsl.py` que haya quedado abierto en otra terminal. Debe haber un solo flujo `EEG` en la red; `python ver_flujos.py` lo muestra.

**Si algo se cae a media demo:** el sistema entra solo en `PAUSA_SEGURA` y sale solo. Si se cerró el orquestador, vuelve a lanzarlo con `--reanudar`.

## Checkpoints (los decide P1)

| CP | Dónde | Criterio | Si falla |
|---|---|---|---|
| 1 | IMPEDANCIAS | Calidad de señal por canal (el Unicorn no mide impedancias) y latencia del ACK en 40 movimientos: MAD ≤ 15 ms, p95 ≤ 60 ms, ACK perdidos ≤ 10 % | Más gel / revisar firmware antes de seguir |
| 2 | CAL_MI | Exactitud balanceada MI ≥ 0.70 (secuencial, mínimo 36 ensayos) | Cambiar piloto o mano vs pies |
| 3 | CAL_ERRP | BA ErrP ≥ 0.75 y especificidad ≥ 0.90, con 120 épocas fijas y umbral anidado | Repetir con `--solo-errp`, seguir con `--saltar-calibracion` (aprende más lento) o plan B |
| 4 | EVALUACION | Recuperación ≤ 120 s | Congelar y usar la ruta de 24 h |

`--forzar` continúa aunque un checkpoint diga NO GO (solo para pruebas).

### Barrido del tamaño de paso y de los pasos por ensayo (5 de octubre, gemelo)

`python estudios/barrido_paso.py` (unos 6 minutos; `informe` rehace tabla y figura): 4 tamaños de paso máximo (= ganancia) × 3 largos de ensayo, 16 sesiones por celda con las mismas semillas, agente bayes de hoy, detector actual y los topes del recorrido. Tabla completa en `docs/barrido_paso.md` y figura en `docs/figuras/barrido_paso.png`. **Es el gemelo, no una persona.**

| Paso máx. | Pasos por ensayo | Pasos en tope | Cierre completo | Se recuperan | Error 2 min |
|---|---|---|---|---|---|
| 0.30 (actual) | 5 | 23 % | 54 % | 14/16 | 0.371 |
| 0.30 | 3 | 5 % | 31 % | 16/16 | 0.309 |
| 0.40 | 3 | 11 % | 46 % | 16/16 | 0.315 |
| 0.40 | 5 | 32 % | 57 % | 14/16 | 0.378 |
| 0.20 | 5 | 11 % | 41 % | 16/16 | 0.328 |
| 0.30 | 8 | 44 % | 61 % | 12/16 | 0.421 |

- **Hay un compromiso, no una celda que gane en todo.** Menos pasos en el tope (y por tanto más épocas de ErrP útiles) y mejor recuperación piden pasos chicos o ensayos cortos; que la órtesis termine cerrada del todo pide lo contrario (cubrir 0.45 del recorrido en pocos pasos). La configuración actual (0.30 × 5) reproduce la corrida de la comparación con baselines (14/16, 0.371).
- **La mejor recuperación sin perder demasiado cierre:** 0.40 × 3 (16/16, 11 % en tope, 46 % de cierre completo) o 0.30 × 3 (16/16, 5 %, 31 %). Con 8 pasos por ensayo el tope se come casi la mitad de los pasos y se recuperan 12 de 16.
- **Lo que este barrido NO mide:** con 3 pasos por ensayo hay ~67 % más ensayos para los mismos pasos, y cada ensayo lleva su ventana de imaginación y su cue: la sesión dura más en tiempo real, y el gemelo no cobra ese tiempo (la ventana de «2 min» son 57 pasos). Tampoco el tiempo que tarda un paso grande en el firmware por Wi-Fi (~430 ms para 0.30). Con una persona, el paso visible (0.08) y la fatiga pueden cambiar el cuadro.
- **No se cambió ningún valor por defecto**: cualquier cambio de `PASO_MAX`, `GANANCIA_PASO` o `PASOS_ENSAYO` mueve los números de referencia de `CLAUDE.md` y lo decide Luis.

## Figuras para la presentación

Están en `docs/figuras/` y cada una se regenera con su estudio. Todas son del simulador o del gemelo, no de una persona.

| Figura | Qué muestra | Cómo se regenera |
|---|---|---|
| `control_negativo.png` | **El resultado central.** Sin la evidencia del ErrP el agente no se recupera de la perturbación (1 de 16 sesiones); con ella sí (10 a 16 de 16), con tres calidades de detector y los topes del recorrido. | `python estudios/paso_sin_movimiento.py` (unos 8 minutos; `informe` rehace la tabla y la figura con lo ya corrido) |
| `comparacion_baselines_es.png` y `_en.png` | **Comparación final con baselines** (estático, eta fija, bayes, sin ErrP; mismas semillas): β aprendida, error y sesiones recuperadas. La tabla está en `docs/comparacion_baselines_es.md` y `_en.md`. Gemelo con los topes del recorrido, detector actual y el prior por paso de hoy; con los detectores medio y débil (en la tabla) bayes baja a 15 y 7 de 16 y eta fija a 0 de 16. | `python estudios/comparacion_baselines.py` (unos 5 minutos; `informe` rehace tablas y figuras) |
| `barrido_paso.png` | Barrido del tamaño de paso y los pasos por ensayo: pasos en el tope, ensayos que terminan con la órtesis completa y sesiones que se recuperan (gemelo). | `python estudios/barrido_paso.py` (unos 6 minutos; `informe` rehace tabla y figura) |
| `transferencia_physionet.png` | BA de un decoder desde cero, pre-entrenado con otras personas y pre-entrenado más ensayos propios, según los ensayos de calibración (EEGMMIDB, 40 personas). | `python estudios/transferencia_physionet.py` (descarga lenta la primera vez; `informe` rehace la tabla y la figura) |
| `curva_robustez.png` | Error tras perturbar y tiempo de recuperación según la BA del detector (0.65 a 0.85). Simulador rápido: no modela los topes del recorrido. | `python estudios/curva_robustez.py` |
| `potencia_iic.png` | Cuántos movimientos ajenos pide el IIC para un intervalo de ±0.2 y para un Spearman significativo. | `python estudios/potencia_iic.py` |
| `agente_lento.png` | Figura técnica: por qué el agente era lento y por qué se descartó la corrección del prior. | `python estudios/agente_lento.py informe` |

## IA sobre una sesión real, con auditoría de cifras (`ia_sesion.py`)

```bash
python ia_sesion.py --sesion resultados/sesion_real_<fecha>.csv            # con la llave de .env: Claude; sin llave: plantillas y reglas
python ia_sesion.py --sesion resultados/sesion_real_<fecha>.csv --sin-ia   # solo plantillas y reglas
```

Corre el copiloto (6 preguntas: recuperación, agente contra sombra, congelamiento, fatiga, excluidos y comparación con la sesión anterior), el co-investigador (propuesta validada contra los rangos seguros y comparada con las reglas deterministas) y el informe clínico (4 Markdown, figuras y la propuesta de la próxima sesión), y **audita** lo que dijeron: cada cifra de una respuesta debe existir en lo que las herramientas devuelven para esa sesión (con el redondeo con que se escribió; los enteros, exactos) y todo «paso N» debe existir. Una cifra sin encontrar no es necesariamente falsa (puede ser una resta o un porcentaje derivado): se lista para revisarla. Deja `sesion_..._ia.md` junto al CSV. **No aplica nada**: las propuestas siguen pendientes de una persona. Prueba `ia_sesion` (65 cifras verificadas en una sesión del simulador y cifras inventadas detectadas). **Con la API de Claude no se ha corrido nunca** (no hay llave en la máquina de desarrollo); con ella la auditoría usa además lo que devolvieron las herramientas que usó el modelo.

## Gemelo personalizado (`gemelo_personal.py`)

```bash
python gemelo_personal.py --mi resultados/calibracion_mi_<n>.npz --errp resultados/calibracion_errp_<n>.npz --sesion resultados/sesion_real_<fecha>.csv
python gemelo_personal.py --demo        # un piloto de mentira (del propio gemelo): comprueba el método
```

Ajusta el cerebro sintético a las calibraciones **reales** de un piloto (ganancia por canal con el RMS en reposo, `erd` con el cociente de potencia en C3, `errp` con la Pe de la onda diferencia en Fz/Cz/Pz sin las épocas con artefacto, `parpadeos` con la fracción de épocas con artefacto) por simulación con rejillas, y **predice** la calibración (BA de MI y del detector) y el lazo (estático y bayes, 4 sujetos × 4 lazos) contra el gemelo estándar; con `--sesion` pone al lado lo medido en la sesión real. Predice, no mide, y el informe lo dice.

- **Comprobación con un piloto de mentira** (`--demo`: erd 0.15, ErrP 4 µV, parpadeos 0.30 y ganancias por canal distintas, medido en el propio gemelo, 4 sujetos × 4 lazos): el método recupera erd 0.15 (0.15), parpadeos 0.25/s (0.30) y un ErrP de 5.3 µV (4.0). **BA del decoder de MI predicha 0.72 contra 0.71 verdadera; BA del detector de ErrP 0.66 contra 0.63; error del agente tras perturbar 0.462 contra 0.497.** Pero **la recuperación sale 8 de 16 contra 0 de 16**: con una BA del detector entre 0.60 y 0.70 la recuperación es un umbral y un error de 0.03 en la BA la cambia por completo. Es un solo piloto de mentira. Lo fiable de la predicción son las BA de la calibración y el rango del error; para la recuperación, ancla con la BA real del CP3 (curva de robustez).
- La ganancia por canal iguala el RMS (error máximo 0 % por construcción), no recupera las ganancias «verdaderas»: el gemelo tiene su propia mezcla entre electrodos (Fz sale ×1.75 contra ×1.2 puesto).
- **Sin datos de P001 en esta máquina:** el método está probado con datos del gemelo; correrlo con una persona está pendiente de sus archivos de calibración (ver `TAREAS.md`).

## Protocolo del ESP32 (para P2)

```
PC -> ESP32   M,<seq>,<angulo 0-1000>,<duracion_ms>\n     0 = abierta, 1000 = cerrada
ESP32 -> PC   A,<seq>,<t_us>\n     ACK al aplicar el primer pulso (marca el inicio del ErrP)
ESP32 -> PC   T,<t_us>,<angulo>,<fsr>\n     telemetría a 50 Hz
```
USB serial a 115200 baudios. El orquestador espera el ACK máximo 300 ms por paso y mide la latencia de cada uno. Si el ACK no llega, el paso queda excluido y el lazo sigue; con tres seguidos entra en `PAUSA_SEGURA`. El ESP32 solo debe devolver el `seq` que recibió: tras reiniciarse no necesita recordar nada.

### Por Wi-Fi: `--ortesis-udp` (firmware 1.2 de la ESP32)

```bash
python orquestador.py real --ortesis-udp                 # la ESP32 en su red «Adaptrode» (192.168.4.1:8888), en lugar de --puerto
python ortesis_udp_sim.py                                # SIN placa: el firmware simulado en la laptop (127.0.0.1:8888)
python demo.py lanzar --plan gemelo --ortesis-udp 127.0.0.1   #   o todo junto: demo.py lanza también el firmware simulado
python orquestador.py real --ortesis-udp 127.0.0.1       #   y el orquestador contra él
```

`hardware.OrtesisUDP` tiene la misma interfaz que `OrtesisSerial` (protocolo JSON en `config.py`, sección *Ortesis por Wi-Fi*). Lo que cambia:

- **Reloj de LSL.** El cliente se crea con `reloj = pylsl.local_clock` (`orquestador.crear_ortesis`), el mismo reloj que estampa el EEG; `time.monotonic` no lo es y en Windows resuelve ~15 ms. `mover()` devuelve como `t_ack` la hora en que la ESP32 **aplicó** la orden: su `t_ms` convertido a ese reloj (`RelojEsp32Wifi`: mediana del desfase de los ACK de menor ida y vuelta de los últimos ~20 s, así sigue la deriva). Si la conversión cae fuera del intervalo envío–llegada del ACK, usa el punto medio. Prueba `ortesis_udp`.
- **Latido.** La ESP32 abre la mano si pasan 0.5 s sin órdenes y el orquestador solo mueve cada ~2 s: un hilo reenvía el estado cada 100 ms. Los latidos numeran aparte (desde 10⁹) para que el `seq` de los pasos siga de uno en uno; sus ACK alimentan el reloj.
- **Sin ACK** en 0.3 s (como por USB): `(seq, None, nan)`, el paso queda excluido y el lazo sigue.
- **Telemetría a 10 Hz** (por USB, 50 Hz). El inicio real del movimiento se extrapola con la velocidad del servo (90 °/s, `config.UDP_VEL_MAX_GRADOS_S`): la interpolación lineal de la telemetría lenta lo adelantaba 30–60 ms. En el firmware simulado el error queda en ~−15 ± 10 ms. **La telemetría del firmware 1.2 es la posición ordenada (`frac`), no la medida:** el retraso físico del servo no se ve.
- **El cerebro enseñando, en la mano.** En cada paso el orquestador manda `p = p'` del agente (`set_p()`, viaja con la orden y en los latidos) y el nervio de luz sube con ella; cuando el detector marca un ErrP (`p_errp` sobre el umbral) en una época **sin artefacto**, manda `errp()` y la órtesis destella en rojo. Una época con artefacto no cuenta como ErrP para el agente y tampoco destella. UDP pierde datagramas (el Wi-Fi del evento), así que `errp` viaja también en los 3 latidos siguientes (`config.UDP_ERRP_LATIDOS`): con la mitad de los datagramas perdidos en el firmware simulado llegan 12 de 12 ErrP contra 6 de 12 con un solo datagrama (prueba `destello_errp_con_perdidas`). Costo: el firmware reinicia su destello de 300 ms con cada mensaje, así que dura hasta ~0.6 s en lugar de 0.3 s. Por USB y con la órtesis simulada no pasa nada (no la tienen). Prueba `ortesis_udp_nervio`. En una sesión `--sham` el nervio y el destello son iguales en los dos bloques (`p'` y la detección reales), así que no delatan cuál es cuál.
- **Paro de emergencia y bloqueo.** El `Vigilante` pasa la órtesis a ROJO (`PAUSA_SEGURA`) con el paro oprimido y a AMARILLO con el bloqueo por corriente.
- **El firmware limita la velocidad:** `dur_ms` se ignora, y un paso de 0.30 del rango tarda ~430 ms (90 °/s sobre 130 °), no los 250 ms del protocolo USB. Un solo grado de libertad: la fracción va a `cierre` y a `pulgar`. 
- **`ortesis_udp_sim.py` no es el firmware:** modela ACK, relojes con otro origen y deriva, vigilancia, velocidad del servo, paro y bloqueo; no mide corriente ni fuerza. Cada cambio de meta empieza a moverse `latencia_mecanica_simulada(seq)` tras el ACK, para que el gemelo (`cerebro_sintetico.py`) siga alineado. **Sin probar con la ESP32 ni por Wi-Fi real.**
- Con la red «Adaptrode» la laptop se queda sin internet (la IA cae a plantillas): usar un segundo adaptador.

**Cierre completo.** Antes la órtesis casi nunca cerraba del todo: cada ensayo seguía desde donde quedó el anterior y los pasos eran de 0.20 como máximo. Ahora cada ensayo empieza en el punto medio (antes del cue llega `M,<seq>,500,400`, con el marcador `centrado`; no es un paso y no lleva época de ErrP, y también se hace al retomar un ensayo tras una pausa) y un paso mueve hasta 0.30 del rango en 250 ms. En el simulador (30 sujetos, `estudios/cierre_completo.py`) la órtesis termina cerrada del todo en el 68 % de los ensayos de cerrar (antes 27 %) y abierta del todo en el 75 % de los de relajar (antes 41 %), con el mismo error del agente. Para P2: el paso más rápido es ahora 0.30 del rango en 250 ms.
