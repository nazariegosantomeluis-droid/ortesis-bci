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

La calibración real decidirá, por validación cruzada, si cada modelo usa los canales de su papel o los 8. El gemelo ya se comporta como un Unicorn (montaje, respuesta visual occipital, IMU, contador y pérdidas de Bluetooth) y puede publicar en el formato del puente o en el de la app UnicornLSL. La selección de canales en la calibración, el CP1 sin impedancias (calidad de señal por canal y latencia del ACK) y el rechazo por movimiento de cabeza ya están implementados; **nada se ha probado todavía con el casco** (`verificar_unicorn.py` es lo primero el domingo).

El orquestador puede leer el EEG de dos fuentes (`--fuente`): `puente` (por defecto: `puente_lsl.py` o el gemelo; flujos `EEG` e `IMU`) o `unicornlsl` (la app de g.tec: un flujo de tipo `Data` con 17 canales, que se resuelve por tipo o con `--eeg-nombre <nombre o número de serie>`). Solo una aplicación puede conectarse al casco a la vez.

## Instalación

```bash
python -m venv .venv
source .venv/Scripts/activate        # Git Bash en Windows (en CMD: .venv\Scripts\activate)
pip install -r requirements.txt
python pruebas.py                    # debe decir 39/39 pruebas pasaron
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
| `cerebro_sintetico.py` | **Gemelo digital del piloto**: publica EEG por LSL que *reacciona* al lazo (ERD al imaginar, ErrP cuando la órtesis se equivoca, N1 visual atenuada según `--embodiment`). Reemplaza al casco para ensayar. |
| `embodiment.py` | Tarea 2 (exploratorio): N1 visual de cada movimiento e índice de integración corporal (IIC) con intervalo y tendencia. |
| `estudios/` | Mediciones offline con el gemelo y el simulador que respaldan cada decisión (ver su README). |
| `salud.py` | `Vigilante`: semáforo VERDE / AMARILLO / ROJO por subsistema (EEG, órtesis, reloj, detector y piloto, que solo avisa) y el retroceso de las reconexiones. |
| `caos.py` | `PlanCaos`: fallas reproducibles por semilla (ingeniería del caos aplicada al lazo). |
| `tablero.py` | Tablero en vivo de 5 paneles, con cuatro semáforos en la cabecera (EEG, órtesis, detector y piloto), el aviso AUTOMATICO de los movimientos ajenos y una línea con el IIC. |
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

Corrida completa con el montaje del Unicorn (`orquestador.py real --ortesis-sim` contra el cerebro sintético, una corrida): CP1 a CP4 en **GO** (MI BA 0.86 con 36 ensayos, ErrP BA 0.79 con 120 épocas). En los 2 min tras la perturbación el agente y la sombra empataron (0.47 contra 0.47; con el montaje anterior, 0.26 contra 0.46): en el gemelo el agente recupera más lento que en el simulador y la causa sigue abierta (sección Resiliencia).

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

**Resultados con caos (exploratorios; simulador y gemelo, no una persona).** Error en los ~2 min tras la perturbación, sobre pasos no excluidos. "Caos leve" (`--caos-nivel leve`) es una falla cada 2 a 3 minutos; el estándar, una cada pocos segundos.

| Medición | Agente | Sombra | Excluidos y pausas |
|---|---|---|---|
| Simulador, 30 sujetos, sin caos | 0.235 | 0.323 | ninguno; agente por debajo en 26 de 30 sujetos |
| Simulador, 30 sujetos, caos leve | 0.232 | 0.326 | 105 de 10 867 filas: `pausa:eeg` 39, `pausa:canal` 28, `epoca_invalida` 20, `sin_ack` 18; agente por debajo en 27 de 30 |
| Simulador, 30 sujetos, caos estándar | 0.222 | 0.333 | 1109 de 11 425 filas: `pausa:eeg` 443, `sin_ack` 332, `pausa:canal` 181, `epoca_invalida` 152, `pausa:ortesis` 1; agente por debajo en 29 de 30 |
| Gemelo (montaje Unicorn), sin caos | 0.47 | 0.47 | 0 de 150 filas; ninguna pausa; CP1 a CP4 en GO (MI 0.86 con 36 ensayos, ErrP 0.79 con 120 épocas); recuperación en 82 s |
| Gemelo (montaje Unicorn), caos leve (semilla 2) | 0.47 | 0.47 | 2 de 151 filas: `pausa:eeg` 1, `epoca_invalida` 1; 1 pausa, reanudada sola; recuperación en 114 s |
| Gemelo (montaje Unicorn), caos estándar (semilla 1) | 0.42 | 0.49 | 23 de 161 filas: `pausa:eeg` 8, `sin_ack` 6, `epoca_invalida` 6, `pausa:canal` 3; 11 pausas, todas reanudadas solas; recuperación en 156 s (CP4 NO GO) |

Las tres filas del gemelo usan los mismos modelos (una sola calibración) y son **una corrida por condición**; los intervalos del 90 % de esos 2 minutos son muy anchos (por ejemplo [0.25, 0.74]). En el bloque adaptativo completo el agente quedó por debajo de la sombra en las tres: 0.44 contra 0.46, 0.29 contra 0.34 y 0.30 contra 0.40. Antes, con el montaje anterior, tres corridas con caos estándar dieron 0.44/0.49, 0.30/0.53 y 0.46/0.47 (agente/sombra).

En el simulador el agente sigue claramente por debajo de la sombra con caos leve y estándar. **En el gemelo con el montaje del Unicorn, en los primeros 2 minutos tras perturbar el agente no supera a la sombra** (recupera en 82 a 156 s, más lento que en el simulador). El agente no se modificó para estas mediciones; la causa está abierta (ver `TAREAS.md`). Todas las sesiones terminaron sin excepción y todas las pausas se reanudaron solas.

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

**Cierre completo.** Antes la órtesis casi nunca cerraba del todo: cada ensayo seguía desde donde quedó el anterior y los pasos eran de 0.20 como máximo. Ahora cada ensayo empieza en el punto medio (antes del cue llega `M,<seq>,500,400`, con el marcador `centrado`; no es un paso y no lleva época de ErrP, y también se hace al retomar un ensayo tras una pausa) y un paso mueve hasta 0.30 del rango en 250 ms. En el simulador (30 sujetos, `estudios/cierre_completo.py`) la órtesis termina cerrada del todo en el 68 % de los ensayos de cerrar (antes 27 %) y abierta del todo en el 75 % de los de relajar (antes 41 %), con el mismo error del agente. Para P2: el paso más rápido es ahora 0.30 del rango en 250 ms.
