# ortesis-bci — contexto para Claude Code

Órtesis de mano controlada por imaginación motora (MI). Un agente corrige el decoder en línea usando el **potencial de error (ErrP)** del cerebro como señal de aprendizaje. Proyecto de Luis (rol P1: orquestador, agente e integración) para una competencia internacional de neurotecnología. La demo con casco real (OpenBCI Cyton, 8 canales, 250 Hz) es el domingo 4 de octubre de 2026; **antes de eso no hay casco**: todo se valida con `cerebro_sintetico.py`.

## Reglas del proyecto (obligatorias)

1. **Código propio.** Se puede analizar qué buscaba el código de otros compañeros, pero no copiarlo. Las soluciones deben ser originales y bien fundamentadas.
2. **Innovación con evidencia.** Ninguna idea se declara mejor sin medirla en `simulador_lazo.py` o en `cerebro_sintetico.py` (banco offline o corrida completa). Si una idea empeora los números, se descarta y se documenta por qué.
3. **El contrato vive en `config.py`.** Los nombres de flujos LSL, marcadores, columnas del CSV, estados, transiciones y umbrales se definen ahí y en ningún otro lado. Si cambias el contrato, actualiza a todos sus consumidores (orquestador, tablero, pruebas, README).
4. **Las pruebas nunca se rompen.** `python pruebas.py` debe pasar antes de cada commit. Cada función nueva lleva su prueba en `pruebas.py`, sin hardware.
5. **Honestidad.** Las métricas exploratorias se etiquetan como exploratorias. No se inventan resultados ni se presentan números del gemelo como si fueran de una persona.
6. **Español** en nombres, comentarios, mensajes y documentación. Sin acentos dentro del código (compatibilidad de consola en Windows); con acentos en README y documentos.
7. Un commit por subtarea, con mensaje descriptivo.

## Entorno

Windows + Git Bash + Python 3.11 en `.venv` (`source .venv/Scripts/activate`). Dependencias en `requirements.txt`. Las líneas `INFO` de liblsl son ruido normal.

## Arquitectura

| Archivo | Rol |
|---|---|
| `config.py` | Contrato: flujos LSL, marcadores, CSV, tiempos, umbrales, protocolo del ESP32, máquina de estados. |
| `salud.py` | `Vigilante` (semáforo VERDE/AMARILLO/ROJO por subsistema: EEG, órtesis, reloj, detector; el detector empieza en CALENTANDO) y `Retroceso` (esperas de reconexión). Clase pura, sin hardware. |
| `caos.py` | `PlanCaos(semilla)`: fallas reproducibles (cortes de EEG, ACK perdidos, picos de latencia, parpadeos, canal despegado). Tasas en `config.CAOS_ESTANDAR`. |
| `agente_errp.py` | `AgenteErrP` (filtro de Kalman sobre la corrección `beta` del logit; P_hat bayesiano; detectores de cambio por sesgo y chequeo predictivo) y `ConfianzaDetector` (sens/espec vivas del detector de ErrP con posteriores Beta; congela el aprendizaje si el detector deja de informar). |
| `simulador_lazo.py` | Piloto sintético rápido a nivel de rasgos, para comparar agentes con muchos sujetos. |
| `cerebro_sintetico.py` | Gemelo digital del piloto: publica EEG por LSL que reacciona al lazo (ERD mu/beta en C3, ErrP fronto-central ante movimientos erróneos, parpadeos, fatiga). Tiene banco offline (`sesion_mi`, `sesion_errp`, `--banco`). |
| `hardware.py` | `EntradaEEG` (LSL con buffer y reloj sincronizado), `OrtesisSerial`/`OrtesisSimulada` (ACK con latencia medida), `DecoderIM` (Riemann con recentrado no supervisado), `DetectorErrP` (fusión temporal + geométrica, calibrado, umbral de Neyman-Pearson, detector de rareza), calibración secuencial (`intervalo_ba`). |
| `orquestador.py` | Máquina de estados, checkpoints go/no go (CP1-CP4), calibraciones, lazo, CSV, flujos `Marcadores`/`Paso`/`Estado`. Backends `sim` y `real`. Revisa la salud antes de cada paso, entra y sale de `PAUSA_SEGURA`, marca pasos excluidos, guarda una instantánea por paso y reanuda con `--reanudar`. |
| `puente_lsl.py` | BrainFlow → LSL (sintética, Cyton, playback) e impedancias. |
| `tablero.py` | Tablero pyqtgraph de 5 paneles que escucha el flujo `Estado` (JSON por paso). |
| `ver_flujos.py`, `pruebas.py` | Diagnóstico LSL y pruebas automáticas. |

Flujo del domingo: `puente_lsl.py` (o `cerebro_sintetico.py`) → LSL `EEG` → `orquestador.py` → órtesis por USB (protocolo `M`/`A`/`T` en `config.py`) y flujos `Marcadores`, `Paso`, `Estado` → `tablero.py` y LabRecorder.

## Comandos

```bash
python pruebas.py                     # rápidas (~30 s)
python pruebas.py --completa          # + lazo real contra el gemelo (~3 min)
python simulador_lazo.py              # comparación de agentes (30 sujetos)
python cerebro_sintetico.py --banco   # decoder y detector offline
python cerebro_sintetico.py           # terminal 1: gemelo
python tablero.py                     # terminal 2
python orquestador.py real --ortesis-sim   # terminal 3: camino real completo
python orquestador.py sim --ciclo 0 --caos 1   # caos estándar en el simulador (--caos-nivel leve: una falla cada 2-3 min)
python orquestador.py real --ortesis-sim --reanudar   # continuar una sesión interrumpida
```

## Cosas que muerden

- **El umbral de renovación del EEG es 1 s** (`config.SALUD['eeg_renovar_s']`, `--renovar-eeg`). No bajarlo sin medir el dongle: `puente_lsl.py` imprime el registro de huecos cada 30 s.
- **Un solo flujo `EEG` en la red.** Un `cerebro_sintetico.py` olvidado en otra terminal contamina cualquier medición contra el gemelo. Antes de medir: `python ver_flujos.py`.
- `pruebas.py --completa`: `lazo_real_sintetico` excedió sus 400 s una vez de tres (2 de octubre) y no se reprodujo; si vuelve a pasar, la prueba ya muestra las últimas líneas de la sesión.
- **No correr `pruebas.py` mientras hay una sesión `real` en marcha:** las pruebas publican flujos `Marcadores` y `Paso` con los mismos nombres.
- Un paso con `excluido` no vacío queda fuera de `EVALUACION`; el agente no aprendió de él.

## Resultados de referencia (para no retroceder)

Medidos el 2 de octubre de 2026 en la máquina de Luis (Windows 11, Python 3.11, numpy 2.5.3, scikit-learn 1.9.1), salvo donde se indica. En otra plataforma o con otras versiones pueden variar en el tercer decimal.

- Simulador (`python simulador_lazo.py --semillas 30`), perturbación de 2.4 logits — error en los primeros 2 min tras perturbar: estático 0.325, eta fijo 0.246, **bayes 0.214**. Antes de perturbar: 0.165 / 0.170 / 0.169; después de los 2 min: 0.316 / 0.178 / 0.175.
- Con el detector degradado a propósito (prueba `confianza_detector`, 12 sujetos): aprendizaje al 8 %, congelado 88 % de la falla, 1 % de congelamientos en falso.
- Banco offline del gemelo (`python cerebro_sintetico.py --banco`): decoder MI BA 0.71; detector ErrP sensibilidad 0.66, especificidad 0.91, BA 0.79.
- Corrida real completa contra el gemelo (una corrida, 2 de octubre): CP1–CP4 en GO (MI BA 0.92, ErrP BA 0.92); error tras perturbar agente 0.26 vs sombra 0.46; ninguna pausa en falso. La calibración no usa semilla fija, así que varía entre corridas (una medida anterior dio 0.14 vs 0.47).
- Caos, simulador (30 sujetos, `orquestador.py sim --caos`): tras perturbar agente/sombra 0.222/0.333 con el estándar (1109 de 11 425 filas excluidas) y 0.232/0.326 con el leve (105 de 10 867); sin caos 0.235/0.323.
- Caos, gemelo (una corrida por condición, mismos modelos que la limpia, renovación a 1 s): leve 0.37/0.47 con 1 pausa; estándar 0.37/0.47 con 10 pausas, todas reanudadas; CP4 en GO. No concluyente. Con renovación a 0.3 s, tres corridas estándar dieron 0.44/0.49, 0.30/0.53 y 0.46/0.47.

Las tareas pendientes, en orden, están en `TAREAS.md`.
