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
| `agente_errp.py` | `AgenteErrP` (filtro de Kalman sobre la corrección `beta` del logit; P_hat bayesiano; detectores de cambio por sesgo y chequeo predictivo) y `ConfianzaDetector` (sens/espec vivas del detector de ErrP con posteriores Beta; congela el aprendizaje si el detector deja de informar). |
| `simulador_lazo.py` | Piloto sintético rápido a nivel de rasgos, para comparar agentes con muchos sujetos. |
| `cerebro_sintetico.py` | Gemelo digital del piloto: publica EEG por LSL que reacciona al lazo (ERD mu/beta en C3, ErrP fronto-central ante movimientos erróneos, parpadeos, fatiga). Tiene banco offline (`sesion_mi`, `sesion_errp`, `--banco`). |
| `hardware.py` | `EntradaEEG` (LSL con buffer y reloj sincronizado), `OrtesisSerial`/`OrtesisSimulada` (ACK con latencia medida), `DecoderIM` (Riemann con recentrado no supervisado), `DetectorErrP` (fusión temporal + geométrica, calibrado, umbral de Neyman-Pearson, detector de rareza), calibración secuencial (`intervalo_ba`). |
| `orquestador.py` | Máquina de estados, checkpoints go/no go (CP1-CP4), calibraciones, lazo, CSV, flujos `Marcadores`/`Paso`/`Estado`. Backends `sim` y `real`. |
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
```

## Resultados de referencia (para no retroceder)

- Simulador, 30 sujetos, perturbación de 2.4 logits — error en los primeros 2 min tras perturbar: estático 0.325, eta fijo 0.243, **bayes 0.211**.
- Con el detector degradado a propósito: aprendizaje al 8 %, congelado 88 % de la falla, 1 % de congelamientos en falso.
- Corrida real completa contra el gemelo: CP1–CP4 en GO; error tras perturbar agente 0.14 vs sombra 0.47.

Las tareas pendientes, en orden, están en `TAREAS.md`.
