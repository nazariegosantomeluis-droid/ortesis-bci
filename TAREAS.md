# Tareas pendientes (en este orden)

Lee `CLAUDE.md` primero. Para cada tarea: diseña, implementa, **mide en el gemelo o el simulador**, agrega pruebas, actualiza README y haz commit. Si una decisión de diseño no está clara, pregúntale a Luis antes de seguir.

---

## Bloque final: la IA como tercera escala de aprendizaje (3 a 5 de octubre de 2026)

Especificación de Luis del 3 de octubre por la noche. **Sustituye** dos decisiones anteriores: el código ya no se congela el sábado a las 20:00 y PhysioNet ya no está fuera.

### Fechas y ramas

- Domingo 4 de octubre: demo con el casco, con la etiqueta **`v-demo` intacta**.
- Lunes 5 de octubre: **el código se congela a las 10:00**; el evento termina a la 1 p.m.
- Todo lo nuevo entra a `main` **detrás de banderas apagadas por defecto**. El domingo en la noche Luis decide qué se enciende para la final.
- Sin retroalimentación táctil: no habrá motores de vibración.

### Visión para la presentación (documentarla así en el README)

El sistema aprende en tres escalas de tiempo:

| Escala | Quién | Cada cuánto |
|---|---|---|
| Rápida | Agente bayesiano | cada paso |
| Media | Detector co-adaptativo | cada ~20 pasos |
| Lenta | Claude como co-investigador | entre bloques |

### Reglas para toda la IA

- Llave en `ANTHROPIC_API_KEY`, leída de un `.env` que está en `.gitignore`.
- Verificar en la documentación oficial el modelo vigente y la API de herramientas.
- A la API solo van métricas agregadas y anónimas: nunca EEG crudo ni nombres.
- Nada de la IA dentro del lazo de control.
- Sin conexión o sin llave, todo funciona con plantillas o reglas deterministas.
- Todo lleva prueba en `pruebas.py` que funciona con la API simulada.

### Orden de trabajo

#### 1. Control causal en vivo con sham (`--sham`)

**Objetivo:** demostrar frente al jurado que es el ErrP del piloto lo que corrige la máquina.

- Analizar la rama `origin/b1-b2-sham-errp` de jusren (`c71f477`), quedarse con las ideas útiles, escribir implementación propia y acreditar la idea a jusren en el README. Luis (3 de octubre): sus dos controles (B1, la órtesis se mueve sola con el piloto en reposo y `p(t)` no debe seguirla; B2, especificidad del ErrP por dirección) son muy útiles y también deben quedar en `main`, con implementación propia.
- Dos bloques adaptativos del mismo largo, real y sham, en orden contrabalanceado (el orden se elige al azar y se registra). Cada bloque arranca con beta, varianza y prior reiniciados y sin perturbación, y recibe su propia perturbación de 2.4 logits en el mismo paso relativo.
- En el bloque sham el agente recibe los `p_errp` permutados al azar entre los pasos recientes del mismo bloque: misma distribución, sin relación con los errores reales. En ese bloque el `ConfianzaDetector` no congela ni escala el aprendizaje (fiabilidad fija en la calibrada), para que el contraste sea limpio: el agente aprende igual de rápido, solo que de una señal sin información.
- Ciego simple: el piloto no sabe qué bloque es cuál; el tablero lo muestra solo cuando el operador lo pide.
- Marcadores `bloque:real` y `bloque:sham` en el contrato. `EVALUACION` y el tablero comparan los bloques lado a lado: error tras perturbar con intervalo del 90 %, tiempo de recuperación y si se recuperó.
- Duración: cada bloque de 60 a 80 pasos, para que los dos quepan en unos 4 minutos.
- **Criterio de aceptación** (gemelo, 16 sesiones, duración de la demo): el bloque real se recupera en al menos 12 de 16 y el sham en 3 de 16 o menos; la diferencia de error tiene un intervalo que excluye el 0. Si con esa duración no se alcanza, reportar a Luis cuánto haría falta **antes de alargar nada**.

#### 2. Copiloto clínico (`copiloto.py`, `--copiloto`)

**Objetivo:** preguntas en lenguaje natural sobre una sesión, respondidas solo con los datos reales.

- Herramientas que consultan el CSV, los marcadores y los checkpoints:
  - `resumen_sesion()`;
  - `eventos(desde, hasta, tipo)`: pausas, congelamientos, cambios detectados, perturbaciones, checkpoints, cambios de semáforo;
  - `metrica(nombre, bloque)`: error del agente y de la sombra con intervalo, tiempo de recuperación, BA viva, fiabilidad, latencia del ACK, excluidos por motivo, alfa occipital;
  - `pasos(desde, hasta, columnas)`: filas puntuales, con un límite de filas;
  - `comparar_sesiones(rutas)`.
- Preguntas que debe responder bien, como prueba: «¿por qué se congeló el aprendizaje en el paso N?», «¿cuánto tardó en recuperarse tras la perturbación?», «¿el agente le ganó a la sombra y con qué certeza?», «¿hubo señales de fatiga?», «¿cuántos pasos se excluyeron y por qué?» y «¿cómo se compara con la sesión anterior?».
- Toda respuesta cita los pasos y valores que usó. Si el dato no existe, lo dice; nunca inventa.
- Interfaz: `python copiloto.py --sesion <csv> "pregunta"`, un modo interactivo y una caja de texto en el tablero.
- Informe entre sesiones: compara la sesión con las anteriores del mismo piloto y genera dos versiones en Markdown, una para el terapeuta y otra sencilla para el paciente, en español e inglés, con las figuras clave. Incluye una propuesta para la próxima sesión en JSON, con el mismo esquema y la misma validación de rangos seguros que el co-investigador, y requiere aprobación humana.
- Pruebas con la API simulada: las herramientas devuelven las cifras correctas, una pregunta sin datos produce «no hay dato» y una propuesta fuera de rango se rechaza.

#### 3. Co-investigador entre bloques

- Al terminar cada bloque, Claude recibe un resumen agregado (exactitud, error contra sombra, BA viva, fiabilidad, pasos excluidos y motivos, alfa occipital, pasos sin ErrP por tamaño de paso) y devuelve una propuesta en JSON con esquema fijo: acción (continuar, pausa, ajustar parámetro, recalibrar) y justificación breve.
- El código valida cada parámetro contra rangos seguros en `config`.
- El tablero muestra la propuesta con botones Aprobar y Rechazar, y todo queda registrado (propuesta, decisión, efecto).
- Sin conexión o sin llave se usa un conjunto de reglas determinista equivalente.
- **Nunca modifica nada sin aprobación humana.**

#### 4. Narrador para el jurado

- Un proceso aparte que escucha el flujo `Estado` y, ante eventos relevantes (perturbación, congelamiento, pausa, recuperación, checkpoints), pide a Claude una frase corta en español o inglés basada solo en los datos del evento.
- Nunca bloquea el lazo; si la API tarda o falla, usa plantillas.

#### 5. Transferencia con PhysioNet

- EEGMMIDB vía `mne.datasets.eegbci`, corridas de imaginación motora (`mne` entra como dependencia).
- Pre-entrenar «mano derecha imaginada contra reposo» con los 8 canales del Unicorn remuestreados a 250 Hz, adaptar con recentrado riemanniano y medir cuántos ensayos de calibración ahorra.
- Si funciona, que la calibración pueda arrancar desde el modelo pre-entrenado.

### Decisiones de implementación y hallazgos (se anotan aquí conforme salen)

**1. Sham (3 de octubre, noche; cifras del gemelo, `estudios/sham_gemelo.py`, 16 sesiones).**

- **Hallazgo que cambia el diseño: el sham que permuta los `p_errp` no cumple el criterio y no puede cumplirlo.** Con 60 pasos por bloque y el detector actual, el bloque real se recupera en 16 de 16 y el sham permutado en 11 de 16 (con 70 pasos, 15 de 16). Tras la perturbación casi todas las decisiones van al mismo lado, y la sola tasa de ErrP ya dice hacia dónde corregir; la permutación conserva la tasa. Un sham con «la misma distribución» conserva justo la información que usa el agente.
- **Decisión tomada, pendiente del visto bueno de Luis:** el sham por defecto quita la evidencia del ErrP (`config.SHAM_ERRP_FUENTE = 'nula'`: tasa base de la calibración, LLR = 0; es el control negativo de siempre, ahora en vivo). Con él: real 16 de 16, sham 0 de 16, sham − real +0.136 [+0.111, +0.161]. El permutado queda disponible con `--sham-fuente recientes`. Se probó una tercera fuente (los `p_errp` del bloque estático, permutados) y quedó en medio (sham 3 a 8 de 16): descartada.
- **Duración:** 60 pasos por bloque alcanzan, pero solo con la perturbación en el paso 10 (50 pasos después). Con la perturbación a un tercio del bloque (paso 20 de 60) el real se recuperaba en 10 de 16; harían falta 80 pasos (5.6 min los dos bloques). Se eligió perturbar en el paso 10 (`config.SHAM_ERRP_PERTURBAR_EN`) en lugar de alargar.
- **Con el detector débil el criterio no se cumple** (real 7 de 16): es el detector, no el sham. Si el CP3 da NO GO el domingo, el contraste en vivo no está garantizado.
- El `ConfianzaDetector` no se reinicia entre bloques (lo que sabe del detector no es del bloque) y sigue midiendo durante el sham, aunque ahí no decide nada. Los bloques del control causal no llevan movimientos ajenos. El orden sale de `random.SystemRandom` (no de `--semilla`, que por defecto es 0 y daría siempre el mismo); `--sham-orden` lo fija para las pruebas.
- Una sola sesión no da para el intervalo de la diferencia de error (±0.2 con 50 pasos): en vivo se muestra recuperó / no se recuperó y el intervalo se reporta tal cual.
- Columna nueva `bloque` en el CSV (contrato), después de `estado`.

---

## Cambio de hardware (2 de octubre de 2026): g.tec Unicorn Hybrid Black

El domingo se usa un Unicorn Hybrid Black, no un Cyton: 8 EEG (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8), 250 Hz, Bluetooth, acelerómetro y giroscopio de 3 ejes, batería y contador de muestras. **El casco no llega hasta el domingo**: todo se prepara con el gemelo.

### Verificado

- BrainFlow 5.23 (instalado): `BoardIds.UNICORN_BOARD`; `Unicorn.dll` viene en el paquete; `serial_number` opcional; hay que emparejar el casco con **su dongle**, no con el Bluetooth de la laptop. Filas: EEG 0–7, acelerómetro 8–10, giroscopio 11–13, batería 14, contador 15, validez 16; 250 Hz.
- UnicornLSL (código fuente de g.tec): un flujo de tipo `Data`, 17 canales `float32` a 250 Hz, **sin etiquetas**, una muestra por envío y sin marca de tiempo propia (LSL estampa la llegada). El nombre es el que se escriba en la app o, vacío, el número de serie. Modo dividido: `<nombre>_EEG` (tipo `EEG`), `_ACC`, `_GYR`, `_CNT`, `_BAT`, `_VALID`.
- Solo una aplicación puede conectarse al casco a la vez.

### Sin verificar hasta tener el casco (lo comprueba `verificar_unicorn.py`)

Orden de los 17 canales del flujo combinado, unidades del EEG, contador de 1 en 1, acelerómetro ~1 g en reposo, batería, validez, rango de ±750 mV, y que `Unicorn.dll` conecte sin la Unicorn Suite.

### Decisiones de Luis

1. **Fuente:** BrainFlow por `puente_lsl.py` como principal; UnicornLSL de respaldo.
2. **Canales:** no se fija el montaje con el gemelo (sería circular). La calibración real elige por validación cruzada entre los canales del papel y los 8, por separado para el decoder (C3/Cz/C4 contra 8) y para el detector (Fz/Cz/Pz contra 8), y registra la elección. El gemelo solo sirve para probar que la selección funciona.
3. **Movimiento de cabeza:** el paso se marca como artefacto (el agente no aprende, no se recentra); sin pausa.
4. **CP1 robusto:**
   - el caos solo actúa desde `LAZO_ESTATICO`, con la opción `--caos-desde calibracion`;
   - 40 movimientos y tres métricas con umbral en `config`: MAD ≤ 15 ms, p95 ≤ 60 ms, ACK perdidos ≤ 10 %;
   - prueba: 39 normales más un pico de 300 ms da GO; jitter típico de 40 ms da NO GO.
   - Sin impedancias: calidad de señal por canal.
5. **Bug de `P_hat`:** en el bloque estático, y con el aprendizaje congelado, se pasa fiabilidad 0 al agente y con salida calibrada `P_hat` queda igual al prior, sin reflejar el ErrP.
6. **Brecha calibración → lazo:** en `resultados/sesion_real_20261002_114317.csv`, MI con BA 0.88 en calibración contra 0.37 de error en el bloque estático; ErrP con especificidad 0.96 calibrada contra 0.72 en vivo. Investigar (recentrado en línea, tiempos de ventana calibración contra lazo, paso fijo de 0.15 en `CAL_ERRP` contra pasos variables) y **reportar la causa antes de corregir**.
7. **Alcance:** P0 más Tarea 2 mínima. De la Tarea 3 solo corregir que la órtesis casi nunca cierra completa. Transferencia con PhysioNet: fuera. Si hay que recortar dentro del P0, lo primero que sale es el semáforo PILOTO.

### Riesgos del domingo

- **Detector de ErrP entre 0.70 y 0.75.** El montaje del Unicorn tiene 3 electrodos fronto-centrales (Fz, Cz, Pz) en lugar de 5; en el gemelo el detector bajó unos puntos. El umbral del CP3 (`config.BA_MIN`) **se queda en 0.75** por decisión de Luis; solo la prueba del gemelo acepta 0.70. Si el detector real queda entre 0.70 y 0.75, el CP3 dará NO GO: tener listo el plan B.
- **Nada se ha probado con el casco.** `verificar_unicorn.py` es lo primero que se corre el domingo.
- **El equipo se suspende por inactividad y al cerrar la tapa.** Desactivar la suspensión antes de la demo.

### Brecha calibración → lazo: causa medida (2 de octubre, tarde; cifras del gemelo)

Scripts en `estudios/`. **Reportado a Luis y corregido** (tabla de abajo).

- **Descartado:** el recentrado en línea y las respuestas cerebrales a cada movimiento dentro de la ventana de MI (ablaciones: ±0.01 de error); el traslape de épocas de ErrP con un paso cada 0.87 s contra 2.5 s (especificidad 0.95 contra 0.96); una carrera en el gemelo entre el flujo `Paso` y el ACK (0 de 220 movimientos juzgados con la dirección equivocada).
- **MI:** (1) la BA que reporta la calibración secuencial es optimista: 0.84 reportada contra 0.80 real (+0.04 de error; +0.06 si para a los 24 ensayos); (2) el primer paso de cada ensayo falla más (0.25 contra 0.18), porque su ventana empieza con la señal e incluye la transición; esperar 2 s más lo baja a 0.12; (3) un bloque estático de 30 pasos tiene desviación de 0.08: el peor de 10 bloques da 0.33. Sesión real de 150 pasos con el montaje nuevo: BA 0.83 en calibración, error 0.20 en el lazo. El 0.37 de la sesión de las 11:43 queda en la cola de ese ruido; la subida de `p'` paso a paso no se reprodujo ni en 300 pasos sin LSL ni en 150 en vivo.
- **ErrP:** la causa es la calibración secuencial. Con el montaje del Unicorn solo 2 de 16 sujetos dan GO, a las 40 a 60 épocas, y lo hacen por suerte: reportan BA 0.87 y especificidad 0.90, y en épocas nuevas dan 0.69 y 0.81. Otros 11 de 16 reciben un NO GO temprano a las 60 épocas, aunque con 120 llegarían a una BA real de 0.72. Con 120 épocas lo reportado coincide con lo real (0.74 contra 0.75). Además, `espec_viva` tiene memoria corta (~14 aciertos) y fluctúa ±0.1.
- **No se pudo probar con el gemelo:** paso fijo de 0.15 en `CAL_ERRP` contra pasos variables en el lazo (el gemelo no hace depender el ErrP del tamaño del paso). Requiere una persona.
- **Bug de `P_hat`:** corregido. `actualizar()` separa la fiabilidad (con la que se calcula `P_hat`) del peso de aprendizaje.

### Hecho el 2 de octubre (noche) y decisiones tomadas

| Pieza | Cifra clave | Decisión |
|---|---|---|
| ErrP: 120 épocas fijas + umbral anidado | reportado 0.70 vs real 0.73 (16 sujetos) | `--min_errp` quitado |
| MI: mínimo 36 ensayos | optimismo +0.02 → +0.00 | |
| Espera de 1 s tras la señal | primer paso 0.22 → 0.15 (~80 % de lo que da 2 s) | `config.ESPERA_PRIMER_PASO_S` |
| Intervalos del 90 % en EVALUACION | remuestreando ensayos de 5 pasos | bootstrap por ensayo, no por paso |
| CP1 sin impedancias | MAD ≤ 15, p95 ≤ 60 ms, ≤ 10 % sin ACK | estado sigue llamándose `IMPEDANCIAS` |
| Selección por validación cruzada | canales (papel / 8) y vistas (dos / tres) | elección anidada; en empate gana la más simple |
| Vista theta | gemelo: el detector la elige cuando hay theta | |
| Detector co-adaptativo | gemelo: BA en vivo 0.65 → 0.73 | prueba en sombra de 30 épocas (20 era muy ruidoso) |
| Época al inicio real (telemetría) | gemelo: BA 0.61 (ACK) → 0.72 (telemetría) | sin telemetría: ACK + latencia media |
| Curva de robustez | BA ≥ 0.75: recupera en ~60 s; 0.65: 162 s | figura en `docs/figuras/` |
| `P_hat` | no frena el aprendizaje (±0.006, 30 sujetos) | `estudios/efecto_p_hat.py` |
| Tarea 2 mínima (IIC, exploratorio) | gemelo, 16 sujetos: sin sesgo; con 12 ajenos no ordena el embodiment (−0.12), con 30 apenas (+0.24) | criterio de aceptación no cumplido por potencia: decide Luis (ver Tarea 2) |
| Semáforo PILOTO | gemelo con fatiga 1: alfa ×2.8 a los 15 min | solo avisa; umbrales ×1.5 y ×2.5 sin validar |
| Órtesis cierra completa | simulador, 30 sujetos: cierra 0.27 → 0.68, abre 0.41 → 0.75; error igual (0.214) | cada ensayo desde 0.5 (400 ms antes del cue, marcador `centrado`, también tras pausa); paso máx. y ganancia 0.20 → 0.30; en `agente_errp.py` solo la ganancia por defecto pasa a leerse de `config` |

**Abierto:** en el gemelo con el montaje del Unicorn el agente no supera a la sombra en los 2 min tras perturbar (0.47 vs 0.47 en dos corridas); en el simulador sí (0.21 vs 0.31). Primer análisis, solo con los CSV de esas corridas (2 de octubre, noche): el agente **aprende lento**. Beta sube +0.2 a +1.2 en los 20 pasos tras perturbar, contra +1.8 a +2.0 con el montaje anterior. No se congela: fiabilidad media de 0.92 a 0.96. En 2 de 3 corridas no cambió ni una decisión respecto a la sombra en esos 57 pasos. Coincide con una BA viva del detector de 0.72–0.74 al perturbar (antes 0.86–0.87), como predice la curva de robustez, pero una corrida con 0.84 también fue lenta. **Explicado el 3 de octubre** con `estudios/agente_lento.py` (ver "Resultado de la prioridad 1" más abajo): la causa es el prior de error del agente, no el modo de salida del detector.

### Sábado 3 de octubre: plan de Luis

**A las 20:00 del sábado se congela el código.** Después de esa hora, solo correcciones de errores.

1. **Prioridad 1 (máximo 3 h): el agente lento en el gemelo.** Hipótesis principal: el simulador usa la salida binaria del detector y el modo real, la calibrada. Si la calibración de Platt sobre un detector débil comprime las probabilidades cerca de la tasa base, cada ErrP aporta poca evidencia y beta se mueve poco, incluso con BA alta. Medir en el gemelo, con 4 semillas: (1) el agente en modo binario contra calibrado sobre las mismas épocas; (2) la distribución de `P_hat` y del LLR tras la perturbación; (3) un diagrama de confiabilidad del detector calibrado. Reportar la causa con números y proponer la corrección.
2. Después, en este orden: el detector co-adaptativo, la alineación al movimiento real con respaldo por ACK y theta solo si sobra tiempo.

#### Resultado de la prioridad 1 (3 de octubre, mañana; `estudios/agente_lento.py`, figura `docs/figuras/agente_lento.png`)

Lazo completo del gemelo sin LSL con la aritmética de `Orquestador.paso`; 4 sujetos × 4 lazos = 16 sesiones por variante, mismo ruido en todas las variantes. Tres regímenes: detector **actual** (elige canales y vistas; BA viva 0.82), detector **de ayer** (gemelo sin theta, 8 canales y dos vistas; 0.74) y **débil** (ErrP de 4 µV; 0.68). Son cifras del gemelo, no de una persona.

- **La hipótesis (binaria contra calibrada) no se confirma.** La salida binaria no es más rápida: error del agente en los 2 min tras perturbar 0.28 contra 0.28 (actual), 0.31 contra 0.29 (ayer; +0.019 ± 0.012 pareado) y 0.36 contra 0.32 (débil; +0.044 ± 0.018). Con detector débil es peor.
- **Platt sí comprime con el detector de ayer, pero no es la causa principal.** Pendiente de calibración 1.31 (1 = calibrado; rango de `p_errp` 0.11 a 0.83) contra 1.05 con el detector actual (0.03 a 0.96). El LLR por error baja de +1.49 (binario) a +1.03 (calibrado). Recalibrar en escala logit mejora poco: −0.016 ± 0.012 y −0.021 ± 0.015 de error.
- **La causa es la evidencia débil del detector contra el prior de error del agente.** `P_hat = sigmoide(logit(prior) + fiabilidad × LLR)`, y el prior (0.2 al empezar) se actualiza con el propio `P_hat` y α = 0.03. Tras la perturbación la mitad de los pasos son errores, pero el prior solo sube a 0.24–0.25. Con prior 0.23 un error necesita LLR > 1.2 para que `P_hat` pase de 0.5: lo logra el 65 % de los errores con el detector actual, el 44 % con el de ayer y el 24 % con el débil. Si `P_hat` no pasa de 0.5, beta se atasca donde p′ ≈ `P_hat` y las decisiones no cambian. Con prior 0.5, sobre las mismas épocas, pasan el 86 %, 78 % y 75 %.
- **Coincide con las corridas lentas por LSL del 2 de octubre:** `P_hat` medio de los errores reales 0.30–0.36 y solo 7–26 % arriba de 0.5, con sens viva 0.67–0.72 y espec 0.85–0.91.
- **Corrección que parecía la mejor y se DESCARTA: subir el prior a 0.5 cuando dispara cualquier detector de cambio.** Es la más rápida (gemelo: error en los 2 min 0.280 → 0.246, 0.294 → 0.257 y 0.316 → 0.274; simulador: recuperación 76 → 44 s y, con detector débil, 179 → 41 s), pero **falla el control negativo**: con un detector sin información (simulador, sens 0.12 y espec 0.90) el agente se recupera igual, 30 de 30 sujetos en 102 s, cuando sin la corrección no se recupera (1 de 30), como debe ser. Lo que mueve a beta ahí no es el ErrP: es el detector de sesgo (que supone metas balanceadas) más un prior de 0.5 que jala todas las decisiones hacia el centro. Rompe el principio del agente ("el ErrP decide qué aprender; los detectores de cambio, cuándo") y no se puede presentar como aprendizaje por ErrP.
- **El mismo control en el gemelo** (la compuerta de confianza abierta con el detector real, pero el agente recibe LLR = 0): el agente de hoy casi no se recupera sin la evidencia del ErrP (4 de 16 sesiones contra 13 de 16 con detector débil y contra 15 de 16 con el de ayer; error +0.121 ± 0.019 y +0.127 ± 0.024). Es decir, **el agente de hoy sí aprende del ErrP**. Con la corrección descartada se recupera igual sin el ErrP (16 de 16 en los dos regímenes).
- **Candidatas que sí pasan el control:** subir el prior solo cuando el cambio lo detecta el chequeo predictivo, que depende del ErrP. No mejora nada medible en el gemelo (−0.004 ± 0.005 y +0.005 ± 0.004) y poco en el simulador (detector débil: 179 → 145 s, −0.011 ± 0.008). El prior por momentos falla a medias el control en el simulador (17 de 30) y un prior más rápido (α 0.10) empeora con detector débil (+0.021 ± 0.010).
- Sin efecto medible: usar el umbral del detector en el chequeo predictivo (hoy cuenta `p_errp > 0.5`; entre el 66 y el 78 % de las detecciones también pasan de 0.5) y no escalar el LLR por la fiabilidad (±0.005).
- **Conclusión: no se cambia el agente.** La lentitud de las corridas del 2 de octubre es la consecuencia honesta de un detector débil (el de entonces: 8 canales, dos vistas, BA viva ~0.74 y probabilidades comprimidas). Con el gemelo y el detector de hoy la BA viva es 0.82 y el agente recupera en 19 pasos de mediana, 16 de 16 sesiones; parte de esa mejora viene de la theta tras el error, que programamos nosotros en el gemelo, así que no dice cómo será con una persona. Lo que acelera de verdad es un detector mejor, como dice la curva de robustez; el domingo lo decide la BA del detector real (CP3).
- **Decisiones de Luis (3 de octubre, mediodía):** el control negativo es el resultado central de la presentación. El detector co-adaptativo se queda encendido por defecto, con `--sin-coadaptativo` para apagarlo, y ningún fallo suyo puede detener el lazo (hecho: `observar()` no lanza, los fallos quedan en `errores` y con 3 seguidos se apaga). Theta se queda como está.
- **Respaldo por ACK en lazo cerrado** (`estudios/respaldo_ack.py`, 16 sesiones por escenario): con telemetría, BA viva del detector 0.87 y error del agente 0.274 (16 de 16 se recuperan, mediana de 16 pasos); sin telemetría nunca, 0.71 y 0.346 (+0.072 ± 0.022; 15 de 16, 25 pasos); si se pierde en el lazo y se usa ACK + latencia media, 0.79 y 0.353 (+0.079 ± 0.021; 15 de 16, 23 pasos). El respaldo funciona; la telemetría vale ~0.07 de error.
- **Detector co-adaptativo en lazo cerrado** (mismas sesiones con y sin él): no cambia el error del agente (+0.001 ± 0.007 con el detector de ayer, −0.001 ± 0.009 con el débil), sube la BA viva tras perturbar de 0.77 a 0.79 y de 0.68 a 0.71, con ~2 cambios de modelo por sesión. Ni ayuda ni estorba en un bloque de 120 pasos.

#### Tarde del 3 de octubre: cierre para la demo

Pedido por Luis, en orden: (a) figuras de la presentación en `docs/figuras/` (curva de robustez, control negativo, potencia del IIC); (b) cifras de referencia de README y CLAUDE.md con la configuración final; (c) `--completa`, merge a `main`, etiqueta `v-demo` y push; (d) `docs/DOMINGO.md`. Además, que en el repositorio solo quede la rama `main`.

Decisiones tomadas sin preguntar (se pueden revertir):

- **Plan B de verdad.** El que había (reproducir el EEG crudo con `puente_lsl.py --placa playback` y correr el orquestador igual) no sirve para el lazo: lo grabado no responde a las señales nuevas. Ahora cada sesión graba junto a su CSV todo lo que publica al tablero (`..._estado.jsonl`) y `repetir_sesion.py` lo repite en el tablero y, con `--puerto`, en la órtesis.
- **`--solo-errp`:** tras un CP3 NO GO repite solo la calibración de ErrP con el decoder de MI guardado.
- **Aviso de edad de los modelos:** al cargar modelos guardados se dice hace cuánto se calibraron y se avisa si pasan de 6 h. En `modelos/` quedan los del gemelo tras las pruebas; el guion del domingo empieza borrándolos.
- **El banco del gemelo** (`cerebro_sintetico.py --banco`) calibra como la calibración real (elige canales y vistas).
- **`docs/DOMINGO.md` se escribió antes de la etiqueta**, para que `v-demo` lo incluya.
- **Pendiente de aclarar con Luis:** la tabla de checkpoints del README dice que si falla el CP4 hay que "congelar y usar la ruta de 24 h"; no sé a qué se refiere y `docs/DOMINGO.md` no lo menciona.

#### Hallazgo de la tarde: los pasos que no mueven la órtesis (`estudios/paso_sin_movimiento.py`)

Al volver a medir contra el gemelo, la sesión con caos leve no se recuperó. Buscando por qué apareció un hueco del lazo que el gemelo tapaba.

- **El hueco.** Con cada ensayo desde el punto medio y pasos de hasta 0.30, la órtesis llega al tope en el segundo paso y los siguientes no la mueven (14 a 28 % de los pasos en las sesiones contra el gemelo; ~27 % en el simulador). Una persona no ve nada ahí y no produce ErrP. El gemelo reaccionaba al marcador `paso_ack` aunque no hubiera movimiento, así que el problema no se veía. Con una persona, el lazo habría leído esos pasos como "sin ErrP": refuerza la decisión y cuenta como detección fallida en la confianza del detector. Tras la perturbación la órtesis se queda en el tope equivocado, justo cuando el agente necesita aprender.
- **La calibración de ErrP tenía el mismo hueco:** ~7 % de los ensayos (0 a 18 % según la sesión) no movían la órtesis porque el recorrido de la calibración chocaba con 0.1 o 0.9.
- **Corrección (decisión tomada sin preguntar; el aprendizaje del agente no se tocó, cambia qué pasos se le dan).** `Orquestador.paso`: si el movimiento es menor que `PASO_VISIBLE`, no hay época, N1 ni aprendizaje, y el paso no cuenta para la confianza del detector; queda con `alineacion = sin_movimiento`, marcador `paso_quieto:<seq>` y sigue contando como decisión en el análisis. `calibrar_errp`: la órtesis vuelve al centro cuando el movimiento no cabe. El gemelo ya no reacciona a esos pasos (solo escucha `paso_ack`). Se revierte con `config.IGNORAR_SIN_MOVIMIENTO = False`.
- **Evidencia** (gemelo sin LSL con los topes del recorrido, 16 sesiones por celda; error del agente en los 2 min tras perturbar y sesiones recuperadas): detector fuerte (BA viva 0.83): leyendo esos pasos 0.409 y 9 de 16, con el aprendizaje congelado 21 % del tiempo; ignorándolos 0.319 y 16 de 16, congelado 0 %. Detector medio (0.73): 0.376 y 11 de 16 contra 0.389 y 10 de 16 (congelado 13 % contra 5 %). Débil (0.67): 0.391 y 7 de 16 contra 0.400 y 11 de 16 (26 % contra 5 %). Es decir: ayuda mucho con un detector bueno; con uno medio o débil el error no cambia y deja de congelarse en falso.
- **Lo que cambia en lo ya reportado.** El estudio del agente lento suponía que todo paso se ve. Con los topes, un tercio de los pasos no informa y el agente es más lento: detector fuerte, 0.319 en lugar de 0.280 (28 pasos de mediana en lugar de 19, 16 de 16); medio, 0.389 en lugar de 0.294 (10 de 16 en lugar de 15); débil, 0.400 en lugar de 0.316 (11 de 16 en lugar de 13). **El control negativo se sostiene y queda más limpio: sin la evidencia del ErrP se recupera 1 de 16 con los tres detectores** (antes 3 o 4 de 16). La figura `control_negativo.png` se rehízo con estos números. Las cifras de la presentación cambian de "sin ErrP 4/16, con ErrP 13/16" a "sin ErrP 1/16, con ErrP 10 a 16 de 16 según el detector".
- **Detector co-adaptativo con la configuración final** (mismas sesiones, con y sin él): −0.005 ± 0.004, −0.024 ± 0.012 y −0.008 ± 0.006 de error (fuerte, medio, débil); recuperan 16, 14 y 11 de 16 contra 16, 10 y 11. No empeora: se queda encendido, como decidió Luis.
- **La sesión con caos leve, repetida dos veces antes de la corrección** (mismos modelos y misma semilla del caos): con el detector fijo se recuperó en 20 pasos (0.23 contra 0.54 de la sombra); con el co-adaptativo, en 44 pasos (0.28 contra 0.54). La original no se recuperó. La variación entre corridas de una sola sesión es grande: por eso las conclusiones salen del estudio de 16 sesiones y no de una corrida.
- **Sesiones reales contra el gemelo con la corrección (una corrida por condición, mismos modelos):** calibración con MI BA 0.88 y ErrP sens 0.83, espec 0.87, BA 0.85 (CP3 en NO GO por especificidad; se siguió con ese detector). Sin caos: agente 0.39 contra sombra 0.54, recupera en 28 pasos = 50 s. Caos estándar: 0.32 contra 0.49, 10 pausas reanudadas, 44 s. Caos leve: 0.23 contra 0.53, 1 pausa, 45 s. Las tres en GO del CP4; entre 23 y 38 pasos por sesión no movieron la órtesis.
- **Simulador con caos (30 sujetos), con la corrección:** agente/sombra 0.225/0.296 sin caos, 0.220/0.289 con el leve y 0.223/0.296 con el estándar; el 27 % de los pasos no mueve la órtesis (casi todos aciertos: el simulador se equivoca con pasos chicos).
- **Respaldo por ACK, medido otra vez con los topes** (`python estudios/respaldo_ack.py 4 4 ignorar`, 16 sesiones por escenario): con telemetría, BA viva 0.86, error 0.329 y 16 de 16 recuperadas (26 pasos); sin telemetría nunca, 0.72, 0.404 (+0.075 ± 0.030) y 10 de 16; perdida en el lazo, 0.81, 0.338 (+0.009 ± 0.026) y 12 de 16. El respaldo sigue funcionando; la telemetría importa más de lo que decía la medición sin topes.
- **`--completa` antes del merge:** la primera corrida dio 53 de 54; falló `entrada_unicorn`. No era la entrada de EEG: la prueba suponía dos cosas que en el gemelo son al azar (que la ventana de MI saliera en 5 s con 40 pérdidas de Bluetooth por minuto, cuando las rachas la tapan hasta 7 s; y que la cabeza se moviera en los 7 s que escuchaba). Se midió (22 repeticiones sin ningún atraso de datos; `EntradaEEG.movimiento` ve todos los movimientos) y la prueba ahora espera esos eventos en lugar de suponerlos: 8 de 8 corridas seguidas en verde.
- **No se hizo (queda como idea, cambia el diseño):** que más pasos informen, por ejemplo con ensayos de 3 pasos en lugar de 5 o con un paso máximo menor. Cambia la duración del lazo y el cierre completo de la órtesis, así que lo decide Luis después de la demo.

### Orden acordado tras el commit 5 (2 de octubre, tarde)

1. Brecha calibración → lazo: ablaciones en el gemelo, reportar la causa con números antes de corregir; el bug de `P_hat` se corrige en el mismo bloque.
2. Repetir las cifras de lazo y caos del README con el montaje nuevo.
3. CP1 robusto y sin impedancias.
4. Rechazo por movimiento de cabeza.
5. Selección de canales por validación cruzada.
6. Pérdidas de Bluetooth como tipo de caos.
7. Tarea 2 mínima. **Hecha** (ver la sección de la Tarea 2).
8. Corrección para que la órtesis cierre completa. **Hecha.**
9. Semáforo PILOTO, solo si alcanza. **Hecho:** alfa occipital (8–13 Hz, PO7/Oz/PO8, últimos 20 s) contra su línea base del primer minuto del lazo; ×1.5 AMARILLO, ×2.5 ROJO; solo avisa (no pausa, no excluye, no cuenta en la escalera); umbrales sin validar en personas. Gemelo con `--fatiga 1`: ×2.8 a los 15 min.

### P0, en orden de commits (rama `p0-unicorn`)

- [x] 1. Contrato: montaje, papeles, flujo `IMU`, fuentes de EEG; gemelo con la topografía del montaje.
- [x] 2. Reloj por contador: hora de cada muestra y huecos de Bluetooth (reemplaza la lógica de paquetes del Cyton).
- [x] 3. Gemelo: N1 visual en PO7/Oz/PO8, IMU con movimientos de cabeza, pérdidas de Bluetooth por contador, formato UnicornLSL.
- [x] 4a. `EntradaEEG` configurable (fuente `puente` o `unicornlsl`, nombre o tipo, canales por índice, hora por contador, IMU); sin suavizado de marcas de LSL; la ventana de MI tolera pérdidas chicas. El puente ya estampa por contador.
- [x] 4b. Selección de canales por validación cruzada en la calibración (papel contra los 8), registrada.
- [x] 4c. Pérdidas de Bluetooth como tipo de caos; repetir las mediciones de caos con el montaje nuevo.
- [x] 5. `puente_lsl.py --placa unicorn --serie <num>` con flujo `IMU`; `verificar_unicorn.py` (probados con la placa sintética y con el gemelo; **nunca con el casco**).
- [x] 6. CP1 robusto y sin impedancias.
- [x] 7. Rechazo por movimiento de cabeza.
- [x] 8. Brecha calibración → lazo (reportar causa) y bug de `P_hat`.
- [x] Corrección de la Tarea 3: la órtesis cierra completa.
- [x] 9. Semáforo PILOTO (alfa occipital), si alcanza.

Después: Tarea 2 mínima, con la atenuación sensorial en PO7/Oz/PO8 como firma principal del contraste movimiento propio contra ajeno.

---

## Tarea 1 — Resiliencia: el lazo que no se cae

> **Hecha el 2 de octubre de 2026** (rama `tarea1-resiliencia`). Especificación y plan en `docs/`; resultados en el README, sección "Resiliencia". Pendiente: probar la reconexión de `puente_lsl.py` con el Cyton real, y repetir la corrida con caos contra el gemelo con más semillas (una sola corrida no es concluyente).

**Objetivo:** si se desconecta el dongle del Cyton, se reinicia el ESP32, se congela LSL o llega una época corrupta a media demo, el sistema lo detecta, se protege, se recupera solo y **no pierde la sesión**. Hoy cualquiera de esas fallas truena el orquestador.

### Diseño

1. **`salud.py` → clase `Vigilante`.** Monitorea en vivo cada subsistema y le asigna un semáforo VERDE / AMARILLO / ROJO:
   - EEG: edad de la última muestra, tasa real contra la nominal, canales planos o saturados.
   - Órtesis: ACK perdidos, picos de latencia, puerto caído.
   - Reloj: deriva entre el reloj del EEG y el del ACK.
   - Detector: fiabilidad del `ConfianzaDetector`.

   Publica el estado de salud en el JSON del flujo `Estado` y, con cada cambio de semáforo, un marcador nuevo `salud:<subsistema>:<color>` (agregarlo al contrato).
2. **Nuevo estado `PAUSA_SEGURA`** en `config.ESTADOS` y `TRANSICIONES`, alcanzable desde cualquier estado de lazo.
   - Al entrar: la órtesis va a posición segura (abrir despacio), el agente no aprende y los pasos de la pausa se marcan en el CSV para excluirlos del análisis.
   - Al salir: se requieren 3 s continuos de salud VERDE y se regresa al estado previo.
3. **Reconexión automática con retroceso exponencial:**
   - `EntradaEEG` vuelve a resolver el flujo si deja de llegar.
   - `OrtesisSerial` reabre el puerto y resincroniza `seq`.
4. **Escalera de degradación**, de mejor a peor:
   1. Todo bien.
   2. ErrP poco fiable: aprendizaje congelado, ya existe.
   3. EEG perdido: pausa segura.
   4. Órtesis perdida: solo registro y aviso.

   Cada escalón queda documentado y probado.
5. **Persistencia de la sesión.** Después de cada paso se guarda una instantánea atómica (archivo temporal + `os.replace`) en `resultados/estado_sesion.json`. Incluye `beta`, varianza, prior, sesgo, CUSUM, contadores del `ConfianzaDetector`, estado de la máquina, ángulo, `seq` y número de paso. `orquestador.py ... --reanudar` continúa la misma sesión y el mismo CSV tras un cierre inesperado.
6. **Modo caos**, ingeniería del caos aplicada a BCI. `--caos <semilla>` hace que `cerebro_sintetico.py` y `OrtesisSimulada` inyecten fallas reproducibles:
   - cortes de EEG de 1 a 5 s;
   - ACK perdidos;
   - picos de latencia de 80 a 300 ms;
   - ráfagas de parpadeos;
   - un canal que se despega.
7. **Tablero:** tres semáforos de salud (EEG, órtesis, detector) en la cabecera; el estado `PAUSA_SEGURA` en rojo.

### Criterios de aceptación (con pruebas en `pruebas.py`)

- Sesión simulada con modo caos: termina sin excepción; ninguna época tomada durante un corte de EEG llega al agente; el agente no aprende en `PAUSA_SEGURA`.
- `--reanudar` tras matar el proceso a mitad de sesión: `beta` y el número de paso continúan exactamente donde se quedaron.
- Con el caos estándar, el error del agente tras la perturbación sigue claramente por debajo del de la sombra. Reportar los números.

---

## Tarea 2 — Embodiment neural en vivo

> **Versión mínima hecha el 2 de octubre de 2026 (noche)**, rama `p0-unicorn`: movimientos ajenos, atenuación sensorial en PO7/Oz/PO8 como única firma, IIC con intervalo y tendencia en CSV, `Estado`, tablero y EVALUACION, `--embodiment` en el gemelo y cuestionario. Decisiones tomadas sin preguntar (se pueden revertir):
> - El movimiento ajeno se **anuncia** (el tablero muestra AUTOMATICO 1 s antes, marcador `aviso_ajeno`) y va **siempre hacia la meta**, 0.15 del rango. Así el piloto sabe que no es suyo y el contraste no se mezcla con la respuesta al error; el IIC compara contra los propios **correctos**.
> - Va en el 2.º paso de uno de cada dos ensayos (1 de cada 10 pasos): tras el centrado y un paso propio la órtesis está entre 0.2 y 0.8 y el movimiento siempre cabe completo. La elección es reproducible por semilla y sin estado (`--reanudar` repite la misma).
> - IIC = d de Cohen de la N1 (media de PO7/Oz/PO8 entre 140 y 200 ms tras el inicio del movimiento, ventana de la literatura) de ajenos contra propios correctos; intervalo bootstrap del 90 %; tendencia = segunda mitad menos primera. Solo en el lazo adaptativo.
> - Selectividad del error y resonancia motora: fuera de la versión mínima.
> - Tablero: una línea con el IIC en lugar de un panel (con ~12 ajenos por sesión la curva sería casi toda ruido).
> - Cuestionario: al final de una sesión `real` con terminal interactiva (`--sin-cuestionario` lo salta); se guarda en `resultados/sesion_..._cuestionario.json` con el IIC.
> - Movimientos ajenos por defecto también en `sim` (`--ajenos-cada 0` los apaga); las pruebas de pausas usan `--ajenos-cada 0`.
>
> **Aceptación en el gemelo (`estudios/embodiment_gemelo.py`, 16 sujetos): NO se cumple.** Con 12 ajenos por sesión (la demo) el IIC no ordena embodiment 0.2 / 0.5 / 0.8 (Spearman de todas las sesiones −0.12 [−0.31, +0.07]); con 30, apenas (+0.24 [−0.03, +0.50]; orden perfecto en 4 de 16 sujetos). Con embodiment 0 el intervalo incluye 0 en 88 % y 100 %, y no hay sesgo (64 sesiones nulas: +0.01). Falta potencia: en el gemelo d ≈ 0.3 con embodiment 0.8, y con 12 ajenos el intervalo mide ±0.5. Las opciones eran: (a) dejarlo así y mostrar el IIC como exploratorio con intervalo ancho; (b) subir la fracción de ajenos; (c) agregar un bloque corto de embodiment al final. **Decisión de Luis (3 de octubre): opción (a), exploratorio.**
>
> **Análisis de potencia** (`estudios/potencia_iic.py`, con las 176 sesiones ya simuladas; figura `docs/figuras/potencia_iic.png`): el IIC de una sesión es su d verdadera más ruido de desviación √(1.14 / ajenos); en el gemelo d = 0.32 × embodiment. Un intervalo de ±0.2 pide **77 ajenos** (~770 pasos del lazo; la demo tiene 120). Para un Spearman significativo (una cola, 0.05) con 16 sujetos por nivel: potencia de 52 % con 12 ajenos y de 80 % con **29**. Un Spearman esperado de 0.5 pide ~60 ajenos; de 0.8, ~300.

**Objetivo:** medir en vivo qué tanto el cerebro del piloto trata a la órtesis como parte de su propio cuerpo. Es el componente científico más novedoso del proyecto y venía en la idea original. Debe presentarse como **métrica exploratoria**, no validada clínicamente.

### Diseño

1. **Ensayos de contraste de agencia.** En alrededor del 10 % de los pasos del lazo adaptativo, la órtesis hace un **movimiento ajeno**: no lo decidió el decoder, el piloto lo sabe por diseño del protocolo y la meta no cambia. Comparar la respuesta cerebral a movimientos *propios* contra *ajenos* es el núcleo del índice, porque aísla el sentido de agencia del simple estímulo visual.
   - El agente **no aprende** de esos pasos.
   - Llevan su propio marcador (agregarlo al contrato).
2. **Tres firmas neurales**, calculadas en línea y cada una con intervalo de confianza bootstrap:
   - **Atenuación sensorial:** la respuesta temprana (unos 100 ms, tipo N1) a movimientos propios debería ser menor que a ajenos.
   - **Selectividad del error:** amplitud Pe − Ne de errores frente a aciertos en Cz y FCz, normalizada por el ruido.
   - **Resonancia motora:** desincronización mu sobre C3 asociada al movimiento de la órtesis, contra una línea base.
   - Opcional: la tendencia a lo largo de la sesión de la respuesta a aciertos (sorpresa decreciente si se aprende el modelo interno).
3. **Índice de Integración Corporal (IIC):** combinación de las firmas en puntajes z, con su intervalo, y tendencia a lo largo de la sesión. Se reporta en el CSV, en el flujo `Estado`, en el tablero (panel nuevo o reemplazo justificado) y en el resumen de `EVALUACION`.
4. **Validación con el gemelo.** Agregar `--embodiment 0..1` a `cerebro_sintetico.py`, que module las tres firmas. Prueba de aceptación: el IIC estimado debe ordenar correctamente sesiones con `embodiment` 0.2, 0.5 y 0.8 (correlación de Spearman alta en varias semillas), y con `embodiment` 0 su intervalo debe incluir 0.
5. **Cuestionario breve** al final de la sesión: tres afirmaciones redactadas por nosotros, escala de 1 a 7, sobre propiedad, agencia y control. Se guardan junto a la sesión para correlacionarlas después con el IIC.

---

## Tarea 3 — Modo jurado

**Objetivo:** un solo comando corre la demo completa y el tablero explica en pantalla qué está pasando, para un jurado internacional que la ve por primera vez.

### Diseño

1. **`demo.py`:**
   - Lanza la fuente de EEG (`--fuente gemelo` o `--fuente cyton --puerto COMx`), el tablero y el orquestador como subprocesos.
   - Verifica la salud antes de empezar.
   - Los cierra limpiamente con Ctrl+C.
2. **Narrador en el tablero:** una franja con una frase corta por estado y fase, con el número clave del momento (por ejemplo: "La órtesis se equivocó y el cerebro lo notó: ErrP detectado, P = 0.82"). Textos en español y en inglés (`--idioma en`), porque la competencia es internacional.
3. **Tarjeta de resultados automática** al terminar: un PNG de una página con los checkpoints, la curva agente contra sombra, la recuperación tras la perturbación, el IIC y la latencia. Además, un JSON con las cifras.
4. **Repetición:** `tablero.py --repetir resultados/sesion_*.csv --velocidad 4` reproduce una sesión grabada con todo el narrador. Es el plan B visible si algo falla en vivo.

### Criterio de aceptación

`python demo.py --fuente gemelo --idioma en` corre de punta a punta sin intervención y deja la tarjeta de resultados en `resultados/`.
