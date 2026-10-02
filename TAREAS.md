# Tareas pendientes (en este orden)

Lee `CLAUDE.md` primero. Para cada tarea: diseña, implementa, **mide en el gemelo o el simulador**, agrega pruebas, actualiza README y haz commit. Si una decisión de diseño no está clara, pregúntale a Luis antes de seguir.

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
