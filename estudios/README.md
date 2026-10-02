# Estudios

Mediciones offline con el gemelo (sin LSL ni casco), para decidir con evidencia. Se corren desde la raíz del repositorio, por ejemplo `python estudios/brecha_mi.py 8`.

| Script | Qué mide |
|---|---|
| `brecha_mi.py [sujetos]` | Brecha calibración → lazo del decoder de MI con ablaciones: esperar más tras la señal, sin recentrado en línea, sin respuestas cerebrales a los movimientos. |
| `brecha_mi_calibracion.py [sujetos]` | Optimismo de la calibración secuencial de MI, efecto del primer paso de cada ensayo y ruido de un bloque de 30 pasos. |
| `brecha_errp_calibracion.py [sujetos]` | Optimismo de la calibración secuencial del detector de ErrP: lo reportado contra lo real en épocas nuevas. |

Resultados del 2 de octubre de 2026 en `TAREAS.md` (sección de la brecha). Son cifras del gemelo, no de una persona.
