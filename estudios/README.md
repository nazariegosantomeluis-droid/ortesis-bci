# Estudios

Mediciones offline con el gemelo (sin LSL ni casco), para decidir con evidencia. Se corren desde la raíz del repositorio, por ejemplo `python estudios/brecha_mi.py 8`.

| Script | Qué mide |
|---|---|
| `brecha_mi.py [sujetos]` | Brecha calibración → lazo del decoder de MI con ablaciones: esperar más tras la señal, sin recentrado en línea, sin respuestas cerebrales a los movimientos. |
| `brecha_mi_calibracion.py [sujetos]` | Optimismo de la calibración secuencial de MI, efecto del primer paso de cada ensayo y ruido de un bloque de 30 pasos. |
| `brecha_errp_calibracion.py [sujetos]` | Optimismo de la calibración secuencial del detector de ErrP: lo reportado contra lo real en épocas nuevas. |
| `curva_robustez.py [sujetos]` | Curva de robustez (solo `simulador_lazo.py`, sin el gemelo ni LSL): error en los 2 min tras la perturbación (agente contra estático) y tiempo de recuperación del agente (β al 70 % de la perturbación) según la BA del detector de ErrP, de 0.65 a 0.85 con especificidad fija 0.90. Por defecto 30 sujetos; guarda `docs/figuras/curva_robustez.png`. |
| `efecto_p_hat.py [sujetos]` | Efecto de calcular `P_hat` con la fiabilidad real del detector cuando el agente no aprende (`actualizar(..., peso)`), con salida calibrada (solo `simulador_lazo.py`, sin el gemelo ni LSL). Variante vieja (fiabilidad 0 sin aprender: el prior no se mueve) contra la nueva (el prior también se adapta en el bloque estático), con detector sens 0.70 y 0.55 (espec 0.90), `p_errp` cruda del simulador (aproximación no calibrada) y exacta (posterior verdadero del simulador), en bloques 60/300 (perturba en 100) y 30/120 (perturba en 40). Mide error en los 2 min tras perturbar, error antes de perturbar, prior y pasos hasta β al 70 %. Con 30 sujetos del simulador, nueva − vieja en el error de 2 min va de −0.005 a +0.006 (IC95 pareado incluye 0); la recuperación tarda a lo más ~4–5 pasos más con la salida exacta. No hace falta corrección. |

| `decisiones_mi.py [sujetos_espera] [sujetos_minimo]` | Espera extra tras la señal antes del primer paso (0, 1 y 2 s) y mínimo de ensayos de la calibración de MI (24 contra 36). |
| `calibracion_errp_fija.py [sujetos]` | Calibración del detector con 120 épocas fijas y umbral anidado: lo reportado contra lo real en 180 épocas nuevas, y cuántos sujetos dan GO en el CP3. |
| `alineacion_epoca.py [sujetos]` | Época del ErrP cortada en el ACK, en el inicio real del movimiento y en el inicio por telemetría. Solo verifica: el efecto lo programamos en el gemelo. |
| `embodiment_gemelo.py [sujetos] [movimientos]` | Aceptación de la Tarea 2 (verificación, no evidencia): IIC de sesiones independientes del gemelo con embodiment 0, 0.2, 0.5 y 0.8 (uno de cada 10 movimientos ajeno); cobertura de 0 con embodiment 0 y Spearman por sujeto y con todas las sesiones juntas (intervalo remuestreando sujetos). |
| `potencia_iic.py` | Análisis de potencia del IIC con las 176 sesiones del gemelo ya simuladas (`datos/iic_gemelo.csv`; no genera EEG): modelo d = κ × embodiment con ruido √(c / ajenos), cuántos movimientos ajenos piden un intervalo de ±0.2 y un Spearman significativo. Guarda `docs/figuras/potencia_iic.png`. |
| `cierre_completo.py [sujetos]` | Por qué la órtesis casi nunca cerraba completa (solo `simulador_lazo.py`): fracción de ensayos que terminan cerrados o abiertos del todo, y error, con paso máximo mayor y con cada ensayo desde el punto medio. |

Resultados del 2 de octubre de 2026 en `TAREAS.md` (sección de la brecha). Son cifras del gemelo, no de una persona.
