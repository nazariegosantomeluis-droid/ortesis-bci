# Comparación con baselines (gemelo digital, 16 sesiones por método, mismas semillas)

| Método | Error antes de perturbar | Error en los 2 min tras perturbar | Se recuperan | Pasos hasta recuperar (mediana) |
|---|---|---|---|---|
| Estático (sin aprender) | 0.172 ± 0.018 | 0.492 ± 0.007 | 0/16 | — |
| Eta fija (0.3) | 0.156 ± 0.022 | 0.414 ± 0.017 | 5/16 | 46 |
| Bayes (el nuestro) | 0.148 ± 0.025 | 0.371 ± 0.017 | 14/16 | 28 |
| Sin ErrP (control negativo) | 0.170 ± 0.018 | 0.489 ± 0.008 | 0/16 | — |

- **Es el gemelo digital, no una persona.** Error = fracción de pasos en que la dirección elegida no fue la meta, media ± error estándar entre sesiones.
- Recuperación: β llega al 70 % de la perturbación (2.4 logits) dentro de los 57 pasos (2 min) del CP4; los pasos son los del lazo, no segundos. «—» = ninguna sesión se recuperó.
- Métodos: *estático* = el decoder calibrado sin corregir; *eta fija* = el agente con eta constante 0.3; *bayes* = el agente de la demo; *sin ErrP* = el agente bayes recibiendo siempre la tasa base como salida del detector (LLR = 0), el control negativo.
- Mismas semillas: la sesión k del sujeto s usa la semilla 1000 + 100 s + k en todos los métodos (mismo piloto, metas, detector y perturbación). Reproducir: `python estudios/comparacion_baselines.py`.

Con los otros dos detectores (error en los 2 min tras perturbar y sesiones que se recuperan):

| | Estático (sin aprender) | Eta fija (0.3) | Bayes (el nuestro) | Sin ErrP (control negativo) |
|---|---|---|---|---|
| detector medio | 0.491; 0/16 | 0.471; 0/16 | 0.368; 15/16 | 0.490; 0/16 |
| detector débil | 0.491; 0/16 | 0.476; 0/16 | 0.434; 7/16 | 0.488; 0/16 |
