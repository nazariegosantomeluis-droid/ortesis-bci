# Comparison with baselines (digital twin, 16 sessions per method, same seeds)

| Method | Error before the perturbation | Error in the 2 min after it | Recovered | Steps to recover (median) |
|---|---|---|---|---|
| Static (no learning) | 0.172 ± 0.018 | 0.492 ± 0.007 | 0/16 | — |
| Fixed eta (0.3) | 0.156 ± 0.022 | 0.414 ± 0.017 | 5/16 | 46 |
| Bayes (ours) | 0.148 ± 0.025 | 0.371 ± 0.017 | 14/16 | 28 |
| No ErrP (negative control) | 0.170 ± 0.018 | 0.489 ± 0.008 | 0/16 | — |

- **This is the digital twin, not a person.** Error = fraction of steps where the chosen direction was not the goal, mean ± standard error across sessions.
- Recovery: β reaches 70 % of the perturbation (2.4 logits) within the 57 steps (2 min) of CP4; steps are loop steps, not seconds. “—” = no session recovered.
- Methods: *static* = the calibrated decoder with no correction; *fixed eta* = the agent with constant eta 0.3; *Bayes* = the demo agent; *no ErrP* = the Bayesian agent always receiving the base rate as the detector output (LLR = 0), the negative control.
- Same seeds: session k of subject s uses seed 1000 + 100 s + k for every method (same pilot, goals, detector and perturbation). Reproduce: `python estudios/comparacion_baselines.py`.

With the other two detectors (error in the 2 min after the perturbation and sessions that recover):

| | Static (no learning) | Fixed eta (0.3) | Bayes (ours) | No ErrP (negative control) |
|---|---|---|---|---|
| medium detector | 0.491; 0/16 | 0.471; 0/16 | 0.368; 15/16 | 0.490; 0/16 |
| weak detector | 0.491; 0/16 | 0.476; 0/16 | 0.434; 7/16 | 0.488; 0/16 |
