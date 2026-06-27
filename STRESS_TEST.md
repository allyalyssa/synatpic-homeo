# Adversarial stress-test — honest summary

The preliminary findings below were deliberately stress-tested with rigorous
controls (seed 1234, subject-level splits throughout). Several did not survive.
Reproduce with `python controls.py A B C D E F H I`.

## Verdicts

| Control | What it tested | Verdict |
|---|---|---|
| A amplitude-matched slope | dissociation: "slope flat" | **FAILS** outside healthy DREAMS (Sleep-EDF matched slope declines, p=0.002) |
| B within-stage density | decline = homeostasis vs N3→N2 shift | **WEAKENED** (partly architectural; clean only in N3 / healthy DREAMS) |
| C density vs SWA power | is density a valid marker | **SURVIVES** (within-subject r 0.82–0.96) |
| D subject-level split | age = recording leakage? | **SURVIVES** (r 0.52→0.49, overlap 0) |
| E confound regression | age = sleep quality? | **SURVIVES** (partial r 0.44, p=1e-8) |
| F static vs trajectory | does the SEQUENCE matter? | **FAILS-as-novelty** (static RF r=0.54 ties LSTM r=0.53) |
| G out-of-cohort transfer | external validity | **NOT RUN** (MESA/MrOS not downloaded; DREAMS has no ages) |
| H un-pool density | per-cohort replication | replicates (all CIs>0) but dz 0.28–1.45 (not uniform) |
| I interpretability hygiene | is saliency trustworthy | **SURVIVES** (IG agrees r=0.93; Adebayo corr −0.13) |

## Still standing
- Slow-wave density declines overnight and validly tracks SWA (C, H) — but this is essentially the long-known overnight SWA decline.
- Sleep EEG predicts age, r≈0.5 (D, E) — real, not leakage or sleep-quality confound.
- Early-NREM is the most age-informative window, and that saliency is trustworthy (I).
- Attention is not a faithful localizer (synthetic demo) — a methods cautionary note.

## Now artifacts / dead
- **The dissociation ("flat slope → fewer not shallower").** Dead as a general claim: amplitude-matched slope *declines* in Sleep-EDF (classical SHY). Survives only in N=20 healthy DREAMS.
- **The "trajectory/sequence encodes a homeostatic signature" thesis.** Dead for age: a static mean-feature model with NO temporal information matches the LSTM. The sequence model, attention, and dissipation-curve framing are unjustified — the age signal is a static SWA *level*, not a trajectory *shape*.
- **The pooled p~4e-10 as a uniform effect.** Overstated (dz 0.28–1.45 across cohorts).

## Defensible version of the paper
1. Methods note: (bi)LSTM attention is not a faithful temporal localizer; use gradient saliency, validated by integrated gradients + the Adebayo randomization test.
2. Replication: slow-wave density/SWA declines overnight and with age (r≈0.5) — a clean re-confirmation of known age-related SWA decline, for which an LSTM is unnecessary.
3. One preliminary, non-generalizing observation: in N=20 healthy DREAMS, the density decline coexists with a flat amplitude-matched slope; this does NOT replicate in Sleep-EDF. Exploratory, explicitly cohort-dependent.

## Single most important missing test
**Out-of-cohort transfer (Control G).** In-cohort r≈0.5 means little until the model is applied to an independent cohort (MESA/MrOS). Until then even the (static) age result is only internally validated.
