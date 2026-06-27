# synaptic-homeo

Does the temporal trajectory of slow-wave EEG activity across a night encode an
individually-varying signature of synaptic homeostatic efficiency — and does a
sequence model trained on these trajectories rediscover or challenge the
Synaptic Homeostasis Hypothesis (SHY; Tononi & Cirelli, 2006)?

The negative half-wave slope of slow waves is a known proxy for synaptic
strength. SHY predicts it is steepest in early NREM and declines across the
night as synapses downscale. We model each subject's per-region **dissipation
curve** (slope *and* slow-wave density across 12 temporal bins) as a multivariate
time series, train a bidirectional LSTM with temporal attention + a dual
classification/regression head, and read **gradient saliency** as the scientific
payload: does the model rely on early NREM and weight frontal regions, as SHY
predicts?

## Key findings (post stress-test — see `STRESS_TEST.md`)

Preliminary findings were deliberately stress-tested with rigorous controls
(`controls.py`). Several did NOT survive; reported honestly rather than hidden.

1. **Survives:** slow-wave density declines overnight and validly tracks SWA
   (within-subject r 0.82–0.96); sleep EEG predicts subject age (held-out r≈0.5,
   not leakage or sleep-quality confound); attention is not a faithful localizer
   (synthetic demo), so claims use gradient saliency, validated by integrated
   gradients + the Adebayo randomization test.
2. **DEAD — the slope/density dissociation** ("flat slope → fewer not shallower")
   does not generalize: once amplitude is matched, slope *declines* in Sleep-EDF
   (classical SHY). It survives only in N=20 healthy DREAMS.
3. **DEAD — the "trajectory/sequence" thesis:** a static mean-feature model with
   no temporal information matches the LSTM at predicting age (r=0.54 vs 0.53), so
   the sequence model/attention add nothing — the age signal is a static SWA
   *level*, essentially re-confirming known age-related SWA decline.
4. **Untested:** out-of-cohort transfer (needs MESA/MrOS) — the single most
   important missing validation.

## Pipeline (9 stages)

| Stage | Module | What it does |
|-------|--------|--------------|
| 1 Download | `download.py` | Fetch + extract DREAMS (Zenodo RAR via bsdtar) and Sleep-EDF (PhysioNet, parallel) |
| 2 Preprocess | `preprocess.py` | EDF → EEG-by-name, auto-scaled, expert-staged clean NREM epochs (+ per-step diagnostics) |
| 3 Slow waves | `slow_waves.py` | `yasa.sw_detect` → per-wave negative slope + time in night |
| 4 Dissipation | `dissipation.py` | Slope **and** density → per-region 12-bin curves + density-decline label |
| 5 Features | `features.py` | Curves → harmonized (N,12,8) tensors (slope+density regions + circadian sin/cos) |
| 6 Model | `model.py` | Bi-LSTM (128, 3 layers, dropout 0.4) + temporal attention + dual head |
| 7 Training | `train.py` | Stratified k-fold CV, AUC + Hanley–McNeil CIs, checkpoints, OOF saliency |
| 8 Interpretation | `interpret.py` | Gradient saliency (temporal/regional) vs SHY; attention shown but flagged |
| — Sanity | `synthetic_check.py` | Separable synthetic data → model must classify it + saliency must localize |
| — Robustness | `robustness.py` | Determinism, label-permutation test, validity report |
| — Compare | `compare.py` | Cross-cohort SHY readouts (patients vs healthy vs Sleep-EDF) |

All paths and hyperparameters live in `config.py`. Run everything with
`python run_pipeline.py dreams dreams_subjects` (raw data must be downloaded
first via `python download.py`). `data/` and `outputs/` are gitignored (multi-GB).

## Honest limitations
- **Small N** (27 DREAMS Patients, 20 Subjects): results are reported with
  Hanley–McNeil AUC CIs and a label-permutation test, not point estimates.
- **Proxy label**: fast/slow "dissipater" is a median split on a fitted density
  decline rate, a deterministic function of the input — so accuracy is partly
  tautological; the saliency *pattern* is the real test.
- **Slope vs density**: the brief's slope proxy is flat overnight here; we report
  this openly and add density rather than force a slope-based story.
- **Cohort/cross-dataset confounds**: DREAMS Patients carry sleep pathology;
  DREAMS/Sleep-EDF differ in montage/hardware/population, so we keep datasets
  separate rather than pool.
- The original Colab notebook is kept as `notebooks` for provenance.
