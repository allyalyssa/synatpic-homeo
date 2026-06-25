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

## Key findings (DREAMS n=27 patients + n=20 healthy; Sleep-EDF n=153)

1. **Slope/density dissociation (all three cohorts at full N).** Slow-wave
   *density* falls steeply overnight (DREAMS healthy 16.6 → 4.6 waves/min, p≈3e-6;
   Sleep-EDF 3.3 → 1.0, p≈4e-10) but per-wave *slope* is essentially flat
   (p=0.13–0.70) — homeostatic downscaling shows up as *fewer* waves, not
   *shallower* ones, so the classic slope proxy alone carries almost no signal. We
   therefore model both measures; the model leans on density (66–79% of saliency).
   (An apparent Sleep-EDF slope decline at n=30 did not survive at n=153 — a
   small-sample fluctuation, which is exactly why we scaled the replication up.)
2. **Attention is not a faithful localizer.** On synthetic data with a known
   answer, the model classifies perfectly yet its attention is flat; only
   gradient saliency localizes. We keep attention but base claims on saliency
   ("Attention is not Explanation", Jain & Wallace 2019).
3. **The model deviates from SHY** in both cohorts: saliency concentrates on
   *late* NREM (p≈0.001–0.003), not early, and shows *no frontal predominance*.
   The late-night emphasis is interpretable — density is near-ceiling early for
   everyone, so the between-subject variance that separates fast/slow dissipaters
   lives late in the night. Classification stays modest (AUC 0.76 / 0.79, CIs
   exclude 0.5 but are wide at this N), so these are exploratory signals.

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
