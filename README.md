# synaptic-homeo

Does the temporal trajectory of slow-wave EEG activity across a night encode an
individually-varying signature of synaptic homeostatic efficiency — and does a
sequence model trained on these trajectories rediscover or challenge the
Synaptic Homeostasis Hypothesis (SHY; Tononi & Cirelli, 2006)?

The negative half-wave slope of slow waves is a known proxy for synaptic
strength. SHY predicts it is steepest in early NREM and declines across the
night as synapses downscale. We model each subject's per-channel **dissipation
curve** (normalized slope across 12 temporal bins) as a multivariate time
series, train a bidirectional LSTM with temporal attention + a dual
classification/regression head, and read the **attention weights and channel
saliency** as the scientific payload: does the model attend to early NREM and
weight frontal channels, as SHY predicts?

## Pipeline (9 stages)

| Stage | Module | What it does |
|-------|--------|--------------|
| 1 Download | `download.py` | Fetch + extract DREAMS (Zenodo) and Sleep-EDF (PhysioNet) |
| 2 Preprocess | `preprocess.py` | EDF → filtered, referenced, expert-staged clean NREM epochs |
| 3 Slow waves | `slow_waves.py` | `yasa.sw_detect` → per-wave negative slope |
| 4 Dissipation | `dissipation.py` | Slopes → per-channel 12-bin curves + fast/slow labels |
| 5 Features | `features.py` | Curves → harmonized tensors (+ circadian time encoding) |
| 6 Model | `model.py` | Bi-LSTM + temporal attention + dual head |
| 7 Training | `train.py` | Stratified k-fold CV, AUC + Hanley–McNeil CIs, checkpoints |
| 8 Interpretation | `interpret.py` | Attention + channel-saliency figures |
| 9 Sanity | `synthetic_check.py` | Synthetic separable data → model must classify it |

All paths and hyperparameters live in `config.py`. Run a stage with, e.g.,
`python download.py`. `data/` and `outputs/` are gitignored (multi-GB).

## Honest limitations
- **Small N** (~26 DREAMS subjects): any classifier result needs confidence
  intervals, not point estimates, and invites overfitting.
- **Proxy labels**: fast/slow "dissipater" is a median split on a fitted decay
  rate, not a biological ground truth.
- **Cross-dataset confounds**: DREAMS (clinical patients) and Sleep-EDF differ
  in montage, hardware, and population; pooling them mixes signal with batch.
- The original Colab notebook is kept as `notebooks` for provenance.
