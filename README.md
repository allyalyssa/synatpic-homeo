# Synaptic Homeostasis Dissipation Fingerprinting

Deep learning model (bidirectional LSTM + attention) trained on 
slow-wave EEG slope dissipation trajectories to classify individual 
differences in synaptic homeostatic efficiency during NREM sleep.

## Datasets
- Sleep-EDF Expanded (PhysioNet)
- ANPHY-Sleep (OSF)
- MESA Sleep (NSRR)
- MrOS Sleep (NSRR)

## Structure
- `src/` — core pipeline modules
- `notebooks/` — Colab runner notebook
- `models/` — trained model checkpoints
- `outputs/` — figures, results, manuscript summary
