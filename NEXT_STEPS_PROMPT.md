# Prompt for Claude chat — critique + roadmap

Copy everything below into claude.ai.

---

I'm working on a neuroscience deep-learning project testing the **Synaptic Homeostasis
Hypothesis (SHY; Tononi & Cirelli 2006)**. I want a hard critical review and a
prioritized roadmap for what to do next. Here is the full, honest state.

**Goal.** Test whether the overnight trajectory of slow-wave EEG activity encodes an
individually-varying, biologically meaningful signature of synaptic homeostatic
efficiency, and whether a sequence model's saliency rediscovers or challenges SHY's
predictions (early-NREM dominance and frontal predominance of the overnight decline).

**What's built (Python package, 9-stage pipeline).**
- Download + preprocess: DREAMS Patients (n=27), DREAMS healthy Subjects (n=20),
  Sleep-EDF cassette (n=153). EEG selected by name, robust amplitude auto-scaling,
  EXPERT hypnograms (AASM / annotations), clean NREM 30 s epochs, per-step diagnostics.
- Slow waves: yasa.sw_detect. Per-subject "dissipation curve" = mean negative SLOPE
  and slow-wave DENSITY per (12 temporal bins x 3 canonical regions: frontal/central/
  occipital), per-subject normalized, plus a circadian sin/cos encoding of bin time.
- Model: bidirectional LSTM (hidden 128, 3 layers, dropout 0.4) + temporal attention +
  dual classification/regression head. Stratified k-fold CV, Hanley-McNeil AUC CIs,
  label-permutation tests, gradient saliency, fixed seeds, a synthetic sanity check.

**Key findings (reported honestly, not inflated).**
1. **Slope/density dissociation.** Across all three cohorts, slow-wave DENSITY declines
   steeply overnight (p down to ~4e-10) while per-wave SLOPE is essentially flat. The
   classic slope proxy carries little signal here; homeostatic downscaling shows up as
   *fewer* waves, not *shallower* ones. The model leans on density (66-79% of saliency).
2. **Attention is not a faithful localizer.** On synthetic data with known ground truth
   the model classifies perfectly yet its attention is flat; only input-gradient saliency
   localizes (cf. Jain & Wallace 2019). We keep attention but base claims on saliency.
3. **The fast/slow-dissipater label is tautological** — it's a deterministic function of
   the input curve, so at n=153 the classifier hit AUC 0.99 (label-recoverability, not
   prediction). This is the core design flaw.
4. **De-circularized with an external target.** Predicting subject AGE (Sleep-EDF, 25-101
   yr; ages from EDF headers) from the dissipation trajectory gives held-out r=0.51
   (p=1.6e-11), R^2=0.25, MAE=15 yr, old/young AUC 0.77. Crucially the saliency for AGE
   peaks at EARLY NREM (bin 0), opposite the tautological label's late-night artifact —
   i.e. the genuine biological signal lives in early NREM, which is SHY-consistent.
   (Spatially it's occipital-dominant, but Sleep-EDF only has a frontal and a posterior
   derivation, so topography is montage-limited.)

**Limitations.** Small N for DREAMS; DREAMS Patients carry sleep pathology; cross-dataset
montage/hardware/population differences (cohorts kept separate, not pooled); only 2-6 EEG
channels so far (no HD-EEG); slope detection uses yasa defaults; age tested in one dataset.

**What I want from you.**
1. Critique the methodology hard — what's weak, wrong, or confounded? Especially the
   dissipation-curve construction, the density-vs-slope choice, the region harmonization,
   the circadian encoding, and the age-prediction framing/leakage risks.
2. A prioritized roadmap. I have ANPHY (83-channel HD-EEG) and MESA + MrOS (large NSRR
   cohorts with cognitive outcomes) available but not yet processed.
3. What controls/analyses would make the density-dissociation and age-via-early-NREM
   findings publishable (e.g. vs slow-wave activity/SWA power, per-cycle modeling,
   matching for total sleep time, brain-age framing)?
4. Relevant literature I should engage with (SWA vs slope vs density as synaptic-strength
   markers; age effects on SWA; brain-age models; attention-as-explanation).
