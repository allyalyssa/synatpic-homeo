"""Stage 2 — turn raw EDFs into clean NREM epochs, with a diagnostic after
every step so subject loss is never silent.

The original pipeline used YASA auto-staging and ignored the expert hypnograms;
for Sleep-EDF that throws away the dataset's entire reason for existing. Here we
parse the expert hypnogram (AASM for DREAMS, annotations for Sleep-EDF) and
print how many epochs survive each step, which is how we locate subject loss.

Two dataset-specific gotchas this code handles explicitly:
  * MNE types every DREAMS channel as "eeg" (even ECG/EMG), so we pick EEG
    channels by name, not by MNE's channel type.
  * DREAMS amplitudes come back ~1e6x too large (labelled V but really uV); a
    median-based auto-scaler rescales them without touching correct files.

Output per subject: data/<dataset>/preprocessed/<subject>.npz with
  epochs     (n_nrem, n_ch, n_samp) float32   cleaned NREM epochs, volts
  stages     (n_nrem,)              int        1/2/3 = N1/N2/N3
  epoch_idx  (n_nrem,)              int        position in recording (x30 s = time)
  ch_names   list[str]
  sfreq      float
"""
import sys

import mne
import numpy as np

from config import (PATHS, PREP, EEG_CHANNELS, DREAMS_HYPNO_STEP_SEC,
                    DREAMS_AASM_MAP, ensure_dirs)

mne.set_log_level("ERROR")

# Sleep-EDF annotation strings -> integer stages (YASA convention). Old R&K S3
# and S4 both fold into N3.
STAGE_STR2INT = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
    "Sleep stage ?": -1, "Movement time": -1,
}


def pick_eeg(raw, dataset):
    """Keep only real EEG channels, by name (see module docstring)."""
    spec = EEG_CHANNELS.get(dataset)
    if spec is None:
        keep = [c for c in raw.ch_names if c.upper().startswith("EEG")]
    else:
        keep = [c for c in raw.ch_names if c in spec]
    if keep:
        raw.pick(keep)
    return keep


def autoscale_to_volts(raw, diag):
    """Rescale uV-labelled-as-V files to true volts using a robust median.

    Real EEG has median |amplitude| ~5-30 uV (5e-6..3e-5 V). A median above 1e-3 V
    means the file is off by ~1e6, so we divide accordingly. Correct files are
    left untouched.
    """
    med = float(np.median(np.abs(raw.get_data())))
    diag["rescaled"] = med > 1e-3
    if diag["rescaled"]:
        raw.apply_function(lambda x: x * 1e-6)


def load_signals(psg_path, dataset, diag):
    """Load EDF, keep EEG only, scale, filter, resample."""
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    diag["channels_all"] = len(raw.ch_names)

    names = pick_eeg(raw, dataset)
    diag["channels_eeg"] = len(names)
    diag["eeg_names"] = names
    if not names:
        return None

    autoscale_to_volts(raw, diag)
    diag["amp_med_uv"] = float(np.median(np.abs(raw.get_data())) * 1e6)

    raw.filter(PREP["l_freq"], PREP["h_freq"], method="fir", phase="zero", verbose=False)
    if raw.info["sfreq"] != PREP["sfreq"]:
        raw.resample(PREP["sfreq"], verbose=False)
    diag["sfreq"] = float(raw.info["sfreq"])
    return raw


def hypnogram_sleep_edf(hyp_path, n_epochs, diag):
    """Expand Sleep-EDF expert annotations onto the 30 s epoch grid."""
    ann = mne.read_annotations(hyp_path)
    stages = np.full(n_epochs, -1, dtype=int)
    seen = set()
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        seen.add(desc)
        start = int(round(onset / PREP["epoch_sec"]))
        n = int(round(dur / PREP["epoch_sec"]))
        stages[start:start + n] = STAGE_STR2INT.get(desc, -1)
    diag["hypno_labels_seen"] = sorted(seen)
    diag["hypno_all_unknown"] = bool((stages == -1).all())
    return stages


def hypnogram_dreams(hyp_path, n_epochs, diag):
    """Read DREAMS AASM scores (5 s) and majority-vote onto the 30 s grid."""
    vals = [int(x) for x in open(hyp_path) if x.strip() and not x.startswith("[")]
    mapped = np.array([DREAMS_AASM_MAP.get(v, -1) for v in vals])
    per = PREP["epoch_sec"] // DREAMS_HYPNO_STEP_SEC          # 6 scores per epoch
    n = len(mapped) // per
    blocks = mapped[:n * per].reshape(n, per)
    stages30 = np.array([_majority(b) for b in blocks])
    out = np.full(n_epochs, -1, dtype=int)
    m = min(n_epochs, len(stages30))
    out[:m] = stages30[:m]
    diag["hypno_labels_seen"] = sorted(set(vals))
    diag["hypno_all_unknown"] = bool((out == -1).all())
    return out


def _majority(block):
    valid = block[block >= 0]
    if valid.size == 0:
        return -1
    return int(np.bincount(valid).argmax())


HYPNO_LOADERS = {"sleep_edf": hypnogram_sleep_edf, "dreams": hypnogram_dreams,
                 "dreams_subjects": hypnogram_dreams}


def preprocess_subject(psg_path, hyp_path, subject, dataset):
    """Full stage-2 pipeline for one subject. Prints diagnostics, returns dict or None."""
    diag = {"subject": subject, "dataset": dataset}
    raw = load_signals(psg_path, dataset, diag)
    if raw is None:
        _report(diag, reason="no EEG channels selected")
        return None

    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    samp = int(PREP["epoch_sec"] * sfreq)
    n_epochs = data.shape[1] // samp
    diag["epochs_total"] = n_epochs
    epochs = data[:, : n_epochs * samp].reshape(data.shape[0], n_epochs, samp).transpose(1, 0, 2)

    stages = HYPNO_LOADERS[dataset](hyp_path, n_epochs, diag)

    ptp = epochs.max(axis=2) - epochs.min(axis=2)
    bad = (ptp > PREP["ptp_reject_uv"] * 1e-6).any(axis=1)
    diag["epochs_clean"] = int((~bad).sum())

    nrem = np.isin(stages, PREP["nrem_stages"])
    diag["epochs_nrem"] = int(nrem.sum())

    keep = nrem & ~bad
    diag["epochs_kept"] = int(keep.sum())
    _report(diag)
    if keep.sum() == 0:
        return None

    return {
        "epochs": epochs[keep].astype(np.float32),
        "stages": stages[keep].astype(int),
        "epoch_idx": np.flatnonzero(keep).astype(int),
        "ch_names": diag["eeg_names"],
        "sfreq": sfreq,
    }


def _report(diag, reason=None):
    d = diag
    print(f"  {d['subject']:14} | eeg {d.get('channels_eeg','?')}/{d.get('channels_all','?')} "
          f"| amp~{d.get('amp_med_uv',float('nan')):.1f}uV{' [rescaled]' if d.get('rescaled') else ''} "
          f"| epochs total={d.get('epochs_total','?')} clean={d.get('epochs_clean','?')} "
          f"nrem={d.get('epochs_nrem','?')} kept={d.get('epochs_kept','?')}")
    if d.get("hypno_all_unknown"):
        print(f"    !! hypnogram parsed to all-unknown; labels seen = {d.get('hypno_labels_seen')}")
    if reason:
        print(f"    !! dropped: {reason}")


def run_sleep_edf():
    ensure_dirs()
    raw_dir, out_dir = PATHS["sleep_edf_raw"], PATHS["sleep_edf_prep"]
    psgs = sorted(raw_dir.glob("*-PSG.edf"))
    print(f"=== Sleep-EDF: {len(psgs)} PSG files ===")
    kept = 0
    for psg in psgs:
        subj = psg.name[:6]
        hyp = next(raw_dir.glob(f"{subj}*-Hypnogram.edf"), None)
        if hyp is None:
            print(f"  {subj}: no hypnogram, skipping")
            continue
        res = preprocess_subject(psg, hyp, subj, "sleep_edf")
        if res:
            np.savez(out_dir / f"{subj}.npz", **res)
            kept += 1
    print(f"Sleep-EDF: {kept}/{len(psgs)} subjects produced clean NREM epochs")


def run_dreams_like(dataset, glob_pat):
    """DREAMS Patients and Subjects share montage + AASM hypnograms; only the
    file-name stem differs (patient* vs subject*)."""
    ensure_dirs()
    raw_dir, out_dir = PATHS[f"{dataset}_raw"], PATHS[f"{dataset}_prep"]
    psgs = sorted(raw_dir.glob(glob_pat))
    print(f"=== {dataset}: {len(psgs)} EDF files ===")
    kept = 0
    for psg in psgs:
        subj = psg.stem
        hyp = raw_dir / f"HypnogramAASM_{subj}.txt"
        if not hyp.exists():
            print(f"  {subj}: no AASM hypnogram, skipping")
            continue
        res = preprocess_subject(psg, hyp, subj, dataset)
        if res:
            np.savez(out_dir / f"{subj}.npz", **res)
            kept += 1
    print(f"{dataset}: {kept}/{len(psgs)} subjects produced clean NREM epochs")


def run_dreams():
    run_dreams_like("dreams", "patient*.edf")


def run_dreams_subjects():
    run_dreams_like("dreams_subjects", "subject*.edf")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "dreams"):
        run_dreams()
    if which in ("all", "dreams_subjects"):
        run_dreams_subjects()
    if which in ("all", "sleep_edf"):
        run_sleep_edf()
