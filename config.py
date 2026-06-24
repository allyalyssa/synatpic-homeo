"""Central configuration for the synaptic-homeostasis project.

Every path, dataset spec, and hyperparameter lives here so the rest of the code
never hardcodes a location or a magic number. Import what you need, e.g.

    from config import PATHS, PREP, SW, MODEL

Paths are resolved relative to this file, so the project runs from any machine
without editing absolute paths.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

# One entry per pipeline stage / dataset. Created on demand by ensure_dirs().
PATHS = {
    # raw downloads (gitignored, large)
    "dreams_raw":            DATA / "dreams" / "raw",
    "dreams_subjects_raw":   DATA / "dreams_subjects" / "raw",
    "sleep_edf_raw":         DATA / "sleep_edf" / "raw",
    # stage 2: cleaned NREM epochs
    "dreams_prep":           DATA / "dreams" / "preprocessed",
    "dreams_subjects_prep":  DATA / "dreams_subjects" / "preprocessed",
    "sleep_edf_prep":        DATA / "sleep_edf" / "preprocessed",
    # stage 3: detected slow waves (one table per subject)
    "dreams_sw":             DATA / "dreams" / "slow_waves",
    "dreams_subjects_sw":    DATA / "dreams_subjects" / "slow_waves",
    "sleep_edf_sw":          DATA / "sleep_edf" / "slow_waves",
    # stage 4: per-channel dissipation curves
    "dreams_curves":         DATA / "dreams" / "dissipation_curves",
    "dreams_subjects_curves":DATA / "dreams_subjects" / "dissipation_curves",
    "sleep_edf_curves":      DATA / "sleep_edf" / "dissipation_curves",
    # stage 5: harmonized model-ready tensors
    "harmonized":      DATA / "harmonized",
    # stage 6-8 artifacts
    "models":          OUTPUTS / "models",
    "figures":         OUTPUTS / "figures",
    "logs":            OUTPUTS / "logs",
}

# Tool we shell out to for RAR extraction. Windows ships bsdtar (libarchive,
# with RAR read support) at this path; on Linux/Colab plain "tar" also works.
import shutil as _shutil
RAR_TAR = r"C:\Windows\System32\tar.exe"
if not Path(RAR_TAR).exists():
    RAR_TAR = _shutil.which("tar") or "tar"


def ensure_dirs():
    """Create every directory in PATHS (idempotent)."""
    for p in PATHS.values():
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# DREAMS: we use the "Patients" database (27 whole-night recordings) to match
# the original pipeline. NOTE: these subjects carry sleep pathologies, which is
# a real confound for a synaptic-homeostasis question framed around healthy
# downscaling. We flag it rather than hide it; the "Subjects" DB (20 healthy)
# is the cleaner long-term choice.
DREAMS = {
    "zenodo_record": "2650142",
    "rar_key": "DatabasePatients.rar",
    "n_expected": 27,
}

# Sleep-EDF Expanded, sleep-cassette (SC) arm. Each subject has an expert
# hypnogram we MUST use (the dataset's whole point) rather than auto-staging.
SLEEP_EDF = {
    "base_url": "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/",
    "max_subjects": None,   # None = all ~153 SC recordings; set an int to cap.
}

# Channel selection is dataset-specific because MNE cannot be trusted to type
# these EDFs correctly (DREAMS labels ECG/EMG/respiration as "eeg"). A None
# spec means "keep channels whose name starts with EEG" (works for Sleep-EDF).
EEG_CHANNELS = {
    "dreams": ["FP1-A2", "FP2-A1", "CZ-A1", "CZ2-A1", "O1-A2", "O2-A1"],
    "dreams_subjects": ["FP1-A2", "FP2-A1", "CZ-A1", "CZ2-A1", "O1-A2", "O2-A1"],
    "sleep_edf": None,
}

# Hypnogram epoch resolution on disk (seconds). DREAMS scores every 5 s; we
# collapse to the 30 s grid by majority vote. Sleep-EDF annotations carry their
# own durations so no fixed step is needed.
DREAMS_HYPNO_STEP_SEC = 5
# DREAMS AASM codes -> YASA integer stages (0=W,1=N1,2=N2,3=N3,4=REM).
DREAMS_AASM_MAP = {5: 0, 4: 4, 3: 1, 2: 2, 1: 3}

# ---------------------------------------------------------------------------
# Stage 2: preprocessing
# ---------------------------------------------------------------------------
PREP = {
    "l_freq": 0.5,
    "h_freq": 40.0,
    "sfreq": 100.0,          # Sleep-EDF is natively 100 Hz; downsample DREAMS to match
    "epoch_sec": 30,
    # Peak-to-peak artifact gate. 500 uV is generous enough to keep real slow
    # waves (which can hit ~200-300 uV) while dropping movement/saturation.
    "ptp_reject_uv": 500.0,
    "nrem_stages": (1, 2, 3),  # N1, N2, N3 in YASA's integer coding
}

# ---------------------------------------------------------------------------
# Stage 3: slow-wave detection (YASA sw_detect)
# ---------------------------------------------------------------------------
# Defaults follow Massimini/Carrier-style criteria that YASA ships with. The
# negative-going slope of each wave is our proxy for synaptic strength.
SW = {
    "freq_sw": (0.3, 1.5),
    "dur_neg": (0.3, 1.5),
    "dur_pos": (0.1, 1.0),
    "amp_ptp": (75, 500),    # uV; lower bound defines a "real" slow wave
}

# ---------------------------------------------------------------------------
# Stage 4: dissipation curves
# ---------------------------------------------------------------------------
DISS = {
    "n_bins": 12,            # temporal bins across the night (the time axis the LSTM sees)
    # Subjects are labelled fast vs slow dissipaters by a median split on the
    # fitted decay rate of mean negative slope across the night.
}

# Canonical scalp regions used to harmonize datasets with different montages
# (DREAMS 6 ch, Sleep-EDF 2 ch, ANPHY 83 ch all collapse to these three). The
# frontal region is what the SHY "frontal predominance" claim is tested against.
REGIONS = ["frontal", "central", "occipital"]

# ---------------------------------------------------------------------------
# Stage 6-7: model + training
# ---------------------------------------------------------------------------
MODEL = {
    "hidden": 128,
    "layers": 3,
    "dropout": 0.4,
    "bidirectional": True,
}

TRAIN = {
    "k_folds": 5,
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 16,
    "patience": 15,          # early stopping
    "reg_loss_weight": 0.5,  # weight on the continuous (regression) head
}

# Single source of truth for reproducibility (see robustness step).
SEED = 1234
