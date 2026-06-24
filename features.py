"""Stage 5 — convert per-subject dissipation curves into harmonized, model-ready
tensors.

Two harmonization moves let datasets with different montages share one model:
  * channels collapse to canonical regions (frontal/central/occipital), so every
    subject becomes a (12 bins x 3 regions) matrix regardless of native channels;
  * amplitude is normalized per subject (divide by the subject's own mean slope),
    which is leakage-free and removes nuisance between-subject gain differences,
    leaving the temporal SHAPE that SHY is about.

We then append a circadian-phase encoding of each bin's time. Time-since-onset is
not just a linear index: sleep sits on a moving circadian background, so we encode
each bin's clock position as sin/cos over a 24 h cycle. This gives the model "where
in the night" each bin is, justified by circadian modulation of homeostasis.

Output: data/harmonized/<dataset>.npz with
  X        (N, 12, 5) float32   features = [frontal, central, occipital, sin, cos]
  y_cls    (N,)        int       fast/slow label
  y_reg    (N,)        float     standardized dissipation rate (regression target)
  y_reg_raw, ids, feature_names
"""
import sys

import numpy as np

from config import PATHS, REGIONS, ensure_dirs

# feature layout: per-region slope, then per-region density, then circadian.
FEATURE_NAMES = ([f"{r}_slope" for r in REGIONS]
                 + [f"{r}_dens" for r in REGIONS]
                 + ["sin_phase", "cos_phase"])


def region_of(name):
    """Map an EEG channel/derivation name to a canonical region (or None)."""
    e = name.upper().replace("EEG", "").strip().split("-")[0].strip()
    if e.startswith(("FP", "AF", "F")):
        return "frontal"
    if e.startswith(("C", "T")):
        return "central"
    if e.startswith(("P", "O")):
        return "occipital"
    return None


def _regionize(curve, ch_names):
    """Per-subject-normalized curve (12, n_ch) -> region curve (12, n_regions)."""
    curve = curve / (curve.mean() + 1e-9)
    nb = curve.shape[0]
    reg = np.full((nb, len(REGIONS)), np.nan)
    for ri, rname in enumerate(REGIONS):
        cols = [i for i, c in enumerate(ch_names) if region_of(c) == rname]
        if cols:
            reg[:, ri] = curve[:, cols].mean(axis=1)
    row_mean = np.nanmean(reg, axis=1, keepdims=True)        # absent region -> neutral
    return np.where(np.isnan(reg), row_mean, reg)


def subject_features(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    ch_names = [str(c) for c in d["ch_names"]]
    slope_reg = _regionize(d["slope_curve"].astype(float), ch_names)
    dens_reg = _regionize(d["density_curve"].astype(float), ch_names)

    phase = 2 * np.pi * d["bin_hours"] / 24.0
    circ = np.stack([np.sin(phase), np.cos(phase)], axis=1)   # (12, 2)

    X = np.concatenate([slope_reg, dens_reg, circ], axis=1).astype("float32")
    return X, int(d["label"]), float(d["rate"])


def build(dataset):
    ensure_dirs()
    files = sorted(PATHS[f"{dataset}_curves"].glob("*.npz"))
    if not files:
        print(f"{dataset}: no dissipation curves yet")
        return
    X, y_cls, y_reg, ids = [], [], [], []
    for f in files:
        x, lab, rate = subject_features(f)
        X.append(x); y_cls.append(lab); y_reg.append(rate); ids.append(f.stem)

    X = np.stack(X)
    y_cls = np.array(y_cls)
    y_reg = np.array(y_reg)
    y_reg_z = (y_reg - y_reg.mean()) / (y_reg.std() + 1e-9)

    out = PATHS["harmonized"] / f"{dataset}.npz"
    np.savez(out, X=X, y_cls=y_cls, y_reg=y_reg_z, y_reg_raw=y_reg,
             ids=ids, feature_names=FEATURE_NAMES)
    print(f"{dataset}: X {X.shape} (subjects, bins, features={FEATURE_NAMES}) | "
          f"fast {int(y_cls.sum())} / slow {int((1 - y_cls).sum())} -> {out.name}")


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["dreams", "sleep_edf"]):
        build(ds)
