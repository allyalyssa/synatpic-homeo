"""Stage 4 — turn detected slow waves into per-channel dissipation curves and
assign each subject a fast/slow-dissipater label.

We carry TWO homeostatic measures per (temporal bin, channel):
  * slope   — mean negative slope of slow waves (the classic SHY proxy);
  * density — slow waves per minute of NREM in that bin.

This is a deliberate, diagnostics-driven choice. In DREAMS, slow-wave DENSITY
falls steeply across the night (the expected homeostatic decline) while per-wave
SLOPE stays essentially flat -- a real dissociation. Modelling slope alone leaves
the classifier almost no signal; density is where the homeostasis actually shows
up. We keep both so the model (and the saliency) can speak to each, and we base
the fast/slow label on the measure that genuinely declines (density).

HONEST CAVEAT (see README): the label is a deterministic function of the curve,
so classification accuracy is partly tautological. The scientific test is the
saliency pattern -- early NREM (temporal) and frontal (spatial) -- not accuracy.

Output per subject: data/<dataset>/dissipation_curves/<subject>.npz with
  slope_curve   (n_bins, n_ch) float
  density_curve (n_bins, n_ch) float   waves/min NREM
  ch_names, bin_hours, rate (density decline), label (median split)
"""
import sys

import numpy as np

from config import PATHS, DISS, PREP, ensure_dirs


def build_curves(sw_path):
    d = np.load(sw_path, allow_pickle=True)
    slope, night, ch_idx = d["slope"], d["night_sec"], d["ch_idx"]
    nrem_sec = d["nrem_sec"]
    ch_names = [str(c) for c in d["ch_names"]]
    n_ch, nb = len(ch_names), DISS["n_bins"]

    edges = np.linspace(night.min(), night.max(), nb + 1)
    sw_bin = np.clip(np.digitize(night, edges) - 1, 0, nb - 1)
    # minutes of NREM falling in each temporal bin (shared across channels)
    ep_bin = np.clip(np.digitize(nrem_sec, edges) - 1, 0, nb - 1)
    nrem_min = np.array([(ep_bin == b).sum() * PREP["epoch_sec"] / 60.0 for b in range(nb)])

    slope_curve = np.full((nb, n_ch), np.nan)
    density_curve = np.zeros((nb, n_ch))
    for b in range(nb):
        for c in range(n_ch):
            m = (sw_bin == b) & (ch_idx == c)
            if m.any():
                slope_curve[b, c] = slope[m].mean()
            density_curve[b, c] = m.sum() / nrem_min[b] if nrem_min[b] > 0 else 0.0

    centers = (edges[:-1] + edges[1:]) / 2
    bin_hours = (centers - night.min()) / 3600.0
    return _fill_nan(slope_curve), density_curve, ch_names, bin_hours


def _fill_nan(curve):
    """Linear-interpolate gaps within each channel; edge-extend if needed."""
    x = np.arange(curve.shape[0])
    for c in range(curve.shape[1]):
        col = curve[:, c]
        ok = ~np.isnan(col)
        curve[:, c] = np.interp(x, x[ok], col[ok]) if ok.any() else 0.0
    return curve


def dissipation_rate(density_curve):
    """Fractional decline per bin of channel-averaged density (higher = faster)."""
    mean_curve = density_curve.mean(axis=1)
    trend = np.polyfit(np.arange(len(mean_curve)), mean_curve, 1)[0]
    return float(-trend / (mean_curve.mean() + 1e-9))


def run(dataset):
    ensure_dirs()
    sw_dir, out_dir = PATHS[f"{dataset}_sw"], PATHS[f"{dataset}_curves"]
    files = sorted(sw_dir.glob("*.npz"))
    print(f"=== {dataset}: building dissipation curves for {len(files)} subjects ===")

    store, rates = {}, {}
    for f in files:
        sc, dc, ch_names, bh = build_curves(f)
        store[f.stem] = (sc, dc, ch_names, bh)
        rates[f.stem] = dissipation_rate(dc)

    if not rates:
        print("  no subjects")
        return

    median = float(np.median(list(rates.values())))
    print(f"  median density-dissipation rate = {median:.4f} (split point)")
    for subj, (sc, dc, ch_names, bh) in store.items():
        label = int(rates[subj] >= median)
        np.savez(out_dir / f"{subj}.npz", slope_curve=sc, density_curve=dc,
                 ch_names=ch_names, bin_hours=bh, rate=rates[subj], label=label)
        print(f"  {subj:14} rate {rates[subj]:+.4f} -> {'FAST' if label else 'slow'}")


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["dreams", "dreams_subjects", "sleep_edf"]):
        run(ds)
