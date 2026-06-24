"""Stage 4 — turn detected slow waves into per-channel dissipation curves and
assign each subject a fast/slow-dissipater label.

For each subject we bin slow waves into DISS['n_bins'] equal-time bins spanning
their slow-wave period and take the mean negative slope per (bin, channel). That
matrix is the dissipation curve: SHY predicts each channel's slope declines from
early to late night.

The label is a median split on the dissipation RATE — the fractional decline of
the channel-averaged curve across the night. The continuous rate is also kept as
the regression target.

HONEST CAVEAT (see README): the label is a deterministic function of the very
curve the model receives, so high classification accuracy is partly tautological.
The scientific test is not the accuracy but the *saliency pattern* — does the
model lean on early bins (temporal) and frontal channels (spatial)? The spatial
question is the more meaningful one, because the label averages over channels and
so privileges no channel a priori.

Output per subject: data/<dataset>/dissipation_curves/<subject>.npz with
  curve     (n_bins, n_ch) float   mean negative slope per bin/channel
  ch_names  list[str]
  rate      float                  dissipation rate (regression target)
  label     int                    1 = fast dissipater, 0 = slow (median split)
"""
import sys

import numpy as np

from config import PATHS, DISS, ensure_dirs


def build_curve(sw_path):
    d = np.load(sw_path, allow_pickle=True)
    slope, night, ch_idx = d["slope"], d["night_sec"], d["ch_idx"]
    ch_names = [str(c) for c in d["ch_names"]]
    n_ch, nb = len(ch_names), DISS["n_bins"]

    edges = np.linspace(night.min(), night.max(), nb + 1)
    bin_of = np.clip(np.digitize(night, edges) - 1, 0, nb - 1)

    curve = np.full((nb, n_ch), np.nan)
    for b in range(nb):
        for c in range(n_ch):
            m = (bin_of == b) & (ch_idx == c)
            if m.any():
                curve[b, c] = slope[m].mean()

    # center time of each bin, in hours since the first slow wave (onset proxy);
    # feature engineering turns this into a circadian-phase encoding.
    centers = (edges[:-1] + edges[1:]) / 2
    bin_hours = (centers - night.min()) / 3600.0
    return _fill_nan(curve), ch_names, bin_hours


def _fill_nan(curve):
    """Linear-interpolate gaps within each channel; edge-extend if needed."""
    x = np.arange(curve.shape[0])
    for c in range(curve.shape[1]):
        col = curve[:, c]
        ok = ~np.isnan(col)
        curve[:, c] = np.interp(x, x[ok], col[ok]) if ok.any() else 0.0
    return curve


def dissipation_rate(curve):
    """Fractional decline per bin of the channel-averaged curve (higher = faster)."""
    mean_curve = curve.mean(axis=1)
    trend = np.polyfit(np.arange(len(mean_curve)), mean_curve, 1)[0]
    return float(-trend / (mean_curve.mean() + 1e-9))


def run(dataset):
    ensure_dirs()
    sw_dir, out_dir = PATHS[f"{dataset}_sw"], PATHS[f"{dataset}_curves"]
    files = sorted(sw_dir.glob("*.npz"))
    print(f"=== {dataset}: building dissipation curves for {len(files)} subjects ===")

    curves, rates, names, hours = {}, {}, {}, {}
    for f in files:
        curve, ch_names, bin_hours = build_curve(f)
        curves[f.stem], names[f.stem], hours[f.stem] = curve, ch_names, bin_hours
        rates[f.stem] = dissipation_rate(curve)

    if not rates:
        print("  no subjects")
        return

    median = float(np.median(list(rates.values())))
    print(f"  median dissipation rate = {median:.4f} (split point)")
    for subj, curve in curves.items():
        label = int(rates[subj] >= median)
        np.savez(out_dir / f"{subj}.npz", curve=curve, ch_names=names[subj],
                 bin_hours=hours[subj], rate=rates[subj], label=label)
        print(f"  {subj:14} rate {rates[subj]:+.4f} -> {'FAST' if label else 'slow'}")


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["dreams", "sleep_edf"]):
        run(ds)
