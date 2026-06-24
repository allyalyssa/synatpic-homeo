"""Cross-cohort comparison of the SHY readouts.

Once two or more datasets have been run through train.cross_validate (which saves
each dataset's out-of-fold saliency bundle), this overlays their temporal and
regional saliency so we can ask the key question directly: do healthy sleepers
(DREAMS Subjects) show the early-NREM / frontal pattern SHY predicts, while
patients (DREAMS Patients) deviate?

Run: python compare.py dreams dreams_subjects sleep_edf
"""
import sys

import numpy as np

from config import PATHS, REGIONS


def _profiles(dataset):
    d = np.load(PATHS["logs"] / f"{dataset}_oof.npz", allow_pickle=True)
    sal = d["oof_sal"]                         # (N, 12, n_features)
    feats = [str(f) for f in d["feature_names"]]
    region_idx = [i for i, f in enumerate(feats) if any(f.startswith(r) for r in REGIONS)]

    temporal = sal[:, :, region_idx].mean(axis=2).mean(0)
    temporal = temporal / temporal.sum()
    region = np.array([sal[:, :, [i for i in region_idx if feats[i].startswith(r)]].mean()
                       for r in REGIONS])
    region = region / region.sum()
    return temporal, region, float(d["auc"]), float(d["auc_lo"]), float(d["auc_hi"])


def main(datasets):
    have = [ds for ds in datasets if (PATHS["logs"] / f"{ds}_oof.npz").exists()]
    if not have:
        print("no datasets run yet")
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    print(f"{'dataset':18} {'AUC [95% CI]':20} {'early/late':14} top-region")
    for ds in have:
        t, r, auc, lo, hi = _profiles(ds)
        ax[0].plot(np.arange(len(t)), t, marker="o", label=ds)
        ax[1].plot(REGIONS, r, marker="o", label=ds)
        print(f"{ds:18} {auc:.2f} [{lo:.2f},{hi:.2f}]    "
              f"{t[:3].sum():.0%} / {t[9:].sum():.0%}     {REGIONS[int(r.argmax())]}")

    ax[0].axvspan(-0.5, 2.5, color="#d62728", alpha=0.08)
    ax[0].set(title="Temporal saliency (early NREM shaded)", xlabel="temporal bin",
              ylabel="relative importance")
    ax[0].legend(fontsize=8)
    ax[1].set(title="Regional saliency", ylabel="relative importance")
    ax[1].legend(fontsize=8)
    fig.suptitle("SHY readouts across cohorts: early-NREM? frontal?", y=1.02)
    fig.tight_layout()
    out = PATHS["figures"] / "cohort_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigure saved: {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["dreams", "dreams_subjects", "sleep_edf"])
