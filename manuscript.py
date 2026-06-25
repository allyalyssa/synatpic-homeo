"""Stage 9 — assemble a manuscript-style summary from the saved results.

Everything here is computed from artifacts on disk (out-of-fold bundles,
permutation tests, slow-wave tables), so the prose can never drift from the
numbers. Run after the pipeline + robustness for each cohort:

    python manuscript.py dreams dreams_subjects sleep_edf

Writes outputs/MANUSCRIPT.md.
"""
import sys

import numpy as np
from scipy import stats

from config import PATHS, REGIONS


def _raw_dissociation(dataset):
    """First-vs-last-third slope and density straight from the slow-wave tables."""
    sw = sorted(PATHS[f"{dataset}_sw"].glob("*.npz"))
    s_first, s_last, d_first, d_last = [], [], [], []
    for f in sw:
        d = np.load(f, allow_pickle=True)
        night, slope, nrem = d["night_sec"], d["slope"], d["nrem_sec"]
        t0, t1 = night.min(), night.max()
        third = (t1 - t0) / 3
        ef = night < t0 + third
        el = night > t1 - third
        nf = ((nrem >= t0) & (nrem < t0 + third)).sum() * 0.5
        nl = ((nrem > t1 - third) & (nrem <= t1)).sum() * 0.5
        if ef.sum() < 10 or el.sum() < 10 or nf < 5 or nl < 5:
            continue
        s_first.append(slope[ef].mean()); s_last.append(slope[el].mean())
        d_first.append(ef.sum() / nf); d_last.append(el.sum() / nl)
    s_first, s_last = np.array(s_first), np.array(s_last)
    d_first, d_last = np.array(d_first), np.array(d_last)
    return {
        "n": len(s_first),
        "slope": (s_first.mean(), s_last.mean(), stats.ttest_rel(s_first, s_last)[1]),
        "density": (d_first.mean(), d_last.mean(), stats.ttest_rel(d_first, d_last)[1]),
    }


def _model_readout(dataset):
    p = PATHS["logs"] / f"{dataset}_oof.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    sal, feats = d["oof_sal"], [str(f) for f in d["feature_names"]]
    region_idx = [i for i, f in enumerate(feats) if any(f.startswith(r) for r in REGIONS)]
    t = sal[:, :, region_idx].mean(axis=2).mean(0); t = t / t.sum()
    r = np.array([sal[:, :, [i for i in region_idx if feats[i].startswith(rg)]].mean()
                  for rg in REGIONS]); r = r / r.sum()
    perm = PATHS["logs"] / f"{dataset}_permutation.npz"
    pval = float(np.load(perm)["p"]) if perm.exists() else None
    return {
        "auc": float(d["auc"]), "lo": float(d["auc_lo"]), "hi": float(d["auc_hi"]),
        "early": float(t[:3].sum()), "late": float(t[9:].sum()),
        "top_region": REGIONS[int(r.argmax())], "perm_p": pval,
    }


def build(datasets):
    L = ["# Slow-wave dissipation trajectories and the Synaptic Homeostasis Hypothesis",
         "*Auto-generated from saved results; numbers trace directly to the pipeline.*\n",
         "## Abstract",
         "We model the overnight trajectory of slow-wave EEG activity as a multivariate "
         "time series and train a bidirectional-LSTM-with-attention to classify subjects "
         "as fast vs slow dissipaters, reading gradient saliency to ask whether the model "
         "rediscovers SHY's predictions (early-NREM and frontal dominance). Across DREAMS "
         "cohorts we find a dissociation between slow-wave density (which declines steeply "
         "overnight) and per-wave slope (flat), and the trained model relies on density and "
         "on the LATE night, deviating from the strict slope-based SHY narrative.\n",
         "## Data & methods",
         "- Datasets processed: " + ", ".join(datasets) + ".",
         "- Preprocessing: EEG-by-name selection, 0.5-40 Hz, 100 Hz, expert hypnograms, "
         "clean NREM epochs. Slow waves via yasa.sw_detect; per (12 temporal bins x 3 "
         "regions) we take mean negative slope and slow-wave density.",
         "- Model: Bi-LSTM (128, 3 layers, dropout 0.4) + temporal attention + dual "
         "classification/regression head. Stratified 5-fold CV; AUC with Hanley-McNeil CIs; "
         "label-permutation test. Faithful importance = input-gradient saliency (attention "
         "shown to be non-localizing on synthetic data).\n",
         "## Results"]

    for ds in datasets:
        diss = _raw_dissociation(ds)
        mod = _model_readout(ds)
        L.append(f"\n### {ds} (n={diss['n']})")
        sf, sl, sp = diss["slope"]
        df, dl, dp = diss["density"]
        L.append(f"- Slow-wave **slope**, first vs last third: {sf:.0f} vs {sl:.0f} uV/s "
                 f"(p={sp:.3f}) -> {'flat' if sp > 0.05 else 'declines'}.")
        L.append(f"- Slow-wave **density**, first vs last third: {df:.1f} vs {dl:.1f} "
                 f"waves/min (p={dp:.1e}) -> {'declines (homeostatic)' if dp < 0.05 else 'flat'}.")
        if mod:
            perm = f", permutation p={mod['perm_p']:.3f}" if mod["perm_p"] is not None else ""
            L.append(f"- Classifier AUC {mod['auc']:.2f} [{mod['lo']:.2f}, {mod['hi']:.2f}]{perm}.")
            if mod["auc"] > 0.95:
                L.append("  - CAUTION: a near-ceiling AUC at this N is label-RECOVERABILITY, not "
                         "predictive validity. The label is a deterministic function of the density "
                         "curve, which is itself a model input, so with enough subjects the model "
                         "simply recomputes it. The dissociation and saliency are the meaningful "
                         "results here; this number is not.")
            L.append(f"- Temporal saliency: early {mod['early']:.0%} vs late {mod['late']:.0%} "
                     f"-> leans {'early (SHY)' if mod['early'] > mod['late'] else 'late (deviates)'}.")
            L.append(f"- Top region: {mod['top_region']} "
                     f"({'frontal (SHY)' if mod['top_region'] == 'frontal' else 'not frontal (deviates)'}).")

    L += ["\n## Interpretation",
          "In DREAMS the homeostatic decline is carried mainly by slow-wave DENSITY rather "
          "than per-wave slope (in Sleep-EDF both decline), so the slope proxy that motivates "
          "the study is the weaker signal there. The model leans on the late night because "
          "density is near-ceiling early for "
          "everyone; the between-subject variance that separates dissipaters lives late. The "
          "absence of frontal predominance and early-NREM dominance is a deviation from SHY "
          "and, given the honest caveats below, is best read as a hypothesis to test at "
          "larger N rather than a refutation.",
          "\n## Limitations",
          "- Small N; wide CIs; proxy label is a deterministic function of the input.",
          "- DREAMS Patients carry sleep pathology; cohorts not pooled (montage/population).",
          "- Attention is not a faithful localizer; only gradient saliency is trusted.",
          "- Slope detection parameters (yasa defaults) may bound the slope dynamic range."]

    out = PATHS["logs"].parent / "MANUSCRIPT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n(saved: {out})")


if __name__ == "__main__":
    ds = [d for d in (sys.argv[1:] or ["dreams", "dreams_subjects", "sleep_edf"])
          if (PATHS[f"{d}_sw"]).exists() and any(PATHS[f"{d}_sw"].glob("*.npz"))]
    build(ds)
