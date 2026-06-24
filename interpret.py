"""Stage 8 — interpretability readouts.

The synthetic sanity check (step 3) established, on data with a known answer,
that the attention weights of a (bi)LSTM do NOT localize to the input bins that
actually drive the prediction — the recurrence diffuses signal across all time
steps. Input-gradient saliency does localize. So:

  * `temporal_saliency` / `channel_saliency` are the FAITHFUL importance maps and
    carry the scientific claims (early-NREM? frontal channels?).
  * `attention_profile` is reported for completeness but is flagged as an
    unfaithful localizer, per Jain & Wallace (2019), "Attention is not Explanation".

All functions take X as a (N, T, F) float tensor and return numpy arrays.
"""
import sys

import numpy as np
import torch
from scipy import stats

from config import PATHS, REGIONS


def attention_profile(model, X, device="cpu"):
    """Mean attention weight per temporal bin (N -> T). NOT a faithful localizer."""
    model = model.to(device).eval()
    with torch.no_grad():
        _, _, attn = model(X.to(device))
    return attn.cpu().numpy().mean(0)


def _abs_grad(model, X, device="cpu"):
    """|d logit / d input|, shape (N, T, F). Saliency for the classification head."""
    model = model.to(device).eval()
    x = X.to(device).clone().requires_grad_(True)
    logit, _, _ = model(x)
    logit.sum().backward()
    return x.grad.abs().cpu().numpy()


def temporal_saliency(model, X, device="cpu"):
    """Faithful per-bin importance (T,): |gradient| averaged over channels + subjects."""
    g = _abs_grad(model, X, device)            # (N, T, F)
    prof = g.mean(axis=(0, 2))
    return prof / prof.sum()


def channel_saliency(model, X, device="cpu"):
    """Faithful per-channel importance (F,): |gradient| averaged over bins + subjects."""
    g = _abs_grad(model, X, device)            # (N, T, F)
    prof = g.mean(axis=(0, 1))
    return prof / prof.sum()


# ---------------------------------------------------------------------------
# Step 6 — read the held-out saliency and ask whether the model rediscovers SHY
# ---------------------------------------------------------------------------
def summarize(dataset):
    """Turn the out-of-fold saliency bundle into figures + an honest writeup.

    SHY makes two testable predictions about what a faithful model should weight:
    early NREM (temporal) and frontal channels (spatial). We test both on
    gradient saliency from subjects each fold never trained on.
    """
    d = np.load(PATHS["logs"] / f"{dataset}_oof.npz", allow_pickle=True)
    sal = d["oof_sal"]                         # (N, 12, 5): regions then sin,cos
    attn = d["oof_attn"].mean(0)              # (12,) flagged-unfaithful readout
    n_reg = len(REGIONS)
    auc, lo, hi = float(d["auc"]), float(d["auc_lo"]), float(d["auc_hi"])

    # temporal importance: region-feature saliency per bin (exclude circadian dims)
    temporal = sal[:, :, :n_reg].mean(axis=2)            # (N, 12)
    t_prof = temporal.mean(0)
    t_prof = t_prof / t_prof.sum()
    early = temporal[:, :3].mean(1)
    late = temporal[:, 9:].mean(1)
    t_stat, t_p = stats.ttest_rel(early, late)

    # spatial importance: per-region saliency over bins
    region = sal[:, :, :n_reg].mean(axis=1)              # (N, 3)
    r_prof = region.mean(0)
    r_prof = r_prof / r_prof.sum()
    fi = REGIONS.index("frontal")
    others = [i for i in range(n_reg) if i != fi]
    f_stat, f_p = stats.ttest_rel(region[:, fi], region[:, others].mean(1))

    _plot_interpretation(dataset, t_prof, attn, r_prof)
    text = _writeup(dataset, auc, lo, hi, t_prof, early, late, t_p,
                    r_prof, region[:, fi].mean(), f_p)
    out = PATHS["logs"].parent / f"{dataset}_interpretation.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n(interpretation saved: {out})")


def _plot_interpretation(dataset, t_prof, attn, r_prof):
    import matplotlib.pyplot as plt
    nb = len(t_prof)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    bins = np.arange(nb)
    early_c = ["#d62728" if b < 3 else "#c6dbef" for b in bins]

    ax[0].bar(bins, t_prof, color=early_c)
    ax[0].set(title="Temporal saliency (faithful)", xlabel="temporal bin (early -> late)",
              ylabel="relative importance")
    ax[0].annotate("early NREM\n(SHY prediction)", (1, t_prof.max()), color="#d62728",
                   ha="center", fontsize=8)

    ax[1].bar(bins, attn, color="#bbbbbb")
    ax[1].set(title="Attention (unfaithful)", xlabel="temporal bin", ylabel="weight")

    colors = ["#d62728" if r == "frontal" else "#9ecae1" for r in REGIONS]
    ax[2].bar(REGIONS, r_prof, color=colors)
    ax[2].set(title="Regional saliency", ylabel="relative importance")

    fig.suptitle(f"{dataset}: does the model rediscover SHY?", y=1.02)
    fig.tight_layout()
    p = PATHS["figures"] / f"{dataset}_interpretation.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"figure saved: {p}")


def _writeup(dataset, auc, lo, hi, t_prof, early, late, t_p, r_prof, frontal_mean, f_p):
    early_share = t_prof[:3].sum()
    late_share = t_prof[9:].sum()
    top_region = REGIONS[int(np.argmax(r_prof))]
    temporal_dir = "early" if early.mean() > late.mean() else "late"
    L = [f"# {dataset}: interpretation\n",
         f"**Classifier (held-out):** AUC {auc:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]. ",
         "The CI " + ("excludes" if lo > 0.5 else "includes") +
         " 0.5, so the model is " +
         ("weakly above chance" if lo > 0.5 else "not reliably above chance") +
         f" at this N. Read the saliency below with that caveat.\n",
         "\n## Temporal: does it attend to early NREM (SHY)?",
         f"- Early bins (0-2) hold {early_share:.0%} of temporal saliency; "
         f"late bins (9-11) hold {late_share:.0%}.",
         f"- Peak importance is at bin {int(np.argmax(t_prof))}; the model leans "
         f"{temporal_dir}-night (paired early-vs-late t-test p={t_p:.3f}).",
         "- SHY predicts early dominance. " +
         ("This is consistent with SHY." if early.mean() > late.mean() and t_p < 0.05
          else "This does NOT clearly support SHY's early-NREM prediction."),
         "\n## Spatial: does it weight frontal channels (SHY)?",
         f"- Regional saliency: " +
         ", ".join(f"{r} {v:.0%}" for r, v in zip(REGIONS, r_prof)) + ".",
         f"- Most-weighted region: {top_region} "
         f"(frontal vs others paired t-test p={f_p:.3f}).",
         "- SHY predicts frontal predominance. " +
         ("This is consistent with SHY." if top_region == "frontal" and f_p < 0.05
          else "This does NOT clearly support frontal predominance."),
         "\n## Honest caveats",
         "- N is tiny; saliency from a barely-above-chance model is itself uncertain.",
         "- The label is a deterministic function of the input curve, so accuracy is "
         "partly tautological; the saliency pattern is the real test, not the accuracy.",
         "- Attention does not localize (shown on synthetic data); only gradient "
         "saliency is treated as faithful.",
         "- DREAMS Patients carry sleep pathology - a confound for a homeostasis claim."]
    return "\n".join(L)


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["dreams"]):
        summarize(ds)
