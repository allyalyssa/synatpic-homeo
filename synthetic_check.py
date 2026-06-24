"""Step 3 — synthetic sanity check.

Before trusting anything on real EEG, prove the model + training code work on
data with a KNOWN answer. We build two obviously-separable classes of
dissipation curve and confine the entire class difference to a few early bins.
A correct model must then (a) classify near-perfectly and (b) put its FAITHFUL
temporal importance on exactly those bins.

Running this is what told us attention is the wrong readout: the model classifies
perfectly but its attention is flat, while input-gradient saliency localizes
cleanly to the signal bins. So the PASS criterion is classification + saliency
localization, and we print attention alongside purely to document its failure.

Run: python synthetic_check.py
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from config import PATHS, SEED, ensure_dirs
from model import make_model
from train import set_seed, fit, predict
from interpret import attention_profile, temporal_saliency

SIGNAL_BINS = (0, 1, 2)     # the only bins that carry class information
N_BINS, N_CHANNELS = 12, 6


def make_synthetic(n=160, noise=0.35, seed=SEED):
    """Two classes of dissipation curve, separable only in SIGNAL_BINS.

    Class 1 ("fast dissipater"): a steep, early, decaying bump in the first 3
    bins. Class 0 ("flat"): no bump. All later bins are identical noise for both
    classes, so a faithful attention map must ignore them.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(0, noise, (n, N_BINS, N_CHANNELS)).astype(np.float32)

    decay = np.array([1.0, 0.6, 0.3], dtype=np.float32)   # steep early decline
    for i, b in enumerate(SIGNAL_BINS):
        X[y == 1, b, :] += decay[i]

    # Continuous "dissipation rate" target: early-minus-late amplitude. High for
    # the bumped class, ~0 for the flat class. This is what the regression head fits.
    rate = X[:, SIGNAL_BINS, :].mean((1, 2)) - X[:, 3:, :].mean((1, 2))
    return X, y.astype(np.float32), rate.astype(np.float32)


def main():
    set_seed()
    ensure_dirs()
    X, y, rate = make_synthetic()

    # stratified train / val / test
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.2, stratify=y[tr], random_state=SEED)

    def t(a):
        return torch.tensor(a)

    model = make_model(N_CHANNELS)
    model, hist = fit(
        model, t(X[tr]), t(y[tr]), t(rate[tr]),
        val=(t(X[va]), t(y[va]), t(rate[va])), verbose=False,
    )

    prob, _, _ = predict(model, t(X[te]))
    acc = float(((prob > 0.5) == y[te]).mean())
    auc = roc_auc_score(y[te], prob)

    attn = attention_profile(model, t(X[te]))         # unfaithful (reported only)
    sal = temporal_saliency(model, t(X[te]))          # faithful localizer

    sal_signal = sal[list(SIGNAL_BINS)].sum()
    attn_signal = attn[list(SIGNAL_BINS)].sum()

    print(f"test accuracy      : {acc:.3f}")
    print(f"test AUC           : {auc:.3f}")
    print(f"saliency/bin       : {np.array2string(sal, precision=3, floatmode='fixed')}")
    print(f"  -> signal bins {SIGNAL_BINS} hold {sal_signal:.0%} of saliency; peak = {sal.argmax()}")
    print(f"attention/bin      : {np.array2string(attn, precision=3, floatmode='fixed')}")
    print(f"  -> signal bins hold only {attn_signal:.0%} of attention; peak = {attn.argmax()}  (unfaithful, as expected)")

    _plot(attn, sal, X, y)

    ok = acc > 0.9 and auc > 0.95 and int(sal.argmax()) in SIGNAL_BINS and sal_signal > 0.5
    print("\nSANITY CHECK:", "PASS (model + faithful saliency work)" if ok else "FAIL")
    return ok


def _plot(attn, sal, X, y):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    bins = np.arange(N_BINS)
    colors = ["#d62728" if b in SIGNAL_BINS else "#c6dbef" for b in bins]

    ax[0].bar(bins, sal, color=colors)
    ax[0].set(title="Temporal saliency (faithful)", xlabel="bin", ylabel="importance")
    ax[1].bar(bins, attn, color=colors)
    ax[1].set(title="Attention (unfaithful, flat)", xlabel="bin", ylabel="weight")

    for label, c in [(0, "#1f77b4"), (1, "#d62728")]:
        ax[2].plot(bins, X[y == label].mean((0, 2)), marker="o", color=c, label=f"class {label}")
    ax[2].set(title="Mean synthetic curves", xlabel="bin", ylabel="amplitude")
    ax[2].legend()
    fig.tight_layout()
    out = PATHS["figures"] / "synthetic_sanity.png"
    fig.savefig(out, dpi=130)
    print(f"figure saved: {out}")


if __name__ == "__main__":
    main()
