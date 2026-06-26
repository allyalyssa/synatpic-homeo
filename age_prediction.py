"""De-circularized analysis: predict subject AGE from the dissipation trajectory.

The fast/slow-dissipater label is a deterministic function of the input, so
classifying it is tautological (at large N the model just recomputes it). AGE is
fully EXTERNAL to the dissipation curve, so above-chance age prediction is a real
test of the project's actual question: does the overnight slow-wave trajectory
carry biologically meaningful homeostatic signal? Slow-wave activity collapses
with age, so there is a genuine hypothesis here.

Sleep-EDF cassette subjects span 25-101 yr; ages come straight from the EDF
headers. We reuse the same Bi-LSTM (regression head -> age, classification head ->
old/young median split) and report held-out correlation / R^2 / MAE plus a
label-permutation test, then read saliency for which part of the night carries
the age signal.

Run: python age_prediction.py            # main CV + figure + writeup
     python age_prediction.py permute    # also the permutation test (slow)
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import mne
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, train_test_split

from config import PATHS, SEED, REGIONS, ensure_dirs
from model import make_model
from train import set_seed, fit, predict, auc_hanley_mcneil_ci, _t

mne.set_log_level("ERROR")


def get_ages(ids):
    """Age per subject id, parsed from the EDF header (cached to disk)."""
    cache = PATHS["sleep_edf_raw"].parent / "ages.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        table = dict(zip(d["ids"], d["ages"]))
    else:
        table = {}
        for f in sorted(PATHS["sleep_edf_raw"].glob("*-PSG.edf")):
            sid = os.path.basename(f)[:6]
            info = mne.io.read_raw_edf(f, preload=False, verbose=False).info
            m = re.search(r"(\d+)\s*yr", str((info.get("subject_info") or {}).get("last_name", "")))
            if m:
                table[sid] = int(m.group(1))
        np.savez(cache, ids=list(table), ages=list(table.values()))
    return np.array([table[i] for i in ids], dtype=float)


def _reg_saliency(model, X):
    """|d(predicted age)/d input| per (bin, feature) -- saliency for the AGE head."""
    model.eval()
    x = _t(X).clone().requires_grad_(True)
    _, rate, _ = model(x)
    rate.sum().backward()
    return x.grad.abs().numpy()


def cross_validate_age(k=5, epochs=120):
    ensure_dirs()
    d = np.load(PATHS["harmonized"] / "sleep_edf.npz", allow_pickle=True)
    X = d["X"].astype("float32")
    ids, feats = list(d["ids"]), [str(f) for f in d["feature_names"]]
    age = get_ages(ids)

    mu, sd = age.mean(), age.std()
    y_reg = ((age - mu) / sd).astype("float32")
    y_cls = (age >= np.median(age)).astype("float32")     # old vs young

    # SUBJECT-grouped CV: Sleep-EDF has ~2 nights/subject, so splitting by
    # recording would leak a subject across train/test (see controls.py D).
    groups = np.array([i[3:5] for i in ids])
    gkf = GroupKFold(n_splits=k)
    pred_age = np.zeros(len(age))
    oof_prob = np.zeros(len(age))
    oof_sal = np.zeros_like(X)
    for fold, (tr, te) in enumerate(gkf.split(X, y_cls, groups)):
        tr2, va = train_test_split(tr, test_size=0.2, stratify=y_cls[tr], random_state=SEED)
        set_seed(SEED + fold)
        model = make_model(X.shape[2])
        model, _ = fit(model, _t(X[tr2]), _t(y_cls[tr2]), _t(y_reg[tr2]),
                       val=(_t(X[va]), _t(y_cls[va]), _t(y_reg[va])),
                       epochs=epochs, reg_weight=1.0)
        prob, rate, _ = predict(model, _t(X[te]))
        pred_age[te] = rate * sd + mu
        oof_prob[te] = prob
        oof_sal[te] = _reg_saliency(model, X[te])

    r, p_r = stats.pearsonr(age, pred_age)
    r2 = r2_score(age, pred_age)
    mae = np.mean(np.abs(age - pred_age))
    auc, lo, hi, _ = auc_hanley_mcneil_ci(y_cls, oof_prob)

    print(f"=== Sleep-EDF age prediction (n={len(age)}, held-out) ===")
    print(f"correlation r = {r:.3f} (p={p_r:.1e}) | R^2 = {r2:.3f} | MAE = {mae:.1f} yr")
    print(f"old/young AUC = {auc:.3f} [{lo:.3f}, {hi:.3f}]")

    np.savez(PATHS["logs"] / "sleep_edf_age_oof.npz", pred_age=pred_age, age=age,
             oof_sal=oof_sal, feature_names=feats, r=r, r2=r2, mae=mae, auc=auc)
    _plot(age, pred_age, oof_sal, feats, r, mae)
    _writeup(r, p_r, r2, mae, auc, lo, hi, oof_sal, feats)
    return {"r": r, "r2": r2, "mae": mae, "auc": auc}


def _region_temporal(oof_sal, feats):
    region_idx = [i for i, f in enumerate(feats) if any(f.startswith(r) for r in REGIONS)]
    temporal = oof_sal[:, :, region_idx].mean(axis=2).mean(0)
    temporal = temporal / temporal.sum()
    region = np.array([oof_sal[:, :, [i for i in region_idx if feats[i].startswith(rg)]].mean()
                       for rg in REGIONS])
    return temporal, region / region.sum()


def _plot(age, pred_age, oof_sal, feats, r, mae):
    import matplotlib.pyplot as plt
    temporal, region = _region_temporal(oof_sal, feats)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].scatter(age, pred_age, alpha=0.5, color="#1f77b4")
    lim = [age.min(), age.max()]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(title=f"Held-out age (r={r:.2f}, MAE={mae:.0f}yr)",
              xlabel="true age", ylabel="predicted age")
    bins = np.arange(len(temporal))
    ax[1].bar(bins, temporal, color=["#d62728" if b < 3 else "#c6dbef" for b in bins])
    ax[1].set(title="Temporal saliency for age", xlabel="bin (early->late)", ylabel="importance")
    ax[2].bar(REGIONS, region, color=["#d62728" if rg == "frontal" else "#9ecae1" for rg in REGIONS])
    ax[2].set(title="Regional saliency for age", ylabel="importance")
    fig.suptitle("Predicting age from the SW dissipation trajectory (external target)", y=1.02)
    fig.tight_layout()
    p = PATHS["figures"] / "sleep_edf_age.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"figure saved: {p}")


def _writeup(r, p_r, r2, mae, auc, lo, hi, oof_sal, feats):
    temporal, region = _region_temporal(oof_sal, feats)
    perm = PATHS["logs"] / "sleep_edf_age_perm.npz"
    perm_line = "- Permutation test: run `python age_prediction.py permute`."
    if perm.exists():
        pp = np.load(perm)
        perm_line = (f"- Permutation test: observed r={float(pp['obs']):.3f} vs null mean "
                     f"{float(pp['null'].mean()):.3f}, **p={float(pp['p']):.3f}**.")
    text = "\n".join([
        "# Sleep-EDF: predicting age from the dissipation trajectory\n",
        "Age is external to the dissipation curve, so this is a non-circular test of "
        "whether the overnight slow-wave trajectory carries biologically meaningful signal.\n",
        f"- Held-out correlation r = {r:.3f} (p={p_r:.1e}), R^2 = {r2:.3f}, MAE = {mae:.1f} yr.",
        f"- Old/young classification AUC = {auc:.3f} [{lo:.3f}, {hi:.3f}].",
        perm_line,
        f"- Temporal saliency: early bins (0-2) {temporal[:3].sum():.0%} vs late (9-11) "
        f"{temporal[9:].sum():.0%}; peak bin {int(temporal.argmax())}.",
        f"- Regional saliency: " + ", ".join(f"{rg} {v:.0%}" for rg, v in zip(REGIONS, region)) + ".",
        "\n## Why this matters",
        "Unlike the fast/slow label, age cannot be read off the curve by construction, so a "
        "real correlation means the trajectory genuinely encodes an individually-varying "
        "biological signature - the project's actual hypothesis. The saliency then shows which "
        "part of the night carries that age signal.",
    ])
    out = PATHS["logs"].parent / "sleep_edf_age.md"
    out.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\n(saved: {out})")


def permutation_age(n_perm=200, epochs=40):
    d = np.load(PATHS["harmonized"] / "sleep_edf.npz", allow_pickle=True)
    X = d["X"].astype("float32")
    ids = [str(i) for i in d["ids"]]
    age = get_ages(ids)
    groups = np.array([i[3:5] for i in ids])
    mu, sd = age.mean(), age.std()

    def cv_r(ages):
        y_reg = ((ages - mu) / sd).astype("float32")
        y_cls = (ages >= np.median(ages)).astype("float32")
        gkf = GroupKFold(5)
        pred = np.zeros(len(ages))
        for fold, (tr, te) in enumerate(gkf.split(X, y_cls, groups)):
            tr2, va = train_test_split(tr, test_size=0.2, stratify=y_cls[tr], random_state=SEED)
            set_seed(SEED + fold)
            m = make_model(X.shape[2])
            m, _ = fit(m, _t(X[tr2]), _t(y_cls[tr2]), _t(y_reg[tr2]),
                       val=(_t(X[va]), _t(y_cls[va]), _t(y_reg[va])), epochs=epochs, reg_weight=1.0)
            _, rate, _ = predict(m, _t(X[te]))
            pred[te] = rate * sd + mu
        return stats.pearsonr(ages, pred)[0]

    obs = cv_r(age)
    rng = np.random.default_rng(SEED)
    null = np.array([cv_r(rng.permutation(age)) for _ in range(n_perm)])
    p = (1 + int((null >= obs).sum())) / (1 + n_perm)
    print(f"age permutation: observed r={obs:.3f}, null mean={null.mean():.3f}, p={p:.3f}")
    np.savez(PATHS["logs"] / "sleep_edf_age_perm.npz", obs=obs, null=null, p=p)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "permute":
        permutation_age()
    else:
        cross_validate_age()
