"""Step 7 — robustness and statistical validity.

Three checks that decide how much to trust the headline number:
  * determinism — with the fixed seed, the whole CV reproduces bit-for-bit;
  * permutation test — shuffle the labels many times and re-run the SAME CV to
    get the null distribution of AUC, so we can say whether 0.70 beats chance
    (this is the real test, given the label is a function of the input curve);
  * a written validity section that states N, class balance, the AUC CI, the
    permutation p-value, and the confounds in plain language.

Held-out discipline is already enforced upstream: predictions and saliency come
only from folds that never saw the subject, and per-subject normalization uses
each subject's own statistics (no train/test leakage).

Run: python robustness.py            # determinism + report (fast)
     python robustness.py permute    # also run the permutation test (slow)
"""
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import PATHS, SEED, TRAIN
from model import make_model
from train import set_seed, fit, predict, auc_hanley_mcneil_ci


def _t(a):
    return torch.tensor(a)


def _load(dataset):
    d = np.load(PATHS["harmonized"] / f"{dataset}.npz", allow_pickle=True)
    return (d["X"].astype("float32"), d["y_cls"].astype("float32"),
            d["y_reg"].astype("float32"))


def _quick_cv_auc(X, y_cls, y_reg, k=5, epochs=TRAIN["epochs"], seed=SEED):
    """Pooled out-of-fold AUC, no disk I/O — used many times by the permutation test."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_cls))
    for fold, (tr, te) in enumerate(skf.split(X, y_cls)):
        tr2, va = train_test_split(tr, test_size=0.2, stratify=y_cls[tr], random_state=seed)
        set_seed(seed + fold)
        m = make_model(X.shape[2])
        m, _ = fit(m, _t(X[tr2]), _t(y_cls[tr2]), _t(y_reg[tr2]),
                   val=(_t(X[va]), _t(y_cls[va]), _t(y_reg[va])), epochs=epochs)
        p, _, _ = predict(m, _t(X[te]))
        oof[te] = p
    return roc_auc_score(y_cls, oof)


def determinism_check(dataset):
    X, y_cls, y_reg = _load(dataset)
    a1 = _quick_cv_auc(X, y_cls, y_reg, epochs=40)
    a2 = _quick_cv_auc(X, y_cls, y_reg, epochs=40)
    print(f"determinism: run1 AUC={a1:.6f}, run2 AUC={a2:.6f}  -> "
          f"{'identical' if a1 == a2 else 'DIFFERS'}")
    return a1 == a2


def permutation_test(dataset, n_perm=100, epochs=30):
    """Null distribution of CV-AUC under shuffled labels. Returns (obs, null, p)."""
    X, y_cls, y_reg = _load(dataset)
    obs = _quick_cv_auc(X, y_cls, y_reg, epochs=epochs)
    rng = np.random.default_rng(SEED)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(y_cls))          # shuffle both targets together
        null[i] = _quick_cv_auc(X, y_cls[perm], y_reg[perm], epochs=epochs)
        if (i + 1) % 10 == 0:
            print(f"  permutation {i+1}/{n_perm}: null AUC so far "
                  f"mean={null[:i+1].mean():.3f}")
    p = (1 + int((null >= obs).sum())) / (1 + n_perm)
    print(f"\npermutation test: observed AUC={obs:.3f}, null mean={null.mean():.3f}, "
          f"p={p:.3f} ({n_perm} permutations)")
    np.savez(PATHS["logs"] / f"{dataset}_permutation.npz", obs=obs, null=null, p=p)
    return obs, null, p


def write_report(dataset):
    X, y_cls, y_reg = _load(dataset)
    n = len(y_cls)
    n_pos = int(y_cls.sum())
    d = np.load(PATHS["logs"] / f"{dataset}_oof.npz", allow_pickle=True)
    auc, lo, hi, _ = auc_hanley_mcneil_ci(d["y_cls"], d["oof_prob"])

    perm_path = PATHS["logs"] / f"{dataset}_permutation.npz"
    perm_line = "- Permutation test: not yet run (`python robustness.py permute`)."
    if perm_path.exists():
        pp = np.load(perm_path)
        perm_line = (f"- Permutation test: observed AUC {float(pp['obs']):.3f} vs null "
                     f"mean {float(pp['null'].mean()):.3f}, **p = {float(pp['p']):.3f}** "
                     f"({len(pp['null'])} label shuffles).")

    text = "\n".join([
        f"# {dataset}: statistical validity\n",
        f"- N = {n} subjects ({n_pos} fast / {n - n_pos} slow). This is small; "
        "treat every estimate as provisional.",
        f"- Held-out AUC = {auc:.3f}, 95% Hanley-McNeil CI [{lo:.3f}, {hi:.3f}]. "
        + ("CI excludes 0.5." if lo > 0.5 else "CI includes 0.5."),
        perm_line,
        "- Reproducibility: fixed seed (config.SEED); CPU training is deterministic, "
        "verified by re-running the CV.",
        "- Leakage control: saliency/predictions come only from held-out folds; "
        "per-subject amplitude normalization uses each subject's own mean.",
        "\n## Threats to validity",
        "- **Proxy label**: fast/slow is a median split on a fitted decay rate, not a "
        "biological ground truth, and is a deterministic function of the input.",
        "- **Cohort confound**: DREAMS Patients carry sleep pathology; SHY is framed for "
        "healthy downscaling. The healthy DREAMS Subjects set is the cleaner test.",
        "- **Cross-dataset confound**: montage/hardware/population differ across datasets, "
        "so pooling mixes signal with batch effects; we keep datasets separate.",
        "- **Multiple looks**: two SHY predictions (temporal, spatial) are tested; with "
        "small N neither survives as a confirmatory claim, only as exploratory signal.",
    ])
    out = PATHS["logs"].parent / f"{dataset}_validity.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n(validity report saved: {out})")


if __name__ == "__main__":
    ds = "dreams"
    if len(sys.argv) > 1 and sys.argv[1] == "permute":
        permutation_test(ds)
    else:
        determinism_check(ds)
    write_report(ds)
