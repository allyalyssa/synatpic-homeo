"""Stage 7 — training utilities.

This file holds the reusable training core (`set_seed`, `fit`, `predict`). The
synthetic sanity check and the real cross-validated experiment both go through
the SAME `fit`, so a green synthetic test really does exercise the training code.

Stratified k-fold CV with AUC confidence intervals is added on top in
`cross_validate` (used by the real experiment, step 5).
"""
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import TRAIN, SEED, PATHS, ensure_dirs
from model import make_model
from interpret import attention_profile, _abs_grad


def set_seed(seed=SEED):
    """One call to make a run reproducible (numpy + torch, CPU/GPU)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit(model, X, y_cls, y_reg, val=None, device="cpu", verbose=False,
        epochs=TRAIN["epochs"], lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"],
        batch_size=TRAIN["batch_size"], reg_weight=TRAIN["reg_loss_weight"],
        patience=TRAIN["patience"]):
    """Train the dual-head model. Returns (model, history).

    X: (N, T, F) float tensor. y_cls: (N,) in {0,1}. y_reg: (N,) continuous.
    If `val=(Xv, yv_cls, yv_reg)` is given, we early-stop on validation loss and
    restore the best weights.
    """
    model = model.to(device)
    X, y_cls, y_reg = X.to(device), y_cls.to(device), y_reg.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, best_state, waited = np.inf, None, 0
    n = X.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logit, rate, _ = model(X[idx])
            loss = bce(logit, y_cls[idx]) + reg_weight * mse(rate, y_reg[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        history["train_loss"].append(epoch_loss / n)

        if val is not None:
            Xv, yv_cls, yv_reg = (t.to(device) for t in val)
            model.eval()
            with torch.no_grad():
                logit, rate, _ = model(Xv)
                vloss = (bce(logit, yv_cls) + reg_weight * mse(rate, yv_reg)).item()
                vacc = ((logit > 0).float() == yv_cls).float().mean().item()
            history["val_loss"].append(vloss)
            history["val_acc"].append(vacc)
            if vloss < best_val - 1e-4:
                best_val, best_state, waited = vloss, _clone(model), 0
            else:
                waited += 1
                if waited >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict(model, X, device="cpu"):
    """Return (probabilities, rates, attention weights) as numpy arrays."""
    model = model.to(device).eval()
    with torch.no_grad():
        logit, rate, attn = model(X.to(device))
    return (torch.sigmoid(logit).cpu().numpy(), rate.cpu().numpy(), attn.cpu().numpy())


def _clone(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Cross-validation, AUC confidence interval, and held-out interpretation
# ---------------------------------------------------------------------------
def auc_hanley_mcneil_ci(y_true, scores, alpha=0.05):
    """AUC with a Hanley & McNeil (1982) standard-error confidence interval.

    The parametric SE is appropriate for the tiny N here, where bootstrap CIs are
    themselves unstable. Returns (auc, lo, hi, se).
    """
    y_true = np.asarray(y_true)
    auc = roc_auc_score(y_true, scores)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return auc, np.nan, np.nan, np.nan
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
                  + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))
    z = 1.959963985
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se), se


def cross_validate(dataset, k=TRAIN["k_folds"], device="cpu"):
    """Stratified k-fold CV producing pooled out-of-fold predictions + saliency.

    Saves a checkpoint per fold, the training curves, and an out-of-fold bundle
    (predictions, attention, gradient saliency on subjects the fold never saw) for
    honest interpretation in step 6.
    """
    ensure_dirs()
    d = np.load(PATHS["harmonized"] / f"{dataset}.npz", allow_pickle=True)
    X = d["X"].astype("float32")
    y_cls = d["y_cls"].astype("float32")            # BCE needs float targets
    y_reg = d["y_reg"].astype("float32")
    ids, feats = list(d["ids"]), list(d["feature_names"])
    n, _, n_features = X.shape

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    oof_prob = np.zeros(n)
    oof_attn = np.zeros((n, X.shape[1]))
    oof_sal = np.zeros_like(X)                       # per-subject |d logit / d input|
    fold_acc, fold_auc, histories = [], [], []

    for fold, (tr, te) in enumerate(skf.split(X, y_cls)):
        tr2, va = train_test_split(tr, test_size=0.2, stratify=y_cls[tr], random_state=SEED)
        set_seed(SEED + fold)
        model = make_model(n_features)
        model, hist = fit(model, _t(X[tr2]), _t(y_cls[tr2]), _t(y_reg[tr2]),
                          val=(_t(X[va]), _t(y_cls[va]), _t(y_reg[va])), device=device)
        histories.append(hist)
        torch.save(model.state_dict(), PATHS["models"] / f"{dataset}_fold{fold}.pt")

        prob, _, _ = predict(model, _t(X[te]), device)
        oof_prob[te] = prob
        oof_attn[te] = attention_profile(model, _t(X[te]), device)
        oof_sal[te] = _abs_grad(model, _t(X[te]), device)
        fold_acc.append(float(((prob > 0.5) == y_cls[te]).mean()))
        fold_auc.append(roc_auc_score(y_cls[te], prob) if len(set(y_cls[te])) > 1 else np.nan)

    auc, lo, hi, se = auc_hanley_mcneil_ci(y_cls, oof_prob)
    oof_acc = float(((oof_prob > 0.5) == y_cls).mean())

    print(f"\n=== {dataset}: {k}-fold CV (n={n}) ===")
    print(f"per-fold accuracy : {np.array2string(np.array(fold_acc), precision=2)} "
          f"(mean {np.mean(fold_acc):.2f})")
    print(f"pooled OOF accuracy: {oof_acc:.3f}")
    print(f"pooled OOF AUC     : {auc:.3f}  (95% Hanley-McNeil CI {lo:.3f}-{hi:.3f}, SE {se:.3f})")
    if lo <= 0.5:
        print("  NOTE: CI includes 0.5 -> not distinguishable from chance at this N.")

    np.savez(PATHS["logs"] / f"{dataset}_oof.npz", oof_prob=oof_prob, oof_attn=oof_attn,
             oof_sal=oof_sal, y_cls=y_cls, ids=ids, feature_names=feats,
             auc=auc, auc_lo=lo, auc_hi=hi)
    _plot_training(dataset, histories)
    return {"auc": auc, "ci": (lo, hi), "oof_acc": oof_acc}


def _plot_training(dataset, histories):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for i, h in enumerate(histories):
        ax[0].plot(h["train_loss"], alpha=0.7, label=f"fold {i}")
        if h["val_loss"]:
            ax[1].plot(h["val_loss"], alpha=0.7, label=f"fold {i}")
    ax[0].set(title=f"{dataset}: train loss", xlabel="epoch", ylabel="loss")
    ax[1].set(title="validation loss", xlabel="epoch", ylabel="loss")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    out = PATHS["figures"] / f"{dataset}_training.png"
    fig.savefig(out, dpi=130)
    print(f"training curves: {out}")


def _t(a):
    return torch.tensor(a)


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["dreams"]):
        cross_validate(ds)
