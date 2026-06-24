"""Stage 7 — training utilities.

This file holds the reusable training core (`set_seed`, `fit`, `predict`). The
synthetic sanity check and the real cross-validated experiment both go through
the SAME `fit`, so a green synthetic test really does exercise the training code.

Stratified k-fold CV with AUC confidence intervals is added on top in
`cross_validate` (used by the real experiment, step 5).
"""
import numpy as np
import torch
import torch.nn as nn

from config import TRAIN, SEED


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
