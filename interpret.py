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
import numpy as np
import torch


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
