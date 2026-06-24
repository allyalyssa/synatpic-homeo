"""Stage 6 — the model whose attention weights are the scientific payload.

A bidirectional LSTM reads each subject's dissipation curve as a sequence of 12
temporal bins (features per bin = per-channel normalized slope, plus a circadian
time encoding). A temporal-attention layer collapses the 12 bins into one
context vector and, crucially, exposes a weight per bin: that is what we later
inspect to ask whether the model attends to early NREM, as SHY predicts.

Two output heads share the context vector:
  * classification — fast vs slow dissipater (binary logit)
  * regression     — the continuous dissipation rate

The regression head is not decoration: forcing the same representation to
predict the continuous rate regularizes the small-N classifier.
"""
import torch
import torch.nn as nn

from config import MODEL


class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention over time steps.

    Returns a context vector and the per-time-step weights (which sum to 1).
    """

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(),
                                   nn.Linear(dim // 2, 1))

    def forward(self, h):                       # h: (B, T, dim)
        weights = torch.softmax(self.score(h).squeeze(-1), dim=1)   # (B, T)
        context = (h * weights.unsqueeze(-1)).sum(dim=1)            # (B, dim)
        return context, weights


class DissipationLSTM(nn.Module):
    def __init__(self, n_features, hidden=MODEL["hidden"], layers=MODEL["layers"],
                 dropout=MODEL["dropout"], bidirectional=MODEL["bidirectional"]):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.attn = TemporalAttention(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.clf_head = nn.Linear(out_dim, 1)
        self.reg_head = nn.Linear(out_dim, 1)

    def forward(self, x):                        # x: (B, T, n_features)
        h, _ = self.lstm(x)
        context, attn = self.attn(h)
        context = self.dropout(context)
        logit = self.clf_head(context).squeeze(-1)
        rate = self.reg_head(context).squeeze(-1)
        return logit, rate, attn


def make_model(n_features):
    return DissipationLSTM(n_features)
