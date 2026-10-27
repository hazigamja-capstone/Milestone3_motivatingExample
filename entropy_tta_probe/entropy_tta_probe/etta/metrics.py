from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- torch losses
def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logp = logits.log_softmax(-1)
    return -(logp.exp() * logp).sum(-1)


def marginal_entropy(logits: torch.Tensor) -> torch.Tensor:
    """TPT objective: entropy of the averaged distribution over views. logits [V, C]."""
    logp = logits.log_softmax(-1)
    avg = torch.logsumexp(logp, dim=0) - math.log(logp.shape[0])
    avg = avg.clamp(min=torch.finfo(avg.dtype).min)
    return -(avg.exp() * avg).sum()


def select_confident(logits: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    """TPT confidence selection: indices of the `ratio` lowest-entropy views."""
    k = max(1, int(logits.shape[0] * ratio))
    return entropy_from_logits(logits).argsort()[:k]


# ----------------------------------------------------------------------------- numpy metrics
def np_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def np_entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(-1)


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def reliability(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append(((lo + hi) / 2, conf[m].mean() if m.any() else np.nan,
                     correct[m].mean() if m.any() else np.nan, int(m.sum())))
    return rows


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean())


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """AUROC of `score` for detecting positive==1 (rank-based, no sklearn dependency)."""
    positive = positive.astype(bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = score.argsort()
    ranks = np.empty(len(score))
    # average ranks for ties
    s_sorted = score[order]
    i = 0
    r = np.arange(1, len(score) + 1, dtype=float)
    while i < len(score):
        j = i
        while j + 1 < len(score) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2
        i = j + 1
    ranks[order] = r
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under risk–coverage curve (lower = confidence ranks errors better)."""
    order = (-conf).argsort()
    err = 1 - correct[order]
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risk.mean())


def rao_quadratic_entropy(p: np.ndarray, sim: np.ndarray, topk: int = 20) -> np.ndarray:
    """Σ_ij p_i p_j (1 - s_ij): uncertainty that accounts for *how different* the competing classes are.
    Computed on the top-k classes per sample (rest of the mass is negligible for CLIP logits)."""
    idx = np.argsort(-p, axis=1)[:, :topk]
    out = np.empty(len(p))
    for n in range(len(p)):
        q = p[n, idx[n]]
        q = q / q.sum()
        d = 1 - sim[np.ix_(idx[n], idx[n])]
        out[n] = q @ d @ q
    return out
