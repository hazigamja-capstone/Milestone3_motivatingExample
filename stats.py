"""샘플 단위 bootstrap 신뢰구간과 Holm 보정. 입력은 샘플마다 맞았는지(bool 배열)."""
import numpy as np


def bootstrap_idx(n, n_boot=10000, seed=0):
    return np.random.default_rng(seed).integers(0, n, (n_boot, n))


def acc_ci(correct, idx):
    """정확도와 95% 신뢰구간."""
    c = np.asarray(correct, np.float64)
    boot = c[idx].mean(1)
    return c.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def paired_diff(a, b, idx):
    """a - b 정확도 차이, 95% 신뢰구간, 양측 p값 (bootstrap 분포가 0을 넘는 비율)."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    boot = d[idx].mean(1)
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), p


def holm(pvals, alpha=0.05):
    """Holm 보정. 보정 p값과 유의 여부."""
    p = np.asarray(pvals, np.float64)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj, adj < alpha
