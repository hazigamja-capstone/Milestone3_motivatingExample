"""공식 구현을 dump(뷰 logits, 특징) 위에서 그대로 옮긴 것. 기본값은 각 저장소의 기본값.

  MTA   (MaxZanella/MTA mta.py)      : inlierness와 mode를 번갈아 최적화, mode로 예측
  R-TPT (TomSheng21/R-TPT rtpt.py)   : k=20 이웃 코사인 평균 신뢰도, softmax(/0.01) 가중으로 logits 앙상블
  ADTE  (Jinx630/ADTE run.py, zero.py): 테스트셋 전체로 클래스 편향 q를 추정 (transductive, 라벨 없음),
                                       클래스별 q의 Tsallis 엔트로피 상위 10% 뷰 확률 합으로 예측
  ZERO  (FarinaMatteo/zero zero.py)  : 엔트로피 상위 30% 뷰의 argmax 다수결, 동률은 나머지 뷰 순서로
"""
import torch


def mta(feats, logits, lambda_y=0.2, lambda_q=4.0, max_iter=5, th=1e-6, uniform_y=False):
    """feats [V,D], logits [V,C] (logit scale 포함). Returns (mode [D], inlierness [V]).
    uniform_y: ablation. inlierness 단계를 건너뛰고 균등 가중 MeanShift만 한다."""
    V = feats.shape[0]
    dist = torch.cdist(feats, feats)
    k = int(0.3 * (V - 1))
    bandwidth = torch.sqrt(0.5 * (dist.sort(1).values[:, 1:k + 1] ** 2).mean(1))
    p = logits.softmax(1)
    affinity = p @ p.T
    y = torch.ones(V) / V
    mode = feats[0]

    def kernel(m):
        return torch.exp(-((feats - m).norm(dim=-1) ** 2) / (2 * bandwidth ** 2))

    for _ in range(max_iter):
        density = kernel(mode)
        for _ in range(0 if uniform_y else max_iter):
            old = y
            y = torch.softmax(1 / lambda_y * (density + lambda_q * (affinity * y[None]).sum(1)), -1)
            if torch.norm(old - y) < th:
                break
        for _ in range(max_iter):
            old = mode
            w = kernel(mode) * y
            mode = (w[:, None] * feats).sum(0) / w.sum()
            mode = mode / mode.norm()
            if torch.norm(old - mode) < th:
                break
    return mode, y


def mta_batch(feats, logits, **kw):
    out = [mta(f, l, **kw) for f, l in zip(feats, logits)]
    return torch.stack([m for m, _ in out]), torch.stack([y for _, y in out])


def rtpt_reliability(feats, k=20):
    """feats [N,V,D]. 자기 자신을 뺀 상위 k 코사인 유사도 평균 [N,V]."""
    sim = feats @ feats.transpose(1, 2)
    eye = torch.eye(sim.shape[1], dtype=torch.bool)
    return sim.masked_fill(eye, float("-inf")).topk(k, -1).values.mean(-1)


def rtpt_ensemble(logits, reliability, temp=0.01):
    """logits [N,V,C]를 softmax(신뢰도/temp) 가중 평균한 logits [N,C]."""
    w = (reliability / temp).softmax(-1)
    return (w[..., None] * logits).sum(1)


def adte_bias(logits, memory_size=20, gap_th=1.0, eps=1e-4, n_iter=500):
    """logits [N,V,C], 테스트셋 순서대로. 클래스별 Tsallis 지수 q [C] (0.01~0.9)."""
    N, _, C = logits.shape
    memory = torch.zeros(C, memory_size, C)
    gap = torch.full((C, memory_size), -1.0)
    for n in range(N):
        out = logits[n].mean(0)
        top = out.topk(2).values
        diff = top[0] - top[1]
        if diff >= gap_th:
            c = out.argmax()
            j = gap[c].argmin()
            memory[c, j], gap[c, j] = out, diff
    P = torch.zeros(C, C)
    for j in range(C):
        valid = memory[j][gap[j] != -1]
        if len(valid):
            P[:, j] = valid.softmax(1).sum(0) / len(valid)
    q = torch.ones(C) / C
    for _ in range(n_iter):
        q_new = P @ q
        q_new = q_new / q_new.norm()
        if torch.norm(q_new - q) < eps:
            break
        q = q_new
    logq = q.clamp_min(torch.finfo(q.dtype).tiny).log()
    return 0.01 + (logq - logq.min()) * (0.9 - 0.01) / (logq.max() - logq.min())


def adte_entropy(p, q):
    """클래스별 지수 q의 Tsallis 엔트로피. 낮을수록 확신."""
    return (p ** q / (1 - q)).sum(-1)


def zero_predict(p, entropy, gamma=0.3):
    """p [N,V,C], entropy [N,V] (낮을수록 확신). Returns [N]."""
    N, V, C = p.shape
    k = int(V * gamma)
    order = entropy.argsort(1)
    preds = p.argmax(-1)
    out = torch.empty(N, dtype=torch.long)
    for n in range(N):
        votes = torch.bincount(preds[n, order[n, :k]], minlength=C)
        ties = (votes == votes.max()).nonzero()[:, 0].tolist()
        if len(ties) == 1:
            out[n] = ties[0]
            continue
        rest = order[n, k:]
        hit = [int(c) for c in preds[n, rest] if int(c) in ties]
        # 남은 뷰 중 동률 클래스를 처음 예측한 뷰를 따르고, 없으면 첫 남은 뷰에서 동률 클래스 중 최대 확률
        out[n] = hit[0] if hit else ties[int(p[n, rest[0], ties].argmax())]
    return out
