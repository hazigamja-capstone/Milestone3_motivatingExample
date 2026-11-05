"""label-free 뷰 신뢰 신호. 모두 '높을수록 믿을 만함' 방향으로 맞춘다.

논문 신호는 공식 구현(methods.py)에서 그대로 뽑는다. 근사 신호와 CLIP에서 원 구현이 성립하지 않는 DeYO(PLPD)는 뺐다.
  mta_inlier  MTA: mode와 함께 최적화한 inlierness
  rtpt_rel    R-TPT: k=20 이웃 코사인 평균 신뢰도
  adte        ADTE: 테스트셋으로 추정한 클래스별 q의 Tsallis 엔트로피 (음수)
"""
import torch

import methods as M

SIGNALS = ["entropy", "maxprob", "margin", "energy", "crop_area", "agreement",
           "mta_inlier", "rtpt_rel", "adte"]


def compute(logits, feats, areas, adte_q=None, mta_out=None):
    """logits [N,V,C], feats [N,V,D], areas [N,V] (torch float). Returns {name: [N,V]}.
    adte_q가 없으면 이 logits 전체(테스트셋)로 추정한다. mta_out = M.mta_batch 결과를 재사용할 때."""
    V = logits.shape[1]
    logp = logits.log_softmax(-1)
    p = logp.exp()
    top2 = p.topk(2, -1).values
    S = {
        "entropy": (p * logp).sum(-1),
        "maxprob": top2[..., 0],
        "margin": top2[..., 0] - top2[..., 1],
        "energy": logits.logsumexp(-1),
        "crop_area": areas,
    }
    pred = p.argmax(-1)
    S["agreement"] = ((pred[:, :, None] == pred[:, None, :]).float().sum(-1) - 1) / (V - 1)
    S["mta_inlier"] = (M.mta_batch(feats, logits) if mta_out is None else mta_out)[1]
    S["rtpt_rel"] = M.rtpt_reliability(feats)
    S["adte"] = -M.adte_entropy(p, M.adte_bias(logits) if adte_q is None else adte_q)
    return S


def to_torch(d, device="cpu"):
    return {k: torch.from_numpy(d[k]).to(device).float() for k in ("logits", "feats", "areas")}
