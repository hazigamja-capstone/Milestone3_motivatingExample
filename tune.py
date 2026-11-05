"""TPT / R-TPT 프롬프트 튜닝 (공식 구현 규칙). 이미지 인코더는 고정이라 dump의 뷰 특징을 그대로 쓰고 텍스트 쪽만 튜닝한다.

공통: 학습 가능한 문맥 4토큰("a photo of a"로 초기화, 클래스명은 뒤), 샘플마다 프롬프트와 optimizer 초기화,
      초기 프롬프트 출력의 엔트로피 상위 10% 뷰 선택, AdamW lr 5e-3, 1 step, fp16 autocast + GradScaler(1000)
TPT  : 손실 = 선택 뷰의 marginal 엔트로피, 예측 = 튜닝한 프롬프트로 원본 뷰(view 0)
       (ablation용으로 64뷰 logits를 모두 저장한다)
R-TPT: 손실 = 선택 뷰의 뷰별 엔트로피 평균, 예측 = 튜닝 후 64뷰 logits의 신뢰도 가중 평균 (가중은 matrix.py에서)

python tune.py --dump runs/dump
python tune.py --columns paper_a --n 500 --tpt-only   # 논문 재현 열
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES
from tqdm import tqdm

from common import LOGIT_SCALE, imagenet_a_classes, load_clip, text_features
from data import COLUMNS, PAPER_COLUMNS


class PromptText:
    """CoOp 방식 프롬프트: [SOS] ctx_1..ctx_n 클래스명 . [EOS]"""

    def __init__(self, model, tok, names, device, init="a photo of a"):
        self.model = model
        self.n_ctx = len(init.split())
        self.tokens = tok([f"{init} {n}." for n in names]).to(device)
        with torch.no_grad():
            emb = model.token_embedding(self.tokens)
        self.prefix, self.suffix = emb[:, :1], emb[:, 1 + self.n_ctx:]
        self.ctx_init = emb[0, 1:1 + self.n_ctx].clone()
        self.eot = self.tokens.argmax(-1)

    def features(self, ctx):
        m = self.model
        x = torch.cat([self.prefix, ctx[None].expand(len(self.prefix), -1, -1), self.suffix], 1)
        x = x + m.positional_embedding
        x = m.transformer(x, attn_mask=m.attn_mask)
        x = m.ln_final(x)
        x = x[torch.arange(len(x)), self.eot] @ m.text_projection
        return F.normalize(x.float(), dim=-1)


def view_entropy(logits):
    return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)


def marginal_entropy(logits):
    """TPT avg_entropy: 선택 뷰 확률 평균의 엔트로피."""
    lp = logits.log_softmax(-1)
    avg = (lp.logsumexp(0) - np.log(lp.shape[0])).clamp_min(torch.finfo(lp.dtype).min)
    return -(avg * avg.exp()).sum(-1)


def tune(pt, feats, loss_fn, lr=5e-3, selection_p=0.1):
    """feats [V,D]. 1 step 튜닝 후 모든 뷰의 logits [V,C]."""
    ctx = torch.nn.Parameter(pt.ctx_init.clone().float())
    opt = torch.optim.AdamW([ctx], lr)
    scaler = torch.amp.GradScaler("cuda", init_scale=1000)
    with torch.autocast("cuda", dtype=torch.float16):
        out = LOGIT_SCALE * feats @ pt.features(ctx).T
        idx = view_entropy(out.float()).argsort()[:int(len(feats) * selection_p)]
        loss = loss_fn(out[idx].float())
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        return (LOGIT_SCALE * feats @ pt.features(ctx).T).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="runs/dump")
    ap.add_argument("--columns", nargs="+", default=[c for c in COLUMNS if c != "ood"])
    ap.add_argument("--n", type=int, default=None, help="열마다 앞에서부터 이만큼만 튜닝")
    ap.add_argument("--tpt-only", action="store_true")
    args = ap.parse_args()

    device = "cuda"
    model, tok = load_clip(device)
    for p in model.parameters():
        p.requires_grad_(False)
    for col in args.columns:
        cfg = COLUMNS.get(col) or PAPER_COLUMNS[col]
        names = IMAGENET_CLASSNAMES if cfg.get("classes") == "in1k" else imagenet_a_classes()[2]
        pt = PromptText(model, tok, names, device)
        # 초기 프롬프트가 dump에 쓴 텍스트 특징과 같은지 확인
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            t0 = pt.features(pt.ctx_init)
        ref = text_features(model, tok, names, device)
        print(f"[{col}] init prompt vs dump text features: min cos {(t0 * ref).sum(-1).min().item():.5f}", flush=True)

        d = np.load(os.path.join(args.dump, f"{col}.npz"))
        N = min(len(d["labels"]), args.n or len(d["labels"]))  # 앞에서부터 N개만 (논문 재현 열은 시간 때문에 일부만)
        feats_all, labels = d["feats"][:N], d["labels"][:N]
        V = feats_all.shape[1]
        tpt = np.zeros((N, V, len(names)), np.float16)
        rtpt = np.zeros((N, V, len(names)), np.float16) if not args.tpt_only else None
        for n in tqdm(range(N), desc=col):
            f = torch.from_numpy(feats_all[n]).to(device).float()
            tpt[n] = tune(pt, f, marginal_entropy).cpu().numpy()
            if rtpt is not None:
                rtpt[n] = tune(pt, f, lambda o: view_entropy(o).mean()).cpu().numpy()
        out = dict(tpt=tpt) if rtpt is None else dict(tpt=tpt, rtpt=rtpt)
        np.savez(os.path.join(args.dump, f"{col}_tuned.npz"), **out)
        print(f"[{col}] N={N} TPT acc={(tpt[:, 0].argmax(-1) == labels).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
