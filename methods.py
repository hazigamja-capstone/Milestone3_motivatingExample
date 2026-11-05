"""CLIP 이미지 인코더의 LayerNorm만 업데이트하는 온라인 TTA (Tent, ETA, SAR).

텍스트 분류기는 고정한다. 각 step은 업데이트 전 logits를 예측으로 돌려준다 (원 논문들과 같음).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy(logits):
    logp = logits.float().log_softmax(1)
    return -(logp.exp() * logp).sum(1)


class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, text_feat):
        super().__init__()
        self.visual = clip_model.visual
        self.register_buffer("text", text_feat)
        self.scale = clip_model.logit_scale.exp().item()

    def forward(self, x):
        f = F.normalize(self.visual(x).float(), dim=-1)
        return self.scale * f @ self.text.float().T


def ln_params(net, exclude_last_blocks=0):
    """visual 안의 LayerNorm affine만 학습. SAR처럼 위쪽 블록과 ln_post를 뺄 수 있다."""
    net.requires_grad_(False)
    n_blocks = len(net.visual.transformer.resblocks)
    params, names = [], []
    for name, m in net.visual.named_modules():
        if not isinstance(m, nn.LayerNorm):
            continue
        if exclude_last_blocks:
            if name == "ln_post":
                continue
            if name.startswith("transformer.resblocks."):
                if int(name.split(".")[2]) >= n_blocks - exclude_last_blocks:
                    continue
        for pn, p in m.named_parameters():
            p.requires_grad_(True)
            params.append(p)
            names.append(f"{name}.{pn}")
    return params, names


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_cls, rho=0.05, **kw):
        super().__init__(params, dict(rho=rho, **kw))
        self.base = base_cls(self.param_groups, **kw)
        self.param_groups = self.base.param_groups

    @torch.no_grad()
    def first_step(self):
        grads = [p.grad.norm(2) for g in self.param_groups for p in g["params"] if p.grad is not None]
        norm = torch.norm(torch.stack(grads), 2) if grads else torch.tensor(0.0)
        for g in self.param_groups:
            scale = g["rho"] / (norm + 1e-12)
            for p in g["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale)
        self.zero_grad()

    @torch.no_grad()
    def second_step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if "old_p" in self.state[p]:
                    p.data = self.state[p].pop("old_p")
        self.base.step()
        self.zero_grad()


class Base:
    exclude_last_blocks = 0

    def __init__(self, net, lr, num_classes, amp=True):
        self.net, self.lr, self.C, self.amp = net, lr, num_classes, amp
        self.params, _ = ln_params(net, self.exclude_last_blocks)
        self.init_state = [p.detach().clone() for p in self.params]
        self.opt = self.make_opt()
        self.e_margin = 0.4 * math.log(num_classes)

    def make_opt(self):
        return torch.optim.SGD(self.params, lr=self.lr, momentum=0.9)

    def fwd(self, x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            return self.net(x).float()

    def reset(self):
        with torch.no_grad():
            for p, p0 in zip(self.params, self.init_state):
                p.copy_(p0)
        self.opt = self.make_opt()
        self.on_reset()

    def on_reset(self):
        pass


class NoAdapt(Base):
    def __init__(self, net, lr, num_classes, amp=True):
        self.net, self.amp = net, amp
        net.requires_grad_(False)

    @torch.no_grad()
    def step(self, x):
        return self.fwd(x)

    def reset(self):
        pass


class Tent(Base):
    def step(self, x):
        logits = self.fwd(x)
        loss = entropy(logits).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return logits.detach()


class ETA(Base):
    """EATA에서 Fisher 정규화를 뺀 버전 (EATA 논문의 ETA)."""

    d_margin = 0.05

    def on_reset(self):
        self.ema_probs = None

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ema_probs = None

    def step(self, x):
        logits = self.fwd(x)
        ent = entropy(logits)
        keep = ent < self.e_margin
        probs = logits.softmax(1)
        if self.ema_probs is not None:
            cos = F.cosine_similarity(self.ema_probs[None], probs, dim=1)
            keep = keep & (cos.abs() < self.d_margin)
        if keep.any():
            with torch.no_grad():
                m = probs[keep].mean(0).detach()
                self.ema_probs = m if self.ema_probs is None else 0.9 * self.ema_probs + 0.1 * m
            w = 1.0 / torch.exp(ent[keep].detach() - self.e_margin)
            loss = (ent[keep] * w).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return logits.detach()


class SAR(Base):
    """SAR (ICLR 2023): 신뢰 샘플만 + SAM + loss EMA가 0.2 아래면 모델 복구."""

    reset_constant = 0.2

    def __init__(self, net, lr, num_classes, amp=True):
        n = len(net.visual.transformer.resblocks)
        self.exclude_last_blocks = max(1, n // 4)  # ViT-B 12블록이면 원 논문처럼 위 3블록 제외
        super().__init__(net, lr, num_classes, amp)
        self.ema = None
        self.n_resets = 0

    def make_opt(self):
        return SAM(self.params, torch.optim.SGD, rho=0.05, lr=self.lr, momentum=0.9)

    def on_reset(self):
        self.ema = None

    def step(self, x):
        logits = self.fwd(x)
        ent = entropy(logits)
        keep = ent < self.e_margin
        if not keep.any():
            return logits.detach()
        ent[keep].mean().backward()
        self.opt.first_step()

        ent2 = entropy(self.fwd(x))[keep]
        keep2 = ent2 < self.e_margin
        if keep2.any():
            loss2 = ent2[keep2].mean()
            v = loss2.item()
            self.ema = v if self.ema is None else 0.9 * self.ema + 0.1 * v
            loss2.backward()
        self.opt.second_step()
        if self.ema is not None and self.ema < self.reset_constant:
            self.reset()
            self.n_resets += 1  # 붕괴 감지로 인한 복구만 센다
        return logits.detach()


METHODS = {"none": NoAdapt, "tent": Tent, "eta": ETA, "sar": SAR}
