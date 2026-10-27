"""
CLIP wrapper (open_clip) + CoOp/TPT-style learnable prompt context.

Only the text-side context vectors are tuned (as in TPT / SwapPrompt); the image encoder stays frozen,
so image features of every view can be computed once and reused across optimisation steps.
"""
from __future__ import annotations

from typing import List, Optional

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

SOT, EOT = 49406, 49407

# CoOp's 7-template subset of the 80 CLIP ImageNet templates (used for the template-sensitivity probe)
IMAGENET_TEMPLATES_SELECT = [
    "itap of a {}.", "a bad photo of the {}.", "a origami {}.", "a photo of the large {}.",
    "a {} in a video game.", "art of the {}.", "a photo of the small {}.",
]


class CLIPBundle:
    def __init__(self, arch: str = "RN50", pretrained: str = "openai", device: str = "cuda"):
        pre = None if pretrained in (None, "", "none") else pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pre)
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.device = device
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.arch = arch
        self.input_size = self.preprocess.transforms[0].size
        if isinstance(self.input_size, (list, tuple)):
            self.input_size = self.input_size[0]

    @property
    def logit_scale(self) -> float:
        return float(self.model.logit_scale.exp().item())

    @torch.no_grad()
    def encode_images(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model.encode_image(x.to(self.device)).float(), dim=-1)

    @torch.no_grad()
    def text_features(self, classnames: List[str], templates=("a photo of a {}.",)) -> torch.Tensor:
        """Zero-shot classifier; with several templates the normalised embeddings are averaged (CLIP ensemble).
        Cached per (classnames, templates): ImageNet-C/V2-C splits share the same 1000 class names."""
        if not hasattr(self, "_tf_cache"):
            self._tf_cache = {}
        key = (tuple(classnames), tuple(templates))
        if key in self._tf_cache:
            return self._tf_cache[key]
        out = []
        for c in classnames:
            tok = self.tokenizer([t.format(c) for t in templates]).to(self.device)
            e = F.normalize(self.model.encode_text(tok).float(), dim=-1).mean(0)
            out.append(F.normalize(e, dim=-1))
        self._tf_cache[key] = torch.stack(out)
        return self._tf_cache[key]


class PromptLearner(nn.Module):
    """Shared (class-agnostic) context vectors initialised from a phrase, e.g. 'a photo of a' (TPT default)."""

    def __init__(self, bundle: CLIPBundle, classnames: List[str], ctx_init: str = "a photo of a"):
        super().__init__()
        m, tok = bundle.model, bundle.tokenizer
        self.bundle = bundle
        dev = bundle.device
        n_ctx = int((tok([ctx_init])[0] != 0).sum().item()) - 2
        prompts = [f"{ctx_init} {c}." for c in classnames]
        tokenized = tok(prompts).to(dev)
        # The text transformer is causal, so positions after EOT never influence the EOT feature.
        # Truncating to the longest EOT position gives identical outputs with ~5x less memory
        # (needed to back-propagate through 1000 class prompts on a 16GB GPU).
        self.seq_len = int(tokenized.argmax(dim=-1).max().item()) + 1
        tokenized = tokenized[:, : self.seq_len]
        with torch.no_grad():
            emb = m.token_embedding(tokenized).float()
        self.register_buffer("tokenized", tokenized)
        self.register_buffer("prefix", emb[:, :1].clone())
        self.register_buffer("suffix", emb[:, 1 + n_ctx:].clone())
        self.ctx_init_state = emb[0, 1:1 + n_ctx].clone()
        self.ctx = nn.Parameter(self.ctx_init_state.clone())
        self.n_ctx = n_ctx

    def reset(self):
        with torch.no_grad():
            self.ctx.copy_(self.ctx_init_state)

    def forward(self) -> torch.Tensor:
        """Returns L2-normalised text features [C, D] (differentiable w.r.t. ctx)."""
        m = self.bundle.model
        C = self.prefix.shape[0]
        cast = m.transformer.get_cast_dtype()
        L = self.seq_len
        x = torch.cat([self.prefix, self.ctx.unsqueeze(0).expand(C, -1, -1), self.suffix], dim=1).to(cast)
        x = x + m.positional_embedding[:L].to(cast)
        mask = m.attn_mask[:L, :L] if m.attn_mask is not None else None
        if not getattr(m.transformer, "batch_first", True):
            x = x.permute(1, 0, 2)
            x = m.transformer(x, attn_mask=mask)
            x = x.permute(1, 0, 2)
        else:
            x = m.transformer(x, attn_mask=mask)
        x = m.ln_final(x)
        x = x[torch.arange(C, device=x.device), self.tokenized.argmax(dim=-1)]
        proj = m.text_projection
        x = proj(x) if isinstance(proj, nn.Linear) else x @ proj
        return F.normalize(x.float(), dim=-1)


def make_prompt_learner(bundle, classnames, ctx_init="a photo of a", lr=5e-3, wd=0.0):
    pl = PromptLearner(bundle, classnames, ctx_init).to(bundle.device)
    opt = torch.optim.AdamW([pl.ctx], lr=lr, weight_decay=wd)
    return pl, opt


def logits_from(img_feats: torch.Tensor, txt_feats: torch.Tensor, scale: float) -> torch.Tensor:
    return scale * img_feats @ txt_feats.t()
