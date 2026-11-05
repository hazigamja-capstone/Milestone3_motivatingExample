import os
import random

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from common import DATA, encode_image, LOGIT_SCALE

V2_DIR = "imagenetv2-matched-frequency-format-val"

# 열 = 실패 모드. 각 지표가 겨냥한 상황을 하나씩 만든다.
#   source: v2 (ImageNet-V2, 200 클래스), a (ImageNet-A), v2_ood (V2의 나머지 800 클래스, 라벨 -1)
#   scale: 뷰 생성 RandomResizedCrop 범위 (TPT 기본은 0.08~1)
COLUMNS = {
    "clean": dict(source="v2", scale=(0.08, 1.0)),
    "aug_noise": dict(source="v2", scale=(0.02, 0.15)),           # MTA, crop 면적이 겨냥
    "adversarial": dict(source="v2", scale=(0.08, 1.0), attack=True),  # R-TPT가 겨냥
    "corruption": dict(source="v2", scale=(0.08, 1.0), corrupt="gaussian_noise"),
    "natural_hard": dict(source="a", scale=(0.08, 1.0)),          # ZERO 등 과신 계열이 겨냥
    "ood": dict(source="v2_ood", scale=(0.08, 1.0)),               # online stream 전용
}

# 논문 재현용 열: 논문 프로토콜 그대로 각 데이터셋의 원래 클래스 공간을 쓴다 (실패 모드 열과 섞지 않음)
PAPER_COLUMNS = {
    "paper_a": dict(source="a", scale=(0.08, 1.0), classes="a200"),
    "paper_v2": dict(source="v2_all", scale=(0.08, 1.0), classes="in1k"),
}


def list_items(source, class_idx, wnids):
    if source == "a":
        root = os.path.join(DATA, "imagenet-a")
        return [(os.path.join(root, w, f), y) for y, w in enumerate(wnids)
                for f in sorted(os.listdir(os.path.join(root, w)))]
    root = os.path.join(DATA, V2_DIR)
    if source == "v2_all":  # 1000 클래스 전체, 라벨 = ImageNet 클래스 인덱스
        return [(os.path.join(root, str(k), f), k) for k in range(1000)
                for f in sorted(os.listdir(os.path.join(root, str(k))))]
    pos = {k: y for y, k in enumerate(class_idx)}
    items = []
    for k in range(1000):
        d = os.path.join(root, str(k))
        in_set = k in pos
        if (source == "v2") != in_set:
            continue
        items += [(os.path.join(d, f), pos[k] if in_set else -1) for f in sorted(os.listdir(d))]
    return items


def sample_items(items, n, seed):
    rng = random.Random(seed)
    return items if n is None or n >= len(items) else rng.sample(items, n)


def load_base(path, size=224):
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, size, interpolation=T.InterpolationMode.BICUBIC)
    return TF.to_tensor(TF.center_crop(img, size))


class BaseImages(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        return load_base(path), y


def corrupt(x, kind, gen):
    if kind == "gaussian_noise":  # ImageNet-C severity 3 수준
        return (x + 0.18 * torch.randn(x.shape, generator=gen).to(x.device)).clamp(0, 1)
    if kind == "gaussian_blur":
        return TF.gaussian_blur(x, kernel_size=9, sigma=3.0)
    raise ValueError(kind)


def pgd(model, x, y, text, eps=4 / 255, alpha=1 / 255, steps=7):
    """zero-shot CLIP 예측을 향한 untargeted L_inf PGD. 원본 이미지를 공격한 뒤 뷰를 만든다 (R-TPT 세팅)."""
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(steps):
        logits = LOGIT_SCALE * encode_image(model, (x + delta).clamp(0, 1)) @ text.T
        loss = F.cross_entropy(logits, y)
        (g,) = torch.autograd.grad(loss, delta)
        delta = (delta + alpha * g.sign()).clamp(-eps, eps).detach().requires_grad_(True)
    return (x + delta).clamp(0, 1).detach()


def make_views(base, n_views, scale, gen):
    """view 0 = 원본. 나머지는 RandomResizedCrop + flip. crop 면적 비율을 같이 반환."""
    _, H, W = base.shape
    views, areas = [base], [1.0]
    for _ in range(n_views - 1):
        area = H * W * torch.empty(1).uniform_(*scale, generator=gen).item()
        logr = torch.empty(1).uniform_(-0.2877, 0.2877, generator=gen).item()  # log(3/4)~log(4/3)
        r = torch.exp(torch.tensor(logr)).item()
        w = min(W, max(1, int(round((area * r) ** 0.5))))
        h = min(H, max(1, int(round((area / r) ** 0.5))))
        i = torch.randint(0, H - h + 1, (1,), generator=gen).item()
        j = torch.randint(0, W - w + 1, (1,), generator=gen).item()
        v = TF.resized_crop(base, i, j, h, w, [H, W], antialias=True)
        if torch.rand(1, generator=gen).item() < 0.5:
            v = TF.hflip(v)
        views.append(v)
        areas.append(h * w / (H * W))
    return torch.stack(views), torch.tensor(areas)


def patch_shuffle(x, g=4, gen=None):
    """DeYO의 객체 파괴 변환: g x g 패치를 섞는다."""
    B, C, H, W = x.shape
    p = H // g
    x = x.view(B, C, g, p, g, p).permute(0, 2, 4, 1, 3, 5).reshape(B, g * g, C, p, p)
    x = x[:, torch.randperm(g * g, generator=gen)]
    return x.view(B, g, g, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
