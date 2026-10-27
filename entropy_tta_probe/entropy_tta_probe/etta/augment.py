"""
TPT-style view generator (mirrors AugMixAugmenter in the TPT repo):
  view 0 = the original image (resize + center crop)
  views 1..V-1 = RandomResizedCrop + flip, plus AugMix(severity=1) for non-ImageNet datasets
The crop area of each view is recorded so E3 can check *what kind* of views entropy selection prefers.
"""
from __future__ import annotations

import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def base_preprocess(size: int = 224):
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class TPTViews:
    def __init__(self, n_views: int = 64, size: int = 224, augmix: bool = False, seed: int | None = None):
        self.n_views = n_views
        self.size = size
        self.augmix = T.AugMix(severity=1) if augmix else None
        self.base_resize = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size)])
        self.to_tensor = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
        self.rng = random.Random(seed)

    def __call__(self, img):
        views = [self.to_tensor(self.base_resize(img))]
        areas = [1.0]
        W, H = img.size
        for _ in range(self.n_views - 1):
            i, j, h, w = T.RandomResizedCrop.get_params(img, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3))
            v = TF.resized_crop(img, i, j, h, w, [self.size, self.size],
                                interpolation=T.InterpolationMode.BICUBIC)
            if self.rng.random() < 0.5:
                v = TF.hflip(v)
            if self.augmix is not None:
                v = self.augmix(v)
            views.append(self.to_tensor(v))
            areas.append((h * w) / float(H * W))
        return torch.stack(views), torch.tensor(areas)


def collate_views(batch):
    # batch size 1 per sample in episodic TTA
    (views, areas), label, idx = batch[0]
    return views, areas, label, idx
