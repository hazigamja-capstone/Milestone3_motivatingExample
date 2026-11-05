"""ImageNet-C 공식 수식(Hendrycks & Dietterich, 2019)을 torch로 옮긴 일부 손상.

입력은 [0,1] 범위의 float 텐서 (B,3,224,224). 같은 seed면 같은 손상이 나온다.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

DOMAINS = ["clean", "gaussian_noise", "shot_noise", "defocus_blur", "contrast", "pixelate", "jpeg"]


def _gen(x, seed):
    g = torch.Generator(device=x.device)
    g.manual_seed(int(seed))
    return g


def gaussian_noise(x, s, seed):
    c = [0.08, 0.12, 0.18, 0.26, 0.38][s - 1]
    return (x + torch.randn(x.shape, generator=_gen(x, seed), device=x.device) * c).clamp(0, 1)


def shot_noise(x, s, seed):
    c = [60, 25, 12, 5, 3][s - 1]
    return (torch.poisson(x * c, generator=_gen(x, seed)) / c).clamp(0, 1)


def contrast(x, s, seed):
    c = [0.4, 0.3, 0.2, 0.1, 0.05][s - 1]
    m = x.mean(dim=(2, 3), keepdim=True)
    return ((x - m) * c + m).clamp(0, 1)


def _gauss1d(ksize, sigma):
    r = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
    k = torch.exp(-(r ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _disk(radius, alias_blur):
    if radius <= 8:
        L, ksize = np.arange(-8, 8 + 1), 3
    else:
        L, ksize = np.arange(-radius, radius + 1), 5
    X, Y = np.meshgrid(L, L)
    k = torch.tensor(((X ** 2 + Y ** 2) <= radius ** 2).astype(np.float32))
    k = k / k.sum()
    g = _gauss1d(ksize, alias_blur)
    g2 = (g[:, None] * g[None, :])[None, None]
    k = F.conv2d(F.pad(k[None, None], (ksize // 2,) * 4, mode="reflect"), g2)[0, 0]
    return k / k.sum()


def defocus_blur(x, s, seed):
    radius, alias = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][s - 1]
    k = _disk(radius, alias).to(x.device, x.dtype)
    p = k.shape[-1] // 2
    w = k[None, None].repeat(3, 1, 1, 1)
    return F.conv2d(F.pad(x, (p,) * 4, mode="reflect"), w, groups=3).clamp(0, 1)


def pixelate(x, s, seed):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][s - 1]
    h = x.shape[-1]
    small = F.interpolate(x, size=int(h * c), mode="area")
    return F.interpolate(small, size=h, mode="nearest")


def jpeg(x, s, seed):
    q = [25, 18, 15, 10, 7][s - 1]
    u8 = (x * 255).round().clamp(0, 255).to(torch.uint8).cpu()
    out = [torchvision.io.decode_jpeg(torchvision.io.encode_jpeg(im, quality=q)) for im in u8]
    return torch.stack(out).to(x.device).float() / 255


_FNS = {
    "clean": lambda x, s, seed: x,
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "defocus_blur": defocus_blur,
    "contrast": contrast,
    "pixelate": pixelate,
    "jpeg": jpeg,
}


def corrupt(x, domain, severity, seed):
    return _FNS[domain](x, severity, seed)
