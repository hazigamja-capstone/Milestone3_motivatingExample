"""
Dataset registry following the CoOp / TPT directory convention (DATASETS.md of CoOp),
so a data root already prepared for TPT / SwapPrompt works as-is.

SwapPrompt (NeurIPS'23) benchmark = 14 datasets:
  ImageNet, ImageNet-V2, ImageNet-A, ImageNet-R, ImageNet-Sketch,
  Caltech101, DTD, Flowers102, OxfordPets, UCF101, StanfordCars, Food101, EuroSAT, SUN397
Extras (optional):
  fgvc_aircraft            -> completes TPT's 10-dataset cross-domain benchmark
  imagenet_c/<corr>/<sev>  -> controllable shift strength (needed for E1 severity sweep, E4 continual stream)

Expected layout under DATA_ROOT:
  imagenet/classnames.txt                  ("n01440764 tench" per line, CoOp file)
  imagenet/images/val/<wnid>/*.JPEG
  imagenetv2/imagenetv2-matched-frequency-format-val/<0..999>/*
  imagenet-adversarial/imagenet-a/<wnid>/*
  imagenet-rendition/imagenet-r/<wnid>/*
  imagenet-sketch/images/<wnid>/*
  imagenet-c/<corruption>/<severity>/<wnid>/*
  caltech-101/101_ObjectCategories + split_zhou_Caltech101.json
  dtd/images + split_zhou_DescribableTextures.json
  oxford_flowers/jpg + split_zhou_OxfordFlowers.json
  oxford_pets/images + split_zhou_OxfordPets.json
  ucf101/UCF-101-midframes + split_zhou_UCF101.json
  stanford_cars/ + split_zhou_StanfordCars.json
  food-101/images + split_zhou_Food101.json
  eurosat/2750 + split_zhou_EuroSAT.json
  sun397/SUN397 + split_zhou_SUN397.json
  fgvc_aircraft/data/{images, variants.txt, images_variant_test.txt}
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

SWAPPROMPT_DATASETS = [
    "imagenet", "imagenet_v2", "imagenet_a", "imagenet_r", "imagenet_sketch",
    "caltech101", "dtd", "flowers102", "oxford_pets", "ucf101",
    "stanford_cars", "food101", "eurosat", "sun397",
]
IMAGENET_FAMILY = {"imagenet", "imagenet_v2", "imagenet_a", "imagenet_r", "imagenet_sketch"}
IMAGENET_C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]

# CoOp renames EuroSAT folder names into readable phrases
EUROSAT_CNAMES = {
    "AnnualCrop": "Annual Crop Land", "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation Land", "Highway": "Highway or Road",
    "Industrial": "Industrial Buildings", "Pasture": "Pasture Land",
    "PermanentCrop": "Permanent Crop Land", "Residential": "Residential Buildings",
    "River": "River", "SeaLake": "Sea or Lake",
}

COOP_JSON = {
    # name: (subdir, image_dir, split_file)
    "caltech101": ("caltech-101", "101_ObjectCategories", "split_zhou_Caltech101.json"),
    "dtd": ("dtd", "images", "split_zhou_DescribableTextures.json"),
    "flowers102": ("oxford_flowers", "jpg", "split_zhou_OxfordFlowers.json"),
    "oxford_pets": ("oxford_pets", "images", "split_zhou_OxfordPets.json"),
    "ucf101": ("ucf101", "UCF-101-midframes", "split_zhou_UCF101.json"),
    "stanford_cars": ("stanford_cars", "", "split_zhou_StanfordCars.json"),
    "food101": ("food-101", "images", "split_zhou_Food101.json"),
    "eurosat": ("eurosat", "2750", "split_zhou_EuroSAT.json"),
    "sun397": ("sun397", "SUN397", "split_zhou_SUN397.json"),
}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class Split:
    name: str
    items: List[Tuple[str, int]]
    classnames: List[str]
    meta: dict = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.classnames)


class ImageListDataset(Dataset):
    """Returns (transform(PIL), label, index). transform may return anything (e.g. a list of views)."""

    def __init__(self, split: Split, transform: Optional[Callable] = None, indices: Optional[List[int]] = None):
        self.split = split
        self.transform = transform
        self.indices = list(range(len(split.items))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        path, label = self.split.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, idx


# --------------------------------------------------------------------------- helpers
def _clean(name: str) -> str:
    return name.replace("_", " ").strip()


def _list_images(d: str) -> List[str]:
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def _imagenet_wnid_to_name(root: str) -> Tuple[List[str], dict]:
    path = os.path.join(root, "imagenet", "classnames.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Use CoOp's imagenet/classnames.txt ('<wnid> <name>' per line)."
        )
    wnids, names = [], {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wnid, name = line.split(" ", 1)
            wnids.append(wnid)
            names[wnid] = name
    return wnids, names


def _wnid_folder_split(name: str, img_root: str, root: str, restrict_to_present: bool) -> Split:
    all_wnids, wnid2name = _imagenet_wnid_to_name(root)
    present = sorted(d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d)))
    wnids = present if restrict_to_present else all_wnids  # A / R use their 200-class subset
    w2i = {w: i for i, w in enumerate(wnids)}
    items = []
    for w in present:
        if w not in w2i:
            continue
        for p in _list_images(os.path.join(img_root, w)):
            items.append((p, w2i[w]))
    return Split(name, items, [wnid2name[w] for w in wnids], {"wnids": wnids})


def _coop_json_split(name: str, root: str) -> Split:
    sub, img_dir, split_file = COOP_JSON[name]
    base = os.path.join(root, sub)
    with open(os.path.join(base, split_file)) as f:
        test = json.load(f)["test"]
    label2name = {}
    items = []
    for impath, label, cname in test:
        items.append((os.path.join(base, img_dir, impath), int(label)))
        label2name[int(label)] = cname
    names = [label2name[i] for i in range(len(label2name))]
    if name == "eurosat":
        names = [EUROSAT_CNAMES.get(n, n) for n in names]
    return Split(name, items, [_clean(n) for n in names])


def _aircraft_split(root: str) -> Split:
    base = os.path.join(root, "fgvc_aircraft", "data")
    with open(os.path.join(base, "variants.txt")) as f:
        names = [l.strip() for l in f if l.strip()]
    n2i = {n: i for i, n in enumerate(names)}
    items = []
    with open(os.path.join(base, "images_variant_test.txt")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            im, var = line.split(" ", 1)
            items.append((os.path.join(base, "images", im + ".jpg"), n2i[var]))
    return Split("fgvc_aircraft", items, names)


# --------------------------------------------------------------------------- public API
def build_split(name: str, root: str) -> Split:
    if name == "imagenet":
        return _wnid_folder_split(name, os.path.join(root, "imagenet", "images", "val"), root, False)
    if name == "imagenet_sketch":
        return _wnid_folder_split(name, os.path.join(root, "imagenet-sketch", "images"), root, False)
    if name == "imagenet_a":
        return _wnid_folder_split(name, os.path.join(root, "imagenet-adversarial", "imagenet-a"), root, True)
    if name == "imagenet_r":
        return _wnid_folder_split(name, os.path.join(root, "imagenet-rendition", "imagenet-r"), root, True)
    if name == "imagenet_v2":
        wnids, w2n = _imagenet_wnid_to_name(root)
        d = os.path.join(root, "imagenetv2", "imagenetv2-matched-frequency-format-val")
        items = []
        for k in range(len(wnids)):
            sub = os.path.join(d, str(k))
            if os.path.isdir(sub):
                items += [(p, k) for p in _list_images(sub)]
        return Split(name, items, [w2n[w] for w in wnids], {"wnids": wnids})
    if name.startswith("imagenet_c"):
        # imagenet_c/<corruption>/<severity>
        _, corr, sev = name.split("/")
        d = os.path.join(root, "imagenet-c", corr, str(sev))
        s = _wnid_folder_split(name, d, root, False)
        s.meta.update(corruption=corr, severity=int(sev))
        return s
    if name in COOP_JSON:
        return _coop_json_split(name, root)
    if name == "fgvc_aircraft":
        return _aircraft_split(root)
    raise KeyError(f"unknown dataset {name}")


def expand_names(names: List[str]) -> List[str]:
    """Shortcuts: 'swapprompt' -> the 14 datasets; 'imagenet_c:sev=5' -> 15 corruptions at sev 5;
    'imagenet_c:corr=gaussian_noise' -> severities 1..5 of one corruption; 'imagenet_c:all' -> 75 splits."""
    out = []
    for n in names:
        if n == "swapprompt":
            out += SWAPPROMPT_DATASETS
        elif n == "tpt_cross":  # TPT cross-dataset = SwapPrompt's 9 + aircraft
            out += [d for d in SWAPPROMPT_DATASETS if d not in IMAGENET_FAMILY] + ["fgvc_aircraft"]
        elif n.startswith("imagenet_c:"):
            arg = n.split(":", 1)[1]
            if arg == "all":
                out += [f"imagenet_c/{c}/{s}" for c in IMAGENET_C_CORRUPTIONS for s in range(1, 6)]
            elif arg.startswith("sev="):
                s = int(arg[4:])
                out += [f"imagenet_c/{c}/{s}" for c in IMAGENET_C_CORRUPTIONS]
            elif arg.startswith("corr="):
                c = arg[5:]
                out += [f"imagenet_c/{c}/{s}" for s in range(1, 6)]
        else:
            out.append(n)
    return out


def subsample(split: Split, n: Optional[int], seed: int = 0) -> List[int]:
    """Class-stratified random subset (keeps label distribution) — returns indices."""
    return subsample_labels([y for _, y in split.items], n, seed)


def subsample_labels(labels, n: Optional[int], seed: int = 0) -> List[int]:
    labels = [int(y) for y in labels]
    idx = list(range(len(labels)))
    if n is None or n >= len(idx):
        return idx
    rng = random.Random(seed)
    by_c = {}
    for i, y in enumerate(labels):
        by_c.setdefault(y, []).append(i)
    for v in by_c.values():
        rng.shuffle(v)
    frac = n / len(idx)
    chosen = []
    for y, v in sorted(by_c.items()):
        chosen += v[: max(1, round(len(v) * frac))]
    if len(chosen) < n:  # rounding shortfall -> top up randomly
        rest = sorted(set(idx) - set(chosen))
        rng.shuffle(rest)
        chosen += rest[: n - len(chosen)]
    rng.shuffle(chosen)
    return sorted(chosen[:n])


def safe_name(name: str) -> str:
    return name.replace("/", "-")
