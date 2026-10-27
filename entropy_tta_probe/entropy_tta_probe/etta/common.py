from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .augment import base_preprocess
from .data import ImageListDataset, build_split, safe_name, subsample
from .model import CLIPBundle


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def base_args(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--data_root", required=True)
    p.add_argument("--datasets", nargs="+", default=["swapprompt"],
                   help="names or shortcuts: swapprompt | tpt_cross | imagenet_c:sev=5 | imagenet_c:corr=<c> | imagenet_c:all")
    p.add_argument("--arch", default="RN50", help="RN50 (SwapPrompt) or ViT-B-16 (TPT)")
    p.add_argument("--pretrained", default="openai", help="'none' = random weights (smoke test only)")
    p.add_argument("--out", default="results")
    p.add_argument("--cache", default="cache")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def load_bundle(args) -> CLIPBundle:
    return CLIPBundle(args.arch, args.pretrained, args.device)


def feature_cache_path(args, name: str) -> str:
    return os.path.join(args.cache, f"{args.arch}_{args.pretrained}", f"{safe_name(name)}.pt")


def get_features(args, bundle: CLIPBundle, name: str, max_samples=None) -> dict:
    """Center-crop image features for a whole split (cached). Returns dict(feats, labels, classnames)."""
    path = feature_cache_path(args, name)
    if os.path.isfile(path):
        d = torch.load(path)
    else:
        split = build_split(name, args.data_root)
        ds = ImageListDataset(split, base_preprocess(bundle.input_size))
        dl = DataLoader(ds, batch_size=256, num_workers=args.workers, shuffle=False)
        feats, labels = [], []
        for x, y, _ in tqdm(dl, desc=f"features {name}"):
            feats.append(bundle.encode_images(x).cpu().half())
            labels.append(y)
        d = dict(feats=torch.cat(feats), labels=torch.cat(labels), classnames=split.classnames, name=name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(d, path)
    if max_samples is not None and max_samples < len(d["labels"]):
        g = torch.Generator().manual_seed(args.seed)
        keep = torch.randperm(len(d["labels"]), generator=g)[:max_samples].sort().values
        d = dict(d, feats=d["feats"][keep], labels=d["labels"][keep])
    return d


def out_dir(args, exp: str) -> str:
    d = os.path.join(args.out, exp, f"{args.arch}_{args.pretrained}")
    os.makedirs(d, exist_ok=True)
    return d


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


__all__ = ["seed_all", "base_args", "load_bundle", "get_features", "out_dir", "dump_json",
           "build_split", "subsample", "safe_name"]
