"""TTA 모델과 zero-shot 모델에 같은 스트림을 넣고 샘플마다 기록한다.

예: python run.py --model ViT-B-16 --method sar
결과: outputs/{model}_{method}_{stream}.npz (샘플 단위 기록, 구간 집계는 analyze.py)
"""
import argparse
import copy
import json
import os
import tarfile
import time
import urllib.request

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES, SIMPLE_IMAGENET_TEMPLATES
from PIL import Image
from torchvision import transforms as T

from corruptions import DOMAINS, corrupt
from methods import METHODS, CLIPClassifier, entropy

V2_URL = "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz"
V2_DRIVE_ID = "1HsymsaWYakVTbWaBzXSsGVk9Wf6j682X"
V2_DIR ="imagenetv2-matched-frequency-format-val"
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def load_v2(root):
    """V2 1만 장을 224x224 uint8로 한 번만 디코딩해서 캐시한다 (약 1.5GB)."""
    cache_x, cache_y = os.path.join(root, "v2_224_x.npy"), os.path.join(root, "v2_224_y.npy")
    if os.path.exists(cache_x):
        return np.load(cache_x, mmap_mode="r"), np.load(cache_y)
    d = os.path.join(root, V2_DIR)
    if not os.path.isdir(d):
        os.makedirs(root, exist_ok=True)
        tgz = os.path.join(root, "v2.tar.gz")
        print("ImageNet-V2 다운로드 중 (1.2GB)")
        try:  # 팀 드라이브 데이터셋 폴더에 있는 같은 파일
            import gdown
            gdown.download(id=V2_DRIVE_ID, output=tgz, quiet=False)
        except Exception as e:
            print("드라이브 다운로드 실패, HuggingFace에서 받음:", e)
            urllib.request.urlretrieve(V2_URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(root)
        os.remove(tgz)
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224)])
    xs, ys = [], []
    for c in range(1000):
        cd = os.path.join(d, str(c))
        for f in sorted(os.listdir(cd)):
            im = tf(Image.open(os.path.join(cd, f)).convert("RGB"))
            xs.append(np.asarray(im).transpose(2, 0, 1))
            ys.append(c)
        if c % 200 == 0:
            print(f"  디코딩 {c}/1000")
    x, y = np.stack(xs), np.array(ys)
    np.save(cache_x, x)
    np.save(cache_y, y)
    return x, y


def dirichlet_order(idx, y, alpha, n_slots, rng):
    """RoTTA/ROID 방식의 클래스 쏠림 스트림. alpha가 작을수록 같은 클래스가 몰려 온다."""
    slots = [[] for _ in range(n_slots)]
    for c in np.unique(y):
        ids = rng.permutation(idx[y == c])
        cuts = (np.cumsum(rng.dirichlet([alpha] * n_slots)) * len(ids)).astype(int)[:-1]
        for s, part in enumerate(np.split(ids, cuts)):
            slots[s].extend(part.tolist())
    return np.array([i for s in slots for i in rng.permutation(s)], dtype=int)


@torch.no_grad()
def text_features(model, name, device, cache_dir):
    path = os.path.join(cache_dir, f"text_{name}.pt")
    if os.path.exists(path):
        return torch.load(path).to(device)
    tok = open_clip.get_tokenizer(name)
    feats = []
    for c in IMAGENET_CLASSNAMES:
        t = tok([tpl(c) for tpl in SIMPLE_IMAGENET_TEMPLATES]).to(device)
        f = F.normalize(model.encode_text(t).float(), dim=-1).mean(0)
        feats.append(F.normalize(f, dim=-1))
    feats = torch.stack(feats)
    torch.save(feats.cpu(), path)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B-16")
    ap.add_argument("--method", default="sar", choices=list(METHODS))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--n_per_domain", type=int, default=10000)
    ap.add_argument("--stream", default="dirichlet", choices=["iid", "dirichlet"])
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--continual", action="store_true", help="도메인이 바뀌어도 모델을 초기화하지 않음")
    ap.add_argument("--grad_ckpt", action="store_true", help="메모리가 모자랄 때 (ViT-L)")
    ap.add_argument("--limit_batches", type=int, default=0, help="도메인당 batch 수 제한 (스모크 테스트)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    X, Y = load_v2(args.data_root)

    clip, _, _ = open_clip.create_model_and_transforms(args.model, pretrained="openai")
    clip = clip.to(dev).eval()
    text = text_features(clip, args.model, dev, args.out)
    if args.grad_ckpt:
        clip.set_grad_checkpointing(True)
    zs_net = CLIPClassifier(copy.deepcopy(clip), text).half().eval().requires_grad_(False)
    net = CLIPClassifier(clip, text).eval()
    del clip.transformer  # 텍스트 인코더는 더 안 씀
    method = METHODS[args.method](net, args.lr, 1000)

    mean, std = MEAN.to(dev), STD.to(dev)
    rng = np.random.default_rng(args.seed)
    rec = {k: [] for k in ["y", "dom", "pred_t", "pred_z", "ent_t", "ent_z", "maxp_t", "maxp_z",
                           "kl_tz", "pz_at_t", "pt_at_z"]}
    domains = args.domains.split(",")
    t0 = time.time()
    for di, dom in enumerate(domains):
        idx = rng.choice(len(Y), min(args.n_per_domain, len(Y)), replace=False)
        if args.stream == "dirichlet":
            idx = dirichlet_order(idx, Y[idx], args.alpha, max(1, len(idx) // args.bs), rng)
        if di > 0 and not args.continual:
            method.reset()
        nb = (len(idx) + args.bs - 1) // args.bs
        if args.limit_batches:
            nb = min(nb, args.limit_batches)
        correct_t = correct_z = n = 0
        for b in range(nb):
            sel = idx[b * args.bs:(b + 1) * args.bs]
            x = torch.from_numpy(np.ascontiguousarray(X[sel])).to(dev).float() / 255
            x = corrupt(x, dom, args.severity, seed=args.seed * 100003 + di * 1009 + b)
            x = (x - mean) / std
            y = torch.from_numpy(Y[sel]).to(dev)

            with torch.no_grad():
                lz = zs_net(x.half()).float()
            lt = method.step(x).float()

            with torch.no_grad():
                pt, pz = lt.softmax(1), lz.softmax(1)
                at, az = pt.argmax(1), pz.argmax(1)
                kl = (pt * (pt.clamp_min(1e-12).log() - pz.clamp_min(1e-12).log())).sum(1)
                vals = dict(y=y, dom=torch.full_like(y, di), pred_t=at, pred_z=az,
                            ent_t=entropy(lt), ent_z=entropy(lz), maxp_t=pt.max(1).values,
                            maxp_z=pz.max(1).values, kl_tz=kl,
                            pz_at_t=pz.gather(1, at[:, None])[:, 0], pt_at_z=pt.gather(1, az[:, None])[:, 0])
                for k, v in vals.items():
                    rec[k].append(v.cpu().numpy())
                correct_t += (at == y).sum().item()
                correct_z += (az == y).sum().item()
                n += len(y)
        print(f"[{dom:15s}] TTA {100 * correct_t / n:5.1f}  zero-shot {100 * correct_z / n:5.1f}  "
              f"차이 {100 * (correct_t - correct_z) / n:+5.1f}  ({time.time() - t0:.0f}s)", flush=True)

    name = f"{args.model}_{args.method}_{args.stream}{'_continual' if args.continual else ''}{args.tag}"
    out = {k: np.concatenate(v) for k, v in rec.items()}
    meta = dict(vars(args), domains=domains, n_resets=getattr(method, "n_resets", 0))
    np.savez_compressed(os.path.join(args.out, name + ".npz"), meta=json.dumps(meta), **out)
    print("저장:", os.path.join(args.out, name + ".npz"), "| SAR 복구 횟수:", meta["n_resets"])


if __name__ == "__main__":
    main()
