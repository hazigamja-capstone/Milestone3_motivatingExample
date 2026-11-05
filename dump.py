"""1단계: 열(실패 모드)마다 샘플 x 뷰 단위 raw 값을 저장한다. 라벨은 저장만 하고 신호 계산에는 쓰지 않는다.

python dump.py --columns clean aug_noise adversarial corruption natural_hard ood --n 500
python dump.py --columns paper_a paper_v2 --n 2000   # 논문 재현용 (원래 클래스 공간, PLPD 생략)
"""
import argparse
import os

import numpy as np
import torch
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import LOGIT_SCALE, encode_image, imagenet_a_classes, load_clip, text_features
from data import COLUMNS, PAPER_COLUMNS, BaseImages, corrupt, list_items, make_views, patch_shuffle, pgd, sample_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns", nargs="+", default=list(COLUMNS), choices=list(COLUMNS) + list(PAPER_COLUMNS))
    ap.add_argument("--n", type=int, default=500, help="열마다 샘플 수 (V2 200클래스는 최대 2000)")
    ap.add_argument("--views", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/dump")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda"
    os.makedirs(args.out, exist_ok=True)
    model, tok = load_clip(device)
    wnids, class_idx, names = imagenet_a_classes()
    text_a200 = text_features(model, tok, names, device)

    for col in args.columns:
        cfg = COLUMNS.get(col) or PAPER_COLUMNS[col]
        in1k = cfg.get("classes") == "in1k"
        text = text_features(model, tok, IMAGENET_CLASSNAMES, device) if in1k else text_a200
        with_plpd = col in COLUMNS
        items = sample_items(list_items(cfg["source"], class_idx, wnids), args.n, args.seed)
        loader = DataLoader(BaseImages(items), batch_size=1, num_workers=args.workers)
        N, V, C = len(items), args.views, text.shape[0]
        logits = np.zeros((N, V, C), np.float16)
        feats = np.zeros((N, V, text.shape[1]), np.float16)
        areas = np.zeros((N, V), np.float32)
        plpd = np.zeros((N, V), np.float32)
        labels = np.array([y for _, y in items], np.int64)

        for n, (base, y) in enumerate(tqdm(loader, desc=col)):
            gen = torch.Generator().manual_seed(args.seed * 100003 + n)
            if "corrupt" in cfg:
                base = corrupt(base, cfg["corrupt"], gen)
            base = base.to(device)
            if cfg.get("attack"):
                base = pgd(model, base, y.to(device), text)
            views, area = make_views(base[0].cpu(), V, cfg["scale"], gen)
            views = views.to(device)
            with torch.no_grad():
                f = encode_image(model, views)
                lg = LOGIT_SCALE * f @ text.T
                p = lg.softmax(-1)
                if with_plpd:
                    yhat = p.argmax(-1)
                    p_sh = (LOGIT_SCALE * encode_image(model, patch_shuffle(views, gen=gen)) @ text.T).softmax(-1)
                    plpd[n] = (p.gather(1, yhat[:, None]) - p_sh.gather(1, yhat[:, None]))[:, 0].cpu().numpy()
            logits[n], feats[n] = lg.cpu().numpy(), f.cpu().numpy()
            areas[n] = area.numpy()

        np.savez(os.path.join(args.out, f"{col}.npz"), logits=logits, feats=feats,
                 areas=areas, plpd=plpd, labels=labels, text=text.cpu().numpy())
        acc0 = (logits[:, 0].argmax(-1) == labels).mean() if labels.min() >= 0 else float("nan")
        print(f"[{col}] N={N} zero-shot acc(view0)={acc0:.4f}")


if __name__ == "__main__":
    main()
