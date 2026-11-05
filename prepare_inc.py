"""공식 ImageNet-C에서 손상 x 강도마다 클래스당 n장을 뽑아 224px uint8 캐시로 저장한다.

출처: huggingface.co/datasets/haideraltahan/wds_imagenetc (공식 ImageNet-C를 webp로 담은 webdataset, 토큰 불필요)
- 샤드(test/0.tar ~ test/540.tar)는 손상 이름 알파벳 순 x 강도 1~5 순, 그 안은 클래스 순으로 정렬돼 있다
- 그래서 필요한 손상의 샤드만 찾아 스트리밍으로 읽고, 원본 샤드는 디스크에 저장하지 않는다
- 클래스 번호는 open_clip IMAGENET_CLASSNAMES 순서와 같다 (tench, goldfish, ...)

예: python prepare_inc.py --corruptions gaussian_noise,shot_noise --severity 5 --per_class 10
"""
import argparse
import io
import os
import tarfile
import time

import urllib.request

import numpy as np
from PIL import Image

REPO = "https://huggingface.co/datasets/haideraltahan/wds_imagenetc/resolve/main/test/{}.tar"
N_SHARDS = 541
PER_SPLIT = 50000
CORRUPTIONS = sorted([
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
    "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
    "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur",
])


def _headers():
    tok = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


_start_cache = {}


def shard_start(i):
    """샤드 i의 첫 샘플 번호 (tar 첫 헤더의 파일 이름 s0001234.0.webp에서 읽음)."""
    if i >= N_SHARDS:
        return 10 ** 12
    if i not in _start_cache:
        # 일부 샤드는 PAX 헤더로 시작해서 앞부분을 넉넉히 읽고 tarfile로 첫 멤버 이름만 꺼낸다
        req = urllib.request.Request(REPO.format(i), headers={**_headers(), "Range": "bytes=0-8191"})
        with urllib.request.urlopen(req, timeout=60) as r:
            head = r.read(8192)
        with tarfile.open(fileobj=io.BytesIO(head), mode="r|") as tf:
            name = tf.next().name
        _start_cache[i] = int(name.split(".")[0][1:])
    return _start_cache[i]


def find_shard(sample):
    """sample 번호를 담고 있는 샤드 (샤드 시작 번호가 단조 증가하므로 이분 탐색)."""
    lo, hi = 0, N_SHARDS - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if shard_start(mid) <= sample:
            lo = mid
        else:
            hi = mid - 1
    return lo


def stream_shard(i):
    req = urllib.request.Request(REPO.format(i), headers=_headers())
    with urllib.request.urlopen(req, timeout=120) as r:
        with tarfile.open(fileobj=r, mode="r|") as tf:
            cur, rec = None, {}
            for m in tf:
                key, ext = m.name.split(".", 1)
                if key != cur:
                    if rec:
                        yield cur, rec
                    cur, rec = key, {}
                rec[ext] = tf.extractfile(m).read()
            if rec:
                yield cur, rec


def prepare(corruption, severity, per_class, out_dir, seed=0, limit_classes=0):
    x_path = os.path.join(out_dir, f"{corruption}_{severity}_x.npy")
    y_path = os.path.join(out_dir, f"{corruption}_{severity}_y.npy")
    if os.path.exists(x_path):
        return x_path, y_path
    os.makedirs(out_dir, exist_ok=True)
    split = f"{corruption}_{severity}"
    k = CORRUPTIONS.index(corruption) * 5 + (severity - 1)
    first, last = k * PER_SPLIT, (k + 1) * PER_SPLIT - 1
    s0, s1 = find_shard(first), find_shard(last)
    n_cls = limit_classes or 1000

    # 클래스마다 50장 중 per_class장의 순번을 미리 골라 둔다 (seed 고정)
    rng = np.random.default_rng(seed)
    pick = [set(rng.choice(50, per_class, replace=False).tolist()) for _ in range(1000)]
    seen = np.zeros(1000, dtype=int)
    xs, ys = [], []
    t0 = time.time()
    print(f"[{split}] 샤드 {s0}~{s1} 스트리밍 (클래스당 {per_class}장)", flush=True)
    done = False
    for sh in range(s0, s1 + 1):
        for attempt in range(5):  # 연결이 끊기면 그 샤드만 처음부터 다시
            seen0, n0 = seen.copy(), len(ys)
            try:
                for key, rec in stream_shard(sh):
                    idx = int(key[1:])
                    if idx < first:
                        continue
                    c = int(rec["cls"].decode())
                    if idx > last or c >= n_cls:
                        done = True
                        break
                    assert rec["split.txt"].decode().strip() == split, (key, rec["split.txt"])
                    if seen[c] in pick[c]:
                        im = Image.open(io.BytesIO(rec["0.webp"])).convert("RGB")
                        assert im.size == (224, 224), im.size
                        xs.append(np.asarray(im).transpose(2, 0, 1))
                        ys.append(c)
                    seen[c] += 1
                break
            except (OSError, tarfile.TarError) as e:
                seen[:] = seen0
                del xs[n0:], ys[n0:]
                done = False
                print(f"  샤드 {sh} 실패 ({e}), 재시도 {attempt + 1}/5", flush=True)
                time.sleep(5)
        else:
            raise RuntimeError(f"샤드 {sh}를 5번 모두 받지 못함")
        print(f"  샤드 {sh} 완료, {len(ys)}장 ({time.time() - t0:.0f}s)", flush=True)
        if done:
            break
    x, y = np.stack(xs), np.array(ys)
    assert len(y) == n_cls * per_class, f"{split}: {len(y)}장만 모임"
    np.save(x_path, x)
    np.save(y_path, y)
    print(f"[{split}] 저장 {x.shape} -> {x_path}", flush=True)
    return x_path, y_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corruptions", default="gaussian_noise,shot_noise,defocus_blur,contrast,pixelate,jpeg_compression")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--per_class", type=int, default=10)
    ap.add_argument("--out", default="data/imagenet_c")
    ap.add_argument("--limit_classes", type=int, default=0, help="스모크 테스트용")
    args = ap.parse_args()
    for c in args.corruptions.split(","):
        prepare(c, args.severity, args.per_class, args.out, limit_classes=args.limit_classes)


if __name__ == "__main__":
    main()
