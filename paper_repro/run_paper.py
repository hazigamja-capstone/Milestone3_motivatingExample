"""A Lost Opportunity (Döbler et al., ECCVW 2024) 공개 코드를 그대로 쓰고, 데이터 입력과 기록만 바꿔 돌린다.

- TTA 방법, 모델 로드, 설정(cfg)은 third_party/test-time-adaptation의 코드를 수정 없이 불러온다.
- 바꾸는 것: (1) ImageNet-C를 디스크 대신 HuggingFace에서 스트리밍 (2) 같은 이미지를 zero-shot 모델에도 넣음
  (3) 이미지마다 기록 (run.py와 같은 필드) (4) 손상 하나가 끝날 때마다 상태를 저장해 끊겨도 이어서 돌림
- 논문 코드 전용 환경(third_party/.venv_paper: torch 2.2.1, open_clip 2.24.0)에서 실행한다.

예 (저장소 루트에서):
  third_party/.venv_paper/Scripts/python.exe paper_repro/run_paper.py --method eta --n_domains 3 --grad_ckpt
결과: outputs_paper/{method}_{arch}_{setting}_s{seed}/dom{i}_{손상}.npz, state.pt, meta.json
"""
import argparse
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "third_party", "test-time-adaptation", "classification")
sys.path.insert(0, ROOT)   # prepare_inc (HF 스트리밍)
sys.path.insert(0, PAPER)  # 논문 코드

from prepare_inc import CORRUPTIONS, PER_SPLIT, find_shard, stream_shard  # noqa: E402

CFG_OF = {"eta": "cfgs/imagenet_c/eta.yaml", "sar": "cfgs/imagenet_c/sar.yaml", "tent": "cfgs/imagenet_c/tent.yaml"}


# ---------------------------------------------------------------- 데이터

def iter_split(corruption, severity, seed, limit_classes=0):
    """손상 하나의 5만 장을 라벨 순으로 내보낸다 (논문의 correlated: 섞은 뒤 라벨로 정렬 = 클래스 안 순서는 무작위).

    HF 샤드는 이미 클래스 순이라, 클래스 하나(50장)를 모아 섞은 뒤 내보낸다. 반환: (클래스, webp 바이트)
    """
    split = f"{corruption}_{severity}"
    k = CORRUPTIONS.index(corruption) * 5 + (severity - 1)
    first, last = k * PER_SPLIT, (k + 1) * PER_SPLIT - 1
    rng = random.Random(seed * 1000 + k)
    buf, cur, seen, stop = [], None, -1, False   # seen: 이미 받은 마지막 샘플 번호
    for sh in range(find_shard(first), find_shard(last) + 1):
        for attempt in range(5):  # 연결이 끊기면 그 샤드를 다시 받되, 이미 받은 샘플은 건너뛴다
            try:
                for key, rec in stream_shard(sh):
                    idx = int(key[1:])
                    if idx < first or idx <= seen:
                        continue
                    c = int(rec["cls"].decode())
                    if idx > last or (limit_classes and c >= limit_classes):
                        stop = True
                        break
                    assert rec["split.txt"].decode().strip() == split, (key, rec["split.txt"])
                    if c != cur and buf:
                        rng.shuffle(buf)
                        yield from buf
                        buf = []
                    cur, seen = c, idx
                    buf.append((c, rec["0.webp"]))
                break
            except Exception as e:  # noqa: BLE001 (네트워크/tar 오류)
                if attempt == 4:
                    raise
                print(f"  샤드 {sh} 오류 ({e}), 재시도 {attempt + 1}/5", flush=True)
                time.sleep(5)
        if stop:
            break
    if buf:
        rng.shuffle(buf)
        yield from buf


def batches(corruption, severity, seed, preprocess, bs, limit_classes=0, prefetch=8):
    """배경 스레드에서 받아 디코딩하고 배치로 묶는다 (계산과 다운로드를 겹침)."""
    from PIL import Image
    import io

    q = queue.Queue(maxsize=prefetch)

    def work():
        try:
            xs, ys = [], []
            for c, b in iter_split(corruption, severity, seed, limit_classes):
                im = Image.open(io.BytesIO(b)).convert("RGB")
                xs.append(preprocess(im))
                ys.append(c)
                if len(xs) == bs:
                    q.put((torch.stack(xs), torch.tensor(ys)))
                    xs, ys = [], []
            if xs:
                q.put((torch.stack(xs), torch.tensor(ys)))
            q.put(None)
        except BaseException as e:  # noqa: BLE001
            q.put(e)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


# ---------------------------------------------------------------- 상태 저장 / 복구

STATE_KEYS = ["current_model_probs", "ema", "num_samples_update_1", "num_samples_update_2", "performed_updates"]


def save_state(path, tta, done):
    extra = {k: getattr(tta, k) for k in STATE_KEYS if hasattr(tta, k)}
    torch.save({"model": tta.model.state_dict(), "optimizer": tta.optimizer.state_dict(),
                "extra": extra, "done": done}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_state(path, tta):
    s = torch.load(path, map_location="cuda")
    tta.model.load_state_dict(s["model"])
    tta.optimizer.load_state_dict(s["optimizer"])
    for k, v in s["extra"].items():
        setattr(tta, k, v)
    return s["done"]


# ---------------------------------------------------------------- 실행

def entropy(logits):
    logp = logits.float().log_softmax(1)
    return -(logp.exp() * logp).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="eta", choices=list(CFG_OF))
    ap.add_argument("--arch", default="ViT-L-14")
    ap.add_argument("--setting", default="correlated")
    ap.add_argument("--seed", type=int, default=1, help="논문 기본값 1")
    ap.add_argument("--n_domains", type=int, default=15, help="논문 순서의 앞쪽 몇 개 손상까지")
    ap.add_argument("--grad_ckpt", action="store_true", help="6GB GPU용. 계산 결과는 같고 메모리만 줄임")
    ap.add_argument("--limit_classes", type=int, default=0, help="스모크 테스트용")
    ap.add_argument("--limit_batches", type=int, default=0, help="스모크 테스트용")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs_paper"))
    ap.add_argument("--opts", nargs="*", default=[], help="논문 cfg에 그대로 넘길 추가 옵션")
    args = ap.parse_args()

    run_dir = os.path.join(args.out, f"{args.method}_{args.arch}_{args.setting}_s{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    os.chdir(PAPER)  # 논문 코드는 상대 경로(robustbench/data 등)를 쓴다
    sys.argv = ["test_time.py", "--cfg", CFG_OF[args.method],
                "MODEL.USE_CLIP", "True", "MODEL.ARCH", args.arch, "MODEL.WEIGHTS", "openai",
                "CLIP.PROMPT_MODE", "ensemble", "SETTING", args.setting, "RNG_SEED", str(args.seed),
                "MIXED_PRECISION", "True",  # 저자 스크립트 scripts/run_clip.sh와 같음 (없으면 fp16 CLIP에서 dtype 오류)
                "SAVE_DIR", os.path.join(run_dir, "paper_log")] + args.opts
    import methods  # noqa: F401  (등록)
    from conf import cfg, load_cfg_from_args
    from models.model import get_model
    from utils.registry import ADAPTATION_REGISTRY

    load_cfg_from_args("paper repro")
    domains = list(cfg.CORRUPTION.TYPE)[:args.n_domains]
    severity = cfg.CORRUPTION.SEVERITY[0]
    dev = "cuda"

    base_model, preprocess = get_model(cfg, 1000, dev)
    base_model.model_preprocess = preprocess
    zs_model, _ = get_model(cfg, 1000, dev)  # 같은 가중치의 고정 복사본
    zs_model.eval().requires_grad_(False)
    if args.grad_ckpt:
        base_model.model.visual.set_grad_checkpointing(True)
    tta = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=base_model, num_classes=1000)

    meta = dict(vars(args), domains=domains, severity=severity, bs=cfg.TEST.BATCH_SIZE, lr=cfg.OPTIM.LR,
                paper_commit=subprocess.run(["git", "-C", PAPER, "rev-parse", "HEAD"], capture_output=True,
                                            text=True).stdout.strip(),
                torch=torch.__version__, cfg=cfg.dump())
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=str)

    state_path = os.path.join(run_dir, "state.pt")
    done = 0
    if os.path.exists(state_path):
        done = load_state(state_path, tta)
        print(f"이어서 돌림: 손상 {done}개 끝난 상태에서 시작", flush=True)
    else:
        tta.reset()  # 논문: 첫 도메인에서 한 번만 초기화 (continual / correlated)

    t0 = time.time()
    for di in range(done, len(domains)):
        dom = domains[di]
        rec = {k: [] for k in ["y", "pred_t", "pred_z", "ent_t", "ent_z", "maxp_t", "maxp_z",
                               "kl_tz", "pz_at_t", "pt_at_z", "rank_z_at_t"]}
        ct = cz = n = 0
        with torch.no_grad():  # 논문 get_accuracy와 같음 (방법 안에서 enable_grad)
            for b, (x, y) in enumerate(batches(dom, severity, args.seed, preprocess, cfg.TEST.BATCH_SIZE,
                                               args.limit_classes)):
                if args.limit_batches and b >= args.limit_batches:
                    break
                x, y = x.to(dev), y.to(dev)
                with torch.autocast("cuda", dtype=torch.float16):  # TTA 쪽과 같은 혼합 정밀도
                    lz = zs_model(x).float()
                lt = tta(x).float()
                pt, pz = lt.softmax(1), lz.softmax(1)
                at, az = pt.argmax(1), pz.argmax(1)
                pz_t = pz.gather(1, at[:, None])
                vals = dict(y=y, pred_t=at, pred_z=az, ent_t=entropy(lt), ent_z=entropy(lz),
                            maxp_t=pt.max(1).values, maxp_z=pz.max(1).values,
                            kl_tz=(pt * (pt.clamp_min(1e-12).log() - pz.clamp_min(1e-12).log())).sum(1),
                            pz_at_t=pz_t[:, 0], pt_at_z=pt.gather(1, az[:, None])[:, 0],
                            rank_z_at_t=(pz > pz_t).sum(1).to(torch.int16))
                for k, v in vals.items():
                    rec[k].append(v.cpu().numpy())
                ct += (at == y).sum().item()
                cz += (az == y).sum().item()
                n += len(y)
                if (b + 1) % 100 == 0:
                    print(f"  [{dom}] 배치 {b + 1}: TTA {100 * ct / n:.1f} zs {100 * cz / n:.1f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
        out = {k: np.concatenate(v) for k, v in rec.items()}
        np.savez_compressed(os.path.join(run_dir, f"dom{di:02d}_{dom}.npz"), **out)
        save_state(state_path, tta, di + 1)
        print(f"[{di + 1}/{len(domains)} {dom}] TTA {100 * ct / n:.1f}  zero-shot {100 * cz / n:.1f}  "
              f"오류율 TTA {100 - 100 * ct / n:.1f} / zs {100 - 100 * cz / n:.1f}  n={n}  "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
