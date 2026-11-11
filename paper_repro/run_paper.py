"""A Lost Opportunity (Döbler et al., ECCVW 2024) 공개 코드를 그대로 쓰고, 데이터 입력과 기록만 바꿔 돌린다.

- TTA 방법, 모델 로드, 설정(cfg)은 third_party/test-time-adaptation의 코드를 수정 없이 불러온다.
- 바꾸는 것: (1) ImageNet-C를 디스크 대신 HuggingFace에서 스트리밍 (2) 같은 이미지를 zero-shot 모델에도 넣음
  (3) 이미지마다 기록 (run.py와 같은 필드) (4) 손상 하나가 끝날 때마다 상태를 저장해 끊겨도 이어서 돌림
- 논문 코드 전용 환경(third_party/.venv_paper: torch 2.2.1, open_clip 2.24.0)에서 실행한다.

예 (저장소 루트에서):
  third_party/.venv_paper/Scripts/python.exe paper_repro/run_paper.py --method eta --n_domains 3 --grad_ckpt
v3 역할: paper(기준 재현), calibration/confirmation(클래스 분리·순서 무작위화).
주 실행은 paper_repro/run_plan.py를 사용한다. --data_revision으로 HF SHA를 고정한다.
결과: outputs_paper/{method}_{arch}_{role}_s{seed}/dom{i}_{손상}.npz, state.pt, meta.json
"""
from contextlib import closing
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
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

def source_split(corruption, severity, limit_classes=0):
    """손상 하나의 5만 장을 라벨 순으로 내보낸다 (논문의 correlated: 섞은 뒤 라벨로 정렬 = 클래스 안 순서는 무작위).

    HF 샤드는 이미 클래스 순이라, 클래스 하나(50장)를 모아 섞은 뒤 내보낸다. 반환: (클래스, webp 바이트)
    """
    split = f"{corruption}_{severity}"
    k = CORRUPTIONS.index(corruption) * 5 + (severity - 1)
    first, last = k * PER_SPLIT, (k + 1) * PER_SPLIT - 1
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
                        yield from buf
                        buf = []
                    cur, seen = c, idx
                    buf.append((c, rec["0.webp"], idx - first))
                break
            except Exception as e:  # noqa: BLE001 (네트워크/tar 오류)
                if attempt == 4:
                    raise
                print(f"  샤드 {sh} 오류 ({e}), 재시도 {attempt + 1}/5", flush=True)
                time.sleep(5)
        if stop:
            break
    if buf:
        yield from buf


from paper_repro.protocol import VERSION, ROLES, role_classes, domain_order, reorder_stream


def iter_split(corruption, severity, seed, limit_classes=0, role="paper"):
    source = source_split(corruption, severity, limit_classes)
    cache_root = os.environ.get("TTA_CACHE_DIR")
    if cache_root:
        from paper_repro.stream_cache import cached_ordered_source
        import prepare_inc
        yield from cached_ordered_source(lambda: source_split(corruption, severity, limit_classes), cache_root,
                               dict(url=prepare_inc.REPO, corruption=corruption, severity=severity),
                               role=role, seed=seed, domain_index=CORRUPTIONS.index(corruption) * 5 + severity - 1,
                               n_classes=limit_classes or 1000)
        return
    yield from reorder_stream(source, role, seed, CORRUPTIONS.index(corruption) * 5 + severity - 1,
                              limit_classes or 1000)


def batches(corruption, severity, seed, preprocess, bs, limit_classes=0, prefetch=8, role="paper"):
    """Ordered deterministic CLIP eval preprocessing with bounded parallel decode."""
    from paper_repro.performance import deterministic_clip_transform, prefetched_batches
    if not deterministic_clip_transform(preprocess):
        raise ValueError("Parallel preprocessing requires the deterministic CLIP eval transform")
    records = iter_split(corruption, severity, seed, limit_classes, role)
    workers = min(4, os.cpu_count() or 1)
    yield from prefetched_batches(records, preprocess, bs, prefetch, workers,
                                  pin_memory=torch.cuda.is_available())


# ---------------------------------------------------------------- 상태 저장 / 복구

from paper_repro.checkpoint import save_state, load_state
from paper_repro.compatibility import execution_sources
from paper_repro.run_identity import prepare_run, mark_complete, file_hash


# ---------------------------------------------------------------- 실행

def entropy(logits):
    logp = logits.float().log_softmax(1)
    return -(logp.exp() * logp).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="eta", choices=list(CFG_OF))
    ap.add_argument("--arch", default="ViT-L-14")
    # All roles remain class-block-correlated; v3 roles randomize the block/domain order.
    ap.add_argument("--setting", default="correlated", choices=["correlated"],
                    help="클래스 블록 correlated 스트림; 역할별 순서는 PREREG v3 참조")
    ap.add_argument("--role", choices=ROLES, default="paper",
                    help="paper: reference only; calibration/confirmation: revised class-disjoint shuffled streams")
    ap.add_argument("--data_revision", required=True, help="Full immutable HF dataset commit SHA")
    ap.add_argument("--seed", type=int, default=1, help="논문 기본값 1")
    ap.add_argument("--n_domains", type=int, default=15, help="논문 순서의 앞쪽 몇 개 손상까지")
    ap.add_argument("--grad_ckpt", action="store_true", help="GPU 메모리 절약; 수치 동등성은 검증 필요")
    ap.add_argument("--limit_classes", type=int, default=0, help="스모크 테스트용")
    ap.add_argument("--limit_batches", type=int, default=0, help="스모크 테스트용")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs_paper"))
    ap.add_argument("--checkpoint_batches", type=int, default=100,
                    help="Save in-progress domain every N batches; 0 disables periodic saves")
    ap.add_argument("--opts", nargs="*", default=[], help="논문 cfg에 그대로 넘길 추가 옵션")
    args = ap.parse_args()

    run_dir = os.path.join(os.path.abspath(args.out), f"{args.method}_{args.arch}_{args.role}_s{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    if args.n_domains < 1 or not 0 <= args.limit_classes <= 1000 or args.limit_batches < 0 or args.checkpoint_batches < 0:
        ap.error("domain count must be positive and limits nonnegative")

    if len(args.data_revision) != 40 or any(c not in "0123456789abcdef" for c in args.data_revision):
        ap.error("data_revision must be a full lowercase 40-character commit SHA")
    import prepare_inc
    prepare_inc.REPO = prepare_inc.REPO.replace("/resolve/main/", f"/resolve/{args.data_revision}/")
    prepare_inc._start_cache.clear()
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
    if args.n_domains > len(cfg.CORRUPTION.TYPE):
        ap.error("n_domains exceeds configured domain count")
    expected_method = {"eta": "eata", "sar": "sar", "tent": "tent"}[args.method]
    if (cfg.MODEL.ADAPTATION != expected_method or cfg.MODEL.ARCH != args.arch
            or cfg.SETTING != args.setting or cfg.RNG_SEED != args.seed or not cfg.MODEL.USE_CLIP):
        ap.error("--opts cannot change the method/model/setting/seed declared in the run name")
    domains = domain_order(list(cfg.CORRUPTION.TYPE)[:args.n_domains], args.role, args.seed)
    severity = cfg.CORRUPTION.SEVERITY[0]
    dev = "cuda"

    base_model, preprocess = get_model(cfg, 1000, dev)
    base_model.model_preprocess = preprocess
    model_module = sys.modules.get("models.model")
    from paper_repro.performance import reuse_fixed_text
    with reuse_fixed_text(model_module, base_model, cfg):
        zs_model, _ = get_model(cfg, 1000, dev)  # independent visual model; identical frozen text
    zs_model.eval().requires_grad_(False)
    if args.grad_ckpt:
        base_model.model.visual.set_grad_checkpointing(True)
    tta = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=base_model, num_classes=1000)

    meta = dict(vars(args), protocol_version=VERSION,
                classes=role_classes(args.role, args.limit_classes or 1000), domains=domains, severity=severity, bs=cfg.TEST.BATCH_SIZE,
                lr=tta.optimizer.param_groups[0]["lr"], cfg_lr=cfg.OPTIM.LR,
                optimizer_lrs=[group["lr"] for group in tta.optimizer.param_groups],
                paper_commit=subprocess.run(["git", "-C", PAPER, "rev-parse", "HEAD"], capture_output=True,
                                            text=True).stdout.strip(),
                torch=torch.__version__, cfg=cfg.dump())
    meta["cuda_runtime"] = torch.version.cuda
    meta["cudnn"] = torch.backends.cudnn.version()
    meta["gpu"] = (dict(name=torch.cuda.get_device_name(),
                        total_memory=torch.cuda.get_device_properties(0).total_memory)
                   if torch.cuda.is_available() else None)
    stable_cfg = cfg.clone()
    stable_cfg.defrost()
    for key in ("SAVE_DIR", "LOG_TIME", "LOG_DEST"):
        if key in stable_cfg:
            del stable_cfg[key]
    meta["identity"] = dict(n_domains=args.n_domains, method=args.method, arch=args.arch, setting=args.setting, seed=args.seed,
                            limit_classes=args.limit_classes, limit_batches=args.limit_batches,
                            checkpoint_batches=args.checkpoint_batches,
                            grad_ckpt=args.grad_ckpt, cfg=stable_cfg.dump(),
                            paper_commit=meta["paper_commit"], torch=meta["torch"],
                            runner_version=VERSION, role=args.role, data_revision=args.data_revision,
                            classes=role_classes(args.role, args.limit_classes or 1000),
                            cli_opts=args.opts, source_sha256=execution_sources(ROOT, args.out))
    prepare_run(run_dir, meta)

    state_path = os.path.join(run_dir, "state.pt")
    done, progress = 0, None
    if os.path.exists(state_path):
        done, progress = load_state(state_path, tta, meta["identity"], with_progress=True)
        print(f"이어서 돌림: 손상 {done}개 끝난 상태에서 시작", flush=True)
    else:
        tta.reset()  # 논문: 첫 도메인에서 한 번만 초기화 (continual / correlated)

    for i in range(done):
        if i >= len(domains) or not os.path.exists(os.path.join(run_dir, f"dom{i:02d}_{domains[i]}.npz")):
            raise ValueError("Checkpoint references missing domain results")
    t0 = time.time()
    for di in range(done, len(domains)):
        dom = domains[di]
        rec = {k: [] for k in ["y", "pred_t", "pred_z", "ent_t", "ent_z", "maxp_t", "maxp_z",
                               "kl_tz", "pz_at_t", "pt_at_z", "rank_z_at_t", "sample_id"]}
        ct = cz = n = 0
        resume_batches = 0
        partial_path = None
        if progress is not None:
            if progress["domain"] != di:
                raise ValueError("Partial checkpoint domain does not match stream")
            partial_path = Path(run_dir) / progress["file"]
            if partial_path.name != progress["file"]:
                raise ValueError("Invalid partial result path")
            if file_hash(partial_path) != progress["sha256"]:
                raise ValueError("Partial results do not match checkpoint hash")
            with np.load(partial_path, allow_pickle=False) as partial:
                rec = {k: [partial[k]] for k in rec}
            resume_batches = progress["batches"]
            n = len(rec["y"][0])
            if n != resume_batches * cfg.TEST.BATCH_SIZE:
                raise ValueError("Partial results do not match checkpoint batch count")
            ct = int((rec["pred_t"][0] == rec["y"][0]).sum())
            cz = int((rec["pred_z"][0] == rec["y"][0]).sum())
            print(f"재개: {dom}, {resume_batches}배치, {n}장", flush=True)
        with torch.no_grad(), closing(batches(dom, severity, args.seed, preprocess, cfg.TEST.BATCH_SIZE,
                                               args.limit_classes, role=args.role)) as stream:
            for b, (x, y, sample_ids) in enumerate(stream):
                if b < resume_batches:
                    continue
                if args.limit_batches and b >= args.limit_batches:
                    break
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):  # TTA 쪽과 같은 혼합 정밀도
                    lz = zs_model(x).float()
                lt = tta(x).float()
                pt, pz = lt.softmax(1), lz.softmax(1)
                at, az = pt.argmax(1), pz.argmax(1)
                pz_t = pz.gather(1, at[:, None])
                vals = dict(y=y, sample_id=sample_ids, pred_t=at, pred_z=az, ent_t=entropy(lt), ent_z=entropy(lz),
                            maxp_t=pt.max(1).values, maxp_z=pz.max(1).values,
                            kl_tz=(pt * (pt.clamp_min(1e-12).log() - pz.clamp_min(1e-12).log())).sum(1),
                            pz_at_t=pz_t[:, 0], pt_at_z=pt.gather(1, az[:, None])[:, 0],
                            rank_z_at_t=(pz > pz_t).sum(1).to(torch.int16))
                for k, v in vals.items():
                    rec[k].append(v.cpu().numpy())
                ct += (at == y).sum().item()
                cz += (az == y).sum().item()
                n += len(y)
                if (args.checkpoint_batches and len(y) == cfg.TEST.BATCH_SIZE
                        and (b + 1) % args.checkpoint_batches == 0):
                    name = f"partial_dom{di:02d}_b{b + 1:06d}.npz"
                    target = Path(run_dir) / name
                    with open(str(target) + ".tmp", "wb") as f:
                        np.savez_compressed(f, **{k: np.concatenate(v) for k, v in rec.items()})
                    os.replace(str(target) + ".tmp", target)
                    save_state(state_path, tta, di, meta["identity"],
                               progress=dict(domain=di, batches=b + 1, file=name, sha256=file_hash(target)))
                    if partial_path is not None and partial_path != target:
                        partial_path.unlink(missing_ok=True)
                    partial_path = target
                if (b + 1) % 100 == 0:
                    print(f"  [{dom}] 배치 {b + 1}: TTA {100 * ct / n:.1f} zs {100 * cz / n:.1f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
        out = {k: np.concatenate(v) for k, v in rec.items()}
        if not args.limit_batches:
            counts = np.bincount(out["y"], minlength=1000)
            expected = np.zeros(1000, dtype=int)
            expected[meta["classes"]] = 50
            if not np.array_equal(counts, expected):
                raise ValueError("Incomplete domain: expected exactly 50 images per selected class")
        result_path = os.path.join(run_dir, f"dom{di:02d}_{dom}.npz")
        with open(result_path + ".tmp", "wb") as f:
            np.savez_compressed(f, **out)
        os.replace(result_path + ".tmp", result_path)
        save_state(state_path, tta, di + 1, meta["identity"])
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)
        progress = None
        print(f"[{di + 1}/{len(domains)} {dom}] TTA {100 * ct / n:.1f}  zero-shot {100 * cz / n:.1f}  "
              f"오류율 TTA {100 - 100 * ct / n:.1f} / zs {100 - 100 * cz / n:.1f}  n={n}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    if not args.limit_classes and not args.limit_batches:
        mark_complete(run_dir, meta["identity"], domains)


if __name__ == "__main__":
    main()
