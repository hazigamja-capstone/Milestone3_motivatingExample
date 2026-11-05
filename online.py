"""3단계 (online): TDA positive cache의 우선순위 신호를 바꿔 끼운다.

공식 TDA (kdiAAA/TDA, ImageNet-A 설정) 규칙 그대로:
  샘플 표현 = 엔트로피 상위 10% 뷰의 특징 평균과 logits 평균, pseudo-label = 그 평균 logits의 argmax
  positive cache: 클래스당 3개, 우선순위 = 선택 뷰 marginal 엔트로피(낮을수록 남음), alpha 2, beta 5
  negative cache: 정규화 엔트로피가 0.2~0.5인 샘플, 클래스당 2개, 확률 0.03~1 마스크, alpha 0.117, beta 1
게이트 실험은 positive cache 우선순위만 다른 신호로 바꾸고 random / oracle과 비교한다 ("entropy" 게이트 = 공식 TDA).
신호마다 스케일이 달라 router에서 섞을 수 있도록 stream 안 percentile rank로 바꿔 쓴다 (단조 변환이라 단일 신호 결과는 같다).

  per-mode stream : 실패 모드 하나만 흐르는 stream (모드별 최적 신호 찾기)
  mixed stream    : 모든 모드 + OOD를 섞고, 모드 정보 없이 돌린다 (실사용 상황)
  oracle_router   : 모드를 안다고 가정하고 그 모드의 최적 신호로 바꿔주는 상한

python online.py --dump runs/dump --out results --seeds 0 1 2
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
import torch

import signals as sig
from data import COLUMNS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 샘플 단위로 쓸 수 없는 신호는 뺀다: crop 면적 평균은 그냥 난수, MTA inlierness는 샘플 안 합이 1이라 평균이 상수
ONLINE_SIGNALS = [s for s in sig.SIGNALS if s not in ("crop_area", "mta_inlier")]
POS = dict(cap=3, alpha=2.0, beta=5.0)
NEG = dict(cap=2, alpha=0.117, beta=1.0, ent_lo=0.2, ent_hi=0.5, mask_lo=0.03, mask_hi=1.0)


def tda_inputs(logits, feats, top=0.1):
    """공식 get_clip_logits: 엔트로피 상위 10% 뷰의 특징 평균(재정규화 없음), logits 평균, marginal 엔트로피, 확률 평균."""
    ent = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
    k = int(logits.shape[1] * top)
    idx = ent.argsort(1)[:, :k]
    sel = logits.gather(1, idx[..., None].expand(-1, -1, logits.shape[-1]))
    feat = feats.gather(1, idx[..., None].expand(-1, -1, feats.shape[-1])).mean(1)
    lp = sel.log_softmax(-1)
    avg = lp.logsumexp(1) - np.log(k)
    loss = -(avg * avg.exp()).sum(-1)
    return feat, sel.mean(1), loss, sel.softmax(-1).mean(1)


def load_modes(dump):
    modes = {}
    for col in COLUMNS:
        path = os.path.join(dump, f"{col}.npz")
        if not os.path.exists(path):
            continue
        d = dict(np.load(path))
        t = sig.to_torch(d)
        S = sig.compute(**t)
        feat, logits, loss, prob = tda_inputs(t["logits"], t["feats"])
        # 공식 TDA 우선순위는 -loss. 다른 신호는 뷰 점수 평균. 어느 신호에도 유리하지 않게 같은 규칙.
        scores = {k: v.mean(1).numpy() for k, v in S.items() if k in ONLINE_SIGNALS}
        scores["entropy"] = -loss.numpy()
        modes[col] = dict(feat=feat.numpy(), logits=logits.numpy(), labels=d["labels"],
                          loss=loss.numpy(), prob=prob.numpy(), scores=scores)
    return modes


def rank01(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def push(bucket, item, cap):
    """item = (score, idx). 높은 점수가 남는다 (공식 TDA는 낮은 loss가 남음 = score -loss)."""
    if len(bucket) < cap:
        bucket.append(item)
    else:
        worst = min(range(cap), key=lambda j: bucket[j][0])
        if item[0] > bucket[worst][0]:
            bucket[worst] = item


def run_stream(feat, logits, labels, score, loss, prob, skip=None):
    """score: [M] 높을수록 positive cache에 남는다. skip: [M] bool, True면 positive cache에 넣지 않음 (oracle 전용).
    공식 TDA 순서: 캐시를 먼저 갱신하고 그 캐시로 현재 샘플을 예측한다."""
    M, C = logits.shape
    pos = {c: [] for c in range(C)}  # c -> [(score, idx)]
    neg = {c: [] for c in range(C)}
    norm_ent = loss / np.log2(C)  # 공식 get_entropy (nats / log2 C)
    preds = np.zeros(M, np.int64)
    contam = np.zeros(M)
    for t in range(M):
        c = int(logits[t].argmax())
        if not (skip is not None and skip[t]):
            push(pos[c], (score[t], t), POS["cap"])
        if NEG["ent_lo"] < norm_ent[t] < NEG["ent_hi"]:
            push(neg[c], (-loss[t], t), NEG["cap"])

        out = logits[t].astype(np.float64)
        keys = [i for b in pos.values() for _, i in b]
        if keys:
            onehot = np.zeros((len(keys), C))
            onehot[np.arange(len(keys)), logits[keys].argmax(-1)] = 1
            out += POS["alpha"] * np.exp(-POS["beta"] * (1 - feat[keys] @ feat[t])) @ onehot
        keys = [i for b in neg.values() for _, i in b]
        if keys:
            mask = ((prob[keys] > NEG["mask_lo"]) & (prob[keys] < NEG["mask_hi"])).astype(np.float64)
            out -= NEG["alpha"] * np.exp(-NEG["beta"] * (1 - feat[keys] @ feat[t])) @ mask
        preds[t] = out.argmax()

        n_in = sum(len(b) for b in pos.values())
        n_bad = sum(labels[i] != logits[i].argmax() for b in pos.values() for _, i in b)
        contam[t] = n_bad / max(n_in, 1)
    return preds, contam


def stream_metrics(preds, contam, labels, mode_ids, mode_names):
    m = labels >= 0
    row = dict(acc=(preds[m] == labels[m]).mean(), contamination_final=contam[-1], contamination_mean=contam.mean())
    for k, name in enumerate(mode_names):
        mk = (mode_ids == k) & m
        if mk.any():
            row[f"acc_{name}"] = (preds[mk] == labels[mk]).mean()
    accs = [v for k, v in row.items() if k.startswith("acc_")]
    row["worst_mode_acc"] = min(accs) if accs else np.nan
    return row


def build(modes, names, seed):
    cat = {k: np.concatenate([modes[n][k] for n in names]) for k in ("feat", "logits", "labels", "loss", "prob")}
    mode_ids = np.concatenate([np.full(len(modes[n]["labels"]), k) for k, n in enumerate(names)])
    raw = {s: np.concatenate([modes[n]["scores"][s] for n in names]) for s in ONLINE_SIGNALS}
    order = np.random.default_rng(seed).permutation(len(mode_ids))
    cat = {k: v[order] for k, v in cat.items()}
    scores = {s: rank01(v[order]) for s, v in raw.items()}
    return cat, mode_ids[order], scores


def run_all(cat, mode_ids, scores, names, seed, router=None):
    feat, logits, labels, loss, prob = (cat[k] for k in ("feat", "logits", "labels", "loss", "prob"))
    rng = np.random.default_rng(1000 + seed)
    pseudo_ok = logits.argmax(-1) == labels
    gates = dict(scores)
    gates["random"] = rng.random(len(labels))
    if router is not None:  # 모드별로 그 모드에서 가장 좋았던 신호의 rank를 쓴다
        gates["oracle_router"] = np.array([scores[router.get(names[k], "entropy")][t]
                                           for t, k in enumerate(mode_ids)])
    rows, curves = [], {}
    for g, s in gates.items():
        preds, contam = run_stream(feat, logits, labels, s, loss, prob)
        rows.append(dict(gate=g, seed=seed, **stream_metrics(preds, contam, labels, mode_ids, names)))
        curves[g] = contam
    preds, contam = run_stream(feat, logits, labels, gates["entropy"], loss, prob, skip=~pseudo_ok)
    rows.append(dict(gate="oracle", seed=seed, **stream_metrics(preds, contam, labels, mode_ids, names)))
    curves["oracle"] = contam
    zs = logits.argmax(-1)
    rows.append(dict(gate="no_cache", seed=seed,
                     **stream_metrics(zs, np.zeros(len(zs)), labels, mode_ids, names)))
    return rows, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="runs/dump")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    modes = load_modes(args.dump)
    labeled = [n for n in modes if n != "ood"]

    # per-mode: 모드별 최적 신호
    per = []
    for n in labeled:
        for seed in args.seeds:
            rows, _ = run_all(*build(modes, [n], seed), [n], seed)
            per += [dict(mode=n, **r) for r in rows]
    per = pd.DataFrame(per)
    per.to_csv(os.path.join(args.out, "online_per_mode.csv"), index=False)
    per_mean = per.groupby(["mode", "gate"])["acc"].mean().unstack(0)
    print(per_mean.round(4).to_string())
    cand = per_mean.drop(index=["oracle", "random", "no_cache"])
    router = cand.idxmax().to_dict()
    print("best signal per mode:", router)

    # mixed: 모드 정보 없이. V2 계열 열은 같은 이미지를 공유하므로 (clean과 aug_noise는 view 0까지 동일)
    # 캐시가 같은 사진을 다시 찾는 누수를 막기 위해 열마다 겹치지 않는 이미지만 쓰고, 열 크기를 맞춘다.
    v2_cols = [n for n in modes if COLUMNS[n]["source"] == "v2"]
    per_mode = min(len(modes[n]["labels"]) for n in v2_cols) // len(v2_cols)
    mix_modes = {}
    for n, m in modes.items():
        idx = (np.arange(v2_cols.index(n), len(m["labels"]), len(v2_cols)) if n in v2_cols
               else np.arange(len(m["labels"])))[:per_mode]
        mix_modes[n] = {k: m[k][idx] for k in ("feat", "logits", "labels", "loss", "prob")}
        mix_modes[n]["scores"] = {k: v[idx] for k, v in m["scores"].items()}
    print(f"mixed stream: {per_mode} disjoint samples per mode")
    mixed, curves = [], {}
    for seed in args.seeds:
        rows, cv = run_all(*build(mix_modes, list(mix_modes), seed), list(mix_modes), seed, router=router)
        mixed += rows
        for g, c in cv.items():
            curves.setdefault(g, []).append(c)
    mixed = pd.DataFrame(mixed)
    mixed.to_csv(os.path.join(args.out, "online_mixed.csv"), index=False)
    agg = mixed.drop(columns="seed").groupby("gate").mean().sort_values("acc", ascending=False)
    ent = agg.loc["entropy"]
    mode_cols = [c for c in agg.columns if c.startswith("acc_")]
    agg["modes_worse_than_entropy"] = (agg[mode_cols] < ent[mode_cols]).sum(1)
    agg.to_csv(os.path.join(args.out, "online_mixed_summary.csv"))
    print(agg.round(4).to_string())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for g, cs in curves.items():
        ax.plot(np.mean(cs, 0), label=g, lw=2.2 if g in ("entropy", "oracle_router", "random") else 1)
    ax.set_xlabel("stream step")
    ax.set_ylabel("cache contamination")
    ax.set_title("mixed stream: wrong pseudo-labels in positive cache")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "online_mixed_contamination.png"), dpi=150)


if __name__ == "__main__":
    main()
