"""논문 재현 판정 (CRITERIA.md 1절). paper_a, paper_v2 열에서 공식 방법을 돌려 논문 수치와 비교한다.

판정: 재현 = 논문 수치가 우리 정확도의 95% CI 안, 부분 재현 = CI 밖이지만 zero-shot 대비 이득의 CI 하한 > 0, 실패 = 그 외.
TPT는 시간 때문에 앞 500샘플만 튜닝해서, TPT 행은 그 500샘플 위에서 판정한다.

python reproduce.py --dump runs/dump --out results
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

import methods as M
import signals as sig
import stats as ST
from matrix import select_predict

# ViT-B/16, 프롬프트 하나, 64뷰. CRITERIA.md 1절 표와 같다.
PAPER = {
    "paper_a": {"zero_shot": 47.87, "tpt": 54.77, "mta": 57.41, "zero": 59.61},
    "paper_v2": {"zero_shot": 60.86, "tpt": 63.45, "mta": 63.61, "zero": 64.16},
}


def predict(d, tuned, chunk=200):
    """1000클래스 열은 float32로 한 번에 올리면 수 GB라 샘플을 나눠 계산한다.
    ADTE 편향 q만 테스트셋 전체(뷰 평균 logits)로 먼저 추정한다."""
    text = torch.from_numpy(d["text"]).float()
    q = M.adte_bias(torch.from_numpy(d["logits"].mean(1, dtype=np.float32))[:, None])
    preds = {k: [] for k in ("zero_shot", "entropy_sel", "zero", "mta", "adte", "rtpt_ens")}
    for s in range(0, len(d["labels"]), chunk):
        t = sig.to_torch({k: d[k][s:s + chunk] for k in ("logits", "feats", "areas")})
        p = t["logits"].softmax(-1)
        mode, y_in = M.mta_batch(t["feats"], t["logits"])
        S = sig.compute(**t, adte_q=q, mta_out=(mode, y_in))
        preds["zero_shot"].append(p[:, 0].argmax(-1))
        preds["entropy_sel"].append(select_predict(p, S["entropy"], 0.1))
        preds["zero"].append(M.zero_predict(p, -S["entropy"]))
        preds["mta"].append((mode @ text.T).argmax(-1))
        preds["adte"].append(select_predict(p, S["adte"], 0.1))
        preds["rtpt_ens"].append(M.rtpt_ensemble(t["logits"], S["rtpt_rel"]).argmax(-1))
    y = torch.from_numpy(d["labels"])
    correct = {k: (torch.cat(v) == y).numpy() for k, v in preds.items()}
    if tuned is not None:
        n = len(tuned["tpt"])
        correct["tpt"] = torch.from_numpy(tuned["tpt"][:, 0]).float().argmax(-1).numpy() == d["labels"][:n]
    return correct


def judge(col, correct):
    rows = []
    n_all = len(correct["zero_shot"])
    for m, c in correct.items():
        n = len(c)
        idx = ST.bootstrap_idx(n)
        acc, lo, hi = ST.acc_ci(c, idx)
        g, glo, ghi, _ = ST.paired_diff(c, correct["zero_shot"][:n], idx)
        ref = PAPER[col].get(m)
        if ref is None:
            verdict = "기준 수치 없음"
        elif lo * 100 <= ref <= hi * 100:
            verdict = "재현"
        elif m != "zero_shot" and glo > 0 and (ref - PAPER[col]["zero_shot"]) > 0:
            verdict = "부분 재현"
        else:
            verdict = "실패"
        rows.append(dict(column=col, method=m, n=n, acc=acc * 100, lo=lo * 100, hi=hi * 100, paper=ref,
                         gain_vs_zs=g * 100, gain_lo=glo * 100, gain_hi=ghi * 100,
                         paper_gain=None if ref is None else ref - PAPER[col]["zero_shot"],
                         verdict=verdict, subset=n < n_all))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="runs/dump")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for col in PAPER:
        path = os.path.join(args.dump, f"{col}.npz")
        if not os.path.exists(path):
            print(f"[{col}] dump 없음, 건너뜀")
            continue
        tuned_path = os.path.join(args.dump, f"{col}_tuned.npz")
        tuned = dict(np.load(tuned_path)) if os.path.exists(tuned_path) else None
        rows += judge(col, predict(dict(np.load(path)), tuned))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "reproduction.csv"), index=False)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
