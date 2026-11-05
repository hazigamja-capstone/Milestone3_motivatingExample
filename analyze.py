"""구간마다 정확도 차이(TTA - zero-shot)와 정답 없는 신호를 모아 분리력을 본다.

예: python analyze.py --seg 512
결과: outputs/segments.csv, outputs/separability.csv, outputs/scatter.png
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

LN_C = math.log(1000)

# 이름: (그룹, 샘플 단위 값을 만드는 함수). 모두 정답 없이 계산된다.
SIGNALS = {
    # TTA 모델만 보는 기존 신호
    "tta_entropy": ("existing", lambda r: r["ent_t"]),
    "tta_maxprob": ("existing", lambda r: r["maxp_t"]),
    "tta_reliable_frac": ("existing", lambda r: (r["ent_t"] < 0.4 * LN_C).astype(float)),
    # zero-shot과 비교하는 신호
    "disagree_rate": ("zs_compare", lambda r: (r["pred_t"] != r["pred_z"]).astype(float)),
    "kl_tta_zs": ("zs_compare", lambda r: r["kl_tz"]),
    "zs_prob_of_tta_pred": ("zs_compare", lambda r: r["pz_at_t"]),
    "conf_gap_on_disagree": ("zs_compare", None),  # 구간 단위로 따로 계산
    # 데이터 난이도 (zero-shot만 봄)
    "zs_entropy": ("data", lambda r: r["ent_z"]),
}


def auroc(score, label):
    """label=1(손해)을 score가 높을수록 잘 고르는 정도. 0.5면 무작위."""
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([pos, neg])).rank().values
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def segments(path, seg):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    r = {k: z[k] for k in z.files if k != "meta"}
    run = os.path.basename(path)[:-4]
    rows = []
    for di, dom in enumerate(meta["domains"]):
        ids = np.where(r["dom"] == di)[0]
        for s in range(0, len(ids), seg):
            j = ids[s:s + seg]
            if len(j) < seg // 2:
                continue
            sub = {k: v[j] for k, v in r.items()}
            acc_t = (sub["pred_t"] == sub["y"]).mean()
            acc_z = (sub["pred_z"] == sub["y"]).mean()
            row = dict(run=run, model=meta["model"], method=meta["method"], domain=dom, seg=s // seg,
                       n=len(j), acc_tta=acc_t, acc_zs=acc_z, delta=acc_t - acc_z)
            for name, (_, fn) in SIGNALS.items():
                if fn is not None:
                    row[name] = float(np.mean(fn(sub)))
            d = sub["pred_t"] != sub["pred_z"]
            # 다르게 답한 샘플에서 zero-shot이 자기 답을 TTA 답보다 얼마나 더 믿는지
            row["conf_gap_on_disagree"] = float(np.mean(sub["maxp_z"][d] - sub["pz_at_t"][d])) if d.any() else 0.0
            rows.append(row)
    return rows


def separability(df):
    out = []
    harm = (df["delta"] < 0).astype(int).values
    for name, (group, _) in SIGNALS.items():
        a = auroc(df[name].values, harm)
        rho = df[[name, "delta"]].rank().corr().iloc[0, 1]
        out.append(dict(signal=name, group=group, auroc_harm=a,
                        separability=max(a, 1 - a) if not np.isnan(a) else np.nan, spearman_delta=rho))
    return pd.DataFrame(out)


def judge(sep):
    def best(g):
        s = sep[sep.group == g].separability
        return s.max() if len(s) else np.nan

    e, z = best("existing"), best("zs_compare")
    tag = lambda v: "나뉨" if v >= 0.8 else ("섞임" if v <= 0.65 else "애매")
    return f"기존 신호 최고 {e:.2f} ({tag(e)}) / zero-shot 비교 최고 {z:.2f} ({tag(z)})"


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(SIGNALS)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    runs = sorted(df.run.unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 1)))
    for ax, name in zip(axes.flat, names):
        for c, run in zip(colors, runs):
            d = df[df.run == run]
            ax.scatter(d[name], 100 * d["delta"], s=12, alpha=0.7, color=c, label=run)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"[{SIGNALS[name][0]}] {name}", fontsize=10)
        ax.set_ylabel("acc(TTA) - acc(zero-shot) [%p]")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seg", type=int, default=512, help="구간 하나의 샘플 수")
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.out, "*.npz"))):
        rows += segments(p, args.seg)
    df = pd.DataFrame(rows)
    df = df[df.method != "none"]
    df.to_csv(os.path.join(args.out, "segments.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n== 실행별 구간 수와 손해 구간 비율 ==")
    print(df.groupby("run").agg(n_seg=("delta", "size"), harm_frac=("delta", lambda d: (d < 0).mean()),
                                mean_delta=("delta", "mean")).round(3))
    print("\n== 도메인별 평균 차이 (%p) ==")
    print((100 * df.pivot_table(index="run", columns="domain", values="delta")).round(1))

    seps = []
    for scope, d in [("pooled", df)] + [(r, g) for r, g in df.groupby("run")]:
        harm = (d.delta < 0).mean()
        if harm == 0 or harm == 1:
            print(f"\n[{scope}] 손해 구간 비율 {harm:.0%}: 이득과 손해가 섞여 있지 않아 분리력을 볼 수 없음")
            continue
        s = separability(d)
        s.insert(0, "scope", scope)
        seps.append(s)
        print(f"\n== [{scope}] 분리력 (손해 구간 비율 {harm:.0%}) ==")
        print(s.drop(columns="scope").round(3).to_string(index=False))
        print("판정:", judge(s))
    if seps:
        pd.concat(seps).to_csv(os.path.join(args.out, "separability.csv"), index=False)
    plot(df, os.path.join(args.out, "scatter.png"))
    print("\n그림:", os.path.join(args.out, "scatter.png"))


if __name__ == "__main__":
    main()
