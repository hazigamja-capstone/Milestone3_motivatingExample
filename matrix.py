"""2단계 (episodic): 지표 x 실패 모드 교차 평가.

모든 신호를 같은 역할에 끼운다: 64개 뷰 중 상위 rho 비율을 골라 확률 평균으로 예측 (TPT의 선택 단계, 프롬프트 업데이트 없음).
셀 값 = 엔트로피 선택 대비 정확도 차이. 라벨은 여기서 채점에만 쓴다.
별도로 각 논문 방법 전체(methods.py, tune.py)를 공식 규칙대로 돌려 TPT, 엔트로피 선택과 비교한다.

python matrix.py --dump runs/dump --out results   (tune.py를 먼저 돌리면 TPT, R-TPT 행이 생긴다)
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
import torch

import methods as M
import signals as sig
import stats as ST
from data import COLUMNS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHODS = ["zero_shot", "entropy_sel", "tpt", "zero", "mta", "adte", "rtpt_ens", "rtpt"]
# 방법마다 겨냥한 열. 여기서 엔트로피 선택을 못 이기면 교차 열은 해석하지 않는다.
# ADTE가 겨냥한 long-tail 열은 없어서 논문 벤치마크(ImageNet-A)로 대체한다.
OWN_COLUMN = {"rtpt_ens": "adversarial", "rtpt": "adversarial", "mta": "aug_noise",
              "zero": "natural_hard", "adte": "natural_hard"}
# 구성요소를 하나씩 빼며 이득이 어디서 사라지는지 본다. 앞 행이 원 방법, 아래로 갈수록 더 많이 뺀다.
ABLATIONS = {
    "MTA": ["mta", "mta_no_affinity", "mta_uniform", "feat_mean", "mta_inlier_top10"],
    "R-TPT": ["rtpt", "rtpt_tpt_loss", "rtpt_uniform", "rtpt_view0", "rtpt_ens", "logit_mean"],
}
ABLATION_DESC = {
    "mta": "원 방법 (inlierness + affinity + mode)",
    "mta_no_affinity": "affinity 항 제거 (lambda_q=0)",
    "mta_uniform": "inlierness 제거 (균등 가중 MeanShift)",
    "feat_mean": "mode 찾기 제거 (뷰 특징 평균)",
    "mta_inlier_top10": "mode 대신 inlierness 상위 10% 선택 (matrix 방식)",
    "rtpt": "원 방법 (뷰별 엔트로피 튜닝 + 신뢰도 가중)",
    "rtpt_tpt_loss": "손실을 TPT marginal 엔트로피로",
    "rtpt_uniform": "신뢰도 가중 제거 (튜닝 후 균등 평균)",
    "rtpt_view0": "앙상블 제거 (튜닝 후 원본 뷰)",
    "rtpt_ens": "튜닝 제거 (신뢰도 가중만)",
    "logit_mean": "튜닝, 가중 둘 다 제거 (logits 균등 평균)",
}


def select_predict(p, score, rho):
    k = max(1, int(round(p.shape[1] * rho)))
    idx = score.topk(k, -1).indices
    ps = p.gather(1, idx[..., None].expand(-1, -1, p.shape[-1]))
    return ps.mean(1).argmax(-1)


def load_text(dump):
    """dump에 쓴 텍스트 특징 [C,D]. MTA는 mode를 텍스트 특징에 투영해 예측한다."""
    path = os.path.join(dump, "text.npy")
    if not os.path.exists(path):
        from common import imagenet_a_classes, load_clip, text_features
        model, tok = load_clip("cuda")
        np.save(path, text_features(model, tok, imagenet_a_classes()[2], "cuda").cpu().numpy())
    return torch.from_numpy(np.load(path))


def evaluate_methods(t, p, y, S, mode, text, tuned):
    """논문 방법 전체를 공식 규칙대로. tuned = tune.py 결과 (없으면 TPT, R-TPT 행 생략)."""
    preds = {
        "zero_shot": p[:, 0].argmax(-1),
        "entropy_sel": select_predict(p, S["entropy"], 0.1),
        "zero": M.zero_predict(p, -S["entropy"]),
        "mta": (mode @ text.T).argmax(-1),
        "adte": select_predict(p, S["adte"], 0.1),
        "rtpt_ens": M.rtpt_ensemble(t["logits"], S["rtpt_rel"]).argmax(-1),
    }
    rel = S["rtpt_rel"]
    preds.update({
        "mta_no_affinity": (M.mta_batch(t["feats"], t["logits"], lambda_q=0.0)[0] @ text.T).argmax(-1),
        "mta_uniform": (M.mta_batch(t["feats"], t["logits"], uniform_y=True)[0] @ text.T).argmax(-1),
        "feat_mean": (t["feats"].mean(1) @ text.T).argmax(-1),
        "mta_inlier_top10": select_predict(p, S["mta_inlier"], 0.1),
        "logit_mean": t["logits"].mean(1).argmax(-1),
    })
    if tuned is not None:
        tpt_v = torch.from_numpy(tuned["tpt"]).float()
        rtpt_v = torch.from_numpy(tuned["rtpt"]).float()
        preds.update({
            "tpt": tpt_v[:, 0].argmax(-1),
            "rtpt": M.rtpt_ensemble(rtpt_v, rel).argmax(-1),
            "rtpt_tpt_loss": M.rtpt_ensemble(tpt_v, rel).argmax(-1),
            "rtpt_uniform": rtpt_v.mean(1).argmax(-1),
            "rtpt_view0": rtpt_v[:, 0].argmax(-1),
        })
    correct = {k: (v == y).numpy() for k, v in preds.items()}
    idx = ST.bootstrap_idx(len(y))
    rows = []
    for k, c in correct.items():
        acc, lo, hi = ST.acc_ci(c, idx)
        row = dict(method=k, acc=acc, acc_lo=lo, acc_hi=hi)
        for base in ("zero_shot", "entropy_sel", "tpt"):
            if base in correct:
                d, dlo, dhi, pv = ST.paired_diff(c, correct[base], idx)
                row.update({f"delta_vs_{base}": d, f"delta_vs_{base}_lo": dlo,
                            f"delta_vs_{base}_hi": dhi, f"p_vs_{base}": pv})
        rows.append(row)
    return pd.DataFrame(rows), correct


# CRITERIA.md 2절의 사전 비교 목록: (열, 방법, 기준)
PREREGISTERED = ([("adversarial", "rtpt", "tpt"), ("adversarial", "rtpt", "entropy_sel")]
                 + [(c, "rtpt", "tpt") for c in ("clean", "aug_noise", "corruption", "natural_hard")]
                 + [(c, "tpt", "zero_shot") for c in ("clean", "aug_noise", "adversarial", "corruption", "natural_hard")])


def preregistered(correct_by_col):
    rows = []
    for col, m, base in PREREGISTERED:
        cc = correct_by_col.get(col, {})
        if m not in cc or base not in cc:
            continue
        idx = ST.bootstrap_idx(len(cc[m]))
        d, lo, hi, p = ST.paired_diff(cc[m], cc[base], idx)
        rows.append(dict(column=col, method=m, baseline=base, delta=d, lo=lo, hi=hi, p=p))
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"], df["significant"] = ST.holm(df["p"])
    return df


def soft_predict(p, score, tau):
    """버리는 뷰 없이 샘플 안에서 표준화한 점수의 softmax로 가중 평균."""
    z = (score - score.mean(1, keepdim=True)) / score.std(1, keepdim=True).clamp_min(1e-6)
    w = (z / tau).softmax(1)
    return (w[..., None] * p).sum(1).argmax(-1)


def rank(score):
    """샘플 안에서의 순위 (0 = 가장 낮음)."""
    return score.argsort(1).argsort(1).float()


def evaluate_variants(p, y, S, rho, tau, filt):
    """원 논문 쓰임에 가까운 변형: 소프트 가중, 엔트로피와 결합.
    soft: 신호 단독 소프트 가중
    filt: 엔트로피 상위 filt 비율로 1차 거른 뒤 그 안에서 신호로 상위 rho 선택 (DeYO 방식)
    rank: 엔트로피 순위 + 신호 순위 합으로 상위 rho 선택
    orig: 원본 뷰(view 0)는 항상 포함하고 나머지만 신호로 선택 (TPT처럼 원본을 묶음에 고정)
    """
    V = p.shape[1]
    ent = S["entropy"]
    ent_acc = (select_predict(p, ent, rho) == y).float().mean().item()
    keep = rank(ent) >= V * (1 - filt)

    def with_orig(s):
        s = s.clone()
        s[:, 0] = float("inf")
        return s

    out = [dict(signal="entropy", variant="soft", acc=(soft_predict(p, ent, tau) == y).float().mean().item()),
           dict(signal="entropy", variant="orig", acc=(select_predict(p, with_orig(ent), rho) == y).float().mean().item())]
    for name in sig.SIGNALS:
        if name == "entropy":
            continue
        s = S[name]
        preds = {
            "soft": soft_predict(p, s, tau),
            "filt": select_predict(p, rank(s) + V * keep, rho),
            "rank": select_predict(p, rank(ent) + rank(s) + 1e-3 * ent, rho),
            "orig": select_predict(p, with_orig(s), rho),
        }
        for v, pr in preds.items():
            out.append(dict(signal=name, variant=v, acc=(pr == y).float().mean().item()))
    df = pd.DataFrame(out)
    df["delta_vs_entropy"] = df["acc"] - ent_acc
    return df


def within_sample_auroc(score, correct):
    """샘플마다 '정답 뷰 점수 > 오답 뷰 점수' 확률. 정답/오답 뷰가 모두 있는 샘플만 의미가 있다."""
    pos, neg = correct.float(), (~correct).float()
    gt = (score[:, :, None] > score[:, None, :]).float() + 0.5 * (score[:, :, None] == score[:, None, :]).float()
    num = (gt * pos[:, :, None] * neg[:, None, :]).sum((1, 2))
    den = pos.sum(1) * neg.sum(1)
    return num / den.clamp_min(1), den > 0


def evaluate_column(p, y, S, rho, n_random=5):
    correct = p.argmax(-1) == y[:, None]
    g = torch.Generator().manual_seed(0)
    rows = {
        "original_only": (p[:, 0].argmax(-1) == y).float().mean().item(),
        "all_views": (p.mean(1).argmax(-1) == y).float().mean().item(),
        "random": np.mean([(select_predict(p, torch.rand(p.shape[:2], generator=g), rho) == y).float().mean().item()
                           for _ in range(n_random)]),
        "oracle": (select_predict(p, correct.float() + 1e-3 * S["entropy"], rho) == y).float().mean().item(),
    }
    ent_pred = select_predict(p, S["entropy"], rho)
    # 구제 가능 케이스: 엔트로피 선택은 틀렸는데 맞힌 뷰가 하나라도 있는 샘플
    rescuable = (ent_pred != y) & correct.any(1)

    out = []
    for name in sig.SIGNALS:
        acc = (select_predict(p, S[name], rho) == y).float().mean().item()
        au, valid = within_sample_auroc(S[name], correct)
        m = valid & rescuable
        # 원본 뷰를 뺀 crop끼리만: 원본 하나가 결과를 끌고 가는지 떼어 보기 위함
        auc, valid_c = within_sample_auroc(S[name][:, 1:], correct[:, 1:])
        out.append(dict(signal=name, acc=acc,
                        auroc_all=au[valid].mean().item(),
                        auroc_rescuable=au[m].mean().item() if m.any() else float("nan"),
                        auroc_crops=auc[valid_c].mean().item()))
    for name, acc in rows.items():
        out.append(dict(signal=name, acc=acc, auroc_all=float("nan"), auroc_rescuable=float("nan")))
    df = pd.DataFrame(out)
    df["delta_vs_entropy"] = df["acc"] - df.loc[df.signal == "entropy", "acc"].item()
    df["n"] = len(y)
    df["rescuable_rate"] = rescuable.float().mean().item()
    return df


def heatmap(pivot, path, title, center=0.0, fmt="{:+.1f}", scale=100):
    fig, ax = plt.subplots(figsize=(1.4 * pivot.shape[1] + 2, 0.45 * pivot.shape[0] + 1.5))
    vals = pivot.values * scale
    lim = np.nanmax(np.abs(vals - center * scale)) or 1
    im = ax.imshow(vals, cmap="RdYlGn", vmin=center * scale - lim, vmax=center * scale + lim, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="runs/dump")
    ap.add_argument("--out", default="results")
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.5, help="소프트 가중 온도 (표준화 점수 기준)")
    ap.add_argument("--filt", type=float, default=0.5, help="결합(filt)에서 엔트로피로 남길 비율")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    text = load_text(args.dump)
    frames, var_frames, meth_frames = [], [], []
    correct_by_col = {}
    for col in COLUMNS:
        path = os.path.join(args.dump, f"{col}.npz")
        if col == "ood" or not os.path.exists(path):
            continue
        d = dict(np.load(path))
        t = sig.to_torch(d)
        y = torch.from_numpy(d["labels"])
        p = t["logits"].softmax(-1)
        mta_out = M.mta_batch(t["feats"], t["logits"])
        S = sig.compute(**t, mta_out=mta_out)
        tuned_path = os.path.join(args.dump, f"{col}_tuned.npz")
        tuned = dict(np.load(tuned_path)) if os.path.exists(tuned_path) else None

        df = evaluate_column(p, y, S, args.rho)
        df.insert(0, "column", col)
        frames.append(df)
        print(df.round(4).to_string(index=False))
        vdf = evaluate_variants(p, y, S, args.rho, args.tau, args.filt)
        vdf.insert(0, "column", col)
        var_frames.append(vdf)
        mdf, correct_by_col[col] = evaluate_methods(t, p, y, S, mta_out[0], text, tuned)
        mdf.insert(0, "column", col)
        meth_frames.append(mdf)
        print(mdf.round(4).to_string(index=False))
    res = pd.concat(frames)
    res.to_csv(os.path.join(args.out, "selection.csv"), index=False)

    order = sig.SIGNALS + ["random", "all_views", "original_only", "oracle"]
    cols = [c for c in COLUMNS if c in set(res.column)]
    delta = res.pivot(index="signal", columns="column", values="delta_vs_entropy").reindex(order)[cols]
    auroc = res.pivot(index="signal", columns="column", values="auroc_rescuable").reindex(sig.SIGNALS)[cols]
    heatmap(delta, os.path.join(args.out, "selection_delta.png"),
            f"top-{args.rho:.0%} view selection: acc - entropy (%p)")
    heatmap(auroc, os.path.join(args.out, "auroc_rescuable.png"),
            "view-level AUROC on rescuable samples", center=0.5, fmt="{:.2f}", scale=1)

    # 실사용 관점 요약: 자기 열에서는 이기지만 다른 열에서 지는가
    summ = delta.drop(index=["entropy", "oracle"], errors="ignore").T.describe().T[["mean", "min", "max"]]
    summ["cols_worse_than_entropy"] = (delta.drop(index=["entropy", "oracle"], errors="ignore") < 0).sum(1)
    summ["best_column"] = delta.drop(index=["entropy", "oracle"], errors="ignore").idxmax(1)
    summ.to_csv(os.path.join(args.out, "selection_summary.csv"))
    print(summ.round(4).to_string())

    # 변형: 단독 하드 선택(기존 행) vs 소프트 가중 vs 엔트로피 결합
    var = pd.concat(var_frames)
    var.to_csv(os.path.join(args.out, "variants.csv"), index=False)
    hard = res[res.signal.isin(sig.SIGNALS) & (res.signal != "entropy")][["column", "signal", "delta_vs_entropy"]]
    var = pd.concat([var, hard.assign(variant="hard")])
    var["row"] = var.signal + " / " + var.variant
    row_order = ["entropy / soft", "entropy / orig"] + [f"{s} / {v}" for s in sig.SIGNALS if s != "entropy"
                                                        for v in ("hard", "soft", "filt", "rank", "orig")]
    vpiv = var.pivot(index="row", columns="column", values="delta_vs_entropy").reindex(row_order)[cols]
    heatmap(vpiv, os.path.join(args.out, "variants_delta.png"),
            f"hard / soft(tau={args.tau}) / entropy filter({args.filt:.0%}) / rank sum / orig fixed: "
            "acc - entropy hard (%p)")
    vsum = vpiv.T.describe().T[["mean", "min", "max"]]
    vsum["cols_worse_than_entropy"] = (vpiv < 0).sum(1)
    vsum.to_csv(os.path.join(args.out, "variants_summary.csv"))
    print(vsum.round(4).to_string())

    # 논문 방법 전체: 각 논문이 실제로 비교한 상대(TPT, zero-shot)와 엔트로피 선택 대비
    meth = pd.concat(meth_frames)
    meth.to_csv(os.path.join(args.out, "methods.csv"), index=False)
    morder = [m for m in METHODS if m in set(meth.method)]
    macc = meth.pivot(index="method", columns="column", values="acc").reindex(morder)[cols]
    macc["mean"] = macc.mean(1)
    print(macc.round(4).to_string())
    base = "tpt" if "tpt" in morder else "entropy_sel"
    heatmap(meth.pivot(index="method", columns="column", values=f"delta_vs_{base}").reindex(morder)[cols],
            os.path.join(args.out, "methods_delta.png"), f"official methods: acc - {base} (%p)")

    # ablation: 원 방법에서 구성요소를 하나씩 뺀다
    abl = []
    for group, rows in ABLATIONS.items():
        rows = [r for r in rows if r in set(meth.method)]
        piv = meth.pivot(index="method", columns="column", values="acc").reindex(rows)[cols]
        piv["mean"] = piv.mean(1)
        piv.insert(0, "desc", [ABLATION_DESC[r] for r in rows])
        piv.insert(0, "group", group)
        abl.append(piv)
    abl = pd.concat(abl)
    abl.to_csv(os.path.join(args.out, "ablation.csv"))
    print(abl.round(4).to_string())

    # ablation 신뢰구간: 원 방법 - 구성요소를 뺀 행 (양수 = 그 구성요소가 기여)
    abl_ci = []
    for group, rows in ABLATIONS.items():
        for col in cols:
            cc = correct_by_col[col]
            if rows[0] not in cc:
                continue
            idx = ST.bootstrap_idx(len(cc[rows[0]]))
            for r in rows[1:]:
                if r in cc:
                    d, lo, hi, pv = ST.paired_diff(cc[rows[0]], cc[r], idx)
                    abl_ci.append(dict(group=group, column=col, removed=r, desc=ABLATION_DESC[r],
                                       full_minus_removed=d, lo=lo, hi=hi, p=pv))
    pd.DataFrame(abl_ci).to_csv(os.path.join(args.out, "ablation_ci.csv"), index=False)

    # 자기 열 게이트: 자기 열에서 엔트로피 선택을 CI 기준으로 이겨야 교차 열 해석으로 넘어간다
    gate = []
    for m, own in OWN_COLUMN.items():
        r = meth[(meth.method == m) & (meth.column == own)]
        if r.empty:
            continue
        r = r.iloc[0]
        passed = r["delta_vs_entropy_sel_lo"] > 0
        for col in cols:
            c = meth[(meth.method == m) & (meth.column == col)].iloc[0]
            gate.append(dict(method=m, own_column=own, gate_passed=passed, column=col, is_own=col == own,
                             delta=c["delta_vs_entropy_sel"], lo=c["delta_vs_entropy_sel_lo"],
                             hi=c["delta_vs_entropy_sel_hi"]))
    gate = pd.DataFrame(gate)
    gate.to_csv(os.path.join(args.out, "gate.csv"), index=False)
    print(gate.round(4).to_string(index=False))

    # CRITERIA.md 2절 사전 비교 (Holm 보정)
    pre = preregistered(correct_by_col)
    pre.to_csv(os.path.join(args.out, "preregistered.csv"), index=False)
    print(pre.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
