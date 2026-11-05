"""후속 실험 분석: 1단계 재현, 2단계 시간 통제, 3단계 다른 실행에 기준값 적용, 4단계 미리 알아채기.

sweep.py로 만든 실행들(outputs_followup/*.npz)을 읽는다. 모든 신호는 정답 없이 계산하고, 정답은 평가에만 쓴다.

예: python followup.py --out outputs_followup
결과: {out}/followup_*.csv, {out}/followup_delta.png
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HARM, GAIN = -0.02, 0.02          # 손해: 차이 < -2%p, 이득: 차이 > +2%p
SILENT_DROP = -0.05               # 1단계: 조용한 손해로 세는 깊이
COLLAPSE_SHARE = 0.1              # 최빈 클래스 비율이 이보다 작으면 붕괴가 아님
TOPK = (3, 5, 10)
EXISTING = ["tta_entropy", "tta_top_class_share"]
ZS = ["disagree_rate", "kl_tta_zs", "zs_prob_of_tta_pred"]


def signal_names(has_rank):
    return EXISTING + ZS + ([f"topk{k}" for k in TOPK] if has_rank else [])


# ---------------------------------------------------------------- 공통

def load(out):
    runs = []
    for p in sorted(glob.glob(os.path.join(out, "*.npz"))):
        z = np.load(p)
        meta = json.loads(str(z["meta"]))
        if meta["method"] == "none":
            continue
        r = {k: z[k] for k in z.files if k != "meta"}
        runs.append(dict(name=os.path.basename(p)[:-4], meta=meta, r=r))
    return runs


def window(r, j):
    """샘플 번호 j 묶음의 정확도와 신호."""
    y, pt, pz = r["y"][j], r["pred_t"][j], r["pred_z"][j]
    d = pt != pz
    f = dict(acc_tta=(pt == y).mean(), acc_zs=(pz == y).mean())
    f["delta"] = f["acc_tta"] - f["acc_zs"]
    f["tta_entropy"] = r["ent_t"][j].mean()
    f["tta_top_class_share"] = np.bincount(pt).max() / len(j)
    f["disagree_rate"] = d.mean()
    f["kl_tta_zs"] = r["kl_tz"][j].mean()
    f["zs_prob_of_tta_pred"] = r["pz_at_t"][j].mean()
    if "rank_z_at_t" in r:
        rk = r["rank_z_at_t"][j][d]
        for k in TOPK:  # 답이 바뀐 샘플 중 TTA 답이 zs 상위 K개 안에 드는 비율
            f[f"topk{k}"] = (rk < k).mean() if len(rk) else 1.0
    return f


def segments(runs, seg):
    rows = []
    for run in runs:
        m, r = run["meta"], run["r"]
        for di, dom in enumerate(m["domains"]):
            ids = np.where(r["dom"] == di)[0]
            for s in range(0, len(ids), seg):
                j = ids[s:s + seg]
                if len(j) < seg // 2:
                    continue
                rows.append(dict(run=run["name"], model=m["model"], method=m["method"], seed=m["seed"],
                                 lr=float(m["lr"]), domain=dom, seg=s // seg, **window(r, j)))
    df = pd.DataFrame(rows)
    df["harm"] = np.where(df.delta < HARM, 1, np.where(df.delta > GAIN, 0, -1))
    return df


def auroc(score, label):
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([pos, neg])).rank().values
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def sep_table(d, signals):
    out = []
    for s in signals:
        a = auroc(d[s].values, d.harm.values)
        out.append(dict(signal=s, auroc=a, separability=max(a, 1 - a) if not np.isnan(a) else np.nan,
                        harm_side="높을수록" if a >= 0.5 else "낮을수록"))
    return pd.DataFrame(out)


def show(title, df):
    print(f"\n== {title} ==")
    print(df.round(3).to_string(index=False) if len(df) else "(없음)")


# ---------------------------------------------------------------- 1단계

def step1(df, out):
    rows = []
    for run, g in df.groupby("run"):
        g = g.sort_values(["domain", "seg"])
        silent = (g.delta < SILENT_DROP) & (g.tta_top_class_share < COLLAPSE_SHARE)
        longest = cur = 0
        for v in silent:  # 조용한 손해가 연속으로 몇 구간 이어졌나
            cur = cur + 1 if v else 0
            longest = max(longest, cur)
        harm = g[g.harm == 1]
        ent_dir = np.nan
        if len(harm):
            first = harm.seg.min()
            before = g[(g.seg < first) & (g.seg >= first - 3)]
            if len(before):
                ent_dir = harm.tta_entropy.mean() - before.tta_entropy.mean()
        rows.append(dict(run=run, model=g.model.iloc[0], method=g.method.iloc[0], seed=g.seed.iloc[0], lr=g.lr.iloc[0],
                         n_harm=len(harm), worst_delta=g.delta.min(), longest_silent=longest,
                         max_top_share_in_harm=harm.tta_top_class_share.max() if len(harm) else np.nan,
                         entropy_change=ent_dir, passed=longest >= 2))
    t = pd.DataFrame(rows).sort_values(["model", "method", "lr", "seed"])
    t.to_csv(os.path.join(out, "followup_step1.csv"), index=False)
    show("1단계: 실행별 손해 (worst_delta는 가장 나쁜 구간의 차이, entropy_change > 0이면 손해 때 엔트로피가 오름)",
         t.drop(columns="run"))

    summ = t.groupby(["model", "method", "lr"]).agg(seeds=("seed", "size"), passed=("passed", "sum")).reset_index()
    show("1단계: 조용한 손해(붕괴 없이 2구간 이상 연속 -5%p 미만)가 나온 seed 수", summ)
    main = summ[(summ.method == "eta") & np.isclose(summ.lr, 1e-3)]
    ok = len(main) > 0 and (main.passed >= 2).any()
    print("1단계 판정:", "통과 (ETA lr 1e-3에서 seed 2개 이상 재현)" if ok else
          "통과 못 함 → 여기서 멈춘다. 조용한 손해를 근거로 쓰지 않는다")
    plot_delta(df, os.path.join(out, "followup_delta.png"))
    return bool(ok)


def plot_delta(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lrs = sorted(df.lr.unique(), reverse=True)
    fig, axes = plt.subplots(1, len(lrs), figsize=(5.5 * len(lrs), 4), sharey=True, squeeze=False)
    for ax, lr in zip(axes[0], lrs):
        for (method, seed), g in df[df.lr == lr].groupby(["method", "seed"]):
            g = g.sort_values(["domain", "seg"])
            ax.plot(np.arange(len(g)), 100 * g.delta, marker="o", ms=3,
                    color={"eta": "tab:red", "sar": "tab:blue"}.get(method, "gray"),
                    ls=["-", "--", ":"][seed % 3], label=f"{method} seed {seed}")
        ax.axhline(0, color="k", lw=0.8)
        ax.axhspan(100 * HARM, 100 * GAIN, color="0.9")
        ax.set_title(f"lr {lr:g}")
        ax.set_xlabel("segment (512 images)")
    axes[0][0].set_ylabel("acc(TTA) - acc(zero-shot) [%p]")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("그림:", path)


# ---------------------------------------------------------------- 2단계

def step2(df, signals, out):
    keys = ["model", "domain", "seed", "lr", "seg"]
    eta = df[(df.method == "eta") & (df.harm == 1)]
    sar = df[(df.method == "sar") & (df.harm == 0)]
    pairs = eta.merge(sar[keys], on=keys)  # 같은 시점에 SAR 이득 구간이 있는 ETA 손해 구간만
    matched = pd.concat([eta.merge(pairs[keys].drop_duplicates(), on=keys),
                         sar.merge(pairs[keys].drop_duplicates(), on=keys)])
    n1, n0 = int((matched.harm == 1).sum()), int((matched.harm == 0).sum())
    res = []
    if n1 >= 3 and n0 >= 3:
        t = sep_table(matched, signals)
        t.insert(0, "analysis", "같은 시점 짝")
        res.append(t)
        show(f"2단계: 같은 시점의 ETA 손해 {n1}개 대 SAR 이득 {n0}개", t.drop(columns="analysis"))
    else:
        print(f"\n2단계: 같은 시점 짝이 부족함 (ETA 손해 {n1}개, SAR 이득 {n0}개)")

    # 보조: 구간 번호마다 평균을 빼서 시간에 따라 커지는 부분을 지운 값
    resid = df.copy()
    for s in signals:
        resid[s] = df[s] - df.groupby(["model", "domain", "seg"])[s].transform("mean")
    lab = resid[resid.harm >= 0]
    if (lab.harm == 1).sum() >= 3:
        t = sep_table(lab, signals)
        t.insert(0, "analysis", "시간 효과 뺀 값")
        res.append(t)
        show(f"2단계 보조: 시간 효과를 뺀 신호 (손해 {(lab.harm == 1).sum()} / 이득 {(lab.harm == 0).sum()})",
             t.drop(columns="analysis"))
    corr = pd.DataFrame([dict(signal=s, spearman_with_seg=df[["seg", s]].corr("spearman").iloc[0, 1])
                         for s in signals])
    show("참고: 구간 번호와 신호의 순위 상관 (클수록 시간에 따라 커짐)", corr)
    if res:
        r = pd.concat(res)
        r.to_csv(os.path.join(out, "followup_step2.csv"), index=False)
        main = r[r.analysis == "같은 시점 짝"]
        if len(main):
            good = main[main.signal.isin([s for s in signals if s not in EXISTING]) & (main.separability >= 0.7)]
            print("2단계 판정:", ("통과: " + ", ".join(good.signal)) if len(good) else
                  "통과 못 함 → zero-shot 비교 신호가 시간만 재고 있을 수 있다")


# ---------------------------------------------------------------- 3단계

def fit_threshold(train, s, direction, fa=0.05):
    gain = direction * train.loc[train.harm == 0, s].values
    return np.quantile(gain, 1 - fa) if len(gain) else np.nan


def direction_of(d, s):
    a = auroc(d[s].values, d.harm.values)
    return 1.0 if np.isnan(a) or a >= 0.5 else -1.0


def loro(lab, signals, level):
    """level='seed': 같은 설정의 다른 seed로 기준값을 정함 / 'config': 다른 설정(방법·학습률)으로 정함."""
    rows, thr_of = [], {}
    for run, test in lab.groupby("run"):
        same_cfg = ((lab.model == test.model.iloc[0]) & (lab.method == test.method.iloc[0])
                    & np.isclose(lab.lr, test.lr.iloc[0]))
        train = lab[same_cfg & (lab.run != run)] if level == "seed" else lab[~same_cfg]
        rest = lab[lab.run != run]  # 방향은 시험 실행을 뺀 전체에서 정한다
        if (train.harm == 0).sum() < 3:
            continue  # 기준값을 정할 이득 구간이 없음 (예: seed가 하나뿐)
        for s in signals:
            dr = direction_of(rest, s)
            thr = fit_threshold(train, s, dr)
            thr_of[(run, s)] = (dr, thr)
            sc = dr * test[s].values
            rows.append(dict(level=level, run=run, signal=s, direction=dr, threshold=dr * thr,
                             n_harm=int((test.harm == 1).sum()), caught=int(((sc > thr) & (test.harm == 1)).sum()),
                             n_gain=int((test.harm == 0).sum()), false_alarm=int(((sc > thr) & (test.harm == 0)).sum())))
    return pd.DataFrame(rows), thr_of


def step3(df, signals, out):
    lab = df[df.harm >= 0]
    res, thr_cfg = [], None
    for level in ["seed", "config"]:
        t, thr_of = loro(lab, signals, level)
        name = "같은 설정의 다른 seed" if level == "seed" else "다른 모델·방법·학습률"
        if level == "config":
            thr_cfg = thr_of
        if t.empty:
            print(f"\n3단계: {name}에서 기준값을 정할 실행이 없어 건너뜀")
            continue
        res.append(t)
        g = t.groupby("signal")[["caught", "n_harm", "false_alarm", "n_gain"]].sum()
        g["recall"] = g.caught / g.n_harm.replace(0, np.nan)
        g["false_alarm_rate"] = g.false_alarm / g.n_gain.replace(0, np.nan)
        g["passed"] = (g.recall >= 0.6) & (g.false_alarm_rate <= 0.1)
        show(f"3단계: {name}에서 정한 기준값을 그대로 적용 (기준값 = 이득 구간 오경보 5%)",
             g.reset_index()[["signal", "recall", "false_alarm_rate", "passed"]])
    if res:
        pd.concat(res).to_csv(os.path.join(out, "followup_step3.csv"), index=False)
    print("3단계 통과 기준: 손해 구간 60% 이상을 잡고 이득 구간 오경보 10% 이하, 두 수준 모두")
    return thr_cfg


# ---------------------------------------------------------------- 4단계

def rolling(r, ids, bs, win):
    """도메인 스트림에서 배치가 끝날 때마다 직전 win장의 정확도 차이와 신호."""
    rows = []
    for end in range(bs, len(ids) + 1, bs):
        if end < win:
            continue
        rows.append(dict(batch=end // bs - 1, **window(r, ids[end - win:end])))
    if len(ids) % bs and len(ids) >= win:
        rows.append(dict(batch=len(ids) // bs, **window(r, ids[len(ids) - win:])))
    return pd.DataFrame(rows)


def runs_of(mask):
    """True가 이어지는 구간들의 (시작, 끝) 위치."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def step4(runs, signals, thr_cfg, seg, out, lookback=24, min_len=4):
    ev_rows, pol_rows, on_rows = [], [], []
    for run in runs:
        m, r, name = run["meta"], run["r"], run["name"]
        bs = m["bs"]
        for di, dom in enumerate(m["domains"]):
            ids = np.where(r["dom"] == di)[0]
            roll = rolling(r, ids, bs, seg)
            if roll.empty:
                continue
            events = [(a, b) for a, b in runs_of((roll.delta < HARM).values) if b - a + 1 >= min_len]
            in_event = np.zeros(len(roll), bool)
            for a, b in events:
                in_event[max(0, a - lookback):b + 1] = True
            y, pt, pz = r["y"][ids], r["pred_t"][ids], r["pred_z"][ids]
            for s in signals:
                if (name, s) not in thr_cfg:
                    continue
                dr, thr = thr_cfg[(name, s)]
                alarm = (dr * roll[s].values > thr)
                for k, (a, b) in enumerate(events):
                    lo = max(0, a - lookback)
                    pre = lo + np.where(alarm[lo:a])[0]  # 사건 직전 lookback배치 안의 경보 위치
                    during = np.where(alarm[a:b + 1])[0]
                    ev_rows.append(dict(run=name, domain=dom, signal=s, event=k, start=int(roll.batch[a]),
                                        length=b - a + 1, early=len(pre) > 0,
                                        lead_batches=a - pre[0] if len(pre) else np.nan,
                                        late=len(pre) == 0 and len(during) > 0))
                on_rows.append(dict(run=name, domain=dom, signal=s,
                                    alarm_on_outside_events=alarm[~in_event].mean() if (~in_event).any() else np.nan))
                # 경보가 처음 울린 다음 배치부터 zero-shot 답을 쓴다
                first = np.where(alarm)[0]
                cut = len(ids) if len(first) == 0 else min(len(ids), (int(roll.batch[first[0]]) + 1) * bs)
                pred_switch = np.concatenate([pt[:cut], pz[cut:]])
                # 경보가 켜져 있는 동안만 zero-shot 답을 쓴다 (TTA는 뒤에서 계속 적응)
                on_batches = set(roll.batch[alarm].astype(int) + 1)
                use_z = np.array([(i // bs) in on_batches for i in range(len(ids))])
                pred_follow = np.where(use_z, pz, pt)
                pol_rows.append(dict(run=name, method=m["method"], domain=dom, signal=s, n=len(ids),
                                     acc_tta=(pt == y).mean(), acc_zs=(pz == y).mean(),
                                     acc_switch=(pred_switch == y).mean(), acc_follow=(pred_follow == y).mean()))
    ev, pol, on = pd.DataFrame(ev_rows), pd.DataFrame(pol_rows), pd.DataFrame(on_rows)
    ev.to_csv(os.path.join(out, "followup_step4_events.csv"), index=False)
    pol.to_csv(os.path.join(out, "followup_step4_policy.csv"), index=False)
    n_ev = ev.groupby(["run", "domain", "event"]).ngroups if len(ev) else 0
    print(f"\n4단계: 손해 사건 {n_ev}개 (512장 이동 창의 차이가 -2%p 아래로 {min_len}배치 이상 이어진 것)")
    if len(ev):
        g = ev.groupby("signal").agg(events=("early", "size"), early=("early", "mean"), late=("late", "mean"),
                                     median_lead_batches=("lead_batches", "median")).reset_index()
        g = g.merge(on.groupby("signal").alarm_on_outside_events.mean().reset_index(), on="signal")
        show(f"4단계: 사건 시작 전 {lookback}배치 안에 경보가 울린 비율 (early), 사건 중에야 울린 비율 (late), "
             "사건 밖에서 경보가 켜져 있던 시간 비율", g)
    if len(pol):
        w = pol.assign(**{c: pol[c] * pol.n for c in ["acc_tta", "acc_zs", "acc_switch", "acc_follow"]})
        g = w.groupby("signal")[["acc_tta", "acc_zs", "acc_switch", "acc_follow", "n"]].sum()
        for c in ["acc_tta", "acc_zs", "acc_switch", "acc_follow"]:
            g[c] = 100 * g[c] / g.n
        show("4단계: 경보에 따라 조치했을 때 전체 정확도 (%). switch: 첫 경보 후 zero-shot으로 전환, "
             "follow: 경보가 켜진 동안만 zero-shot", g.drop(columns="n").reset_index())
    print("4단계 통과 기준: 사건의 절반 넘게 early, 그리고 조치한 정확도가 acc_tta보다 높음")
    print("주의: TTA 초기화는 다시 돌려야 알 수 있어서 여기서는 zero-shot 전환만 시뮬레이션한다")


# ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs_followup")
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--force", action="store_true", help="1단계를 통과하지 못해도 2~4단계를 돌린다")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    runs = load(args.out)
    if not runs:
        raise SystemExit(f"{args.out}에 npz가 없다. sweep.py를 먼저 돌린다")
    has_rank = all("rank_z_at_t" in x["r"] for x in runs)
    if not has_rank:
        print("주의: 순위 기록(rank_z_at_t)이 없는 실행이 있어 상위 K 비율은 빼고 분석한다")
    signals = signal_names(has_rank)
    df = segments(runs, args.seg)
    df.to_csv(os.path.join(args.out, "followup_segments.csv"), index=False)
    pd.set_option("display.width", 220)
    print(f"실행 {len(runs)}개, 구간 {len(df)}개 (손해 {(df.harm == 1).sum()}, 이득 {(df.harm == 0).sum()})")

    ok = step1(df, args.out)
    if not ok and not args.force:
        return
    step2(df, signals, args.out)
    thr_cfg = step3(df, signals, args.out)
    step4(runs, signals, thr_cfg, args.seg, args.out)


if __name__ == "__main__":
    main()
