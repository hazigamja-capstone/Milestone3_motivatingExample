"""실험 #3 판정 (PREREG.md 4절 기준을 그대로 코드로 옮김).

입력: run_paper.py 결과 폴더들 (outputs_paper/{method}_{arch}_{setting}_s{seed}/dom*.npz)
출력: 손상별 오류율, 512장 구간 표, 손해 사건 목록과 유형, (실행이 2개 이상이면) 교차 기준 경보 시점.

예:
  python paper_repro/analyze_paper.py outputs_paper/eta_ViT-L-14_correlated_s1
  python paper_repro/analyze_paper.py "outputs_paper/eta_*"     # 여러 실행: 사건 빈도 + 교차 기준 경보
  python paper_repro/analyze_paper.py RUN --exclude_seed 1       # 조합을 고른 1차 실행은 빈도에서 뺀다

구간은 손상 경계와 상관없이 실행 전체 스트림을 이어서 512장씩 자른다 (continual이라 시간축이 하나).
마지막 512장 미만 조각은 버린다.

사건 유형 (주 기준, 정답 라벨 대비): 구간 TTA 최빈 비율 / 구간 정답 라벨 최빈 비율
  ≥ 4배 붕괴, ≤ 1.5배 조용한 손해, 그 사이는 별도. 라벨은 평가에만 쓴다.
  zero-shot 대비 기준은 쓰지 않는다 (zero-shot으로 유형을 나누고 zero-shot 신호를 평가하면 순환).
  민감도 분석: absolute 원안 (≥ 50% 붕괴, < 10% 조용한 손해)도 함께 보고한다.

교차 기준 경보: 다른 실행의 비손해 구간(Δ > -5%p, 이득·노이즈 구간)으로 기준값을 정한다.
  오경보율 주 기준 5%, 보조 1%·10%. 보정용 비손해 구간이 100개 미만인 실행이 있으면 주 기준을 10%로 올린다.
  이어진 구간은 독립이 아니므로 구간 단위 오경보율과 함께 오경보 묶음 수(연속 경보 덩어리)도 센다.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

SEG = 512
HARM_DELTA = -5.0      # %p, 이하이면 손해 구간
HARM_RUN = 3           # 연속 구간 수
TOPK = 5               # 상위 K 비율의 K
COLLAPSE_X, SILENT_X = 4.0, 1.5   # 정답 라벨 최빈 비율 대비 배수
FA_RATES = [0.01, 0.05, 0.10]
FA_PRIMARY, FA_FALLBACK, MIN_CALIB = 0.05, 0.10, 100
SIGNALS = ["ent_t", "maxp_t", "mode_t", "disagree", "kl", "topk", "seg_idx"]
BASELINES = ["ent_t", "seg_idx"]   # 신호 통과 기준: 이 둘보다 먼저 반응해야 한다
TYPES = ["붕괴", "조용한 손해", "그 사이"]


# ---------------------------------------------------------------- 불러오기

def load_run(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "dom*.npz")))
    if not files:
        raise SystemExit(f"{run_dir}: dom*.npz 없음")
    parts, doms = [], []
    for f in files:
        d = dict(np.load(f))
        dom = re.match(r"dom\d+_(.+)\.npz", os.path.basename(f)).group(1)
        d["dom"] = np.full(len(d["y"]), len(doms))
        doms.append(dom)
        parts.append(d)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    meta = {}
    if os.path.exists(os.path.join(run_dir, "meta.json")):
        with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
    return data, doms, meta


def mode_ratio(a):
    return np.bincount(a).max() / len(a)


def segments(d):
    rows = []
    for i in range(len(d["y"]) // SEG):
        s = slice(i * SEG, (i + 1) * SEG)
        y, pt, pz = d["y"][s], d["pred_t"][s], d["pred_z"][s]
        acc_t, acc_z = 100 * (pt == y).mean(), 100 * (pz == y).mean()
        mt, my = mode_ratio(pt), mode_ratio(y)
        rows.append(dict(
            seg_idx=i, dom=int(np.bincount(d["dom"][s]).argmax()),
            acc_t=acc_t, acc_z=acc_z, delta=acc_t - acc_z,
            ent_t=d["ent_t"][s].mean(), maxp_t=d["maxp_t"][s].mean(),
            mode_t=mt, mode_y=my, mode_lab=mt / my, mode_z=mode_ratio(pz),
            disagree=(pt != pz).mean(), kl=d["kl_tz"][s].mean(),
            topk=(d["rank_z_at_t"][s] < TOPK).mean() if "rank_z_at_t" in d else np.nan,
        ))
    return rows


# ---------------------------------------------------------------- 사건

def classify_label(ev):
    m = ev["mode_lab"]
    return "붕괴" if m >= COLLAPSE_X else "조용한 손해" if m <= SILENT_X else "그 사이"


def classify_abs(ev):
    m = ev["mode_t"]
    return "붕괴" if m >= 0.5 else "조용한 손해" if m < 0.10 else "그 사이"


def find_events(rows):
    harm = [r["delta"] <= HARM_DELTA for r in rows]
    events, i = [], 0
    while i < len(rows):
        if not harm[i]:
            i += 1
            continue
        j = i
        while j < len(rows) and harm[j]:
            j += 1
        if j - i >= HARM_RUN:
            seg = rows[i:j]
            ev = dict(start=i, end=j - 1, n_seg=j - i, doms=sorted({r["dom"] for r in seg}),
                      delta_mean=np.mean([r["delta"] for r in seg]), delta_min=min(r["delta"] for r in seg),
                      **{k: np.mean([r[k] for r in seg]) for k in ["mode_t", "mode_y", "mode_lab", "mode_z"]})
            ev["type"], ev["type_abs"] = classify_label(ev), classify_abs(ev)
            events.append(ev)
        i = j
    return events


# ---------------------------------------------------------------- 교차 기준 경보

def calib_rows(rows):
    return [r for r in rows if r["delta"] > HARM_DELTA]


def thresholds(ref_rows, fa):
    """기준 실행의 비손해 구간 분포로 기준값을 정한다 (양쪽 꼬리, 구간 번호만 위쪽 꼬리)."""
    ok = calib_rows(ref_rows)
    th = {}
    for s in SIGNALS:
        v = np.array([r[s] for r in ok], dtype=float)
        v = v[~np.isnan(v)]
        if not len(v):
            th[s] = None
        elif s == "seg_idx":  # 시간은 한쪽 방향: "이 시점 이후면 경보"
            th[s] = (-np.inf, np.quantile(v, 1 - fa))
        else:
            th[s] = (np.quantile(v, fa / 2), np.quantile(v, 1 - fa / 2))
    return th


def alarms(rows, th, s):
    if th[s] is None:
        return None
    lo, hi = th[s]
    return [not (lo <= r[s] <= hi) for r in rows]


def first_alarm(on, ev):
    """사건 시작 기준 첫 경보 구간의 상대 위치 (음수 = 사건 전에 울림). 사건 끝까지 안 울리면 None.
    사건 안에서 처음 켜진 경보가 사건 전부터 이어져 왔으면 그 연속 덩어리의 시작점을 잡는다."""
    if on is None:
        return None
    for k in range(ev["start"], ev["end"] + 1):
        if on[k]:
            a = k
            while a - 1 >= 0 and on[a - 1]:
                a -= 1
            return a - ev["start"]
    return None


def false_alarms(rows, on):
    """대상 실행의 비손해 구간에서 실제 오경보율과 오경보 묶음 수 (연속으로 켜진 비손해 구간 덩어리)."""
    if on is None:
        return None, None
    ok = [r["delta"] > HARM_DELTA for r in rows]
    fa = [o and k for o, k in zip(on, ok)]
    n_ok = sum(ok)
    clusters = sum(1 for i, f in enumerate(fa) if f and (i == 0 or not fa[i - 1]))
    return (sum(fa) / n_ok if n_ok else np.nan), clusters


def cross_alarm(all_rows, all_events, names, fa, primary):
    tag = " (주 기준)" if primary else " (보조)"
    print(f"\n----- 오경보율 {fa:.0%}{tag}")
    print("사건 시작 대비 첫 경보 구간 (음수 = 미리 울림, - = 사건 끝까지 안 울림). 기준 실행이 여럿이면 중앙값")
    print("사건".ljust(40) + "".join(s.rjust(9) for s in SIGNALS))
    wins = {s: [0, 0] for s in SIGNALS}
    fa_stats = {s: [] for s in SIGNALS}
    for i, events in enumerate(all_events):
        refs = [thresholds(all_rows[j], fa) for j in range(len(all_rows)) if j != i]
        ons = {s: [alarms(all_rows[i], th, s) for th in refs] for s in SIGNALS}
        for s in SIGNALS:
            for on in ons[s]:
                r, c = false_alarms(all_rows[i], on)
                if r is not None:
                    fa_stats[s].append((r, c))
        for e in events:
            lead = {}
            for s in SIGNALS:
                v = [x for x in (first_alarm(on, e) for on in ons[s]) if x is not None]
                lead[s] = int(np.median(v)) if v else None
            label = f"{names[i][:24]} {e['start']}~{e['end']} {e['type']}"
            print(label.ljust(40) + "".join(("  -  " if lead[s] is None else f"{lead[s]:+4d}").rjust(9) for s in SIGNALS))
            for s in SIGNALS:
                if s in BASELINES:
                    continue
                wins[s][1] += 1
                if lead[s] is not None and all(lead[b] is None or lead[s] < lead[b] for b in BASELINES):
                    wins[s][0] += 1
    print("실제 오경보 (대상 실행 비손해 구간, 실행×기준 조합 평균): 구간 비율 / 묶음 수")
    print("  " + "  ".join(f"{s} {np.mean([x[0] for x in v]):.3f}/{np.mean([x[1] for x in v]):.1f}"
                           for s, v in fa_stats.items() if v))
    print("신호 통과 (엔트로피와 구간 번호보다 둘 다 먼저 반응한 사건 비율):")
    print("  " + "  ".join(f"{s} {w[0]}/{w[1]}" for s, w in wins.items() if s not in BASELINES and w[1]))


# ---------------------------------------------------------------- 출력

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--exclude_seed", type=int, action="append", default=[],
                    help="빈도에서 뺄 1차 실행 seed (조합 선택에 쓴 실행)")
    ap.add_argument("--csv", default="", help="구간 표 저장 폴더")
    args = ap.parse_args()

    runs = [rd for r in args.runs for rd in (sorted(glob.glob(r)) or [r]) if os.path.isdir(rd)]
    all_rows, all_events, names, seeds = [], [], [], []
    for rd in runs:
        d, doms, meta = load_run(rd)
        rows = segments(d)
        name = os.path.basename(os.path.normpath(rd))
        print(f"\n===== {name}  (seed {meta.get('seed')}, lr {meta.get('lr')}, 배치 {meta.get('bs')}, n={len(d['y'])})")
        print("손상별 오류율 (%)   TTA    zs    차이")
        for i, dom in enumerate(doms):
            m = d["dom"] == i
            et, ez = 100 - 100 * (d["pred_t"][m] == d["y"][m]).mean(), 100 - 100 * (d["pred_z"][m] == d["y"][m]).mean()
            print(f"  {dom:<18} {et:5.1f} {ez:5.1f} {et - ez:+6.1f}")
        et, ez = 100 - 100 * (d["pred_t"] == d["y"]).mean(), 100 - 100 * (d["pred_z"] == d["y"]).mean()
        print(f"  {'전체':<18} {et:5.1f} {ez:5.1f} {et - ez:+6.1f}")
        if not rows:
            print(f"구간 없음 ({SEG}장 미만)")
            continue
        events = find_events(rows)
        names.append(name)
        seeds.append(meta.get("seed"))
        all_rows.append(rows)
        all_events.append(events)
        my = np.array([r["mode_y"] for r in rows])
        print(f"정답 라벨 구간 최빈 비율: 중앙값 {np.median(my):.3f} (범위 {my.min():.3f}~{my.max():.3f})")
        print(f"손해 구간 (Δ ≤ {HARM_DELTA}%p): {sum(r['delta'] <= HARM_DELTA for r in rows)} / {len(rows)}, "
              f"보정용 비손해 구간 {len(calib_rows(rows))}개")
        print(f"손해 사건 ({HARM_RUN}구간 이상 연속): {len(events)}건")
        for e in events:
            print(f"  구간 {e['start']}~{e['end']} ({e['n_seg']}개, 손상 {[doms[k] for k in e['doms']]}) "
                  f"Δ 평균 {e['delta_mean']:+.1f} 최저 {e['delta_min']:+.1f}  "
                  f"최빈 TTA {e['mode_t']:.3f} / 라벨 {e['mode_y']:.3f} (×{e['mode_lab']:.1f})  "
                  f"→ {e['type']}  [absolute: {e['type_abs']}]")
        if args.csv:
            import csv
            os.makedirs(args.csv, exist_ok=True)
            with open(os.path.join(args.csv, f"{name}_segments.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    # 빈도 / 재현 판정
    fresh = [i for i, s in enumerate(seeds) if s not in args.exclude_seed]
    print("\n===== 재현 판정 (PREREG: 새 seed 3개 중 2개 이상에서 같은 유형의 사건)")
    print(f"빈도에 넣는 실행: {[names[i] for i in fresh]}  (뺀 seed: {args.exclude_seed or '없음'})")
    for key, label in [("type", "주 기준 (정답 라벨 대비)"), ("type_abs", "민감도 (absolute)")]:
        print(f" {label}")
        for t in TYPES:
            k = sum(any(e[key] == t for e in all_events[i]) for i in fresh)
            verdict = "판정 불가 (새 seed 3개 미만)" if len(fresh) < 3 else ("통과" if k >= 2 else "실패")
            print(f"  {t:<8}: {k} / {len(fresh)} 실행  → {verdict}")

    # 교차 기준 경보
    if len(all_rows) < 2:
        print("\n교차 기준 경보: 실행이 2개 이상 있어야 한다 (기준값을 다른 실행에서 정함)")
        return
    n_calib = min(len(calib_rows(r)) for r in all_rows)
    primary = FA_PRIMARY if n_calib >= MIN_CALIB else FA_FALLBACK
    print(f"\n===== 교차 기준 경보 (보정용 비손해 구간 최소 {n_calib}개 → 주 기준 {primary:.0%}"
          f"{'' if primary == FA_PRIMARY else f', {MIN_CALIB}개 미만이라 올림'})")
    for fa in FA_RATES:
        cross_alarm(all_rows, all_events, names, fa, fa == primary)


if __name__ == "__main__":
    main()
