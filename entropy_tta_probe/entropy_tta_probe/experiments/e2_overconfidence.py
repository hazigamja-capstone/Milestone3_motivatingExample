"""
E2 — Entropy minimisation buys (a little) accuracy with (a lot of) over-confidence.
Reads the npz files of run_episodic.py.

Per dataset, for zero-shot / TPT after 1..S steps / confident-view ensemble (no tuning):
  acc, ECE(15 bins), NLL, mean confidence, over-confidence gap (conf - acc), AURC
Prediction flips zero-shot -> TPT(1 step):
  W->W with confidence increase  = confirmation bias (wrong answer made more certain)
  R->W                           = entropy objective pulled a correct answer away
  W->R                           = the gain TPT is credited for
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from etta.common import base_args, out_dir  # noqa: E402
from etta.metrics import aurc, ece, nll, np_softmax, reliability  # noqa: E402
from etta.plot import plt, save  # noqa: E402


def calib(prob, labels):
    conf, pred = prob.max(1), prob.argmax(1)
    corr = (pred == labels).astype(float)
    return dict(acc=corr.mean(), ece=ece(conf, corr), nll=nll(prob, labels),
                mean_conf=conf.mean(), overconf_gap=conf.mean() - corr.mean(), aurc=aurc(conf, corr)), conf, corr


if __name__ == "__main__":
    ap = base_args(__doc__)
    args = ap.parse_args()
    src = out_dir(args, "episodic")
    od = out_dir(args, "e2")
    rows, flips = [], []
    pooled = {"zero-shot": ([], []), "TPT (1 step)": ([], [])}
    files = sorted(glob.glob(os.path.join(src, "*.npz")))
    if not files:
        sys.exit(f"no npz in {src}; run experiments/run_episodic.py first")
    for fpath in files:
        r = np.load(fpath)
        meta = json.load(open(fpath.replace(".npz", ".json")))
        name, y = meta["dataset"], r["labels"]
        zs = np_softmax(r["zs_logits"].astype(np.float32))
        variants = {"zero-shot": zs}
        for k in range(r["tpt_logits"].shape[0]):
            variants[f"TPT ({k + 1} step{'s' if k else ''})"] = np_softmax(r["tpt_logits"][k].astype(np.float32))
        variants["conf-view ensemble (no tuning)"] = r["ens_sel_prob"].astype(np.float32)
        variants["all-view ensemble (no tuning)"] = r["ens_all_prob"].astype(np.float32)
        for vname, prob in variants.items():
            m, conf, corr = calib(prob, y)
            rows.append(dict(dataset=name, method=vname, **m))
            if vname in pooled:
                pooled[vname][0].append(conf); pooled[vname][1].append(corr)

        tp = variants["TPT (1 step)"]
        zc, tc = zs.argmax(1) == y, tp.argmax(1) == y
        dconf = tp.max(1) - zs.max(1)
        ww = ~zc & ~tc
        flips.append(dict(
            dataset=name, n=len(y),
            R_to_W=(zc & ~tc).mean(), W_to_R=(~zc & tc).mean(), W_to_W=ww.mean(), R_to_R=(zc & tc).mean(),
            W_to_W_conf_up=(ww & (dconf > 0)).sum() / max(1, ww.sum()),
            mean_dconf_W_to_W=dconf[ww].mean() if ww.any() else np.nan,
            mean_dconf_R_to_R=dconf[zc & tc].mean() if (zc & tc).any() else np.nan,
            # confidently wrong: wrong predictions with confidence > 0.9 before/after
            conf_wrong_gt90_zs=((~zc) & (zs.max(1) > 0.9)).mean(),
            conf_wrong_gt90_tpt=((~tc) & (tp.max(1) > 0.9)).mean(),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(od, "calibration.csv"), index=False)
    fl = pd.DataFrame(flips)
    fl.to_csv(os.path.join(od, "flips.csv"), index=False)
    piv = df.pivot_table(index="dataset", columns="method", values=["acc", "ece"])
    piv.to_csv(os.path.join(od, "acc_ece_pivot.csv"))
    print(df.groupby("method")[["acc", "ece", "nll", "mean_conf", "overconf_gap"]].mean().round(4))
    print(fl.round(4).to_string(index=False))

    # reliability diagram (pooled over datasets)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for vname, (cs, ks) in pooled.items():
        rel = reliability(np.concatenate(cs), np.concatenate(ks))
        xs = [r_[1] for r_ in rel if r_[3] > 0]; ys = [r_[2] for r_ in rel if r_[3] > 0]
        ax[0].plot(xs, ys, "o-", ms=3, label=vname)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set_xlabel("confidence"); ax[0].set_ylabel("accuracy")
    ax[0].set_title("reliability (pooled)"); ax[0].legend()
    steps = df[df.method.str.startswith("TPT") | (df.method == "zero-shot")]
    order = ["zero-shot"] + sorted([m for m in steps.method.unique() if m.startswith("TPT")])
    g = steps.groupby("method")[["acc", "ece", "mean_conf"]].mean().loc[order]
    x = range(len(order))
    ax[1].plot(x, g.acc, "o-", label="acc"); ax[1].plot(x, g.mean_conf, "s-", label="mean conf")
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels(["ZS"] + [f"{i}" for i in range(1, len(order))])
    ax[1].set_xlabel("TPT steps"); ax[1].legend(); ax[1].set_title("acc vs confidence")
    ax[2].plot(x, g.ece, "o-", color="C3"); ax[2].set_xticks(list(x))
    ax[2].set_xticklabels(["ZS"] + [f"{i}" for i in range(1, len(order))]); ax[2].set_title("ECE vs TPT steps")
    save(fig, os.path.join(od, "overconfidence.png"))
    print(f"saved -> {od}")
