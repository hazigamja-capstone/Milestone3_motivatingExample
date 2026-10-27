"""
E3 — What does confidence-based view selection actually select, and where does TPT's gain come from?
Reads the npz files of run_episodic.py.

  * accuracy of individual views per within-sample entropy rank (rank 1 = most confident view)
  * crop area of selected vs. non-selected views (do low-entropy views tend to be small crops?)
  * unanimous-wrong rate: all selected views agree on the same wrong class
  * harm cases: original view correct but the selected-view ensemble wrong (and vice versa)
  * accuracy of: original view | all-view mean | selected-view mean | ZERO vote | TPT
    and the agreement of TPT with ZERO (if ~1, the entropy gradient adds little beyond view selection)
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from etta.common import base_args, out_dir  # noqa: E402
from etta.plot import plt, save  # noqa: E402

N_RANK_BINS = 8

if __name__ == "__main__":
    args = base_args(__doc__).parse_args()
    src, od = out_dir(args, "episodic"), out_dir(args, "e3")
    files = sorted(glob.glob(os.path.join(src, "*.npz")))
    if not files:
        sys.exit(f"no npz in {src}; run experiments/run_episodic.py first")
    rows, rank_curves, areas_sel, areas_rest = [], {}, [], []
    for fpath in files:
        r = np.load(fpath)
        name = json.load(open(fpath.replace(".npz", ".json")))["dataset"]
        y = r["labels"]
        vp, H, area, sel = r["view_pred"], r["view_entropy"], r["view_area"], r["selected"]
        vc = vp == y[:, None]
        N, V = vp.shape
        aug = np.arange(1, V)  # exclude the original view from view-level stats
        rank = H[:, aug].argsort(1).argsort(1)  # 0 = lowest entropy among augmented views
        b = (rank * N_RANK_BINS) // (V - 1)
        rank_curves[name] = [vc[:, aug][b == k].mean() for k in range(N_RANK_BINS)]
        a_sel, a_rest = area[:, aug][sel[:, aug]], area[:, aug][~sel[:, aug]]
        areas_sel.append(a_sel); areas_rest.append(a_rest)

        orig = vp[:, 0] == y
        ens_all = r["ens_all_prob"].argmax(1) == y
        ens_sel = r["ens_sel_prob"].argmax(1) == y
        zero = r["zero_pred"] == y
        tpt = r["tpt_logits"][0].argmax(1) == y
        sel_preds = np.where(sel, vp, -1)
        unanimous = np.array([len(set(row[row >= 0])) == 1 for row in sel_preds])
        unan_label = np.array([row[row >= 0][0] for row in sel_preds])
        rows.append(dict(
            dataset=name, n=N,
            acc_orig=orig.mean(), acc_all_views=ens_all.mean(), acc_selected_views=ens_sel.mean(),
            acc_zero_vote=zero.mean(), acc_tpt=tpt.mean(),
            tpt_agrees_with_zero=(r["tpt_logits"][0].argmax(1) == r["zero_pred"]).mean(),
            tpt_agrees_with_sel_ens=(r["tpt_logits"][0].argmax(1) == r["ens_sel_prob"].argmax(1)).mean(),
            view_acc_selected=vc[sel].mean(), view_acc_unselected=vc[~sel].mean(),
            view_acc_most_confident_bin=rank_curves[name][0], view_acc_least_confident_bin=rank_curves[name][-1],
            area_selected=a_sel.mean(), area_unselected=a_rest.mean(),
            orig_view_selected=sel[:, 0].mean(),
            unanimous_wrong=(unanimous & (unan_label != y)).mean(),
            harm_orig_right_sel_wrong=(orig & ~ens_sel).mean(),
            help_orig_wrong_sel_right=(~orig & ens_sel).mean(),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(od, "summary.csv"), index=False)
    print(df.round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for n, c in rank_curves.items():
        ax[0].plot(np.arange(1, N_RANK_BINS + 1), c, marker=".", label=n)
    ax[0].set_xlabel(f"entropy rank bin of augmented view (1 = lowest, {N_RANK_BINS} bins)")
    ax[0].set_ylabel("view accuracy"); ax[0].legend(fontsize=6, ncol=2)
    cols = ["acc_orig", "acc_all_views", "acc_selected_views", "acc_zero_vote", "acc_tpt"]
    m = df[cols].mean()
    ax[1].bar(range(len(cols)), m.values, color=["C7", "C0", "C1", "C2", "C3"])
    ax[1].set_xticks(range(len(cols))); ax[1].set_xticklabels(["orig", "all", "sel", "ZERO", "TPT"])
    ax[1].set_ylim(m.min() - 0.05, m.max() + 0.03); ax[1].set_title("mean acc over datasets")
    ax[2].hist(np.concatenate(areas_rest), bins=30, alpha=0.6, density=True, label="not selected")
    ax[2].hist(np.concatenate(areas_sel), bins=30, alpha=0.6, density=True, label="selected (low H)")
    ax[2].set_xlabel("crop area / image area"); ax[2].legend()
    save(fig, os.path.join(od, "view_selection.png"))
    print(f"saved -> {od}")
