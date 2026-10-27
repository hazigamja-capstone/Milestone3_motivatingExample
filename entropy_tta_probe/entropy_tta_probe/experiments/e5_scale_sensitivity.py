"""
E5 — Entropy's absolute scale depends on things unrelated to correctness:
     logit scale (temperature), number of classes K, and the prompt template.

(a) temperature  : logit scale in {--scales}; per dataset mean H, H/log K, and what a FIXED threshold
                   (EATA-style H < 0.4·ln 1000) would select: selection rate and precision of the selected set.
(b) #classes     : random class subsets of --k_source (default ImageNet) with K in {--ks}, 5 seeds;
                   mean H, mean H/log K, accuracy, AUROC, fixed-threshold selection rate.
(c) templates    : 'a photo of a {}.', '{}', CoOp's 7 templates, and their ensemble;
                   accuracy, mean H, AUROC, Spearman rank correlation of per-sample entropy vs. the default
                   template and Jaccard overlap of the 10% lowest-entropy samples
                   (= is "which samples are confident" a property of the image or of the prompt?).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from etta.common import base_args, get_features, load_bundle, out_dir  # noqa: E402
from etta.data import expand_names  # noqa: E402
from etta.metrics import auroc, np_entropy, np_softmax  # noqa: E402
from etta.model import IMAGENET_TEMPLATES_SELECT  # noqa: E402
from etta.plot import plt, save  # noqa: E402

FIXED_H = 0.4 * np.log(1000)


def spearman(a, b):
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def low_set(H, q=0.1):
    return set(np.flatnonzero(H <= np.quantile(H, q)))


if __name__ == "__main__":
    ap = base_args(__doc__)
    ap.add_argument("--scales", type=float, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 5, 10, 50, 100, 200, 500, 1000])
    ap.add_argument("--k_source", default="imagenet", help="dataset for the #classes sweep ('none' to skip)")
    ap.add_argument("--k_seeds", type=int, default=5)
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()
    bundle = load_bundle(args)
    od = out_dir(args, "e5")
    dev = bundle.device
    names = expand_names(args.datasets)
    rows_t, rows_tpl = [], []
    for n in names:
        d = get_features(args, bundle, n, args.max_samples)
        f, y = d["feats"].float().to(dev), d["labels"].numpy()
        cos = (f @ bundle.text_features(d["classnames"]).t()).cpu().numpy()
        K = cos.shape[1]
        # (a) temperature
        for sc in args.scales:
            p = np_softmax(sc * cos)
            H = np_entropy(p)
            corr = p.argmax(1) == y
            selm = H < FIXED_H
            rows_t.append(dict(dataset=n, K=K, scale=sc, acc=corr.mean(), mean_H=H.mean(),
                               mean_H_norm=H.mean() / np.log(K), auroc=auroc(-H, corr.astype(float)),
                               fixed_thr_select_rate=selm.mean(),
                               fixed_thr_precision=corr[selm].mean() if selm.any() else np.nan))
        # (c) templates
        tpls = {"a photo of a {}.": ["a photo of a {}."], "{}": ["{}"]}
        tpls.update({t: [t] for t in IMAGENET_TEMPLATES_SELECT})
        tpls["ensemble(7)"] = IMAGENET_TEMPLATES_SELECT
        Hs = {}
        for tn, tl in tpls.items():
            lg = bundle.logit_scale * (f @ bundle.text_features(d["classnames"], tl).t()).cpu().numpy()
            p = np_softmax(lg)
            H = np_entropy(p)
            Hs[tn] = H
            corr = p.argmax(1) == y
            rows_tpl.append(dict(dataset=n, template=tn, acc=corr.mean(), mean_H=H.mean(),
                                 auroc=auroc(-H, corr.astype(float)),
                                 fixed_thr_select_rate=(H < FIXED_H).mean()))
        ref, ref_set = Hs["a photo of a {}."], low_set(Hs["a photo of a {}."])
        for r in rows_tpl[-len(tpls):]:
            Ht = Hs[r["template"]]
            r["spearman_vs_default"] = spearman(ref, Ht)
            s_ = low_set(Ht)
            r["jaccard_low10_vs_default"] = len(s_ & ref_set) / max(1, len(s_ | ref_set))
        print(f"{n}: done")

    dt, dtp = pd.DataFrame(rows_t), pd.DataFrame(rows_tpl)
    dt.to_csv(os.path.join(od, "temperature.csv"), index=False)
    dtp.to_csv(os.path.join(od, "templates.csv"), index=False)

    # (b) number of classes
    rows_k = []
    if args.k_source.lower() != "none":
        d = get_features(args, bundle, args.k_source, args.max_samples)
        f, y = d["feats"].float().to(dev), d["labels"].numpy()
        cos_full = (f @ bundle.text_features(d["classnames"]).t()).cpu().numpy()
        Kfull = cos_full.shape[1]
        for K in [k for k in args.ks if k <= Kfull]:
            for sd in range(args.k_seeds):
                rng = np.random.default_rng(sd)
                cls = np.sort(rng.choice(Kfull, K, replace=False))
                m = np.isin(y, cls)
                remap = {c: i for i, c in enumerate(cls)}
                yy = np.array([remap[c] for c in y[m]])
                p = np_softmax(bundle.logit_scale * cos_full[m][:, cls])
                H = np_entropy(p)
                corr = p.argmax(1) == yy
                rows_k.append(dict(K=K, seed=sd, n=int(m.sum()), acc=corr.mean(), mean_H=H.mean(),
                                   mean_H_norm=H.mean() / np.log(K), auroc=auroc(-H, corr.astype(float)),
                                   fixed_thr_select_rate=(H < FIXED_H).mean(),
                                   fixed_thr_precision=corr[H < FIXED_H].mean() if (H < FIXED_H).any() else np.nan))
    dk = pd.DataFrame(rows_k)
    dk.to_csv(os.path.join(od, "num_classes.csv"), index=False)

    print(dt[dt.scale == 100][["dataset", "K", "acc", "mean_H", "mean_H_norm", "fixed_thr_select_rate",
                               "fixed_thr_precision"]].round(3).to_string(index=False))
    print(dtp.groupby("template")[["acc", "mean_H", "auroc", "spearman_vs_default",
                                   "jaccard_low10_vs_default"]].mean().round(3))

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for n_, g in dt.groupby("dataset"):
        ax[0].plot(g.scale, g.fixed_thr_select_rate, marker=".", label=n_)
    ax[0].set_xscale("log"); ax[0].set_xlabel("logit scale"); ax[0].set_ylabel(f"rate H < {FIXED_H:.2f}")
    ax[0].legend(fontsize=6, ncol=2); ax[0].set_title("same threshold, different datasets")
    if len(dk):
        gk = dk.groupby("K").mean(numeric_only=True)
        ax[1].plot(gk.index, gk.mean_H, "o-", label="mean H")
        ax[1].plot(gk.index, gk.mean_H_norm, "s-", label="mean H / log K")
        ax[1].plot(gk.index, gk.acc, "^-", label="acc")
        ax[1].set_xscale("log"); ax[1].set_xlabel(f"#classes K ({args.k_source} subsets)"); ax[1].legend()
    gt = dtp.groupby("template")[["spearman_vs_default", "jaccard_low10_vs_default"]].mean()
    ax[2].barh(range(len(gt)), gt.jaccard_low10_vs_default, label="Jaccard of lowest-10% H set")
    ax[2].set_yticks(range(len(gt))); ax[2].set_yticklabels(gt.index, fontsize=6)
    ax[2].set_title("agreement with 'a photo of a {}.'")
    save(fig, os.path.join(od, "scale_sensitivity.png"))
    print(f"saved -> {od}")
