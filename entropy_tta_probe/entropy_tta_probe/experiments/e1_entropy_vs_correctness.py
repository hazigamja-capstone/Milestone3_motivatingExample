"""
E1 — Low entropy ≠ correct prediction, and it gets worse as the shift grows.

For every split (zero-shot CLIP, 'a photo of a {}.'):
  * AUROC of (-entropy) for separating correct / wrong predictions (0.5 = entropy carries no signal)
  * error rate inside the 10% / 30% lowest-entropy samples (= what TPT/EATA-style selection would trust)
  * share of ALL errors that sit below the median entropy (errors entropy minimisation will reinforce)
  * accuracy per entropy decile (calibration of entropy as a correctness proxy)
With imagenet_c:* splits the same numbers are aggregated per severity -> trend with shift strength.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from etta.common import base_args, get_features, load_bundle, out_dir  # noqa: E402
from etta.data import expand_names  # noqa: E402
from etta.metrics import aurc, auroc, np_entropy, np_softmax  # noqa: E402
from etta.plot import plt, save  # noqa: E402


def analyse(name, logits, labels):
    p = np_softmax(logits)
    pred = p.argmax(1)
    correct = (pred == labels).astype(float)
    H = np_entropy(p)
    K = p.shape[1]
    q10, q30, q50 = np.quantile(H, [0.1, 0.3, 0.5])
    low10, low30 = H <= q10, H <= q30
    wrong = correct == 0
    dec = np.digitize(H, np.quantile(H, np.linspace(0, 1, 11)[1:-1]))
    row = dict(
        dataset=name, n=len(labels), num_classes=K, acc=correct.mean(),
        mean_entropy=H.mean(), mean_norm_entropy=H.mean() / np.log(K),
        auroc_negH_correct=auroc(-H, correct), aurc_negH=aurc(-H, correct),
        auroc_maxprob_correct=auroc(p.max(1), correct),
        err_in_lowest10pct_H=1 - correct[low10].mean(), err_in_lowest30pct_H=1 - correct[low30].mean(),
        share_of_errors_below_median_H=(wrong & (H <= q50)).sum() / max(1, wrong.sum()),
        mean_conf_of_errors=p.max(1)[wrong].mean() if wrong.any() else np.nan,
    )
    deciles = [correct[dec == d].mean() if (dec == d).any() else np.nan for d in range(10)]
    return row, deciles


if __name__ == "__main__":
    ap = base_args(__doc__)
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()
    bundle = load_bundle(args)
    od = out_dir(args, "e1")
    rows, dec_curves = [], {}
    for name in expand_names(args.datasets):
        d = get_features(args, bundle, name, args.max_samples)
        T = bundle.text_features(d["classnames"])
        logits = bundle.logit_scale * d["feats"].float().to(bundle.device) @ T.t()
        row, dec = analyse(name, logits.cpu().numpy(), d["labels"].numpy())
        rows.append(row)
        dec_curves[name] = dec
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(od, "summary.csv"), index=False)

    # severity aggregation for ImageNet-C
    c = df[df.dataset.str.startswith("imagenet_c")].copy()
    if len(c):
        c["severity"] = c.dataset.str.split("/").str[-1].astype(int)
        sev = c.groupby("severity")[["acc", "auroc_negH_correct", "err_in_lowest10pct_H",
                                     "share_of_errors_below_median_H", "mean_conf_of_errors"]].mean()
        sev.to_csv(os.path.join(od, "imagenet_c_by_severity.csv"))
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        ax[0].plot(sev.index, sev.acc, "o-", label="accuracy")
        ax[0].plot(sev.index, sev.auroc_negH_correct, "s-", label="AUROC(-H → correct)")
        ax[0].set_xlabel("ImageNet-C severity"); ax[0].legend()
        ax[1].plot(sev.index, sev.err_in_lowest10pct_H, "o-", label="error rate in lowest-10% H")
        ax[1].plot(sev.index, sev.share_of_errors_below_median_H, "s-", label="errors below median H")
        ax[1].set_xlabel("ImageNet-C severity"); ax[1].legend()
        save(fig, os.path.join(od, "imagenet_c_severity.png"))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for n, dec in dec_curves.items():
        if not n.startswith("imagenet_c"):
            ax[0].plot(range(1, 11), dec, marker=".", label=n)
    ax[0].set_xlabel("entropy decile (1 = most confident)"); ax[0].set_ylabel("accuracy")
    ax[0].legend(fontsize=6, ncol=2)
    ax[1].scatter(df.acc, df.err_in_lowest10pct_H, s=12)
    for _, r in df.iterrows():
        if not r.dataset.startswith("imagenet_c"):
            ax[1].annotate(r.dataset, (r.acc, r.err_in_lowest10pct_H), fontsize=6)
    ax[1].set_xlabel("zero-shot accuracy"); ax[1].set_ylabel("error rate among lowest-10% entropy")
    save(fig, os.path.join(od, "entropy_deciles.png"))
    print(f"saved -> {od}")
