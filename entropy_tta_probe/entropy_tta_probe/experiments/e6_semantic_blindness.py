"""
E6 — Entropy is blind to the structure of the label / embedding space.

(a) semantic severity of errors: an error between 'husky' and 'malamute' and one between 'husky' and
    'teapot' can have the same entropy. Errors are split into near / far by the text-embedding cosine of
    (predicted, true) class; AUROC of entropy for telling near from far errors (≈0.5 = blind).
    Rao's quadratic entropy  Q = Σ p_i p_j (1 - cos(t_i, t_j))  is reported as a structure-aware contrast.
(b) iso-entropy spread: within each entropy decile, the spread (IQR) of Q — same entropy, very
    different kinds of uncertainty.
(c) cosine geometry: CLIP image–text cosines are small and tightly packed (modality gap); entropy is
    driven by top-1 − top-2 cosine gaps of ~0.01 amplified by the logit scale. Reports mean max cosine,
    gap statistics for correct vs wrong, and corr(H, gap).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from etta.common import base_args, get_features, load_bundle, out_dir  # noqa: E402
from etta.data import expand_names  # noqa: E402
from etta.metrics import auroc, np_entropy, np_softmax, rao_quadratic_entropy  # noqa: E402
from etta.plot import plt, save  # noqa: E402

if __name__ == "__main__":
    ap = base_args(__doc__)
    ap.add_argument("--max_samples", type=int, default=10000)
    args = ap.parse_args()
    bundle = load_bundle(args)
    od = out_dir(args, "e6")
    rows, scatter = [], {}
    for n in expand_names(args.datasets):
        d = get_features(args, bundle, n, args.max_samples)
        T = bundle.text_features(d["classnames"])
        sim = (T @ T.t()).cpu().numpy()
        cos = (d["feats"].float().to(bundle.device) @ T.t()).cpu().numpy()
        y = d["labels"].numpy()
        p = np_softmax(bundle.logit_scale * cos)
        pred = p.argmax(1)
        corr = (pred == y).astype(float)
        H = np_entropy(p)
        Q = rao_quadratic_entropy(p, sim)
        wrong = corr == 0
        s_err = sim[pred[wrong], y[wrong]]
        near = s_err >= np.median(s_err) if wrong.any() else np.array([])
        top2 = np.sort(cos, 1)[:, -2:]
        gap = top2[:, 1] - top2[:, 0]
        dec = np.digitize(H, np.quantile(H, np.linspace(0, 1, 11)[1:-1]))
        iqr = [np.subtract(*np.percentile(Q[dec == k], [75, 25])) if (dec == k).sum() > 3 else np.nan
               for k in range(10)]
        rows.append(dict(
            dataset=n, n=len(y), K=len(T), acc=corr.mean(),
            auroc_negH_correct=auroc(-H, corr), auroc_negQ_correct=auroc(-Q, corr),
            auroc_H_near_vs_far_err=auroc(-H[wrong], near.astype(float)) if wrong.sum() > 10 else np.nan,
            auroc_Q_near_vs_far_err=auroc(-Q[wrong], near.astype(float)) if wrong.sum() > 10 else np.nan,
            mean_H_near_err=H[wrong][near].mean() if wrong.any() else np.nan,
            mean_H_far_err=H[wrong][~near].mean() if wrong.any() else np.nan,
            mean_Q_iqr_within_H_decile=np.nanmean(iqr),
            mean_text_text_cos=(sim.sum() - np.trace(sim)) / (len(T) * (len(T) - 1)),
            mean_max_img_text_cos=cos.max(1).mean(),
            mean_top12_cos_gap_correct=gap[corr == 1].mean() if (corr == 1).any() else np.nan,
            mean_top12_cos_gap_wrong=gap[wrong].mean() if wrong.any() else np.nan,
            corr_H_gap=float(np.corrcoef(H, gap)[0, 1]),
        ))
        scatter[n] = (H, Q, corr)
        print(f"{n}: done")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(od, "summary.csv"), index=False)
    print(df.round(4).to_string(index=False))

    show = list(scatter)[:4]
    fig, ax = plt.subplots(1, max(1, len(show)), figsize=(3.2 * max(1, len(show)), 3), squeeze=False)
    for a, n in zip(ax[0], show):
        H, Q, c = scatter[n]
        a.scatter(H[c == 1], Q[c == 1], s=2, alpha=0.3, label="correct")
        a.scatter(H[c == 0], Q[c == 0], s=2, alpha=0.3, label="wrong")
        a.set_xlabel("entropy H"); a.set_ylabel("Rao Q (text-space)"); a.set_title(n, fontsize=8)
    ax[0][0].legend(markerscale=4)
    save(fig, os.path.join(od, "entropy_vs_rao.png"))
    print(f"saved -> {od}")
