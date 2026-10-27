"""
E4 — Confirmation bias, error accumulation and collapse when entropy minimisation runs ONLINE
(no reset between batches, as in TENT / continual TTA, unlike episodic TPT).

Stream = the given datasets one after another (e.g. imagenet_c:sev=5 -> 15 corruptions = CoTTA-style
continual shift). Within each dataset the order is
  iid           random order
  dirichlet     non-i.i.d. label shift (NOTE/RoTTA-style Dirichlet over time slots, --alpha)
  class_sorted  extreme label shift (sorted by class)
Parameters adapted (--params):
  prompt   shared context vectors (TPT/SwapPrompt parameter set; image features cached -> fast)
  norm     normalisation affine params of the image encoder (TENT on the VLM visual tower)
Objective (--objective): ent  = mean per-sample entropy (TENT)
                         ent_div = ent - div_w * H(batch-mean prediction)  (diversity term; diagnostic)
Logged per window: online acc vs zero-shot acc on the same samples, mean confidence/entropy,
#distinct predicted classes relative to #distinct true classes, share of the most predicted class.
At the end the final parameters are re-evaluated on every dataset (permanent damage?).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from etta.augment import base_preprocess  # noqa: E402
from etta.common import base_args, dump_json, get_features, load_bundle, out_dir, seed_all  # noqa: E402
from etta.data import ImageListDataset, build_split, expand_names, subsample_labels  # noqa: E402
from etta.metrics import entropy_from_logits  # noqa: E402
from etta.model import PromptLearner  # noqa: E402
from etta.plot import plt, save  # noqa: E402


def make_order(labels: np.ndarray, idx: list, mode: str, alpha: float, rng: np.random.Generator):
    idx = np.array(idx)
    if mode == "iid":
        return list(rng.permutation(idx))
    if mode == "class_sorted":
        return list(idx[np.argsort(labels[idx], kind="stable")])
    if mode == "dirichlet":
        classes = np.unique(labels[idx])
        K = len(classes)
        slots = [[] for _ in range(K)]
        for c in classes:
            members = rng.permutation(idx[labels[idx] == c])
            p = rng.dirichlet(alpha * np.ones(K))
            counts = rng.multinomial(len(members), p)
            pos = 0
            for k, n in enumerate(counts):
                slots[k] += list(members[pos:pos + n]); pos += n
        return [i for s in slots for i in rng.permutation(s)]
    raise ValueError(mode)


def configure_norm(model, arch):
    params = []
    for m in model.visual.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train(); m.track_running_stats = False; m.running_mean = None; m.running_var = None
            m.requires_grad_(True); params += [m.weight, m.bias]
        elif isinstance(m, nn.LayerNorm):
            m.requires_grad_(True); params += [m.weight, m.bias]
    return params


def objective(logits, kind, div_w):
    loss = entropy_from_logits(logits).mean()
    if kind == "ent_div":
        pm = logits.softmax(-1).mean(0)
        loss = loss + div_w * (pm * pm.clamp_min(1e-12).log()).sum()  # = - div_w * H(mean p)
    return loss


def window_stats(df, w):
    out = []
    for s in range(0, len(df), w):
        g = df.iloc[s:s + w]
        if len(g) < w // 2:
            break
        nt = g.label.nunique()
        out.append(dict(
            start=s, dataset=g.dataset.iloc[0],
            acc=(g.pred == g.label).mean(), acc_zs=(g.zs_pred == g.label).mean(),
            mean_conf=g.conf.mean(), mean_entropy=g.entropy.mean(),
            pred_classes_over_true=g.pred.nunique() / nt, zs_pred_classes_over_true=g.zs_pred.nunique() / nt,
            top_class_share=g.pred.value_counts().iloc[0] / len(g),
            true_top_class_share=g.label.value_counts().iloc[0] / len(g),
            wrong_conf_gt90=((g.pred != g.label) & (g.conf > 0.9)).mean(),
        ))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = base_args(__doc__)
    ap.add_argument("--params", choices=["prompt", "norm"], default="prompt")
    ap.add_argument("--order", choices=["iid", "dirichlet", "class_sorted"], default="iid")
    ap.add_argument("--alpha", type=float, default=0.1, help="Dirichlet concentration (smaller = stronger label shift)")
    ap.add_argument("--objective", choices=["ent", "ent_div"], default="ent")
    ap.add_argument("--div_w", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--steps", type=int, default=1, help="gradient steps per incoming batch")
    ap.add_argument("--max_per_dataset", type=int, default=5000)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--ctx_init", default="a photo of a")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    bundle = load_bundle(args)
    names = expand_names(args.datasets)
    tag = args.tag or f"{args.params}_{args.order}_{args.objective}_lr{args.lr}_s{args.steps}"
    od = os.path.join(out_dir(args, "e4"), tag)
    os.makedirs(od, exist_ok=True)
    s = bundle.logit_scale
    dev = bundle.device

    # ---- learnable parameters
    learners, zs_text = {}, {}
    shared_ctx = None
    caches, splits = {}, {}
    for n in names:
        caches[n] = get_features(args, bundle, n)
        zs_text[n] = bundle.text_features(caches[n]["classnames"])
        if args.params == "prompt":
            pl = PromptLearner(bundle, caches[n]["classnames"], args.ctx_init).to(dev)
            if shared_ctx is None:
                shared_ctx = pl.ctx
            else:
                pl.ctx = shared_ctx  # one context shared across the whole stream
            learners[n] = pl
        else:
            splits[n] = build_split(n, args.data_root)
    if args.params == "prompt":
        opt = torch.optim.AdamW([shared_ctx], lr=args.lr, weight_decay=0.0)
    else:
        params = configure_norm(bundle.model, args.arch)
        opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9)

    def forward(n, x_or_f):
        if args.params == "prompt":
            return s * x_or_f @ learners[n]().t()
        f = F.normalize(bundle.model.encode_image(x_or_f).float(), dim=-1)
        return s * f @ zs_text[n].t()

    # ---- stream
    records = []
    for n in names:
        labels = caches[n]["labels"].numpy()
        sub = subsample_labels(labels, args.max_per_dataset, args.seed)
        order = make_order(labels, sub, args.order, args.alpha, rng)
        feats = caches[n]["feats"].float()
        zs_pred_all = (feats.to(dev) @ zs_text[n].t()).argmax(1).cpu().numpy()
        if args.params == "prompt":
            batches = ((order[i:i + args.bs], feats[order[i:i + args.bs]].to(dev)) for i in range(0, len(order), args.bs))
        else:
            ds = ImageListDataset(splits[n], base_preprocess(bundle.input_size), order)
            dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers)
            batches = ((ix.tolist(), x.to(dev)) for x, _, ix in dl)
        for ids, x in batches:
            for k in range(args.steps):
                logits = forward(n, x)
                if k == 0:  # online prediction = output of the first forward on this batch (TENT protocol)
                    with torch.no_grad():
                        p = logits.softmax(-1)
                        conf, pred = p.max(-1)
                        H = entropy_from_logits(logits)
                    for j, i in enumerate(ids):
                        records.append((n, int(i), int(labels[i]), int(pred[j]), float(conf[j]), float(H[j]),
                                        int(zs_pred_all[i])))
                loss = objective(logits, args.objective, args.div_w)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        print(f"{n}: streamed {len(order)} samples")

    df = pd.DataFrame(records, columns=["dataset", "index", "label", "pred", "conf", "entropy", "zs_pred"])
    df.to_csv(os.path.join(od, "stream.csv.gz"), index=False)
    window = min(args.window, max(2, len(df) // 10))  # short streams: keep ~10 windows
    if window != args.window:
        print(f"stream has {len(df)} samples -> window {args.window} reduced to {window}")
    win = window_stats(df, window)
    win.to_csv(os.path.join(od, "windows.csv"), index=False)

    # ---- final re-evaluation with the adapted parameters
    final = []
    with torch.no_grad():
        for n in names:
            f = caches[n]["feats"].float().to(dev)
            y = caches[n]["labels"].numpy()
            zs = (f @ zs_text[n].t()).argmax(1).cpu().numpy()
            if args.params == "prompt":
                fin = (f @ learners[n]().t()).argmax(1).cpu().numpy()
            else:
                sub = subsample_labels(y, args.max_per_dataset, args.seed)
                ds = ImageListDataset(splits[n], base_preprocess(bundle.input_size), sub)
                preds = []
                for x, _, _ in DataLoader(ds, batch_size=256, num_workers=args.workers):
                    preds.append(forward(n, x.to(dev)).argmax(1).cpu())
                fin, y, zs = torch.cat(preds).numpy(), y[sub], zs[sub]
            online = df[df.dataset == n]
            final.append(dict(dataset=n, zero_shot_acc=(zs == y).mean(),
                              online_acc=(online.pred == online.label).mean(),
                              final_params_acc=(fin == y).mean(),
                              final_distinct_pred_classes=len(np.unique(fin)), num_classes=len(caches[n]["classnames"])))
    fdf = pd.DataFrame(final)
    fdf.to_csv(os.path.join(od, "final_eval.csv"), index=False)
    dump_json(vars(args), os.path.join(od, "args.json"))
    print(fdf.round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    x = win.start
    ax[0].plot(x, win.acc, label="online (entropy-min)"); ax[0].plot(x, win.acc_zs, label="zero-shot", ls="--")
    ax[0].set_title("accuracy per window"); ax[0].legend()
    ax[1].plot(x, win.pred_classes_over_true, label="adapted"); ax[1].plot(x, win.zs_pred_classes_over_true, ls="--", label="zero-shot")
    ax[1].set_title("#pred classes / #true classes"); ax[1].legend()
    ax[2].plot(x, win.mean_conf, label="mean conf"); ax[2].plot(x, win.wrong_conf_gt90, label="wrong & conf>0.9")
    ax[2].plot(x, win.top_class_share, label="top-class share"); ax[2].legend(); ax[2].set_title("confidence / collapse")
    for a in ax:
        a.set_xlabel("stream position (samples)")
        for b in np.flatnonzero(win.dataset.values[1:] != win.dataset.values[:-1]):
            a.axvline(x.iloc[b + 1], color="k", lw=0.4, alpha=0.4)
    save(fig, os.path.join(od, "stream.png"))
    print(f"saved -> {od}")
