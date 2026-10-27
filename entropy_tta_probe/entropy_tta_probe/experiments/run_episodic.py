"""
Episodic TPT pass (the expensive one) — shared by E2 (over-confidence) and E3 (view selection).

Per test image (TPT protocol, prompt reset after every sample):
  1. 64 views (orig + 63 random-resized crops, AugMix on non-ImageNet sets), frozen image encoder
  2. zero-shot logits for every view with ctx = 'a photo of a'
  3. TPT: select the 10% lowest-entropy views, minimise their marginal entropy w.r.t. ctx (AdamW, lr 5e-3),
     re-selecting each step; after step k predict on the original view  (k = 1..--steps; k=1 is TPT)
  4. training-free baselines on the same views: mean over all views, mean over the selected views,
     ZERO (majority vote of argmax over the selected views)
Everything is dumped to <out>/episodic/<arch>/<dataset>.npz.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from etta.augment import TPTViews, collate_views  # noqa: E402
from etta.common import base_args, dump_json, load_bundle, out_dir, seed_all  # noqa: E402
from etta.data import IMAGENET_FAMILY, ImageListDataset, build_split, expand_names, safe_name, subsample  # noqa: E402
from etta.metrics import entropy_from_logits, marginal_entropy, select_confident  # noqa: E402
from etta.model import make_prompt_learner  # noqa: E402


def run_dataset(args, bundle, name, od):
    path = os.path.join(od, f"{safe_name(name)}.npz")
    if os.path.isfile(path) and not args.overwrite:
        print(f"skip {name} (exists)")
        return
    split = build_split(name, args.data_root)
    idx = subsample(split, args.n_samples, args.seed)
    use_augmix = not (name in IMAGENET_FAMILY or name.startswith("imagenet_c"))
    views_tf = TPTViews(args.n_views, bundle.input_size, augmix=use_augmix, seed=args.seed)
    dl = DataLoader(ImageListDataset(split, views_tf, idx), batch_size=1, shuffle=False,
                    num_workers=args.workers, collate_fn=collate_views)
    pl, _ = make_prompt_learner(bundle, split.classnames, args.ctx_init, args.lr)
    s = bundle.logit_scale
    with torch.no_grad():
        t0 = pl()

    N, V, C, S = len(idx), args.n_views, split.num_classes, args.steps
    rec = dict(
        labels=np.zeros(N, np.int64), index=np.zeros(N, np.int64),
        zs_logits=np.zeros((N, C), np.float16), tpt_logits=np.zeros((S, N, C), np.float16),
        view_entropy=np.zeros((N, V), np.float32), view_pred=np.zeros((N, V), np.int32),
        view_conf=np.zeros((N, V), np.float32), view_area=np.zeros((N, V), np.float32),
        selected=np.zeros((N, V), bool),
        ens_all_prob=np.zeros((N, C), np.float16), ens_sel_prob=np.zeros((N, C), np.float16),
        zero_pred=np.zeros(N, np.int64),
    )
    t_start = time.time()
    for n, (views, areas, label, i) in enumerate(tqdm(dl, desc=name)):
        f = bundle.encode_images(views)  # [V, D] frozen
        with torch.no_grad():
            lg = s * f @ t0.t()
            prob = lg.softmax(-1)
            H = entropy_from_logits(lg)
            sel = select_confident(lg, args.selection_ratio)
            rec["zs_logits"][n] = lg[0].cpu().numpy()
            rec["view_entropy"][n] = H.cpu().numpy()
            conf, pred = prob.max(-1)
            rec["view_pred"][n] = pred.cpu().numpy()
            rec["view_conf"][n] = conf.cpu().numpy()
            rec["view_area"][n] = areas.numpy()
            rec["selected"][n, sel.cpu().numpy()] = True
            rec["ens_all_prob"][n] = prob.mean(0).cpu().numpy()
            sel_prob = prob[sel].mean(0)
            rec["ens_sel_prob"][n] = sel_prob.cpu().numpy()
            votes = torch.bincount(pred[sel], minlength=C).float()
            votes = votes + 1e-3 * sel_prob  # tie-break by soft score
            rec["zero_pred"][n] = int(votes.argmax())

        # ---- TPT (episodic: reset ctx and optimizer for every sample)
        pl.reset()
        opt = torch.optim.AdamW([pl.ctx], lr=args.lr, weight_decay=0.0)
        for k in range(S):
            t = pl()
            lg_k = s * f @ t.t()
            sel_k = select_confident(lg_k.detach(), args.selection_ratio)
            loss = marginal_entropy(lg_k[sel_k])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                rec["tpt_logits"][k, n] = (s * f[:1] @ pl().t())[0].cpu().numpy()
        rec["labels"][n] = label
        rec["index"][n] = i
    np.savez_compressed(path, **rec)
    dump_json(dict(dataset=name, classnames=split.classnames, logit_scale=s, n_views=V,
                   steps=S, lr=args.lr, selection_ratio=args.selection_ratio, augmix=use_augmix,
                   ctx_init=args.ctx_init, seconds=time.time() - t_start),
              path.replace(".npz", ".json"))
    zs_acc = (rec["zs_logits"].argmax(1) == rec["labels"]).mean()
    tpt_acc = (rec["tpt_logits"][0].argmax(1) == rec["labels"]).mean()
    print(f"{name}: zero-shot {zs_acc:.4f}  TPT(1 step) {tpt_acc:.4f}")


if __name__ == "__main__":
    ap = base_args(__doc__)
    ap.add_argument("--n_samples", type=int, default=1000, help="class-stratified subset per dataset (None=all)")
    ap.add_argument("--n_views", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4, help="record predictions after 1..steps TPT updates")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--selection_ratio", type=float, default=0.1)
    ap.add_argument("--ctx_init", default="a photo of a")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.n_samples is not None and args.n_samples <= 0:
        args.n_samples = None
    seed_all(args.seed)
    bundle = load_bundle(args)
    od = out_dir(args, "episodic")
    for name in expand_names(args.datasets):
        run_dataset(args, bundle, name, od)
