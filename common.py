import json
import os

import open_clip
import torch
import torch.nn.functional as F
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
LOGIT_SCALE = 100.0


def load_clip(device):
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-16-quickgelu")
    return model.to(device).eval(), tokenizer


def imagenet_a_classes():
    """ImageNet-A의 200 wnid와 그에 대응하는 1000-class 인덱스. 모든 열이 이 라벨 공간을 공유한다."""
    with open(os.path.join(DATA, "imagenet_class_index.json")) as f:
        w2i = {v[0]: int(k) for k, v in json.load(f).items()}
    root = os.path.join(DATA, "imagenet-a")
    wnids = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    idx = [w2i[w] for w in wnids]
    names = [IMAGENET_CLASSNAMES[i] for i in idx]
    return wnids, idx, names


@torch.no_grad()
def text_features(model, tokenizer, names, device, template="a photo of a {}."):
    tokens = tokenizer([template.format(n) for n in names]).to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        t = model.encode_text(tokens)
    return F.normalize(t.float(), dim=-1)


def encode_image(model, x):
    """x: [B,3,224,224] in [0,1]. Returns L2-normalized float32 features."""
    x = (x - MEAN.to(x.device)) / STD.to(x.device)
    with torch.autocast("cuda", dtype=torch.float16):
        f = model.encode_image(x)
    return F.normalize(f.float(), dim=-1)
