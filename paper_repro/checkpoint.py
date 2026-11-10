"""Versioned domain-boundary checkpoints. Load only locally trusted checkpoints."""
import os
import random

import numpy as np
import torch

STATE_KEYS = ("current_model_probs", "ema", "num_samples_update_1",
              "num_samples_update_2", "performed_updates", "input_buffer", "pointer")


def mutable_model_state(model):
    """TTA changes norm parameters, not frozen CLIP weights; buffers may change."""
    keys = {name for name, p in model.named_parameters() if p.requires_grad}
    keys.update(name for name, _ in model.named_buffers())
    return {name: value for name, value in model.state_dict().items() if name in keys}


def save_state(path, tta, done, identity, progress=None):
    scaler = getattr(tta, "scaler", None)
    base = getattr(tta.optimizer, "base_optimizer", None)
    state = dict(version=3, identity=identity, done=done, progress=progress,
                 model=mutable_model_state(tta.model), optimizer=tta.optimizer.state_dict(),
                 base_optimizer=base.state_dict() if base is not None else None,
                 scaler=scaler.state_dict() if scaler is not None else None,
                 extra={k: getattr(tta, k) for k in STATE_KEYS if hasattr(tta, k)},
                 rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                          torch=torch.get_rng_state(),
                          cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    torch.save(state, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)


def load_state(path, tta, identity, with_progress=False):
    # CPU loading keeps RNG ByteTensors on CPU; model/optimizer loading moves weights as needed.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("version") != 3 or state.get("identity") != identity:
        raise ValueError("Checkpoint is legacy or incompatible; use a new --out directory. Do not overwrite old results.")
    expected = set(mutable_model_state(tta.model))
    if set(state["model"]) != expected:
        raise ValueError("Checkpoint trainable parameters/buffers differ from reconstructed model")
    tta.model.load_state_dict(state["model"], strict=False)
    tta.optimizer.load_state_dict(state["optimizer"])
    base = getattr(tta.optimizer, "base_optimizer", None)
    if base is not None:
        base.load_state_dict(state["base_optimizer"])
        # SAM and SGD must continue to reference the same parameter groups.
        tta.optimizer.param_groups = base.param_groups
    scaler = getattr(tta, "scaler", None)
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    device = next(tta.model.parameters()).device
    for k, v in state["extra"].items():
        if torch.is_tensor(v):
            v = v.to(device)
        elif isinstance(v, list):
            v = [x.to(device) if torch.is_tensor(x) else x for x in v]
        setattr(tta, k, v)
    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return (state["done"], state.get("progress")) if with_progress else state["done"]
