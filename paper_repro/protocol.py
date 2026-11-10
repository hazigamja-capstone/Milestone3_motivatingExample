"""Experiment v3: fixed, class-disjoint calibration and confirmation streams."""
import pickle
from pathlib import Path
import random
import tempfile

VERSION = 3
PARTITION_SEED = 1729
CALIBRATION_SEEDS = (101, 102)
CONFIRMATION_SEEDS = tuple(range(201, 211))
ROLES = ('paper', 'calibration', 'confirmation')


def role_classes(role, n_classes=1000):
    if role not in ROLES:
        raise ValueError(f'Unknown role: {role}')
    ids = list(range(n_classes))
    if role == 'paper':
        return ids
    random.Random(PARTITION_SEED).shuffle(ids)
    split = n_classes * 2 // 5
    return sorted(ids[:split] if role == 'calibration' else ids[split:])


def domain_order(domains, role, seed):
    domains = list(domains)
    if role != 'paper':
        random.Random(seed).shuffle(domains)
    return domains


def reorder_stream(source, role, seed, domain_index, n_classes=1000):
    """Spool compressed class blocks to disk, retaining only one class in RAM.

    Source yields (label, encoded bytes, sample_id), sorted by label. No model
    predictions or outcome labels enter randomization. Temporary files are ours.
    """
    selected = set(role_classes(role, n_classes))
    order = sorted(selected)
    rng = random.Random(seed * 1000 + domain_index)
    if role != 'paper':
        rng.shuffle(order)
    with tempfile.TemporaryDirectory(prefix='tta-stream-') as tmp:
        current, block = None, []
        def flush():
            if current in selected:
                with open(Path(tmp) / str(current), 'wb') as f:
                    pickle.dump(block, f)
        seen = set()
        for c, encoded, sample_id in source:
            if c != current:
                if current is not None:
                    flush()
                    seen.add(current)
                if c in seen:
                    raise ValueError('Source classes must form contiguous blocks')
                current, block = c, []
            if c in selected:
                block.append((c, encoded, sample_id))
        if current is not None:
            flush()
        for c in order:
            path = Path(tmp) / str(c)
            if not path.exists():
                raise ValueError(f'Missing class {c}')
            with open(path, 'rb') as f:
                items = pickle.load(f)
            rng.shuffle(items)
            yield from items
