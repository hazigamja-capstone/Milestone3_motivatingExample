"""Deterministic input acceleration; no model, optimizer, or RNG policy changes."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import io
import threading


def ordered_preprocess(records, preprocess, workers=4, pending=16):
    """Bounded, ordered map. Only use with the deterministic CLIP eval transform.

    Tasks decode images and run CPU transforms only. They must never draw from
    Python/NumPy/torch RNGs. Closing the iterator joins workers, including on error.
    """
    from PIL import Image
    if workers < 1 or pending < 1:
        raise ValueError('workers and pending must be positive')

    def convert(record):
        label, encoded, sample_id = record
        with Image.open(io.BytesIO(encoded)) as image:
            tensor = preprocess(image.convert('RGB'))
        return tensor, label, sample_id

    records = iter(records)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='tta-decode') as pool:
        futures = deque()
        try:
            for _ in range(pending):
                record = next(records, None)
                if record is None:
                    break
                futures.append(pool.submit(convert, record))
            while futures:
                yield futures.popleft().result()
                record = next(records, None)
                if record is not None:
                    futures.append(pool.submit(convert, record))
        finally:
            for future in futures:
                future.cancel()
            close = getattr(records, 'close', None)
            if close:
                close()


def deterministic_clip_transform(preprocess):
    """Fail closed for unknown/stochastic transforms instead of changing RNG order."""
    import torchvision.transforms as transforms
    from open_clip.transform import _convert_to_rgb
    allowed = (transforms.Resize, transforms.CenterCrop, transforms.ToTensor,
               transforms.Normalize)
    return (type(preprocess) is transforms.Compose
            and all(type(t) in allowed or t is _convert_to_rgb for t in preprocess.transforms))


def prefetched_batches(records, preprocess, bs, prefetch=8, workers=4, pin_memory=False):
    """Overlap bounded CPU preparation with compute; cancel producers on early exit."""
    import queue
    import torch
    if bs < 1 or prefetch < 1:
        raise ValueError('bs and prefetch must be positive')
    items = queue.Queue(maxsize=prefetch)
    stop = threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                items.put(item, timeout=.05)
                return
            except queue.Full:
                pass

    def pack(rows):
        x = torch.stack([r[0] for r in rows])
        y = torch.tensor([r[1] for r in rows])
        ids = torch.tensor([r[2] for r in rows])
        if pin_memory:
            x, y = x.pin_memory(), y.pin_memory()
        return x, y, ids

    def work():
        converted = ordered_preprocess(records, preprocess, workers, max(2 * workers, 1))
        try:
            rows = []
            for row in converted:
                if stop.is_set():
                    break
                rows.append(row)
                if len(rows) == bs:
                    put(pack(rows)); rows = []
            if rows and not stop.is_set():
                put(pack(rows))
            put(None)
        except BaseException as exc:
            put(exc)
        finally:
            converted.close()

    thread = threading.Thread(target=work, name='tta-input', daemon=True)
    thread.start()
    try:
        while True:
            item = items.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join()


@contextmanager
def reuse_fixed_text(model_module, reference, cfg):
    """Reuse the first model's exact frozen ensemble tensors in its identical twin.

    The second visual model is still constructed normally, preserving its RNG
    consumption and independent parameters. No serialized cache or dtype change.
    Restricted to the pinned ZeroShotCLIP constructor and frozen ensemble setup.
    """
    import torch
    import hashlib
    from pathlib import Path
    original = getattr(model_module, "ZeroShotCLIP", None)
    source = getattr(model_module, '__file__', None)
    pinned = (source is not None and Path(source).is_file() and
              hashlib.sha256(Path(source).read_bytes()).hexdigest() ==
              'd61ac5b3f6d1a7064fb16b27ada2604255ad08d6f8f6af14df2b4217b839906d')
    if (not pinned or type(reference) is not original or not cfg.CLIP.FREEZE_TEXT_ENCODER
            or cfg.CLIP.PROMPT_MODE != 'ensemble'
            or cfg.MODEL.ARCH not in ('ViT-B-16', 'ViT-L-14')
            or cfg.MODEL.WEIGHTS != 'openai'):
        yield
        return

    class ReuseText(original):
        def __init__(self, new_cfg, model, device, normalize):
            if new_cfg.dump() != cfg.dump():
                raise ValueError('Text feature reuse requires identical CLIP configuration')
            torch.nn.Module.__init__(self)
            self.cfg, self.model, self.device = new_cfg, model, device
            self.normalize = normalize
            self.prompt_mode = reference.prompt_mode
            self.freeze_text_encoder = reference.freeze_text_encoder
            self.class_names = list(reference.class_names)
            self.tokenize = reference.tokenize
            self.logit_scale = model.logit_scale.data
            self.text_features = reference.text_features.detach().clone()
            self.tokenized_texts_all = reference.tokenized_texts_all.detach().clone()
            self.model.transformer = None

    model_module.ZeroShotCLIP = ReuseText
    try:
        yield
    finally:
        model_module.ZeroShotCLIP = original
