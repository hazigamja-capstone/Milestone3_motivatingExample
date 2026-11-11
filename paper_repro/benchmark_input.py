"""Synthetic CPU input microbenchmark; no weights/data download or experiment edits.

Run from the repository root: python -m paper_repro.benchmark_input
Uses one intra-op torch thread for both paths, 1,000 WebP images, three repeats.
This measures input stages only, not end-to-end GPU training throughput.
"""
import io
import random
import statistics
import tempfile
import time

from PIL import Image
import torch
from open_clip.transform import image_transform

from paper_repro.performance import ordered_preprocess
from paper_repro.protocol import reorder_stream
from paper_repro.stream_cache import cached_source, cached_ordered_source


def timing(function):
    times = []
    for _ in range(3):
        start = time.perf_counter()
        count = function()
        times.append(time.perf_counter() - start)
    return round(statistics.median(times), 4), count


def main():
    torch.set_num_threads(1)
    rng = random.Random(4)
    encoded = []
    for _ in range(32):
        image = Image.frombytes('RGB', (256, 256), rng.randbytes(256 * 256 * 3))
        buffer = io.BytesIO()
        image.save(buffer, format='WEBP')
        encoded.append(buffer.getvalue())
    records = [(i // 50, encoded[i % 32], i) for i in range(1000)]
    preprocess = image_transform(224, is_train=False)
    preprocess.transforms = preprocess.transforms[:-1]
    for workers in (1, 4):
        print('preprocess', workers, timing(lambda: sum(
            1 for _ in ordered_preprocess(records, preprocess, workers))))
    with tempfile.TemporaryDirectory() as temporary:
        key = 'benchmark'
        list(cached_source(lambda: iter(records), temporary, key, 20))
        print('cache_original', timing(lambda: sum(1 for _ in reorder_stream(
            cached_source(lambda: iter(records), temporary, key, 20), 'confirmation', 201, 3, 20))))
        print('cache_direct', timing(lambda: sum(1 for _ in cached_ordered_source(
            lambda: iter(records), temporary, key, 'confirmation', 201, 3, 20))))


if __name__ == '__main__':
    main()
