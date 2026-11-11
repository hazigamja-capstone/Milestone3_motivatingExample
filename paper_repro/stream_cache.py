"""Verified local cache of sorted, compressed ImageNet-C source records.

Use a dedicated local disk directory, not Google Drive. The cache identity must
include immutable dataset revision, corruption and severity; role/order seed are
intentionally absent so all experiment runs reuse the same downloaded bytes.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile

# uint32 label, uint64 source sample ID, uint32 compressed byte count.
_HEADER = struct.Struct('>IQI')
_VERSION = 1


def _digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(key, n_classes):
    if type(n_classes) is not int or not 1 <= n_classes <= 1000:
        raise ValueError('n_classes must be between 1 and 1000')
    identity = dict(version=_VERSION, source=key, n_classes=n_classes)
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()
    return identity, hashlib.sha256(encoded).hexdigest()


def _valid(root, identity):
    try:
        if root.is_symlink() or (root/'records.bin').is_symlink() or (root/'manifest.json').is_symlink():
            raise ValueError('Cache paths must not be symlinks')
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        data = root/'records.bin'
        return (manifest['identity'] == identity
                and manifest['records'] == identity['n_classes'] * 50
                and data.stat().st_size == manifest['bytes']
                and _digest_file(data) == manifest['sha256'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _build(source_factory, staging, identity):
    counts = [0] * identity['n_classes']
    seen_ids = set()
    previous = -1
    sha = hashlib.sha256()
    size = 0
    with (staging/'records.bin').open('wb') as stream:
        for label, encoded, sample_id in source_factory():
            if (type(label) is not int or not 0 <= label < len(counts)
                    or type(sample_id) is not int or sample_id // 50 != label
                    or label < previous or sample_id in seen_ids):
                raise ValueError('Source must contain sorted class blocks with unique valid sample IDs')
            if not isinstance(encoded, bytes) or not encoded:
                raise ValueError('Source image must contain nonempty encoded bytes')
            counts[label] += 1
            if counts[label] > 50:
                raise ValueError('Source contains more than 50 samples per class')
            seen_ids.add(sample_id)
            previous = label
            header = _HEADER.pack(label, sample_id, len(encoded))
            stream.write(header); stream.write(encoded)
            sha.update(header); sha.update(encoded)
            size += len(header) + len(encoded)
        if any(count != 50 for count in counts):
            raise ValueError('Source must contain exactly 50 samples for each selected class')
        stream.flush()
        os.fsync(stream.fileno())
    manifest = dict(identity=identity, records=sum(counts), bytes=size, sha256=sha.hexdigest())
    with (staging/'manifest.json').open('w', encoding='utf-8') as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def _records(path):
    with path.open('rb') as stream:
        while True:
            header = stream.read(_HEADER.size)
            if not header:
                return
            if len(header) != _HEADER.size:
                raise ValueError('Truncated source cache header')
            label, sample_id, size = _HEADER.unpack(header)
            encoded = stream.read(size)
            if len(encoded) != size:
                raise ValueError('Truncated source cache image')
            yield label, encoded, sample_id


@contextmanager
def _verified_cache(source_factory, cache_root, key, n_classes):
    # Colab is Linux. flock avoids duplicate downloads in concurrent local jobs.
    import fcntl

    identity, name = _identity(key, n_classes)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root/name
    with (cache_root/f'{name}.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not _valid(target, identity):
            if target.is_symlink():
                raise ValueError('Refusing a symlink in the source cache')
            staging = Path(tempfile.mkdtemp(prefix=f'.{name}-', dir=cache_root))
            try:
                _build(source_factory, staging, identity)
                if target.exists():
                    shutil.rmtree(target)
                os.replace(staging, target)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        # Keep the lock while iterating, so another cache user cannot replace the
        # directory while these verified records are being read.
        yield target


def cached_source(source_factory, cache_root, key, n_classes=1000):
    """Yield a fully SHA-256 verified source; retain the v1 cache on disk."""
    with _verified_cache(source_factory, cache_root, key, n_classes) as target:
        yield from _records(target/'records.bin')


def cached_ordered_source(source_factory, cache_root, key, role, seed, domain_index, n_classes=1000):
    """Read class permutations directly from verified v1 compressed bytes.

    Keep one small offset index in RAM instead of writing and rereading every
    selected class to temporary pickle files. Validation still reads ALL bytes
    on each opening. The cache format/identity and the RNG calls match v3.
    """
    import mmap
    import random
    from paper_repro.protocol import role_classes
    selected = set(role_classes(role, n_classes))
    order = sorted(selected)
    rng = random.Random(seed * 1000 + domain_index)
    if role != 'paper':
        rng.shuffle(order)
    with _verified_cache(source_factory, cache_root, key, n_classes) as target:
        with (target/'records.bin').open('rb') as stream:
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
                blocks = {c: [] for c in selected}
                offset = 0
                while offset < len(data):
                    if len(data) - offset < _HEADER.size:
                        raise ValueError('Truncated source cache header')
                    label, sample_id, size = _HEADER.unpack_from(data, offset)
                    offset += _HEADER.size
                    if offset + size > len(data):
                        raise ValueError('Truncated source cache image')
                    if label in selected:
                        blocks[label].append((offset, size, sample_id))
                    offset += size
                for label in order:
                    items = blocks[label]
                    if len(items) != 50:
                        raise ValueError(f'Missing/incomplete class {label}')
                    rng.shuffle(items)
                    for offset, size, sample_id in items:
                        yield label, data[offset:offset+size], sample_id
