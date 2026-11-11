"""Explicit, finite compatibility bridge for the published v3 source bundle.

Scientific identity stays attached to the original study. Execution provenance
records the actual optimized sources separately; unknown/edited code never maps
onto that identity. This does not monkeypatch hashes, checkpoint loading, or the
bundle installer. All original identity and artifact comparisons still run.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path

LEGACY_SOURCE_NAMES = ('run_paper.py', 'checkpoint.py', 'run_identity.py', 'protocol.py', 'stream_cache.py')
PLAN_NAMES = ('run_plan.py', 'run_paper.py', 'protocol.py', 'checkpoint.py', 'run_identity.py',
              'analyze_paper.py', 'evaluate_detector.py', 'stream_cache.py')


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(data)
    return sha.hexdigest()


def legacy_execution(root, study):
    """Return the known legacy manifest only after verifying both source versions."""
    root, study = Path(root), Path(study)
    manifest = json.loads((root/'paper_repro/compatibility_manifest.json').read_text())
    bundle = study/'source_bundle.zip'
    if not bundle.exists() or digest(bundle) != manifest['legacy_bundle_sha256']:
        return None
    actual = {name: digest(root/name) for name in manifest['optimized']}
    if actual != manifest['optimized']:
        raise ValueError('Optimized source differs from the verified compatibility manifest')
    record = dict(version=1, scientific_bundle_sha256=manifest['legacy_bundle_sha256'],
                  execution_source_sha256=actual,
                  validation='CPU regression/order/resume verified; CUDA equivalence requires Colab smoke')
    encoded = json.dumps(record, sort_keys=True, indent=2).encode()
    name = hashlib.sha256(encoded).hexdigest()
    directory = study/'execution_backends'
    directory.mkdir(exist_ok=True)
    target = directory/f'{name}.json'
    # Publish a complete record atomically; concurrent/restarted jobs must
    # never observe a partially written provenance file.
    fd, temporary = tempfile.mkstemp(prefix='.backend-', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != encoded:
                raise ValueError('Execution provenance was modified; refusing to overwrite')
    finally:
        os.unlink(temporary)
    return manifest


def execution_sources(root, study):
    from paper_repro.run_identity import source_hashes
    manifest = legacy_execution(root, study)
    if manifest is None:
        return source_hashes(root)
    names = ['paper_repro/'+n for n in LEGACY_SOURCE_NAMES]
    names += ['prepare_inc.py', 'paper_repro/requirements_colab.lock']
    return {name: manifest['legacy'][name] for name in names}


def execution_plan_code(root, study):
    manifest = legacy_execution(root, study)
    names = PLAN_NAMES if manifest else PLAN_NAMES + ('performance.py', 'compatibility.py', 'compatibility_manifest.json')
    return {name: manifest['legacy']['paper_repro/'+name] if manifest else
            digest(Path(root)/'paper_repro'/name) for name in names}
