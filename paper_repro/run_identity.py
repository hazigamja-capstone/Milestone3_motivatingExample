"""Fail closed before reusing an output directory; preserve the original metadata."""
import json
from pathlib import Path


def prepare_run(run_dir, meta):
    path = Path(run_dir) / "meta.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("identity") != meta["identity"]:
            raise ValueError("Existing run has incompatible/legacy metadata; choose a new --out directory.")
        if old.get("domains", []) != meta["domains"]:
            raise ValueError("Existing metadata is immutable; use a new output root for different domains.")
        return
    elif list(Path(run_dir).glob("dom*.npz")) or (Path(run_dir) / "state.pt").exists():
        raise ValueError("Results without metadata cannot be resumed safely.")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def file_hash(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mark_complete(run_dir, identity, domains):
    root = Path(run_dir)
    files = [root / "meta.json"] + [root / f"dom{i:02d}_{dom}.npz" for i, dom in enumerate(domains)]
    marker = dict(identity=identity, files={p.name: file_hash(p) for p in files})
    tmp = root / "complete.json.tmp"
    tmp.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    tmp.replace(root / "complete.json")


def completed_run(run_dir, expected):
    """Validate finished output before costly model construction on Colab reconnect."""
    root = Path(run_dir)
    path = root / "complete.json"
    if not path.exists():
        return False
    marker = json.loads(path.read_text())
    identity = marker["identity"]
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError("Completed run has a different identity; use a new output root")
    if not (root / "state.pt").exists():
        raise ValueError("Completed run is missing its final checkpoint")
    for name, digest in marker["files"].items():
        path = root / name
        if path.name != name or not path.is_file() or file_hash(path) != digest:
            raise ValueError("Completed run output changed or is missing")
    return True


def source_hashes(root):
    names = ("run_paper.py", "checkpoint.py", "run_identity.py", "protocol.py", "stream_cache.py",
             "performance.py", "compatibility.py", "compatibility_manifest.json")
    paths = ["paper_repro/" + name for name in names] + ["prepare_inc.py"]
    lock = Path(root) / "paper_repro/requirements_colab.lock"
    if lock.exists():
        paths.append("paper_repro/requirements_colab.lock")
    return {name: file_hash(Path(root) / name) for name in paths}
