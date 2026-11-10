"""Standard-library-only bootstrap, also embedded verbatim in Colab notebooks."""
import hashlib
import io
from pathlib import Path
import zipfile


def install_bundle(payload, repo, study):
    repo, study = Path(repo), Path(study)
    study.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    saved = study / 'source_bundle.zip'
    if saved.exists():
        if hashlib.sha256(saved.read_bytes()).hexdigest() != digest:
            raise ValueError('Drive study has different source code. Open the original saved notebook or choose a NEW study directory.')
    else:
        with saved.open('xb') as f:
            f.write(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or '..' in relative.parts or member.is_dir():
                raise ValueError('Invalid source bundle path')
            target = repo / relative
            if not target.resolve().is_relative_to(repo.resolve()):
                raise ValueError('Source path escapes repository')
            data = archive.read(member)
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError(f'Local source was modified: {relative}. Preserve it and select a fresh runtime/source directory.')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return digest


def run_notebook_command(command):
    """Forward Colab stop-cell to the helper so it can stop its scientific job group."""
    import os
    import signal
    import subprocess
    process = subprocess.Popen(list(map(str, command)), start_new_session=True)
    try:
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                # colab_session.run_logged has its own 10s scientific-group cleanup.
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
