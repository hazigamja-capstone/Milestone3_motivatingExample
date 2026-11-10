"""Prepare the pinned Linux/CPython 3.11 paper environment without altering Colab's torch.

Invoke with the Colab kernel's Python. The scientific process uses --venv/bin/python.
TLS/checksum validation stays enabled. No experiment or model-weight download runs here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

PAPER_COMMIT = '8b2ed63255d906b6b0afaefc6ae630b9ec4bdaa1'
UV_VERSION = '0.13.5'
ROOT = Path(__file__).resolve().parents[1]


def run(args, **kwargs):
    print('+', ' '.join(map(str,args)), flush=True)
    try:
        return subprocess.run(list(map(str,args)),check=True,**kwargs)
    except subprocess.CalledProcessError as exc:
        if kwargs.get('capture_output'):
            if exc.stdout:
                print(exc.stdout,flush=True)
            if exc.stderr:
                print(exc.stderr,file=sys.stderr,flush=True)
        raise


def prepare_upstream(paper_root):
    if not paper_root.exists():
        paper_root.parent.mkdir(parents=True,exist_ok=True)
        run(['git','clone','--no-checkout','https://github.com/mariodoebler/test-time-adaptation.git',paper_root])
        run(['git','-C',paper_root,'checkout','--detach',PAPER_COMMIT])
    top = run(['git','-C',paper_root,'rev-parse','--show-toplevel'],capture_output=True,text=True).stdout.strip()
    if Path(top).resolve() != paper_root.resolve():
        raise RuntimeError('paper-root must be the actual upstream repository root')
    sha = run(['git','-C',paper_root,'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
    if sha != PAPER_COMMIT:
        raise RuntimeError(f'Upstream is not pinned to {PAPER_COMMIT}; preserve existing work and use another paper-root')
    changed = run(['git','-C',paper_root,'status','--porcelain','--untracked-files=no'],capture_output=True,text=True).stdout
    if changed.strip():
        raise RuntimeError('Pinned upstream has modified tracked files; use a clean checkout')


def uv_command():
    path = shutil.which('uv')
    if path:
        version = subprocess.run([path,'--version'],check=True,capture_output=True,text=True).stdout.split()
        if len(version)>1 and version[1] == UV_VERSION:
            return [path]
    run([sys.executable,'-m','pip','install','--disable-pip-version-check','uv=='+UV_VERSION])
    return [sys.executable,'-m','uv']


def verification_code(paper_root, allow_cpu):
    return '''import importlib.metadata as metadata, json, pathlib, platform, sys
import torch, torchvision, open_clip
from PIL import features
paper = pathlib.Path(%r).resolve() / 'classification'
sys.path.insert(0,str(paper))
import robustbench, methods, conf
from models.model import get_model
from utils.registry import ADAPTATION_REGISTRY
expected = {'torch':'2.2.1+cu121','torchvision':'0.17.1+cu121','open-clip-torch':'2.24.0',
            'timm':'0.9.16','numpy':'1.26.4','Pillow':'10.2.0','nvidia-nvjitlink-cu12':'12.1.105'}
actual = {name:metadata.version(name) for name in expected}
assert actual == expected, (actual,expected)
assert sys.version_info[:2] == (3,11), sys.version
assert torch.version.cuda == '12.1', torch.version.cuda
assert pathlib.Path(robustbench.__file__).resolve().is_relative_to(paper), robustbench.__file__
autoattack_source = json.loads(metadata.distribution('autoattack').read_text('direct_url.json'))
autoattack_commit = autoattack_source['vcs_info']['commit_id']
assert autoattack_commit == 'a39220048b3c9f2cca9a4d3a54604793c68eca7e', autoattack_commit
assert features.check('webp'), 'Pillow WebP decoding unavailable'
for method in ('eata','tent','sar'):
    assert ADAPTATION_REGISTRY.get(method) is not None
# Exercise a compiled torchvision op to catch mismatched torch/torchvision wheels.
boxes = torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]])
assert torchvision.ops.nms(boxes,torch.tensor([.9,.8]),.5).tolist() == [0]
cuda = torch.cuda.is_available()
if not %r:
    assert cuda, 'CUDA is unavailable. In Colab select Runtime > Change runtime type > GPU, then rerun setup.'
if cuda:
    a = torch.randn((32,32),device='cuda')
    assert torch.isfinite(a @ a).all().item()
print(json.dumps({'python':platform.python_version(),'versions':actual,'cuda_runtime':torch.version.cuda,
                  'cuda_available':cuda,'gpu':torch.cuda.get_device_name() if cuda else None,
                  'vendored_robustbench':str(pathlib.Path(robustbench.__file__).resolve()),
                  'registered_methods':['eata','tent','sar'],'torchvision_nms':'passed','autoattack_commit':autoattack_commit,
                  'validation_mode':'cpu-imports-only' if not cuda else 'cuda-imports-and-matmul'},sort_keys=True))
''' % (str(paper_root), allow_cpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv',default='/content/venv_paper')
    parser.add_argument('--paper-root',default=str(ROOT/'third_party/test-time-adaptation'))
    parser.add_argument('--allow-cpu',action='store_true',help='Only for setup/import testing; not a GPU readiness claim')
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64','AMD64'):
        parser.error('The pinned cu121 CPython3.11 wheel lock targets Linux x86_64 (Colab)')
    paper_root = Path(args.paper_root).resolve(); venv = Path(args.venv).resolve()
    prepare_upstream(paper_root)
    # Large CUDA wheels share bandwidth with small packages in Colab. Limit
    # concurrency and allow slow transfers without weakening TLS/hash checks.
    os.environ.setdefault('UV_CONCURRENT_DOWNLOADS','4')
    os.environ.setdefault('UV_HTTP_TIMEOUT','120')
    uv = uv_command()
    vpy = venv/'bin/python'
    if vpy.exists():
        version = run([vpy,'-c','import sys; print("%s.%s"%sys.version_info[:2])'],capture_output=True,text=True).stdout.strip()
        if version != '3.11':
            parser.error('Existing venv uses Python '+version+'; choose a new --venv directory')
    else:
        run(uv+['venv','--python','3.11',str(venv),'--system-certs'])
    # The lock has exact official CUDA wheel URLs; all other artifacts use PyPI.
    # This dedicated venv is synchronized to the lock: stale external robustbench
    # and packages left by interrupted/older setup do not survive. Colab's kernel
    # environment is not modified.
    lock = ROOT/'paper_repro/requirements_colab.lock'
    run(uv+['pip','sync','--python',str(vpy),'--require-hashes','--system-certs',
            '--default-index','https://pypi.org/simple',str(lock)])
    run(uv+['pip','install','--python',str(vpy),'--system-certs','--no-deps',
            '-r',str(ROOT/'paper_repro/requirements_paper_nodeps.txt')])
    run(uv+['pip','check','--python',str(vpy)])
    result = run([vpy,'-c',verification_code(paper_root,args.allow_cpu)],capture_output=True,text=True)
    print(result.stdout,flush=True)
    info = json.loads(result.stdout.strip().splitlines()[-1])
    info['paper_commit'] = PAPER_COMMIT
    info['requirements_sha256'] = hashlib.sha256(lock.read_bytes()).hexdigest()
    freeze = run(uv+['pip','freeze','--python',str(vpy)],capture_output=True,text=True).stdout
    (venv/'paper-environment.freeze.txt').write_text(freeze,encoding='utf-8')
    (venv/'paper-environment.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Ready:',vpy, '| evidence:',venv/'paper-environment.json')


if __name__ == '__main__':
    main()
