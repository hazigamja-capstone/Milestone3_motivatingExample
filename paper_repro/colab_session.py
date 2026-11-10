"""Colab session-sized jobs. All experiment imports run in the isolated venv."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COMBOS = {'A': ('eta','ViT-L-14'), 'B1': ('eta','ViT-B-16'),
          'B2': ('tent','ViT-B-16'), 'B3': ('sar','ViT-L-14')}


def run_logged(command, log):
    """Unmasked exit status, live stdout, persistent log, and child cancellation."""
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    print('RUN', ' '.join(map(str, command)), flush=True)
    with log.open('a', encoding='utf-8') as f:
        process = subprocess.Popen(list(map(str, command)), cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                f.write(line); f.flush()
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            process.stdout.close()
        if code:
            raise subprocess.CalledProcessError(code, command)


def gpu_info(python):
    code = '''import json, torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA GPU missing: select Colab Runtime > Change runtime type > GPU')
p=torch.cuda.get_device_properties(0)
print(json.dumps(dict(name=p.name,total_gib=p.total_memory/1024**3,torch=torch.__version__)))'''
    return json.loads(subprocess.check_output([str(python), '-c', code], text=True))


def checkpoint_choice(requested, plan, arch, gpu=None):
    if plan is not None:
        saved = plan['grad_ckpt']
        if requested != 'auto' and (requested == 'on') != saved:
            raise ValueError('Checkpointing is already frozen in the plan; keep the original choice')
        return saved
    if requested != 'auto':
        return requested == 'on'
    return arch == 'ViT-L-14' and gpu is not None and gpu['total_gib'] < 24


def data_revision(python, out, requested=None):
    path = out / 'dataset_revision.json'
    if path.exists():
        value = json.loads(path.read_text())['data_revision']
        if requested and requested != value:
            raise ValueError('Dataset SHA already frozen for this study')
        return value
    code = 'from paper_repro.run_plan import revision; print(revision())'
    value = requested or subprocess.check_output([str(python), '-c', code], cwd=ROOT, text=True).strip()
    if len(value) != 40 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Dataset revision must be a full immutable SHA')
    with path.open('x') as f:
        json.dump({'data_revision':value}, f, indent=2)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, help='Isolated Python3.11 executable')
    parser.add_argument('--combo', choices=COMBOS, required=True)
    parser.add_argument('--phase', choices=('smoke','paper','calibration','freeze','confirmation','evaluate','status'), required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--grad_ckpt', choices=('auto','on','off'), default='auto')
    parser.add_argument('--data_revision')
    args = parser.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.phase == 'status':
        for role, seeds in [('paper',[1]),('calibration',[101,102]),('confirmation',list(range(201,211)))]:
            method, arch = COMBOS[args.combo]
            for seed in seeds:
                root = out / f'{method}_{arch}_{role}_s{seed}'
                state = 'complete marker' if (root/'complete.json').exists() else f'{len(list(root.glob("dom*.npz")))}/3 domains written'
                print(role, seed, state)
        print('Frozen calibration:', (out/f'{args.combo}_calibration.json').exists())
        print('Evaluation:', (out/f'{args.combo}_evaluation.json').exists())
        return
    plan_path = out/f'{args.combo}_plan.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    compute = args.phase in ('smoke','paper','calibration','confirmation')
    gpu = gpu_info(args.python) if compute else None
    if gpu:
        print('GPU:', json.dumps(gpu), flush=True)
        if shutil.disk_usage(tempfile.gettempdir()).free < 5*1024**3:
            parser.error('Need at least 5 GiB free local temporary disk for compressed stream spooling')
    method, arch = COMBOS[args.combo]
    ckpt = checkpoint_choice(args.grad_ckpt, plan, arch, gpu)
    sha = data_revision(args.python, out, args.data_revision)
    print('Dataset:', sha, '| gradient checkpointing:', ckpt, flush=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log = out/'logs'/f'{args.combo}_{args.phase}_{args.seed or "all"}_{timestamp}.log'
    if args.phase == 'smoke':
        # Fresh path every time: smoke may never be an accidental completed-run no-op.
        smoke_root = Path(tempfile.mkdtemp(prefix=f'tta-smoke-{args.combo}-'))
        command = [args.python, ROOT/'paper_repro/run_paper.py','--method',method,'--arch',arch,
                   '--role','paper','--n_domains','1','--limit_classes','3',
                   '--data_revision',sha,'--out',smoke_root]
        if ckpt:
            command.append('--grad_ckpt')
        run_logged(command, log)
        print('Smoke outputs:', smoke_root, flush=True)
        return
    command = [args.python, ROOT/'paper_repro/run_plan.py','--combo',args.combo,
               '--phase',args.phase,'--out',out,'--data_revision',sha]
    if ckpt:
        command.append('--grad_ckpt')
    if args.seed is not None:
        command += ['--seed',str(args.seed)]
    run_logged(command, log)
    print('Saved log:', log, flush=True)


if __name__ == '__main__':
    main()
