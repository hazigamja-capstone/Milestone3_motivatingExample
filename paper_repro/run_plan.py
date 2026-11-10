"""Execute a fixed v3 plan, one combination at a time. No experiments run on import."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_repro.protocol import CALIBRATION_SEEDS, CONFIRMATION_SEEDS, VERSION
from paper_repro.evaluate_detector import check_artifact
from paper_repro.run_identity import completed_run, source_hashes

COMBOS = {'A': ('eta','ViT-L-14'), 'B1': ('eta','ViT-B-16'),
          'B2': ('tent','ViT-B-16'), 'B3': ('sar','ViT-L-14')}
PHASES = ('paper','calibration','freeze','confirmation','evaluate')


def revision():
    headers = {}
    if os.environ.get('HF_TOKEN'):
        headers['Authorization'] = 'Bearer ' + os.environ['HF_TOKEN']
    request = urllib.request.Request('https://huggingface.co/api/datasets/haideraltahan/wds_imagenetc',headers=headers)
    with urllib.request.urlopen(request,timeout=60) as response:
        return json.load(response)['sha']


def commands(combo, phase, out, data_revision, grad_ckpt=False, seed=None):
    method, arch = COMBOS[combo]
    base = [sys.executable,str(ROOT/'paper_repro/run_paper.py'),'--method',method,'--arch',arch,
            '--n_domains','3','--data_revision',data_revision,'--out',str(out)]
    if grad_ckpt:
        base += ['--grad_ckpt']
    if phase in ('paper','calibration','confirmation'):
        seeds = {'paper':(1,), 'calibration':CALIBRATION_SEEDS,'confirmation':CONFIRMATION_SEEDS}[phase]
        if seed is not None:
            if seed not in seeds:
                raise ValueError(f"Seed {seed} is not registered for {phase}")
            seeds = (seed,)
        return [base+['--role',phase,'--seed',str(item)] for item in seeds]
    artifact = out/f'{combo}_calibration.json'
    evaluator = [sys.executable,str(ROOT/'paper_repro/evaluate_detector.py')]
    if phase == 'freeze':
        return [evaluator+['calibrate',str(out/f'{method}_{arch}_calibration_s*'),'--output',str(artifact)]]
    if phase == 'evaluate':
        return [evaluator+['evaluate',str(out/f'{method}_{arch}_confirmation_s*'),
                          '--calibration',str(artifact),'--output',str(out/f'{combo}_evaluation.json')]]
    raise ValueError(phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--combo',choices=COMBOS,required=True)
    parser.add_argument('--phase',choices=PHASES+('all',),required=True)
    parser.add_argument('--out',required=True,help='Shared study output root for all four combinations')
    parser.add_argument('--data_revision',help='Resolved and frozen automatically before first run if omitted')
    parser.add_argument('--grad_ckpt',action='store_true')
    parser.add_argument('--seed', type=int, help='Run one registered seed in a compute phase (Colab session-sized job)')
    args = parser.parse_args()
    if args.seed is not None:
        if args.phase not in ('paper', 'calibration', 'confirmation'):
            parser.error('--seed is only valid for a single compute phase')
        try:
            commands(args.combo, args.phase, Path(args.out), '0' * 40, args.grad_ckpt, args.seed)
        except ValueError as exc:
            parser.error(str(exc))
    out = Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    plan_path = out/f'{args.combo}_plan.json'
    old = json.loads(plan_path.read_text()) if plan_path.exists() else None
    dataset_path = out/'dataset_revision.json'
    dataset = json.loads(dataset_path.read_text()) if dataset_path.exists() else None
    data_revision = args.data_revision or (dataset['data_revision'] if dataset else
                                            old['data_revision'] if old else revision())
    if len(data_revision) != 40 or any(c not in '0123456789abcdef' for c in data_revision):
        parser.error('Dataset revision must be an immutable 40-character lowercase SHA')
    if dataset and dataset['data_revision'] != data_revision:
        parser.error('All combinations must share the study dataset SHA; use a new study root')
    if dataset is None:
        with dataset_path.open('x') as f:
            json.dump({'data_revision':data_revision},f,indent=2)
    names = ('run_plan.py','run_paper.py','protocol.py','checkpoint.py','run_identity.py',
             'analyze_paper.py','evaluate_detector.py','stream_cache.py')
    plan = dict(version=VERSION,combo=args.combo,data_revision=data_revision,grad_ckpt=args.grad_ckpt,
                n_domains=3,calibration_seeds=list(CALIBRATION_SEEDS),confirmation_seeds=list(CONFIRMATION_SEEDS),
                code={n:hashlib.sha256((ROOT/'paper_repro'/n).read_bytes()).hexdigest() for n in names},
                prepare_inc_sha256=hashlib.sha256((ROOT/'prepare_inc.py').read_bytes()).hexdigest())
    if old and old != plan:
        parser.error('Frozen plan changed; use a new output root instead of modifying an existing experiment')
    if old is None:
        with plan_path.open('x') as f:
            json.dump(plan,f,indent=2)
    phases = PHASES if args.phase == 'all' else (args.phase,)
    for phase in phases:
        artifact = out/f'{args.combo}_calibration.json'
        if phase in ('confirmation','evaluate'):
            if not artifact.exists():
                parser.error('Freeze calibration before starting confirmation')
            frozen = json.loads(artifact.read_text())
            check_artifact(frozen)
            method, arch = COMBOS[args.combo]
            if (frozen['signature']['method'] != method or frozen['signature']['arch'] != arch
                    or frozen['signature']['data_revision'] != data_revision):
                parser.error('Frozen calibration belongs to a different combination or dataset')
            for name, sha in frozen['signature']['source_sha256'].items():
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != sha:
                    parser.error('Frozen calibration was produced by different source code')
        if phase == 'freeze' and artifact.exists():
            check_artifact(json.loads(artifact.read_text()))
            print('Reusing frozen calibration:',artifact)
            continue
        for command in commands(args.combo,phase,out,data_revision,args.grad_ckpt,args.seed):
            if phase in ('paper','calibration','confirmation'):
                method, arch = COMBOS[args.combo]
                seed = int(command[-1])
                import torch
                expected = dict(method=method, arch=arch, role=phase, seed=seed,
                                data_revision=data_revision, grad_ckpt=args.grad_ckpt,
                                n_domains=3, torch=torch.__version__, limit_classes=0, limit_batches=0,
                                checkpoint_batches=100, cli_opts=[], source_sha256=source_hashes(ROOT))
                run_dir = out / f'{method}_{arch}_{phase}_s{seed}'
                if completed_run(run_dir, expected):
                    print('SKIP verified complete run:', run_dir, flush=True)
                    continue
            print('RUN', ' '.join(command),flush=True)
            subprocess.run(command,cwd=ROOT,check=True)


if __name__ == '__main__':
    main()
