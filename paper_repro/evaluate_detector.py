"""v3: freeze unlabeled thresholds, then evaluate untouched confirmation runs.

Run from repository root. Calibration never selects rows using delta or labels.
"""
import argparse
import glob
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from paper_repro.analyze_paper import load_run, validate_runs, segments, find_events, SEG
from paper_repro.protocol import CALIBRATION_SEEDS, CONFIRMATION_SEEDS, VERSION

SIGNALS = ('disagree', 'ent_t', 'ent_z', 'progress', 'kl', 'topk')
PRIMARY = 'disagree'
ALPHA = .05
CALIB_SEGMENTS = 100
LOOKBACK = 2
BOOTSTRAPS = 2000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(meta):
    keys = ('method', 'arch', 'setting', 'severity', 'bs', 'lr', 'paper_commit',
            'torch', 'grad_ckpt', 'data_revision', 'protocol_version')
    out = {k: meta.get(k) for k in keys}
    out['domains'] = sorted(meta['domains'])
    out['cfg'] = '\n'.join(line for line in meta['cfg'].splitlines()
                           if not line.startswith(('RNG_SEED:', 'SAVE_DIR:', 'LOG_TIME:', 'LOG_DEST:')))
    out['source_sha256'] = meta['identity']['source_sha256']
    return out


def rows_for(data):
    rows = segments(data)
    for r in rows:
        # Planned dataset size is known before inference; no future outcomes used.
        r['progress'] = (r['seg_idx'] + 1) * SEG / len(data['y'])
    return rows


def unlabeled_thresholds(row_sets):
    if any(len(rows) < CALIB_SEGMENTS for rows in row_sets):
        raise ValueError('Each calibration run needs at least 100 complete segments; no FPR fallback')
    selected = [r for rows in row_sets for r in rows[:CALIB_SEGMENTS]]
    thresholds = {}
    for signal in SIGNALS:
        values = np.array([r[signal] for r in selected], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f'Missing/nonfinite calibration signal: {signal}')
        if signal in ('disagree', 'progress', 'kl'):
            lo, hi = None, float(np.quantile(values, 1-ALPHA))
        elif signal == 'topk':
            lo, hi = float(np.quantile(values, ALPHA)), None
        else:
            lo, hi = map(float, np.quantile(values, [ALPHA/2, 1-ALPHA/2]))
        thresholds[signal] = [lo, hi]
    return thresholds


def alarms_for(rows, thresholds, signal):
    lo, hi = thresholds[signal]
    return [bool((lo is not None and r[signal] < lo) or
                 (hi is not None and r[signal] > hi)) for r in rows]


def event_detection(on, event, lookback=LOOKBACK):
    """Only new alarm onsets in a bounded lookback qualify; time is sample-end."""
    start, end = event['start'], event['end']
    for k in range(max(0, start-lookback), end+1):
        if on[k] and (k == 0 or not on[k-1]):
            available = (k+1)*SEG
            return dict(alarm_segment=k, available_sample=available,
                        delay_from_start_samples=available-start*SEG,
                        delay_from_confirmation_samples=available-(start+3)*SEG)
    return None


def signal_metrics(rows, events, on):
    safe = np.array([r['delta'] > -5 for r in rows])
    false = np.asarray(on) & safe
    false_clusters = sum(bool(v) and (i == 0 or not false[i-1]) for i,v in enumerate(false))
    detected = [event_detection(on,e) for e in events]
    delays = [v['delay_from_start_samples'] if v is not None else math.inf for v in detected]
    median = sorted(delays)[len(delays)//2] if delays else math.inf
    return dict(events=len(events), detected=sum(v is not None for v in detected),
                missed=sum(v is None for v in detected), event_details=detected,
                median_delay_samples=None if not math.isfinite(median) else median,
                median_delay_is_infinite=bool(delays and not math.isfinite(median)),
                false_alarm_segment_rate=float(false.mean()/safe.mean()) if safe.any() else None,
                false_alarm_clusters=false_clusters,
                false_alarm_clusters_per_10000_safe_samples=(false_clusters/(safe.sum()*SEG)*10000
                                                            if safe.any() else None))


def utility(data, on, batch_size):
    """First alarm irreversibly switches at the next batch; no labels in policy.

    Shadow TTA continues for measurement. Only its predictions before switch are
    consumed by policy, so stopping its updates after switch is equivalent.
    """
    n = len(data['y'])
    edges = [i for i,v in enumerate(on) if v]
    switch = min(n, math.ceil(((edges[0]+1)*SEG)/batch_size)*batch_size) if edges else n
    t = data['pred_t'] == data['y']; z = data['pred_z'] == data['y']
    policy = np.concatenate([t[:switch],z[switch:]])
    # Label-aware per-batch choice is an unrealizable upper bound.
    oracle_correct = sum(max(int(t[i:i+batch_size].sum()), int(z[i:i+batch_size].sum()))
                         for i in range(0,n,batch_size))
    def stats(correct):
        delta = (correct.astype(float)-z.astype(float))*100
        windows = [float(delta[i:i+SEG].mean()) for i in range(0,n-SEG+1,SEG)]
        return dict(accuracy=float(correct.mean()*100), delta_vs_zs_pp=float(delta.mean()),
                    worst_segment_delta_pp=min(windows) if windows else None)
    return dict(zero_shot=stats(z), always_tta=stats(t), switch_policy=stats(policy),
                oracle_batch_accuracy=oracle_correct/n*100, switch_sample=switch,
                switched=bool(switch<n))


def resolve(patterns):
    paths = []
    for pattern in patterns:
        found = sorted(glob.glob(pattern)) or [pattern]
        if any(not Path(p).is_dir() for p in found):
            raise ValueError(f'No run directory: {pattern}')
        paths.extend(found)
    return paths


def checked(patterns, role, expected_seeds):
    runs = validate_runs(resolve(patterns))
    if {m['seed'] for _,_,_,m in runs} != set(expected_seeds):
        raise ValueError(f'{role} requires exactly seeds {list(expected_seeds)}')
    for path,data,_,meta in runs:
        if meta.get('role') != role or meta.get('protocol_version') != VERSION:
            raise ValueError(f'{path}: expected v{VERSION} {role} stream')
        if SEG % meta['bs'] != 0:
            raise ValueError('Batch size must divide 512 for causal alarm timing')
    return runs


def calibration_artifact(runs):
    return dict(version=VERSION, signature=signature(runs[0][3]), alpha=ALPHA,
                calibration_segments_per_run=CALIB_SEGMENTS, primary=PRIMARY, lookback=LOOKBACK,
                evaluator_sha256=digest(__file__),
                analysis_sha256=digest(Path(__file__).with_name('analyze_paper.py')),
                thresholds=unlabeled_thresholds([rows_for(d) for _,d,_,_ in runs]),
                calibration=[dict(seed=m['seed'], classes=m['classes'],
                                  meta_sha256=digest(Path(p)/'meta.json'),
                                  files={f.name:digest(f) for f in sorted(Path(p).glob('dom*.npz'))})
                             for p,_,_,m in runs])


def check_artifact(artifact):
    if (artifact.get('version') != VERSION or artifact.get('alpha') != ALPHA
            or artifact.get('primary') != PRIMARY or artifact.get('lookback') != LOOKBACK
            or artifact.get('calibration_segments_per_run') != CALIB_SEGMENTS
            or artifact.get('evaluator_sha256') != digest(__file__)
            or artifact.get('analysis_sha256') != digest(Path(__file__).with_name('analyze_paper.py'))):
        raise ValueError('Frozen artifact does not match the registered evaluator/protocol')
    if {c['seed'] for c in artifact['calibration']} != set(CALIBRATION_SEEDS):
        raise ValueError('Invalid calibration provenance')


def evaluate(runs, artifact):
    check_artifact(artifact)
    results = []
    for path,data,_,meta in runs:
        if signature(meta) != artifact['signature']:
            raise ValueError('Confirmation protocol differs from frozen calibration')
        if any(set(meta['classes']) & set(c['classes']) for c in artifact['calibration']):
            raise ValueError('Calibration and confirmation classes overlap')
        rows = rows_for(data); events = find_events(rows)
        signals = {s:signal_metrics(rows,events,alarms_for(rows,artifact['thresholds'],s)) for s in SIGNALS}
        results.append(dict(seed=meta['seed'], name=Path(path).name, signals=signals,
                            events=events,
                            utility=utility(data,alarms_for(rows,artifact['thresholds'],PRIMARY),meta['bs'])))
    deltas = np.array([r['utility']['switch_policy']['delta_vs_zs_pp'] for r in results])
    rng = np.random.default_rng(1729)
    boot = rng.choice(deltas, size=(BOOTSTRAPS,len(deltas)), replace=True).mean(1)
    return dict(version=VERSION, primary=PRIMARY, alpha=ALPHA, runs=results,
                switch_delta_vs_zs_mean_pp=float(deltas.mean()),
                switch_delta_vs_zs_order_bootstrap_95ci=list(map(float,np.quantile(boot,[.025,.975]))),
                interpretation='Conditional on the fixed ImageNet-C class/image pool; order seeds are not independent deployment datasets. Nominal alpha is not guaranteed under shift.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    cal = sub.add_parser('calibrate'); cal.add_argument('runs',nargs='+'); cal.add_argument('--output',required=True)
    test = sub.add_parser('evaluate'); test.add_argument('runs',nargs='+'); test.add_argument('--calibration',required=True); test.add_argument('--output',required=True)
    args = parser.parse_args()
    try:
        if args.command == 'calibrate':
            result = calibration_artifact(checked(args.runs,'calibration',CALIBRATION_SEEDS))
        else:
            artifact = json.loads(Path(args.calibration).read_text())
            result = evaluate(checked(args.runs,'confirmation',CONFIRMATION_SEEDS),artifact)
            result['calibration_sha256'] = digest(args.calibration)
        # Exclusive create: never silently replace a frozen threshold or result.
        with open(args.output,'x',encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    except (ValueError,KeyError,FileExistsError) as exc:
        parser.error(str(exc))
    print(args.output)


if __name__ == '__main__':
    main()
