"""Wrap the published scientific payload without changing existing study hashes."""
import json
import ast
import base64
import hashlib
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def precision_fixed_bootstrap(cell):
    """Patch only the embedded runner; keep every other archived file intact."""
    source = ''.join(cell['source'])
    tree = ast.parse(source)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in node.targets))
    encoded = ast.literal_eval(assignment.value.args[0])
    payload = base64.b64decode(encoded)
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(payload)) as old, zipfile.ZipFile(output, 'w') as new:
        for info in old.infolist():
            data = old.read(info.filename)
            if info.filename == 'paper_repro/run_paper.py':
                before = b'                lt = tta(x).float()\n'
                after = (b'                # SAR needs autocast around its forward/adaptation call.\n'
                         b'                with torch.autocast("cuda", dtype=torch.float16):\n'
                         b'                    lt = tta(x).float()\n')
                if data.count(before) != 1:
                    raise ValueError('Expected exactly one unpatched TTA call in the original payload')
                data = data.replace(before, after)
            new.writestr(info, data)
    fixed = output.getvalue()
    source = source.replace(encoded, base64.b64encode(fixed).decode('ascii'))
    source = source.replace(hashlib.sha256(payload).hexdigest(), hashlib.sha256(fixed).hexdigest())
    cell['source'] = source.splitlines(keepends=True)


def automatic_notebook(original, fix_precision=True):
    nb = json.loads(json.dumps(original))
    if fix_precision:
        precision_fixed_bootstrap(nb['cells'][4])
    nb['cells'][0]['source'] = '''# CLIP TTA v3 — 자동 실행·재개

GPU 런타임을 선택하고 위에서부터 실행하면 됩니다.
B1 → B2 → A → B3 각각 smoke → paper → 보정101/102 → 동결 → 확인201–210 → 평가를 수행합니다.
이미 완료된 계산은 검증 후 건너뛰며 오류가 발생하면 즉시 중단합니다.
전체468만 장의 장시간 작업입니다. 중단 후 같은 STUDY로 다시 실행하세요.
SAR의 FP32 입력/FP16 가중치 오류를 수정한 내장 소스를 사용합니다.
수정 전 결과와 섞지 않도록 새 STUDY를 사용합니다. 기존 Drive 결과는 삭제하지 마세요.
'''.splitlines(keepends=True)
    nb['cells'][1]['source'] = '''# 최초 한 번만 설정. 이후 PHASE/SEED를 수동 변경할 필요가 없습니다.
COMBOS = ['B1', 'B2', 'A', 'B3']  # 한 조합만 원하면 ['B1']
GRAD_CKPT = 'auto'
STUDY = 'motivating_exp3_v3_sar_fp16fix'  # 이전 소스의 STUDY와 분리; 재개 시 이 이름 유지
DATA_REVISION = None
PHASE = 'auto'                  # 설치 셀의 GPU 검사에 사용
JOBS = []
for combo in COMBOS:
    JOBS += [(combo, 'smoke', None), (combo, 'paper', 1)]
    JOBS += [(combo, 'calibration', seed) for seed in (101, 102)]
    JOBS += [(combo, 'freeze', None)]
    JOBS += [(combo, 'confirmation', seed) for seed in range(201, 211)]
    JOBS += [(combo, 'evaluate', None)]
print(f'자동 실행: {COMBOS}, 총 {len(JOBS)}개 작업 (완료 작업은 건너뜀)')
'''.splitlines(keepends=True)
    nb['cells'][7]['source'] = '''## 자동 실행 / 재개

아래 셀 하나가 모든 단계와 seed를 순서대로 처리합니다.
중단 후 위에서부터 같은 설정으로 재실행하면 완료 seed를 검증하고 이어갑니다.
Drive의 결과와 source_bundle.zip을 삭제하지 마세요.
실패한 작업 뒤의 실험은 진행하지 않습니다. 완료해도 GPU가 자동 해제되지는 않으므로 런타임을 종료하세요.
'''.splitlines(keepends=True)
    nb['cells'][8]['source'] = '''# 노트북 호스트에서 하위 프로세스 출력을 직접 받아 셀에 표시합니다.
import subprocess, signal, json, hashlib

def run_visible(command):
    process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        process.stdout.close()

for index, (combo, phase, seed) in enumerate(JOBS, 1):
    print(f'\\n[{index}/{len(JOBS)}] {combo} / {phase} / seed={seed}', flush=True)
    if phase == 'smoke':
        marker = OUT/f'{combo}_notebook_smoke.json'
        if (OUT/f'{combo}_plan.json').exists() or (
            marker.exists() and json.loads(marker.read_text()).get('bundle_sha') == BUNDLE_SHA
            and json.loads(marker.read_text()).get('grad_ckpt') == GRAD_CKPT):
            print('SKIP: 기존 계획 또는 같은 소스/설정의 스모크 기록', flush=True)
            continue
    if phase == 'evaluate' and (OUT/f'{combo}_evaluation.json').exists():
        result = json.loads((OUT/f'{combo}_evaluation.json').read_text())
        digest = hashlib.sha256((OUT/f'{combo}_calibration.json').read_bytes()).hexdigest()
        if (result.get('version') != 3 or result.get('calibration_sha256') != digest
            or {r['seed'] for r in result['runs']} != set(range(201, 211))):
            raise ValueError('기존 평가 파일이 현재 동결/seed 집합과 다릅니다. 덮어쓰지 않습니다.')
        print('SKIP: 동일 동결 임계값으로 완료된 평가', flush=True)
        continue
    command = [sys.executable, str(REPO/'paper_repro/colab_session.py'),
        '--python', str(VPY), '--combo', combo, '--phase', phase,
        '--out', str(OUT), '--grad_ckpt', GRAD_CKPT]
    if seed is not None:
        command += ['--seed', str(seed)]
    if DATA_REVISION is not None:
        command += ['--data_revision', DATA_REVISION]
    run_visible(command)
    if phase == 'smoke':
        (OUT/f'{combo}_notebook_smoke.json').write_text(json.dumps(
            {'bundle_sha': BUNDLE_SHA, 'grad_ckpt': GRAD_CKPT}))
print('선택한 모든 조합의 평가까지 완료했습니다. GPU 런타임을 종료하세요.', flush=True)
'''.splitlines(keepends=True)
    if not fix_precision:
        nb['cells'][1]['source'] = [line.replace('motivating_exp3_v3_sar_fp16fix', 'motivating_exp3_v3')
                                   for line in nb['cells'][1]['source']]
        nb['cells'][0]['source'] = original['cells'][0]['source'][:]
    return nb


if __name__ == '__main__':
    original = json.loads((ROOT/'colab_exp3_v3_ready.ipynb').read_text())
    target = ROOT/'colab_exp3_v3_auto.ipynb'
    target.write_text(json.dumps(automatic_notebook(original),ensure_ascii=False,indent=1))
    print(target)
