"""Build self-contained Colab notebooks from the CURRENT working-tree source.

No git push, notebook upload of a second file, or kernel dependency replacement
is needed. A compressed, checksummed source archive is embedded in each notebook.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def source_archive(include_notebook_fixture=False):
    paths = [Path('prepare_inc.py')]
    if include_notebook_fixture:
        # The fast bundle's regression suite verifies the immutable legacy payload.
        paths.append(Path('colab_exp3_v3_ready.ipynb'))
    paths += [p.relative_to(ROOT) for p in sorted((ROOT/'paper_repro').glob('*.py'))]
    paths += [Path('paper_repro')/n for n in ('requirements_paper.txt','requirements_paper_nodeps.txt',
                                               'requirements_colab.lock','compatibility_manifest.json')]
    paths += [p.relative_to(ROOT) for p in sorted((ROOT/'paper_repro').glob('*.md'))]
    paths += [p.relative_to(ROOT) for p in sorted((ROOT/'tests').glob('test_*.py'))]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for relative in paths:
            info = zipfile.ZipInfo(relative.as_posix(),date_time=(2026,11,10,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info,(ROOT/relative).read_bytes())
    return buffer.getvalue()


def cells(combo, payload=None):
    payload = source_archive() if payload is None else payload
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode('ascii')
    bootstrap = (ROOT/'paper_repro/colab_bundle.py').read_text()
    return [
        ('markdown', '''# CLIP TTA 실험 #3 — Colab 실행 노트북

**이 파일 하나를 Colab에 업로드**하고 GPU 런타임을 선택한 뒤 위에서부터 실행하세요.
현재 수정 코드가 내장되어 있어 원격 브랜치에 push할 필요가 없습니다.
설치는 별도 Python 3.11 가상환경에서 진행하며 Colab 기본 torch를 바꾸지 않습니다.

결과·체크포인트·로그·동결 소스는 Drive에 보관합니다. 원본 데이터 캐시는 `/content`에만 둡니다.
처음에는 `PHASE='smoke'`로 검증하세요. 이후 `paper → calibration → freeze → confirmation → evaluate` 순서입니다.
조합당 117만 장 처리이므로 한 번에 전체를 시작하지 않습니다. 세션이 끝나면 같은 파일을 다시 열어 재개하세요.'''),
        ('code', f'''# 사용자 설정: 네 조합 모두 같은 STUDY 이름을 사용합니다.
COMBO = {combo!r}       # A / B1 / B2 / B3; 권장 순서 B1 → B2 → A → B3
PHASE = 'smoke'         # smoke / paper / calibration / freeze / confirmation / evaluate / status
SEED = None            # calibration:101 또는102, confirmation:201..210; None이면 해당 단계 전체
GRAD_CKPT = 'auto'      # auto / on / off. 최초 계획 이후 자동으로 저장된 선택을 유지
STUDY = 'motivating_exp3_v3'  # 다른 설계/코드를 실행하려면 새 이름을 사용
DATA_REVISION = None    # None이면 첫 HF SHA를 동결. 이후 같은 SHA 자동 재사용
'''),
        ('code', '''# Drive 연결과 선택적 HF 토큰. 토큰 값은 출력/파일에 저장하지 않습니다.
import os, sys, subprocess
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
assert STUDY and Path(STUDY).name == STUDY and STUDY not in ('.','..')
OUT = Path('/content/drive/MyDrive')/STUDY
REPO = Path('/content')/('motivating_exp3_' + STUDY)
VPY = Path('/content/venv_paper/bin/python')
OUT.mkdir(parents=True,exist_ok=True)
# Compressed dataset cache and temporary files stay on fast local disk.
os.environ['TTA_CACHE_DIR'] = '/content/tta_data_cache'
os.environ['PYTHONUNBUFFERED'] = '1'
try:
    from google.colab import userdata
    token = userdata.get('HF_TOKEN')
    if token:
        os.environ['HF_TOKEN'] = token
    del token
except Exception:
    pass
'''),
        ('markdown', '''## 소스 복원

처음 실행하면 내장 소스와 이 노트북을 Drive에 저장합니다. 재접속 시 같은 소스만 허용합니다.
다른 버전이 감지되면 덮어쓰지 않고 중단합니다. Drive의 `source_bundle.zip`과
`experiment_notebook.ipynb`를 보관하면 해당 버전으로 돌아올 수 있습니다.'''),
        ('code', bootstrap + f'''\n# Embedded source payload; generated from the current project files.\nimport base64\nPAYLOAD = base64.b64decode({encoded!r})\nassert hashlib.sha256(PAYLOAD).hexdigest() == {digest!r}\nBUNDLE_SHA = install_bundle(PAYLOAD, REPO, OUT)\nos.chdir(REPO)\nprint('Source restored:', BUNDLE_SHA, '|', REPO)\n# Persist this exact notebook where supported by Colab.\ntry:\n    import json\n    from google.colab import _message\n    notebook = _message.blocking_request('get_ipynb', timeout_sec=20)['ipynb']\n    saved_notebook = OUT/'experiment_notebook.ipynb'\n    if not saved_notebook.exists():\n        saved_notebook.write_text(json.dumps(notebook,ensure_ascii=False),encoding='utf-8')\nexcept Exception:\n    print('Notebook auto-save unavailable; keep your original .ipynb alongside source_bundle.zip.')\n'''),
        ('markdown', '''## 고정 환경 설치 및 확인

첫 설치에는 수 분 이상 걸릴 수 있습니다. Python 3.11, torch 2.2.1/cu121와 논문 코드를 고정합니다.
해시 검증된 lock으로 설치하고 torchvision 연산·WebP·방법 등록·CUDA 연산을 검사합니다.
설치가 중단되면 이 셀을 다시 실행하세요. 분석 단계는 GPU 없이도 실행할 수 있습니다.'''),
        ('code', '''setup_command = [sys.executable, str(REPO/'paper_repro/setup_colab.py'),
                 '--venv', str(VPY.parent.parent),
                 '--paper-root', str(REPO/'third_party/test-time-adaptation')]
if PHASE in ('freeze','evaluate','status'):
    setup_command.append('--allow-cpu')
if PHASE == 'status':
    print('Status only: dependency installation is unnecessary.')
else:
    run_notebook_command(setup_command)
    # Keep setup evidence on Drive; it contains versions, not credentials.
    import shutil
    for name in ('paper-environment.json','paper-environment.freeze.txt'):
        target = OUT/'environment'/name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(VPY.parent.parent/name,target)
'''),
        ('markdown', '''## 실행 / 재개

- 처음에는 smoke. T4의 ViT-L은 auto가 gradient checkpointing을 선택하지만 OOM 가능성은 스모크로 확인합니다.
- paper는 seed1, calibration은101·102. freeze로 임계값을 저장한 뒤 confirmation201–210을 실행합니다.
- Colab 세션이 짧으면 SEED를 하나씩 지정하세요. 완료된 seed는 해시를 확인하고 건너뜁니다.
- 기본100배치마다 진행 상태를 저장합니다. 연결이 끊기면 같은 설정으로 재실행합니다.
  진행 중 domain은 데이터를 다시 읽을 수 있지만 저장된 지점 이전 GPU 적응은 반복하지 않습니다.
- `status`는 저장된 완료 표시/도메인 수를 보여줍니다. `evaluate`는 확인10개가 모두 있어야 합니다.
- 오류는 해당 셀을 실패시키며 Drive/logs에 기록됩니다. CUDA OOM 때 배치 크기를 임의로 줄이지 마세요.
  첫 본실험 전 checkpointing을 결정하거나 더 큰 GPU로 바꾸세요.'''),
        ('code', '''command = [sys.executable, str(REPO/'paper_repro/colab_session.py'),
           '--python', str(VPY), '--combo', COMBO, '--phase', PHASE,
           '--out', str(OUT), '--grad_ckpt', GRAD_CKPT]
if SEED is not None and PHASE in ('paper','calibration','confirmation'):
    command += ['--seed',str(SEED)]
if DATA_REVISION is not None:
    command += ['--data_revision',DATA_REVISION]
run_notebook_command(command)
'''),
        ('markdown', '''## 결과 확인

`<COMBO>_evaluation.json`에 누적 정확도·최악 구간 손해·미감지·지연·실제 오경보가 저장됩니다.
확인 seed는 같은 이미지 풀의 순서 반복이며 독립 배포 데이터셋이 아닙니다.
주 기준과 해석 범위는 내장 `paper_repro/PREREG.md`, 상세 실행 안내는 `paper_repro/COLAB.md`입니다.'''),
    ]


def write(combo, name, payload=None):
    notebook = dict(nbformat=4,nbformat_minor=5,
                    metadata=dict(accelerator='GPU',colab=dict(provenance=[]),
                                  kernelspec=dict(name='python3',display_name='Python 3')),
                    cells=[])
    for index,(kind,text) in enumerate(cells(combo,payload)):
        cell = dict(cell_type=kind,id=f'cell-{index:02d}',metadata={},source=text.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None,outputs=[])
        notebook['cells'].append(cell)
    (ROOT/name).write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')


if __name__ == '__main__':
    payload = source_archive()
    write('A','colab_exp3.ipynb',payload)
    write('B1','colab_exp3b.ipynb',payload)
    write('B1','colab_exp3_v3_ready.ipynb',payload)
