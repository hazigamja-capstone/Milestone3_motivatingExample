"""Add a separately hashed execution backend to the original published notebook.

Never regenerate the original scientific payload for an existing Drive study.
Both archives remain immutable and are installed in separate source directories.
"""
import base64
import hashlib
import json
from pathlib import Path

from paper_repro.make_auto_colab import automatic_notebook
from paper_repro.make_colab import source_archive

ROOT = Path(__file__).resolve().parents[1]


def fast_notebook(original, payload=None):
    notebook = automatic_notebook(original)
    payload = source_archive(include_notebook_fixture=True) if payload is None else payload
    digest = hashlib.sha256(payload).hexdigest()
    notebook['cells'][0]['source'] += ['\n## 입력 파이프라인 최적화\n',
        '기존 source_bundle·metadata·체크포인트를 보존하고 별도 해시의 실행 소스를 사용합니다.\n',
        '배치64·학습률·정밀도·샘플 순서·업데이트 규칙은 유지합니다.\n',
        'CPU 호환성 회귀 검증 완료. CUDA 재개 동등성은 실제 Colab GPU에서 확인해야 합니다.\n']
    # Run the unmodified original bootstrap first, including its Drive/source
    # lock. Then install the backend via the same strict installer in a separate
    # directory. No replacing/unlinking source_bundle or source-hash checks.
    overlay = f'''
# Separate execution backend; original Drive source_bundle stays immutable.
FAST_PAYLOAD = base64.b64decode({base64.b64encode(payload).decode('ascii')!r})
FAST_SHA = hashlib.sha256(FAST_PAYLOAD).hexdigest()
assert FAST_SHA == {digest!r}
ORIGINAL_REPO = REPO
REPO = ORIGINAL_REPO.parent / (ORIGINAL_REPO.name + '-fast-' + FAST_SHA[:16])
install_bundle(FAST_PAYLOAD, REPO, OUT/'execution_backends'/FAST_SHA)
os.chdir(REPO)
print('Verified execution backend:', FAST_SHA, '|', REPO)
'''
    notebook['cells'][4]['source'] += overlay.splitlines(keepends=True)
    jobs = ''.join(notebook['cells'][8]['source'])
    jobs = jobs.replace("f'{combo}_notebook_smoke.json'", "f'{combo}_fast_{FAST_SHA[:16]}_smoke.json'")
    jobs = jobs.replace("if (OUT/f'{combo}_plan.json').exists() or (", "if (")
    jobs = jobs.replace('기존 계획 또는 같은 소스/설정의 스모크 기록', '같은 최적화 소스/설정의 GPU 스모크 기록')
    notebook['cells'][8]['source'] = jobs.splitlines(keepends=True)
    return notebook


if __name__ == '__main__':
    original = json.loads((ROOT/'colab_exp3_v3_ready.ipynb').read_text())
    target = ROOT/'colab_exp3_v3_fast.ipynb'
    target.write_text(json.dumps(fast_notebook(original), ensure_ascii=False, indent=1)+'\n')
    print(target)
