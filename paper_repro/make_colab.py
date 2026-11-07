"""Colab 노트북을 만든다 (노트북 JSON을 손으로 고치지 않으려고). 실행: python paper_repro/make_colab.py

- colab_exp3.ipynb: 기준 조합 A (ViT-L/14 + ETA)
- colab_exp3b.ipynb: 추가 조합 B1–B3 (PREREG 변경 기록 2026-11-08). 노트북 안에서 COMBO만 바꿔 고른다
"""
import json
import os

BRANCH = "hyunjin_expriments#2"
BRANCH_B = "seoy_experiments3"
PAPER_COMMIT = "8b2ed63"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_cells(branch):
    """코드 받기, 논문 환경, 드라이브 (두 노트북 공통)."""
    return [
        ("code", f"""# 1. 코드 받기 (저장소 + 논문 코드)
import os
if not os.path.isdir('/content/repo'):
    !git clone -q -b '{branch}' https://github.com/hazigamja-capstone/Milestone3_motivatingExample.git /content/repo
else:
    !git -C /content/repo pull -q
P = '/content/repo/third_party/test-time-adaptation'
if not os.path.isdir(P):
    !git clone -q https://github.com/mariodoebler/test-time-adaptation.git {{P}}
!git -C {{P}} checkout -q {PAPER_COMMIT} && git -C {{P}} log --oneline -1
%cd /content/repo"""),
        ("code", """# 2. 논문 환경 (Colab 기본 torch와 섞이지 않게 별도 가상환경, 처음 한 번 수 분)
VPY = '/content/venv_paper/bin/python'
if not os.path.exists(VPY):
    !pip -q install uv
    !uv venv -q --python 3.11 /content/venv_paper
    !uv pip install -q --python {VPY} -r paper_repro/requirements_paper.txt --index-strategy unsafe-best-match
    !uv pip install -q --python {VPY} --no-deps -r paper_repro/requirements_paper_nodeps.txt
!{VPY} -c "import torch, open_clip; print(torch.__version__, open_clip.__version__, torch.cuda.get_device_name())"
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
        ("code", """# 3. 드라이브 + (선택) HF 토큰: 왼쪽 열쇠 아이콘(보안 비밀)에 HF_TOKEN을 넣어 두면 쓴다. 없어도 받을 수 있지만 느릴 수 있다
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/motivating_exp3'
os.makedirs(OUT, exist_ok=True)
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    print('HF_TOKEN 사용')
except Exception:
    print('HF_TOKEN 없음')"""),
    ]


# ---------------------------------------------------------------- A: 기준 조합

cells_a = [
    ("md", f"""# 실험 #3: A Lost Opportunity 재현 (ViT-L/14 + ETA, correlated)

런타임 > 런타임 유형 변경 > GPU (A100 또는 L4 권장)

- 설정과 판정 기준은 `paper_repro/PREREG.md`. **PREREG가 확정·커밋된 뒤에 돌린다.**
- 논문 코드(커밋 `{PAPER_COMMIT}`)는 수정 없이 쓰고, 논문 환경(torch 2.2.1, open_clip 2.24.0)을 별도 가상환경으로 만든다.
- 결과는 드라이브 `MyDrive/motivating_exp3/`에 바로 저장한다. 손상 하나가 끝날 때마다 상태가 저장되므로,
  런타임이 끊기면 처음부터 다시 실행하면 이어서 돈다 (진행 중이던 손상 하나는 처음부터 다시)."""),
    *setup_cells(BRANCH),
    ("md", """## 스모크 테스트 (1~2분)

클래스 3개(150장)만 돌려서 환경·다운로드·기록이 되는지 본다. 결과는 `/content/smoke`에만 남는다."""),
    ("code", """!{VPY} paper_repro/run_paper.py --method eta --n_domains 1 --limit_classes 3 --out /content/smoke 2>&1 | grep -v -i warn | tail -3
!ls /content/smoke/*"""),
    ("md", """## 1단계: 기준점 재현

- PREREG 1단계: 앞쪽 손상 3개(gaussian, shot, impulse)를 논문 seed 1로. 손상당 5만 장.
- GPU 메모리가 16GB 미만(T4 등)이면 `GRAD_CKPT = True`. 계산 결과는 같고 메모리만 줄인다 (PREREG 3절 3번).
- 논문 Table 3 비교용으로 15개 전부 돌리려면 `N_DOMAINS = 15`. 같은 폴더에서 이어서 돈다."""),
    ("code", """N_DOMAINS = 3
GRAD_CKPT = False
SEED = 1
ck = '--grad_ckpt' if GRAD_CKPT else ''
!{VPY} paper_repro/run_paper.py --method eta --seed {SEED} --n_domains {N_DOMAINS} {ck} --out {OUT} 2>&1 | grep -v -i warn | tee -a {OUT}/run_eta_s{SEED}.log"""),
    ("md", """## 판정 (GPU 불필요)

- 사건 유형은 정답 라벨 대비(주 기준)와 absolute(민감도)를 함께 출력한다. 오경보율은 1·5·10%를 모두 출력한다 (PREREG 4절).
- 3단계(빈도)에서는 seed 1을 빼고(`--exclude_seed 1`) 새 seed만 센다. 새 seed는 위 셀에서 `SEED`만 바꿔 돌린다."""),
    ("code", """!{VPY} paper_repro/analyze_paper.py "{OUT}/eta_ViT-L-14_correlated_s*" --exclude_seed 1 --csv {OUT}/segments | tee {OUT}/analyze_log.txt"""),
]

# ---------------------------------------------------------------- B1–B3: 추가 조합

cells_b = [
    ("md", f"""# 실험 #3 추가 조합: B1–B3 (correlated)

런타임 > 런타임 유형 변경 > GPU (A100 또는 L4 권장)

- 조합과 판정 규칙은 `paper_repro/PREREG.md`의 **변경 기록 2026-11-08**. **그 항목이 확정·커밋된 뒤에 돌린다.**
- 논문 코드(커밋 `{PAPER_COMMIT}`)와 논문 환경은 기준 조합 노트북(`colab_exp3.ipynb`)과 같다.
- 결과는 기준 조합과 같은 드라이브 폴더 `MyDrive/motivating_exp3/`. 실행 폴더 이름이 조합마다 달라 섞이지 않는다.

| COMBO | 모델 + 방법 | 역할 |
|---|---|---|
| B1 | ViT-B/16 + ETA | 큰 손해 사례 |
| B2 | ViT-B/16 + Tent | 붕괴 대조군 |
| B3 | ViT-L/14 + SAR | 조용한 손해 후보 (느림: SAM이라 forward 2번) |"""),
    *setup_cells(BRANCH_B),
    ("md", """## 조합 고르기

PREREG 순서: B1 → B2 → B3. 각 조합은 seed 1 다음에 새 seed 2, 3, 4 (결과를 본 뒤 바꾸지 않는다)."""),
    ("code", """COMBOS = {'B1': ('eta', 'ViT-B-16'), 'B2': ('tent', 'ViT-B-16'), 'B3': ('sar', 'ViT-L-14')}
COMBO = 'B1'
SEED = 1          # 1 → 2 → 3 → 4
N_DOMAINS = 3     # PREREG: 앞쪽 손상 3개 (gaussian, shot, impulse)
GRAD_CKPT = False # GPU 메모리가 모자라면 True (계산 결과는 같음). ViT-L-14는 16GB 미만이면 True
METHOD, ARCH = COMBOS[COMBO]
RUN_GLOB = f'{OUT}/{METHOD}_{ARCH}_correlated_s*'
ck = '--grad_ckpt' if GRAD_CKPT else ''
print(COMBO, METHOD, ARCH, 'seed', SEED, '| 판정 대상:', RUN_GLOB)"""),
    ("md", """## 스모크 테스트 (1~2분)

클래스 3개(150장)만 돌려서 이 조합의 모델 로드·방법·기록이 되는지 본다. 결과는 `/content/smoke_<COMBO>`에만 남는다."""),
    ("code", """!{VPY} paper_repro/run_paper.py --method {METHOD} --arch {ARCH} --n_domains 1 --limit_classes 3 {ck} --out /content/smoke_{COMBO} 2>&1 | grep -v -i warn | tail -3
!ls /content/smoke_{COMBO}/*"""),
    ("md", """## 실행

손상 하나가 끝날 때마다 상태가 저장된다. 끊기면 이 셀을 다시 실행하면 이어서 돈다."""),
    ("code", """!{VPY} paper_repro/run_paper.py --method {METHOD} --arch {ARCH} --seed {SEED} --n_domains {N_DOMAINS} {ck} --out {OUT} 2>&1 | grep -v -i warn | tee -a {OUT}/run_{COMBO}_s{SEED}.log"""),
    ("md", """## 판정 (GPU 불필요)

- 조합마다 따로 판정한다. 기준값도 같은 조합의 다른 실행에서만 정한다 (다른 조합과 섞지 않음).
- seed 1은 빈도에서 뺀다 (`--exclude_seed 1`). 새 seed 3개가 다 돌기 전에는 "판정 불가"로 나온다."""),
    ("code", """!{VPY} paper_repro/analyze_paper.py "{RUN_GLOB}" --exclude_seed 1 --csv {OUT}/segments_{COMBO} | tee {OUT}/analyze_{COMBO}.txt"""),
    ("md", """## B3 탐색 기준 (통과 판정에 쓰지 않음)

B3만: 손해 구간 기준을 −3%p로 낮춘 "약한 손해 사건"을 따로 센다. 출력 맨 위와 판정 줄에 탐색이라고 표시된다."""),
    ("code", """if COMBO == 'B3':
    !{VPY} paper_repro/analyze_paper.py "{RUN_GLOB}" --exclude_seed 1 --harm_delta -3 | tee {OUT}/analyze_B3_explore.txt
else:
    print('B3에서만 돌린다')"""),
]


def write(cells, name):
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "cells": []}
    for kind, src in cells:
        lines = src.splitlines(keepends=True)
        if kind == "md":
            nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})
    path = os.path.join(ROOT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(path)


write(cells_a, "colab_exp3.ipynb")
write(cells_b, "colab_exp3b.ipynb")
