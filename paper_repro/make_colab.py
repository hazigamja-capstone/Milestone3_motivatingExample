"""colab_exp3.ipynb를 만든다 (노트북 JSON을 손으로 고치지 않으려고). 실행: python paper_repro/make_colab.py"""
import json
import os

BRANCH = "hyunjin_expriments#2"
PAPER_COMMIT = "8b2ed63"

cells = [
    ("md", f"""# 실험 #3: A Lost Opportunity 재현 (ViT-L/14 + ETA, correlated)

런타임 > 런타임 유형 변경 > GPU (A100 또는 L4 권장)

- 설정과 판정 기준은 `paper_repro/PREREG.md`. **PREREG가 확정·커밋된 뒤에 돌린다.**
- 논문 코드(커밋 `{PAPER_COMMIT}`)는 수정 없이 쓰고, 논문 환경(torch 2.2.1, open_clip 2.24.0)을 별도 가상환경으로 만든다.
- 결과는 드라이브 `MyDrive/motivating_exp3/`에 바로 저장한다. 손상 하나가 끝날 때마다 상태가 저장되므로,
  런타임이 끊기면 처음부터 다시 실행하면 이어서 돈다 (진행 중이던 손상 하나는 처음부터 다시)."""),
    ("code", f"""# 1. 코드 받기 (저장소 + 논문 코드)
import os
if not os.path.isdir('/content/repo'):
    !git clone -q -b '{BRANCH}' https://github.com/hazigamja-capstone/Milestone3_motivatingExample.git /content/repo
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

path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "colab_exp3.ipynb")
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(path)
