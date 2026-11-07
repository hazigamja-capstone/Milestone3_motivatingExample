# 프로젝트 안내

CLIP 온라인 TTA가 해로워지는 순간을 zero-shot 비교로 정답 없이 찾는 연구 (Milestone 3, motivating example).

## 세션 시작 시

1. `docs/research_context.md`를 읽는다. 연구 질문, 선행 연구, 해석 기준, 현재 상태, 다음 단계가 있다.
2. `docs/session_log.md`의 최근 항목을 읽는다.

## 세션 종료 전

- `docs/session_log.md` 맨 위에 오늘 날짜로 항목을 추가한다(한 일 / 알게 된 것 / 다음).
- 연구 질문, 현재 상태, 다음 단계가 바뀌었으면 `docs/research_context.md`도 고친다.
- 새 실험을 마치면 `docs/expN_<주제>.md`로 보고서를 남기고, 그림은 `docs/figures/`에 복사한다(`docs/`, `outputs/`, `results/`, `runs/`는 모두 git 미추적이라 이 컴퓨터에만 있다).

## 코드

- `run.py`: TTA 모델과 zero-shot 모델에 같은 스트림을 넣고 샘플 단위로 기록 (`outputs/*.npz`).
- `methods.py`: Tent, ETA, SAR (LayerNorm affine만 갱신).
- `analyze.py`: 512장 구간 단위 정확도 차이와 신호별 분리력.
- 실행은 Colab GPU (`colab_run.ipynb`). 로컬 파이썬은 `.venv/Scripts/python.exe`.

## 문서

- `docs/exp2_silent_harm.md`: 실험 #2 보고서와 후속 연구 설계(8절).
