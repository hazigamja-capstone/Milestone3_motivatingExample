# TTA 신뢰 신호 교차 평가

가설: 기존 VLM TTA 신뢰 지표는 각자 하나의 실패 모드만 겨냥한다. 그래서 자기 열에서는 엔트로피를 이기지만 다른 열에서는 지거나 해가 되고, 실패 모드를 모르는 실사용 stream에서는 단일 지표가 어딘가에서 반드시 깨진다.

## 구성

| 단계 | 파일 | 하는 일 |
|---|---|---|
| 1 | `dump.py` | 열(실패 모드)마다 샘플 x 64뷰의 logits, 특징, crop 면적을 저장 (PLPD도 저장하지만 쓰지 않음) |
| 1b | `tune.py` | TPT, R-TPT 프롬프트 튜닝 (dump의 이미지 특징 재사용, 텍스트 쪽만 튜닝) |
| 2 | `matrix.py` | 모든 신호를 같은 역할(top 10% 뷰 선택 후 확률 평균)에 끼워 지표 x 실패 모드 heatmap, 논문 방법 전체 비교 |
| 3 | `online.py` | 공식 TDA에서 positive cache 우선순위 신호를 바꿔 끼우고, 모드별 stream과 mixed stream 비교 |

모델은 CLIP ViT-B/16 (OpenAI), 프롬프트는 `a photo of a {}.` 하나. 라벨 공간은 ImageNet-A의 200 클래스로 통일했다 (mixed stream에서 캐시 하나를 공유하기 위함).

## 열 (실패 모드)

| 열 | 만드는 방법 | 겨냥하는 지표 |
|---|---|---|
| clean | ImageNet-V2 200클래스 | 기준 |
| aug_noise | 같은 이미지, crop 면적 2~15% | MTA, crop 면적 |
| adversarial | 같은 이미지, PGD eps 4/255 7step 후 뷰 생성 | R-TPT |
| corruption | 같은 이미지, gaussian noise sigma 0.18 | 없음 |
| natural_hard | ImageNet-A | ZERO 등 과신 계열 |
| ood | V2의 나머지 800클래스 (online 전용) | energy |

clean, aug_noise, adversarial, corruption은 같은 V2 이미지 집합에서 나와서 열끼리 paired 비교가 된다.

## 신호 (모두 label-free, 높을수록 믿음)

단순 신호: entropy(기준), maxprob, margin, energy, crop_area, agreement. 추가 행으로 random, all_views, original_only, oracle(상한)이 있다.

논문 신호는 공식 구현에서 그대로 뽑는다 (`methods.py`, 기본값은 각 저장소 기본값).
- `mta_inlier`: MTA (MaxZanella/MTA) inlierness. mode와 번갈아 최적화
- `rtpt_rel`: R-TPT (TomSheng21/R-TPT) k=20 이웃 코사인 평균 신뢰도
- `adte`: ADTE (Jinx630/ADTE) 테스트셋으로 추정한 클래스별 q의 Tsallis 엔트로피

## 논문 방법 전체 (`methods.csv`)

신호만 떼지 않고 각 논문의 방법을 공식 규칙대로 돌린 행. 각 논문이 실제로 비교한 상대는 TPT다.

| 행 | 내용 |
|---|---|
| zero_shot | 원본 뷰 하나 |
| entropy_sel | 엔트로피 상위 10% 뷰 확률 평균 (TPT 선택 단계만) |
| tpt | 프롬프트 1 step 튜닝 (marginal 엔트로피) 후 원본 뷰 예측 |
| zero | 엔트로피 상위 30% 뷰 argmax 다수결 |
| mta | MeanShift mode 예측 |
| adte | ADTE 엔트로피 상위 10% 뷰 확률 합 |
| rtpt_ens | 튜닝 없이 R-TPT 신뢰도 가중 앙상블 |
| rtpt | R-TPT 튜닝 (뷰별 엔트로피 평균) 후 신뢰도 가중 앙상블 |

뺀 것과 공식 설정과 다른 점:
- DeYO(PLPD)는 뺐다. CLIP용 방법이 아니고 원래는 스트림에서 모델 LayerNorm을 업데이트하는 방법이라 이 설정에서 원 구현이 성립하지 않는다
- C-TPT 계열은 선택 신호가 아니라 손실 정규화라서 뺐다
- 모든 방법이 프롬프트 하나(`a photo of a {}.`)를 쓴다. TDA, ADTE 공식 결과는 템플릿 앙상블을 쓴다
- 뷰는 모든 방법에 RandomResizedCrop + flip 63장 + 원본. TDA 공식 코드는 ImageNet-A에 AugMix 뷰를 쓴다
- R-TPT 공식 코드는 fp16 유사도가 1 이상인 항목만 빼서 자기 자신이 남을 수 있다. 여기서는 자기 자신을 명시적으로 뺀다
- ZERO 공식 동률 처리(`break_sample_tie`)는 인덱싱 버그가 있어 의도대로(동률 클래스 중 최대 확률) 구현했다
- online에서 crop_area(view 0이 항상 1)와 mta_inlier(샘플 안 합이 1이라 평균이 상수)는 샘플 점수로 못 써서 뺐다

## 실행

```bash
.venv/Scripts/python dump.py --n 500          # 열마다 500샘플, RTX 4050 기준 열당 수십 분
.venv/Scripts/python tune.py                   # TPT, R-TPT 튜닝
.venv/Scripts/python matrix.py
.venv/Scripts/python online.py --seeds 0 1 2
```

빠른 확인은 `--n 50`.

## 결과 파일 (`results/`)

- `selection_delta.png`: 핵심 그림. 대각선만 초록이면 가설 지지
- `auroc_rescuable.png`: 구제 가능 케이스(엔트로피 선택은 틀렸는데 맞힌 뷰가 있음)에서 뷰 AUROC
- `selection_summary.csv`: 신호별 평균, 최저, 최고 열, 엔트로피보다 못한 열 수
- `methods.csv`, `methods_delta.png`: 논문 방법 전체의 정확도와 TPT 대비 차이
- `variants_summary.csv`, `variants_delta.png`: 신호를 hard, soft, 엔트로피 결합, 원본 고정으로 쓸 때 비교
- `online_per_mode.csv`: 모드별 stream에서 게이트별 정확도
- `online_mixed_summary.csv`: mixed stream의 전체, worst-mode 정확도, 캐시 오염률, oracle_router와의 차이
- `online_mixed_contamination.png`: stream 진행에 따른 캐시 오염률

## 읽는 법

- 단일 신호가 mixed stream에서 worst-mode가 엔트로피보다 낮고, oracle_router와 차이가 크면 "통합이 필요하다"는 동기가 성립한다
- oracle_router와 최강 단일 신호의 차이가 작으면 모드별로 신호를 고르는 것 자체의 이득이 작다는 뜻
- random과 비슷한 신호는 선택 효과가 없는 것

## 남은 것

- long-tail / 사전학습 편향 열 (ADTE가 겨냥)
- corruption 종류 확장 (blur는 `data.corrupt`에 있음)
