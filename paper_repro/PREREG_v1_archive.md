# 실험 #3 사전 등록

> 상태: **확정.** 첫 결과를 보기 전에 커밋했다. 이후 바꾸는 내용은 이 문서를 고치지 않고 아래 '변경 기록'에 날짜와 이유를 덧붙인다.
> 확정일: 2026-11-07 · 확정한 사람: ____

## 1. 목적

A Lost Opportunity(Döbler et al., ECCVW 2024)에서 TTA가 zero-shot보다 크게 나빴던 조합을 재현해 **손해 사례를 확보**한다. 이 단계의 목적은 원인 분석이 아니라 사례 확보다.

## 2. 진행 순서

| 단계 | 할 일 | 목적 |
|---|---|---|
| 0. 설정 확인 | 논문 코드에서 학습률, 초기화, 스트림, 배치, 손상 순서 확인 | 재현 기준 고정 (완료, 3절) |
| 1. 기준점 재현 | ViT-L/14 + ETA를 논문 설정 그대로 실행 | 손해가 나오는 기준점 확보 |
| 2. 사례 확보 | 재현되면 신호를 기록해 붕괴와 조용한 손해로 분류 | 사례 수집 |
| 3. 빈도 측정 | 같은 조합을 새 seed와 손상 순서로 반복 | 실제 빈도 확인 |
| 4. 원인 분리 | 필요할 때만 조건을 하나씩 변경 | 무엇이 손해를 만드는지 확인 |

1단계는 먼저 논문 순서의 앞쪽 손상 3개(gaussian_noise, shot_noise, impulse_noise)만 돌린다. 논문 전체 평균과 비교하려면 15개를 모두 돌린다.

**기준 조합이 ViT-L/14 + ETA인 이유:** 논문 Table 3(correlated 설정)에서 zero-shot 대비 오류율 차이가 SAR는 +2.1%p(58.6 → 60.7)로 작아 재현 여부를 판단하기 어렵다. ETA는 +19.1%p(58.6 → 77.7)라 훨씬 잘 보인다.

## 3. 실행 설정 (논문 코드 커밋 `8b2ed63`)

| 항목 | 값 | 출처 |
|---|---|---|
| 모델 | CLIP ViT-L/14, OpenAI 가중치, fp16으로 로드 | `CLIP.PRECISION` 기본값 |
| 프롬프트 | Radford 80개 템플릿 묶음 (`CLIP.PROMPT_MODE ensemble`) | 논문 본문 |
| 데이터 | ImageNet-C severity 5, 손상당 5만 장 전부 | correlated 설정에서 50k 목록 사용 |
| 손상 순서 | gaussian, shot, impulse, defocus, glass, motion, zoom, snow, frost, fog, brightness, contrast, elastic, pixelate, jpeg | `cfgs/imagenet_c/eta.yaml` |
| 스트림 | 손상 안에서 섞은 뒤 라벨로 정렬 (같은 클래스 50장 연속) | `SETTING correlated` |
| 초기화 | 처음 한 번만. 손상이 바뀌어도 초기화 안 함 | correlated = continual + 정렬 |
| ETA | SGD lr 2.5e-4, momentum 0.9, 배치 64, Fisher 0, D_MARGIN 0.05, E0 = 0.4·ln1000 | `eta.yaml`, `conf.py` |
| seed | 1 (논문 기본값) | `conf.py` |
| 환경 | torch 2.2.1, open_clip 2.24.0 (논문 `environment.yml`) | |

**논문 코드와 다른 점 (A안: 논문 코드 + HF 스트리밍):**

1. 데이터를 디스크 대신 HuggingFace(`haideraltahan/wds_imagenetc`)에서 스트리밍한다. 원본 JPEG를 webp로 다시 압축한 데이터라 픽셀이 조금 다를 수 있다.
2. 같은 클래스 50장 안의 순서는 우리 seed로 섞는다. 논문과 같은 무작위 순서는 아니다.
3. GPU 메모리(6GB) 때문에 gradient checkpointing을 켠다. 계산 결과는 같고 메모리만 줄어든다.
4. 같은 이미지를 고정된 zero-shot 모델에도 넣고, 이미지마다 기록한다. TTA 계산에는 영향이 없다.

## 4. 판정 기준 (실행 전에 고정)

| 항목 | 기준 |
|---|---|
| 구간 | 512장. 손상 경계와 상관없이 실행 전체 스트림을 이어서 자른다. 512장 미만 끝 조각은 버린다 |
| 손해 사건 | 정확도 차이(TTA − zero-shot)가 −5%p 이하인 구간이 3개 이상 연속 |
| 사건 유형 (주 기준) | 사건 구간의 TTA 최빈 비율 ÷ 같은 구간 정답 라벨 최빈 비율 |
| 붕괴 | 4배 이상 |
| 조용한 손해 | 1.5배 이하 |
| 그 사이 | 별도 분류 |
| 민감도 분석 | absolute 원안(최빈 비율 50% 이상 붕괴, 10% 미만 조용한 손해) 결과도 함께 보고한다 |
| 빈도 계산 | 조합을 고른 1차 실행(seed 1)은 빼고, 새 seed만 센다 |
| 재현 통과 | 새 seed 3개 중 2개 이상에서 같은 유형(주 기준)의 사건이 생긴다 |
| 비교 신호 | 엔트로피, 최대 확률, 최빈 클래스 비율, 불일치율, KL, 상위 K 비율(K=5), 구간 번호 |
| 기준값 정하기 | 다른 실행의 비손해 구간(Δ > −5%p, 이득 구간과 노이즈 구간)에서 정한다. 양쪽 꼬리, 구간 번호만 위쪽 꼬리 |
| 오경보율 | 주 기준 5%. 1%와 10%도 함께 보고한다 |
| 표본 크기 규칙 | 보정용 비손해 구간이 100개 미만인 실행이 있으면 주 기준을 10%로 올린다 |
| 오경보 단위 | 구간 단위 오경보율과 함께 오경보 묶음 수(연속으로 켜진 비손해 구간 덩어리)도 센다 |
| 신호 통과 | 주 기준 오경보율에서, 사건 시작 대비 첫 경보가 엔트로피와 구간 번호보다 둘 다 먼저다 |

**사건 유형을 정답 라벨 대비로 나누는 이유**

- absolute 원안: 정렬 스트림에서는 같은 클래스가 50장씩 이어져, 완벽한 모델도 512장 구간 최빈 비율이 약 50/512 ≈ 0.10이다. "조용한 손해 < 10%"가 스트림 구조와 겹친다.
- zero-shot 대비: 사건 유형을 zero-shot으로 나눈 뒤 "zero-shot 신호가 조용한 손해를 더 잘 잡는다"고 하면 결론이 정의에 들어간다(순환).
- 정답 라벨 대비: 라벨은 평가에만 쓰므로 정답 없는 감지라는 전제를 어기지 않고, 스트림 쏠림과 순환 문제가 둘 다 없다.

## 변경 기록

(확정 이후 바뀐 점을 날짜와 이유와 함께 적는다.)

### 2026-11-08 · 추가 조합 B1–B3 등록 (초안)

> 상태: **초안.** 아래 조합을 하나라도 돌리기 전에 커밋해야 확정된다.
> 작성자: ____ · 이 항목을 쓸 때 기준 조합(ViT-L/14 + ETA) 결과를 봤는가: ☐ 안 봄 ☐ 봄 (본 범위: ____)

**무엇을 바꾸나:** 기준 조합은 그대로 두고, 손해 사례를 더 확보하기 위한 조합 3개를 추가한다. 1–4절의 본문(목적, 설정, 판정 기준)은 바꾸지 않는다.

**이유:**

1. 기준 조합 하나만으로는 사례 수가 부족할 수 있다. 실험 #2의 조용한 손해는 재현되지 않았다(`run_lr1e-3.log`: defocus, ETA·SAR, seed 0–2 모두 TTA가 zero-shot보다 +1.2~+4.2%p 좋음).
2. 논문 Table 3(correlated)에는 더 크거나 성격이 다른 손해가 있다. ViT-B/16은 ViT-L/14보다 훨씬 빠르게 돈다.
3. 붕괴 사례(대조군)와 조용한 손해 후보를 따로 확보해야 사건 유형별로 신호를 비교할 수 있다.

**추가 조합** (모두 ImageNet-C severity 5, `SETTING correlated`, 논문 코드 cfg 기본값 그대로; 학습률 등 실제 값은 실행마다 `meta.json`에 기록됨)

| ID | 모델 + 방법 | 논문 Table 3 ImageNet-C 오류율 (zero-shot → TTA) | 역할 |
|---|---|---|---|
| A (기존) | ViT-L/14 + ETA | 58.6 → 77.7 | 기준점 (1–4절 그대로) |
| B1 | ViT-B/16 + ETA | 73.8 → 88.4 | 큰 손해 사례. A와 같은 방법이라 모델 크기만 다름 |
| B2 | ViT-B/16 + Tent | 73.8 → 93.9 | 붕괴 대조군. 붕괴로 분류되리라 예상 |
| B3 | ViT-L/14 + SAR | 58.6 → 60.7 | 조용한 손해 후보. SAR는 붕괴 시 모델을 복구한다 |

논문 수치는 손상 15개 평균이다. 앞쪽 손상 3개에서도 같은 크기의 손해가 나온다는 보장은 없다.

**진행 순서와 seed** (2절과 같은 규칙)

- 각 조합은 먼저 논문 seed 1로, 앞쪽 손상 3개(gaussian_noise, shot_noise, impulse_noise)만 돌린다.
- 그다음 새 seed **2, 3, 4**로 반복한다. seed는 지금 고정하고 결과를 본 뒤 바꾸지 않는다.
- 돌리는 순서: B1 → B2 → B3. B3는 SAM 때문에 forward가 두 번이라 가장 느리다.
- 계산 자원이 모자라 일부만 돌리면, 돌리지 못한 조합과 seed를 그대로 보고한다.

**판정** (4절 기준을 바꾸지 않고 조합마다 따로 적용)

| 항목 | 기준 |
|---|---|
| 손해 사건, 사건 유형, 경보, 오경보율 | 4절 그대로 |
| 재현 통과 | 조합마다 따로. seed 1을 빼고 새 seed 3개 중 2개 이상에서 같은 유형(주 기준)의 사건이 생긴다 |
| 교차 기준 경보의 기준값 | 같은 조합의 다른 실행에서만 정한다. 조합을 섞지 않는다 |
| 조합 간 합산 | 하지 않는다. 다른 조합의 사건을 더해 2/3을 채우지 않는다 |
| 여러 조합 보고 | 통과 여부와 상관없이 A와 B1–B3를 모두 보고한다. 통과한 조합만 골라 쓰지 않는다 |
| B2 기대 결과 | 붕괴(주 기준). 붕괴가 아니면 그 자체로 보고한다 |
| B3 사건이 없을 때 | "사건 없음"으로 보고한다. −5%p 기준을 결과를 본 뒤 낮추지 않는다 |
| B3 탐색 기준 (통과 판정에 쓰지 않음) | 정확도 차이가 −3%p 이하인 구간이 3개 이상 연속이면 "약한 손해 사건"으로 따로 센다. 이때 보정용 비손해 구간도 같은 기준(Δ > −3%p)을 따른다. 탐색 결과로만 보고한다 |

탐색 기준은 `analyze_paper.py --harm_delta -3`으로 돌린다. 기본값(−5)이 아닌 값을 넣으면 출력 맨 위와 재현 판정 줄에 "탐색"이라고 표시되고, 통과/실패 대신 "2개 이상/미만"만 나온다.

**이번 추가에서 뺀 것** (지금 `run_paper.py`로는 돌릴 수 없어 코드를 고쳐야 함)

| 후보 | 근거 | 빼는 이유 |
|---|---|---|
| ViT-B/16 + DeYO | 논문 Table 3: 73.8 → 99.7 | `run_paper.py`의 `CFG_OF`에 deyo가 없음 |
| ViT-L/14 + ETA, ImageNet-D109 | 22.4 → 40.5 | 데이터 로더가 HF ImageNet-C만 지원. DomainNet-126을 따로 받아야 함 |
| `gradual_correlated`, `mixed_domains_correlated` | ROID (Marsden et al., WACV 2024) | `iter_split`이 손상 하나씩 라벨 정렬로만 내보냄 |
| CCC | RDumb (Press et al., NeurIPS 2023): CIN-C에서는 안정적이던 ETA가 CCC에서 붕괴 | 별도 webdataset이고 correlated와 함께 쓸 수 없음 |

주의: `run_paper.py --setting`에 `correlated`가 아닌 값을 넘기면 논문 cfg만 바뀌고 데이터는 여전히 라벨 정렬 스트림으로 들어와, 실행 이름과 실제 스트림이 어긋났다. 그래서 `--setting`은 이제 `correlated`만 받는다. 위 후보를 하려면 로더를 먼저 고치고 새 변경 기록을 남긴다.

**실행 명령**

```
# B1 (메모리가 모자라면 --grad_ckpt를 붙인다. 계산 결과는 같다)
third_party/.venv_paper/Scripts/python.exe paper_repro/run_paper.py --method eta  --arch ViT-B-16 --n_domains 3 --seed 1
# B2
third_party/.venv_paper/Scripts/python.exe paper_repro/run_paper.py --method tent --arch ViT-B-16 --n_domains 3 --seed 1
# B3
third_party/.venv_paper/Scripts/python.exe paper_repro/run_paper.py --method sar  --arch ViT-L-14 --n_domains 3 --seed 1 --grad_ckpt
# 새 seed: 같은 명령에서 --seed 2, 3, 4
```

**판정 명령** (조합마다 따로)

```
python paper_repro/analyze_paper.py "outputs_paper/eta_ViT-B-16_correlated_s*"  --exclude_seed 1
python paper_repro/analyze_paper.py "outputs_paper/tent_ViT-B-16_correlated_s*" --exclude_seed 1
python paper_repro/analyze_paper.py "outputs_paper/sar_ViT-L-14_correlated_s*"  --exclude_seed 1
# B3 탐색 기준 (통과 판정에 쓰지 않음)
python paper_repro/analyze_paper.py "outputs_paper/sar_ViT-L-14_correlated_s*"  --exclude_seed 1 --harm_delta -3
```

**Colab:** `colab_exp3b.ipynb` (`paper_repro/make_colab.py`로 생성, 브랜치 `seoy_experiments3`를 받음). 노트북 안에서 `COMBO`와 `SEED`만 바꿔 돌린다.

## 5. 실행 명령

```
third_party/.venv_paper/Scripts/python.exe paper_repro/run_paper.py --method eta --n_domains 3 --grad_ckpt
```

손상 하나가 끝날 때마다 `outputs_paper/<실행>/state.pt`에 모델 상태가 저장된다. 끊기면 같은 명령으로 이어서 돌린다.

**Colab:** `colab_exp3.ipynb` (`paper_repro/make_colab.py`로 생성). 논문 환경을 `uv`로 따로 만든다
(`requirements_paper.txt` + `requirements_paper_nodeps.txt`; robustbench가 timm≥1.0.9를 요구해 `--no-deps`로 설치).
결과는 드라이브 `MyDrive/motivating_exp3/`. GPU 메모리가 16GB 이상이면 gradient checkpointing을 끈다(결과는 같음).

**판정:**

```
python paper_repro/analyze_paper.py "outputs_paper/eta_ViT-L-14_correlated_s*" --exclude_seed 1
```

4절 기준을 그대로 구현했다(`paper_repro/analyze_paper.py`). 주 기준(정답 라벨 대비)과 민감도(absolute)를 함께 출력하고, 교차 기준 경보는 오경보율 1%, 5%, 10%를 모두 출력하며 표본 크기 규칙에 따라 주 기준을 표시한다.

### 2026-11-10 · 구현 오류 수정 및 분석 집계 명시

기존 1–4절과 B1–B3의 과거 확정/초안 상태를 변경하지 않는다. 실험 #3 결과는 이 검토에 제공되지 않았으며, 과거 결과 열람 여부는 미확인이다. 아래는 코드 검토와 합성 데이터에서 확인한 오류의 수정 기록이다.

- 재개: scaler, SAR 내부 SGD momentum, 난수 상태를 보존하는 checkpoint v2 도입. 기존 checkpoint는 필요한 정보가 없어 자동 재개하지 않는다.
- 결과 혼합 방지: 설정/코드가 다르면 기존 결과를 재사용하지 않는다. 같은 도메인 prefix의 확장만 허용한다.
- 재현: 동일 경로 중복 제거, 중복 seed/상이한 조합/미완료/스모크 결과 차단. 확인 seed 기본값은 B 조합에 명시된 2·3·4이며 A도 같은 기본값을 사용한다. 다른 사전 지정 seed는 `--confirm_seeds`로 명시한다.
- 경보 집계 수정: 미감지를 버리던 중앙값을 폐기하고 +∞를 포함한 상위 중앙값과 감지 수/기준 수를 출력한다. 신호 통과는 서로 다른 기준의 중앙값을 비교하지 않고 **동일 기준 실행별**로 두 baseline과 짝비교한다. 사건×기준 비율이며 독립 사건 비율이 아니다. 보정 불가능한 비교는 통과 분모에서 제외한다.
- 기존 임계값은 라벨로 비손해 구간을 고르는 사후 진단임을 명시한다. 온라인 무라벨 보정이나 개입 효용을 입증했다고 해석하지 않는다.

이 집계 수정 전후의 숫자는 분석 버전을 명시해 구분한다. 사건 기준과 오경보 예산은 소급 변경하지 않았다. 새 감지 설계는 `DESIGN_REVIEW.md`의 **제안**이며 확인 결과를 보기 전 별도로 고정해야 한다.
