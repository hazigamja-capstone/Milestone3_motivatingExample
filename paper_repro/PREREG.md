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
