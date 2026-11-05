# VLM TTA 방법의 실패 모드별 교차 평가: 현재 결론

## 실험 설계

**질문:** 기존 TTA 방법은 각자 겨냥한 실패 모드에서만 이기고, 다른 실패 모드에서는 이득을 잃거나 손해를 보는가.

**모델:** CLIP ViT-B/16, 프롬프트 `a photo of a {}.` 하나, 샘플당 뷰 64개 (원본 1 + 랜덤 crop 63)

**실패 모드 열** (열당 500샘플, ImageNet-A 200클래스 공간)

| 열 | 구성 | 겨냥하는 방법 |
|---|---|---|
| clean | ImageNet-V2 | 기준 |
| aug_noise | crop 면적 2~15% | MTA |
| adversarial | 원본에 PGD (eps 4/255, 7 step) 후 뷰 생성 | R-TPT |
| corruption | 가우시안 노이즈 (sigma 0.18) | 없음 |
| natural_hard | ImageNet-A | ZERO, ADTE |

**방법:** 공식 코드의 규칙과 기본값을 그대로 옮겼다 (TPT, R-TPT, MTA, ZERO, ADTE). CLIP에서 원 구현이 성립하지 않는 DeYO는 제외했다.

**기준선:** 엔트로피 선택 (엔트로피 상위 10% 뷰의 확률 평균, TPT의 선택 단계), zero-shot (원본 뷰), TPT

**통계:** 샘플 단위 paired bootstrap 95% CI. R-TPT, TPT 비교는 결과를 보기 전에 판정 기준을 정했고 (`CRITERIA.md`) Holm 보정을 적용했다. MTA, ZERO, ADTE는 결과를 본 뒤 판정해 탐색적이다.

**구성요소 분해:** MTA와 R-TPT에서 구성요소를 하나씩 빼고 정확도 변화를 본다.

## 결론

### 1. R-TPT는 자기 실패 모드에 특화되어 있다 (사전 기준)

엔트로피 선택 대비 차이 (%p)

| clean | aug_noise | adversarial | corruption | natural_hard |
|---|---|---|---|---|
| -0.6 [-2.0, 0.8] | **-21.6** [-25.6, -17.6] | **+4.6** [2.0, 7.4] | **-3.2** [-5.6, -1.0] | -1.4 [-4.0, 1.2] |

자기 열에서만 유의하게 이기고, 다른 열에서는 이득이 없거나 손해다.

### 2. 이득의 출처는 방법마다 다르다

| 방법 | 이득의 출처 | 근거 |
|---|---|---|
| MTA | inlierness 신호 | 빼면 aug_noise -32.0%p |
| R-TPT | 뷰 앙상블 (신호나 튜닝이 아님) | 앙상블을 빼면 adversarial -43.2%p, 튜닝을 빼면 -1.0%p (유의하지 않음) |

신호만 떼어 공통 선택기에 끼우는 비교로는 방법을 공정하게 평가할 수 없다.

### 3. 적대적 공격 상황에서 정답 뷰를 찾는 label-free 신호는 없다

adversarial 열의 구제 가능 케이스(엔트로피 선택은 틀렸지만 맞힌 뷰가 있는 샘플)에서 모든 신호의 뷰 AUROC가 0.21~0.42로 랜덤(0.5)보다 낮다. 방법들의 이득은 정답 뷰를 찾아서가 아니라 공격받은 원본에서 멀어지거나 평균을 내서 생긴다.

### 조건부 결론 (탐색적)

- **MTA:** 자기 열에서 +7.8%p [4.8, 10.8], 다른 열에서 유의한 손해 없음
- **ZERO, ADTE:** 자기 열(natural_hard)에서 엔트로피 선택 대비 이득이 확인되지 않음 (0.0, +1.0)

## 한계

- **ImageNet-A 재현이 부분적이다.** zero-shot은 재현되지만 (46.7 vs 논문 47.87), ZERO와 MTA의 이득은 논문의 절반이다 (ZERO +5.9 vs +11.7, MTA +5.0 vs +9.5). ImageNet-V2는 모두 재현된다. 원인 후보는 뷰 생성 방식이다. 공식 코드는 원본 전체 이미지에서 crop을 자르고, 이 실험은 224 중앙 crop 안에서 자른다. 이 차이는 모든 열에 같이 적용되어 있어 뷰를 쓰는 방법의 이득이 작게 나왔을 수 있다.
- 실패 모드 열이 5개뿐이고, 열당 500샘플이라 ±2~4%p 차이는 구분이 어렵다.
- online 실험(TDA 캐시)은 스트림이 짧아 캐시가 작동하지 않아 결론에서 뺐다.

## 첨부 이미지

| 파일 | 내용 |
|---|---|
| `results/methods_delta.png` | 방법 x 실패 모드, TPT 대비 정확도 차이 |
| `results/selection_delta.png` | 신호 x 실패 모드, 상위 10% 뷰 선택 시 엔트로피 대비 정확도 차이 |
| `results/auroc_rescuable.png` | 구제 가능 케이스의 뷰 AUROC |
| `results/variants_delta.png` | 신호 사용 방식별 (hard, soft, 엔트로피 결합, 원본 고정) 엔트로피 대비 차이 |

수치 원본: `results/methods.csv`, `results/ablation_ci.csv`, `results/preregistered.csv`, `results/gate.csv`, `results/reproduction.csv`
