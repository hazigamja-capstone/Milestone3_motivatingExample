# entropy_tta_probe — VLM TTA에서 엔트로피 목적함수의 한계 검증 코드

CLIP 기반 테스트타임 적응(TPT 계열)에서 **엔트로피 최소화 / 엔트로피 기반 선택**이 가진 다섯 가지 한계를
각각 독립된 실험으로 수치와 그림으로 보여주는 코드입니다.

## 한계 ↔ 실험 대응

| # | 한계 | 스크립트 | "한계가 드러났다"고 볼 신호 |
|---|------|----------|---------------------------|
| 1a | 낮은 엔트로피 ≠ 정답, shift가 클수록 악화 | `e1_entropy_vs_correctness.py` | AUROC(−H→정답)이 ImageNet-C severity↑에 따라 하락, 엔트로피 하위 10% 안에서도 오류율이 높고 severity↑에 따라 상승 |
| 1b | 온라인 적응 시 확증 편향·오류 누적·붕괴 | `e4_continual_collapse.py` | 윈도우별 정확도가 zero-shot 아래로 떨어짐, 예측 클래스 수 감소·최빈 클래스 비중 증가, 최종 파라미터 재평가 정확도 < zero-shot. `--objective ent_div`로 다양성 항을 넣으면 붕괴가 완화되면 원인이 엔트로피 항임을 확인 |
| 2 | 과신 / 보정 악화 | `run_episodic.py` → `e2_overconfidence.py` | TPT 스텝↑에 따라 정확도는 거의 그대로인데 mean conf·ECE·NLL 상승, W→W 샘플의 confidence 상승 비율 높음, conf>0.9 오답 비율 증가 |
| 3 | 신뢰도 기반 뷰 선택의 허위 단서 위험 / 이득의 출처 | `run_episodic.py` → `e3_view_selection.py` | 선택된 뷰의 crop 면적이 작음, 선택 뷰 전원이 같은 오답에 동의하는 비율 존재, 원본은 맞는데 선택 앙상블이 틀리는 사례 존재, TPT와 ZERO(학습 없는 다수결)의 예측 일치율이 매우 높고 정확도 차이 작음 |
| 4 | 엔트로피 절대 크기가 온도·클래스 수·템플릿에 좌우 | `e5_scale_sensitivity.py` | 같은 고정 임계값(0.4·ln1000)의 선택률이 데이터셋·logit scale마다 극단적으로 다름, K에 따라 mean H가 log K처럼 변함, 템플릿을 바꾸면 "확신하는 10%" 집합의 Jaccard가 낮음 |
| 5 | 클래스 간 의미 구조·임베딩 기하 무시 | `e6_semantic_blindness.py` | near/far 오류 구분 AUROC(H)≈0.5 vs Rao Q는 더 높음, 같은 엔트로피 구간 안에서 Q의 IQR이 큼, top-1/top-2 cosine 간격이 ~0.01 수준인데 logit scale로 증폭되어 엔트로피를 결정 |

## 벤치마크 선택

- **기본값 = SwapPrompt 구성 그대로 (14개)**: ImageNet, ImageNet-V2/A/R/Sketch + Caltech101, DTD, Flowers102, OxfordPets, UCF101, StanfordCars, Food101, EuroSAT, SUN397. `--datasets swapprompt`
- 이 14개는 사실상 **TPT 벤치마크(OOD 5개 + cross-dataset 10개)에서 FGVC-Aircraft만 빠진 것**이라, 통용 프로토콜과 비교하려면 `fgvc_aircraft`를 추가하는 것을 권장합니다 (`--datasets swapprompt fgvc_aircraft`, 또는 cross만 `tpt_cross`).
- **ImageNet-C 추가를 권장**: SwapPrompt 구성에는 shift 강도를 조절할 수 있는 데이터가 없습니다. 한계 1a(shift↑ → 엔트로피 신뢰도↓)와 1b(연속 도메인 변화에서의 붕괴, CoTTA 프로토콜)는 ImageNet-C로 보여주는 것이 가장 깔끔합니다. `imagenet_c:all`(15종×5단계), `imagenet_c:sev=5`, `imagenet_c:corr=gaussian_noise` 단축명 지원.
- 백본: SwapPrompt와 같은 `RN50`이 기본, TPT와 맞추려면 `--arch ViT-B-16`.

## 데이터 경로

CoOp `DATASETS.md` 규약을 그대로 따릅니다 (TPT/SwapPrompt용으로 이미 준비된 폴더면 그대로 동작).
```
DATA/
  imagenet/classnames.txt               # CoOp 파일: "<wnid> <name>" 1000줄 (A/R/Sketch/V2/C 모두 이 파일로 클래스명 매핑)
  imagenet/images/val/<wnid>/*
  imagenetv2/imagenetv2-matched-frequency-format-val/<0..999>/*
  imagenet-adversarial/imagenet-a/<wnid>/*
  imagenet-rendition/imagenet-r/<wnid>/*
  imagenet-sketch/images/<wnid>/*
  imagenet-c/<corruption>/<severity>/<wnid>/*          # 선택
  caltech-101/{101_ObjectCategories, split_zhou_Caltech101.json}
  dtd/{images, split_zhou_DescribableTextures.json}
  oxford_flowers/{jpg, split_zhou_OxfordFlowers.json}
  oxford_pets/{images, split_zhou_OxfordPets.json}
  ucf101/{UCF-101-midframes, split_zhou_UCF101.json}
  stanford_cars/{..., split_zhou_StanfordCars.json}
  food-101/{images, split_zhou_Food101.json}
  eurosat/{2750, split_zhou_EuroSAT.json}
  sun397/{SUN397, split_zhou_SUN397.json}
  fgvc_aircraft/data/{images, variants.txt, images_variant_test.txt}   # 선택
```
ImageNet-A/R은 폴더에 있는 200개 wnid만 라벨 공간으로 씁니다(TPT와 동일).

## 실행

```bash
pip install -r requirements.txt
bash run_all.sh /path/to/DATA RN50        # 전체 (Linux / macOS)
```
Windows PowerShell: `powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Data "D:\DATA" -Arch RN50`

**Colab**: `colab_run.ipynb`를 열고 위에서부터 실행합니다. 데이터셋 다운로드(공개 링크 + CoOp split json),
ImageNet-V2-C 생성(ImageNet-C 공식 corruption 수식을 V2 이미지에 적용, 공식 ImageNet-C 아님), 단계별 실행, 결과 표시까지 포함합니다.
Colab에서 GitHub의 노트북을 열면 저장소 코드는 함께 오지 않으므로, 저장소 폴더를 Drive의 `MyDrive/entropy_tta_probe`에 두거나 노트북 0번 셀의 `REPO_DIR`을 지정하세요.
개별 실행 예:
```bash
python experiments/cache_features.py --data_root DATA --datasets swapprompt imagenet_c:all
python experiments/e1_entropy_vs_correctness.py --data_root DATA --datasets swapprompt imagenet_c:all
python experiments/run_episodic.py --data_root DATA --datasets swapprompt --n_samples 1000 --steps 4
python experiments/e2_overconfidence.py --data_root DATA
python experiments/e3_view_selection.py --data_root DATA
python experiments/e4_continual_collapse.py --data_root DATA --datasets imagenet_c:sev=5 --order iid
python experiments/e4_continual_collapse.py --data_root DATA --datasets imagenet_c:sev=5 --order dirichlet --alpha 0.1
python experiments/e4_continual_collapse.py --data_root DATA --datasets imagenet_c:sev=5 --objective ent_div   # 대조군
python experiments/e4_continual_collapse.py --data_root DATA --datasets imagenet_c:sev=5 --params norm --lr 1e-3  # TENT(시각 인코더 norm)
python experiments/e5_scale_sensitivity.py --data_root DATA --datasets swapprompt
python experiments/e6_semantic_blindness.py --data_root DATA --datasets swapprompt
```
결과는 `results/<실험>/<arch>_<pretrained>/` 아래 CSV와 PNG로 저장됩니다.

## 구현 메모

- **TPT 재현 설정**: 64뷰(원본 + RandomResizedCrop, ImageNet 계열 외에는 AugMix), 엔트로피 하위 10% 뷰 선택, 선택 뷰의 marginal entropy 최소화, AdamW lr 5e-3, ctx 초기값 "a photo of a", 샘플마다 ctx·옵티마이저 리셋, 업데이트 후 원본 뷰로 예측. `--steps 4`면 1~4스텝 후 예측을 모두 저장(1스텝 = 표준 TPT).
- 이미지 인코더가 고정이므로 뷰 특징은 한 번만 계산하고 텍스트 쪽 ctx만 미분합니다. 프롬프트 러너는 초기 상태에서 `encode_text` 출력과 1e-7 이내로 일치하는 것을 확인했습니다.
- E1/E4(prompt)/E5/E6는 center-crop 특징 캐시만 쓰므로 빠릅니다. 비싼 건 `run_episodic.py`(샘플당 64뷰 인코딩 + 텍스트 인코더 역전파)라 기본적으로 데이터셋당 클래스 층화 1000개 샘플을 씁니다(`--n_samples 0`이면 전체).
- ZERO는 선택 뷰들의 argmax 다수결(동률은 soft 점수로 결정)로 구현했습니다.
- Rao Q는 텍스트 임베딩 cosine 거리 기반이며 샘플별 top-20 클래스에서 계산합니다.

## 주의

이 코드는 가짜 이미지 + 랜덤 가중치로 모든 스크립트가 끝까지 도는지만 확인했습니다. 실제 CLIP 가중치·데이터로는
아직 돌리지 않았으므로, 위 표의 "신호"는 기대되는 방향이지 측정된 결과가 아닙니다.
