# v3 실행 및 검증 안내

활성 설계는 `PREREG.md`. 실험 #3 본실험 미실행 확인에 따라 설계를 실행 전에 개정했다. `analyze_paper.py`는 과거 paper 기준의 사후 진단용이며 v3 calibration/confirmation 입력을 거부한다. **주 실험은 `run_plan.py`와 `evaluate_detector.py`를 사용한다.**

## 실행 순서

Python3.11 / 논문 의존성 / CUDA 환경에서 저장소 루트를 작업 디렉터리로 사용한다. Colab은 `colab_exp3.ipynb`(A), `colab_exp3b.ipynb`(B1/B2/B3)를 사용한다. 두 파일은 `make_colab.py`로 생성하며 실행 소스와 의존성 lock이 내장되어 있다. 원격 push 없이 노트북 파일 하나를 업로드할 수 있다. Colab 상세 안내는 `COLAB.md`를 따른다.

```sh
# 예: B1. 기본적으로 HF API에서 현재 데이터 SHA를 최초 한 번 조회해 계획에 고정한다.
python paper_repro/run_plan.py --combo B1 --phase paper --out outputs_v3
python paper_repro/run_plan.py --combo B1 --phase calibration --out outputs_v3
python paper_repro/run_plan.py --combo B1 --phase freeze --out outputs_v3
python paper_repro/run_plan.py --combo B1 --phase confirmation --out outputs_v3
python paper_repro/run_plan.py --combo B1 --phase evaluate --out outputs_v3
```

재개 시 저장된 SHA와 코드/정밀도 설정을 유지한다. `--phase all`은 위 순서 전체를 실행한다. 조합 순서 B1 → B2 → A → B3. `--grad_ckpt`는 첫 실행 전 결정하고 모든 단계에 동일하게 지정한다. 명시적 SHA를 쓸 때에는 첫 명령에 `--data_revision <40자리SHA>`를 추가한다. 이후 계획 파일을 재사용하며 SHA를 바꾸지 않는다.

네 조합 모두 같은 `--out outputs_v3`를 사용한다. 첫 조합에서 `dataset_revision.json`을 만들고, 이후 조합도 그 SHA를 자동 재사용한다. 다른 SHA를 지정하면 실행 전에 거부한다. 조합별 결과 폴더와 계획/임계값/평가 파일은 서로 다른 이름으로 저장된다.

- `paper`: 전체1000클래스, seed1, 손상당50,000장. 논문순서의 참고 재현.
- `calibration`: 고정400클래스, seed101/102, 손상당20,000장. 각 실행의 첫100구간 전체로 무라벨 보정.
- `freeze`: `<COMBO>_calibration.json`을 새로 작성. 임계값 파일을 덮어쓰지 않음.
- `confirmation`: 고정600클래스, seed201–210, 손상당30,000장. 동결 없이는 계획 실행기가 시작하지 않음.
- `evaluate`: 확인10개가 모두 완료되어야 `<COMBO>_evaluation.json` 생성. 기존 평가 파일을 덮어쓰지 않음.

조합당1,170,000장(두 모델 계산 포함). 주 실험의 domain 수는3으로 고정하며 수/순서/역할/코드가 달라지면 새 출력 루트를 쓴다. v3에서는 3→15 domain 확장도 같은 실행으로 이어 붙이지 않는다.

## GPU 스모크와 재개 검증

본실험과 다른 출력 루트에서 `run_paper.py --role paper --limit_classes 3 --n_domains 1 --data_revision <SHA>`로 스모크한다. calibration/confirmation 역할도 클래스 분할·기록이 작동하는지 별도 스모크한다. 스모크 결과는 주 분석에 들어갈 수 없다.

GPU 재개 동등성은 같은 고정3-domain 계획을 사용해 한 번은 무중단 실행하고 다른 번은 기본100배치의 부분 checkpoint 이후 중단→재개하여 비교한다. domain 수를 바꾸는 방식으로 중단을 흉내 내지 않는다. 출력 배열·누적 정확도·업데이트 수·scaler/optimizer 상태를 비교하고 수치 차이를 기록한다. CPU 테스트만으로 CUDA AMP/checkpointing 동등성을 주장하지 않는다.

기존 checkpoint/결과를 자동 이관하지 않는다. 상태 복원에 필요한 값 또는 새 역할/샘플ID가 없으므로 보존한 채 새 출력 루트에서 시작한다. v3 checkpoint는 갱신 파라미터·버퍼·optimizer·SAR내부SGD·scaler·ETA/SAR상태·Python/NumPy/torch/CUDA난수와 부분 결과 위치/해시를 보존한다. 전체 frozen CLIP은 초기 가중치로 재구성하므로 Drive에 seed별 중복 저장하지 않는다.

각 확인 seed를 세션 하나에 맞추려면 `--phase confirmation --seed 201`처럼 지정한다. 나머지 seed도201–210을 모두 완료해야 한다. 이미 완료된 seed는 metadata/출력 해시를 확인하고 모델 생성 전에 건너뛴다.

## 회귀 검증

저장소 루트에서 torch·numpy가 있는 Python으로:

```sh
python -m unittest discover -s tests -v
```

고정 upstream SAM 추가 검증:

```sh
git clone https://github.com/mariodoebler/test-time-adaptation.git /tmp/seoy-upstream
git -C /tmp/seoy-upstream checkout 8b2ed63
PAPER_SAR_SOURCE=/tmp/seoy-upstream/classification/methods/sar.py python -m unittest discover -s tests -v
```

SAM 클래스만 AST로 로드해 외부 모델 의존성 없이 검증한다. 경로를 주지 않으면 이 추가 통합 테스트만 skip이다. 고정 Python3.11.17/torch2.2.1+cu121 환경의 설치·import·WebP·torchvision NMS·방법 등록·의존성 검사를 CPU에서 수행했고, 같은 환경의 설치 재실행도 검증했다. CUDA runtime wheel이 설치되어도 GPU가 없으면 GPU연산 검증은 아니다. 별도로 Python3.12/CPU torch2.15.0에서도 회귀 검증을 수행한다.

검증 범위: 역할별 클래스 분리·블록/손상 순서, 결과 중복/완결성, 보정의 라벨 비의존성과 prefix 제한, 보정 부족 실패, 경보의 실제 이용 시각/lookback, 미감지/오경보, 정책 전환의 인과성, 동결파일 보호, 보정2개→확인10개의 합성 결과 전체 CLI, checkpoint 재개.

## 현재 실행 한계

본실험은 실행하지 않았다. 현재 머신에는 CUDA GPU가 없고 HF API 요청은 프록시 터널403으로 실패했다. 환경 네트워크 설정에서 `huggingface.co` 접근과 실제 다운로드 리다이렉트 목적지를 허용한 뒤 재검증하거나 Colab에서 수행해야 한다. 모델 가중치 목적지 접근도 실제 스모크로 검증해야 한다. 데이터 SHA를 추측하거나 TLS 검증을 비활성화하지 않는다.

### 2026-11-10 Colab 배포물 검증 결과

- Python3.11.17 / torch2.2.1+cu121 / torchvision0.17.1+cu121 / open_clip2.24.0의 해시 잠금 설치와 동일 환경 재실행 통과.
- 내부 robustbench, 고정 AutoAttack, EATA/Tent/SAR 등록, WebP 디코딩, torchvision CPU NMS, 의존성 검사 통과.
- 노트북에 넣을 소스를 별도 디렉터리에 복원한 뒤 해당 고정 환경에서 **회귀 테스트43개 모두 통과**. 실제 runner의 중간 재개·출력 중복 방지, 셀 중단 시 과학 자식 프로세스 종료도 포함.
- CUDA GPU는 없으므로 GPU 행렬연산·실제 CLIP GPU 스모크·HF 데이터 다운로드·본실험은 미실행. Colab에서 스모크로 이어서 검증한다.
