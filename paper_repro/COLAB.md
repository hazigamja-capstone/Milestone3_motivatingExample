# Colab 실행 안내

## 자동 실행 노트북

`colab_exp3_v3_auto.ipynb`는 PHASE/SEED 수동 변경 없이 B1→B2→A→B3의 스모크·기준·보정·동결·확인·평가를 순서대로 실행한다. 기존 `STUDY`를 유지한다. 기존 계획이 있으면 스모크를 건너뛰고, 완료 seed는 원래 실행기가 검증 후 재사용한다. 오류 뒤 작업은 실행하지 않는다. 화면에 로그를 직접 표시한다.

기존 수동 노트북의 과학 소스 payload를 그대로 유지하므로 소스 동결과 이미 생성한 계획을 변경하지 않는다. `python paper_repro/make_auto_colab.py`로 수동 노트북을 감싸 생성한다. 이 변경을 적용하기 위해 기존 수동 노트북의 payload를 재생성하거나 STUDY를 변경할 필요가 없다.

## 시작

1. **`colab_exp3_v3_ready.ipynb`를 Google Colab에 업로드**한다. A만 기본 선택된 파일은 `colab_exp3.ipynb`다. `colab_exp3b.ipynb`도 같은 내용이며 모든 파일에서 COMBO를 바꿀 수 있다.
2. Colab의 **런타임 → 런타임 유형 변경 → GPU**를 선택한다. ViT-B는 T4부터 스모크하고, ViT-L은 L4/A100을 권장한다. 작은 GPU에서 메모리 동작은 스모크로 확인한다.
3. 설정 셀에서 처음에는 `COMBO='B1'`, `PHASE='smoke'`, `SEED=None`, `GRAD_CKPT='auto'`를 유지한다.
4. 위에서부터 실행하고 Drive 연결을 승인한다. 선택적으로 Colab 보안 비밀의 `HF_TOKEN`을 사용할 수 있다. 공개 데이터는 토큰 없이도 시도하며 토큰을 코드/로그/Drive에 저장하지 않는다.
5. 환경 검사와 스모크가 성공하면 아래 단계로 진행한다. 환경 설치만으로 모델/데이터 동작을 검증한 것은 아니다.

**수정 코드는 노트북에 내장되어 있다.** 별도 ZIP 업로드나 원격 브랜치 push가 필요 없다. 현재 작업 트리에서 `python paper_repro/make_colab.py`를 실행하면 노트북 두 개가 같은 소스로 재생성된다.

## 실행 단계

설정 셀과 마지막 실행 셀을 다음 조합으로 재실행한다. 세션이 바뀌면 설치/복원 셀도 위에서부터 실행한다.

| PHASE | SEED | 실행 |
|---|---|---|
| smoke | None | 매번 새 로컬 경로에서150장 검증, 본실험에 포함하지 않음 |
| paper | 1 또는 None | 논문 순서 기준 실행 |
| calibration | 101 | 첫 무라벨 보정 실행 |
| calibration | 102 | 둘째 무라벨 보정 실행 |
| freeze | None | 기준값 동결. 두 보정 실행 완료 필요 |
| confirmation | 201,202,…,210을 하나씩 | 고정 확인 실행10개 |
| evaluate | None | 확인10개가 모두 완료되면 JSON 평가 |
| status | None | 완료 표시와 기록된 domain 수 확인 |

`SEED=None`이면 해당 계산 단계의 모든 등록 seed를 순서대로 실행한다. 조합 순서는 **B1 → B2 → A → B3**. 네 조합에서 `STUDY`는 동일하게 유지한다. 모델/조합/seed를 바꾸기 위해 배치 크기나 논문 설정을 임의 변경하지 않는다.

기본 `GRAD_CKPT='auto'`는 이미 저장한 계획이 있으면 원래 선택을 유지한다. 첫 계획 전에는 ViT-L에서 GPU메모리가24GiB 미만이면 켠다. 이는 메모리 성공 보장이 아니다. 본실험 전 OOM이면 더 큰 GPU 또는 checkpointing으로 스모크를 다시 검증한다. 계획 동결 이후 설정 변경은 새 STUDY로 구분한다.

## 설치와 사전 검사

- Colab kernel의 Python/torch를 덮어쓰지 않고 `/content/venv_paper`에 **Python3.11 + torch2.2.1/cu121 + open_clip2.24.0**을 설치한다.
- `requirements_colab.lock`의 해시를 검증한다. 별도 AutoAttack도 전체 커밋으로 고정한다. upstream 내부 robustbench를 사용하며 충돌하는 외부 robustbench를 추가하지 않는다.
- Python3.11은 이 저장소가 선택한 실행 환경이다. upstream Conda의 Python3.12.2와 byte-for-byte 동일한 환경이라는 뜻은 아니다.
- 설치 완료 후 핵심 버전, 고정 upstream 커밋, 내부 robustbench 경로, EATA/Tent/SAR 등록, Pillow WebP, torchvision NMS, CUDA행렬연산을 검사한다. 설치/검사 실패는 셀을 실패시킨다.
- freeze/evaluate/status는 GPU 없이도 실행 가능하다. 새 런타임에는 동일 가상환경 의존성 설치가 필요하다.
- 드라이버는 Colab이 제공하며 로컬 CUDA toolkit 설치가 필요하지 않다. CPU만 선택되어 있으면 계산 단계 전에 중단한다.

## 저장 위치와 재개

Drive: `MyDrive/<STUDY>/`

- `source_bundle.zip`: 노트북에 들어 있던 정확한 소스와 의존성 lock
- `experiment_notebook.ipynb`: 지원될 때 자동 저장한 원본 노트북. 자동 저장 안내가 실패하면 내려받은 .ipynb를 직접 보관
- `environment/`: 실제 설치 버전과 환경 검사 결과(인증정보 없음)
- `dataset_revision.json`, `<COMBO>_plan.json`: 데이터 SHA와 동결 설정
- `<method>_<arch>_<role>_s<seed>/`: 샘플별 NPZ, 작은 `state.pt`, metadata, 완료 marker
- `logs/`: stdout/stderr와 실제 종료 상태
- `<COMBO>_calibration.json`, `<COMBO>_evaluation.json`: 동결 기준값과 평가

실행 코드는 `/content/motivating_exp3_<STUDY>`에 복원한다. 다른 코드가 같은 Drive STUDY에 들어가면 덮어쓰지 않고 중단한다. 새 실험 코드는 새 STUDY를 사용한다.

체크포인트는 **TTA가 갱신하는 파라미터·버퍼와 optimizer/scaler/난수 상태만 저장**한다. 전체 CLIP 가중치를 seed마다 Drive에 중복 저장하지 않는다. 최초 OpenAI 가중치는 고정 upstream/모델 로더로 재구성한다. 기본100개의 **완전한 배치**마다 진행 중 domain의 예측과 상태를 함께 저장한다.

연결이 끊기거나 중지한 경우 같은 노트북·STUDY·COMBO·PHASE·SEED로 다시 실행한다. 저장 시점 이후 최대99개 완전 배치와 마지막 미완성 배치의 계산을 다시 할 수 있다. 완료된 seed는 출력/metadata 해시를 검증하고 모델 생성 전에 건너뛴다. 진행 중 domain은 스트림을 재생하여 저장된 배치까지 건너뛰며, 그 이전 GPU 적응은 반복하지 않는다.

`status`의 완료 표시만으로 분석 완결성을 주장하지 않는다. 실행기가 해시를 검증하고 최종 평가가 표본 수/역할/seed/샘플ID를 다시 검사한다.

## 데이터와 자원

처음 HF dataset SHA를 고정한 후 네 조합이 같은 SHA를 사용한다. 압축 원본 캐시는 `/content/tta_data_cache`에 저장하며 SHA·손상·severity별로 다른 seed와 역할이 재사용한다. 캐시는 전체 바이트 해시를 검증하고 불완전하거나 손상되면 다시 받는다. Colab 런타임 초기화 시 캐시는 사라질 수 있으며, 이때 다운로드가 다시 필요하다. 사진별 Drive I/O는 하지 않는다.

로컬 임시 디스크는 최소5GiB 여유를 검사하지만 전체 환경·모델·캐시를 위한 실제 여유 공간은 더 필요하다. 원본 캐시+재배열 중 임시 블록이 동시에 존재할 수 있다. Drive에도 결과와 체크포인트 임시 교체를 위한 여유를 남긴다. 실제 필요량은 모델/압축률에 따라 다르다.

조합당117만 장, 전체468만 장을 두 모델에 입력한다. Colab 무료 세션 하나에 끝난다고 가정하지 않는다. 환경 설치시간·프롬프트 생성·데이터다운로드가 포함되므로150장 스모크 처리량을 전체 시간 보장으로 해석하지 않는다.

## 문제 해결

- **CUDA unavailable**: GPU 런타임 선택 후 설치 셀 재실행.
- **CUDA out of memory**: 본실험 전 GPU를 키우거나 checkpointing을 켜고 스모크. 배치64를 줄이면 다른 설계가 된다.
- **다른 소스/설정 감지**: 기존 STUDY를 덮어쓰지 말고 해당 소스 노트북으로 복귀하거나 새 STUDY 사용.
- **HF403/429/다운로드 오류**: 로그에서 실제 오류를 확인. 필요하면 Colab 보안 비밀에 HF_TOKEN 설정 후 재시도. TLS/해시 검증을 끄지 않는다.
- **세션 중단**: 동일 설정 재실행. Drive에 마지막으로 완전히 저장된 checkpoint에서 이어간다. Google Drive 동기화 자체가 실패하면 이전 정상 저장 지점으로 돌아갈 수 있다.
- **설치 중단**: 설치 셀 재실행. 중간에 venv 파일이 생겼다는 이유만으로 설치 성공으로 처리하지 않는다.

## 검증 범위

개발 머신에서 고정 Python3.11 환경의 설치/import 검사와 CPU 회귀/합성 파이프라인을 실행한다. Colab 실제 GPU의 모델 스모크·데이터 다운로드·정확도 재현은 이 노트북의 스모크와 본실험으로 별도 검증해야 한다. 연구 기준은 `PREREG.md`를 따른다.
