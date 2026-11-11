# Colab 입력 파이프라인 최적화

`colab_exp3_v3_fast.ipynb`를 Colab에 업로드하고 기존 `STUDY`, 조합, seed,
데이터 SHA, gradient checkpointing 선택을 유지한다. 기존 수동/자동 노트북은
변경하지 않았다. 새 최적화 노트북도 같은 64 배치, 학습률, AMP 정밀도,
모델/가중치/ensemble 프롬프트, 샘플 순서, TTA 업데이트 규칙을 사용한다.

## 기존 결과와 재개

지원하는 기존 버전은 `753f405329fe3569291af926a5ee2adf54eeea10`에 포함된
`colab_exp3_v3_ready.ipynb`의 과학 소스 bundle이다. 정확한 ZIP SHA와 파일별
기존/최적화 SHA는 `compatibility_manifest.json`에 고정되어 있다.

1. 기존 설치 셀을 먼저 그대로 실행한다. 기존 Drive `source_bundle.zip`과
   원본 로컬 소스 검사가 그대로 적용된다. 다른 bundle이면 계속 거부한다.
2. 최적화 bundle은 별도 로컬 디렉터리에 설치하고, Drive의
   `execution_backends/<backend ZIP SHA>/source_bundle.zip`에 별도로 보존한다.
   원본 소스 디렉터리와 원본 `source_bundle.zip`은 수정하지 않는다.
3. 실행 시 기존 bundle의 정확한 ZIP SHA와 최적화 소스의 실제 파일별 SHA를
   모두 검증한다. 이 특정 조합만 기존 과학 identity에 명시적으로 연결한다.
   실제 실행 소스 SHA와 검증 범위는 `execution_backends/<record SHA>.json`에
   추가 기록한다. 이미 존재하는 이력은 덮어쓰지 않으며 변조 시 중단한다.
4. 기존 계획/metadata/완료 출력/체크포인트 identity와 데이터 해시 검사는
   계속 수행한다. 기존 `meta.json`은 재접속 시 그대로 보존한다. domain 변경도
   기존 metadata를 덮어쓰지 않고 새 출력 루트를 요구한다. 완료 seed는 모델
   생성 전에 검증하여 건너뛴다. 중간 체크포인트는 원래 identity로 복원한다.
5. 기존 계획이 있어도 최적화 backend마다 GPU smoke를 한 번 실행한다.
   알 수 없는 bundle/소스, 설정/torch 버전/seed 변경, 변조된 결과는 자동 이관하지 않는다.

원본 코드만 CLI로 사용하던 출력 폴더에 기존 `source_bundle.zip`이 없으면
이 호환 경로가 적용되지 않는다. 원본 bundle이나 metadata를 새로 조작해
재개하지 말고, 원본 실행 환경에서 이어서 실행한다. 새 출력 루트에서 직접
실행하는 최적화 CLI는 실제 새 소스 SHA를 identity로 사용한다.

## 변경한 입력 처리

- 결정론적인 CLIP eval transform만 최대 4 CPU 스레드에서 병렬 처리한다.
  완료 순서와 관계없이 입력 순서대로 반환하며, stochastic/알 수 없는 변환은 거부한다.
  배치 경계와 마지막 불완전 배치를 유지한다. 대기 작업과 배치 큐는 제한되어 있다.
  조기 종료/예외 때 worker를 회수하고 원본 캐시 잠금을 해제한다.
- 기존 v1 압축 캐시를 그대로 읽는다. 전체 SHA-256 검증은 매번 유지하고,
  메모리의 작은 offset index와 read-only mmap으로 같은 클래스/샘플 순열을 읽는다.
  클래스별 pickle 임시 파일 재기록·재읽기가 사라진다. 새 캐시 포맷이나
  압축 이미지 전체 RAM 적재는 필요하지 않다.
- 고정 upstream `models/model.py` 해시, frozen text encoder, ensemble 설정이
  일치할 때 두 번째 zero-shot 모델에 첫 모델의 정확한 텍스트 특징/토큰 텐서를
  복제한다. visual 모델은 독립적으로 정상 생성하므로 모델 초기화 RNG 소비는
  유지한다. 알려진 upstream이 아니면 원래 텍스트 특징 계산을 사용한다.
- GPU 입력 배치를 pinned memory에 준비하고 `non_blocking=True`로 전송한다.
  전송과 모델 연산은 같은 CUDA stream에서 순서를 유지한다. AMP와 dtype은 유지한다.

## 검증 범위

합성 이미지로 직렬/병렬 전처리 텐서, label, sample ID, 64/64/22 배치,
RNG 상태, 모든 역할/여러 seed의 캐시 순열, 캐시 손상 복구, 예외/조기 종료를
검증한다. 실제 open_clip eval transform도 직렬 결과와 정확히 비교한다.
고정 upstream 생성자의 AST를 사용해 텍스트 특징/로짓/RNG/모델 독립성을 비교한다.
기존 identity의 합성 체크포인트 중단→재개는 연속 실행과 정확히 같은 결과와
모델/업데이트 상태를 내며, 실제 runner의 기존 중간 재개 회귀 검사도 유지한다.
기존 bundle/plan/metadata/완료 결과 재사용과 변경 거부, 두 bundle의 독립 설치도 검사한다.
실제 개인 Drive 결과는 이 환경에 제공되지 않았으므로 합성 fixture로 검증했다.

```sh
PAPER_MODEL_SOURCE=/path/to/pinned/classification/models/model.py \
PAPER_SAR_SOURCE=/path/to/pinned/classification/methods/sar.py \
python -m unittest discover -s tests -v
python -m paper_repro.benchmark_input
```

Python3.11.17 / torch2.2.1+cpu / torchvision0.17.1+cpu / open_clip2.24.0와
Python3.12 / torch2.15.0+cpu 환경에서 회귀 검사를 수행했다.
원본·최적화 bundle을 별도 디렉터리에 복원한 뒤의 회귀 실행도 검증했다.

CPU 입력 마이크로벤치마크(Python3.12 / torch2.15.0+cpu, 1,000개 256px WebP, torch intra-op 1 thread,
3회 중앙값): 전처리 1 worker **3.7735초** → 4 workers **1.0598초**,
confirmation 600개 캐시 읽기 **0.0485초** → **0.0346초**.
이 수치는 실제 CLIP GPU 학습 속도 향상률을 뜻하지 않는다.

CUDA GPU가 없는 개발 환경에서는 pinned/비동기 전송 검사를 skip한다.
실제 Colab의 고정 Python3.11/torch2.2.1+cu121 환경에서 smoke를 실행하고,
본실험 전에 `VALIDATION.md`의 동일 계획 GPU 중단→재개 비교를 수행한다.
CPU 검사는 CUDA AMP 수치 동등성이나 전체 학습 처리량을 보증하지 않는다.

최적화 노트북 재생성은 저장소 루트에서 `python -m paper_repro.make_fast_colab`.
원본 노트북을 재생성하거나 기존 study의 source_bundle을 교체하지 않는다.
코드가 변경되어 manifest 검증에 실패하면 임의로 해시 검사를 끄지 않는다.
