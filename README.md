# Milestone3_motivatingExample

CLIP 온라인 TTA가 zero-shot보다 해로워지는 상황과 무라벨 감지·전환 정책을 평가하는 실험입니다.

## Colab에서 실행

**입력 처리 최적화·기존 결과 재개:** [colab_exp3_v3_fast.ipynb](colab_exp3_v3_fast.ipynb). 기존 `STUDY`와 실험 설정을 유지하세요. 원본 bundle/metadata를 보존하며 알려진 버전만 호환 재개합니다. [호환 범위와 검증 결과](paper_repro/PERFORMANCE.md)를 확인하세요.

**단계·seed 자동 실행:** [colab_exp3_v3_auto.ipynb](colab_exp3_v3_auto.ipynb). 첫 설정 셀에 작업 목록이 만들어지며 위에서부터 실행하면 네 조합을 순서대로 완료합니다. 기존 v3 내장 소스를 유지하므로 같은 STUDY의 완료 결과를 재사용합니다. 장시간 작업이며 중단 후 재실행할 수 있습니다.

[Colab에서 v3 노트북 열기](https://colab.research.google.com/github/hazigamja-capstone/Milestone3_motivatingExample/blob/seoy_experiments3/colab_exp3_v3_ready.ipynb)

1. [colab_exp3_v3_ready.ipynb](colab_exp3_v3_ready.ipynb)를 내려받아 **Google Colab → 노트북 업로드**로 엽니다.
2. **런타임 → 런타임 유형 변경 → GPU**를 선택합니다.
3. 기본 `COMBO='B1'`, `PHASE='smoke'`로 위에서부터 실행하고 Drive를 연결합니다.
4. 성공하면 `paper → calibration → freeze → confirmation → evaluate` 순서로 단계를 바꿉니다. 짧은 세션에서는 seed 하나씩 실행할 수 있습니다.

노트북에 현재 소스와 고정 의존성이 들어 있어 Git push나 별도 소스 업로드 없이 사용할 수 있습니다.
Python3.11/torch2.2.1 환경은 Colab 기본 환경과 분리합니다. 결과·로그·체크포인트는 Drive에 저장하고 원본 데이터 캐시는 로컬 디스크에 둡니다.

- [Colab 상세 안내·재개·문제 해결](paper_repro/COLAB.md)
- [실험 사전등록 v3](paper_repro/PREREG.md)
- [설계 검토와 한계](paper_repro/DESIGN_REVIEW.md)
- [검증 절차](paper_repro/VALIDATION.md)

A 조합이 기본 선택된 노트북은 [colab_exp3.ipynb](colab_exp3.ipynb)입니다. 모든 노트북에서 COMBO를 바꿀 수 있습니다. `colab_exp3b.ipynb`도 동일한 B1 기본 설정으로 유지합니다.

전체 계획은 네 조합 합계468만 장을 처리합니다. 스모크부터 시작하고 Colab 세션 한 번에 끝난다고 가정하지 마세요. 데이터/모델 GPU 스모크와 연구 본실험은 실제 Colab에서 별도로 검증해야 합니다.
