"""후속 실험 1단계(재현): seed와 학습률을 바꿔 가며 run.py를 돌린다.

run.py 프로세스 하나에서 데이터를 한 번만 받고 모든 조합을 차례로 돌린다.
이미 결과 파일이 있는 조합은 건너뛰므로, 중간에 끊겨도 같은 명령으로 이어서 돌릴 수 있다.

예: python sweep.py                       # 기본 격자 (ETA, SAR x seed 0,1,2 x lr 1e-3, 5e-4, 2.5e-4)
    python sweep.py --lrs 1e-3            # 학습률 1e-3만 먼저
    python sweep.py --no_cache            # 로컬: 데이터를 디스크에 저장하지 않고 메모리에만 올림
    python sweep.py --dry                 # 실행할 명령만 출력
결과: outputs_followup/*.npz → python followup.py
"""
import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--methods", default="eta,sar")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lrs", default="1e-3,5e-4,2.5e-4", help="앞에 둔 학습률부터 돌린다")
    ap.add_argument("--domains", default="defocus_blur")
    ap.add_argument("--out", default="outputs_followup")
    ap.add_argument("--dry", action="store_true")
    args, rest = ap.parse_known_args()  # 나머지 인자(--no_cache 등)는 run.py로 그대로 넘긴다

    cmd = [sys.executable, "run.py", "--model", args.model, "--method", args.methods, "--seed", args.seeds,
           "--lr", args.lrs, "--domains", args.domains, "--out", args.out] + rest
    if "L-14" in args.model and "--grad_ckpt" not in rest:
        cmd.append("--grad_ckpt")
    print(" ".join(cmd[1:]), flush=True)
    if not args.dry:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
