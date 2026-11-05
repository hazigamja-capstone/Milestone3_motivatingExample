"""후속 실험 1단계(재현): seed와 학습률을 바꿔 가며 run.py를 반복 실행한다.

이미 결과 파일이 있는 조합은 건너뛰므로, 중간에 끊겨도 같은 명령으로 이어서 돌릴 수 있다.

예: python sweep.py                       # 기본 격자 (ETA, SAR x seed 0,1,2 x lr 2.5e-4, 5e-4, 1e-3)
    python sweep.py --lrs 1e-3            # 학습률 1e-3만 먼저
    python sweep.py --dry                 # 실행할 명령만 출력
결과: outputs_followup/*.npz → python followup.py
"""
import argparse
import os
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
    args, rest = ap.parse_known_args()  # 나머지 인자는 run.py로 그대로 넘긴다

    jobs = []
    for lr in args.lrs.split(","):
        for seed in args.seeds.split(","):
            for method in args.methods.split(","):
                name = f"imagenet_c_{args.model}_{method}_dirichlet_s{seed}_lr{float(lr):g}.npz"
                if os.path.exists(os.path.join(args.out, name)):
                    print("있음, 건너뜀:", name)
                    continue
                cmd = [sys.executable, "run.py", "--model", args.model, "--method", method, "--seed", seed,
                       "--lr", lr, "--domains", args.domains, "--out", args.out] + rest
                if "L-14" in args.model and "--grad_ckpt" not in rest:
                    cmd.append("--grad_ckpt")
                jobs.append(cmd)

    print(f"실행할 조합 {len(jobs)}개")
    for i, cmd in enumerate(jobs):
        print(f"\n[{i + 1}/{len(jobs)}]", " ".join(cmd[1:]), flush=True)
        if not args.dry:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
