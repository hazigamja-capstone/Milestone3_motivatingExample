#!/usr/bin/env bash
# usage: bash run_all.sh /path/to/DATA [RN50|ViT-B-16]
set -euo pipefail
DATA=${1:?data root}
ARCH=${2:-RN50}
C="--data_root $DATA --arch $ARCH"
MAIN="swapprompt"
HAS_C=$( [ -d "$DATA/imagenet-c" ] && echo 1 || echo 0 )
EXTRA=$( [ "$HAS_C" = 1 ] && echo "imagenet_c:all" || echo "" )

python experiments/cache_features.py $C --datasets $MAIN $EXTRA
python experiments/e1_entropy_vs_correctness.py $C --datasets $MAIN $EXTRA

python experiments/run_episodic.py $C --datasets $MAIN --n_samples 1000 --steps 4
python experiments/e2_overconfidence.py $C
python experiments/e3_view_selection.py $C

if [ "$HAS_C" = 1 ]; then STREAM="imagenet_c:sev=5"; else STREAM="imagenet_a imagenet_r imagenet_sketch imagenet_v2"; fi
for ORDER in iid dirichlet class_sorted; do
  python experiments/e4_continual_collapse.py $C --datasets $STREAM --order $ORDER
done
python experiments/e4_continual_collapse.py $C --datasets $STREAM --order dirichlet --objective ent_div
python experiments/e4_continual_collapse.py $C --datasets $STREAM --order iid --params norm --lr 1e-3

python experiments/e5_scale_sensitivity.py $C --datasets $MAIN
python experiments/e6_semantic_blindness.py $C --datasets $MAIN
