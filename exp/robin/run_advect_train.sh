#!/bin/bash
# Merge the advection dataset, then train all surrogates on T = 2 (3 seeds each), 3 models per GPU.
cd -- "$(dirname -- "$0")/../.." || exit 1
PY="${TURNPIKE_PYTHON:-python}"
D=runs/advect_data
.env/bin/python exp/robin/gen_advect_data.py --merge || exit 1
mkdir -p $D/models $D/pred
launch() {  # gpu model extra-args...
  local g=$1 m=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" -u exp/robin/advect_surrogate.py --data $D/advect_data.npz --out $D/models/$m.json --model $m \
    --save-pred $D/pred "$@" > $D/logs/train_$m.log 2>&1 < /dev/null &
}
launch 0 tp_transport --epochs 300 --batch 16
launch 0 cnn_per --epochs 300 --batch 16
launch 0 per_only
launch 1 cnn --epochs 300 --batch 16
launch 1 deeponet --epochs 300 --batch 16
launch 1 fno --epochs 200 --batch 8
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv
