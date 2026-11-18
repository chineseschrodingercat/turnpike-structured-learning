#!/bin/bash
# Train all multi-spot models on runs/multispot_data/data.npz (after gen_multispot_data.py --merge).
# GPU 0: semilinear tp_local, three seeds in parallel. GPU 1: everything else, in sequence.
cd -- "$(dirname -- "$0")/../.." || exit 1
PY="${TURNPIKE_PYTHON:-python}"
D=runs/multispot_data/data.npz
O=runs/multispot_data/models
mkdir -p $O
S="exp/robin/multispot_surrogate.py --data $D --fp32 --save-pred $O/pred"
for s in 0 1 2; do
  nohup "$PY" $S --kind nl --model tp_local --seeds $s --epochs 800 --batch 64 --device cuda:0 --out $O/nl_tp_local_s$s.json > $O/nl_tp_local_s$s.log 2>&1 < /dev/null &
done
(
  "$PY" $S --kind lin --model ref_only --device cuda:1 --out $O/lin_ref_only.json > $O/lin_ref_only.log 2>&1
  "$PY" $S --kind nl --model ref_only --device cuda:1 --out $O/nl_ref_only.json > $O/nl_ref_only.log 2>&1
  "$PY" $S --kind lin --model tp --seeds 0 1 2 --epochs 3000 --batch 64 --device cuda:1 --out $O/lin_tp.json > $O/lin_tp.log 2>&1
  "$PY" $S --kind nl --model tp --seeds 0 --epochs 3000 --batch 64 --device cuda:1 --out $O/nl_tp_nocorr.json > $O/nl_tp_nocorr.log 2>&1
  "$PY" $S --kind nl --model cnn --seeds 0 1 2 --epochs 300 --batch 64 --train-T 1 2 --device cuda:1 --out $O/nl_cnn.json > $O/nl_cnn.log 2>&1
  "$PY" $S --kind nl --model cnn_res --seeds 0 1 2 --epochs 300 --batch 64 --train-T 1 2 --device cuda:1 --out $O/nl_cnn_res.json > $O/nl_cnn_res.log 2>&1
  "$PY" $S --kind lin --model cnn --seeds 0 1 2 --epochs 300 --batch 64 --train-T 1 2 --device cuda:1 --out $O/lin_cnn.json > $O/lin_cnn.log 2>&1
) > $O/gpu1_queue.log 2>&1 < /dev/null &
wait
echo "all done"
