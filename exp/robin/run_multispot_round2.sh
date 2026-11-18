#!/bin/bash
# Round 2: phase-aware correction (tp_qp), remaining tp_local_bnd seeds, and fair baselines (same training horizons,
# longer CNN training). GPU 0: tp_qp x 3 seeds. GPU 1: tp_local_bnd seeds 1-2, tp_qp on T = {1, 2}, CNN queue.
cd -- "$(dirname -- "$0")/../.." || exit 1
PY="${TURNPIKE_PYTHON:-python}"
D=runs/multispot_data/data.npz
O=runs/multispot_data/models
S="exp/robin/multispot_surrogate.py --data $D --fp32 --save-pred $O/pred"
for s in 0 1 2; do
  "$PY" $S --kind nl --model tp_qp --seeds $s --epochs 800 --device cuda:0 --out $O/nl_tp_qp_s$s.json > $O/nl_tp_qp_s$s.log 2>&1 < /dev/null &
done
for s in 1 2; do
  "$PY" $S --kind nl --model tp_local_bnd --seeds $s --epochs 800 --device cuda:1 --out $O/nl_tp_local_bnd_s$s.json > $O/nl_tp_local_bnd_s$s.log 2>&1 < /dev/null &
done
"$PY" $S --kind nl --model tp_qp --seeds 0 --epochs 800 --train-T 1 2 --tag _T12 --device cuda:1 --out $O/nl_tp_qp_T12_s0.json > $O/nl_tp_qp_T12_s0.log 2>&1 < /dev/null &
(
  "$PY" $S --kind nl --model cnn_res --epochs 300 --train-T 1 --tag _T1 --device cuda:1 --out $O/nl_cnn_res_T1.json > $O/nl_cnn_res_T1.log 2>&1
  "$PY" $S --kind lin --model cnn_res --epochs 300 --train-T 1 2 --device cuda:1 --out $O/lin_cnn_res.json > $O/lin_cnn_res.log 2>&1
  "$PY" $S --kind lin --model cnn_res --epochs 300 --train-T 1 --tag _T1 --device cuda:1 --out $O/lin_cnn_res_T1.json > $O/lin_cnn_res_T1.log 2>&1
  "$PY" $S --kind nl --model cnn_res --epochs 1200 --train-T 1 2 --tag _e1200 --device cuda:1 --out $O/nl_cnn_res_e1200.json > $O/nl_cnn_res_e1200.log 2>&1
  "$PY" $S --kind nl --model cnn_res --epochs 1200 --train-T 1 --tag _T1_e1200 --device cuda:1 --out $O/nl_cnn_res_T1_e1200.json > $O/nl_cnn_res_T1_e1200.log 2>&1
  "$PY" $S --kind nl --model cnn --epochs 1200 --train-T 1 2 --tag _e1200 --device cuda:1 --out $O/nl_cnn_e1200.json > $O/nl_cnn_e1200.log 2>&1
  "$PY" $S --kind lin --model cnn --epochs 1200 --train-T 1 2 --tag _e1200 --device cuda:1 --out $O/lin_cnn_e1200.json > $O/lin_cnn_e1200.log 2>&1
) < /dev/null &
wait
echo "$(date) round 2 done"
