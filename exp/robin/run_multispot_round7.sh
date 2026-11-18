#!/bin/bash
# Round 7: linear case. Horizon-independent residual CNN (cnn_res_tc, W = 63 / L = 0.3 and W = 127 / L = 0.1) on T = 1,
# plus the zero-padded residual CNN at the same 1200 epochs on T = 1 and T = {1, 2} (round 1 ran it for 300 only); 3 seeds each.
cd -- "$(dirname -- "$0")/../.." || exit 1
PY="${TURNPIKE_PYTHON:-python}"
D=runs/multispot_data/data.npz
O=runs/multispot_data/models
S="exp/robin/multispot_surrogate.py --data $D --fp32 --save-pred $O/pred --kind lin --epochs 1200 --seeds 0 1 2"
"$PY" $S --model cnn_res_tc --device cuda:0 --out $O/lin_cnn_res_tc.json > $O/lin_cnn_res_tc.log 2>&1 < /dev/null &
"$PY" $S --model cnn_res_tc --tdil 1 2 4 8 16 32 64 --edge-L 0.1 --tag _W127 --device cuda:1 \
  --out $O/lin_cnn_res_tc_W127.json > $O/lin_cnn_res_tc_W127.log 2>&1 < /dev/null &
"$PY" $S --model cnn_res --train-T 1 2 --tag _e1200 --device cuda:0 --out $O/lin_cnn_res_e1200.json > $O/lin_cnn_res_e1200.log 2>&1 < /dev/null &
"$PY" $S --model cnn_res --tag _T1_e1200 --device cuda:1 --out $O/lin_cnn_res_T1_e1200.json > $O/lin_cnn_res_T1_e1200.log 2>&1 < /dev/null &
wait
echo "$(date) round 7 done"
