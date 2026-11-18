#!/bin/bash
# Round 6: horizon-independent residual CNN (cnn_res_tc), 1200 epochs as cnn_res_e1200, 3 seeds each:
# W = 63 (L = 0.3) and W = 127 (L = 0.1) on T = 1, and W = 63 on T = {1, 2} for an equal-data comparison.
cd -- "$(dirname -- "$0")/../.." || exit 1
PY="${TURNPIKE_PYTHON:-python}"
D=runs/multispot_data/data.npz
O=runs/multispot_data/models
S="exp/robin/multispot_surrogate.py --data $D --fp32 --save-pred $O/pred --kind nl --model cnn_res_tc --epochs 1200 --seeds 0 1 2"
"$PY" $S --device cuda:0 --out $O/nl_cnn_res_tc.json > $O/nl_cnn_res_tc.log 2>&1 < /dev/null &
"$PY" $S --tdil 1 2 4 8 16 32 64 --edge-L 0.1 --tag _W127 --device cuda:1 --out $O/nl_cnn_res_tc_W127.json > $O/nl_cnn_res_tc_W127.log 2>&1 < /dev/null &
"$PY" $S --train-T 1 2 --tag _T12 --device cuda:0 --out $O/nl_cnn_res_tc_T12.json > $O/nl_cnn_res_tc_T12.log 2>&1 < /dev/null &
wait
echo "$(date) round 6 done"
