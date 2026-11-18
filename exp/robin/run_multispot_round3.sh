#!/bin/bash
# Round 3: temporal-context correction (tp_tc), 3 seeds on T = 1, plus one seed on T = {1, 2}.
cd -- "$(dirname -- "$0")/../.." || exit 1
PY="${TURNPIKE_PYTHON:-python}"
D=runs/multispot_data/data.npz
O=runs/multispot_data/models
S="exp/robin/multispot_surrogate.py --data $D --fp32 --save-pred $O/pred"
"$PY" $S --kind nl --model tp_tc --seeds 0 --epochs 800 --device cuda:0 --out $O/nl_tp_tc_s0.json > $O/nl_tp_tc_s0.log 2>&1 < /dev/null &
"$PY" $S --kind nl --model tp_tc --seeds 1 --epochs 800 --device cuda:0 --out $O/nl_tp_tc_s1.json > $O/nl_tp_tc_s1.log 2>&1 < /dev/null &
"$PY" $S --kind nl --model tp_tc --seeds 2 --epochs 800 --device cuda:1 --out $O/nl_tp_tc_s2.json > $O/nl_tp_tc_s2.log 2>&1 < /dev/null &
"$PY" $S --kind nl --model tp_tc --seeds 0 --epochs 800 --train-T 1 2 --tag _T12 --device cuda:1 --out $O/nl_tp_tc_T12_s0.json > $O/nl_tp_tc_T12_s0.log 2>&1 < /dev/null &
wait
echo "$(date) round 3 done"
