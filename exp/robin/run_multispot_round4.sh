#!/bin/bash
# Round 4: temporal correction with clipped distance-to-end channels, with (tp_tce) and without (tp_tce_nolayer)
# the snapshot end layers; 3 seeds each, trained on T = 1 only.
cd -- "$(dirname -- "$0")/../.." || exit 1
PY="${TURNPIKE_PYTHON:-python}"
D=runs/multispot_data/data.npz
O=runs/multispot_data/models
S="exp/robin/multispot_surrogate.py --data $D --fp32 --save-pred $O/pred"
for s in 0 1 2; do
  "$PY" $S --kind nl --model tp_tce --seeds $s --epochs 800 --device cuda:0 --out $O/nl_tp_tce_s$s.json > $O/nl_tp_tce_s$s.log 2>&1 < /dev/null &
  "$PY" $S --kind nl --model tp_tce_nolayer --seeds $s --epochs 800 --device cuda:1 --out $O/nl_tp_tce_nolayer_s$s.json > $O/nl_tp_tce_nolayer_s$s.log 2>&1 < /dev/null &
done
wait
echo "$(date) round 4 done"
