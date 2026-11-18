#!/bin/bash
# Round 5: tp_tce_nolayer for 1600 epochs, window W = 63 (L = 0.3, as round 4) vs W = 127 (L = 0.1, so that
# T >= 2 (L + W dt) still holds at T = 1); 2 seeds each, trained on T = 1 only. Separates training budget from receptive field.
cd -- "$(dirname -- "$0")/../.." || exit 1
PY="${TURNPIKE_PYTHON:-python}"
D=runs/multispot_data/data.npz
O=runs/multispot_data/models
S="exp/robin/multispot_surrogate.py --data $D --fp32 --save-pred $O/pred --kind nl --model tp_tce_nolayer --epochs 1600"
for s in 0 1; do
  "$PY" $S --seeds $s --tag _e1600 --device cuda:$s --out $O/nl_tp_tce_nolayer_e1600_s$s.json > $O/nl_tp_tce_nolayer_e1600_s$s.log 2>&1 < /dev/null &
  "$PY" $S --seeds $s --tdil 1 2 4 8 16 32 64 --edge-L 0.1 --tag _W127_e1600 --device cuda:$s \
    --out $O/nl_tp_tce_nolayer_W127_e1600_s$s.json > $O/nl_tp_tce_nolayer_W127_e1600_s$s.log 2>&1 < /dev/null &
done
wait
echo "$(date) round 5 done"
