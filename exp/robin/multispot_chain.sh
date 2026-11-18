#!/bin/bash
# Server-side chain: wait for gen_multispot_data.py, merge, short timing runs on real data, then full training.
cd -- "$(dirname -- "$0")/../.." || exit 1
D=runs/multispot_data
PY="${TURNPIKE_PYTHON:-python}"
while pgrep -f "gen_multispot_data.py --out" > /dev/null; do sleep 60; done
n=$(ls $D/samples | wc -l)
echo "$(date) generation finished with $n samples"
[ "$n" -ge 1350 ] || { echo "incomplete, stopping"; exit 1; }
.env/bin/python exp/robin/gen_multispot_data.py --out $D/data.npz --merge > $D/merge.log 2>&1 || { echo "merge failed"; exit 1; }
cat $D/merge.log
mkdir -p $D/timing
S="exp/robin/multispot_surrogate.py --data $D/data.npz --fp32"
/usr/bin/time -f "%e s" "$PY" $S --kind nl --model tp_local --seeds 0 --epochs 3 --device cuda:0 --out $D/timing/tp_local.json > $D/timing/tp_local.log 2>&1 &
/usr/bin/time -f "%e s" "$PY" $S --kind nl --model cnn --seeds 0 --epochs 2 --train-T 1 2 --device cuda:1 --out $D/timing/cnn.json > $D/timing/cnn.log 2>&1 &
wait
echo "$(date) timing runs done"
tail -n 3 $D/timing/*.log
bash exp/robin/run_multispot_train.sh
echo "$(date) training done"
