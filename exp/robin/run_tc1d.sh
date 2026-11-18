#!/bin/bash
# Log section 32: the convolutional implementation (cnn_res_tc) and the hybrid on the 1D benchmarks E01 (linear) and
# E08 (semilinear). Window half-width L + W dt = T_train / 2 for tc (dt = 1/320), shorter for the hybrid. 3 seeds each.
# Usage: bash run_tc1d.sh [all|e01|e08|hyb2]. One BLAS thread per process: the E01 loader computes lambda_1(beta) for all
# 1000 samples with small dense eigensolves, which stall when 12 processes each start a full BLAS thread pool.
cd -- "$(dirname -- "$0")/../.." || exit 1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
PART=${1:-all}
PY="${TURNPIKE_PYTHON:-python}"
S=exp/robin/tc1d_surrogate.py
D1=runs/turnpike/turnpike_data.npz
D8=runs/semilinear/data_semilinear.npz
O=runs/tc1d
mkdir -p $O/e01 $O/e08
W1="--tdil 1 2 4 8 16 32 64 --edge-L 0.1"           # 0.1 + 127/320 = 0.497
W2="--tdil 1 2 4 8 16 32 64 128 --edge-L 0.2"       # 0.2 + 255/320 = 0.997
W4="--tdil 1 2 4 8 16 32 64 128 256 --edge-L 0.4"   # 0.4 + 511/320 = 1.997
WH="--tdil 1 2 4 8 16 32 --edge-L 0.05"             # 0.05 + 63/320 = 0.247 (hybrid, T_train = 1)
G="--fp32 --epochs 1200 --batch 64 --lr 1e-3"
TP="--epochs 6000 --batch 700 --lr 2e-3 --threads 8"
run() { local dir=$1 tag=$2; shift 2; "$PY" $S --out $O/$dir/$tag.json --save-pred $O/$dir/pred "$@" > $O/$dir/$tag.log 2>&1 < /dev/null; }
# non-learned structural floor of tc; TurnpikeNet references (float64, CPU, the E01/E08 protocol);
# convolutional implementation and hybrid (float32, GPU)
if [ "$PART" = all ] || [ "$PART" = e01 ]; then
  for T in 1 2 4; do run e01 oracle_T$T --data $D1 --model local_oracle --train-T $T --tag _T$T & done
  run e01 tp_T1 --data $D1 --model tp --train-T 1 $TP --tag _T1 &
  run e01 tc_T1 --data $D1 --model tc --train-T 1 $W1 $G --device cuda:0 --tag _T1 &
  run e01 tc_T2 --data $D1 --model tc --train-T 2 $W2 $G --device cuda:0 --tag _T2 &
  run e01 tc_T4 --data $D1 --model tc --train-T 4 $W4 $G --device cuda:0 --tag _T4 &
  run e01 hyb_T1 --data $D1 --model hyb --train-T 1 $WH $G --device cuda:0 --tag _T1 &
fi
if [ "$PART" = all ] || [ "$PART" = e08 ]; then
  for T in 1 2; do run e08 oracle_T$T --data $D8 --model local_oracle --train-T $T --tag _T$T & done
  run e08 tp_T1 --data $D8 --model tp --train-T 1 $TP --tag _T1 &
  run e08 tc_T1 --data $D8 --model tc --train-T 1 $W1 $G --device cuda:1 --tag _T1 &
  run e08 tc_T2 --data $D8 --model tc --train-T 2 $W2 $G --device cuda:1 --tag _T2 &
  run e08 hyb_T1 --data $D8 --model hyb --train-T 1 $WH $G --device cuda:1 --tag _T1 &
fi
# two-stage hybrid: TurnpikeNet first (stage 1, float32 here), then frozen, then the convolutional correction
if [ "$PART" = all ] || [ "$PART" = hyb2 ]; then
  run e01 hyb2_T1 --data $D1 --model hyb2 --train-T 1 $WH $G --device cuda:1 --tag _T1 &
  run e08 hyb2_T1 --data $D8 --model hyb2 --train-T 1 $WH $G --device cuda:1 --tag _T1 &
fi
wait
echo "$(date) tc1d done"
