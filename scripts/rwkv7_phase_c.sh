#!/usr/bin/env bash
# RWKV-7 receiver, phase C (after the bridge is trained): bridged evaluation with controls,
# hand-off sweep, observability, compute report, comparison.  GPUs are shared with other jobs,
# so each step waits for enough free memory and retries on CUDA out-of-memory.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'pkill -P $$; exit 130' INT TERM   # killing this script must also kill its running python child
CFG=configs/qwen3p5_9b_to_rwkv7_1p5b.yaml
NAME=qwen3p5_9b_to_rwkv7_1p5b
R=runs/$NAME
while [ ! -f "$R/bridge.pt" ]; do sleep 30; done

pick_pair() {  # GS: sender gpu (>= 21 GB free), GR: receiver gpu (>= 7 GB free, same card if >= 29 GB)
  GS=$(bash scripts/wait_gpu.sh 20000)
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GS" | tr -d ' ')
  if [ "$FREE" -ge 27000 ]; then GR=$GS; else GR=$(bash scripts/wait_gpu.sh 6000 "$GS"); fi
}
run_retry() {  # run_retry <logfile> <cmd...>; retries on OOM up to 8 times
  local log=$1; shift
  for attempt in 1 2 3 4 5 6 7 8; do
    pick_pair
    "$@" models.sender.device=cuda:$GS models.receiver.device=cuda:$GR > "$log" 2>&1 && return 0
    if grep -q -E "out of memory|OutOfMemoryError" "$log"; then echo "OOM on attempt $attempt ($log), retrying"; sleep 120; else return 1; fi
  done
  return 1
}

run_retry "$R/eval_bridged.log" python -m state_bridge eval -c "$CFG" --modes bridged,bridged_shuffled,bridged_ablated eval.batch_size=16
run_retry "$R/handoff.log"      python -m state_bridge handoff -c "$CFG" handoff.batch_size=16
run_retry "$R/observe.log"      python -m state_bridge observe -c "$CFG" eval.batch_size=16
python -m state_bridge compute -c "$CFG" > "$R/compute.log" 2>&1
python scripts/compare_runs.py --base "$R" --runs "$R" "runs/${NAME}_state_tuning" --out "$R/comparison.md" > "$R/compare.log" 2>&1
echo PHASE_C_DONE
