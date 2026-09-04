#!/usr/bin/env bash
# RWKV-7 receiver, phase B (needs phase A's targets_train.jsonl): train the state bridge and
# the state-tuning control, evaluate with controls, hand-off sweep, observability, compute.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/qwen3p5_9b_to_rwkv7_1p5b.yaml
NAME=qwen3p5_9b_to_rwkv7_1p5b
R=runs/$NAME
while [ ! -f "$R/targets_train.jsonl" ]; do sleep 30; done

pick_pair() {  # sets GS (sender gpu) and GR (receiver gpu)
  GS=$(bash scripts/wait_gpu.sh 22000)
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GS" | tr -d ' ')
  if [ "$FREE" -ge 36000 ]; then GR=$GS; else GR=$(bash scripts/wait_gpu.sh 12000 "$GS"); fi
}

# 1. bridge (sender + receiver) and control (receiver only) in parallel when memory allows
pick_pair
python -m state_bridge train -c "$CFG" models.sender.device=cuda:$GS models.receiver.device=cuda:$GR "data.sender_generations=$R/gen_sender_train.*.jsonl" > "$R/train.log" 2>&1 &
P_MAIN=$!
sleep 90
GC=$(bash scripts/wait_gpu.sh 12000)
python -m state_bridge train -c "$CFG" run_name=${NAME}_state_tuning bridge.type=prompt_tuning models.receiver.device=cuda:$GC > "runs/${NAME}_state_tuning.train.log" 2>&1 &
P_CTRL=$!
wait $P_CTRL
GC=$(bash scripts/wait_gpu.sh 12000)
python -m state_bridge eval -c "$CFG" run_name=${NAME}_state_tuning --modes bridged models.receiver.device=cuda:$GC > "runs/${NAME}_state_tuning.eval.log" 2>&1 &
P_CTRL_EVAL=$!
wait $P_MAIN

# 2. bridged evaluation with controls
pick_pair
python -m state_bridge eval -c "$CFG" --modes bridged,bridged_shuffled,bridged_ablated models.sender.device=cuda:$GS models.receiver.device=cuda:$GR > "$R/eval_bridged.log" 2>&1
wait $P_CTRL_EVAL

# 3. hand-off sweep, observability, compute, comparison
pick_pair
python -m state_bridge handoff -c "$CFG" models.sender.device=cuda:$GS models.receiver.device=cuda:$GR > "$R/handoff.log" 2>&1
pick_pair
python -m state_bridge observe -c "$CFG" eval.batch_size=24 models.sender.device=cuda:$GS models.receiver.device=cuda:$GR > "$R/observe.log" 2>&1
python -m state_bridge compute -c "$CFG" > "$R/compute.log" 2>&1
python scripts/compare_runs.py --base "$R" --runs "$R" "runs/${NAME}_state_tuning" --out "$R/comparison.md" > "$R/compare.log" 2>&1
echo PHASE_B_DONE
