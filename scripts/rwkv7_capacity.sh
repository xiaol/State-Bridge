#!/usr/bin/env bash
# Step 1 of docs/next-steps.md: capacity test on the RNN pair.  Train the state bridge with the
# sender reading the worked solution, evaluate with shuffled and mean-state controls, probe.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'pkill -P $$; exit 130' INT TERM
CFG=configs/qwen3p5_9b_to_rwkv7_1p5b_capacity.yaml
R=runs/qwen3p5_9b_to_rwkv7_1p5b_capacity
BASE=runs/qwen3p5_9b_to_rwkv7_1p5b
G=${1:-0}
mkdir -p "$R"
cp "$BASE/eval_receiver.json" "$BASE/eval_receiver.jsonl" "$BASE/eval_sender.json" "$BASE/eval_sender.jsonl" "$R/" 2>/dev/null || true
python -m state_bridge train   -c "$CFG" models.sender.device=cuda:$G models.receiver.device=cuda:$G > "$R/train.log" 2>&1
python -m state_bridge eval    -c "$CFG" models.sender.device=cuda:$G models.receiver.device=cuda:$G > "$R/eval.log" 2>&1
python -m state_bridge observe -c "$CFG" eval.batch_size=16 models.sender.device=cuda:$G models.receiver.device=cuda:$G > "$R/observe.log" 2>&1
python scripts/compare_runs.py --base "$R" --runs "$R" "runs/qwen3p5_9b_to_rwkv7_1p5b" "runs/qwen3p5_9b_to_rwkv7_1p5b_state_tuning" --out "$R/comparison.md" > "$R/compare.log" 2>&1
echo CAPACITY_DONE
