#!/usr/bin/env bash
# After the capacity bridge is trained: the three eval modes in parallel on separate GPUs
# (each in its own run dir to avoid clobbering), then the probes, then the comparison.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/qwen3p5_9b_to_rwkv7_1p5b_capacity.yaml
N=qwen3p5_9b_to_rwkv7_1p5b_capacity
R=runs/$N
while [ ! -f "$R/bridge.pt" ]; do sleep 20; done; sleep 20
python -m state_bridge eval -c "$CFG" --modes bridged models.sender.device=cuda:0 models.receiver.device=cuda:0 > "$R/eval_bridged.log" 2>&1 &
P0=$!
python -m state_bridge eval -c "$CFG" run_name=${N}_shuf eval.checkpoint=$R/bridge.pt --modes bridged_shuffled models.sender.device=cuda:2 models.receiver.device=cuda:2 > "$R/eval_shuffled.log" 2>&1 &
P2=$!
python -m state_bridge eval -c "$CFG" run_name=${N}_abl eval.checkpoint=$R/bridge.pt --modes bridged_ablated models.sender.device=cuda:3 models.receiver.device=cuda:3 > "$R/eval_ablated.log" 2>&1 &
P3=$!
wait $P0
python -m state_bridge observe -c "$CFG" eval.batch_size=16 models.sender.device=cuda:0 models.receiver.device=cuda:0 > "$R/observe.log" 2>&1
wait $P2 $P3
cp runs/${N}_shuf/eval_bridged_shuffled.json runs/${N}_shuf/eval_bridged_shuffled.jsonl runs/${N}_abl/eval_bridged_ablated.json runs/${N}_abl/eval_bridged_ablated.jsonl "$R/"
python scripts/compare_runs.py --base "$R" --runs "$R" runs/qwen3p5_9b_to_rwkv7_1p5b runs/qwen3p5_9b_to_rwkv7_1p5b_state_tuning --out "$R/comparison.md" > "$R/compare.log" 2>&1
echo CAPACITY_EVAL_DONE
