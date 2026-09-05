#!/usr/bin/env bash
# Capacity test v2: answer-only targets.  Bridge (sender reads question + answer) on GPU A,
# state-tuning control (no sender) on GPU B; then bridged / shuffled / mean-state / control evals
# and the probes.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'pkill -P $$; exit 130' INT TERM
CFG=configs/qwen3p5_9b_to_rwkv7_1p5b_answer.yaml
N=qwen3p5_9b_to_rwkv7_1p5b_answer
R=runs/$N
BASE=runs/qwen3p5_9b_to_rwkv7_1p5b
GA=${1:-2}; GB=${2:-3}
mkdir -p "$R"
cp "$BASE/eval_receiver.json" "$BASE/eval_receiver.jsonl" "$BASE/eval_sender.json" "$BASE/eval_sender.jsonl" "$R/" 2>/dev/null || true
python -m state_bridge train -c "$CFG" models.sender.device=cuda:$GA models.receiver.device=cuda:$GA > "$R/train.log" 2>&1 &
PA=$!
python -m state_bridge train -c "$CFG" run_name=${N}_state_tuning bridge.type=prompt_tuning models.receiver.device=cuda:$GB > "runs/${N}_state_tuning.train.log" 2>&1 &
PB=$!
wait $PB
python -m state_bridge eval -c "$CFG" run_name=${N}_state_tuning --modes bridged models.receiver.device=cuda:$GB > "runs/${N}_state_tuning.eval.log" 2>&1 &
PB=$!
wait $PA
python -m state_bridge eval -c "$CFG" --modes bridged,bridged_shuffled,bridged_ablated models.sender.device=cuda:$GA models.receiver.device=cuda:$GA > "$R/eval.log" 2>&1
wait $PB
python -m state_bridge observe -c "$CFG" eval.batch_size=16 observe.limit=300 models.sender.device=cuda:$GA models.receiver.device=cuda:$GA > "$R/observe.log" 2>&1
python scripts/compare_runs.py --base "$R" --runs "$R" "runs/${N}_state_tuning" --out "$R/comparison.md" > "$R/compare.log" 2>&1
echo ANSWER_TEST_DONE
