#!/usr/bin/env bash
# Step 2 of docs/next-steps.md: hand-off-heavy training on the RNN pair.  Every training
# problem has a sender-written solution; 80% of examples hand off after 1-256 sender tokens,
# 20% stay pure prefill.  Then the text-vs-latent hand-off sweep and the k=0 evaluation.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'pkill -P $$; exit 130' INT TERM
CFG=configs/qwen3p5_9b_to_rwkv7_1p5b.yaml
BASE=runs/qwen3p5_9b_to_rwkv7_1p5b
N=qwen3p5_9b_to_rwkv7_1p5b_handoff
R=runs/$N
G=${1:-1}
mkdir -p "$R"
cp "$BASE/eval_receiver.json" "$BASE/eval_receiver.jsonl" "$BASE/eval_sender.json" "$BASE/eval_sender.jsonl" "$R/" 2>/dev/null || true
OV="run_name=$N data.sender_generations=$BASE/gen_sender_train.*.jsonl data.handoff_prob=0.8 data.handoff_max=256 models.sender.device=cuda:$G models.receiver.device=cuda:$G"
python -m state_bridge train   -c "$CFG" $OV > "$R/train.log" 2>&1
python -m state_bridge handoff -c "$CFG" $OV handoff.batch_size=16 > "$R/handoff.log" 2>&1
python -m state_bridge eval    -c "$CFG" $OV --modes bridged,bridged_shuffled eval.batch_size=16 > "$R/eval.log" 2>&1
python scripts/compare_runs.py --base "$R" --runs "$R" runs/qwen3p5_9b_to_rwkv7_1p5b runs/qwen3p5_9b_to_rwkv7_1p5b_state_tuning --out "$R/comparison.md" > "$R/compare.log" 2>&1
echo HANDOFF_HEAVY_DONE
