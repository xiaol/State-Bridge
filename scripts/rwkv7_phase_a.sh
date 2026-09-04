#!/usr/bin/env bash
# RWKV-7 receiver, phase A: baselines and model-written training targets.
#   1. receiver-alone GSM8K eval (RWKV-7 G1j 1.5B)
#   2. receiver writes solutions for the train split
#   3. the sender (Qwen3.5-9B) writes solutions for the problems the receiver got wrong that no
#      earlier sender file covers (sender files from the Qwen3-0.6B run are reused: same sender,
#      same prompts)
#   4. build targets_train.jsonl
# The sender-alone eval is copied from the Qwen3-0.6B run: same sender, same test prompts.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/qwen3p5_9b_to_rwkv7_1p5b.yaml
R=runs/qwen3p5_9b_to_rwkv7_1p5b
OLD=runs/qwen3p5_9b_to_qwen3_0p6b
mkdir -p "$R"
cp "$OLD/eval_sender.json" "$OLD/eval_sender.jsonl" "$R/" 2>/dev/null || true
cp "$OLD/gen_sender_train.wrong0.jsonl" "$R/gen_sender_train.qwenwrong0.jsonl" 2>/dev/null || true
cp "$OLD/gen_sender_train.wrong1.jsonl" "$R/gen_sender_train.qwenwrong1.jsonl" 2>/dev/null || true

G=$(bash scripts/wait_gpu.sh 12000)
python -m state_bridge eval -c "$CFG" --modes receiver models.receiver.device=cuda:$G > "$R/eval_receiver.log" 2>&1
G=$(bash scripts/wait_gpu.sh 12000)
python -m state_bridge precompute -c "$CFG" --role receiver --device cuda:$G > "$R/precompute_receiver.log" 2>&1

python - <<EOF
import json, glob
R = "$R"
recv = {json.loads(l)["id"]: json.loads(l) for f in glob.glob(f"{R}/gen_receiver_train.*.jsonl") for l in open(f) if l.strip()}
have = {json.loads(l)["id"] for f in glob.glob(f"{R}/gen_sender_train.*.jsonl") for l in open(f) if l.strip()}
todo = [r for i, r in recv.items() if not r["correct"] and i not in have]
with open(f"{R}/receiver_wrong_uncovered.jsonl", "w") as f:
    for r in todo:
        f.write(json.dumps(r) + "\n")
print("receiver wrong:", sum(not r["correct"] for r in recv.values()), "of", len(recv), "| uncovered by sender files:", len(todo))
EOF

G=$(bash scripts/wait_gpu.sh 30000)
python -m state_bridge precompute -c "$CFG" --role sender --subset "wrong:$R/receiver_wrong_uncovered.jsonl" --tag rwkvwrong --device cuda:$G > "$R/precompute_sender.log" 2>&1
python -m state_bridge targets -c "$CFG" > "$R/targets.log" 2>&1
echo PHASE_A_DONE
