#!/usr/bin/env bash
# Full replication for one config: train the bridge, train the prompt-tuning control,
# evaluate every mode, then hand-off sweep, geometry, observability and compute report.
# usage: scripts/run_all.sh configs/qwen3p5_9b_to_qwen3_0p6b.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${1:-configs/qwen3p5_9b_to_qwen3_0p6b.yaml}"
NAME=$(python -c "import yaml,sys;print(yaml.safe_load(open('$CFG'))['run_name'])")
RUNS=$(python -c "import yaml,sys;print(yaml.safe_load(open('$CFG')).get('runs_dir','runs'))")
T="$RUNS/$NAME/targets_train.jsonl"
# 1. baselines and model-written training targets
python -m state_bridge eval -c "$CFG" --modes receiver,sender
python -m state_bridge precompute -c "$CFG" --role receiver
python -m state_bridge precompute -c "$CFG" --role sender --subset wrong
python -m state_bridge targets -c "$CFG"
# 2. bridge and prompt-tuning control, both frozen models, same targets
python -m state_bridge train -c "$CFG" data.targets="$T"
python -m state_bridge train -c "$CFG" data.targets="$T" run_name="${NAME}_prompt_tuning" bridge.type=prompt_tuning
python -m state_bridge eval  -c "$CFG" --modes bridged,bridged_shuffled,bridged_ablated
python -m state_bridge eval  -c "$CFG" run_name="${NAME}_prompt_tuning" --modes bridged
python scripts/compare_runs.py --base "$RUNS/$NAME" --runs "$RUNS/$NAME" "$RUNS/${NAME}_prompt_tuning" --out "$RUNS/$NAME/comparison.md"
python -m state_bridge handoff  -c "$CFG"
python -m state_bridge geometry -c "$CFG"
python -m state_bridge observe  -c "$CFG"
python -m state_bridge compute  -c "$CFG"
