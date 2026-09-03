#!/usr/bin/env bash
# Full replication for one config: train the bridge, train the prompt-tuning control,
# evaluate every mode, then hand-off sweep, geometry, observability and compute report.
# usage: scripts/run_all.sh configs/qwen3p5_9b_to_qwen3_0p6b.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${1:-configs/qwen3p5_9b_to_qwen3_0p6b.yaml}"
NAME=$(python -c "import yaml,sys;print(yaml.safe_load(open('$CFG'))['run_name'])")
python -m state_bridge train -c "$CFG"
python -m state_bridge train -c "$CFG" run_name="${NAME}_prompt_tuning" bridge.type=prompt_tuning
python -m state_bridge eval  -c "$CFG"
python -m state_bridge eval  -c "$CFG" run_name="${NAME}_prompt_tuning" --modes bridged
python -m state_bridge handoff  -c "$CFG"
python -m state_bridge geometry -c "$CFG"
python -m state_bridge observe  -c "$CFG"
python -m state_bridge compute  -c "$CFG"
