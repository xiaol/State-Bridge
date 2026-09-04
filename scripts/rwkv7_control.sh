#!/usr/bin/env bash
# RWKV-7 receiver: the state-tuning control (constant initial state, no sender), trained with the
# same targets / steps as the bridge, then evaluated.  Waits for a GPU with enough free memory.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/qwen3p5_9b_to_rwkv7_1p5b.yaml
NAME=qwen3p5_9b_to_rwkv7_1p5b
R=runs/$NAME
while [ ! -f "$R/targets_train.jsonl" ]; do sleep 30; done
G=$(bash scripts/wait_gpu.sh ${1:-11000})
python -m state_bridge train -c "$CFG" run_name=${NAME}_state_tuning bridge.type=prompt_tuning models.receiver.device=cuda:$G > "runs/${NAME}_state_tuning.train.log" 2>&1
G=$(bash scripts/wait_gpu.sh 8000)
python -m state_bridge eval -c "$CFG" run_name=${NAME}_state_tuning --modes bridged models.receiver.device=cuda:$G > "runs/${NAME}_state_tuning.eval.log" 2>&1
echo CONTROL_DONE
