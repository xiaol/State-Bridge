#!/usr/bin/env bash
# Re-run the observability probes with standardised ridge features (and save the recorded
# features) for the runs whose probe tables were computed with the unscaled probe.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'pkill -P $$; exit 130' INT TERM
G=${1:-0}
python -m state_bridge observe -c configs/qwen3p5_9b_to_rwkv7_1p5b_capacity.yaml eval.batch_size=16 models.sender.device=cuda:$G models.receiver.device=cuda:$G > runs/qwen3p5_9b_to_rwkv7_1p5b_capacity/observe.log 2>&1
python -m state_bridge observe -c configs/qwen3p5_9b_to_rwkv7_1p5b.yaml eval.batch_size=16 models.sender.device=cuda:$G models.receiver.device=cuda:$G > runs/qwen3p5_9b_to_rwkv7_1p5b/observe.log 2>&1
python -m state_bridge observe -c configs/qwen3p5_9b_to_qwen3_0p6b.yaml run_name=qwen3p5_9b_to_qwen3_0p6b_kv eval.batch_size=24 models.sender.device=cuda:$G models.receiver.device=cuda:$G > runs/qwen3p5_9b_to_qwen3_0p6b_kv.observe.log 2>&1
echo REPROBE_DONE
