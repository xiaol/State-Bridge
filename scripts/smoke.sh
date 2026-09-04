#!/usr/bin/env bash
# End-to-end offline smoke test on CPU with tiny random models (~1 minute).
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/make_tiny_models.py --out runs/tiny
python -m state_bridge train    -c configs/smoke.yaml
python -m state_bridge eval     -c configs/smoke.yaml
python -m state_bridge handoff  -c configs/smoke.yaml
python -m state_bridge geometry -c configs/smoke.yaml
python -m state_bridge observe  -c configs/smoke.yaml
python -m state_bridge compute  -c configs/smoke.yaml
# RNN receiver (RWKV-7, state injection)
python -m state_bridge train    -c configs/smoke_rwkv7.yaml
python -m state_bridge eval     -c configs/smoke_rwkv7.yaml
python -m state_bridge handoff  -c configs/smoke_rwkv7.yaml
python -m state_bridge geometry -c configs/smoke_rwkv7.yaml
python -m state_bridge observe  -c configs/smoke_rwkv7.yaml
python -m state_bridge compute  -c configs/smoke_rwkv7.yaml
echo "smoke test passed"
