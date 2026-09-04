#!/usr/bin/env bash
# Print the index of the first GPU with at least <min_free_mib> free memory, polling until one
# appears.  usage: wait_gpu.sh <min_free_mib> [comma-separated indices to exclude]
# Indices follow nvidia-smi; run CUDA jobs with CUDA_DEVICE_ORDER=PCI_BUS_ID so cuda:i matches.
NEED=${1:-30000}
EX=",${2:-},"
while true; do
  while IFS=, read -r idx free; do
    idx=${idx// /}; free=${free// /}
    if [[ "$EX" != *",$idx,"* ]] && [ "$free" -ge "$NEED" ]; then echo "$idx"; exit 0; fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  sleep 30
done
