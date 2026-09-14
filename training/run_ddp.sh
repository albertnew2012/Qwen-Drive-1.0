#!/usr/bin/env bash
# Launch multi-GPU training. Consumer 3090s have no NVLink and P2P is disabled,
# so NCCL is told explicitly or its collectives can hang.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src:.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PATH="$PWD/.venv/bin:$PATH"
NPROC=${NPROC:-2}
exec .venv/bin/torchrun --nproc_per_node=$NPROC --master_port=${PORT:-29571} "$@"
