#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m emotion_et.train_trt \
  --backend tiny \
  --pretrain-csv data/final_training/final_pretrain_trt_scaled.csv \
  --finetune-csv data/final_training/final_finetune_trt_scaled.csv \
  --output-dir "${OUT_DIR:-artifacts/trt_only_tiny_smoke_1out}" \
  --pretrain-epochs 1 \
  --finetune-epochs 1 \
  --batch-size 4 \
  --max-length 64 \
  --max-pretrain-sentences 8 \
  --max-finetune-train-sentences 8 \
  --max-valid-sentences 4 \
  --loss huber \
  --seed 42 \
  --device cpu
