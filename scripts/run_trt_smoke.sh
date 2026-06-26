#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m emotion_et.train_trt \
  --backend hf \
  --model-name "${MODEL_NAME:-roberta-base}" \
  --pretrain-csv data/final_training/final_pretrain_trt_scaled.csv \
  --finetune-csv data/final_training/final_finetune_trt_scaled.csv \
  --output-dir "${OUT_DIR:-artifacts/trt_only_roberta_smoke_1out}" \
  --pretrain-epochs 1 \
  --finetune-epochs 1 \
  --batch-size 1 \
  --max-length 64 \
  --max-pretrain-sentences 2 \
  --max-finetune-train-sentences 2 \
  --max-valid-sentences 2 \
  --loss huber \
  --seed 42 \
  --device "${DEVICE:-cpu}"

python scripts/check_final_data.py --root "$ROOT" --output-json artifacts/data_status.json
