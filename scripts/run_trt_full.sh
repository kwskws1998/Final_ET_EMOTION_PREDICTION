#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-auto}"
MODEL_NAME="${MODEL_NAME:-roberta-base}"
OUT_DIR="${OUT_DIR:-artifacts/trt_only_roberta}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-5e-5}"
PRETRAIN_CSV="${PRETRAIN_CSV:-data/final_training/final_pretrain_trt_scaled.csv}"
FINETUNE_CSV="${FINETUNE_CSV:-data/final_training/final_finetune_trt_scaled.csv}"

missing=()
for path in "$PRETRAIN_CSV" "$FINETUNE_CSV"; do
  if [[ ! -f "$path" ]]; then
    missing+=("$path")
  fi
done

if (( ${#missing[@]} > 0 )); then
  printf '[error] missing required training CSVs:\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  cat >&2 <<'EOF'

Do not substitute data/pretrain_data/train_and_valid.csv here. That file is a
legacy processed ZuCo aggregate and would duplicate official ZuCo-derived rows.

Build official ZuCo CSVs first:

  bash scripts/build_official_zuco_trt.sh
  python scripts/build_final_trt_training_files.py

or pass explicit paths:

  PRETRAIN_CSV=/path/to/final_pretrain_trt_scaled.csv \
  FINETUNE_CSV=/path/to/final_finetune_trt_scaled.csv \
  bash scripts/run_trt_full.sh
EOF
  exit 1
fi

python -m emotion_et.train_trt \
  --backend hf \
  --model-name "$MODEL_NAME" \
  --pretrain-csv "$PRETRAIN_CSV" \
  --finetune-csv "$FINETUNE_CSV" \
  --output-dir "$OUT_DIR" \
  --pretrain-epochs "$PRETRAIN_EPOCHS" \
  --finetune-epochs "$FINETUNE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --max-length 512 \
  --device "$DEVICE" \
  --loss huber \
  --seed 42
