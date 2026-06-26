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
LR="${LR:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SEED="${SEED:-42}"
LOSS="${LOSS:-huber}"
PRETRAIN_VALID_RATIO="${PRETRAIN_VALID_RATIO:-0.10}"
VALID_RATIO="${VALID_RATIO:-0.15}"
PAD_MODE="${PAD_MODE:-dataset}"
PRETRAIN_CSV="${PRETRAIN_CSV:-data/final_training/final_pretrain_trt_scaled.csv}"
PRETRAIN_VALID_CSV="${PRETRAIN_VALID_CSV:-}"
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

pretrain_valid_args=(--pretrain-valid-ratio "$PRETRAIN_VALID_RATIO")
if [[ -n "$PRETRAIN_VALID_CSV" ]]; then
  if [[ ! -f "$PRETRAIN_VALID_CSV" ]]; then
    printf '[error] missing PRETRAIN_VALID_CSV: %s\n' "$PRETRAIN_VALID_CSV" >&2
    exit 1
  fi
  pretrain_valid_args=(--pretrain-valid-csv "$PRETRAIN_VALID_CSV")
fi

python -m emotion_et.train_trt \
  --backend hf \
  --model-name "$MODEL_NAME" \
  --pretrain-csv "$PRETRAIN_CSV" \
  "${pretrain_valid_args[@]}" \
  --finetune-csv "$FINETUNE_CSV" \
  --output-dir "$OUT_DIR" \
  --pretrain-epochs "$PRETRAIN_EPOCHS" \
  --finetune-epochs "$FINETUNE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --max-length "$MAX_LENGTH" \
  --device "$DEVICE" \
  --loss "$LOSS" \
  --seed "$SEED" \
  --valid-ratio "$VALID_RATIO" \
  --pad-mode "$PAD_MODE"
