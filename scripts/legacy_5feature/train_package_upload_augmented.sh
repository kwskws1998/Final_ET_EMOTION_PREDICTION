#!/usr/bin/env bash
set -euo pipefail

LEGACY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$LEGACY_DIR/../.." && pwd)"

RUN_DIR="${RUN_DIR:-${OUTPUT_DIR:-emotion_et_prediction/runs/repro_cmcl_to_iitb_augmented_roberta}}"
TARGET_HF_MODEL_REPO="${HF_MODEL_REPO:-}"

HF_MODEL_REPO="" OUTPUT_DIR="$RUN_DIR" bash "$LEGACY_DIR/train_repro_cmcl_to_iitb_augmented.sh"

RUN_DIR="$RUN_DIR" HF_MODEL_REPO="$TARGET_HF_MODEL_REPO" bash "$LEGACY_DIR/package_hf_model.sh"
