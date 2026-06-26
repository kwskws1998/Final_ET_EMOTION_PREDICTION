#!/usr/bin/env bash
set -euo pipefail

LEGACY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$LEGACY_DIR/../.." && pwd)"

export IITB_CSV="${IITB_CSV:-emotion_et_prediction/data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv}"
export OUTPUT_DIR="${OUTPUT_DIR:-emotion_et_prediction/runs/repro_cmcl_to_iitb_augmented_roberta}"

bash "$LEGACY_DIR/train_repro_cmcl_to_iitb.sh"
