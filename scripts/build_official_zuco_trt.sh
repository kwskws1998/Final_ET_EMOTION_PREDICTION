#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ZUCO1_RESULTS_GLOB="${ZUCO1_RESULTS_GLOB:-data/zuco_raw/zuco1/task1- SR/Matlab files/results*_SR.mat}"
ZUCO2_NR_RESULTS_GLOB="${ZUCO2_NR_RESULTS_GLOB:-data/zuco_raw/zuco2/task1 - NR/Matlab files/results*_NR.mat}"
ZUCO2_TSR_RESULTS_GLOB="${ZUCO2_TSR_RESULTS_GLOB:-data/zuco_raw/zuco2/task2 - TSR/Matlab files/results*_TSR.mat}"

ZUCO1_OUT="${ZUCO1_OUT:-data/finetune_data/zuco1_sentiment_official_trt.csv}"
ZUCO2_OUT="${ZUCO2_OUT:-data/pretrain_data/zuco2_official_trt.csv}"

python scripts/extract_zuco_results_mat.py \
  --results-mat "$ZUCO1_RESULTS_GLOB" \
  --output-csv "$ZUCO1_OUT" \
  --raw-output-csv data/finetune_data/zuco1_sentiment_official_trt_raw.csv \
  --stats-json artifacts/zuco1_sentiment_official_trt_report.json

python scripts/extract_zuco_results_mat.py \
  --results-mat "$ZUCO2_NR_RESULTS_GLOB" \
  --results-mat "$ZUCO2_TSR_RESULTS_GLOB" \
  --output-csv "$ZUCO2_OUT" \
  --raw-output-csv data/pretrain_data/zuco2_official_trt_raw.csv \
  --stats-json artifacts/zuco2_official_trt_report.json
