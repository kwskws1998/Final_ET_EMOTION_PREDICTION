# Final Emotion ET Prediction Data Plan

This project separates general reading-time pretraining from emotion-domain
fine-tuning.

## Training Stages

Pretraining:

- Provo: `data/pretrain_data/provo.csv`
- ZuCo 2.0 official extraction:
  `data/pretrain_data/zuco2_official_trt.csv`

Fine-tuning:

- ZuCo 1.0 Task 1 Sentiment: expected processed CSV at
  `data/finetune_data/zuco1_sentiment_official_trt.csv`
- IITB SA-I/SA-II sentiment gaze: `data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv`

## Official ZuCo Sources

- ZuCo 1.0 OSF node: `q3zws`
- ZuCo 2.0 OSF node: `2urht`

ZuCo 1.0 has `task1- SR`, which is the sentiment-reading task. ZuCo 2.0 has
normal reading and task-specific relation annotation, so it belongs in the
general pretraining stage rather than the emotion fine-tuning stage.

## Current Status

The existing Provo and IITB processed CSVs are included from the previous
project. The ZuCo source downloader is present.

Use:

```bash
python scripts/download_zuco_osf.py --output-dir data/zuco_raw
```

For small corrected ET and wordbounds `.mat` files, run:

```bash
python scripts/download_zuco_osf.py --output-dir data/zuco_raw --include-et-mat
```

For the large subject `.mat` files, run intentionally:

```bash
python scripts/download_zuco_osf.py --output-dir data/zuco_raw --include-matlab
```

## Official ZuCo Preprocessing

The final training path should use official ZuCo-derived CSVs, not the legacy
CMCL aggregate. Build them from official result MAT files:

```bash
bash scripts/build_official_zuco_trt.sh
```

Default outputs:

```text
data/pretrain_data/zuco2_official_trt.csv
data/finetune_data/zuco1_sentiment_official_trt.csv
```

`data/pretrain_data/train_and_valid.csv` is an old processed ZuCo aggregate. It
must not be mixed with official ZuCo1/ZuCo2 extraction outputs in the same
training run, because that duplicates ZuCo examples.

After official preprocessing finishes, run:

```bash
python scripts/compare_official_zuco_to_legacy.py
```

The comparison checks whether official ZuCo 1.0 + ZuCo 2.0 matches the legacy
aggregate in row count, sentence count, and normalized sentence-text overlap.

## Strict Raw Result-MAT Path

When the large official `results*.mat` files are available outside the git tree,
use explicit globs:

```bash
ZUCO1_RESULTS_GLOB="/path/to/zuco1/task1- SR/Preprocessed/*/results*_SR*.mat" \
ZUCO2_NR_RESULTS_GLOB="/path/to/zuco2/task1 - NR/Preprocessed/*/results*_NR*.mat" \
ZUCO2_TSR_RESULTS_GLOB="/path/to/zuco2/task2 - TSR/Preprocessed/*/results*_TSR*.mat" \
bash scripts/build_official_zuco_trt.sh
```

## Legacy Text-Match Split

`scripts/legacy_5feature/build_zuco_cmcl_splits.py` is retained only as a
diagnostic helper for the old aggregate file. It writes `*_cmcl_textmatch.csv`
outputs by default, so they are not confused with official preprocessing
outputs.

Previously generated text-match CSVs are kept under `data/legacy_reference/`.
