# Final Emotion ET Prediction

This directory is the cleaned final workspace for emotion-specific TRT
prediction. It keeps the previous 5-feature ET predictor code, but the main path
for the current study is a TRT-only RoBERTa-style token regressor.

Run commands from this directory unless noted otherwise.

## Data Roles

- General TRT pretrain:
  - Provo: `data/pretrain_data/provo.csv`
  - ZuCo 2.0 official extraction:
    `data/pretrain_data/zuco2_official_trt.csv`
- Emotion-domain TRT fine-tune:
  - ZuCo 1.0 Task 1 Sentiment: expected processed CSV at
    `data/finetune_data/zuco1_sentiment_official_trt.csv`
  - IITB SA-I/SA-II sentiment gaze:
    `data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv`

The model target contract is:

```text
sentence_id,word_id,word,nFix,FFD,GPT,TRT,fixProp
```

The TRT-only training path reads the same contract but supervises only the `TRT`
column. IITB V2 raw fixation durations are event-level milliseconds, so the
converter first aggregates participant-word features, then scales the resulting
features to the CMCL/ZuCo target statistics used by the CMCL Provo preprocessing
notebook. Words with no fixation from any participant are kept as all-zero rows
by default.

See `DATA_PLAN.md` for the ZuCo 1.0/2.0 split.

## Download Official ZuCo Sources

Small task-material files:

```bash
python scripts/download_zuco_osf.py --output-dir data/zuco_raw
```

Small corrected ET and wordbounds `.mat` files, excluding large EEG/result
files:

```bash
python scripts/download_zuco_osf.py --output-dir data/zuco_raw --include-et-mat
```

Large subject `.mat` files:

```bash
python scripts/download_zuco_osf.py --output-dir data/zuco_raw --include-matlab
```

The current checked-in data already contains Provo and IITB processed CSVs. The
two expected final ZuCo processed files must come from official ZuCo files:

```text
data/pretrain_data/zuco2_official_trt.csv
data/finetune_data/zuco1_sentiment_official_trt.csv
```

Build both from official ZuCo result MAT files:

```bash
bash scripts/build_official_zuco_trt.sh
```

This writes:

```text
data/pretrain_data/zuco2_official_trt.csv
data/finetune_data/zuco1_sentiment_official_trt.csv
artifacts/zuco1_sentiment_official_trt_report.json
artifacts/zuco2_official_trt_report.json
```

The legacy `data/pretrain_data/train_and_valid.csv` file is an already processed
ZuCo aggregate. Do not use it together with official ZuCo-derived CSVs in the
same training run, because that duplicates ZuCo examples.

If the official result MAT files are outside this git tree, pass explicit globs:

```bash
ZUCO1_RESULTS_GLOB="/path/to/zuco1/task1- SR/Preprocessed/*/results*_SR*.mat" \
ZUCO2_NR_RESULTS_GLOB="/path/to/zuco2/task1 - NR/Preprocessed/*/results*_NR*.mat" \
ZUCO2_TSR_RESULTS_GLOB="/path/to/zuco2/task2 - TSR/Preprocessed/*/results*_TSR*.mat" \
bash scripts/build_official_zuco_trt.sh
```

Those result files are intentionally not tracked because the official files are
large. The extractor scales features to the CMCL/Provo target statistics by
default.

Run a status check:

```bash
python scripts/check_final_data.py
```

Build the final training CSVs after official ZuCo and IITB preprocessing:

```bash
python scripts/build_final_trt_training_files.py
```

This writes:

```text
data/final_training/final_pretrain_trt_scaled.csv
data/final_training/final_finetune_trt_scaled.csv
artifacts/final_trt_training_files_report.json
```

The final training files remap sentence ids across sources and keep source
metadata. They do not rescale values again; inputs are already on the
CMCL/Provo-style feature scale.

After official ZuCo preprocessing finishes, compare official ZuCo 1.0 + ZuCo
2.0 against the old aggregate `train_and_valid.csv`:

```bash
python scripts/compare_official_zuco_to_legacy.py
```

This reports row count, sentence count, and normalized sentence-text overlap.

## Convert IITB V2

If the CSVs are not already present, download CMCL pretraining CSVs into the
workspace:

```bash
bash scripts/legacy_5feature/download_cmcl_data.sh
```

```bash
python -m emotion_et.preprocess_iitb \
  --fixation-csv data/iitb_sentiment_gaze_raw/extracted/v2/Eye-tracking_and_SA-II_released_dataset/Fixation_sequence.csv \
  --text-csv data/iitb_sentiment_gaze_raw/extracted/v2/Eye-tracking_and_SA-II_released_dataset/text_and_annorations.csv \
  --output-csv data/finetune_data/iitb_v2_cmcl_scaled.csv \
  --raw-output-csv data/finetune_data/iitb_v2_raw_word_features.csv \
  --stats-json data/finetune_data/iitb_v2_preprocess_stats.json
```

## Convert IITB/CFILT SA-I

Keep the CFILT raw archive outside tracked files unless you have permission to
redistribute it. The converter reads `Eye-Tracking-Sentiment-Analysis.tar.gz`,
keeps binary sentiment snippets by default, preserves all-zero rows, scales to
CMCL nonzero target statistics, and can write a duplicate-filtered SA-I+SA-II
fine-tune CSV.

```bash
python -m emotion_et.preprocess_iitb_sa1 \
  --sa1-archive /path/to/Eye-Tracking-Sentiment-Analysis.tar.gz \
  --output-csv data/finetune_data/iitb_sa1_snippet_cmcl_scaled.csv \
  --raw-output-csv data/finetune_data/iitb_sa1_snippet_raw_word_features.csv \
  --stats-json data/finetune_data/iitb_sa1_snippet_preprocess_stats.json \
  --combined-with-csv data/finetune_data/iitb_v2_cmcl_scaled.csv \
  --combined-output-csv data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv \
  --combined-stats-json data/finetune_data/iitb_sa1_sa2_preprocess_stats.json
```

## Train TRT-Only Model

Smoke test:

```bash
bash scripts/run_trt_smoke.sh
```

This is a real RoBERTa smoke run using `emotion_et.train_trt` and a
`hidden_size -> 1` TRT head. For a no-download structural check, use:

```bash
bash scripts/run_trt_tiny_smoke.sh
```

Full TRT-only run:

```bash
DEVICE=cuda \
PRETRAIN_EPOCHS=100 \
FINETUNE_EPOCHS=150 \
BATCH_SIZE=16 \
LR=2e-5 \
PRETRAIN_VALID_RATIO=0.10 \
VALID_RATIO=0.15 \
bash scripts/run_trt_full.sh
```

The full script trains `emotion_et.train_trt`, not the legacy 5-feature
`emotion_et.train_et` path. Its RoBERTa head is `hidden_size -> 1`, and the only
supervised target is `TRT`. It uses:

```text
data/final_training/final_pretrain_trt_scaled.csv
data/final_training/final_finetune_trt_scaled.csv
```

It does not fall back to `data/pretrain_data/train_and_valid.csv`.
By default, the pretraining file is split by sentence into a 90/10
train/validation split. That pretrain validation metric is diagnostic only; the
saved `checkpoint_best.pt` is still selected by fine-tune validation TRT MAE.
Use `PRETRAIN_VALID_RATIO=0` to disable pretrain validation, or
`PRETRAIN_VALID_CSV=/path/to/pretrain_valid.csv` for an explicit pretrain
validation file.

## Legacy ZuCo Aggregate

`data/pretrain_data/train_and_valid.csv` is kept only as a legacy reference from
the older CMCL-style pipeline. `scripts/legacy_5feature/build_zuco_cmcl_splits.py`
can still split that aggregate by text matching, but those outputs are not
official ZuCo preprocessing outputs and are not used by `scripts/run_trt_full.sh`.

Previously generated text-match outputs are stored only under:

```text
data/legacy_reference/zuco2_cmcl_textmatch.csv
data/legacy_reference/zuco1_sentiment_cmcl_textmatch.csv
```

## Legacy 5-Feature Code

The old five-output ET predictor path is preserved for inspection and
reproduction only. It predicts `[nFix, FFD, GPT, TRT, fixProp]` with
`emotion_et.train_et` and `emotion_et.models.HFTokenRegressor`.

Legacy scripts live under:

```text
scripts/legacy_5feature/
```

Direct legacy training command:

```bash
python -m emotion_et.train_et \
  --backend hf \
  --model-name roberta-base \
  --pretrain-csv data/pretrain_data/provo.csv \
  --pretrain-csv data/pretrain_data/train_and_valid.csv \
  --finetune-csv data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv \
  --output-dir runs/cmcl_to_iitb_augmented_roberta \
  --pretrain-epochs 100 \
  --finetune-epochs 150 \
  --batch-size 16 \
  --lr 5e-5 \
  --best-metric all
```

For the notebook-compatible ET Predictor 2 export path, this is the default final training command:

```bash
bash scripts/legacy_5feature/train_repro_cmcl_to_iitb.sh
```

That script uses `data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv` by default.
The older SA-II-only file remains available for ablations by passing
`IITB_CSV=emotion_et_prediction/data/finetune_data/iitb_v2_cmcl_scaled.csv`.

The direct `train_et` command writes `checkpoint_best.pt`, `checkpoint_last.pt`,
and `checkpoint.pt`. `checkpoint.pt` is an alias of the best validation
checkpoint. Use `--best-metric TRT` when TRT MAE should choose the best
checkpoint instead of the mean `all` MAE.

The notebook-compatible script writes `et_predictor2_seed42.safetensors`,
`metrics_best.json`, predictions, and logs. Use this path for the Hugging Face
bundle below.

Use the tiny backend only for local smoke tests when `roberta-base` is not cached.

```bash
bash scripts/legacy_5feature/smoke_run.sh
```

After downloading `roberta-base`, run the real Hugging Face backend smoke:

```bash
hf download roberta-base
bash scripts/legacy_5feature/smoke_run_roberta.sh
```

## Package the Hugging Face Model

After the TRT-only run finishes, package it into a self-contained Hugging Face
upload folder:

```bash
LOCAL_FILES_ONLY=1 \
bash scripts/package_hf_model.sh
```

To upload after packaging, first authenticate with `hf auth login`, then upload
the generated TRT-only package directory:

```bash
hf upload <hf-user>/<repo-name> hf_emotion_trt_roberta . --type model \
  --commit-message "Add emotion TRT predictor"
```

The TRT-only package includes:

```text
.gitattributes
README.md
model.py
config.json
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
special_tokens_map.json
emotion_trt_predictor_seed42.safetensors
metrics_best.json
lr_grid_summary.tsv
manifest.env
```

The exported `model.py` auto-detects the decoder output dimension from the
weight file. For the new TRT-only model, `predict_word_trt(...)` reads the
single output directly. Older 5-feature exports remain loadable for legacy
inspection.

Upload the contents of `OUTPUT_DIR` to the Hugging Face model repo. If using the
zip, unzip it first and upload the files inside it rather than uploading the zip
as a single model artifact.
