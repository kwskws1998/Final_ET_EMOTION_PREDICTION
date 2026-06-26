"""Build final scaled TRT training CSVs from official/processed sources."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["sentence_id", "word_id", "word", "TRT"]
FEATURE_COLUMNS = ["nFix", "FFD", "GPT", "TRT", "fixProp"]


@dataclass(frozen=True)
class SourceSpec:
    stage: str
    dataset_source: str
    path: Path


def summarize_frame(path: Path, df: pd.DataFrame) -> dict[str, object]:
    trt = pd.to_numeric(df["TRT"], errors="coerce")
    return {
        "path": str(path),
        "rows": int(len(df)),
        "sentences": int(df["sentence_id"].nunique()),
        "trt_mean": float(trt.mean()),
        "trt_std": float(trt.std()),
        "trt_na": int(trt.isna().sum()),
    }


def load_source(spec: SourceSpec, sentence_offset: int) -> tuple[pd.DataFrame, dict[str, object], int]:
    if not spec.path.exists():
        raise FileNotFoundError(spec.path)
    df = pd.read_csv(spec.path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{spec.path} missing required columns: {missing}")

    clean = df.copy()
    clean = clean.dropna(subset=REQUIRED_COLUMNS)
    clean["original_sentence_id"] = clean["sentence_id"].astype(int)
    clean["sentence_id"] = clean["sentence_id"].astype(int)
    clean["word_id"] = clean["word_id"].astype(int)
    clean["word"] = clean["word"].astype(str)
    clean["TRT"] = pd.to_numeric(clean["TRT"], errors="coerce")
    clean = clean.dropna(subset=["TRT"])
    clean = clean[clean["word"].str.replace("<EOS>", "", regex=False).str.strip().ne("")]

    for column in FEATURE_COLUMNS:
        if column not in clean.columns:
            clean[column] = pd.NA

    sentence_ids = clean["sentence_id"].drop_duplicates().tolist()
    sentence_map = {old_id: sentence_offset + index for index, old_id in enumerate(sentence_ids)}
    clean["sentence_id"] = clean["sentence_id"].map(sentence_map).astype(int)
    clean["dataset_source"] = spec.dataset_source
    clean["stage"] = spec.stage

    keep_columns = [
        "sentence_id",
        "word_id",
        "word",
        *FEATURE_COLUMNS,
        "dataset_source",
        "stage",
        "original_sentence_id",
    ]
    for optional in ["task", "source_sentence_id"]:
        if optional in clean.columns:
            keep_columns.append(optional)

    clean = clean[keep_columns].sort_values(["sentence_id", "word_id"]).reset_index(drop=True)
    summary = summarize_frame(spec.path, clean)
    summary["dataset_source"] = spec.dataset_source
    summary["stage"] = spec.stage
    return clean, summary, sentence_offset + len(sentence_ids)


def build_stage(specs: list[SourceSpec]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    offset = 0
    for spec in specs:
        frame, summary, offset = load_source(spec, sentence_offset=offset)
        frames.append(frame)
        summaries.append(summary)
    if not frames:
        raise ValueError("No sources were provided.")
    return pd.concat(frames, ignore_index=True), summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provo-csv", type=Path, default=Path("data/pretrain_data/provo.csv"))
    parser.add_argument("--zuco2-csv", type=Path, default=Path("data/pretrain_data/zuco2_official_trt.csv"))
    parser.add_argument(
        "--zuco1-sentiment-csv",
        type=Path,
        default=Path("data/finetune_data/zuco1_sentiment_official_trt.csv"),
    )
    parser.add_argument("--iitb-csv", type=Path, default=Path("data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/final_training"))
    parser.add_argument("--report-json", type=Path, default=Path("artifacts/final_trt_training_files_report.json"))
    args = parser.parse_args()

    pretrain_specs = [
        SourceSpec("pretrain", "provo", args.provo_csv),
        SourceSpec("pretrain", "zuco2_official_nr_tsr", args.zuco2_csv),
    ]
    finetune_specs = [
        SourceSpec("finetune", "zuco1_official_sentiment", args.zuco1_sentiment_csv),
        SourceSpec("finetune", "iitb_sa1_sa2_cmcl_scaled", args.iitb_csv),
    ]

    pretrain_df, pretrain_summaries = build_stage(pretrain_specs)
    finetune_df, finetune_summaries = build_stage(finetune_specs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    pretrain_path = args.output_dir / "final_pretrain_trt_scaled.csv"
    finetune_path = args.output_dir / "final_finetune_trt_scaled.csv"
    pretrain_df.to_csv(pretrain_path, index=False)
    finetune_df.to_csv(finetune_path, index=False)

    report = {
        "pretrain_output": str(pretrain_path),
        "finetune_output": str(finetune_path),
        "pretrain": {
            "rows": int(len(pretrain_df)),
            "sentences": int(pretrain_df["sentence_id"].nunique()),
            "sources": pretrain_summaries,
        },
        "finetune": {
            "rows": int(len(finetune_df)),
            "sentences": int(finetune_df["sentence_id"].nunique()),
            "sources": finetune_summaries,
        },
        "note": "Inputs are already scaled to the CMCL/Provo-style feature scale; this script validates, remaps sentence ids, and concatenates sources without rescaling.",
    }
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
