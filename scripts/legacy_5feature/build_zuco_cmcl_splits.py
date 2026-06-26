"""Legacy helper: split the old CMCL processed ZuCo aggregate by text match.

The CMCL `train_and_valid.csv` already contains word-level ZuCo ET features in
the same scale as Provo. This script separates the ZuCo 1.0 sentiment sentences
that can be matched to the official ZuCo 1.0 sentiment material, and separates
the ZuCo 2.0 sentences that can be matched to the official ZuCo 2.0 task
materials for pretraining.

This is not the official ZuCo preprocessing path. It is kept only as a legacy
diagnostic/reference helper. Do not use these outputs together with official
ZuCo-derived CSVs in the same training run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = ["nFix", "FFD", "GPT", "TRT", "fixProp"]
OUTPUT_COLUMNS = ["sentence_id", "word_id", "word", *FEATURE_COLUMNS]


def normalize_text(text: object) -> str:
    value = str(text)
    value = value.replace("<EOS>", "")
    value = value.replace("emp11111ty", "empty")
    value = value.replace("``", '"').replace("''", '"').replace("`", "'")
    value = value.replace("�", "")
    value = re.sub(r"[^\w]+", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip()


def sentence_texts(df: pd.DataFrame) -> dict[int, str]:
    texts: dict[int, str] = {}
    for sentence_id, group in df.groupby("sentence_id", sort=False):
        words = group.sort_values("word_id")["word"].astype(str).tolist()
        texts[int(sentence_id)] = " ".join(word.replace("<EOS>", "") for word in words)
    return texts


def read_sentiment_material(path: Path) -> list[dict[str, str]]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    reader = csv.DictReader(lines, delimiter=";")
    required = {"sentence_id", "sentence", "sentiment_label"}
    if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
        raise ValueError(f"{path} must contain columns: {sorted(required)}")
    return list(reader)


def read_zuco2_materials(root: Path) -> dict[str, dict[str, object]]:
    if not root.exists():
        raise FileNotFoundError(root)
    matches: dict[str, dict[str, object]] = {}
    for path in sorted(root.glob("*.csv")):
        if "control_questions" in path.name:
            continue
        for row_index, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if not line.strip() or ";" not in line:
                continue
            parts = next(csv.reader([line], delimiter=";"))
            if len(parts) < 3:
                continue
            text = parts[2]
            normalized = normalize_text(text)
            if not normalized:
                continue
            matches[normalized] = {
                "material_file": str(path),
                "row_index": row_index,
                "paragraph_id": parts[0],
                "sentence_id": parts[1],
                "task": "TSR" if path.name.startswith("tsr_") else "NR",
                "label": parts[3] if len(parts) > 3 else "",
                "source_text": text,
            }
    return matches


def renumber(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    sentence_ids = clean["sentence_id"].drop_duplicates().tolist()
    mapping = {old: index for index, old in enumerate(sentence_ids)}
    clean["source_sentence_id"] = clean["sentence_id"].astype(int)
    clean["sentence_id"] = clean["sentence_id"].map(mapping).astype(int)
    return clean


def summarize(df: pd.DataFrame) -> dict[str, object]:
    payload: dict[str, object] = {
        "rows": int(len(df)),
        "sentences": int(df["sentence_id"].nunique()) if "sentence_id" in df else 0,
    }
    for column in FEATURE_COLUMNS:
        if column in df:
            values = pd.to_numeric(df[column], errors="coerce")
            payload[f"{column}_mean"] = float(values.mean())
            payload[f"{column}_std"] = float(values.std())
    return payload


def build_splits(
    cmcl_zuco_csv: Path,
    sentiment_material_csv: Path,
    zuco2_material_dir: Path,
    output_zuco1_sentiment: Path,
    output_zuco2: Path,
    metadata_json: Path,
) -> dict[str, object]:
    cmcl = pd.read_csv(cmcl_zuco_csv)
    missing = set(OUTPUT_COLUMNS) - set(cmcl.columns)
    if missing:
        raise ValueError(f"Missing CMCL columns in {cmcl_zuco_csv}: {sorted(missing)}")
    cmcl = cmcl[OUTPUT_COLUMNS].copy()

    cmcl_text_by_id = sentence_texts(cmcl)
    normalized_to_sentence_id = {normalize_text(text): sid for sid, text in cmcl_text_by_id.items()}
    sentiment_rows = read_sentiment_material(sentiment_material_csv)
    zuco2_materials = read_zuco2_materials(zuco2_material_dir)

    matched_source_ids: dict[int, dict[str, object]] = {}
    unmatched: list[dict[str, str]] = []
    for row in sentiment_rows:
        normalized = normalize_text(row["sentence"])
        source_id = normalized_to_sentence_id.get(normalized)
        if source_id is None:
            unmatched.append(row)
            continue
        matched_source_ids[source_id] = {
            "zuco1_sentence_id": row["sentence_id"],
            "sentiment_label": row["sentiment_label"],
            "source_text": row["sentence"],
        }

    matched_zuco2_ids: dict[int, dict[str, object]] = {}
    unmatched_zuco2_materials = 0
    for normalized, source in zuco2_materials.items():
        source_id = normalized_to_sentence_id.get(normalized)
        if source_id is None:
            unmatched_zuco2_materials += 1
            continue
        if source_id in matched_source_ids:
            continue
        matched_zuco2_ids[source_id] = source

    sentiment_df = cmcl[cmcl["sentence_id"].isin(matched_source_ids)].copy()
    sentiment_df = renumber(sentiment_df)
    sentiment_df["zuco1_sentence_id"] = sentiment_df["source_sentence_id"].map(
        lambda sid: matched_source_ids[int(sid)]["zuco1_sentence_id"]
    )
    sentiment_df["sentiment_label"] = sentiment_df["source_sentence_id"].map(
        lambda sid: matched_source_ids[int(sid)]["sentiment_label"]
    )

    zuco2_df = cmcl[cmcl["sentence_id"].isin(matched_zuco2_ids)].copy()
    zuco2_df = renumber(zuco2_df)
    zuco2_df["zuco2_task"] = zuco2_df["source_sentence_id"].map(
        lambda sid: matched_zuco2_ids[int(sid)]["task"]
    )
    zuco2_df["zuco2_material_file"] = zuco2_df["source_sentence_id"].map(
        lambda sid: matched_zuco2_ids[int(sid)]["material_file"]
    )

    output_zuco1_sentiment.parent.mkdir(parents=True, exist_ok=True)
    output_zuco2.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    sentiment_df.to_csv(output_zuco1_sentiment, index=False)
    zuco2_df.to_csv(output_zuco2, index=False)

    metadata = {
        "cmcl_zuco_csv": str(cmcl_zuco_csv),
        "sentiment_material_csv": str(sentiment_material_csv),
        "zuco2_material_dir": str(zuco2_material_dir),
        "output_zuco1_sentiment": str(output_zuco1_sentiment),
        "output_zuco2": str(output_zuco2),
        "method": "text match from CMCL processed ZuCo train_and_valid.csv to official ZuCo 1.0 sentiment and ZuCo 2.0 task materials",
        "cmcl_sentences": int(cmcl["sentence_id"].nunique()),
        "cmcl_rows": int(len(cmcl)),
        "official_sentiment_rows": int(len(sentiment_rows)),
        "matched_sentiment_sentences": int(len(matched_source_ids)),
        "unmatched_sentiment_rows": int(len(unmatched)),
        "official_zuco2_material_rows": int(len(zuco2_materials)),
        "matched_zuco2_sentences": int(len(matched_zuco2_ids)),
        "unmatched_zuco2_material_rows": int(unmatched_zuco2_materials),
        "zuco1_sentiment_summary": summarize(sentiment_df),
        "zuco2_summary": summarize(zuco2_df),
        "unmatched_sentiment_examples": unmatched[:25],
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmcl-zuco-csv", type=Path, default=Path("data/pretrain_data/train_and_valid.csv"))
    parser.add_argument(
        "--sentiment-material-csv",
        type=Path,
        default=Path("data/zuco_raw/zuco1/task_materials/sentiment_labels_task1.csv"),
    )
    parser.add_argument("--zuco2-material-dir", type=Path, default=Path("data/zuco_raw/zuco2/task_materials"))
    parser.add_argument(
        "--output-zuco1-sentiment",
        type=Path,
        default=Path("data/finetune_data/zuco1_sentiment_cmcl_textmatch.csv"),
    )
    parser.add_argument(
        "--output-zuco2",
        type=Path,
        default=Path("data/pretrain_data/zuco2_cmcl_textmatch.csv"),
    )
    parser.add_argument("--metadata-json", type=Path, default=Path("artifacts/zuco_cmcl_textmatch_report.json"))
    args = parser.parse_args()

    metadata = build_splits(
        cmcl_zuco_csv=args.cmcl_zuco_csv,
        sentiment_material_csv=args.sentiment_material_csv,
        zuco2_material_dir=args.zuco2_material_dir,
        output_zuco1_sentiment=args.output_zuco1_sentiment,
        output_zuco2=args.output_zuco2,
        metadata_json=args.metadata_json,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
