"""Extract CMCL-style word TRT rows from official ZuCo result MAT files.

This strict path expects the large official `results*_*.mat` files that contain
`sentenceData`. Keep those files outside git and pass their paths explicitly.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.io
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotion_et.constants import CMCL_NONZERO_TARGET_STATS, CMCL_TARGET_STATS, FEATURE_NAMES
from emotion_et.preprocess_iitb import scale_features_to_cmcl, summarize_features


def as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value.ravel())
    return [value]


def scalar_feature(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return 0.0
        value = value.item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0


def clean_word(value: object, append_eos: bool = False) -> str:
    text = str(value).strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if append_eos:
        text = f"{text}<EOS>"
    return text


def clean_sentence_text(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def infer_subject_and_task(path: Path) -> tuple[str, str]:
    match = re.search(r"results(?P<subject>[A-Z0-9]+)_(?P<task>[A-Z0-9]+)\.mat$", path.name)
    if match:
        return match.group("subject"), match.group("task")
    return path.stem, "unknown"


def load_sentence_data(path: Path):
    try:
        payload = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as exc:
        raise RuntimeError(
            f"{path} appears to be a MATLAB v7.3/HDF5 file. Convert it or use an "
            "h5py reader before this extractor."
        ) from exc
    if "sentenceData" not in payload:
        raise ValueError(f"{path} does not contain `sentenceData`.")
    return as_list(payload["sentenceData"])


def load_hdf5_string(data_container: h5py.File, reference: object) -> str:
    values = np.array(data_container[reference]).ravel()
    return "".join(chr(int(value)) for value in values)


def scalar_hdf5_feature(data_container: h5py.File, reference: object) -> float:
    values = np.array(data_container[reference])
    if values.size != 1:
        return 0.0
    result = float(values.ravel()[0])
    return result if np.isfinite(result) else 0.0


def extract_subject_rows_hdf5(path: Path, max_sentences: int | None = None) -> pd.DataFrame:
    subject, task = infer_subject_and_task(path)
    rows: list[dict[str, object]] = []
    with h5py.File(path, "r") as data_container:
        sentence_data = data_container["sentenceData"]
        word_data = sentence_data["word"]
        for sentence_index in range(len(word_data)):
            if max_sentences is not None and sentence_index >= max_sentences:
                break
            sentence_text = clean_sentence_text(load_hdf5_string(data_container, sentence_data["content"][sentence_index][0]))
            word_group = data_container[word_data[sentence_index][0]]
            if "content" not in word_group:
                continue
            content = word_group["content"]
            for word_index in range(len(content)):
                token = clean_word(load_hdf5_string(data_container, content[word_index][0]))
                if not token:
                    continue
                rows.append(
                    {
                        "source_file": str(path),
                        "subject_id": subject,
                        "task": task,
                        "source_sentence_id": sentence_index,
                        "sentence_text": sentence_text,
                        "word_id": word_index,
                        "word": token,
                        "nFix": scalar_hdf5_feature(data_container, word_group["nFixations"][word_index][0]),
                        "FFD": scalar_hdf5_feature(data_container, word_group["FFD"][word_index][0]),
                        "GPT": scalar_hdf5_feature(data_container, word_group["GPT"][word_index][0]),
                        "TRT": scalar_hdf5_feature(data_container, word_group["TRT"][word_index][0]),
                    }
                )
    return pd.DataFrame(rows)


def extract_subject_rows(path: Path, max_sentences: int | None = None) -> pd.DataFrame:
    subject, task = infer_subject_and_task(path)
    try:
        sentence_data = load_sentence_data(path)
    except RuntimeError as exc:
        if "MATLAB v7.3" in str(exc):
            return extract_subject_rows_hdf5(path, max_sentences=max_sentences)
        raise
    rows: list[dict[str, object]] = []
    for sentence_index, sentence in enumerate(sentence_data):
        if max_sentences is not None and sentence_index >= max_sentences:
            break
        sentence_text = clean_sentence_text(getattr(sentence, "content", ""))
        words = as_list(getattr(sentence, "word", None))
        for word_index, word in enumerate(words):
            token = clean_word(getattr(word, "content", ""))
            if not token:
                continue
            rows.append(
                {
                    "source_file": str(path),
                    "subject_id": subject,
                    "task": task,
                    "source_sentence_id": sentence_index,
                    "sentence_text": sentence_text,
                    "word_id": word_index,
                    "word": token,
                    "nFix": scalar_feature(getattr(word, "nFixations", None)),
                    "FFD": scalar_feature(getattr(word, "FFD", None)),
                    "GPT": scalar_feature(getattr(word, "GPT", None)),
                    "TRT": scalar_feature(getattr(word, "TRT", None)),
                }
            )
    return pd.DataFrame(rows)


def aggregate_subject_rows(raw_subject_rows: pd.DataFrame, append_eos: bool = True) -> pd.DataFrame:
    if raw_subject_rows.empty:
        raise ValueError("No word rows were extracted from the provided MAT files.")
    grouped = (
        raw_subject_rows.groupby(["task", "sentence_text", "word_id", "word"], sort=True)
        .agg(
            source_sentence_id=("source_sentence_id", "min"),
            nFix=("nFix", "mean"),
            FFD=("FFD", "mean"),
            GPT=("GPT", "mean"),
            TRT=("TRT", "mean"),
            fixProp=("nFix", lambda values: float((values > 0).mean() * 100.0)),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(["task", "source_sentence_id", "sentence_text", "word_id"]).reset_index(drop=True)
    sentence_keys = grouped[["task", "sentence_text"]].drop_duplicates().itertuples(index=False)
    mapping = {(task, sentence_text): index for index, (task, sentence_text) in enumerate(sentence_keys)}
    grouped["sentence_id"] = [
        mapping[(task, sentence_text)]
        for task, sentence_text in zip(grouped["task"], grouped["sentence_text"])
    ]
    if append_eos:
        last_word_ids = grouped.groupby("sentence_id")["word_id"].transform("max")
        mask = grouped["word_id"].eq(last_word_ids)
        grouped.loc[mask, "word"] = grouped.loc[mask, "word"].astype(str) + "<EOS>"
    return grouped[["sentence_id", "word_id", "word", *FEATURE_NAMES, "task", "source_sentence_id"]]


def expand_paths(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            paths.extend(Path(item) for item in expanded)
        else:
            paths.append(Path(pattern))
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing MAT files: {[str(path) for path in missing[:10]]}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-mat", action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--raw-output-csv", type=Path, default=None)
    parser.add_argument("--stats-json", type=Path, default=None)
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--scale-zero-rows", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--no-append-eos", action="store_true")
    args = parser.parse_args()

    paths = expand_paths(args.results_mat)
    if args.max_files is not None:
        paths = paths[: args.max_files]
    subject_frames = [extract_subject_rows(path, max_sentences=args.max_sentences) for path in paths]
    raw_df = aggregate_subject_rows(pd.concat(subject_frames, ignore_index=True), append_eos=not args.no_append_eos)

    if args.raw_output_csv is not None:
        args.raw_output_csv.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(args.raw_output_csv, index=False)

    target_stats = CMCL_NONZERO_TARGET_STATS if not args.scale_zero_rows else CMCL_TARGET_STATS
    output_df = (
        raw_df.copy()
        if args.no_scale
        else scale_features_to_cmcl(
            raw_df,
            target_stats=target_stats,
            preserve_zero_rows=not args.scale_zero_rows,
        )
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)

    summary = {
        "results_mat": [str(path) for path in paths],
        "output_csv": str(args.output_csv),
        "raw_output_csv": str(args.raw_output_csv) if args.raw_output_csv else None,
        "scaled": not args.no_scale,
        "scale_zero_rows": args.scale_zero_rows,
        "rows": int(len(output_df)),
        "sentences": int(output_df["sentence_id"].nunique()),
        "raw_summary": summarize_features(raw_df, output_df, target_stats, not args.scale_zero_rows),
    }
    if args.stats_json is not None:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
