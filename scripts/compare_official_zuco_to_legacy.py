"""Compare official ZuCo extraction outputs against the legacy aggregate CSV."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def normalize_text(text: object) -> str:
    value = str(text).replace("<EOS>", "")
    value = value.replace("``", '"').replace("''", '"').replace("`", "'")
    value = re.sub(r"[^\w]+", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip()


def sentence_texts(df: pd.DataFrame, key_columns: list[str] | None = None) -> dict[str, str]:
    required = {"sentence_id", "word_id", "word"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    group_keys = key_columns or ["sentence_id"]
    texts: dict[str, str] = {}
    for sentence_key, group in df.groupby(group_keys, sort=False):
        words = group.sort_values("word_id")["word"].astype(str).tolist()
        texts[str(sentence_key)] = normalize_text(" ".join(words))
    return texts


def summarize(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    df = pd.read_csv(path)
    payload: dict[str, object] = {
        "path": str(path),
        "exists": True,
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    if "sentence_id" in df.columns:
        payload["sentences"] = int(df["sentence_id"].nunique())
    if "TRT" in df.columns:
        payload["trt_mean"] = float(pd.to_numeric(df["TRT"], errors="coerce").mean())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-csv", type=Path, default=Path("data/pretrain_data/train_and_valid.csv"))
    parser.add_argument("--zuco1-csv", type=Path, default=Path("data/finetune_data/zuco1_sentiment_official_trt.csv"))
    parser.add_argument("--zuco2-csv", type=Path, default=Path("data/pretrain_data/zuco2_official_trt.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/official_vs_legacy_zuco_compare.json"))
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    summaries = {
        "legacy": summarize(args.legacy_csv),
        "official_zuco1": summarize(args.zuco1_csv),
        "official_zuco2": summarize(args.zuco2_csv),
    }
    missing = [
        key
        for key, item in summaries.items()
        if not item.get("exists")
    ]
    if missing:
        payload = {"ready": False, "missing": missing, "summaries": summaries}
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        raise SystemExit(1)

    legacy = pd.read_csv(args.legacy_csv)
    zuco1 = pd.read_csv(args.zuco1_csv)
    zuco2 = pd.read_csv(args.zuco2_csv)
    official = pd.concat(
        [zuco1.assign(_official_source="zuco1"), zuco2.assign(_official_source="zuco2")],
        ignore_index=True,
    )

    legacy_texts = set(sentence_texts(legacy).values())
    official_texts = set(sentence_texts(official, ["_official_source", "sentence_id"]).values())
    legacy_only = sorted(legacy_texts - official_texts)
    official_only = sorted(official_texts - legacy_texts)

    payload = {
        "ready": True,
        "summaries": summaries,
        "combined_official": {
            "rows": int(len(official)),
            "sentences": int(zuco1["sentence_id"].nunique() + zuco2["sentence_id"].nunique()),
            "unique_normalized_sentence_texts": int(len(official_texts)),
        },
        "legacy": {
            "unique_normalized_sentence_texts": int(len(legacy_texts)),
        },
        "row_delta_official_minus_legacy": int(len(official) - len(legacy)),
        "sentence_delta_official_minus_legacy": int(
            zuco1["sentence_id"].nunique() + zuco2["sentence_id"].nunique() - legacy["sentence_id"].nunique()
        ),
        "normalized_text_overlap": {
            "intersection": int(len(legacy_texts & official_texts)),
            "legacy_only": int(len(legacy_only)),
            "official_only": int(len(official_only)),
            "legacy_only_examples": legacy_only[: args.max_examples],
            "official_only_examples": official_only[: args.max_examples],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
