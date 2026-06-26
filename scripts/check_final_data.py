"""Report which data files are ready for the final TRT-only training plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED = {
    "pretrain": [
        "data/pretrain_data/provo.csv",
        "data/pretrain_data/zuco2_official_trt.csv",
    ],
    "finetune": [
        "data/finetune_data/zuco1_sentiment_official_trt.csv",
        "data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv",
    ],
}

LEGACY_REFERENCE = {
    "legacy_reference": [
        "data/pretrain_data/train_and_valid.csv",
        "data/legacy_reference/zuco2_cmcl_textmatch.csv",
        "data/legacy_reference/zuco1_sentiment_cmcl_textmatch.csv",
    ]
}


def summarize_csv(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    df = pd.read_csv(path)
    summary = {
        "path": str(path),
        "exists": True,
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    if "sentence_id" in df.columns:
        summary["sentences"] = int(df["sentence_id"].nunique())
    if "TRT" in df.columns:
        summary["trt_mean"] = float(pd.to_numeric(df["TRT"], errors="coerce").mean())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/data_status.json"))
    args = parser.parse_args()

    payload = {
        split: [summarize_csv(args.root / rel_path) for rel_path in rel_paths]
        for split, rel_paths in REQUIRED.items()
    }
    payload.update(
        {
            split: [summarize_csv(args.root / rel_path) for rel_path in rel_paths]
            for split, rel_paths in LEGACY_REFERENCE.items()
        }
    )
    payload["ready"] = all(
        item.get("exists") is True
        for split_name in REQUIRED
        for item in payload[split_name]
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
