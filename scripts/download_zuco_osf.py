"""Download selected official ZuCo files from OSF.

By default this script downloads only small task-material CSV files and text CSVs.
Pass --include-results-mat to download only official `results*.mat` files for
word-level ZuCo extraction. Pass --include-et-mat to download the small corrected
ET/wordbounds MAT files. Pass --include-matlab when you intentionally want every
MAT file, including large EEG files.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

OSF_NODES = {
    "zuco1": "q3zws",
    "zuco2": "2urht",
}

KEEP_SMALL = {
    "sentiment_labels_task1.csv",
    "sentiment_normal_reading.csv",
    "relations_normal_reading.csv",
    "relations_task_specific.csv",
}

KEEP_READER_FILES = {
    "utils_ZuCo.py",
    "README_ZuCo_DataLoader.pdf",
    "requirements.txt",
    "data_loading_helpers.py",
    "read_matlab_files.py",
}

SKIP_DEFAULT_FOLDERS = {
    "answers",
    "Matlab files",
    "scripts",
}


def read_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def iter_osf_files(
    url: str,
    include_matlab: bool,
    include_et_mat: bool,
    include_results_mat: bool,
    parent_path: str = "",
):
    while url:
        payload = read_json(url)
        for item in payload.get("data", []):
            attrs = item.get("attributes", {})
            kind = attrs.get("kind")
            name = attrs.get("name")
            item_path = f"{parent_path}/{name}" if parent_path else str(name)
            if kind == "folder":
                if (
                    not include_matlab
                    and not include_et_mat
                    and not include_results_mat
                    and name in SKIP_DEFAULT_FOLDERS
                ):
                    continue
                if (
                    not include_matlab
                    and not include_et_mat
                    and not include_results_mat
                    and name
                    and (name.startswith("Z") or name.startswith("Y"))
                    and len(name) == 3
                ):
                    continue
                child = (
                    item.get("relationships", {})
                    .get("files", {})
                    .get("links", {})
                    .get("related", {})
                    .get("href")
                )
                if child:
                    yield from iter_osf_files(
                        child,
                        include_matlab=include_matlab,
                        include_et_mat=include_et_mat,
                        include_results_mat=include_results_mat,
                        parent_path=item_path,
                    )
            elif kind == "file":
                yield item
        url = payload.get("links", {}).get("next")


def should_download(name: str, include_matlab: bool, include_et_mat: bool, include_results_mat: bool) -> bool:
    if name in KEEP_SMALL:
        return True
    if name.endswith(".csv") and ("nr_" in name or "tsr_" in name):
        return True
    if include_et_mat and name in KEEP_READER_FILES:
        return True
    if include_et_mat and (
        name.endswith("_corrected_ET.mat")
        or name.startswith("wordbounds_")
        or name in {"sentencesSR.mat", "sentencesNR.mat", "sentencesTSR.mat"}
    ):
        return True
    if include_results_mat and name.startswith("results") and name.endswith(".mat"):
        return True
    if name.endswith(".mat") and include_matlab:
        return True
    return False


def infer_results_task(name: str) -> str | None:
    match = re.match(r"results[A-Z0-9]+_(?P<task>[A-Z0-9]+)\.mat$", name)
    return match.group("task") if match else None


def should_download_with_task_filter(
    name: str,
    include_matlab: bool,
    include_et_mat: bool,
    include_results_mat: bool,
    result_tasks: set[str] | None,
) -> bool:
    if include_results_mat and name.startswith("results") and name.endswith(".mat"):
        task = infer_results_task(name)
        if result_tasks and task not in result_tasks:
            return False
    return should_download(name, include_matlab, include_et_mat, include_results_mat)


def output_path(root: Path, dataset: str, item: dict) -> Path:
    attrs = item.get("attributes", {})
    path = attrs.get("materialized_path") or f"/{attrs.get('name')}"
    parts = [part for part in path.strip("/").split("/") if part]
    return root / dataset / Path(*parts)


def download_file(url: str, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"skip existing {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"download {url} -> {path}")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/zuco_raw"))
    parser.add_argument("--datasets", nargs="+", choices=sorted(OSF_NODES), default=["zuco1", "zuco2"])
    parser.add_argument("--include-results-mat", action="store_true")
    parser.add_argument("--results-task", nargs="+", default=None, help="Limit results MAT download to task suffixes, e.g. SR NR TSR.")
    parser.add_argument("--include-et-mat", action="store_true")
    parser.add_argument("--include-matlab", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_tasks = {task.upper() for task in args.results_task} if args.results_task else None
    manifest: list[dict[str, object]] = []
    for dataset in args.datasets:
        node = OSF_NODES[dataset]
        api_url = f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/"
        for item in iter_osf_files(
            api_url,
            include_matlab=args.include_matlab,
            include_et_mat=args.include_et_mat,
            include_results_mat=args.include_results_mat,
        ):
            attrs = item.get("attributes", {})
            links = item.get("links", {})
            name = attrs.get("name", "")
            download_url = links.get("download")
            if not download_url or not should_download_with_task_filter(
                name,
                args.include_matlab,
                args.include_et_mat,
                args.include_results_mat,
                result_tasks,
            ):
                continue
            path = output_path(args.output_dir, dataset, item)
            manifest.append(
                {
                    "dataset": dataset,
                    "name": name,
                    "materialized_path": attrs.get("materialized_path"),
                    "size": attrs.get("size"),
                    "download": download_url,
                    "local_path": str(path),
                }
            )
            download_file(download_url, path, overwrite=args.overwrite)
            time.sleep(args.sleep)

    manifest_path = args.output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"downloaded_or_existing": len(manifest), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
