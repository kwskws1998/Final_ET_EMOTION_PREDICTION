"""Stage-wise TRT-only training for emotion-specific ET prediction."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .trt_data import (
    IGNORE_VALUE,
    HFTRTDataset,
    SimpleTRTDataset,
    SimpleVocab,
    collate_trt_batch,
    limit_sentences,
    load_and_concat_csvs,
    split_by_sentence,
)
from .trt_models import HFTokenTRTRegressor, TinyTRTRegressor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def label_mask(labels: torch.Tensor) -> torch.Tensor:
    return labels.ne(IGNORE_VALUE)


def token_loss_values(predictions: torch.Tensor, labels: torch.Tensor, loss: str) -> torch.Tensor:
    mask = label_mask(labels)
    if not mask.any():
        return predictions.new_empty((0,))
    if loss == "mse":
        return torch.nn.functional.mse_loss(predictions[mask], labels[mask], reduction="none")
    if loss == "huber":
        return torch.nn.functional.huber_loss(predictions[mask], labels[mask], delta=1.0, reduction="none")
    raise ValueError(f"Unsupported loss: {loss}")


def training_loss(predictions: torch.Tensor, labels: torch.Tensor, loss: str) -> torch.Tensor:
    values = token_loss_values(predictions, labels, loss)
    if values.numel() == 0:
        return predictions.sum() * 0.0
    return values.mean()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_name: str,
) -> dict[str, float]:
    model.eval()
    predictions_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    losses_all: list[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        mask = label_mask(labels)
        if mask.any():
            predictions_all.append(predictions[mask].clamp_min(0.0).detach().cpu().numpy())
            labels_all.append(labels[mask].detach().cpu().numpy())
            losses_all.append(token_loss_values(predictions, labels, loss_name).detach().cpu().numpy())
    if not predictions_all:
        return {"TRT": float("nan"), "loss": float("nan")}
    predictions_np = np.concatenate(predictions_all)
    labels_np = np.concatenate(labels_all)
    losses_np = np.concatenate(losses_all)
    return {
        "TRT": float(np.abs(predictions_np - labels_np).mean()),
        "loss": float(losses_np.mean()),
    }


def make_loader(
    df: pd.DataFrame,
    backend: str,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    pad_mode: str,
    vocab: SimpleVocab | None = None,
    tokenizer=None,
) -> torch.utils.data.DataLoader:
    pad_to_dataset_max = pad_mode == "dataset"
    if backend == "tiny":
        if vocab is None:
            raise ValueError("Tiny backend requires a SimpleVocab.")
        dataset = SimpleTRTDataset(df, vocab=vocab, max_length=max_length, pad_to_dataset_max=pad_to_dataset_max)
        pad_id = vocab.pad_id
    elif backend == "hf":
        if tokenizer is None:
            raise ValueError("HF backend requires a tokenizer.")
        dataset = HFTRTDataset(df, tokenizer=tokenizer, max_length=max_length, pad_to_dataset_max=pad_to_dataset_max)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    if pad_mode == "max-length":
        pad_to_length = max_length
    elif pad_mode == "dataset":
        pad_to_length = dataset.pad_to_length
    else:
        pad_to_length = None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_trt_batch(batch, pad_id=pad_id, pad_to_length=pad_to_length),
    )


def build_backend_objects(
    backend: str,
    model_name: str,
    train_frames: list[pd.DataFrame],
    freeze_encoder: bool,
    local_files_only: bool,
):
    if backend == "tiny":
        vocab = SimpleVocab.build(train_frames)
        model = TinyTRTRegressor(vocab_size=len(vocab.token_to_id), pad_id=vocab.pad_id)
        return model, vocab, None
    if backend == "hf":
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                add_prefix_space=True,
                local_files_only=local_files_only,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = HFTokenTRTRegressor(
            model_name=model_name,
            freeze_encoder=freeze_encoder,
            local_files_only=local_files_only,
        )
        return model, None, tokenizer
    raise ValueError(f"Unsupported backend: {backend}")


def checkpoint_payload(
    model: torch.nn.Module,
    vocab: SimpleVocab | None,
    args: argparse.Namespace,
    stage: str,
    epoch: int,
    valid_mae: dict[str, float],
) -> dict[str, object]:
    serialized_args: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized_args[key] = str(value)
        elif isinstance(value, list):
            serialized_args[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            serialized_args[key] = value
    return {
        "state_dict": model.state_dict(),
        "backend": args.backend,
        "model_name": args.model_name,
        "target_feature": "TRT",
        "vocab": vocab.token_to_id if vocab is not None else None,
        "args": serialized_args,
        "stage": stage,
        "epoch": epoch,
        "valid_mae": valid_mae,
        "selected_metric": "TRT",
        "selected_score": float(valid_mae["TRT"]),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    vocab: SimpleVocab | None,
    args: argparse.Namespace,
    stage: str,
    epoch: int,
    valid_mae: dict[str, float],
) -> None:
    torch.save(checkpoint_payload(model, vocab, args, stage, epoch, valid_mae), path)


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_name: str,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = training_loss(predictions, labels, loss_name)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def run_stage(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    label: str,
    loss_name: str,
    valid_loader: torch.utils.data.DataLoader | None = None,
    on_epoch_end=None,
) -> list[dict[str, object]]:
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
    )
    logs: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_name)
        metrics = evaluate(model, valid_loader, device, loss_name) if valid_loader is not None else None
        row = {"stage": label, "epoch": epoch, "train_loss": train_loss, "valid_mae": metrics}
        logs.append(row)
        if metrics is None:
            print(f"[{label}] epoch {epoch}/{epochs} train_loss={train_loss:.4f}", flush=True)
        else:
            print(
                f"[{label}] epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
                f"valid_loss={metrics['loss']:.4f} valid_trt_mae={metrics['TRT']:.4f}",
                flush=True,
            )
        if on_epoch_end is not None:
            on_epoch_end(row)
    return logs


def frame_summary(df: pd.DataFrame | None) -> dict[str, int] | None:
    if df is None:
        return None
    return {
        "rows": int(len(df)),
        "sentences": int(df["sentence_id"].nunique()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hf", "tiny"], default="hf")
    parser.add_argument("--model-name", type=str, default="roberta-base")
    parser.add_argument("--pretrain-csv", type=Path, action="append", default=[])
    parser.add_argument("--pretrain-valid-csv", type=Path, action="append", default=[])
    parser.add_argument("--finetune-csv", type=Path, action="append", required=True)
    parser.add_argument("--valid-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--pretrain-valid-ratio", type=float, default=0.10)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--pad-mode", choices=["batch", "dataset", "max-length"], default="dataset")
    parser.add_argument("--max-pretrain-sentences", type=int, default=None)
    parser.add_argument("--max-pretrain-valid-sentences", type=int, default=None)
    parser.add_argument("--max-finetune-train-sentences", type=int, default=None)
    parser.add_argument("--max-valid-sentences", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not 0.0 <= args.pretrain_valid_ratio < 1.0:
        raise ValueError("--pretrain-valid-ratio must be in [0, 1).")
    if not 0.0 < args.valid_ratio < 1.0:
        raise ValueError("--valid-ratio must be in (0, 1).")

    pretrain_df = None
    pretrain_valid_df = None
    if args.pretrain_csv:
        pretrain_full_df = load_and_concat_csvs(args.pretrain_csv)
        if args.pretrain_valid_csv:
            pretrain_df = pretrain_full_df
            pretrain_valid_df = load_and_concat_csvs(args.pretrain_valid_csv)
        elif args.pretrain_valid_ratio > 0.0:
            pretrain_df, pretrain_valid_df = split_by_sentence(
                pretrain_full_df,
                valid_ratio=args.pretrain_valid_ratio,
                seed=args.seed,
            )
        else:
            pretrain_df = pretrain_full_df
        pretrain_df = limit_sentences(pretrain_df, args.max_pretrain_sentences)
        if pretrain_valid_df is not None:
            pretrain_valid_df = limit_sentences(pretrain_valid_df, args.max_pretrain_valid_sentences)

    finetune_df = load_and_concat_csvs(args.finetune_csv)
    if args.valid_csv is not None:
        finetune_train_df = finetune_df
        valid_df = limit_sentences(pd.read_csv(args.valid_csv), args.max_valid_sentences)
    else:
        finetune_train_df, valid_df = split_by_sentence(finetune_df, valid_ratio=args.valid_ratio, seed=args.seed)
        finetune_train_df = limit_sentences(finetune_train_df, args.max_finetune_train_sentences)
        valid_df = limit_sentences(valid_df, args.max_valid_sentences)

    backend_frames = [finetune_train_df, valid_df]
    if pretrain_df is not None:
        backend_frames.append(pretrain_df)
    if pretrain_valid_df is not None:
        backend_frames.append(pretrain_valid_df)
    model, vocab, tokenizer = build_backend_objects(
        backend=args.backend,
        model_name=args.model_name,
        train_frames=backend_frames,
        freeze_encoder=args.freeze_encoder,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    print(f"Device: {device}", flush=True)
    print(f"Target: TRT only", flush=True)
    print(f"Loss: {args.loss}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    split_summary = {
        "pretrain_train": frame_summary(pretrain_df),
        "pretrain_valid": frame_summary(pretrain_valid_df),
        "finetune_train": frame_summary(finetune_train_df),
        "finetune_valid": frame_summary(valid_df),
    }
    print(f"Data split: {json.dumps(split_summary, sort_keys=True)}", flush=True)
    (args.output_dir / "data_split_summary.json").write_text(
        json.dumps(split_summary, indent=2),
        encoding="utf-8",
    )

    valid_loader = make_loader(
        valid_df,
        backend=args.backend,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
        pad_mode=args.pad_mode,
        vocab=vocab,
        tokenizer=tokenizer,
    )
    logs: list[dict[str, object]] = []

    if pretrain_df is not None and args.pretrain_epochs > 0:
        pretrain_loader = make_loader(
            pretrain_df,
            backend=args.backend,
            batch_size=args.batch_size,
            max_length=args.max_length,
            shuffle=True,
            pad_mode=args.pad_mode,
            vocab=vocab,
            tokenizer=tokenizer,
        )
        pretrain_valid_loader = None
        if pretrain_valid_df is not None:
            pretrain_valid_loader = make_loader(
                pretrain_valid_df,
                backend=args.backend,
                batch_size=args.batch_size,
                max_length=args.max_length,
                shuffle=False,
                pad_mode=args.pad_mode,
                vocab=vocab,
                tokenizer=tokenizer,
            )
        logs.extend(
            run_stage(
                model=model,
                train_loader=pretrain_loader,
                device=device,
                epochs=args.pretrain_epochs,
                lr=args.lr,
                label="pretrain",
                loss_name=args.loss,
                valid_loader=pretrain_valid_loader,
            )
        )

    finetune_loader = make_loader(
        finetune_train_df,
        backend=args.backend,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        pad_mode=args.pad_mode,
        vocab=vocab,
        tokenizer=tokenizer,
    )
    best_state = {"score": float("inf"), "epoch": None, "metrics": None}

    def save_best_if_needed(row: dict[str, object]) -> None:
        metrics = row["valid_mae"]
        if not isinstance(metrics, dict):
            return
        score = float(metrics["TRT"])
        if np.isfinite(score) and score < float(best_state["score"]):
            epoch = int(row["epoch"])
            best_state.update({"score": score, "epoch": epoch, "metrics": metrics})
            save_checkpoint(args.output_dir / "checkpoint_best.pt", model, vocab, args, "finetune", epoch, metrics)
            (args.output_dir / "metrics_best.json").write_text(
                json.dumps({"stage": "finetune", "epoch": epoch, "valid_mae": metrics}, indent=2),
                encoding="utf-8",
            )
            print(f"[finetune] new best TRT_mae={score:.4f} at epoch {epoch}", flush=True)

    logs.extend(
        run_stage(
            model=model,
            train_loader=finetune_loader,
            device=device,
            epochs=args.finetune_epochs,
            lr=args.lr,
            label="finetune",
            loss_name=args.loss,
            valid_loader=valid_loader,
            on_epoch_end=save_best_if_needed,
        )
    )
    last_metrics = evaluate(model, valid_loader, device, args.loss)
    save_checkpoint(args.output_dir / "checkpoint_last.pt", model, vocab, args, "finetune", args.finetune_epochs, last_metrics)
    (args.output_dir / "metrics_last.json").write_text(
        json.dumps({"stage": "finetune", "epoch": args.finetune_epochs, "valid_mae": last_metrics}, indent=2),
        encoding="utf-8",
    )
    if best_state["metrics"] is None:
        save_checkpoint(args.output_dir / "checkpoint_best.pt", model, vocab, args, "finetune", args.finetune_epochs, last_metrics)
        (args.output_dir / "metrics_best.json").write_text(
            json.dumps({"stage": "finetune", "epoch": args.finetune_epochs, "valid_mae": last_metrics}, indent=2),
            encoding="utf-8",
        )
        best_state.update({"score": float(last_metrics["TRT"]), "epoch": args.finetune_epochs, "metrics": last_metrics})
    shutil.copy2(args.output_dir / "checkpoint_best.pt", args.output_dir / "checkpoint.pt")
    shutil.copy2(args.output_dir / "metrics_best.json", args.output_dir / "metrics.json")
    (args.output_dir / "train_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    print(f"Last TRT MAE: {last_metrics['TRT']:.4f}", flush=True)
    print(f"Best TRT MAE: {float(best_state['score']):.4f} at epoch {best_state['epoch']}", flush=True)


if __name__ == "__main__":
    main()
