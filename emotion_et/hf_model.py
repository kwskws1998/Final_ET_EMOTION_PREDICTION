"""Self-contained Hugging Face inference wrapper for ET predictor exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, RobertaConfig, RobertaModel

FULL_FEATURE_NAMES = ["nFix", "FFD", "GPT", "TRT", "fixProp"]
TRT_FEATURE_NAMES = ["TRT"]
FEATURE_NAMES = FULL_FEATURE_NAMES
TRT_INDEX = 3
DEFAULT_WEIGHT = "emotion_trt_predictor_seed42.safetensors"


def feature_names_for_output_dim(output_dim: int) -> list[str]:
    if output_dim == 1:
        return TRT_FEATURE_NAMES
    if output_dim == len(FULL_FEATURE_NAMES):
        return FULL_FEATURE_NAMES
    return [f"feature_{index}" for index in range(output_dim)]


def infer_output_dim(state: dict[str, torch.Tensor]) -> int:
    decoder_weight = state.get("decoder.weight")
    if decoder_weight is None:
        raise KeyError("Missing decoder.weight in ET predictor weights.")
    return int(decoder_weight.shape[0])


class RobertaRegressionModel(torch.nn.Module):
    """RoBERTa-base encoder with a token regression head."""

    def __init__(self, config_path: str | Path = ".", output_dim: int = 1):
        super().__init__()
        config = RobertaConfig.from_pretrained(config_path)
        self.output_dim = output_dim
        self.feature_names = feature_names_for_output_dim(output_dim)
        self.roberta = RobertaModel(config)
        self.decoder = torch.nn.Linear(config.hidden_size, output_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.decoder(hidden)


def load_et_predictor(
    model_dir: str | Path,
    weight_name: str = DEFAULT_WEIGHT,
    device: str | torch.device | None = None,
) -> tuple[RobertaRegressionModel, AutoTokenizer]:
    """Load the exported ET predictor and tokenizer from a local or downloaded HF repo."""

    model_dir = Path(model_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
    state = load_file(str(model_dir / weight_name), device=str(device))
    model = RobertaRegressionModel(config_path=model_dir, output_dim=infer_output_dim(state)).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict_word_features(
    text: str,
    model: RobertaRegressionModel,
    tokenizer,
    device: str | torch.device | None = None,
    max_length: int = 512,
) -> tuple[list[str], np.ndarray]:
    """Predict word-level ET features by taking the first RoBERTa subword per word."""

    device = torch.device(device or next(model.parameters()).device)
    words = text.strip().split()
    encoded = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    predictions = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = predictions.squeeze(0).clamp_min(0.0).cpu().numpy()
    if predictions.ndim == 1:
        predictions = predictions[:, None]

    word_ids = encoded.word_ids(batch_index=0)
    output = np.zeros((len(words), model.output_dim), dtype=np.float32)
    seen: set[int] = set()
    for token_index, word_index in enumerate(word_ids):
        if word_index is None or word_index in seen or word_index >= len(words):
            continue
        output[word_index] = predictions[token_index]
        seen.add(word_index)
    return words, output


@torch.no_grad()
def predict_word_trt(
    text: str,
    model: RobertaRegressionModel,
    tokenizer,
    device: str | torch.device | None = None,
    max_length: int = 512,
) -> tuple[list[str], np.ndarray]:
    """Predict word-level TRT values only."""

    words, features = predict_word_features(text, model, tokenizer, device=device, max_length=max_length)
    if features.shape[1] == 1:
        return words, features[:, 0]
    return words, features[:, TRT_INDEX]
