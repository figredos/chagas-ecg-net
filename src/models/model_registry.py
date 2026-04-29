from typing import Any

import os
import json
import torch

from src.models.base import ECGClassifier
from src.models.cnn_bert import CNNBertClassifier
from src.models.swin_transformer import SwinTransformer
from src.models.pn_encoder import PreNormEncoderClassifier


def instantiate_model_with_params(
    model_name: str, constructor_kwargs: dict[str, Any]
) -> ECGClassifier:
    model_map = {
        "pn_encoder": PreNormEncoderClassifier,
        "cnn_bert": CNNBertClassifier,
        "swin_transformer": SwinTransformer,
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model name: '{model_name}'")

    return model_map[model_name](**constructor_kwargs)


def load_model_from_registry(
    root_path: str,
    model_name: str,
    model_version: str = "latest",
    device: str = "cpu",
) -> ECGClassifier:
    registry_base = os.path.join(
        root_path,
        "model_registry",
        model_name,
    )

    if model_version == "latest":
        versions = sorted(
            [d for d in os.listdir(registry_base) if d.startswith("v")],
            key=lambda v: int(v[1:]),
        )
        if not versions:
            raise FileNotFoundError(
                f"No versions found for model '{model_name}' in '{registry_base}'."
            )
        model_version = versions[-1]

    model_registry_path = os.path.join(
        registry_base,
        model_version,
    )
    metadata_path = os.path.join(
        model_registry_path,
        "metadata.json",
    )
    state_dict_path = os.path.join(
        model_registry_path,
        "model.pth",
    )

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    constructor_kwargs = metadata["constructor_kwargs"].copy()
    constructor_kwargs["device"] = device

    model = instantiate_model_with_params(
        model_name=metadata["model_name"],
        constructor_kwargs=constructor_kwargs,
    )

    state_dict = torch.load(state_dict_path, map_location=device)

    model.load_state_dict(state_dict)

    model.eval()

    return model
