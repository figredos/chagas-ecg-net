from typing import Any

import os
import json
import torch


from datetime import datetime

from omegaconf import ListConfig, DictConfig


def _to_serializable(obj):
    if isinstance(obj, (ListConfig, list, tuple, torch.Size)):
        return [_to_serializable(v) for v in obj]
    elif isinstance(obj, DictConfig):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)


def convert_checkpoint(
    checkpoint_path: str,
    output_dir_path: str,
    model_name: str,
    task: str,
    version: str,
    class_names: list[str],
    input_shape: list[int] | torch.Size,
    constructor_kwargs: dict[str, Any],
) -> None:
    checkpoint = torch.load(checkpoint_path)

    state_dict = checkpoint["model_state_dict"]

    registry_dir = os.path.join(
        output_dir_path,
        model_name,
        version,
    )
    os.makedirs(registry_dir, exist_ok=True)

    torch.save(
        state_dict,
        f=os.path.join(registry_dir, "model.pth"),
    )

    metadata = {
        "model_name": model_name,
        "task": task,
        "class_names": class_names,
        "test_acc": checkpoint["test_acc"],
        "test_loss": checkpoint["test_loss"],
        "train_date": datetime.now().strftime('%Y%m%d'),
        "training_epochs": checkpoint["epoch"],
        "input_shape": input_shape,
        "constructor_kwargs": constructor_kwargs,
    }

    with open(os.path.join(registry_dir, "metadata.json"), "w") as f:
        json.dump(_to_serializable(metadata), f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--class_names", required=True)
    parser.add_argument("--input_shape", required=True)
    parser.add_argument("--constructor_kwargs", required=True)

    args = parser.parse_args()

    convert_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_dir_path=args.output_dir_path,
        model_name=args.model_name,
        task=args.task,
        version=args.version,
        class_names=args.class_names,
        input_shape=args.input_shape,
        constructor_kwargs=args.constructor_kwargs,
    )
