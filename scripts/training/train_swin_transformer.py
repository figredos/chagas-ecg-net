import os
import json
from datetime import datetime

import torch
import hydra
from omegaconf import DictConfig

from src.training.trainers.swin_transformer_trainer import SwinTransformerTrainer
from scripts.utils.convert_checkpoint import convert_checkpoint


@hydra.main(
    config_path="../../configs", config_name="train_swin_transformer", version_base=None
)
def main(cfg: DictConfig) -> None:
    print(f"Device: {cfg.device} | CUDA available: {torch.cuda.is_available()}")

    trainer = SwinTransformerTrainer(cfg)

    history = trainer.fit()

    filename = os.path.join(
        cfg.swin_transformer.paths.checkpoints_dir,
        f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        json.dump(history, f)

    constructor_kwargs = {
        "hidden_dim": cfg.swin_transformer.model.hidden_dim,
        "layers": cfg.swin_transformer.model.layers,
        "heads": cfg.swin_transformer.model.heads,
        "channels": cfg.swin_transformer.model.channels,
        "num_classes": cfg.swin_transformer.model.num_classes,
        "head_dim": cfg.swin_transformer.model.head_dim,
        "window_size": cfg.swin_transformer.model.window_size,
        "downscaling_factors": cfg.swin_transformer.model.downscaling_factors,
        "relative_pos_embedding": cfg.swin_transformer.model.relative_pos_embedding,
        "class_names": cfg.swin_transformer.data.class_names,
        "device": cfg.device,
    }

    convert_checkpoint(
        checkpoint_path="./checkpoints/swin_transformer/swin_transformer_artifact.pth",
        output_dir_path="./model_registry",
        model_name="swin_transformer",
        task=cfg.task,
        version=cfg.cnn_bert.mlflow.version,
        class_names=cfg.swin_transformer.data.class_names,
        input_shape=trainer.data_shape,
        constructor_kwargs=constructor_kwargs,
    )

    print("Swin Transformer was trained successfully.")


if __name__ == "__main__":
    main()
