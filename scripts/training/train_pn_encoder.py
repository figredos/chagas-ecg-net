import os
import json
from datetime import datetime

import torch
import hydra
from omegaconf import DictConfig

from src.training.trainers.pn_encoder_trainer import PreNormEncoderTrainer
from scripts.utils.convert_checkpoint import convert_checkpoint


@hydra.main(
    config_path="../../configs", config_name="train_pn_encoder", version_base=None
)
def main(cfg: DictConfig) -> None:
    print(f"Device: {cfg.device} | CUDA available: {torch.cuda.is_available()}")

    trainer = PreNormEncoderTrainer(cfg)

    history = trainer.fit()

    filename = os.path.join(
        cfg.pn_encoder.paths.checkpoints_dir,
        f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        json.dump(history, f)

    constructor_kwargs = {
        "num_leads": cfg.pn_encoder.model.num_leads,
        "seq_len": trainer.seq_len,
        "embed_dim": cfg.pn_encoder.model.embed_dim,
        "num_layers": cfg.pn_encoder.model.num_layers,
        "num_heads": cfg.pn_encoder.model.num_heads,
        "num_classes": cfg.pn_encoder.model.num_classes,
        "dropout": cfg.pn_encoder.model.dropout,
        "ff_multiplier": cfg.pn_encoder.model.ff_multiplier,
        "class_names": cfg.pn_encoder.data.class_names,
        "device": cfg.device,
    }
    dataset_kwargs = {
        "wavelet": cfg.pn_encoder.data.wavelet,
        "level": cfg.pn_encoder.data.level,
        "window_augment": cfg.pn_encoder.data.window_augment,
        "window_size": cfg.pn_encoder.data.window_size,
        "stride": cfg.pn_encoder.data.stride,
        "filter": cfg.pn_encoder.data.filter,
    }

    convert_checkpoint(
        checkpoint_path="./checkpoints/pn_encoder/pn_encoder_artifact.pth",
        output_dir_path="./model_registry",
        model_name="pn_encoder",
        task=cfg.task,
        version=cfg.pn_encoder.mlflow.version,
        class_names=cfg.pn_encoder.data.class_names,
        input_shape=trainer.data_shape,
        constructor_kwargs=constructor_kwargs,
        dataset_kwargs=dataset_kwargs,
    )

    print("Pre-Norm Encoder was trained successfully.")


if __name__ == "__main__":
    main()
