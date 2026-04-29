import os
import json
from datetime import datetime

import torch
import hydra
from omegaconf import DictConfig

from src.training.trainers.cnn_bert_trainer import CNNBertTrainer
from scripts.utils.convert_checkpoint import convert_checkpoint


@hydra.main(
    config_path="../../configs", config_name="train_cnn_bert", version_base=None
)
def main(cfg: DictConfig) -> None:
    print(f"Device: {cfg.device} | CUDA available: {torch.cuda.is_available()}")

    trainer = CNNBertTrainer(cfg)

    history = trainer.fit()

    filename = os.path.join(
        cfg.cnn_bert.paths.checkpoints_dir,
        f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        json.dump(history, f)

    constructor_kwargs = {
        "d_model": cfg.cnn_bert.model.d_model,
        "num_heads": cfg.cnn_bert.model.num_heads,
        "num_leads": cfg.cnn_bert.model.num_leads,
        "num_classes": cfg.cnn_bert.model.num_classes,
        "num_transformer_layers": cfg.cnn_bert.model.num_transformer_layers,
        "cnn_downsample_factor": cfg.cnn_bert.model.cnn_downsample_factor,
        "device": cfg.device,
        "class_names": cfg.cnn_bert.data.class_names,
    }

    convert_checkpoint(
        checkpoint_path="./checkpoints/cnn_bert/cnn_bert_artifact.pth",
        output_dir_path="./model_registry",
        model_name="cnn_bert",
        task=cfg.task,
        version=cfg.cnn_bert.mlflow.version,
        class_names=cfg.cnn_bert.data.class_names,
        input_shape=trainer.data_shape,
        constructor_kwargs=constructor_kwargs,
    )

    print("CNN BERT was trained successfully.")


if __name__ == "__main__":
    main()
