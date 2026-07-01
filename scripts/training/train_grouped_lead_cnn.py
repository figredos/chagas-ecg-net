import os
import json
from datetime import datetime

import torch
import hydra
from omegaconf import DictConfig


from scripts.utils.convert_checkpoint import convert_checkpoint
from src.training.trainers.grouped_lead_cnn_trainer import GroupedLeadCNNTrainer


@hydra.main(
    config_path="../../configs",
    config_name="train_grouped_lead_cnn",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    print(f"Device: {cfg.device} | CUDA available: {torch.cuda.is_available()}")

    trainer = GroupedLeadCNNTrainer(cfg)
    history = trainer.fit()

    filename = os.path.join(
        cfg.grouped_lead_cnn.paths.checkpoints_dir,
        f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        json.dump(history, f)
    constructor_kwargs = {
        "feature_dim": cfg.grouped_lead_cnn.model.feature_dim,
        "num_heads": cfg.grouped_lead_cnn.model.num_heads,
        "num_classes": cfg.grouped_lead_cnn.model.num_classes,
        "ffn_dropout_p": cfg.grouped_lead_cnn.model.ffn_dropout_p,
        "clf_dropout_p": cfg.grouped_lead_cnn.model.clf_dropout_p,
        "class_names": cfg.grouped_lead_cnn.data.class_names,
    }

    convert_checkpoint(
        checkpoint_path="./checkpoints/grouped_lead_cnn/grouped_lead_cnn_artifact.pth",
        output_dir_path="./model_registry",
        model_name="grouped_lead_cnn",
        task=cfg.task,
        version=cfg.grouped_lead_cnn.mlflow.version,
        class_names=cfg.grouped_lead_cnn.data.class_names,
        input_shape=trainer.data_shape,
        constructor_kwargs=constructor_kwargs,
    )

    print("Grouped-Lead CNN Classifier was trained successfully.")


if __name__ == "__main__":
    main()
