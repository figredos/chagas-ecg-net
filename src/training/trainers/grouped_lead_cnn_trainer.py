import os

import mlflow
from mlflow.pytorch import log_model

import matplotlib.pyplot as plt

from omegaconf import DictConfig

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import train_test_split

from src.training.engine import train
from src.data.datasets import RawECGDataset
from src.utils.callbacks import EarlyStopping
from src.training.trainers.base_trainer import BaseTrainer
from src.models.group_lead_cnn import GroupedLeadCNNClassifier


class GroupedLeadCNNTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.train_dataset, self.test_dataset = self._build_datasets(
            self.cfg.grouped_lead_cnn.data.use_pre_split
        )
        self.train_loader, self.test_loader = self._build_dataloaders()

    def _build_model(self) -> GroupedLeadCNNClassifier:
        return GroupedLeadCNNClassifier(
            feature_dim=self.cfg.grouped_lead_cnn.model.feature_dim,
            num_heads=self.cfg.grouped_lead_cnn.model.num_heads,
            num_classes=self.cfg.grouped_lead_cnn.model.num_classes,
            ffn_dropout_p=self.cfg.grouped_lead_cnn.model.ffn_dropout_p,
            clf_dropout_p=self.cfg.grouped_lead_cnn.model.clf_dropout_p,
            class_names=self.cfg.grouped_lead_cnn.data.class_names,
        )

    def _build_loss_fn(self) -> nn.CrossEntropyLoss:
        try:
            train_ds = torch.load(
                self.cfg.grouped_lead_cnn.data.split_train_dataset_path
                if self.cfg.grouped_lead_cnn.data.use_pre_split
                else self.cfg.grouped_lead_cnn.data.complete_dataset_path
            )
            train_labels = train_ds["labels"]
            counts = torch.bincount(train_labels)
            weights = counts.sum().float() / (len(counts) * counts.float())
            weights = weights.clamp(0.8, 1.2)
            weights = weights.to(self.device)
            print(
                f"  Loss weights: {dict(zip(self.cfg.grouped_lead_cnn.data.class_names, weights.tolist()))}"
            )
            return nn.CrossEntropyLoss(weight=weights)
        except Exception as e:
            print(
                f"  Warning: could not compute class weights ({e}), using unweighted loss"
            )
            return nn.CrossEntropyLoss()

    def _build_optimizer(self) -> torch.optim.Adam:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.grouped_lead_cnn.training.lr,
            weight_decay=self.cfg.grouped_lead_cnn.training.weight_decay,
        )

    def _build_scheduler(self) -> CosineAnnealingLR:
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.grouped_lead_cnn.training.epochs,
            eta_min=1e-6,
        )

    def _build_early_stopping(self) -> EarlyStopping:
        return EarlyStopping(
            patience=self.cfg.grouped_lead_cnn.training.es_patience,
            min_delta=self.cfg.grouped_lead_cnn.training.min_delta,
            filename=self.cfg.grouped_lead_cnn.paths.save_filename,
            file_dir=self.cfg.grouped_lead_cnn.paths.checkpoints_dir,
            mode="min",
        )

    def _build_datasets(
        self, use_pre_split: bool
    ) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        if use_pre_split:
            train_dataset = torch.load(
                self.cfg.grouped_lead_cnn.data.split_train_dataset_path
            )
            train_data = train_dataset["data"].to(torch.float32)
            train_labels = train_dataset["labels"]

            test_dataset = torch.load(
                self.cfg.grouped_lead_cnn.data.split_test_dataset_path
            )
            test_data = test_dataset["data"].to(torch.float32)
            test_labels = test_dataset["labels"]

        else:
            dataset = torch.load(self.cfg.grouped_lead_cnn.data.complete_dataset_path)
            data = dataset["data"].to(torch.float32)
            labels = dataset["labels"]

            train_data, test_data, train_labels, test_labels = train_test_split(
                data,
                labels,
                train_size=self.cfg.grouped_lead_cnn.data.train_size,
                random_state=self.cfg.seed,
                stratify=labels,
            )
            self._save_split_tensors(train_data, test_data, train_labels, test_labels)

        self.data_shape = train_data.shape

        train_dataset = RawECGDataset(train_data, train_labels)
        test_dataset = RawECGDataset(test_data, test_labels)

        return train_dataset, test_dataset

    def _build_dataloaders(
        self,
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.grouped_lead_cnn.model.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.grouped_lead_cnn.model.batch_size,
            shuffle=False,
        )

        return train_loader, test_loader

    def _build_experiment_tracking(
        self, tracking_uri: str | None = None, experiment_name: str | None = None
    ) -> None:
        mlflow.set_tracking_uri(
            tracking_uri or self.cfg.grouped_lead_cnn.mlflow.tracking_uri
        )
        mlflow.set_experiment(
            experiment_name or self.cfg.grouped_lead_cnn.mlflow.experiment_name
        )

    def fit(self) -> dict:
        os.makedirs(self.cfg.grouped_lead_cnn.paths.checkpoints_dir, exist_ok=True)
        self._save_config()

        with mlflow.start_run(
            run_name=self.cfg.grouped_lead_cnn.mlflow.run_name
        ) as run:
            mlflow.log_params(
                {
                    "epochs": self.cfg.grouped_lead_cnn.training.epochs,
                    "lr": self.cfg.grouped_lead_cnn.training.lr,
                    "weight_decay": self.cfg.grouped_lead_cnn.training.weight_decay,
                    "batch_size": self.cfg.grouped_lead_cnn.model.batch_size,
                    "feature_dim": self.cfg.grouped_lead_cnn.model.feature_dim,
                    "num_heads": self.cfg.grouped_lead_cnn.model.num_heads,
                    "ffn_dropout_p": self.cfg.grouped_lead_cnn.model.ffn_dropout_p,
                    "clf_dropout_p": self.cfg.grouped_lead_cnn.model.clf_dropout_p,
                }
            )

            history = train(
                model=self.model,
                early_stopping=self.early_stopping,
                train_dataloader=self.train_loader,
                test_dataloader=self.test_loader,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                loss_fn=self.loss_fn,
                run_id=run.info.run_id,
                epochs=self.cfg.grouped_lead_cnn.training.epochs,
                device=self.cfg.device,
                notebook=self.cfg.grouped_lead_cnn.notebook,
            )

            disp = self._build_confusion_matrix(
                best_model_path=self.early_stopping.checkpoint_path,
                class_names=self.cfg.grouped_lead_cnn.data.class_names,
                test_dataloader=self.test_loader,
                notebook=self.cfg.grouped_lead_cnn.notebook,
            )

            mlflow.log_figure(disp.figure_, "confusion_matrix.png")

            plt.close(disp.figure_)

            checkpoint = torch.load(self.early_stopping.checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            log_model(
                self.model,
                name="model",
                registered_model_name="grouped_lead_cnn",
            )
            mlflow.log_artifact(
                os.path.join(
                    self.cfg.grouped_lead_cnn.paths.checkpoints_dir,
                    "data",
                ),
                artifact_path="data_split",
            )
        return history

    def _save_split_tensors(
        self,
        train_data: torch.Tensor,
        test_data: torch.Tensor,
        train_labels: torch.Tensor,
        test_labels: torch.Tensor,
        file_dir: str | None = None,
    ) -> None:
        return super()._save_split_tensors(
            train_data,
            test_data,
            train_labels,
            test_labels,
            file_dir or self.cfg.grouped_lead_cnn.paths.checkpoints_dir,
        )

    def _save_config(self, path: str | None = None) -> None:
        return super()._save_config(
            path or self.cfg.grouped_lead_cnn.paths.checkpoints_dir
        )
