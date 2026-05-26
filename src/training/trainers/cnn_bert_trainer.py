import os

import mlflow
from mlflow.pytorch import log_model

import matplotlib.pyplot as plt

from omegaconf import DictConfig

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from src.training.engine import train
from src.data.datasets import RawECGDataset
from src.utils.callbacks import EarlyStopping
from src.models.cnn_bert import CNNBertClassifier
from src.training.trainers.base_trainer import BaseTrainer


class CNNBertTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.train_dataset, self.test_dataset = self._build_datasets(
            self.cfg.cnn_bert.data.use_pre_split
        )
        self.train_loader, self.test_loader = self._build_dataloaders()

    def _build_model(self) -> CNNBertClassifier:
        return CNNBertClassifier(
            d_model=self.cfg.cnn_bert.model.d_model,
            num_heads=self.cfg.cnn_bert.model.num_heads,
            num_leads=self.cfg.cnn_bert.model.num_leads,
            num_classes=self.cfg.cnn_bert.model.num_classes,
            num_transformer_layers=self.cfg.cnn_bert.model.num_transformer_layers,
            cnn_downsample_factor=self.cfg.cnn_bert.model.cnn_downsample_factor,
            device=self.cfg.device,
            class_names=self.cfg.cnn_bert.data.class_names,
        )

    def _build_loss_fn(self) -> nn.CrossEntropyLoss:
        return nn.CrossEntropyLoss()

    def _build_optimizer(self) -> torch.optim.Adam:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.cnn_bert.training.lr,
            weight_decay=self.cfg.cnn_bert.training.weight_decay,
        )

    def _build_scheduler(self) -> ReduceLROnPlateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.cfg.cnn_bert.training.scheduler_factor,
            patience=self.cfg.cnn_bert.training.scheduler_patience,
        )

    def _build_early_stopping(self) -> EarlyStopping:
        return EarlyStopping(
            patience=self.cfg.cnn_bert.training.es_patience,
            min_delta=self.cfg.cnn_bert.training.min_delta,
            filename=self.cfg.cnn_bert.paths.save_filename,
            file_dir=self.cfg.cnn_bert.paths.checkpoints_dir,
            mode="min",
        )

    def _build_datasets(
        self, use_pre_split: bool
    ) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        if use_pre_split:
            train_dataset = torch.load(self.cfg.cnn_bert.data.split_train_dataset_path)
            train_data = train_dataset["data"].to(torch.float32)
            train_labels = train_dataset["labels"]

            test_dataset = torch.load(self.cfg.cnn_bert.data.split_test_dataset_path)
            test_data = test_dataset["data"].to(torch.float32)
            test_labels = test_dataset["labels"]

        else:
            dataset = torch.load(self.cfg.cnn_bert.data.complete_dataset_path)
            data = dataset["data"].to(torch.float32)
            labels = dataset["labels"]

            train_data, test_data, train_labels, test_labels = train_test_split(
                data,
                labels,
                train_size=self.cfg.cnn_bert.data.train_size,
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
            batch_size=self.cfg.cnn_bert.model.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.cnn_bert.model.batch_size,
            shuffle=False,
        )

        return train_loader, test_loader

    def _build_experiment_tracking(
        self, tracking_uri: str | None = None, experiment_name: str | None = None
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri or self.cfg.cnn_bert.mlflow.tracking_uri)
        mlflow.set_experiment(
            experiment_name or self.cfg.cnn_bert.mlflow.experiment_name
        )

    def fit(self) -> dict:
        os.makedirs(self.cfg.cnn_bert.paths.checkpoints_dir, exist_ok=True)
        self._save_config()

        with mlflow.start_run(run_name=self.cfg.cnn_bert.mlflow.run_name) as run:
            mlflow.log_params(
                {
                    "epochs": self.cfg.cnn_bert.training.epochs,
                    "lr": self.cfg.cnn_bert.training.lr,
                    "weight_decay": self.cfg.cnn_bert.training.weight_decay,
                    "batch_size": self.cfg.cnn_bert.model.batch_size,
                    "d_model": self.cfg.cnn_bert.model.d_model,
                    "num_heads": self.cfg.cnn_bert.model.num_heads,
                    "num_transformer_layers": self.cfg.cnn_bert.model.num_transformer_layers,
                    "cnn_downsample_factor": self.cfg.cnn_bert.model.cnn_downsample_factor,
                    "es_patience": self.cfg.cnn_bert.training.es_patience,
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
                epochs=self.cfg.cnn_bert.training.epochs,
                device=self.cfg.device,
                notebook=self.cfg.cnn_bert.notebook,
            )

            disp = self._build_confusion_matrix(
                best_model_path=self.early_stopping.checkpoint_path,
                class_names=self.cfg.cnn_bert.data.class_names,
                test_dataloader=self.test_loader,
                notebook=self.cfg.cnn_bert.notebook,
            )

            mlflow.log_figure(disp.figure_, "confusion_matrix.png")

            plt.close(disp.figure_)

            checkpoint = torch.load(self.early_stopping.checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            log_model(
                self.model,
                name="model",
                registered_model_name="cnn_bert",
            )
            mlflow.log_artifact(
                os.path.join(
                    self.cfg.cnn_bert.paths.checkpoints_dir,
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
            file_dir or self.cfg.cnn_bert.paths.checkpoints_dir,
        )

    def _save_config(self, path: str | None = None) -> None:
        return super()._save_config(path or self.cfg.cnn_bert.paths.checkpoints_dir)
