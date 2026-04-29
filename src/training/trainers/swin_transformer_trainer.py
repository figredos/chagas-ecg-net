import os

import mlflow
from mlflow.pytorch import log_model

from omegaconf import DictConfig

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer as Optimizer

from sklearn.model_selection import train_test_split

from src.data.datasets import STRawECGDataset
from src.utils.callbacks import EarlyStopping
from src.training.trainers.base_trainer import BaseTrainer
from src.models.swin_transformer import SwinTransformer

from src.training.engine import train


class SwinTransformerTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.train_dataset, self.test_dataset = self._build_datasets(
            self.cfg.swin_transformer.data.use_pre_split
        )
        self.train_loader, self.test_loader = self._build_dataloaders()

    def _build_model(self) -> SwinTransformer:
        return SwinTransformer(
            hidden_dim=self.cfg.swin_transformer.model.hidden_dim,
            layers=self.cfg.swin_transformer.model.layers,
            heads=self.cfg.swin_transformer.model.heads,
            channels=self.cfg.swin_transformer.model.channels,
            num_classes=self.cfg.swin_transformer.model.num_classes,
            head_dim=self.cfg.swin_transformer.model.head_dim,
            window_size=self.cfg.swin_transformer.model.window_size,
            downscaling_factors=self.cfg.swin_transformer.model.downscaling_factors,
            relative_pos_embedding=self.cfg.swin_transformer.model.relative_pos_embedding,
            class_names=self.cfg.swin_transformer.data.class_names,
        )

    def _build_loss_fn(self) -> nn.CrossEntropyLoss:
        return nn.CrossEntropyLoss()

    def _build_optimizer(self) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.cfg.swin_transformer.training.lr,
            weight_decay=self.cfg.swin_transformer.training.weight_decay,
            betas=self.cfg.swin_transformer.training.betas,
            eps=self.cfg.swin_transformer.training.eps,
        )

    def _build_scheduler(self) -> ReduceLROnPlateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.cfg.swin_transformer.training.scheduler_factor,
            patience=self.cfg.swin_transformer.training.scheduler_patience,
        )

    def _build_early_stopping(self) -> EarlyStopping:
        return EarlyStopping(
            patience=self.cfg.swin_transformer.training.es_patience,
            min_delta=self.cfg.swin_transformer.training.min_delta,
            filename=self.cfg.swin_transformer.paths.save_filename,
            file_dir=self.cfg.swin_transformer.paths.checkpoints_dir,
            mode="min",
        )

    def _build_datasets(self, use_pre_split: bool) -> tuple[Dataset, Dataset]:
        if use_pre_split:
            train_dataset = torch.load(
                self.cfg.swin_transformer.data.split_train_dataset_path
            )
            train_data = train_dataset["data"].to(torch.float32)
            train_labels = train_dataset["labels"]

            test_dataset = torch.load(
                self.cfg.swin_transformer.data.split_test_dataset_path
            )
            test_data = test_dataset["data"].to(torch.float32)
            test_labels = test_dataset["labels"]

        else:
            dataset = torch.load(self.cfg.swin_transformer.data.complete_dataset_path)
            data = dataset["data"].to(torch.float32)
            labels = dataset["labels"]

            train_data, test_data, train_labels, test_labels = train_test_split(
                data,
                labels,
                train_size=self.cfg.swin_transformer.data.train_size,
                random_state=self.cfg.seed,
                stratify=labels,
            )
            self._save_split_tensors(train_data, test_data, train_labels, test_labels)

        self.data_shape = train_data.shape

        self._train_labels = train_labels

        train_dataset = STRawECGDataset(train_data, train_labels)
        test_dataset = STRawECGDataset(test_data, test_labels)

        return train_dataset, test_dataset

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        class_counts = torch.bincount(self._train_labels)
        class_weights_sampling = 1.0 / class_counts.float()
        sample_weights = class_weights_sampling[self._train_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,  # type: ignore
            num_samples=len(self._train_labels),
            replacement=True,
        )

        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.swin_transformer.model.batch_size,
            sampler=sampler,
        )

        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.swin_transformer.model.batch_size,
            shuffle=False,
        )

        return train_dataloader, test_dataloader

    def _build_experiment_tracking(
        self, tracking_uri: str | None = None, experiment_name: str | None = None
    ) -> None:
        mlflow.set_tracking_uri(
            tracking_uri or self.cfg.swin_transformer.mlflow.tracking_uri
        )
        mlflow.set_experiment(
            experiment_name or self.cfg.swin_transformer.mlflow.experiment_name
        )

    def fit(self) -> dict:
        os.makedirs(self.cfg.swin_transformer.paths.checkpoints_dir, exist_ok=True)
        self._save_config()

        with mlflow.start_run(
            run_name=self.cfg.swin_transformer.mlflow.run_name
        ) as run:
            mlflow.log_params(
                {
                    "batch_size": self.cfg.swin_transformer.model.batch_size,
                    "hidden_dim": self.cfg.swin_transformer.model.hidden_dim,
                    "window_size": self.cfg.swin_transformer.model.window_size,
                    "layers": self.cfg.swin_transformer.model.layers,
                    "heads": self.cfg.swin_transformer.model.heads,
                    "downscaling_factors": self.cfg.swin_transformer.model.downscaling_factors,
                    "num_classes": self.cfg.swin_transformer.model.num_classes,
                    "head_dim": self.cfg.swin_transformer.model.head_dim,
                    "relative_pos_embedding": self.cfg.swin_transformer.model.relative_pos_embedding,
                    "epochs": self.cfg.swin_transformer.training.epochs,
                    "lr": self.cfg.swin_transformer.training.lr,
                    "weight_decay": self.cfg.swin_transformer.training.weight_decay,
                    "es_patience": self.cfg.swin_transformer.training.es_patience,
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
                epochs=self.cfg.swin_transformer.training.epochs,
                device=self.cfg.device,
                notebook=self.cfg.swin_transformer.notebook,
            )

            log_model(
                self.model,
                name="model",
            )
            mlflow.log_artifact(
                os.path.join(
                    self.cfg.swin_transformer.paths.checkpoints_dir,
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
            file_dir or self.cfg.swin_transformer.paths.checkpoints_dir,
        )

    def _save_config(self, path: str | None = None) -> None:
        return super()._save_config(
            path or self.cfg.swin_transformer.paths.checkpoints_dir
        )
