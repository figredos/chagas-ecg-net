import os
import matplotlib.pyplot as plt

import mlflow
from mlflow.pytorch import log_model

from omegaconf import DictConfig

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer as Optimizer

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from src.data.datasets import CD2ECGDataset
from src.utils.callbacks import EarlyStopping
from src.training.trainers.base_trainer import BaseTrainer
from src.models.pn_encoder import PreNormEncoderClassifier

from src.training.engine import get_predictions_and_targets, train


class PreNormEncoderTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.train_loader, self.test_loader = self._build_dataloaders()

    def _build_model(self) -> PreNormEncoderClassifier:
        self.train_dataset, self.test_dataset = self._build_datasets(
            self.cfg.pn_encoder.data.use_pre_split
        )
        self.seq_len = self.train_dataset[0][0].shape[0]

        return PreNormEncoderClassifier(
            num_leads=self.cfg.pn_encoder.model.num_leads,
            seq_len=self.seq_len,
            embed_dim=self.cfg.pn_encoder.model.embed_dim,
            num_layers=self.cfg.pn_encoder.model.num_layers,
            num_heads=self.cfg.pn_encoder.model.num_heads,
            num_classes=self.cfg.pn_encoder.model.num_classes,
            dropout=self.cfg.pn_encoder.model.dropout,
            ff_multiplier=self.cfg.pn_encoder.model.ff_multiplier,
            class_names=self.cfg.pn_encoder.data.class_names,
            device=self.cfg.device,
        )

    def _build_loss_fn(self) -> nn.CrossEntropyLoss:
        return nn.CrossEntropyLoss()

    def _build_optimizer(self) -> torch.optim.Adam:
        return torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.cfg.pn_encoder.training.lr,
        )

    def _build_scheduler(self) -> ReduceLROnPlateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.cfg.pn_encoder.training.scheduler_factor,
            patience=self.cfg.pn_encoder.training.scheduler_patience,
        )

    def _build_early_stopping(self) -> EarlyStopping:
        return EarlyStopping(
            patience=self.cfg.pn_encoder.training.es_patience,
            min_delta=self.cfg.pn_encoder.training.min_delta,
            filename=self.cfg.pn_encoder.paths.save_filename,
            file_dir=self.cfg.pn_encoder.paths.checkpoints_dir,
            mode="min",
        )

    def _build_datasets(self, use_pre_split: bool) -> tuple[Dataset, Dataset]:
        if use_pre_split:
            train_dataset = torch.load(
                self.cfg.pn_encoder.data.split_train_dataset_path
            )
            train_data = train_dataset["data"].to(torch.float32)
            train_labels = train_dataset["labels"]

            test_dataset = torch.load(self.cfg.pn_encoder.data.split_test_dataset_path)
            test_data = test_dataset["data"].to(torch.float32)
            test_labels = test_dataset["labels"]

        else:
            dataset = torch.load(self.cfg.pn_encoder.data.complete_dataset_path)
            data = dataset["data"].to(torch.float32)
            labels = dataset["labels"]

            train_data, test_data, train_labels, test_labels = train_test_split(
                data,
                labels,
                train_size=self.cfg.pn_encoder.data.train_size,
                random_state=self.cfg.seed,
                stratify=labels,
            )
            self._save_split_tensors(train_data, test_data, train_labels, test_labels)

        self.data_shape = train_data.shape

        train_dataset = CD2ECGDataset(
            data=train_data,
            labels=train_labels,
            wavelet=self.cfg.pn_encoder.data.wavelet,
            level=self.cfg.pn_encoder.data.wavelet_level,
            window_augment=self.cfg.pn_encoder.data.window_augment,
            window_size=self.cfg.pn_encoder.data.window_size,
            stride=self.cfg.pn_encoder.data.stride,
            filter=self.cfg.pn_encoder.data.filter,
        )
        test_dataset = CD2ECGDataset(
            data=test_data,
            labels=test_labels,
            wavelet=self.cfg.pn_encoder.data.wavelet,
            level=self.cfg.pn_encoder.data.wavelet_level,
            window_augment=self.cfg.pn_encoder.data.window_augment,
            window_size=self.cfg.pn_encoder.data.window_size,
            stride=self.cfg.pn_encoder.data.stride,
            filter=self.cfg.pn_encoder.data.filter,
        )

        return train_dataset, test_dataset

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.pn_encoder.model.batch_size,
            shuffle=True,
        )
        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.pn_encoder.model.batch_size,
            shuffle=False,
        )

        return train_dataloader, test_dataloader

    def _build_experiment_tracking(
        self, tracking_uri: str | None = None, experiment_name: str | None = None
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri or self.cfg.pn_encoder.mlflow.tracking_uri)
        mlflow.set_experiment(
            experiment_name or self.cfg.pn_encoder.mlflow.experiment_name
        )

    def _build_confusion_matrix(
        self, best_model_path: str, class_names: list[str]
    ) -> ConfusionMatrixDisplay:

        self.model.load_state_dict(torch.load(best_model_path, weights_only=True))

        final_preds, final_labels = get_predictions_and_targets(
            self.model,
            self.test_loader,
            notebook=self.cfg.pn_encoder.notebook,
            device=self.cfg.device,
        )

        disp = ConfusionMatrixDisplay.from_predictions(
            y_true=final_labels.cpu(),
            y_pred=final_preds.cpu(),
            display_labels=class_names,
            normalize='true',
            cmap="Blues",
        )
        disp.figure_.set_size_inches(8, 6)

        return disp

    def fit(self) -> dict:
        os.makedirs(self.cfg.pn_encoder.paths.checkpoints_dir, exist_ok=True)
        self._save_config()

        with mlflow.start_run(run_name=self.cfg.pn_encoder.mlflow.run_name) as run:
            mlflow.log_params(
                {
                    "epochs": self.cfg.pn_encoder.training.epochs,
                    "lr": self.cfg.pn_encoder.training.lr,
                    "batch_size": self.cfg.pn_encoder.model.batch_size,
                    "num_layers": self.cfg.pn_encoder.model.num_layers,
                    "num_classes": self.cfg.pn_encoder.model.num_classes,
                    "dropout": self.cfg.pn_encoder.model.dropout,
                    "ff_multiplier": self.cfg.pn_encoder.model.ff_multiplier,
                    "seq_len": self.seq_len,
                    "wavelet": self.cfg.pn_encoder.data.wavelet,
                    "wavelet_level": self.cfg.pn_encoder.data.wavelet_level,
                    "embed_dim": self.cfg.pn_encoder.model.embed_dim,
                    "num_heads": self.cfg.pn_encoder.model.num_heads,
                    "es_patience": self.cfg.pn_encoder.training.es_patience,
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
                epochs=self.cfg.pn_encoder.training.epochs,
                device=self.cfg.device,
                notebook=self.cfg.pn_encoder.notebook,
            )

            disp = self._build_confusion_matrix(
                best_model_path=self.early_stopping.checkpoint_path,
                class_names=self.cfg.pn_encoder.data.class_names,
            )

            mlflow.log_figure(disp.figure_, "confusion_matrix.png")

            plt.close(disp.figure_)

            log_model(
                self.model,
                name="model",
            )
            mlflow.log_artifact(
                os.path.join(
                    self.cfg.pn_encoder.paths.checkpoints_dir,
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
            file_dir or self.cfg.pn_encoder.paths.checkpoints_dir,
        )

    def _save_config(self, path: str | None = None) -> None:
        return super()._save_config(path or self.cfg.pn_encoder.paths.checkpoints_dir)
