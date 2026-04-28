import os

import mlflow
from abc import ABC, abstractmethod

from omegaconf import DictConfig, OmegaConf

import torch

from torch.utils.data import DataLoader

from src.models.base import ECGClassifier
from src.utils.callbacks import EarlyStopping


class BaseTrainer(ABC):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = self._build_model().to(self.device)
        self.loss_fn = self._build_loss_fn()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.early_stopping = self._build_early_stopping()
        self._build_experiment_tracking()

    @abstractmethod
    def _build_model(self) -> ECGClassifier: ...

    def _build_loss_fn(self) -> torch.nn.Module: ...

    @abstractmethod
    def _build_optimizer(self) -> torch.optim.Optimizer: ...

    @abstractmethod
    def _build_scheduler(self) -> torch.optim.lr_scheduler.ReduceLROnPlateau: ...

    @abstractmethod
    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]: ...

    @abstractmethod
    def _build_early_stopping(self) -> EarlyStopping: ...

    @abstractmethod
    def _build_datasets(
        self, use_pre_split: bool
    ) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: ...

    @abstractmethod
    def fit(self) -> dict: ...

    def _build_experiment_tracking(
        self, tracking_uri: str | None = None, experiment_name: str | None = None
    ) -> None:
        if tracking_uri is None or experiment_name is None:
            return

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def _save_split_tensors(
        self,
        train_data: torch.Tensor,
        test_data: torch.Tensor,
        train_labels: torch.Tensor,
        test_labels: torch.Tensor,
        file_dir: str | None = None,
    ) -> None:
        if file_dir is None:
            return

        file_dir = file_dir
        data_dir = f"{file_dir}/data"
        os.makedirs(data_dir, exist_ok=True)

        torch.save(
            {"data": train_data, "labels": train_labels},
            os.path.join(data_dir, "train_data.pt"),
        )
        torch.save(
            {"data": test_data, "labels": test_labels},
            os.path.join(data_dir, "test_data.pt"),
        )

    def _save_config(self, path: str | None = None) -> None:
        if path is None:
            return

        path = os.path.join(path, "config.yaml")
        OmegaConf.save(self.cfg, path)
