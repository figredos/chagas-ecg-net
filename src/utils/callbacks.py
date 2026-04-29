from typing import Literal

import torch


class EarlyStopping:
    """Early Stopper for training

    Args:
        patience (int): Number of epochs before stopping.
        min_delta (float): Minimum threshold of improvement.
        filename (str): Naming convention for model file.
        file_dir (str): Directory in which to save the file.
        mode ("min" | "max"): Either "min" or "max". If "min" will save the models with the lowest metric value, with "max" will save the model with largest metric value.
    """

    def __init__(
        self,
        patience: int,
        min_delta: float,
        filename: str,
        file_dir: str,
        mode: Literal["min", "max"] = "min",
        include_acc: bool = False,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.filename = filename
        self.file_dir = file_dir
        self.mode = mode
        self.include_acc = include_acc
        self.metric = None
        self.counter = 0

    def __call__(
        self,
        model: torch.nn.Module,
        test_loss: float,
        test_acc: float,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> bool:
        """Updates metric history.

        When called updates the best metric value and saves current state, or increases counter until early stopping.

        If self.mode is "min" analyses `test_loss`, otherwise analyses `test_acc`.

        Args:
            model (torch.nn.Module): Model that will have its state saved.
            test_loss (float): Value for loss of model at a given time.
            test_acc (float): Value for accuracy of model at a given time.
            optimizer (torch.optim.Optimizer): Optimizer that will have its sate saved.
            epoch (int): Current epoch of training.

        Returns:
            `bool`: A boolean of whether the model has surpassed the maximum number of allowed epochs.
        """
        metric = test_loss if self.mode == "min" else test_acc

        if self.metric is None:
            self.metric = metric
            self.save_checkpoint(model, test_loss, test_acc, optimizer, epoch)
        elif self._is_better(metric):
            self.metric = metric
            self.counter = 0
            self.save_checkpoint(model, test_loss, test_acc, optimizer, epoch)

        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True

        return False

    def _is_better(self, metric):
        assert self.metric is not None

        if self.mode == "min":
            return metric < self.metric - self.min_delta
        else:
            return metric > self.metric + self.min_delta

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        test_loss: float,
        test_acc: float,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> None:
        """Save model checkpoint.

        Args:
            model (torch.nn.Module): Model that will have its state saved.
            test_loss (float): Value for loss of model at a given time.
            test_acc (float): Value for accuracy of model at a given time.
            optimizer (torch.optim.Optimizer): Optimizer that will have its sate saved.
            epoch (int): Current epoch of training.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': test_loss,
            'test_acc': test_acc,
        }

        if self.include_acc:
            checkpoint_path = f"{self.file_dir}/{self.filename}_{test_acc}.pth"
        else:
            checkpoint_path = f"{self.file_dir}/{self.filename}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(
            f"""
######################################      
Saving model at Epoch:       |   {epoch}   |        
######################################      
Test Loss:                   | {test_loss:.3f} |

Test Accuracy:               | {test_acc:.3f} |
######################################      
              """
        )
