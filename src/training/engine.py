"""
Contains functions for training and testing a PyTorch model.
"""

from typing import Any

import torch
import mlflow

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from sklearn.metrics import classification_report, accuracy_score

from tqdm import tqdm as traditional_tqdm
from tqdm.notebook import tqdm as notebook_tqdm

from src.utils.callbacks import EarlyStopping


def compute_and_extract_metrics(
    preds: torch.Tensor | list[float] | list[torch.Tensor],
    labels: torch.Tensor | list[float] | list[torch.Tensor],
) -> tuple[float, float, float, float]:
    report: dict[str, Any] = classification_report(
        y_pred=preds, y_true=labels, output_dict=True, zero_division=0
    )  # type: ignore[assignment]

    return (
        float(report["accuracy"]),
        float(report["macro avg"]["precision"]),
        float(report["macro avg"]["recall"]),
        float(report["macro avg"]["f1-score"]),
    )


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    notebook: bool = True,
) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
    """
    Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode then runs through all of the required training steps
    (forward pass, loss calculation, optimizer step).

    Args:
        model (torch.nn.Module): A PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be trained on.
        loss_fn (torch.nn.Module): A PyTorch loss function to minimize.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
        device (str): A target device's name to compute on.

    Returns:
        `Tuple[float,float]`: Tuple of training loss and training accuracy metrics.
    """
    tqdm = notebook_tqdm if notebook else traditional_tqdm

    model.train()

    train_loss = 0
    train_preds, train_labels = [], []

    for _, (X, y) in tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Train Step"
    ):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        train_preds.append(y_pred_class)
        train_labels.append(y)

    train_loss = train_loss / len(dataloader)

    train_preds = torch.cat(train_preds)
    train_labels = torch.cat(train_labels)

    return (train_loss, (train_preds, train_labels))


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: str | torch.device,
    notebook: bool = True,
) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
    """
    Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to eval mode then runs through forward pass on test dataset.

    Args:
        model (torch.nn.Module): A PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be tested on.
        loss_fn (torch.nn.Module): A PyTorch loss function to calculate loss on test data.
        device (str): A target device's name to compute on.

    Returns:
        `tuple[float,float]`: tuple of testing loss and testing accuracy metrics.
    """

    tqdm = notebook_tqdm if notebook else traditional_tqdm

    model.eval()

    test_loss = 0
    test_preds, test_labels = [], []

    with torch.inference_mode():
        for _, (X, y) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Test Step"
        ):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)

            test_preds.append(test_pred_labels)
            test_labels.append(y)

    test_loss = test_loss / len(dataloader)

    test_preds = torch.cat(test_preds)
    test_labels = torch.cat(test_labels)

    return (test_loss, (test_preds, test_labels))


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau | CosineAnnealingLR,
    loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
    run_id: str | None = None,
    early_stopping: EarlyStopping | None = None,
    epochs: int = 5,
    device: str | torch.device = "cpu",
    notebook: bool = True,
) -> dict[str, list[float]]:
    """
    Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode then runs through all of the required training steps
    (forward pass, loss calculation, optimizer step).

    Args:
        model (torch.nn.Module): A PyTorch model to be trained.
        train_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be trained on.
        test_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be tested on.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
        loss_fn (torch.nn.Module): A PyTorch loss function to minimize.
        writer (torch.utils.tensorboard.writer.SummaryWriter | None): Instance of Summary to track experiments.
        epochs (int): Number of epochs to train on.
        device (str): A target device's name to compute on.

    Returns:
        `Dict[str,List[float]]`: Dictionary of training and testing loss, as well as training and testing
        accuracy metrics. Each metric has a value in a list for each epoch.

    """
    tqdm = notebook_tqdm if notebook else traditional_tqdm

    results = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
    }

    for epoch in tqdm(range(epochs), total=epochs, desc="Epochs"):

        train_loss, (train_preds, train_labels) = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            notebook=notebook,
        )

        test_loss, (test_preds, test_labels) = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            notebook=notebook,
        )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(test_loss)
        else:
            scheduler.step()

        train_acc = accuracy_score(y_pred=train_preds.cpu(), y_true=train_labels.cpu())

        test_acc, test_precision, test_recall, test_f1 = compute_and_extract_metrics(
            test_preds.cpu(), test_labels.cpu()
        )

        print(f"""
======================================      
Epoch:                       |   {epoch + 1}   |  
======================================      
Train Loss:                  | {train_loss:.3f} |

Train Accuracy:              | {train_acc:.3f} |            
--------------------------------------
Test Loss:                   | {test_loss:.3f} |

Test Accuracy:               | {test_acc:.3f} |
              """)

        results["train_loss"].append(
            train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
        )
        results["test_loss"].append(
            test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss
        )

        results["train_acc"].append(train_acc)

        results["test_acc"].append(test_acc)
        results["test_precision"].append(test_precision)
        results["test_recall"].append(test_recall)
        results["test_f1"].append(test_f1)

        test_metrics = {
            "accuracy": test_acc,
            "f1": test_f1,
            "precision": test_precision,
            "recall": test_recall,
        }

        if run_id is not None:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_acc": float(train_acc),
                    "test_acc": test_acc,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "test_f1": test_f1,
                },
                step=epoch,
            )

        if early_stopping is not None:
            if early_stopping(
                model=model,
                test_loss=test_loss,
                test_metrics=test_metrics,
                optimizer=optimizer,
                epoch=epoch + 1,
            ):
                print("""
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      Early stopping triggered!
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    """)
                break

    return results


def get_predictions_and_targets(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: str | torch.device = "cpu",
    notebook: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get predictions and targets tensors.

    Args:
        model (torch.nn.Module): The model to extract predictions.
        test_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be tested on.
        device (str): A target device's name to compute on.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: tuple of prediction tensor and targets tensor.
    """
    tqdm = notebook_tqdm if notebook else traditional_tqdm

    model.eval()
    predictions = []
    targets = []

    with torch.inference_mode():
        for _, (X, y) in tqdm(
            enumerate(test_dataloader),
            total=len(test_dataloader),
            desc="Getting Predictions",
        ):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)
            test_pred_labels = test_pred_logits.argmax(dim=1)

            predictions.append(test_pred_labels.cpu())
            targets.append(y.cpu())

    return torch.cat(predictions, dim=0), torch.cat(targets, dim=0)
