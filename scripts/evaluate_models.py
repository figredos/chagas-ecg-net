import os
from pathlib import Path
from typing import Any
from argparse import ArgumentParser, Namespace

import time
import json

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report

from src.data.datasets import CD2ECGDataset, RawECGDataset, STRawECGDataset

from src.training.engine import get_predictions_and_targets
from src.models.model_registry import load_model_from_registry


def _string_to_len(
    value: str | int | float, output_len: int, start_with: str = ""
) -> str:
    if not isinstance(value, str):
        value = f"{value:.4f}" if isinstance(value, float) else str(value)

    space_len = output_len - len(value)
    space_len = space_len if space_len else 0

    space_suffix = space_len // 2
    space_prefix = space_suffix + 1 if space_len % 2 else space_suffix

    return start_with + space_prefix * " " + value + space_suffix * " "


def _print_eval_table(
    evaluations: dict[str, Any] | tuple[dict[str, Any]] | list[dict[str, Any]],
    num_classes: int,
    class_labels: list[str] | None = None,
) -> None:
    output_len = 20
    if not class_labels:
        class_labels = [f"{label}" for label in range(num_classes)]

    if len(class_labels) != num_classes:
        raise ValueError()

    if isinstance(evaluations, dict):
        evaluations = [evaluations]

    col_names_list = [
        "Model Name",
        "Inference Time (s)",
        "Accuracy",
        "Class",
        "F1-Score",
        "Recall",
        "Precision",
    ]
    col_names_list = [
        _string_to_len(print_item, output_len, "|") for print_item in col_names_list
    ]

    print(
        "_" * 148,
        "\n",
        *col_names_list,
        "|\n",
        "_" * 148,
        sep="",
        end="\n",
    )

    for evaluation_dict in evaluations:
        assert isinstance(evaluation_dict, dict)
        metric_dict = evaluation_dict["classification_report"]
        model_name = evaluation_dict["model_name"]

        eval_list_print = [
            model_name,
            evaluation_dict["process_time"],
            metric_dict["accuracy"],
        ]
        eval_list_print = [
            _string_to_len(print_item, output_len, "|")
            for print_item in eval_list_print
        ]
        sep_print = _string_to_len("", output_len=output_len, start_with="|")
        print(
            *eval_list_print,
            sep_print * 4,
            "|",
            sep="",
        )

        for pos, label in enumerate(class_labels):
            label_list_print = [
                label,
                metric_dict[str(pos)]["recall"],
                metric_dict[str(pos)]["precision"],
                metric_dict[str(pos)]["f1-score"],
            ]

            label_list_print = [
                _string_to_len(print_item, output_len, "|")
                for print_item in label_list_print
            ]

            print(
                sep_print * 3,
                *label_list_print,
                "|",
                sep="",
            )

        print("_" * 148)


def _evaluate_model(
    model_name: str,
    num_classes: int,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    device: str = "cpu",
) -> dict[str, Any]:
    model, _ = load_model_from_registry(
        registry_root="model_registry",
        model_name=model_name,
        model_version="latest",
        device=device,
    )

    model = model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    start = time.perf_counter()

    print(30 * "=", f"Getting predictions for: {model_name}", 30 * "=")

    predictions, targets = get_predictions_and_targets(
        model=model,
        test_dataloader=loader,
        device=device,
        notebook=False,
    )

    process_time = (time.perf_counter() - start) * 1000

    if num_classes < 3 and model_name == "cnn_bert":
        predictions = torch.where(
            predictions == 2,
            torch.zeros_like(predictions),
            predictions,
        )

    metrics = classification_report(
        y_true=targets.cpu(), y_pred=predictions.cpu(), output_dict=True
    )

    return {
        "model_name": model_name,
        "predictions": predictions.cpu().numpy().tolist(),
        "targets": targets.cpu().numpy().tolist(),
        "process_time": process_time,
        "classification_report": metrics,
    }


def parse_arguments() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        choices=["cnn_bert", "swin_transformer", "pn_encoder"],
        help="Model to evaluate. If omitted, evaluates all models.",
    )
    parser.add_argument(
        "-n",
        "--num_classes",
        help="""
        Number of classes to predict on. 2 or 3 
        (note that the only model for 3-class prediction is \"cnn_bert\").
        """,
        default=2,
        type=int,
    )
    parser.add_argument(
        "-json",
        "--json_output_path",
        help="JSON output path.",
        default="eval_output.json",
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Accelerator device.",
        default="cpu",
    )

    return parser


def evaluate_models(args: Namespace) -> None:

    if args.num_classes == 3 and args.model not in (None, "cnn_bert"):
        raise ValueError("The only model that supports 3-class prediction is cnn_bert")

    if args.num_classes not in (2, 3):
        raise ValueError("num_classes must be 2 or 3")

    # Loading Dataset
    if args.num_classes == 3:
        dataset = torch.load("datasets/complete/3_class_dataset.pt")
        data = dataset["data"].to(torch.float32)
        labels = dataset["labels"]
    else:
        dataset = torch.load("datasets/complete/2_class_dataset.pt")
        data = dataset["data"].to(torch.float32)
        labels = dataset["labels"]

    class_labels = (
        ["non-Chagas", "Chagas"]
        if args.num_classes < 3
        else ["Normal", "Chagas", "Structural"]
    )

    eval_dicts = []
    # Evaluating all models
    if args.model is None and args.num_classes < 3:
        cnn_bert_dataset = RawECGDataset(data, labels)
        st_dataset = STRawECGDataset(data, labels)
        pn_encoder_dataset = CD2ECGDataset(
            data,
            labels,
            wavelet="bior3.3",
            level=1,
            window_augment=True,
            window_size=200,
            stride=100,
            filter=0.5,
        )

        cnn_bert_eval = _evaluate_model(
            model_name="cnn_bert",
            num_classes=args.num_classes,
            dataset=cnn_bert_dataset,
            batch_size=128,
            device=args.device,
        )
        swin_transformer_eval = _evaluate_model(
            model_name="swin_transformer",
            num_classes=args.num_classes,
            dataset=st_dataset,
            batch_size=128,
            device=args.device,
        )
        pn_encoder_eval = _evaluate_model(
            model_name="pn_encoder",
            num_classes=args.num_classes,
            dataset=pn_encoder_dataset,
            batch_size=128,
            device=args.device,
        )
        eval_dicts.extend([cnn_bert_eval, swin_transformer_eval, pn_encoder_eval])
    # Evaluating only CNN-Bert
    elif args.model == "cnn_bert":
        cnn_bert_dataset = RawECGDataset(data, labels)

        cnn_bert_eval = _evaluate_model(
            model_name="cnn_bert",
            num_classes=args.num_classes,
            dataset=cnn_bert_dataset,
            batch_size=128,
            device=args.device,
        )
        eval_dicts.append(cnn_bert_eval)
    # Evaluating only SwinTransformer
    elif args.model == "swin_transformer" and args.num_classes < 3:
        st_dataset = STRawECGDataset(data, labels)

        swin_transformer_eval = _evaluate_model(
            model_name="swin_transformer",
            num_classes=args.num_classes,
            dataset=st_dataset,
            batch_size=128,
            device=args.device,
        )
        eval_dicts.append(swin_transformer_eval)

    # Evaluating only Pre-Norm Encoder
    elif args.model == "pn_encoder" and args.num_classes < 3:
        pn_encoder_dataset = CD2ECGDataset(
            data,
            labels,
            wavelet="bior3.3",
            level=1,
            window_augment=True,
            window_size=200,
            stride=100,
            filter=0.5,
        )
        pn_encoder_eval = _evaluate_model(
            model_name="pn_encoder",
            num_classes=args.num_classes,
            dataset=pn_encoder_dataset,
            batch_size=128,
            device=args.device,
        )

        eval_dicts.append(pn_encoder_eval)

    _print_eval_table(
        eval_dicts,
        num_classes=args.num_classes,
        class_labels=class_labels,
    )

    output_path = Path(args.json_output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(json.dumps(eval_dicts))


def main():
    parser = parse_arguments()
    evaluate_models(parser.parse_args())


if __name__ == "__main__":
    main()
