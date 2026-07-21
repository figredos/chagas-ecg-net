import logging
from typing import Any

import os
import json

from datetime import datetime, timezone


def load_model_eval_json(model_name: str, eval_suffix: str = "eval") -> dict[str, Any]:
    with open(f"outputs/eval_jsons/{model_name}_{eval_suffix}.json", "r") as file:
        eval_data = json.load(file)

    return eval_data[0]


def create_reference_baseline(model_name: str) -> dict[str, Any]:
    model_eval_json = load_model_eval_json(model_name)

    test_error_rate = 1 - model_eval_json["classification_report"]["accuracy"]

    test_sample_count = len(model_eval_json["predictions"])

    return {
        "model_name": model_name,
        "test_error_rate": test_error_rate,
        "test_sample_count": test_sample_count,
    }


def load_feedback(feedback_path: str, window: int) -> list[dict[str, Any]]:
    with open(f"{feedback_path}/feedback.jsonl", "r") as jsonl_file:
        raw_feedback = jsonl_file.readlines()

        if window > len(raw_feedback):
            logging.getLogger(__name__).warning(
                f"""
Window size {window} larger than the {len(raw_feedback)} available feedback lines.
Loading all available lines.""",
            )

        lines = [json.loads(jsonl) for jsonl in raw_feedback]
        return lines[-window:]


def compute_disagreement_rate(feedback: list[dict]) -> dict[str, Any]:
    error_count = 0

    for feedback_line in feedback:
        if feedback_line["predicted_class"] != feedback_line["true_class"]:
            error_count += 1

    sample_count = len(feedback)

    disagreement_rate = error_count / sample_count

    return {
        "disagreement_rate": disagreement_rate,
        "sample_count": sample_count,
        "error_count": error_count,
    }


def check_drift(
    model_name: str,
    feedback_path: str,
    threshold: float,
    feedback_window: int,
    min_samples: int,
) -> dict[str, Any]:
    reference_baseline = create_reference_baseline(model_name)
    feedback = load_feedback(feedback_path, feedback_window)
    disagreement_rate = compute_disagreement_rate(feedback)

    drifted = (
        disagreement_rate["disagreement_rate"]
        > reference_baseline["test_error_rate"] * threshold
    )

    sample_count = len(feedback)
    insufficient_data = sample_count < min_samples

    return {
        "drifted": drifted and not insufficient_data,
        "disagreement_rate": disagreement_rate["disagreement_rate"],
        "reference_error_rate": reference_baseline["test_error_rate"],
        "sample_count": sample_count,
        "insufficient_data": insufficient_data,
    }


def run_drift_check(
    model_name: str,
    feedback_path: str,
    output_path: str,
    threshold: float = 2.0,
    feedback_window: int = 50,
    min_samples: int = 50,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    check = check_drift(
        model_name,
        feedback_path,
        threshold,
        feedback_window,
        min_samples,
    )

    os.makedirs(f"{output_path}/drift/", exist_ok=True)
    with open(f"{output_path}/drift/drift_report_{timestamp}.json", "w") as drift_file:
        drift_file.write(json.dumps(check))

    return check
