#!/usr/bin/env python3
"""Evaluate PromptGuard filters on a labeled dataset."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_nlp_filter():
    from promptguard.nlp_filter import NLPFilter

    return NLPFilter()


def _load_ml_filter():
    from promptguard.ml_filter import MLFilter

    return MLFilter()


def _load_llm_filter():
    from promptguard.llm_filter import LLMFilter

    return LLMFilter()


FILTER_BUILDERS = {
    "nlp_filter": _load_nlp_filter,
    "ml_filter": _load_ml_filter,
    "llm_filter": _load_llm_filter,
}


@dataclass
class EvaluationResult:
    filter_name: str
    evaluated: int
    skipped: int
    tp: int
    tn: int
    fp: int
    fn: int

    @property
    def total(self) -> int:
        return self.evaluated

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        denom = precision + recall
        return 2 * precision * recall / denom if denom else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PromptGuard filters.")
    parser.add_argument(
        "--dataset-path",
        default="datasets/eval/prompt_injection_eval.csv",
        help="CSV file containing 'text' and 'label' columns.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory used to store confusion matrices.",
    )
    parser.add_argument(
        "--filters",
        nargs="+",
        choices=tuple(FILTER_BUILDERS.keys()),
        default=list(FILTER_BUILDERS.keys()),
        help="Subset of filters to evaluate (default: all).",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> List[Tuple[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    rows: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            text = (row.get("text") or "").strip()
            label_raw = row.get("label", "").strip()
            if not text:
                continue
            try:
                label = int(label_raw)
            except ValueError as exc:
                raise ValueError(f"Invalid label on row {idx}: {label_raw}") from exc
            rows.append((text, 1 if label else 0))

    if not rows:
        raise ValueError(f"No usable rows found in {path}")
    return rows


def extract_label(score: object) -> int:
    if isinstance(score, dict) and "label" in score:
        return 1 if int(score["label"]) else 0
    if isinstance(score, (int, float)):
        return 1 if int(score) else 0
    raise ValueError(f"Unrecognized score payload: {score!r}")


def evaluate_filter(
    name: str, filter_obj: object, dataset: Sequence[Tuple[str, int]]
) -> EvaluationResult:
    tp = tn = fp = fn = skipped = 0

    for text, label in dataset:
        try:
            result = filter_obj.safe_eval(text)  # type: ignore[attr-defined]
            predicted = extract_label(result.score)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - keep evaluating despite errors
            skipped += 1
            print(f"[{name}] Skipping prompt due to error: {exc}")
            continue

        if label == 1 and predicted == 1:
            tp += 1
        elif label == 0 and predicted == 0:
            tn += 1
        elif label == 0 and predicted == 1:
            fp += 1
        elif label == 1 and predicted == 0:
            fn += 1

    evaluated = len(dataset) - skipped
    return EvaluationResult(name, evaluated, skipped, tp, tn, fp, fn)


def save_confusion_matrix(result: EvaluationResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / f"{result.filter_name}_confusion_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", "predicted_label", "count"])
        writer.writerow([1, 1, result.tp])
        writer.writerow([1, 0, result.fn])
        writer.writerow([0, 1, result.fp])
        writer.writerow([0, 0, result.tn])
    return matrix_path


def main() -> None:
    args = parse_args()
    dataset = load_dataset(Path(args.dataset_path))
    results_dir = Path(args.results_dir)

    print(f"Loaded {len(dataset)} rows from {args.dataset_path}")

    selected_filters = args.filters or list(FILTER_BUILDERS.keys())

    for name in selected_filters:
        builder = FILTER_BUILDERS[name]
        print(f"\n=== Evaluating {name} ===")
        try:
            filter_obj = builder()
        except Exception as exc:
            print(f"Could not initialize {name}: {exc}")
            continue
        try:
            result = evaluate_filter(name, filter_obj, dataset)
        except Exception as exc:
            print(f"Failed to evaluate {name}: {exc}")
            continue

        if result.total == 0:
            print(f"No successful evaluations for {name} (skipped {result.skipped}).")
            continue

        matrix_path = save_confusion_matrix(result, results_dir)
        print(f"Samples evaluated: {result.total} (skipped {result.skipped})")
        print(f"Accuracy : {result.accuracy:.3f}")
        print(f"Precision: {result.precision:.3f}")
        print(f"Recall   : {result.recall:.3f}")
        print(f"F1-score : {result.f1:.3f}")
        print(f"Confusion matrix saved to {matrix_path}")


if __name__ == "__main__":
    main()
