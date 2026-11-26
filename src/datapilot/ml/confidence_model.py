"""A small classifier that predicts whether an agent answer is likely to be correct.

It is trained on the per-question logs written by the evaluation (where correctness is
known from the benchmark) and only uses signals that are also available at run time:
retries, error types, result size and query complexity.
"""

import argparse
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from datapilot.errors import EXECUTION_ERRORS

FEATURE_NAMES = (
    "retries",
    "attempts",
    "first_attempt_failed",
    "execution_errors",
    "sanity_flags",
    "distinct_error_types",
    "result_rows_log",
    "result_columns",
    "empty_result",
    "code_length_log",
    "join_count",
    "subquery_count",
    "has_group_by",
    "has_order_by",
    "has_window",
    "has_limit",
    "has_aggregate",
    "plan_steps",
    "tables_in_schema",
    "question_words_log",
)
MIN_TRAINING_ROWS = 30


def extract_features(summary: Mapping[str, Any]) -> dict[str, float]:
    code = str(summary.get("code") or "").lower()
    categories = [str(category) for category in summary.get("error_categories") or []]
    rows = int(summary.get("result_rows") or 0)
    return {
        "retries": float(summary.get("retries") or 0),
        "attempts": float(summary.get("attempts") or 1),
        "first_attempt_failed": float(not summary.get("first_attempt_ok", True)),
        "execution_errors": float(sum(category in EXECUTION_ERRORS for category in categories)),
        "sanity_flags": float(sum(category not in EXECUTION_ERRORS for category in categories)),
        "distinct_error_types": float(len(set(categories))),
        "result_rows_log": math.log1p(rows),
        "result_columns": float(summary.get("result_columns") or 0),
        "empty_result": float(rows == 0),
        "code_length_log": math.log1p(len(code)),
        "join_count": float(len(re.findall(r"\bjoin\b", code))),
        "subquery_count": float(max(0, len(re.findall(r"\bselect\b", code)) - 1)),
        "has_group_by": float("group by" in code),
        "has_order_by": float("order by" in code),
        "has_window": float(re.search(r"\bover\s*\(", code) is not None),
        "has_limit": float(re.search(r"\blimit\b", code) is not None),
        "has_aggregate": float(re.search(r"\b(count|sum|avg|min|max)\s*\(", code) is not None),
        "plan_steps": float(summary.get("plan_steps") or 1),
        "tables_in_schema": float(summary.get("tables_retrieved") or 0),
        "question_words_log": math.log1p(len(str(summary.get("question") or "").split())),
    }


@dataclass
class ConfidenceModel:
    pipeline: Any
    model_name: str
    trained_on: int
    cv_auc: float
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def score(self, summary: Mapping[str, Any]) -> float:
        features = extract_features(summary)
        row = pd.DataFrame(
            [[features[name] for name in self.feature_names]], columns=list(self.feature_names)
        )
        return float(self.pipeline.predict_proba(row)[0, 1])


def load_confidence_model(path: Path) -> ConfidenceModel | None:
    if not path.exists():
        return None
    try:
        model = joblib.load(path)
    except Exception:
        return None
    return model if isinstance(model, ConfidenceModel) else None


def candidate_models(random_state: int) -> dict[str, Any]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05, random_state=random_state
        ),
    }


def train_confidence_model(
    records: Iterable[Mapping[str, Any]], random_state: int = 42
) -> tuple[ConfidenceModel, pd.DataFrame]:
    records = [record for record in records if "correct" in record]
    if len(records) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Need at least {MIN_TRAINING_ROWS} labelled runs to train, got {len(records)}."
        )
    features = pd.DataFrame([extract_features(record) for record in records])[list(FEATURE_NAMES)]
    labels = np.array([int(bool(record["correct"])) for record in records])
    if labels.min() == labels.max():
        raise ValueError("All runs have the same label; the model needs both outcomes.")

    folds = min(5, int(np.bincount(labels).min()))
    splitter = StratifiedKFold(max(folds, 2), shuffle=True, random_state=random_state)
    rows, fitted = [], {}
    for name, model in candidate_models(random_state).items():
        scores = cross_validate(
            model,
            features,
            labels,
            cv=splitter,
            scoring={"roc_auc": "roc_auc", "brier": "neg_brier_score", "accuracy": "accuracy"},
        )
        rows.append(
            {
                "model": name,
                "cv_roc_auc": scores["test_roc_auc"].mean(),
                "cv_roc_auc_std": scores["test_roc_auc"].std(),
                "cv_brier": -scores["test_brier"].mean(),
                "cv_accuracy": scores["test_accuracy"].mean(),
            }
        )
        fitted[name] = model
    results = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False, ignore_index=True)
    best = results.iloc[0]
    pipeline = fitted[best["model"]].fit(features, labels)
    model = ConfidenceModel(pipeline, str(best["model"]), len(records), float(best["cv_roc_auc"]))
    return model, results


def read_run_logs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            records += [json.loads(line) for line in handle if line.strip()]
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the answer-confidence model.")
    parser.add_argument("--runs-dir", type=Path, default=Path("reports/eval_runs"))
    parser.add_argument("--model-path", type=Path, default=Path("models/confidence.joblib"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    # Baseline runs have no agent trace, so only agent variants are used for training.
    paths = sorted(args.runs_dir.glob("agent_*.jsonl"))
    if not paths:
        raise SystemExit(f"No agent run logs in {args.runs_dir}; run datapilot-eval first.")
    records = read_run_logs(paths)
    model, results = train_confidence_model(records)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_path)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.report_dir / "confidence_model.csv", index=False)
    positive_rate = np.mean([bool(record["correct"]) for record in records if "correct" in record])
    lines = [
        "# Answer-confidence model",
        "",
        f"Trained on {model.trained_on} agent runs from {len(paths)} log files "
        f"({positive_rate:.1%} of answers correct).",
        "",
        "| Model | CV ROC-AUC | Std | CV Brier | CV accuracy |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {row.model} | {row.cv_roc_auc:.3f} | {row.cv_roc_auc_std:.3f} | "
        f"{row.cv_brier:.3f} | {row.cv_accuracy:.3f} |"
        for row in results.itertuples(index=False)
    ]
    lines += ["", f"Selected: **{model.model_name}**, saved to `{args.model_path.as_posix()}`."]
    (args.report_dir / "confidence_model.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
