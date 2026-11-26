"""Evaluate the ML Analyst's training pipeline on public tabular datasets.

The target column is fixed per dataset here, so the numbers measure the modelling
pipeline itself (preprocessing, model comparison, cross-validation), not the LLM's
choice of target.
"""

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml

from datapilot.ml.pipelines import MLReport, run_model_comparison, validate_setup


@dataclass(frozen=True)
class Dataset:
    name: str
    source: str
    target: str
    load: Callable[[], pd.DataFrame]


def load_churn() -> pd.DataFrame:
    return pd.read_excel(Path("data/samples/customer_churn.xlsx"))


def load_housing() -> pd.DataFrame:
    return fetch_california_housing(as_frame=True).frame


def load_adult() -> pd.DataFrame:
    return fetch_openml("adult", version=2, as_frame=True).frame


DATASETS = (
    Dataset("Customer churn", "UCI Iranian Churn (3,150 customers)", "churn", load_churn),
    Dataset(
        "California housing",
        "1990 US census via scikit-learn (20,640 districts)",
        "MedHouseVal",
        load_housing,
    ),
    Dataset("Adult income", "UCI Adult via OpenML (48,842 people)", "class", load_adult),
)


def result_rows(dataset: Dataset, report: MLReport, seconds: float) -> list[dict]:
    rows = []
    for score in report.model_scores:
        selected = score.model == report.best_model
        row = {
            "dataset": dataset.name,
            "task": report.task,
            "rows": report.n_rows,
            "features": len(report.features),
            "model": score.model,
            "primary_metric": report.primary_metric,
            "cv_mean": round(score.cv_mean, 4),
            "cv_std": round(score.cv_std, 4),
            "selected": selected,
            "train_seconds": round(seconds, 1),
        }
        if selected:
            row.update(
                {f"test_{key}": round(value, 4) for key, value in report.test_metrics.items()}
            )
        rows.append(row)
    return rows


def importance_chart(dataset: Dataset, report: MLReport, out_dir: Path) -> Path:
    data = report.feature_importance.head(10).iloc[::-1]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.barh(data["feature"], data["importance"], xerr=data["std"], color="#1f4e9e")
    axis.set_title(f"{dataset.name}: permutation importance ({report.best_model})")
    axis.set_xlabel(f"drop in {report.primary_metric} when shuffled")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    path = out_dir / f"ml_importance_{dataset.name.lower().replace(' ', '_')}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def report_markdown(results: pd.DataFrame, reports: dict[str, MLReport], charts: list[Path]) -> str:
    lines = [
        "# ML Analyst evaluation",
        "",
        "Each dataset goes through the same pipeline the ML Analyst uses: validation of "
        "the feature set, imputation + scaling + one-hot encoding, an 80/20 stratified "
        "split, 5-fold cross-validation on the training part for three model families, "
        "then the best model is refit and scored once on the held-out test set.",
        "",
        "| Dataset | Task | Rows | Model | CV score (mean ± std) | Selected |",
        "|---|---|---|---|---|---|",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.task} | {row.rows:,} | {row.model} | "
            f"{row.primary_metric} {row.cv_mean:.3f} ± {row.cv_std:.3f} | "
            f"{'yes' if row.selected else ''} |"
        )
    lines += ["", "## Held-out test set (selected model)", ""]
    for name, report in reports.items():
        metrics = ", ".join(f"{key} {value:.3f}" for key, value in report.test_metrics.items())
        lines.append(f"- **{name}** ({report.best_model}): {metrics}")
        top = ", ".join(report.feature_importance["feature"].head(3))
        lines.append(f"  - strongest features: {top}")
        for warning in report.warnings:
            lines.append(f"  - warning: {warning}")
        if report.dropped:
            dropped = "; ".join(f"{k} ({v})" for k, v in report.dropped.items())
            lines.append(f"  - dropped by validation: {dropped}")
    lines += ["", *[f"![{path.stem}]({path.name})" for path in charts]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the ML pipeline on public datasets.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, reports, charts = [], {}, []
    for dataset in DATASETS:
        print(f"{dataset.name}: loading")
        frame = dataset.load()
        started = time.perf_counter()
        report = run_model_comparison(frame, validate_setup(frame, dataset.target, None))
        seconds = time.perf_counter() - started
        print(f"{dataset.name}: best {report.best_model} in {seconds:.0f}s, {report.test_metrics}")
        rows += result_rows(dataset, report, seconds)
        reports[dataset.name] = report
        charts.append(importance_chart(dataset, report, args.out_dir))

    results = pd.DataFrame(rows)
    results.to_csv(args.out_dir / "ml_results.csv", index=False)
    markdown_text = report_markdown(results, reports, charts)
    (args.out_dir / "ml_results.md").write_text(markdown_text, encoding="utf-8")
    print(markdown_text)
