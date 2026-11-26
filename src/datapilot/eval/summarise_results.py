"""Turn the per-question evaluation logs into tables and charts in reports/."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from datapilot.eval.run_eval import VARIANTS

LABELS = {
    "baseline": "Single LLM call",
    "agent_no_critic": "Agent, no critic",
    "agent_full": "Agent + critic",
}
COLOURS = {"baseline": "#9aa5b1", "agent_no_critic": "#5b8def", "agent_full": "#1f4e9e"}


def load_runs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(runs_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows += [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise SystemExit(f"No evaluation logs found in {runs_dir}; run datapilot-eval first.")
    frame = pd.DataFrame(rows)
    frame["tokens"] = frame["prompt_tokens"] + frame["completion_tokens"]
    return frame


def common_questions(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare variants on the same questions, even if one run was cut short."""
    per_variant = frame.groupby("variant")["id"].apply(set)
    shared = set.intersection(*per_variant)
    return frame[frame["id"].isin(shared)]


def difficulty(sql: str) -> str:
    lowered = " ".join(sql.lower().split())
    if len(re.findall(r"\bselect\b", lowered)) > 1:
        return "nested / set ops"
    if " join " in lowered:
        return "join"
    if "group by" in lowered:
        return "group by"
    return "single table"


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("variant")
    table = pd.DataFrame(
        {
            "questions": grouped.size(),
            "execution_accuracy": grouped["correct"].mean(),
            "valid_sql_rate": grouped["executed"].mean(),
            "avg_retries": grouped["retries"].mean(),
            "avg_llm_calls": grouped["llm_calls"].mean(),
            "avg_latency_s": grouped["latency_ms"].mean() / 1000,
            "median_latency_s": grouped["latency_ms"].median() / 1000,
            "avg_tokens": grouped["tokens"].mean(),
        }
    )
    order = [variant for variant in VARIANTS if variant in table.index]
    return table.loc[order]


def difficulty_table(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.assign(difficulty=frame["gold_sql"].map(difficulty))
    table = frame.pivot_table(
        index="difficulty", columns="variant", values="correct", aggfunc="mean"
    )
    counts = frame[frame["variant"] == frame["variant"].iloc[0]].groupby("difficulty").size()
    table.insert(0, "questions", counts)
    return table[["questions", *[v for v in VARIANTS if v in table.columns]]]


def error_table(frame: pd.DataFrame) -> pd.DataFrame:
    exploded = frame[["variant", "error_categories"]].explode("error_categories").dropna()
    if exploded.empty:
        return pd.DataFrame()
    return (
        exploded.groupby(["error_categories", "variant"]).size().unstack(fill_value=0).sort_index()
    )


def markdown(table: pd.DataFrame, percent: tuple[str, ...] = ()) -> str:
    columns = [table.index.name or "", *map(str, table.columns)]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for index, row in table.iterrows():
        cells = [LABELS.get(str(index), str(index))]
        for column, value in row.items():
            if str(column).lower() == "questions":
                cells.append(f"{int(value):,}")
            elif column in percent or percent == ("*",):
                cells.append("" if pd.isna(value) else f"{value:.1%}")
            elif isinstance(value, float):
                cells.append(f"{value:,.2f}")
            else:
                cells.append(f"{value:,}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_charts(frame: pd.DataFrame, table: pd.DataFrame, out_dir: Path) -> list[Path]:
    variants = list(table.index)
    labels = [LABELS[variant] for variant in variants]
    colours = [COLOURS[variant] for variant in variants]
    paths = []

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, column, title in [
        (axes[0], "execution_accuracy", "Execution accuracy"),
        (axes[1], "valid_sql_rate", "Valid SQL rate"),
    ]:
        bars = axis.bar(labels, table[column] * 100, color=colours)
        axis.bar_label(bars, fmt="%.1f%%")
        axis.set_title(title)
        axis.set_ylim(0, 105)
        axis.set_ylabel("% of questions")
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    paths.append(out_dir / "accuracy_by_variant.png")
    figure.savefig(paths[-1], dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, column, title in [
        (axes[0], "avg_latency_s", "Average latency (s)"),
        (axes[1], "avg_tokens", "Average tokens per question"),
        (axes[2], "avg_retries", "Average retries per question"),
    ]:
        bars = axis.bar(labels, table[column], color=colours)
        axis.bar_label(bars, fmt="%.2f" if column != "avg_tokens" else "%.0f")
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    paths.append(out_dir / "cost_by_variant.png")
    figure.savefig(paths[-1], dpi=150)
    plt.close(figure)

    errors = error_table(frame)
    if not errors.empty:
        errors = errors[[variant for variant in variants if variant in errors.columns]]
        axis = errors.plot.barh(
            figsize=(9, 5), color=[COLOURS[variant] for variant in errors.columns]
        )
        axis.set_title("Errors seen during runs (including ones fixed by retries)")
        axis.set_xlabel("occurrences")
        axis.legend([LABELS[variant] for variant in errors.columns])
        axis.spines[["top", "right"]].set_visible(False)
        axis.figure.tight_layout()
        paths.append(out_dir / "error_categories.png")
        axis.figure.savefig(paths[-1], dpi=150)
        plt.close(axis.figure)
    return paths


def build_report(runs_dir: Path, out_dir: Path) -> str:
    all_runs = load_runs(runs_dir)
    frame = common_questions(all_runs)
    table = summary_table(frame)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "text_to_sql_results.csv", float_format="%.4f")
    by_difficulty = difficulty_table(frame)
    by_difficulty.to_csv(out_dir / "text_to_sql_by_difficulty.csv", float_format="%.4f")
    charts = save_charts(frame, table, out_dir)

    models = ", ".join(sorted(frame["model"].dropna().unique()))
    display = table.copy()
    display.index.name = "Variant"
    display.columns = [
        "Questions",
        "Execution accuracy",
        "Valid SQL",
        "Avg retries",
        "Avg LLM calls",
        "Avg latency (s)",
        "Median latency (s)",
        "Avg tokens",
    ]
    by_difficulty.index.name = "Gold query type"
    by_difficulty.columns = ["Questions", *[LABELS[c] for c in by_difficulty.columns[1:]]]
    lines = [
        "# Text-to-SQL evaluation",
        "",
        f"Spider dev subset, {table['questions'].iloc[0]} questions answered by every variant. "
        f"Model: {models}.",
        "",
        markdown(display, percent=("Execution accuracy", "Valid SQL")),
        "",
        "## Accuracy by type of gold query",
        "",
        markdown(by_difficulty, percent=("*",)),
        "",
        "Execution accuracy: the result set matches the gold query's result on the original "
        "SQLite database (column order ignored, row order only checked when the gold query "
        "has ORDER BY). Latency counts LLM time plus query execution.",
        "",
        *[f"![{path.stem}]({path.name})" for path in charts],
    ]
    report = "\n".join(lines) + "\n"
    (out_dir / "text_to_sql_results.md").write_text(report, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise evaluation logs into reports/.")
    parser.add_argument("--runs-dir", type=Path, default=Path("reports/eval_runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    print(build_report(args.runs_dir, args.out_dir))
