from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

ChartKind = Literal["bar", "line", "scatter", "pie", "histogram"]
CHART_KINDS: tuple[str, ...] = ("bar", "line", "scatter", "pie", "histogram")
TIME_HINTS = ("date", "month", "year", "week", "day", "quarter", "period", "time")
MAX_BAR_CATEGORIES = 30
MAX_PIE_SLICES = 8
LAYOUT = {"template": "plotly_white", "margin": {"l": 20, "r": 20, "t": 60, "b": 20}}


@dataclass(frozen=True)
class ChartSpec:
    kind: ChartKind
    x: str
    y: str | None = None
    color: str | None = None
    title: str = ""


def is_key_column(name: str) -> bool:
    lowered = name.lower()
    return lowered == "id" or lowered.endswith("_id")


def is_measure(frame: pd.DataFrame, column: str) -> bool:
    series = frame[column]
    return (
        is_numeric_dtype(series)
        and not is_bool_dtype(series)
        and not is_key_column(column)
        and not is_time_column(frame, column)
    )


def is_time_column(frame: pd.DataFrame, column: str) -> bool:
    series = frame[column]
    if is_datetime64_any_dtype(series):
        return True
    lowered = column.lower()
    if not any(hint in lowered for hint in TIME_HINTS):
        return False
    # A "year" or "month" column holding small integers is still a time axis.
    return not is_numeric_dtype(series) or series.dropna().between(1, 3000).all()


def suggest_chart(frame: pd.DataFrame | None, title: str = "") -> ChartSpec | None:
    """Pick a sensible default chart from the shape and types of the result."""
    if frame is None or len(frame) < 2:
        return None
    columns = [str(column) for column in frame.columns]
    measures = [column for column in columns if is_measure(frame, column)]
    times = [column for column in columns if is_time_column(frame, column)]
    labels = [column for column in columns if column not in measures and column not in times]
    if not measures:
        return None

    if times:
        color = next((col for col in labels if frame[col].nunique() <= 10), None)
        return ChartSpec("line", x=times[0], y=measures[0], color=color, title=title)
    if labels and frame[labels[0]].nunique() <= MAX_BAR_CATEGORIES:
        return ChartSpec("bar", x=labels[0], y=measures[0], title=title)
    if len(measures) >= 2:
        return ChartSpec("scatter", x=measures[0], y=measures[1], title=title)
    if len(frame) >= 10:
        return ChartSpec("histogram", x=measures[0], title=title)
    return None


def validate_chart(raw: Mapping[str, Any] | None, frame: pd.DataFrame | None) -> ChartSpec | None:
    """Accept a chart proposed by the model only if it fits the actual result."""
    if not raw or frame is None or frame.empty:
        return None
    kind = str(raw.get("type") or "none").lower()
    if kind not in CHART_KINDS:
        return None
    columns = {str(column) for column in frame.columns}
    x, y, color = raw.get("x"), raw.get("y"), raw.get("color")
    if x not in columns:
        return None
    if kind != "histogram" and (y not in columns or not is_numeric_dtype(frame[y])):
        return None
    if color not in columns:
        color = None
    if kind == "pie" and (frame[x].nunique() > MAX_PIE_SLICES or (frame[y] < 0).any()):
        kind = "bar"
    title = str(raw.get("title") or "")
    return ChartSpec(kind, x=x, y=None if kind == "histogram" else y, color=color, title=title)


def build_figure(frame: pd.DataFrame, spec: ChartSpec) -> go.Figure:
    if spec.kind == "line":
        figure = px.line(
            frame.sort_values(spec.x), x=spec.x, y=spec.y, color=spec.color, markers=True
        )
    elif spec.kind == "bar":
        data = frame if spec.color else frame.sort_values(spec.y, ascending=False)
        figure = px.bar(data, x=spec.x, y=spec.y, color=spec.color)
    elif spec.kind == "scatter":
        figure = px.scatter(frame, x=spec.x, y=spec.y, color=spec.color)
    elif spec.kind == "pie":
        figure = px.pie(frame, names=spec.x, values=spec.y)
    else:
        figure = px.histogram(frame, x=spec.x)
    figure.update_layout(title=spec.title, **LAYOUT)
    return figure


def feature_importance_figure(importance: pd.DataFrame, top: int = 15) -> go.Figure:
    data = importance.head(top).iloc[::-1]
    figure = px.bar(
        data, x="importance", y="feature", orientation="h", error_x="std",
        title="Permutation feature importance (test set)",
    )  # fmt: skip
    figure.update_layout(**LAYOUT)
    return figure


def confusion_matrix_figure(matrix: Sequence[Sequence[int]], labels: Sequence[str]) -> go.Figure:
    figure = px.imshow(
        matrix,
        x=list(labels),
        y=list(labels),
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual", "color": "Rows"},
        title="Confusion matrix (test set)",
    )
    figure.update_layout(**LAYOUT)
    return figure
