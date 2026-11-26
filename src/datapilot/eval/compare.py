"""Execution-accuracy comparison between a gold result and a predicted result."""

import datetime as dt
import math
import numbers
import sqlite3
import time
from collections import Counter
from collections.abc import Sequence
from contextlib import closing
from decimal import Decimal
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MAX_PERMUTED_COLUMNS = 6


def normalise_value(value: Any) -> Any:
    """Make SQLite and DuckDB values comparable (types, float noise, date formats)."""
    if value is None:
        return None
    if isinstance(value, float | np.floating) and math.isnan(value):
        return None
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, bool | np.bool_):
        return float(int(value))
    if isinstance(value, numbers.Number | Decimal):
        return round(float(value), 4) + 0.0
    if isinstance(value, dt.datetime | pd.Timestamp):
        if value.hour == value.minute == value.second == 0 and not value.microsecond:
            return value.date().isoformat()
        return value.isoformat(sep=" ")
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    text = str(value).strip()
    try:
        return round(float(text), 4) + 0.0
    except ValueError:
        return text


def normalise_rows(rows: Sequence[Sequence[Any]]) -> list[tuple[Any, ...]]:
    return [tuple(normalise_value(value) for value in row) for row in rows]


def frame_rows(frame: pd.DataFrame | None) -> list[tuple[Any, ...]]:
    if frame is None:
        return []
    return [tuple(row) for row in frame.itertuples(index=False, name=None)]


def results_match(
    gold: Sequence[Sequence[Any]], predicted: Sequence[Sequence[Any]], ordered: bool
) -> bool:
    """Spider-style execution match: same rows, columns may come in any order."""
    gold_rows, predicted_rows = normalise_rows(gold), normalise_rows(predicted)
    if len(gold_rows) != len(predicted_rows):
        return False
    if not gold_rows:
        return True
    width = len(gold_rows[0])
    if any(len(row) != width for row in predicted_rows):
        return False

    gold_columns = [Counter(column) for column in zip(*gold_rows, strict=True)]
    predicted_columns = [Counter(column) for column in zip(*predicted_rows, strict=True)]
    candidates = (
        permutations(range(width)) if width <= MAX_PERMUTED_COLUMNS else [tuple(range(width))]
    )
    for order in candidates:
        if any(predicted_columns[order[i]] != gold_columns[i] for i in range(width)):
            continue
        reordered = [tuple(row[i] for i in order) for row in predicted_rows]
        if ordered and reordered == gold_rows:
            return True
        if not ordered and Counter(reordered) == Counter(gold_rows):
            return True
    return False


def is_ordered(gold_sql: str) -> bool:
    return "order by" in " ".join(gold_sql.lower().split())


def execute_gold(sqlite_path: Path, sql: str, timeout_seconds: float = 30.0) -> list[tuple]:
    deadline = time.monotonic() + timeout_seconds
    uri = f"file:{sqlite_path.as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.text_factory = lambda raw: raw.decode("utf-8", "replace")
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        return connection.execute(sql).fetchall()
