from dataclasses import dataclass

import pandas as pd

from datapilot.errors import ErrorCategory


@dataclass
class ExecutionResult:
    ok: bool
    frame: pd.DataFrame | None = None
    error: str | None = None
    category: ErrorCategory | None = None
    truncated: bool = False
    elapsed_ms: float = 0.0

    @property
    def row_count(self) -> int:
        return 0 if self.frame is None else len(self.frame)


def dedupe_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique = []
    for column in map(str, columns):
        count = seen.get(column, 0)
        seen[column] = count + 1
        unique.append(column if count == 0 else f"{column}_{count + 1}")
    return unique


def preview(frame: pd.DataFrame | None, max_rows: int = 15) -> str:
    """Plain-text table for prompts; to_string avoids an extra markdown dependency."""
    if frame is None:
        return "(no result)"
    if frame.empty:
        return f"(0 rows; columns: {', '.join(map(str, frame.columns))})"
    shown = frame.head(max_rows).to_string(index=False, max_colwidth=40)
    more = len(frame) - max_rows
    return shown + (f"\n... {more:,} more rows" if more > 0 else "")
