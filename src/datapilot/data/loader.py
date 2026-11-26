import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd
from pandas.api.types import is_numeric_dtype

CSV_SUFFIXES = {".csv", ".tsv", ".txt"}
EXCEL_SUFFIXES = {".xlsx", ".xls"}
SQLITE_SUFFIXES = {".sqlite", ".sqlite3", ".db"}
PARQUET_SUFFIXES = {".parquet"}
SUPPORTED_SUFFIXES = CSV_SUFFIXES | EXCEL_SUFFIXES | SQLITE_SUFFIXES | PARQUET_SUFFIXES

# Words that would force the model to quote identifiers in every query.
RESERVED_WORDS = frozenset(
    """
    all analyse analyze and any array as asc between both by case cast check collate column
    constraint create default delete desc distinct do drop else end except false fetch for
    foreign from grant group having in initially insert intersect into is join lateral leading
    limit not null offset on only or order placing primary references returning select some
    symmetric table then to trailing true union unique update using variadic when where window
    with
    """.split()
)


class UnsupportedFileError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedTable:
    name: str
    source: str
    row_count: int
    renamed_columns: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Workspace:
    """A DuckDB file holding every table the user uploaded."""

    db_path: Path
    tables: list[LoadedTable]

    @property
    def table_names(self) -> list[str]:
        return [table.name for table in self.tables]


def safe_identifier(raw: str, taken: set[str], fallback: str = "col") -> str:
    """Turn an arbitrary header into a lowercase snake_case SQL identifier."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(raw).strip()).strip("_").lower() or fallback
    if name[0].isdigit():
        name = f"{fallback}_{name}"
    if name in RESERVED_WORDS:
        name = f"{name}_value"
    candidate, suffix = name, 2
    while candidate in taken:
        candidate = f"{name}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def tidy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Make mixed-type object columns safe to hand to DuckDB.

    SQLite does not enforce column types, so a column can hold ints and strings at once.
    Columns that are purely numeric become numbers; anything else becomes text.
    """
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype != object:
            continue
        values = frame[column].dropna()
        if values.empty:
            frame[column] = frame[column].astype("string")
        elif values.map(lambda value: isinstance(value, int | float)).all():
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        elif values.map(lambda value: isinstance(value, bytes)).any():
            frame[column] = frame[column].map(
                lambda value: value.decode("utf-8", "replace") if isinstance(value, bytes) else value
            )
            frame[column] = frame[column].astype("string")
        else:
            frame[column] = frame[column].map(lambda value: None if pd.isna(value) else str(value))
    return frame


def rename_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    taken: set[str] = set()
    mapping = {str(column): safe_identifier(str(column), taken) for column in frame.columns}
    renamed = {new: old for old, new in mapping.items() if old != new}
    return frame.set_axis(list(mapping.values()), axis=1), renamed


def write_frame(
    connection: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    table_name: str,
    source: str,
) -> LoadedTable:
    frame, renamed = rename_columns(tidy_frame(frame))
    connection.register("staging_frame", frame)
    try:
        connection.execute(
            f"CREATE TABLE {quote_identifier(table_name)} AS SELECT * FROM staging_frame"
        )
    finally:
        connection.unregister("staging_frame")
    return LoadedTable(table_name, source, len(frame), renamed)


def load_csv(
    connection: duckdb.DuckDBPyConnection, path: Path, taken: set[str]
) -> list[LoadedTable]:
    # DuckDB's CSV sniffer is better than pandas at spotting dates and delimiters.
    relation = connection.read_csv(str(path))
    column_names: set[str] = set()
    projections, renamed = [], {}
    for column in relation.columns:
        new_name = safe_identifier(column, column_names)
        projections.append(f"{quote_identifier(column)} AS {quote_identifier(new_name)}")
        if new_name != column:
            renamed[new_name] = column
    table_name = safe_identifier(path.stem, taken, fallback="tbl")
    relation.project(", ".join(projections)).create(table_name)
    row_count = connection.execute(f"SELECT count(*) FROM {quote_identifier(table_name)}")
    return [LoadedTable(table_name, path.name, row_count.fetchall()[0][0], renamed)]


def load_excel(
    connection: duckdb.DuckDBPyConnection, path: Path, taken: set[str]
) -> list[LoadedTable]:
    sheets = pd.read_excel(path, sheet_name=None)
    loaded = []
    for sheet_name, frame in sheets.items():
        if frame.empty:
            continue
        stem = path.stem if len(sheets) == 1 else f"{path.stem}_{sheet_name}"
        table_name = safe_identifier(stem, taken, fallback="tbl")
        loaded.append(write_frame(connection, frame, table_name, f"{path.name} [{sheet_name}]"))
    return loaded


def parse_declared_dates(frame: pd.DataFrame, declared_types: Mapping[str, str]) -> pd.DataFrame:
    """SQLite stores dates as text; parse columns declared as dates when nearly all values fit."""
    frame = frame.copy()
    for column, declared in declared_types.items():
        if column not in frame.columns or not re.search(r"DATE|TIME", declared, re.IGNORECASE):
            continue
        values = frame[column]
        if is_numeric_dtype(values) or values.dropna().empty:
            continue
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
        if parsed.notna().sum() >= 0.95 * values.notna().sum():
            frame[column] = parsed
    return frame


def load_sqlite(
    connection: duckdb.DuckDBPyConnection, path: Path, taken: set[str]
) -> list[LoadedTable]:
    loaded = []
    with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as source:
        source.text_factory = lambda raw: raw.decode("utf-8", "replace")
        names = [
            row[0]
            for row in source.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite%' ORDER BY name"
            )
        ]
        for name in names:
            frame = pd.read_sql_query(f"SELECT * FROM {quote_identifier(name)}", source)
            declared = source.execute(f"PRAGMA table_info({quote_identifier(name)})").fetchall()
            frame = parse_declared_dates(frame, {row[1]: str(row[2]) for row in declared})
            table_name = safe_identifier(name, taken, fallback="tbl")
            loaded.append(write_frame(connection, frame, table_name, f"{path.name} [{name}]"))
    return loaded


def load_parquet(
    connection: duckdb.DuckDBPyConnection, path: Path, taken: set[str]
) -> list[LoadedTable]:
    frame = connection.read_parquet(str(path)).df()
    table_name = safe_identifier(path.stem, taken, fallback="tbl")
    return [write_frame(connection, frame, table_name, path.name)]


def load_files(paths: Sequence[Path], db_path: Path) -> Workspace:
    """Load CSV, Excel, SQLite and Parquet files into a fresh DuckDB database file."""
    for path in paths:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise UnsupportedFileError(f"{path.name}: unsupported file type {path.suffix!r}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    taken: set[str] = set()
    tables: list[LoadedTable] = []
    with duckdb.connect(str(db_path)) as connection:
        for path in paths:
            suffix = path.suffix.lower()
            if suffix in CSV_SUFFIXES:
                tables += load_csv(connection, path, taken)
            elif suffix in EXCEL_SUFFIXES:
                tables += load_excel(connection, path, taken)
            elif suffix in SQLITE_SUFFIXES:
                tables += load_sqlite(connection, path, taken)
            else:
                tables += load_parquet(connection, path, taken)
    return Workspace(db_path=db_path, tables=tables)


def load_frames(frames: Mapping[str, pd.DataFrame], db_path: Path) -> Workspace:
    """Build a workspace straight from DataFrames (handy for tests and notebooks)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    taken: set[str] = set()
    with duckdb.connect(str(db_path)) as connection:
        tables = [
            write_frame(connection, frame, safe_identifier(name, taken, "tbl"), name)
            for name, frame in frames.items()
        ]
    return Workspace(db_path=db_path, tables=tables)


def connect_read_only(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open the workspace so that queries can neither write nor touch outside files."""
    return duckdb.connect(
        str(db_path), read_only=True, config={"enable_external_access": False}
    )


def read_tables(
    db_path: Path, names: Iterable[str], row_cap: int
) -> dict[str, pd.DataFrame]:
    with connect_read_only(db_path) as connection:
        return {
            name: connection.execute(
                f"SELECT * FROM {quote_identifier(name)} LIMIT {int(row_cap)}"
            ).df()
            for name in names
        }
