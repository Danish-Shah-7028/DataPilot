from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from datapilot.data.loader import connect_read_only, quote_identifier

NUMERIC_TYPES = (
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
    "REAL",
)
TEMPORAL_TYPES = ("DATE", "TIMESTAMP", "TIME")
MAX_SAMPLE_CHARS = 40
LOW_CARDINALITY = 12


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    null_fraction: float = 0.0
    distinct_count: int = 0
    sample_values: tuple[str, ...] = ()
    min_value: str | None = None
    max_value: str | None = None
    original_name: str | None = None

    @property
    def is_numeric(self) -> bool:
        return self.dtype.upper().startswith(NUMERIC_TYPES)

    @property
    def is_temporal(self) -> bool:
        return self.dtype.upper().startswith(TEMPORAL_TYPES)

    def describe(self) -> str:
        header = f"{self.name} {self.dtype}"
        if self.original_name:
            header += f' (originally "{self.original_name}")'
        details = []
        if self.min_value is not None and (self.is_numeric or self.is_temporal):
            details.append(f"range {self.min_value} to {self.max_value}")
        if self.sample_values:
            details.append("values like " + ", ".join(repr(value) for value in self.sample_values))
        if self.distinct_count:
            details.append(f"~{self.distinct_count:,} distinct")
        if self.null_fraction > 0:
            details.append(f"{self.null_fraction:.0%} null")
        return header + (": " + "; ".join(details) if details else "")


@dataclass(frozen=True)
class TableProfile:
    name: str
    row_count: int
    columns: tuple[ColumnProfile, ...]

    @property
    def column_names(self) -> list[str]:
        return [column.name for column in self.columns]

    def column(self, name: str) -> ColumnProfile | None:
        lowered = name.lower()
        return next((col for col in self.columns if col.name.lower() == lowered), None)

    def render(self, column_names: Collection[str] | None = None) -> str:
        lines = [f"TABLE {self.name} ({self.row_count:,} rows)"]
        shown = [
            column
            for column in self.columns
            if column_names is None or column.name in column_names
        ]
        lines += [f"  - {column.describe()}" for column in shown]
        hidden = len(self.columns) - len(shown)
        if hidden:
            lines.append(f"  - ... {hidden} more columns not shown")
        return "\n".join(lines)


@dataclass(frozen=True)
class SchemaProfile:
    tables: tuple[TableProfile, ...]

    @property
    def column_count(self) -> int:
        return sum(len(table.columns) for table in self.tables)

    @property
    def table_names(self) -> list[str]:
        return [table.name for table in self.tables]

    def table(self, name: str) -> TableProfile | None:
        lowered = name.lower()
        return next((table for table in self.tables if table.name.lower() == lowered), None)

    def render(self, selection: Mapping[str, Collection[str] | None] | None = None) -> str:
        """Render the schema for a prompt; `selection` maps table -> columns (None = all)."""
        if selection is None:
            return "\n\n".join(table.render() for table in self.tables)
        blocks = []
        for table in self.tables:
            if table.name in selection:
                blocks.append(table.render(selection[table.name]))
        return "\n\n".join(blocks)

    def summary(self) -> str:
        """One line per table, used where the full profile would waste tokens."""
        return "\n".join(
            f"{table.name}({', '.join(table.column_names)})" for table in self.tables
        )


def clip(value: object) -> str:
    text = str(value)
    return text if len(text) <= MAX_SAMPLE_CHARS else text[: MAX_SAMPLE_CHARS - 3] + "..."


def profile_table(connection, table_name: str, sample_size: int) -> TableProfile:
    table = quote_identifier(table_name)
    columns = connection.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position",
        [table_name],
    ).fetchall()

    aggregates = ["count(*)"]
    for name, dtype in columns:
        column = quote_identifier(name)
        probe = ColumnProfile(name, dtype)
        aggregates.append(f"count({column})")
        aggregates.append(f"approx_count_distinct({column})")
        if probe.is_numeric or probe.is_temporal:
            aggregates += [f"min({column})::VARCHAR", f"max({column})::VARCHAR"]
        else:
            aggregates += ["NULL", "NULL"]
    stats = connection.execute(f"SELECT {', '.join(aggregates)} FROM {table}").fetchall()[0]
    row_count = stats[0]

    profiles = []
    for position, (name, dtype) in enumerate(columns):
        non_null, distinct, min_value, max_value = stats[1 + position * 4 : 5 + position * 4]
        probe = ColumnProfile(name, dtype)
        samples: tuple[str, ...] = ()
        if non_null and (not probe.is_numeric or distinct <= LOW_CARDINALITY):
            column = quote_identifier(name)
            rows = connection.execute(
                f"SELECT {column}::VARCHAR AS value, count(*) AS frequency FROM {table} "
                f"WHERE {column} IS NOT NULL GROUP BY 1 ORDER BY frequency DESC, value "
                f"LIMIT {int(sample_size)}"
            ).fetchall()
            samples = tuple(clip(value) for value, unused_frequency in rows)
        profiles.append(
            ColumnProfile(
                name=name,
                dtype=dtype,
                null_fraction=round(1 - non_null / row_count, 3) if row_count else 0.0,
                distinct_count=int(distinct or 0),
                sample_values=samples,
                min_value=min_value,
                max_value=max_value,
            )
        )
    return TableProfile(table_name, row_count, tuple(profiles))


def profile_database(
    db_path: Path,
    renamed_columns: Mapping[str, Mapping[str, str]] | None = None,
    sample_size: int = 5,
) -> SchemaProfile:
    """Collect types, ranges and sample values for every table in the workspace."""
    renamed_columns = renamed_columns or {}
    with connect_read_only(db_path) as connection:
        names = [
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
        ]
        tables = []
        for name in names:
            profile = profile_table(connection, name, sample_size)
            originals = renamed_columns.get(name, {})
            if originals:
                profile = TableProfile(
                    profile.name,
                    profile.row_count,
                    tuple(
                        ColumnProfile(**{**vars(column), "original_name": originals.get(column.name)})
                        for column in profile.columns
                    ),
                )
            tables.append(profile)
    return SchemaProfile(tuple(tables))
