import math
import re
import threading
import time
from pathlib import Path

import duckdb

from datapilot.data.loader import connect_read_only
from datapilot.errors import ErrorCategory
from datapilot.tools.execution import ExecutionResult, dedupe_columns

DUCKDB_VECTOR_SIZE = 2048


class UnsafeQueryError(ValueError):
    pass


def check_read_only(sql: str) -> None:
    """Accept exactly one SELECT statement (CTEs included)."""
    statements = duckdb.extract_statements(sql)
    if len(statements) != 1:
        raise UnsafeQueryError(f"Expected exactly one SQL statement, found {len(statements)}.")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise UnsafeQueryError(f"Only SELECT queries are allowed, got {statements[0].type.name}.")


def classify_duckdb_error(error: Exception) -> ErrorCategory:
    message = str(error)
    if isinstance(error, duckdb.InterruptException):
        return ErrorCategory.TIMEOUT
    if isinstance(error, duckdb.ParserException):
        return ErrorCategory.SYNTAX_ERROR
    if isinstance(error, duckdb.CatalogException):
        if re.search(r"Table .* does not exist", message):
            return ErrorCategory.UNKNOWN_TABLE
        return ErrorCategory.RUNTIME_ERROR
    if isinstance(error, duckdb.BinderException):
        if "Referenced column" in message or "not found in FROM clause" in message:
            return ErrorCategory.UNKNOWN_COLUMN
        if "No function matches" in message or "Cannot compare" in message:
            return ErrorCategory.TYPE_MISMATCH
        return ErrorCategory.RUNTIME_ERROR
    if isinstance(error, duckdb.ConversionException | duckdb.TypeMismatchException):
        return ErrorCategory.TYPE_MISMATCH
    if isinstance(error, duckdb.PermissionException):
        return ErrorCategory.UNSAFE_CODE
    return ErrorCategory.RUNTIME_ERROR


def run_sql(
    db_path: Path,
    sql: str,
    *,
    row_limit: int = 5000,
    timeout_seconds: float = 15.0,
) -> ExecutionResult:
    """Run one read-only query with a row cap and a wall-clock timeout."""
    started = time.perf_counter()

    def elapsed() -> float:
        return (time.perf_counter() - started) * 1000

    sql = sql.strip().rstrip(";").strip()
    if not sql:
        return ExecutionResult(False, error="Empty query.", category=ErrorCategory.NO_CODE)
    try:
        check_read_only(sql)
    except UnsafeQueryError as error:
        return ExecutionResult(
            False, error=str(error), category=ErrorCategory.UNSAFE_CODE, elapsed_ms=elapsed()
        )
    except duckdb.Error as error:
        return ExecutionResult(
            False, error=str(error), category=classify_duckdb_error(error), elapsed_ms=elapsed()
        )

    connection = connect_read_only(db_path)
    # DuckDB has no per-query timeout setting, so a timer interrupts the connection instead.
    timer = threading.Timer(timeout_seconds, connection.interrupt)
    timer.start()
    try:
        cursor = connection.execute(sql)
        vectors = math.ceil((row_limit + 1) / DUCKDB_VECTOR_SIZE)
        frame = cursor.fetch_df_chunk(vectors)
    except duckdb.Error as error:
        category = classify_duckdb_error(error)
        message = (
            f"Query exceeded the {timeout_seconds:g}s time limit."
            if category is ErrorCategory.TIMEOUT
            else str(error)
        )
        return ExecutionResult(False, error=message, category=category, elapsed_ms=elapsed())
    finally:
        timer.cancel()
        connection.close()

    truncated = len(frame) > row_limit
    frame = frame.head(row_limit).set_axis(dedupe_columns(list(frame.columns)), axis=1)
    return ExecutionResult(True, frame=frame, truncated=truncated, elapsed_ms=elapsed())
