from typing import Any

from datapilot.graph.context import AgentContext
from datapilot.graph.state import AgentState, Attempt
from datapilot.tools.python_sandbox import run_python
from datapilot.tools.sql_executor import run_sql


def execute_code(state: AgentState, context: AgentContext) -> dict[str, Any]:
    code = state.get("code", "")
    language = state.get("language", "sql")
    config = context.config
    if language == "sql":
        result = run_sql(
            context.db_path,
            code,
            row_limit=config.row_limit,
            timeout_seconds=config.sql_timeout_seconds,
        )
    else:
        tables = state.get("retrieved_tables") or context.profile.table_names
        result = run_python(
            code,
            context.table_frames(tables),
            row_limit=config.row_limit,
            timeout_seconds=config.sandbox_timeout_seconds,
        )

    attempt = Attempt(
        step=state.get("step_index", 0),
        language=language,
        code=code,
        ok=result.ok,
        error=result.error,
        category=None if result.category is None else str(result.category),
        row_count=result.row_count,
    )
    detail = f"{result.row_count} rows" if result.ok else f"error: {result.category}"
    return {
        "execution": result,
        "attempts": [*state.get("attempts", []), attempt],
        "trace_detail": detail,
    }
