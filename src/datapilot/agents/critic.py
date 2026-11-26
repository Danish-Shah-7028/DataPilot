import difflib
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from datapilot.agents.formatting import clip, messages
from datapilot.data.schema import SchemaProfile
from datapilot.errors import ErrorCategory
from datapilot.graph.context import AgentContext
from datapilot.graph.state import AgentState, StepResult
from datapilot.graph.tracing import summarise_run
from datapilot.llm.json_output import LLMOutputError, parse_json_object
from datapilot.tools.execution import ExecutionResult, preview

SINGLE_VALUE_QUESTION = re.compile(
    r"^\s*(how many|how much|what is the (total|average|mean|number|sum|overall)|"
    r"what's the (total|average|number))\b",
    re.IGNORECASE,
)
GROUPING_WORDS = re.compile(r"\b(per|by|each|every|for all|breakdown|across|top)\b", re.IGNORECASE)
COUNT_COLUMN = re.compile(r"(^|_)(count|num|number|n)(_|$)|_count$", re.IGNORECASE)
PERCENT_COLUMN = re.compile(r"(percent|pct|percentage)", re.IGNORECASE)

REVIEW_PROMPT = """You are the verifier of DataPilot, an AI data analyst. Check whether \
the result correctly answers the current step. Judge only the current step, not the \
rest of the question; later steps will handle the rest.

Look for: wrong aggregation (sum vs average vs count), missing or wrong filters, wrong \
grouping, joins that duplicate rows, wrong sort direction, the wrong time period and \
impossible values.

Only reject a result when you can name a specific, concrete error in the code or the \
values. If the result reasonably answers the step, or you are unsure, accept it. Never \
ask for cosmetic changes such as column names, rounding, extra columns, ordering of \
columns or formatting.

Reply with JSON only: {"ok": true | false, "issue": "what is wrong, or empty", \
"fix": "a concrete instruction for the coder, or empty"}"""

STRICT_NOTE = """

The automatic confidence score for this answer is low, so check it carefully against \
the question before accepting it."""


@dataclass(frozen=True)
class Review:
    ok: bool
    category: ErrorCategory | None = None
    feedback: str = ""


def missing_name(error: str) -> str | None:
    match = re.search(r'(?:column|Table with name|name) "?([A-Za-z0-9_]+)"?', error)
    return match.group(1) if match else None


def execution_hint(result: ExecutionResult, profile: SchemaProfile) -> str:
    category = result.category
    if category in (ErrorCategory.UNKNOWN_COLUMN, ErrorCategory.UNKNOWN_TABLE):
        candidates = (
            [f"{table.name}.{column}" for table in profile.tables for column in table.column_names]
            if category is ErrorCategory.UNKNOWN_COLUMN
            else profile.table_names
        )
        name = missing_name(result.error or "")
        short = [candidate.split(".")[-1] for candidate in candidates]
        close = difflib.get_close_matches(name or "", short, n=3, cutoff=0.5) if name else []
        if close:
            return f"Did you mean: {', '.join(close)}? Use only names from the schema."
        return "Use only table and column names that appear in the schema."
    if category is ErrorCategory.TYPE_MISMATCH:
        return "Check the column types in the schema and cast text values explicitly."
    if category is ErrorCategory.TIMEOUT:
        return "Simplify the query: filter and aggregate before joining, avoid cross joins."
    if category is ErrorCategory.UNSAFE_CODE:
        return "Only read-only analysis is allowed: one SELECT statement or plain pandas code."
    if category is ErrorCategory.NO_CODE:
        return "Return the code in the JSON 'code' field."
    return "Fix the error and keep the same intent."


def check_execution(result: ExecutionResult, profile: SchemaProfile) -> Review | None:
    if result.ok:
        return None
    feedback = f"The code failed with: {clip(result.error or 'unknown error', 600)}\n"
    return Review(False, result.category, feedback + execution_hint(result, profile))


def referenced_tables(code: str, profile: SchemaProfile) -> list[int]:
    lowered = code.lower()
    return [
        table.row_count
        for table in profile.tables
        if re.search(rf"\b{re.escape(table.name.lower())}\b", lowered)
    ]


def check_sanity(
    step: str,
    frame: pd.DataFrame,
    code: str,
    profile: SchemaProfile,
    previous_empty: bool,
) -> Review | None:
    """Cheap rule-based checks that catch the classic silent text-to-SQL failures."""
    if frame.empty:
        if previous_empty:
            return None  # Two empty results in a row: probably the honest answer.
        return Review(
            False,
            ErrorCategory.EMPTY_RESULT,
            "The result is empty. Check that filter values match the data exactly (see the "
            "sample values in the schema), that date ranges exist and that joins use the "
            "right keys. If no rows really is the right answer, return the same code again.",
        )
    if frame.isna().all().all():
        return Review(
            False,
            ErrorCategory.ALL_NULL_RESULT,
            "Every value in the result is NULL. Check the column choices and join conditions.",
        )
    for column in frame.columns:
        series = frame[column]
        if not is_numeric_dtype(series):
            continue
        if COUNT_COLUMN.search(str(column)) and (series < 0).any():
            return Review(
                False,
                ErrorCategory.IMPOSSIBLE_VALUE,
                f"Column {column!r} looks like a count but has negative values.",
            )
        if PERCENT_COLUMN.search(str(column)) and (series.abs() > 100).any():
            return Review(
                False,
                ErrorCategory.IMPOSSIBLE_VALUE,
                f"Column {column!r} looks like a percentage but has values above 100.",
            )
    sizes = referenced_tables(code, profile)
    if " join " in f" {code.lower()} " and sizes and len(frame) > max(sizes):
        return Review(
            False,
            ErrorCategory.ROW_EXPLOSION,
            f"The result has {len(frame):,} rows, more than any input table ({max(sizes):,}). "
            "A join is probably duplicating rows; check the join keys or aggregate first.",
        )
    if SINGLE_VALUE_QUESTION.search(step) and not GROUPING_WORDS.search(step) and len(frame) > 1:
        return Review(
            False,
            ErrorCategory.AGGREGATION_MISMATCH,
            f"The step asks for a single value but the result has {len(frame)} rows. "
            "Aggregate over the whole selection unless a breakdown was requested.",
        )
    return None


def llm_review(state: AgentState, context: AgentContext, strict: bool) -> Review:
    plan = state.get("plan") or [state["standalone_question"]]
    step = plan[state.get("step_index", 0)]
    execution = state["execution"]
    user = (
        f"Question: {state['standalone_question']}\n"
        f"Current step: {step}\n\n"
        f"Schema:\n{state.get('schema_context', '')}\n\n"
        f"Code:\n{state.get('code', '')}\n\n"
        f"Result ({execution.row_count} rows"
        f"{', truncated' if execution.truncated else ''}):\n{preview(execution.frame, 15)}"
    )
    system = REVIEW_PROMPT + (STRICT_NOTE if strict else "")
    response = context.llm.complete(messages(system, user), json_mode=True)
    try:
        payload = parse_json_object(response.text)
    except LLMOutputError:
        return Review(True)  # An unreadable review should not block a working result.
    if payload.get("ok", True) is not False:
        return Review(True)
    issue = str(payload.get("issue") or "").strip()
    fix = str(payload.get("fix") or "").strip()
    return Review(
        False, ErrorCategory.SEMANTIC_MISMATCH, " ".join(part for part in (issue, fix) if part)
    )


def record_step(state: AgentState, warning: str | None = None) -> dict[str, Any]:
    """Store the current step's result and move on to the next step (or finish)."""
    plan = state.get("plan") or [state["standalone_question"]]
    index = state.get("step_index", 0)
    execution = state["execution"]
    result = StepResult(
        step=index,
        description=plan[index],
        language=state.get("language", "sql"),
        code=state.get("code", ""),
        frame=execution.frame,
        ok=execution.ok,
        truncated=execution.truncated,
        error=execution.error,
        warning=warning,
    )
    finished = index + 1 >= len(plan)
    return {
        "step_results": [*state.get("step_results", []), result],
        "step_index": index + 1,
        "retries": 0,
        "review_retries": 0,
        "feedback": "",
        "verdict": "done" if finished else "next",
    }


def review_result(state: AgentState, context: AgentContext) -> dict[str, Any]:
    execution = state["execution"]
    retries = state.get("retries", 0)
    plan = state.get("plan") or [state["standalone_question"]]
    index = state.get("step_index", 0)
    step_attempts = [attempt for attempt in state.get("attempts", []) if attempt.step == index]
    previous_empty = (
        len(step_attempts) >= 2 and step_attempts[-2].ok and step_attempts[-2].row_count == 0
    )

    review = check_execution(execution, context.profile)
    if review is None:
        review = check_sanity(
            plan[index], execution.frame, state.get("code", ""), context.profile, previous_empty
        )
    if review is None and context.config.llm_review:
        review = llm_review(state, context, strict=False)

    update: dict[str, Any] = {}
    if review is None or review.ok:
        score = None
        if context.confidence_model is not None:
            score = context.confidence_model.score(summarise_run(state))
        low_confidence = score is not None and score < context.config.confidence_threshold
        if low_confidence and context.config.llm_review and not state.get("deep_review_done"):
            update["deep_review_done"] = True
            review = llm_review(state, context, strict=True)
        if review is None or review.ok:
            accepted = record_step(state)
            detail = "accepted" if score is None else f"accepted (confidence {score:.2f})"
            return {**update, **accepted, "trace_detail": detail}

    categories = [*state.get("error_categories", []), str(review.category)]
    subjective = review.category is ErrorCategory.SEMANTIC_MISMATCH
    review_retries = state.get("review_retries", 0)
    # Execution errors and rule violations are objective, so they get the full retry
    # budget. An LLM reviewer's objection is an opinion: it gets one retry per step, after
    # which a working result is kept with a warning instead of looping on vague advice.
    out_of_review_retries = subjective and review_retries >= context.config.max_review_retries
    if retries >= context.config.max_retries or out_of_review_retries:
        if execution.ok:
            warning = f"The verifier still had doubts: {clip(review.feedback, 200)}"
            return {
                **update,
                **record_step(state, warning),
                "error_categories": categories,
                "trace_detail": f"accepted with warning ({review.category})",
            }
        return {
            **update,
            "verdict": "give_up",
            "failed": True,
            "error_categories": categories,
            "feedback": review.feedback,
            "trace_detail": f"gave up ({review.category})",
        }
    return {
        **update,
        "verdict": "retry",
        "retries": retries + 1,
        "review_retries": review_retries + int(subjective),
        "total_retries": state.get("total_retries", 0) + 1,
        "feedback": review.feedback,
        "error_categories": categories,
        "trace_detail": f"retry ({review.category}): {clip(review.feedback, 160)}",
    }


def accept_without_review(state: AgentState, context: AgentContext) -> dict[str, Any]:
    """Stand-in for the critic in the 'no critic' ablation: keep whatever came back."""
    execution = state["execution"]
    if not execution.ok:
        return {
            "verdict": "give_up",
            "failed": True,
            "error_categories": [*state.get("error_categories", []), str(execution.category)],
            "feedback": execution.error or "",
            "trace_detail": f"failed ({execution.category})",
        }
    return {**record_step(state), "trace_detail": "accepted without review"}
