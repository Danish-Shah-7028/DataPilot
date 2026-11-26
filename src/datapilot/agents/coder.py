from typing import Any

from datapilot.agents.formatting import (
    clip,
    format_history,
    format_plan,
    format_step_results,
    messages,
)
from datapilot.graph.context import AgentContext
from datapilot.graph.state import AgentState
from datapilot.llm.json_output import LLMOutputError, parse_json_object, strip_code_fences

SQL_PROMPT = """You are the SQL coder of DataPilot, an AI data analyst. Write one DuckDB \
SQL query for the current step.

Rules:
- Use only the tables and columns listed in the schema. Never invent columns.
- Write a single read-only SELECT statement; WITH clauses are fine. No comments.
- Return only the columns needed to answer, in the order the question asks for them.
- Use ILIKE for case-insensitive text matching.
- For dates use date_trunc, strftime or EXTRACT. If a date is stored as text, convert it \
with TRY_CAST or strptime first.
- Only add LIMIT when the question asks for a specific number of results.
- If earlier steps found specific values (a region, a date), use them directly.

Reply with JSON only: {"code": "<the SQL query>", "explanation": "<one short sentence>"}"""

PANDAS_PROMPT = """You are the pandas coder of DataPilot, an AI data analyst. Write Python \
code for the current step.

Environment:
- Each table is a pandas DataFrame, available as a variable named after the table and in \
the dict `tables`.
- pandas is imported as pd and numpy as np. You may also import math, statistics, \
datetime, re, collections or itertools.
- There is no file, network or system access. Names starting with an underscore are \
not allowed.

Rules:
- Assign the final answer to a variable named `result` (DataFrame, Series or single value).
- Use only the columns listed in the schema. Do not print or plot.
- Keep the code short and vectorised.

Reply with JSON only: {"code": "<python code>", "explanation": "<one short sentence>"}"""


def build_request(state: AgentState, context: AgentContext) -> str:
    plan = state.get("plan") or [state["standalone_question"]]
    step_index = state.get("step_index", 0)
    parts = [f"Schema:\n{state['schema_context']}"]

    history = state.get("history", [])
    if history:
        parts.append(
            "Earlier in this conversation (reuse this code when the question is a follow-up):\n"
            + format_history(history, context.config.history_turns)
        )

    parts.append(f"Question: {state['standalone_question']}")
    if len(plan) > 1:
        parts.append(f"Plan:\n{format_plan(plan, step_index)}")
        parts.append(f"Write code for step {step_index + 1}: {plan[step_index]}")
    finished = state.get("step_results", [])
    if finished:
        parts.append(
            "Results of earlier steps (for reference only: they are not tables you can "
            "query, so copy the values you need or recompute them in a WITH clause):\n"
            + format_step_results(finished)
        )

    feedback = state.get("feedback")
    attempts = [attempt for attempt in state.get("attempts", []) if attempt.step == step_index]
    if feedback and attempts:
        parts.append(
            "Your previous attempt for this step was rejected.\n"
            f"Previous code:\n{clip(attempts[-1].code, 1500)}\n"
            f"Problem: {feedback}\n"
            "Write a corrected version."
        )
    return "\n\n".join(parts)


def write_code(state: AgentState, context: AgentContext) -> dict[str, Any]:
    language = "python" if state["route"] == "pandas" else "sql"
    prompt = PANDAS_PROMPT if language == "python" else SQL_PROMPT
    response = context.llm.complete(messages(prompt, build_request(state, context)), json_mode=True)
    try:
        payload = parse_json_object(response.text)
        code = str(payload.get("code") or "")
        explanation = str(payload.get("explanation") or "")
    except LLMOutputError:
        code, explanation = response.text, ""
    code = strip_code_fences(code)
    return {
        "language": language,
        "code": code,
        "explanation": explanation,
        "trace_detail": f"step {state.get('step_index', 0) + 1}, {len(code)} chars",
    }
