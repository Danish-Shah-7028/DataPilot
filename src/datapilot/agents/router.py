from typing import Any

from datapilot.agents.formatting import format_history, messages
from datapilot.graph.context import AgentContext
from datapilot.graph.state import ROUTES, AgentState
from datapilot.llm.json_output import LLMOutputError, parse_json_object

ROUTER_PROMPT = """You are the router of DataPilot, an AI data analyst. Decide how the latest \
question about the user's dataset should be handled.

Routes:
- "sql": descriptive questions that SQL can answer: filters, counts, totals, averages, \
rankings, trends over time, comparisons between groups.
- "pandas": analysis that is awkward in SQL: correlations, distributions and outliers, \
statistical summaries across many columns, pivots and multi-stage reshaping.
- "ml": predictive questions that need a trained model: who or what is likely to churn, \
buy or default, predicting a value for records, or what drives an outcome.
- "out_of_scope": not answerable from the data, small talk, requests to change or delete \
data, or anything involving files, the operating system, the internet or harmful actions.

Set "complex" to true only when the question has several parts that need separate \
queries, for example "which region dropped the most and why".

If the question is a follow-up such as "now only for Europe" or "and per month?", rewrite \
it as a complete standalone question using the conversation. Otherwise repeat it unchanged.

Reply with JSON only:
{"route": "sql" | "pandas" | "ml" | "out_of_scope", "complex": true | false, \
"standalone_question": "...", "reason": "one short sentence"}"""


def route_question(state: AgentState, context: AgentContext) -> dict[str, Any]:
    question = state["question"]
    history = state.get("history", [])
    user = (
        f"Tables:\n{context.profile.summary()}\n\n"
        f"Conversation so far:\n{format_history(history, context.config.history_turns)}\n\n"
        f"Latest question: {question}"
    )
    response = context.llm.complete(messages(ROUTER_PROMPT, user), json_mode=True)
    try:
        payload = parse_json_object(response.text)
    except LLMOutputError:
        payload = {}

    route = payload.get("route")
    if route not in ROUTES:
        route = "sql"
    allowed = context.config.allowed_routes
    if allowed and route not in allowed:
        route = allowed[0]
    standalone = str(payload.get("standalone_question") or "").strip() or question
    return {
        "route": route,
        "is_complex": bool(payload.get("complex")) and route in ("sql", "pandas"),
        "standalone_question": standalone,
        "route_reason": str(payload.get("reason") or ""),
        "plan": [standalone],
        "step_index": 0,
        "retries": 0,
        "review_retries": 0,
        "total_retries": 0,
        "attempts": [],
        "step_results": [],
        "error_categories": [],
        "deep_review_done": False,
        "failed": False,
        "trace_detail": f"route={route} complex={bool(payload.get('complex'))}",
    }
