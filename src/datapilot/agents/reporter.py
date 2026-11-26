from typing import Any

from datapilot.agents.formatting import clip, format_step_results, messages
from datapilot.graph.context import AgentContext
from datapilot.graph.state import AgentState
from datapilot.graph.tracing import summarise_run
from datapilot.llm.json_output import LLMOutputError, parse_json_object
from datapilot.tools.charts import suggest_chart, validate_chart

REPORTER_PROMPT = """You are the reporter of DataPilot, an AI data analyst. Write the \
answer to the user's question from the results below.

- Lead with the direct answer, in 2 to 5 sentences of plain language.
- Quote concrete numbers from the results. Never invent numbers that are not there.
- Do not add currency symbols or units unless the data or the question states them.
- Write rates and shares between 0 and 1 as percentages (0.168 becomes 16.8%), and round \
other numbers sensibly.
- Refer to columns in everyday words (say "revenue", not "total_revenue").
- If the question asks why, explain using the evidence in the step results, and say \
plainly when the data cannot fully explain it.
- If a result is truncated, empty or carries a warning, mention it briefly.
- Suggest a chart for the final result table only if it makes the answer clearer.

Reply with JSON only:
{"answer": "...", "chart": {"type": "bar" | "line" | "scatter" | "pie" | "histogram" | \
"none", "x": "column", "y": "column or null", "color": "column or null", "title": "..."}}"""

ML_NOTE = """

This was a predictive question answered with trained models. Say which model did best \
and how good it is in everyday words (for example what ROC-AUC or RMSE means here), name \
the strongest drivers, and repeat any warnings about leakage, imbalance or small samples. \
Make clear that predictions are probabilities, not certainties. Do not suggest a chart."""

OUT_OF_SCOPE = (
    "I can only answer questions about the loaded data, and I never change data or "
    "access files, the system or the internet."
)


def write_report(state: AgentState, context: AgentContext) -> dict[str, Any]:
    route = state.get("route")
    if route == "out_of_scope":
        reason = state.get("route_reason", "").strip().rstrip(".")
        because = f" ({reason[0].lower()}{reason[1:]})" if reason else ""
        return {
            "answer": f"I can't help with that request{because}. {OUT_OF_SCOPE}",
            "chart": None,
            "final_frame": None,
            "confidence": None,
            "trace_detail": "out of scope",
        }

    results = [result for result in state.get("step_results", []) if result.ok]
    if state.get("failed") or not results:
        attempts = state.get("attempts", [])
        problem = state.get("feedback") or (attempts[-1].error if attempts else "") or "unknown"
        retries = state.get("total_retries", 0)
        tried = f" after {retries} {'retry' if retries == 1 else 'retries'}" if retries else ""
        return {
            "answer": (
                f"I couldn't produce a reliable answer{tried}. "
                f"The last problem was: {clip(str(problem), 400)}"
            ),
            "chart": None,
            "final_frame": None,
            "failed": True,
            "confidence": None,
            "trace_detail": "reported failure",
        }

    final_frame = results[-1].frame
    report = state.get("ml_report")
    parts = [f"Question: {state.get('standalone_question') or state['question']}"]
    if report is not None:
        parts.append(f"Model results:\n{report.summary_text()}")
    parts.append(f"Step results:\n{format_step_results(results, max_rows=20)}")
    system = REPORTER_PROMPT + (ML_NOTE if report is not None else "")
    response = context.llm.complete(messages(system, "\n\n".join(parts)), json_mode=True)
    try:
        payload = parse_json_object(response.text)
        answer = str(payload.get("answer") or "").strip()
        raw_chart = payload.get("chart")
    except LLMOutputError:
        answer, raw_chart = response.text.strip(), None

    chart = None
    if report is None:
        wants_no_chart = isinstance(raw_chart, dict) and raw_chart.get("type") == "none"
        chart = validate_chart(raw_chart if isinstance(raw_chart, dict) else None, final_frame)
        if chart is None and not wants_no_chart:
            chart = suggest_chart(final_frame)

    confidence = None
    if context.confidence_model is not None and report is None:
        confidence = context.confidence_model.score(
            summarise_run({**state, "final_frame": final_frame})
        )
    return {
        "answer": answer or "Here is the result.",
        "chart": chart,
        "final_frame": final_frame,
        "confidence": confidence,
        "trace_detail": f"chart={chart.kind if chart else 'none'}",
    }
