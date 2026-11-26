from typing import Any

from datapilot.agents.formatting import messages
from datapilot.graph.context import AgentContext
from datapilot.graph.state import AgentState
from datapilot.llm.json_output import LLMOutputError, parse_json_object

MAX_STEPS = 4

PLANNER_PROMPT = """You are the planner of DataPilot, an AI data analyst. Break the \
question into a short ordered list of analysis steps (2 to 4).

- Each step must be answerable with one {language} over the tables below.
- Later steps can use facts found by earlier steps (for example the region found in step 1).
- The final step must produce the table that best answers the question.
- Name the tables and columns each step needs.
- Do not add steps for drawing charts or writing the final answer.

Reply with JSON only: {{"steps": ["...", "..."]}}"""


def plan_steps(state: AgentState, context: AgentContext) -> dict[str, Any]:
    question = state["standalone_question"]
    language = "pandas computation" if state["route"] == "pandas" else "SQL query"
    user = f"Schema:\n{state['schema_context']}\n\nQuestion: {question}"
    response = context.llm.complete(
        messages(PLANNER_PROMPT.format(language=language), user), json_mode=True
    )
    try:
        steps = parse_json_object(response.text).get("steps") or []
    except LLMOutputError:
        steps = []
    steps = [str(step).strip() for step in steps if str(step).strip()][:MAX_STEPS]
    if not steps:
        steps = [question]
    return {"plan": steps, "step_index": 0, "trace_detail": f"{len(steps)} steps"}
