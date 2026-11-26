from collections.abc import Sequence
from typing import Any

from langgraph.graph import END, START, StateGraph

from datapilot.agents.coder import write_code
from datapilot.agents.critic import accept_without_review, review_result
from datapilot.agents.executor import execute_code
from datapilot.agents.ml_analyst import analyse_prediction
from datapilot.agents.planner import plan_steps
from datapilot.agents.reporter import write_report
from datapilot.agents.router import route_question
from datapilot.agents.schema_retriever import retrieve_schema_node
from datapilot.graph.context import AgentContext
from datapilot.graph.state import AgentState, Turn
from datapilot.graph.tracing import traced

# Worst case: 4 plan steps x 4 attempts x (coder, executor, critic) plus the fixed nodes.
RECURSION_LIMIT = 80


def after_router(state: AgentState) -> str:
    return "reporter" if state["route"] == "out_of_scope" else "schema_retriever"


def after_retrieval(state: AgentState) -> str:
    if state["route"] == "ml":
        return "ml_analyst"
    return "planner" if state.get("is_complex") else "coder"


def after_review(state: AgentState) -> str:
    return "coder" if state.get("verdict") in ("retry", "next") else "reporter"


def build_workflow(context: AgentContext) -> Any:
    """Compile the agent graph. With use_critic=False the result is accepted as-is."""
    review_name = "critic" if context.config.use_critic else "collect"
    review_node = review_result if context.config.use_critic else accept_without_review
    nodes = {
        "router": route_question,
        "schema_retriever": retrieve_schema_node,
        "planner": plan_steps,
        "coder": write_code,
        "executor": execute_code,
        review_name: review_node,
        "ml_analyst": analyse_prediction,
        "reporter": write_report,
    }
    graph = StateGraph(AgentState)
    for name, node in nodes.items():
        graph.add_node(name, traced(name, node, context))

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", after_router, ["schema_retriever", "reporter"])
    graph.add_conditional_edges(
        "schema_retriever", after_retrieval, ["ml_analyst", "planner", "coder"]
    )
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "executor")
    graph.add_edge("executor", review_name)
    graph.add_conditional_edges(review_name, after_review, ["coder", "reporter"])
    graph.add_edge("ml_analyst", "reporter")
    graph.add_edge("reporter", END)
    return graph.compile()


def run_question(workflow: Any, question: str, history: Sequence[Turn] = ()) -> AgentState:
    initial: AgentState = {"question": question, "history": list(history), "trace": []}
    return workflow.invoke(initial, config={"recursion_limit": RECURSION_LIMIT})


def make_turn(state: AgentState) -> Turn:
    return Turn(
        question=state.get("question", ""),
        standalone_question=state.get("standalone_question", ""),
        route=str(state.get("route", "")),
        code=state.get("code", "") if not state.get("failed") else "",
        answer=state.get("answer", ""),
    )
