import operator
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

import pandas as pd

from datapilot.ml.pipelines import MLReport
from datapilot.tools.charts import ChartSpec
from datapilot.tools.execution import ExecutionResult

Route = Literal["sql", "pandas", "ml", "out_of_scope"]
Language = Literal["sql", "python"]
Verdict = Literal["retry", "next", "done", "give_up"]
ROUTES: tuple[Route, ...] = ("sql", "pandas", "ml", "out_of_scope")


class Turn(TypedDict):
    """What the agents remember about an earlier question in the conversation."""

    question: str
    standalone_question: str
    route: str
    code: str
    answer: str


@dataclass(frozen=True)
class Attempt:
    step: int
    language: Language
    code: str
    ok: bool
    error: str | None
    category: str | None
    row_count: int


@dataclass(frozen=True)
class StepResult:
    step: int
    description: str
    language: str
    code: str
    frame: pd.DataFrame | None
    ok: bool
    truncated: bool = False
    error: str | None = None
    warning: str | None = None


@dataclass(frozen=True)
class StepLog:
    node: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    cached_calls: int = 0
    retry: int = 0
    detail: str = ""


class AgentState(TypedDict, total=False):
    question: str
    history: list[Turn]

    route: Route
    route_reason: str
    is_complex: bool
    standalone_question: str

    plan: list[str]
    step_index: int
    schema_context: str
    retrieved_tables: list[str]

    language: Language
    code: str
    explanation: str
    execution: ExecutionResult | None
    attempts: list[Attempt]
    step_results: list[StepResult]

    retries: int
    review_retries: int
    total_retries: int
    feedback: str
    verdict: Verdict
    error_categories: list[str]
    deep_review_done: bool

    ml_report: MLReport | None

    answer: str
    chart: ChartSpec | None
    final_frame: pd.DataFrame | None
    confidence: float | None
    failed: bool

    trace: Annotated[list[StepLog], operator.add]
