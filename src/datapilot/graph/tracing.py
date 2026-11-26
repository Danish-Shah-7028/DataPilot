import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from datapilot.graph.context import AgentContext
from datapilot.graph.state import AgentState, StepLog
from datapilot.llm.provider import ChatMessage, ChatModel, LLMResponse

NodeFunction = Callable[[AgentState, AgentContext], dict[str, Any]]


@dataclass
class MeteredChatModel:
    """Counts calls and tokens so each graph step can be logged with its own cost."""

    inner: ChatModel
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cached_calls: int = 0
    reported_ms: float = 0.0
    waited_ms: float = 0.0

    @property
    def model(self) -> str:
        return self.inner.model

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        response = self.inner.complete(messages, json_mode=json_mode, temperature=temperature)
        self.waited_ms += (time.perf_counter() - started) * 1000
        self.reported_ms += response.latency_ms
        self.calls += 1
        self.cached_calls += int(response.cached)
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        return response


def traced(name: str, node: NodeFunction, context: AgentContext) -> Callable[[AgentState], dict]:
    def run(state: AgentState) -> dict[str, Any]:
        meter = MeteredChatModel(context.llm)
        started = time.perf_counter()
        update = node(state, replace(context, llm=meter))
        elapsed_ms = (time.perf_counter() - started) * 1000
        # A cache hit returns instantly but carries the latency of the original call;
        # counting that instead keeps re-runs of the evaluation comparable.
        latency_ms = elapsed_ms - meter.waited_ms + meter.reported_ms
        log = StepLog(
            node=name,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=meter.prompt_tokens,
            completion_tokens=meter.completion_tokens,
            llm_calls=meter.calls,
            cached_calls=meter.cached_calls,
            retry=int(state.get("retries", 0)),
            detail=str(update.pop("trace_detail", "")),
        )
        return {**update, "trace": [log]}

    return run


def summarise_run(state: AgentState) -> dict[str, Any]:
    """Flatten a finished (or in-progress) run into one JSON-friendly log record."""
    attempts = state.get("attempts", [])
    trace = state.get("trace", [])
    frame = state.get("final_frame")
    execution = state.get("execution")
    if frame is None and execution is not None and execution.ok:
        frame = execution.frame
    code = state.get("code") or (attempts[-1].code if attempts else "")
    return {
        "question": state.get("question", ""),
        "standalone_question": state.get("standalone_question", ""),
        "route": state.get("route"),
        "language": state.get("language"),
        "code": code,
        "success": not state.get("failed", False) and state.get("route") != "out_of_scope",
        "retries": int(state.get("total_retries", 0)),
        "attempts": len(attempts),
        "first_attempt_ok": attempts[0].ok if attempts else True,
        "error_categories": list(state.get("error_categories", [])),
        "result_rows": 0 if frame is None else len(frame),
        "result_columns": 0 if frame is None else len(frame.columns),
        "plan_steps": max(1, len(state.get("plan", []))),
        "tables_retrieved": len(state.get("retrieved_tables", [])),
        "llm_calls": sum(log.llm_calls for log in trace),
        "cached_calls": sum(log.cached_calls for log in trace),
        "prompt_tokens": sum(log.prompt_tokens for log in trace),
        "completion_tokens": sum(log.completion_tokens for log in trace),
        "latency_ms": round(sum(log.latency_ms for log in trace), 2),
        "confidence": state.get("confidence"),
        "nodes": [log.node for log in trace],
        "answer": state.get("answer", ""),
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
