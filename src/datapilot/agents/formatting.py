from collections.abc import Sequence

from datapilot.graph.state import StepResult, Turn
from datapilot.llm.provider import ChatMessage
from datapilot.tools.execution import preview


def clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_history(history: Sequence[Turn], turns: int) -> str:
    if not history:
        return "(no earlier questions)"
    blocks = []
    for turn in history[-turns:]:
        block = f"User asked: {turn['question']}"
        if turn["code"]:
            block += f"\nCode used:\n{clip(turn['code'], 800)}"
        if turn["answer"]:
            block += f"\nAnswer given: {clip(turn['answer'], 300)}"
        blocks.append(block)
    return "\n\n".join(blocks)


def format_plan(plan: Sequence[str], current: int) -> str:
    lines = []
    for position, step in enumerate(plan):
        marker = "  <- current step" if position == current else ""
        lines.append(f"{position + 1}. {step}{marker}")
    return "\n".join(lines)


def format_step_results(results: Sequence[StepResult], max_rows: int = 10) -> str:
    blocks = []
    for result in results:
        body = preview(result.frame, max_rows) if result.ok else f"failed: {result.error}"
        notes = []
        if result.truncated:
            notes.append("result truncated by the row limit")
        if result.warning:
            notes.append(result.warning)
        note = f"\n(note: {'; '.join(notes)})" if notes else ""
        blocks.append(f"Step {result.step + 1}: {result.description}\nResult:\n{body}{note}")
    return "\n\n".join(blocks)


def messages(system: str, user: str) -> list[ChatMessage]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
