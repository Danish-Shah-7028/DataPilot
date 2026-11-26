import json
import re
from typing import Any

FENCE_PATTERN = re.compile(r"```(?:[a-zA-Z0-9]+)?\s*(.*?)```", re.DOTALL)


class LLMOutputError(ValueError):
    """The model's reply could not be parsed into the expected structure."""


def strip_code_fences(text: str) -> str:
    match = FENCE_PATTERN.search(text)
    return match.group(1).strip() if match else text.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model reply, tolerating fences and stray prose."""
    # Plain JSON first: a fenced code block can legitimately sit inside a string value.
    for candidate in (text.strip(), strip_code_fences(text)):
        try:
            value = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMOutputError(f"No JSON object found in reply: {text[:200]!r}")
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as error:
            raise LLMOutputError(f"Malformed JSON in reply: {error}") from error

    if not isinstance(value, dict):
        raise LLMOutputError(f"Expected a JSON object, got: {text[:100]!r}")
    return value
