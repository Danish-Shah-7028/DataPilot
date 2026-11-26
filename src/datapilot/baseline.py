"""Single-call text-to-SQL: the baseline the agent is compared against."""

from datapilot.data.schema import SchemaProfile
from datapilot.llm.json_output import LLMOutputError, parse_json_object, strip_code_fences
from datapilot.llm.provider import ChatModel, LLMResponse

BASELINE_PROMPT = """You are an expert data analyst. Write one DuckDB SQL query that \
answers the question using only the schema below. Return only the columns needed to \
answer, in the order the question asks for them. Only add LIMIT when the question asks \
for a specific number of results.

Reply with JSON only: {"code": "<the SQL query>"}"""


def baseline_sql(question: str, profile: SchemaProfile, llm: ChatModel) -> tuple[str, LLMResponse]:
    response = llm.complete(
        [
            {"role": "system", "content": BASELINE_PROMPT},
            {"role": "user", "content": f"Schema:\n{profile.render()}\n\nQuestion: {question}"},
        ],
        json_mode=True,
    )
    try:
        code = str(parse_json_object(response.text).get("code") or "")
    except LLMOutputError:
        code = response.text
    return strip_code_fences(code), response
