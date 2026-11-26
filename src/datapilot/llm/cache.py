import hashlib
import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

from datapilot.llm.provider import ChatMessage, ChatModel, LLMResponse


def cache_key(
    model: str,
    messages: Sequence[ChatMessage],
    json_mode: bool,
    temperature: float | None,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": list(messages),
            "json_mode": json_mode,
            "temperature": temperature,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CachedChatModel:
    """Wraps a chat model with a SQLite-backed reply cache.

    Cached replies keep the token counts and latency of the original call, so a re-run
    of the evaluation reports the same cost numbers as the first run.
    """

    inner: ChatModel
    path: Path

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
        key = cache_key(self.inner.model, messages, json_mode, temperature)
        stored = self.lookup(key)
        if stored is not None:
            return stored

        response = self.inner.complete(messages, json_mode=json_mode, temperature=temperature)
        self.store(key, response)
        return response

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        return connection

    def lookup(self, key: str) -> LLMResponse | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT payload FROM responses WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        payload["cached"] = True
        return LLMResponse(**payload)

    def store(self, key: str, response: LLMResponse) -> None:
        payload = asdict(response)
        payload.pop("cached")
        with closing(self.connect()) as connection, connection:
            connection.execute(
                "INSERT OR REPLACE INTO responses (key, payload) VALUES (?, ?)",
                (key, json.dumps(payload)),
            )
