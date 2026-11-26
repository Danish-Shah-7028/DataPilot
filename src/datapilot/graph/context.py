from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import pandas as pd

from datapilot.config import Settings
from datapilot.data.loader import read_tables
from datapilot.data.schema import SchemaProfile
from datapilot.llm.provider import ChatModel
from datapilot.ml.confidence_model import ConfidenceModel
from datapilot.rag.schema_index import SchemaIndex


@dataclass(frozen=True)
class AgentConfig:
    row_limit: int = 5000
    sql_timeout_seconds: float = 15.0
    sandbox_timeout_seconds: float = 30.0
    sandbox_table_row_cap: int = 200_000
    ml_max_rows: int = 50_000
    max_retries: int = 3
    max_review_retries: int = 1
    llm_review: bool = True
    use_critic: bool = True
    full_schema_max_columns: int = 40
    confidence_threshold: float = 0.5
    history_turns: int = 3
    allowed_routes: tuple[str, ...] = ()

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            row_limit=settings.sql_row_limit,
            sql_timeout_seconds=settings.sql_timeout_seconds,
            sandbox_timeout_seconds=settings.sandbox_timeout_seconds,
            sandbox_table_row_cap=settings.sandbox_table_row_cap,
            ml_max_rows=settings.ml_max_rows,
            max_retries=settings.critic_max_retries,
            llm_review=settings.critic_llm_review,
            full_schema_max_columns=settings.rag_full_schema_max_columns,
            confidence_threshold=settings.confidence_threshold,
        )


@dataclass
class AgentContext:
    """Everything the nodes need besides the state: model, data, schema index and limits."""

    llm: ChatModel
    db_path: Path
    profile: SchemaProfile
    index: SchemaIndex | None = None
    config: AgentConfig = field(default_factory=AgentConfig)
    confidence_model: ConfidenceModel | None = None
    frame_cache: dict[str, pd.DataFrame] = field(default_factory=dict)

    def table_frames(self, names: Iterable[str]) -> dict[str, pd.DataFrame]:
        names = list(names)
        missing = [name for name in names if name not in self.frame_cache]
        if missing:
            self.frame_cache.update(
                read_tables(self.db_path, missing, self.config.sandbox_table_row_cap)
            )
        return {name: self.frame_cache[name] for name in names}
