from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datapilot.config import Settings, get_settings
from datapilot.data.loader import Workspace, load_files
from datapilot.data.schema import SchemaProfile, profile_database
from datapilot.graph.context import AgentConfig, AgentContext
from datapilot.graph.state import AgentState, Turn
from datapilot.graph.tracing import append_jsonl, summarise_run
from datapilot.graph.workflow import build_workflow, make_turn, run_question
from datapilot.llm.provider import ChatModel, build_chat_model
from datapilot.ml.confidence_model import load_confidence_model
from datapilot.rag.schema_index import (
    Embedder,
    SchemaIndex,
    build_schema_index,
    embedder_for,
)


@dataclass
class AnalysisSession:
    """One loaded dataset plus the conversation about it."""

    workspace: Workspace
    profile: SchemaProfile
    index: SchemaIndex
    context: AgentContext
    workflow: Any
    history: list[Turn] = field(default_factory=list)
    log_path: Path | None = None

    def ask(self, question: str) -> AgentState:
        state = run_question(self.workflow, question, self.history)
        self.history.append(make_turn(state))
        if self.log_path is not None:
            append_jsonl(self.log_path, summarise_run(state))
        return state

    def reset(self) -> None:
        self.history.clear()


def open_session(
    paths: Sequence[Path],
    workdir: Path,
    *,
    llm: ChatModel | None = None,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
    config: AgentConfig | None = None,
) -> AnalysisSession:
    settings = settings or get_settings()
    workspace = load_files(paths, workdir / "workspace.duckdb")
    renamed = {table.name: table.renamed_columns for table in workspace.tables}
    profile = profile_database(workspace.db_path, renamed)
    index = build_schema_index(profile, embedder or embedder_for(settings.embedding_model))
    context = AgentContext(
        llm=llm or build_chat_model(settings),
        db_path=workspace.db_path,
        profile=profile,
        index=index,
        config=config or AgentConfig.from_settings(settings),
        confidence_model=load_confidence_model(settings.confidence_model_path),
    )
    return AnalysisSession(
        workspace=workspace,
        profile=profile,
        index=index,
        context=context,
        workflow=build_workflow(context),
        log_path=settings.log_dir / "runs.jsonl",
    )
