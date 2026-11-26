from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from datapilot.data.schema import SchemaProfile
from datapilot.graph.context import AgentContext
from datapilot.graph.state import AgentState
from datapilot.rag.schema_index import SchemaIndex, tokenize

LEXICAL_BOOST = 0.15
TABLE_MARGIN = 0.2


@dataclass(frozen=True)
class RetrievedSchema:
    selection: dict[str, set[str] | None]
    text: str
    used_full_schema: bool

    @property
    def tables(self) -> list[str]:
        return list(self.selection)


def is_key(column: str) -> bool:
    lowered = column.lower()
    return lowered == "id" or lowered.endswith("id")


def lexical_hits(profile: SchemaProfile, question: str) -> set[tuple[str, str | None]]:
    """Tables/columns whose names or sample values appear literally in the question."""
    words = set(tokenize(question))
    lowered = question.lower()
    hits: set[tuple[str, str | None]] = set()
    for table in profile.tables:
        if set(tokenize(table.name)) & words:
            hits.add((table.name, None))
        for column in table.columns:
            name_tokens = set(tokenize(column.name)) - {"id"}
            if name_tokens and name_tokens <= words:
                hits.add((table.name, column.name))
            if any(len(value) >= 3 and value.lower() in lowered for value in column.sample_values):
                hits.add((table.name, column.name))
    return hits


def bridge_tables(profile: SchemaProfile, chosen: list[str]) -> list[str]:
    """Add tables that link two chosen tables through shared key columns."""
    if len(chosen) < 2:
        return chosen
    keys = {
        table.name: {column.name for column in table.columns if is_key(column.name)}
        for table in profile.tables
    }
    extra = []
    for table in profile.tables:
        if table.name in chosen:
            continue
        linked = [other for other in chosen if keys[table.name] & keys[other]]
        if len(linked) >= 2:
            extra.append(table.name)
    return chosen + extra


def retrieve_schema(
    index: SchemaIndex,
    question: str,
    *,
    full_schema_max_columns: int = 40,
    max_tables: int = 4,
    max_columns_per_table: int = 15,
) -> RetrievedSchema:
    profile = index.profile
    if profile.column_count <= full_schema_max_columns:
        # Small schemas fit comfortably in the prompt, and dropping a column there
        # would only cost accuracy.
        return RetrievedSchema(dict.fromkeys(profile.table_names), profile.render(), True)

    hits = lexical_hits(profile, question)
    table_scores: dict[str, float] = defaultdict(float)
    column_scores: dict[tuple[str, str], float] = {}
    for document, score in index.search(question):
        boosted = score + (LEXICAL_BOOST if (document.table, document.column) in hits else 0.0)
        table_scores[document.table] = max(table_scores[document.table], boosted)
        if document.column is not None:
            column_scores[(document.table, document.column)] = boosted

    ranked = sorted(table_scores, key=table_scores.get, reverse=True)
    best = table_scores[ranked[0]]
    chosen = [name for name in ranked if table_scores[name] >= best - TABLE_MARGIN][:max_tables]
    chosen = bridge_tables(profile, chosen)

    selection: dict[str, set[str] | None] = {}
    for name in chosen:
        table = profile.table(name)
        if table is None or len(table.columns) <= max_columns_per_table:
            selection[name] = None
            continue
        ordered = sorted(
            table.column_names,
            key=lambda column, table_name=name: column_scores.get((table_name, column), 0.0),
            reverse=True,
        )
        keys = {column for column in table.column_names if is_key(column)}
        selection[name] = set(ordered[:max_columns_per_table]) | keys
    return RetrievedSchema(selection, profile.render(selection), False)


def retrieve_schema_node(state: AgentState, context: AgentContext) -> dict[str, Any]:
    question = state.get("standalone_question") or state["question"]
    if context.index is None:
        retrieved = RetrievedSchema(
            dict.fromkeys(context.profile.table_names), context.profile.render(), True
        )
    else:
        retrieved = retrieve_schema(
            context.index,
            question,
            full_schema_max_columns=context.config.full_schema_max_columns,
        )
    detail = "full schema" if retrieved.used_full_schema else ", ".join(retrieved.tables)
    return {
        "schema_context": retrieved.text,
        "retrieved_tables": retrieved.tables,
        "trace_detail": detail,
    }
