"""Run the three-variant text-to-SQL comparison on the Spider subset.

Variants:
- baseline: one LLM call with the full schema, executed once.
- agent_no_critic: the LangGraph agent with the critic replaced by a pass-through.
- agent_full: the agent with the critic, sanity checks and up to three retries.

Every answer is written to reports/eval_runs/<variant>.jsonl as soon as it is scored, so
a run interrupted by rate limits can simply be started again and resumes where it stopped.
"""

import argparse
import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datapilot.baseline import baseline_sql
from datapilot.config import get_settings
from datapilot.data.loader import load_files
from datapilot.data.schema import SchemaProfile, profile_database
from datapilot.eval.compare import execute_gold, frame_rows, is_ordered, results_match
from datapilot.graph.context import AgentConfig, AgentContext
from datapilot.graph.tracing import append_jsonl, summarise_run
from datapilot.graph.workflow import build_workflow, run_question
from datapilot.llm.cache import CachedChatModel
from datapilot.llm.provider import ChatModel, LLMError, build_chat_model
from datapilot.rag.schema_index import (
    Embedder,
    HashingEmbedder,
    SchemaIndex,
    SentenceTransformerEmbedder,
    build_schema_index,
)
from datapilot.tools.sql_executor import run_sql

VARIANTS = ("baseline", "agent_no_critic", "agent_full")
EVAL_ROW_LIMIT = 100_000
EVAL_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str
    db_id: str
    question: str
    gold_sql: str


@dataclass(frozen=True)
class PreparedDatabase:
    sqlite_path: Path
    db_path: Path
    profile: SchemaProfile
    index: SchemaIndex


def load_items(path: Path, limit: int | None = None) -> list[BenchmarkItem]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    items = [
        BenchmarkItem(row["id"], row["db_id"], row["question"], row["gold_sql"]) for row in rows
    ]
    return items[:limit] if limit else items


def prepare_database(
    db_id: str, databases_dir: Path, work_dir: Path, embedder: Embedder
) -> PreparedDatabase:
    sqlite_path = databases_dir / db_id / f"{db_id}.sqlite"
    workspace = load_files([sqlite_path], work_dir / f"{db_id}.duckdb")
    renamed = {table.name: table.renamed_columns for table in workspace.tables}
    profile = profile_database(workspace.db_path, renamed)
    return PreparedDatabase(
        sqlite_path, workspace.db_path, profile, build_schema_index(profile, embedder)
    )


def run_baseline(
    item: BenchmarkItem, database: PreparedDatabase, llm: ChatModel
) -> tuple[dict, Any]:
    code, response = baseline_sql(item.question, database.profile, llm)
    execution = run_sql(
        database.db_path, code, row_limit=EVAL_ROW_LIMIT, timeout_seconds=EVAL_TIMEOUT_SECONDS
    )
    record = {
        "code": code,
        "executed": execution.ok,
        "error": execution.error,
        "error_categories": [] if execution.ok else [str(execution.category)],
        "retries": 0,
        "attempts": 1,
        "first_attempt_ok": execution.ok,
        "llm_calls": 1,
        "cached_calls": int(response.cached),
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "latency_ms": round(response.latency_ms + execution.elapsed_ms, 2),
        "result_rows": execution.row_count,
        "result_columns": 0 if execution.frame is None else len(execution.frame.columns),
    }
    return record, execution.frame


def run_agent(
    item: BenchmarkItem, database: PreparedDatabase, llm: ChatModel, use_critic: bool
) -> tuple[dict, Any]:
    context = AgentContext(
        llm=llm,
        db_path=database.db_path,
        profile=database.profile,
        index=database.index,
        # The benchmark is pure text-to-SQL, so the router may not send questions elsewhere.
        config=AgentConfig(
            row_limit=EVAL_ROW_LIMIT,
            sql_timeout_seconds=EVAL_TIMEOUT_SECONDS,
            use_critic=use_critic,
            allowed_routes=("sql",),
        ),
    )
    state = run_question(build_workflow(context), item.question)
    frame = state.get("final_frame")
    attempts = state.get("attempts", [])
    record = summarise_run(state)
    record.pop("answer", None)
    record["executed"] = frame is not None
    record["error"] = None if frame is not None else (attempts[-1].error if attempts else None)
    return record, frame


def evaluate(
    items: Sequence[BenchmarkItem],
    variants: Sequence[str],
    *,
    databases_dir: Path,
    out_dir: Path,
    work_dir: Path,
    llm: ChatModel,
    embedder: Embedder,
    fresh: bool = False,
    log: Callable[[str], None] = print,
) -> bool:
    """Score every item for every variant. Returns False if the LLM provider gave up."""
    out_dir.mkdir(parents=True, exist_ok=True)
    databases: dict[str, PreparedDatabase] = {}
    gold_results: dict[str, list[tuple] | None] = {}

    for variant in variants:
        out_path = out_dir / f"{variant}.jsonl"
        if fresh:
            out_path.unlink(missing_ok=True)
        done = set()
        if out_path.exists():
            with out_path.open(encoding="utf-8") as handle:
                done = {json.loads(line)["id"] for line in handle if line.strip()}

        for position, item in enumerate(items, start=1):
            if item.item_id in done:
                continue
            if item.db_id not in databases:
                databases[item.db_id] = prepare_database(
                    item.db_id, databases_dir, work_dir, embedder
                )
            database = databases[item.db_id]
            if item.item_id not in gold_results:
                try:
                    gold_results[item.item_id] = execute_gold(database.sqlite_path, item.gold_sql)
                except sqlite3.Error as error:
                    log(f"skipping {item.item_id}: gold SQL failed ({error})")
                    gold_results[item.item_id] = None
            gold = gold_results[item.item_id]
            if gold is None:
                continue

            try:
                if variant == "baseline":
                    record, frame = run_baseline(item, database, llm)
                else:
                    record, frame = run_agent(item, database, llm, variant == "agent_full")
            except LLMError as error:
                log(f"stopping: {error}\nRe-run the same command to resume.")
                return False

            correct = frame is not None and results_match(
                gold, frame_rows(frame), is_ordered(item.gold_sql)
            )
            append_jsonl(
                out_path,
                {
                    "id": item.item_id,
                    "db_id": item.db_id,
                    "variant": variant,
                    "model": llm.model,
                    "question": item.question,
                    "gold_sql": item.gold_sql,
                    **record,
                    "correct": correct,
                },
            )
            log(
                f"[{position:>3}/{len(items)}] {variant:<16} {item.db_id:<28} "
                f"{'correct' if correct else 'wrong':<8} retries={record['retries']} "
                f"{record['latency_ms'] / 1000:.1f}s"
            )
    return True


def check_harness(
    items: Sequence[BenchmarkItem], databases_dir: Path, work_dir: Path
) -> dict[str, list[BenchmarkItem]]:
    """Replay the gold SQL on the DuckDB copies, without any LLM.

    Gold queries are written for SQLite, so some use syntax DuckDB rejects (double-quoted
    strings, bare columns in GROUP BY); those do not affect scoring because gold always
    runs on SQLite. The interesting bucket is "differs": valid DuckDB SQL whose result
    does not match, which points at data conversion or comparison problems.
    """
    buckets: dict[str, list[BenchmarkItem]] = {"reproduces": [], "differs": [], "sqlite_only": []}
    db_paths: dict[str, Path] = {}
    for item in items:
        sqlite_path = databases_dir / item.db_id / f"{item.db_id}.sqlite"
        if item.db_id not in db_paths:
            db_paths[item.db_id] = load_files(
                [sqlite_path], work_dir / f"{item.db_id}.duckdb"
            ).db_path
        gold = execute_gold(sqlite_path, item.gold_sql)
        result = run_sql(db_paths[item.db_id], item.gold_sql, row_limit=EVAL_ROW_LIMIT)
        if not result.ok:
            buckets["sqlite_only"].append(item)
        elif results_match(gold, frame_rows(result.frame), is_ordered(item.gold_sql)):
            buckets["reproduces"].append(item)
        else:
            buckets["differs"].append(item)
    return buckets


def harness_report(items: Sequence[BenchmarkItem], buckets: dict[str, list[BenchmarkItem]]) -> str:
    runnable = len(buckets["reproduces"]) + len(buckets["differs"])
    lines = [
        "# Evaluation harness check",
        "",
        f"All {len(items)} gold queries of the subset were replayed on the DuckDB copies of "
        "their databases and compared with the result on the original SQLite file, using "
        "the same comparison as the evaluation. No LLM is involved.",
        "",
        f"- Valid DuckDB SQL whose result matches SQLite: **{len(buckets['reproduces'])}** "
        f"of {runnable}",
        f"- Valid DuckDB SQL whose result differs: **{len(buckets['differs'])}**",
        f"- SQLite-only syntax that DuckDB rejects: {len(buckets['sqlite_only'])} "
        "(double-quoted string literals, bare columns in GROUP BY). These do not affect "
        "scoring because gold queries always run on SQLite.",
        "",
    ]
    if buckets["differs"]:
        lines += ["Queries whose results differ:", ""]
        lines += [
            f"- `{item.item_id}` ({item.db_id}): `{item.gold_sql}`" for item in buckets["differs"]
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DataPilot on the Spider subset.")
    parser.add_argument(
        "--subset", type=Path, default=Path("data/benchmark/spider_dev_subset.jsonl")
    )
    parser.add_argument("--databases", type=Path, default=Path("data/benchmark/databases"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/eval_runs"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--fresh", action="store_true", help="discard earlier results")
    parser.add_argument("--work-dir", type=Path, help="where the DuckDB copies are built")
    parser.add_argument(
        "--embedder",
        choices=["minilm", "hashing"],
        default="minilm",
        help="schema embeddings (hashing needs no model download)",
    )
    parser.add_argument(
        "--check-harness",
        action="store_true",
        help="run the gold SQL on the DuckDB copies (no LLM) and report mismatches",
    )
    args = parser.parse_args()

    settings = get_settings()
    if not args.subset.exists():
        raise SystemExit(f"{args.subset} not found; run datapilot-prepare-benchmark first.")
    if args.check_harness:
        items = load_items(args.subset, args.limit)
        buckets = check_harness(items, args.databases, settings.cache_dir / "eval_duckdb")
        report = harness_report(items, buckets)
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        (settings.reports_dir / "harness_check.md").write_text(report, encoding="utf-8")
        print(report)
        return
    # Always cache during evaluation: re-runs cost nothing and give identical answers.
    inner = build_chat_model(settings.model_copy(update={"llm_cache_enabled": False}))
    llm = CachedChatModel(inner=inner, path=settings.llm_cache_path)
    embedder = (
        HashingEmbedder()
        if args.embedder == "hashing"
        else SentenceTransformerEmbedder(settings.embedding_model)
    )
    finished = evaluate(
        load_items(args.subset, args.limit),
        args.variants,
        databases_dir=args.databases,
        out_dir=args.out_dir,
        work_dir=args.work_dir or settings.cache_dir / "eval_duckdb",
        llm=llm,
        embedder=embedder,
        fresh=args.fresh,
    )
    if finished:
        print("Done. Run datapilot-summarise to build the tables and charts.")
