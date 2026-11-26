from typing import Any

import pandas as pd

from datapilot.agents.formatting import messages
from datapilot.graph.context import AgentContext
from datapilot.graph.state import AgentState, StepResult
from datapilot.llm.json_output import LLMOutputError, parse_json_object
from datapilot.ml.pipelines import (
    MLReport,
    MLSetup,
    MLSetupError,
    run_model_comparison,
    validate_setup,
)

RANKING_ROWS = 20

ML_PROMPT = """You are the machine learning analyst of DataPilot, an AI data analyst. The \
user asked a predictive question. Pick one table and propose a supervised learning setup.

- target: the column to predict (for churn questions, the churn flag).
- features: columns that would be known before the outcome happens. Leave out IDs, free \
text and anything that leaks the answer (recorded after, or derived from, the target).
- id_column: the column identifying each row (for example customer_id), or null.

Reply with JSON only:
{"table": "...", "target": "...", "features": ["..."], "id_column": "..." | null, \
"reason": "one short sentence"}"""


def describe_pipeline(table: str, setup: MLSetup, report: MLReport) -> str:
    dropped = ", ".join(f"{name} ({why})" for name, why in setup.dropped.items()) or "none"
    return "\n".join(
        [
            f"# table: {table}   target: {setup.target}   task: {setup.task}",
            f"# features: {', '.join(setup.features)}",
            f"# dropped: {dropped}",
            f"# 80/20 train/test split, 5-fold CV on train, best model by {report.primary_metric}",
            report.pipeline_description,
        ]
    )


def risk_ranking(frame: pd.DataFrame, setup: MLSetup, report: MLReport) -> pd.DataFrame:
    """Rows that have not (yet) hit the positive outcome, ordered by predicted risk."""
    scored = frame.loc[report.risk_scores.index].copy()
    scored["risk_score"] = report.risk_scores.round(3)
    scored = scored[scored[setup.target].astype(str) != report.positive_label]
    drivers = [name for name in report.feature_importance["feature"] if name in scored.columns][:3]
    identifier = [setup.id_column] if setup.id_column in scored.columns else []
    if not identifier:
        scored = scored.reset_index(names="row")
        identifier = ["row"]
    ranked = scored.sort_values("risk_score", ascending=False).head(RANKING_ROWS)
    return ranked[[*identifier, "risk_score", *drivers]].reset_index(drop=True)


def failure(message: str) -> dict[str, Any]:
    return {"failed": True, "feedback": message, "step_results": [], "trace_detail": message}


def analyse_prediction(state: AgentState, context: AgentContext) -> dict[str, Any]:
    question = state.get("standalone_question") or state["question"]
    user = f"Schema:\n{state['schema_context']}\n\nQuestion: {question}"
    response = context.llm.complete(messages(ML_PROMPT, user), json_mode=True)
    try:
        proposal = parse_json_object(response.text)
    except LLMOutputError:
        proposal = {}

    profile = context.profile
    table = profile.table(str(proposal.get("table") or ""))
    if table is None and len(profile.tables) == 1:
        table = profile.tables[0]
    if table is None:
        return failure("I could not tell which table holds the outcome to predict.")
    target = table.column(str(proposal.get("target") or ""))
    if target is None:
        return failure(f"I could not find a suitable target column in {table.name}.")
    features = [str(name) for name in proposal.get("features") or []]
    id_column = proposal.get("id_column")
    id_column = id_column if isinstance(id_column, str) and table.column(id_column) else None

    frame = context.table_frames([table.name])[table.name]
    try:
        setup = validate_setup(frame, target.name, features, id_column)
        report = run_model_comparison(frame, setup, max_rows=context.config.ml_max_rows)
    except (MLSetupError, ValueError) as error:
        return failure(f"Model training failed: {error}")

    code = describe_pipeline(table.name, setup, report)
    steps = [
        StepResult(
            step=0,
            description=f"Compare models predicting {setup.target} with 5-fold cross-validation",
            language="python",
            code=code,
            frame=report.comparison_frame(),
            ok=True,
        )
    ]
    if report.risk_scores is not None:
        steps.append(
            StepResult(
                step=1,
                description=(
                    f"Rank rows whose {setup.target} is not {report.positive_label} by the "
                    "out-of-fold predicted probability"
                ),
                language="python",
                code=code,
                frame=risk_ranking(frame, setup, report),
                ok=True,
            )
        )
    return {
        "ml_report": report,
        "step_results": steps,
        "plan": [step.description for step in steps],
        "language": "python",
        "code": code,
        "trace_detail": f"{report.best_model}, {report.primary_metric}="
        f"{report.test_metrics.get(report.primary_metric, float('nan')):.3f}",
    }
