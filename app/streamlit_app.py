"""DataPilot web app. Run with: streamlit run app/streamlit_app.py"""

import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import SecretStr

from datapilot.config import Settings, get_settings
from datapilot.data.loader import SUPPORTED_SUFFIXES
from datapilot.graph.state import AgentState
from datapilot.llm.provider import LLMError, build_chat_model
from datapilot.session import AnalysisSession, open_session
from datapilot.tools.charts import build_figure, confusion_matrix_figure, feature_importance_figure

SAMPLES = {
    "Online retail, 5 linked tables (SQLite)": Path("data/samples/retail.sqlite"),
    "Online retail, one flat table (CSV)": Path("data/samples/online_retail.csv"),
    "Telecom customer churn (Excel)": Path("data/samples/customer_churn.xlsx"),
}
EXAMPLES = {
    "Online retail, 5 linked tables (SQLite)": [
        "Show the monthly revenue trend for 2011 as a chart.",
        "Which region had the largest drop in revenue from Q3 to Q4 2010, and why?",
        "Who are the top 10 customers by total spend?",
        "Now only for Europe.",
    ],
    "Online retail, one flat table (CSV)": [
        "What are the 5 best-selling products by quantity?",
        "Is there a correlation between unit price and quantity per line?",
        "How does the average order value differ by country?",
    ],
    "Telecom customer churn (Excel)": [
        "What is the churn rate by tariff plan?",
        "Which customers are most likely to churn next month?",
        "How are complaints related to churn?",
    ],
}


PROVIDERS = {"Groq": "groq", "OpenAI-compatible": "openai"}


def init_state() -> None:
    defaults = {
        "session": None,
        "messages": [],
        "dataset_label": None,
        "workdir": None,
        "llm_settings": None,
        "llm_fingerprint": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def workdir() -> Path:
    if st.session_state.workdir is None:
        st.session_state.workdir = tempfile.mkdtemp(prefix="datapilot-")
    return Path(st.session_state.workdir)


def api_key(settings: Settings) -> str:
    secret = settings.groq_api_key if settings.llm_provider == "groq" else settings.openai_api_key
    return secret.get_secret_value() if secret else ""


def fingerprint(settings: Settings) -> tuple[str, ...]:
    key_hash = hashlib.sha256(api_key(settings).encode()).hexdigest()
    return (settings.llm_provider, settings.llm_model, settings.openai_base_url or "", key_hash)


def model_settings() -> Settings:
    """LLM fields in the sidebar. A pasted key only lives in this browser session."""
    base = get_settings()
    st.sidebar.subheader("Language model")
    labels = list(PROVIDERS)
    default = labels.index(next(k for k, v in PROVIDERS.items() if v == base.llm_provider))
    provider = PROVIDERS[st.sidebar.selectbox("Provider", labels, index=default)]
    configured = provider == base.llm_provider

    if provider == "groq":
        model = st.sidebar.text_input(
            "Model", base.llm_model if configured else "llama-3.3-70b-versatile"
        )
        update = {"llm_provider": "groq", "llm_model": model.strip()}
        key_field, server_key = "groq_api_key", base.groq_api_key
        help_text = "Free key at console.groq.com."
    else:
        base_url = st.sidebar.text_input(
            "Base URL", base.openai_base_url or "https://api.openai.com/v1"
        )
        model = st.sidebar.text_input("Model", base.llm_model if configured else "gpt-4.1-mini")
        update = {
            "llm_provider": "openai",
            "llm_model": model.strip(),
            "openai_base_url": base_url.strip() or None,
        }
        key_field, server_key = "openai_api_key", base.openai_api_key
        help_text = "Any OpenAI-compatible key, for example from OpenRouter."

    pasted = st.sidebar.text_input(
        "API key",
        type="password",
        placeholder="using the key from .env" if server_key else "paste your key",
        help=f"{help_text} It stays in this browser session and is never stored or logged.",
    )
    if pasted.strip():
        update[key_field] = SecretStr(pasted.strip())
    return base.model_copy(update=update)


def use_model(session: AnalysisSession, settings: Settings) -> bool:
    """Point the session at the model in the sidebar, rebuilding it only when it changed."""
    if st.session_state.llm_fingerprint == fingerprint(settings):
        return True
    try:
        session.context.llm = build_chat_model(settings)
    except LLMError as error:
        st.error(str(error))
        return False
    st.session_state.llm_fingerprint = fingerprint(settings)
    return True


def load_dataset(paths: list[Path], label: str) -> None:
    settings: Settings = st.session_state.llm_settings
    if not api_key(settings):
        st.sidebar.warning("Paste an API key above first.")
        return
    try:
        with st.spinner("Loading data and indexing the schema..."):
            st.session_state.session = open_session(paths, workdir(), settings=settings)
    except LLMError as error:
        st.session_state.session = None
        st.error(str(error))
        return
    st.session_state.llm_fingerprint = fingerprint(settings)
    st.session_state.messages = []
    st.session_state.dataset_label = label


def sidebar() -> None:
    st.sidebar.title("DataPilot")
    st.session_state.llm_settings = model_settings()
    st.sidebar.divider()

    source = st.sidebar.radio("Data", ["Sample dataset", "Upload files"], horizontal=True)
    if source == "Sample dataset":
        label = st.sidebar.selectbox("Sample", list(SAMPLES))
        if st.sidebar.button("Load sample", width="stretch"):
            load_dataset([SAMPLES[label]], label)
    else:
        uploads = st.sidebar.file_uploader(
            "CSV, Excel, SQLite or Parquet",
            type=[suffix.lstrip(".") for suffix in sorted(SUPPORTED_SUFFIXES)],
            accept_multiple_files=True,
        )
        if uploads and st.sidebar.button("Load files", width="stretch"):
            folder = workdir() / "uploads"
            folder.mkdir(exist_ok=True)
            paths = []
            for upload in uploads:
                path = folder / Path(upload.name).name
                path.write_bytes(upload.getbuffer())
                paths.append(path)
            load_dataset(paths, ", ".join(upload.name for upload in uploads))

    session: AnalysisSession | None = st.session_state.session
    if session is None:
        return
    st.sidebar.divider()
    st.sidebar.subheader("Tables")
    for table in session.profile.tables:
        with st.sidebar.expander(f"{table.name} ({table.row_count:,} rows)"):
            st.text(table.render())
    if st.sidebar.button("Clear conversation", width="stretch"):
        session.reset()
        st.session_state.messages = []
    if session.context.confidence_model is None:
        st.sidebar.caption(
            "Confidence scores appear once the confidence model is trained "
            "(datapilot-eval, then datapilot-train-confidence)."
        )


def confidence_label(score: float | None) -> str:
    if score is None:
        return "n/a"
    level = "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
    return f"{score:.0%} ({level})"


def render_ml(state: AgentState) -> None:
    report = state["ml_report"]
    tabs = st.tabs(["Model comparison", "Feature importance", "Confusion matrix", "Notes"])
    with tabs[0]:
        st.dataframe(report.comparison_frame(), hide_index=True, width="stretch")
        test = ", ".join(f"{name} {value:.3f}" for name, value in report.test_metrics.items())
        st.caption(f"Held-out test set ({report.best_model}): {test}")
    with tabs[1]:
        st.plotly_chart(feature_importance_figure(report.feature_importance), width="stretch")
    with tabs[2]:
        if report.confusion_matrix:
            figure = confusion_matrix_figure(report.confusion_matrix, report.labels)
            st.plotly_chart(figure, width="stretch")
        else:
            st.caption("Not applicable for regression.")
    with tabs[3]:
        for warning in report.warnings or ["No warnings."]:
            st.markdown(f"- {warning}")
        if report.dropped:
            st.markdown("**Columns left out:**")
            for name, reason in report.dropped.items():
                st.markdown(f"- `{name}`: {reason}")


def render_answer(state: AgentState) -> None:
    st.markdown(state.get("answer", ""))
    trace = state.get("trace", [])
    columns = st.columns(5)
    columns[0].metric("Route", str(state.get("route", "")).replace("_", " "))
    columns[1].metric("Retries", state.get("total_retries", 0))
    columns[2].metric("Time", f"{sum(log.latency_ms for log in trace) / 1000:.1f}s")
    columns[3].metric(
        "Tokens", f"{sum(log.prompt_tokens + log.completion_tokens for log in trace):,}"
    )
    columns[4].metric("Confidence", confidence_label(state.get("confidence")))

    frame = state.get("final_frame")
    chart = state.get("chart")
    if frame is not None and chart is not None:
        st.plotly_chart(build_figure(frame, chart), width="stretch")
    if state.get("ml_report") is not None:
        render_ml(state)
    if frame is not None:
        st.dataframe(frame, hide_index=True, width="stretch")

    steps = state.get("step_results", [])
    if state.get("code"):
        with st.expander("Generated code"):
            language = "sql" if state.get("language") == "sql" else "python"
            if len(steps) > 1 and state.get("ml_report") is None:
                for step in steps:
                    st.markdown(f"**Step {step.step + 1}.** {step.description}")
                    st.code(step.code, language=language)
            else:
                st.code(state["code"], language=language)
    if len(steps) > 1 and state.get("ml_report") is None:
        with st.expander("Plan and intermediate results"):
            for step in steps[:-1]:
                st.markdown(f"**Step {step.step + 1}.** {step.description}")
                if step.frame is not None:
                    st.dataframe(step.frame.head(50), hide_index=True)
    with st.expander("Agent trace"):
        st.dataframe(
            pd.DataFrame(
                {
                    "node": [log.node for log in trace],
                    "ms": [round(log.latency_ms) for log in trace],
                    "tokens": [log.prompt_tokens + log.completion_tokens for log in trace],
                    "retry": [log.retry for log in trace],
                    "detail": [log.detail for log in trace],
                }
            ),
            hide_index=True,
            width="stretch",
        )


def ask(question: str) -> None:
    session: AnalysisSession = st.session_state.session
    if not use_model(session, st.session_state.llm_settings):
        return
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Planning, querying and checking..."):
                state = session.ask(question)
        except LLMError as error:
            st.error(f"The language model could not be reached: {error}")
            return
        render_answer(state)
    st.session_state.messages.append({"role": "assistant", "state": state})


LANDING_STYLE = """
<style>
.dp-eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: .78rem;
              opacity: .6; margin: 1.5rem 0 .4rem; }
.dp-title { font-size: 2.6rem; font-weight: 700; line-height: 1.15; margin-bottom: .7rem; }
.dp-lead { font-size: 1.1rem; opacity: .8; max-width: 44rem; line-height: 1.55; }
.dp-flow { display: flex; flex-wrap: wrap; gap: .45rem; margin: .3rem 0 1.2rem; }
.dp-step { border: 1px solid rgba(128, 128, 128, .35); border-radius: 999px;
           padding: .25rem .8rem; font-size: .88rem; }
.dp-step b { color: #2f6fdb; margin-right: .35rem; }
.dp-footer { font-size: .85rem; opacity: .6; margin-top: 2rem; }
</style>
"""
FEATURES = [
    (
        "Ask in plain English",
        'Trends, rankings and comparisons, plus follow-ups like "now only for Europe". '
        "Works with CSV, Excel and SQLite.",
    ),
    (
        "Checks its own work",
        "A critic catches errors, empty results and joins that duplicate rows, then sends "
        "the code back with a concrete fix.",
    ),
    (
        "Real models for predictions",
        '"Who is likely to churn?" trains and compares scikit-learn models with '
        "cross-validation instead of guessing.",
    ),
]
FLOW = [
    "Route the question",
    "Find the right tables",
    "Plan",
    "Write SQL or pandas",
    "Run it in a sandbox",
    "Check and retry",
    "Explain",
]


def landing_page() -> None:
    st.markdown(LANDING_STYLE, unsafe_allow_html=True)
    st.markdown(
        '<div class="dp-eyebrow">Multi-agent data analyst</div>'
        '<div class="dp-title">Ask your data a question.</div>'
        '<p class="dp-lead">DataPilot plans the analysis, writes the code, runs it safely and '
        "fixes its own mistakes. You get a short answer, the table, a chart and the exact "
        "SQL or Python behind it.</p>",
        unsafe_allow_html=True,
    )
    st.info(
        "To start, paste an API key in the sidebar (a free Groq key works), load a sample "
        "dataset or upload your own file, then ask away."
    )

    for column, (title, text) in zip(st.columns(3), FEATURES, strict=True):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(text)

    st.markdown("#### How it works")
    steps = "".join(
        f'<span class="dp-step"><b>{number}</b>{step}</span>'
        for number, step in enumerate(FLOW, start=1)
    )
    st.markdown(f'<div class="dp-flow">{steps}</div>', unsafe_allow_html=True)

    st.markdown("#### Things to try")
    for label in ("Online retail, 5 linked tables (SQLite)", "Telecom customer churn (Excel)"):
        st.markdown(f"*{label}*")
        for example in EXAMPLES[label][:2]:
            st.markdown(f"- {example}")

    st.markdown(
        '<div class="dp-footer">Your API key stays in this browser session and is never '
        "stored. Built with LangGraph, DuckDB, scikit-learn and Streamlit.</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="DataPilot", layout="wide")
    init_state()
    sidebar()

    session: AnalysisSession | None = st.session_state.session
    if session is None:
        landing_page()
        return

    st.title("DataPilot")
    st.caption(f"Talking to: {st.session_state.dataset_label}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_answer(message["state"])

    examples = EXAMPLES.get(st.session_state.dataset_label, [])
    chosen = None
    if examples and not st.session_state.messages:
        st.markdown("**Try one of these:**")
        for column, example in zip(st.columns(len(examples)), examples, strict=True):
            if column.button(example, width="stretch"):
                chosen = example

    question = st.chat_input("Ask a question about your data") or chosen
    if question:
        ask(question)


main()
