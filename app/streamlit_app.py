"""DataPilot web app. Run with: streamlit run app/streamlit_app.py"""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from datapilot.config import get_settings
from datapilot.data.loader import SUPPORTED_SUFFIXES
from datapilot.graph.state import AgentState
from datapilot.llm.provider import LLMError
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


def init_state() -> None:
    defaults = {"session": None, "messages": [], "dataset_label": None, "workdir": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def workdir() -> Path:
    if st.session_state.workdir is None:
        st.session_state.workdir = tempfile.mkdtemp(prefix="datapilot-")
    return Path(st.session_state.workdir)


def load_dataset(paths: list[Path], label: str) -> None:
    try:
        with st.spinner("Loading data and indexing the schema..."):
            st.session_state.session = open_session(paths, workdir())
    except LLMError as error:
        st.session_state.session = None
        st.error(f"{error}")
        return
    st.session_state.messages = []
    st.session_state.dataset_label = label


def sidebar() -> None:
    settings = get_settings()
    st.sidebar.title("DataPilot")
    st.sidebar.caption(f"Model: `{settings.llm_model}` via {settings.llm_provider}")

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


def main() -> None:
    st.set_page_config(page_title="DataPilot", layout="wide")
    init_state()
    sidebar()

    st.title("DataPilot")
    st.caption(
        "Ask questions about your data in plain English. A team of agents plans the analysis, "
        "writes and runs the SQL or pandas code, checks the result and explains it."
    )
    session: AnalysisSession | None = st.session_state.session
    if session is None:
        st.info("Load a sample dataset or upload your own files from the sidebar to start.")
        return

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
