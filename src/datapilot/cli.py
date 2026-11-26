import argparse
import tempfile
from pathlib import Path

from datapilot.graph.state import AgentState
from datapilot.session import AnalysisSession, open_session
from datapilot.tools.execution import preview


def print_answer(state: AgentState) -> None:
    print(f"\n{state.get('answer', '')}\n")
    frame = state.get("final_frame")
    if frame is not None:
        print(preview(frame, 20))
    if state.get("code"):
        print(f"\n--- {state.get('language', 'code')} ---\n{state['code']}")
    trace = state.get("trace", [])
    tokens = sum(log.prompt_tokens + log.completion_tokens for log in trace)
    latency = sum(log.latency_ms for log in trace) / 1000
    confidence = state.get("confidence")
    print(
        f"\nroute={state.get('route')} retries={state.get('total_retries', 0)} "
        f"tokens={tokens} time={latency:.1f}s"
        + (f" confidence={confidence:.2f}" if confidence is not None else "")
    )


def chat(session: AnalysisSession) -> None:
    print("Ask a question about your data (empty line to quit, 'reset' to forget context).")
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            break
        if not question:
            break
        if question.lower() == "reset":
            session.reset()
            continue
        print_answer(session.ask(question))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask DataPilot questions from the terminal.")
    parser.add_argument("--data", type=Path, action="append", required=True,
                        help="CSV, Excel, SQLite or Parquet file (repeatable)")  # fmt: skip
    parser.add_argument("question", nargs="?", help="ask one question and exit")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as workdir:
        session = open_session(args.data, Path(workdir))
        print("Loaded tables: " + ", ".join(session.workspace.table_names))
        if args.question:
            print_answer(session.ask(args.question))
        else:
            chat(session)
