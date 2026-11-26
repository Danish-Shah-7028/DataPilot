"""Restricted execution of model-written pandas code.

Three layers, cheapest first:
1. A static AST check rejects imports outside a small allow-list, dunder access and the
   pandas/numpy I/O helpers (read_csv, to_parquet, np.load, ...).
2. The code runs with a trimmed set of builtins (no open, exec, getattr, ...).
3. It runs in a separate isolated Python process with a timeout, where an audit hook
   refuses file writes, reads outside the Python installation, sockets and subprocesses.
   The result comes back as Apache Arrow rather than pickle, so nothing the code
   produces can execute in the main process when it is read.

This is defence in depth for a single-user analyst tool, not a hardened multi-tenant
sandbox; for that you would run the child inside a container with no network.
"""

import ast
import builtins
import contextlib
import importlib
import importlib.util
import json
import keyword
import math
import os
import pickle
import site
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from datapilot.errors import ErrorCategory
from datapilot.tools.execution import ExecutionResult, dedupe_columns

WORKER_COMMAND = "from datapilot.tools.python_sandbox import worker_main; worker_main()"

ALLOWED_MODULES = frozenset(
    {"pandas", "numpy", "math", "statistics", "datetime", "re", "collections", "itertools"}
)
BLOCKED_NAMES = frozenset(
    {
        "open", "exec", "eval", "compile", "input", "breakpoint", "globals", "locals",
        "vars", "getattr", "setattr", "delattr", "help", "exit", "quit", "memoryview",
        "dir", "type", "object", "super",
    }
)  # fmt: skip
BLOCKED_ATTRIBUTES = frozenset(
    {
        "eval", "load", "save", "savez", "savez_compressed", "loadtxt", "savetxt",
        "genfromtxt", "fromfile", "tofile", "memmap", "ctypeslib", "io", "os", "sys", "lib",
        "testing", "f2py", "compat", "core", "api", "plotting", "util", "system", "popen",
        "subprocess", "builtins", "modules", "import_module", "DataSource",
    }
)  # fmt: skip
ALLOWED_TO_METHODS = frozenset(
    {
        "to_frame", "to_dict", "to_list", "to_numpy", "to_period", "to_timestamp",
        "to_datetime", "to_numeric", "to_timedelta", "to_records", "to_pydatetime",
        "to_string", "to_offset",
    }
)  # fmt: skip
SAFE_BUILTINS = (
    "abs", "all", "any", "bool", "callable", "chr", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "hash", "int", "isinstance", "issubclass", "iter", "len",
    "list", "map", "max", "min", "next", "ord", "pow", "range", "repr", "reversed", "round",
    "set", "slice", "sorted", "str", "sum", "tuple", "zip", "ArithmeticError", "Exception",
    "IndexError", "KeyError", "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
)  # fmt: skip
BLOCKED_EVENTS = (
    "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.startfile", "os.fork",
    "os.kill", "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.chown",
    "os.link", "os.symlink", "os.truncate", "os.putenv", "os.unsetenv", "subprocess.",
    "socket.", "shutil.", "ctypes.", "urllib.", "http.", "ftplib.", "smtplib.",
    "webbrowser.", "winreg.",
)  # fmt: skip
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


class UnsafeCodeError(ValueError):
    pass


def validate_code(code: str) -> None:
    """Raise UnsafeCodeError (or SyntaxError) if the code uses anything off-limits."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES:
                    raise UnsafeCodeError(f"Importing {alias.name!r} is not allowed.")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                raise UnsafeCodeError(f"Importing from {node.module!r} is not allowed.")
        elif isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in BLOCKED_NAMES:
                raise UnsafeCodeError(f"Use of {node.id!r} is not allowed.")
        elif isinstance(node, ast.Attribute):
            check_attribute(node.attr)
        elif isinstance(node, ast.FunctionDef | ast.arg):
            name = node.name if isinstance(node, ast.FunctionDef) else node.arg
            if name.startswith("_"):
                raise UnsafeCodeError("Names starting with an underscore are not allowed.")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Blocks dunder tricks smuggled through strings, e.g. df.query("@x.__class__").
            if "__" in node.value:
                raise UnsafeCodeError("Strings containing '__' are not allowed.")
        elif isinstance(
            node,
            ast.Global | ast.Nonlocal | ast.ClassDef | ast.AsyncFunctionDef | ast.Await,
        ):
            raise UnsafeCodeError("Classes, async code and global/nonlocal are not allowed.")


def check_attribute(attribute: str) -> None:
    if attribute.startswith("_"):
        raise UnsafeCodeError(f"Access to private attribute {attribute!r} is not allowed.")
    if attribute in BLOCKED_ATTRIBUTES or attribute.startswith("read_"):
        raise UnsafeCodeError(f"{attribute!r} is not allowed (no file, network or system access).")
    if attribute.startswith("to_") and attribute not in ALLOWED_TO_METHODS:
        raise UnsafeCodeError(f"{attribute!r} is not allowed (results must stay in memory).")


def guarded_import(name: str, *args: Any) -> Any:
    level = args[3] if len(args) > 3 else 0
    if level or name.split(".")[0] not in ALLOWED_MODULES:
        raise ImportError(f"Importing {name!r} is not allowed in the sandbox.")
    from_list = args[2] if len(args) > 2 else None
    module = importlib.import_module(name)
    return module if from_list else importlib.import_module(name.split(".")[0])


def safe_builtins() -> dict[str, Any]:
    allowed = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
    allowed["print"] = lambda *values, **options: None
    allowed["__import__"] = guarded_import
    return allowed


def allowed_roots() -> tuple[str, ...]:
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, *site.getsitepackages()}
    spec = importlib.util.find_spec("datapilot")
    if spec and spec.submodule_search_locations:
        roots.update(spec.submodule_search_locations)
    return tuple(os.path.normcase(os.path.abspath(root)) for root in roots)


def install_guards(roots: tuple[str, ...]) -> None:
    """Audit hook for the child process; once added it cannot be removed."""

    def inside_roots(path: object) -> bool:
        if isinstance(path, int):
            return True
        try:
            text = os.path.normcase(os.path.abspath(os.fsdecode(path)))
        except TypeError:
            return False
        return text.startswith(roots)

    def guard(event: str, args: tuple[Any, ...]) -> None:
        if event == "open":
            path, mode, flags = args
            read_only = (
                not any(flag in mode for flag in "wax+")
                if isinstance(mode, str)
                else not (flags or 0) & WRITE_FLAGS
            )
            if not (read_only and inside_roots(path)):
                raise PermissionError(f"File access is not allowed in the sandbox: {path}")
        elif event in ("os.listdir", "os.scandir"):
            if not inside_roots(args[0] if args else "."):
                raise PermissionError("Listing directories is not allowed in the sandbox.")
        elif event.startswith(BLOCKED_EVENTS):
            raise PermissionError(f"{event} is not allowed in the sandbox.")

    sys.addaudithook(guard)


def to_frame(value: Any) -> pd.DataFrame:
    """Normalise whatever the code assigned to `result` into a DataFrame."""
    if isinstance(value, pd.DataFrame):
        frame = value
        if isinstance(frame.columns, pd.MultiIndex):
            frame = frame.set_axis(
                ["_".join(str(part) for part in column if str(part)) for column in frame.columns],
                axis=1,
            )
        if not isinstance(frame.index, pd.RangeIndex):
            frame = frame.reset_index(allow_duplicates=True)
    elif isinstance(value, pd.Series):
        series = value.rename(value.name if value.name is not None else "value")
        frame = (
            series.to_frame()
            if isinstance(series.index, pd.RangeIndex)
            else series.reset_index(allow_duplicates=True)
        )
    elif isinstance(value, dict):
        frame = pd.DataFrame({"key": list(value.keys()), "value": list(value.values())})
    elif isinstance(value, list | tuple | set):
        frame = pd.DataFrame({"value": list(value)})
    elif isinstance(value, np.ndarray):
        frame = pd.DataFrame(value)
    else:
        frame = pd.DataFrame({"value": [value]})
    return frame.set_axis(dedupe_columns([str(column) for column in frame.columns]), axis=1)


def build_namespace(tables: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins(),
        "pd": pd,
        "np": np,
        "math": math,
        "tables": dict(tables),
    }
    for name, frame in tables.items():
        if name.isidentifier() and not keyword.iskeyword(name) and not name.startswith("_"):
            namespace[name] = frame
    return namespace


def frame_to_arrow(frame: pd.DataFrame) -> bytes:
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
    except (pa.ArrowException, TypeError, ValueError):
        # Mixed-type object columns cannot be typed by Arrow; fall back to text.
        frame = frame.copy()
        for column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = frame[column].map(
                    lambda value: None if value is None else str(value)
                )
        table = pa.Table.from_pandas(frame, preserve_index=False)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def arrow_to_frame(payload: bytes) -> pd.DataFrame:
    return pa.ipc.open_stream(payload).read_all().to_pandas()


def execute_untrusted(code: str, tables: dict[str, pd.DataFrame], row_limit: int) -> bytes:
    """Run the code (inside the worker) and encode the outcome as header line + Arrow."""
    try:
        install_guards(allowed_roots())
        namespace = build_namespace(tables)
        with contextlib.redirect_stdout(sys.stderr):
            exec(compile(code, "<analysis>", "exec"), namespace)
        if "result" not in namespace:
            raise NameError("The code must assign its answer to a variable named `result`.")
        frame = to_frame(namespace["result"])
        header = {"status": "ok", "truncated": len(frame) > row_limit, "message": ""}
        body = frame_to_arrow(frame.head(row_limit))
    except BaseException as error:
        header = {"status": "error", "truncated": False, "message": f"{error_kind(error)}: {error}"}
        body = b""
    return json.dumps(header).encode() + b"\n" + body


def worker_main() -> None:
    """Entry point of the sandbox subprocess: job in on stdin, outcome out on stdout."""
    # The job comes from the trusted parent process, so unpickling it is fine. The
    # result goes back as Arrow, which is data only and cannot run code when read.
    job = pickle.load(sys.stdin.buffer)
    outcome = execute_untrusted(job["code"], job["tables"], job["row_limit"])
    sys.stdout.buffer.write(outcome)
    sys.stdout.buffer.flush()


def error_kind(error: BaseException) -> str:
    # repr(KeyError('x')) == "KeyError('x')"; avoids reaching for dunder attributes.
    return repr(error).split("(", 1)[0]


def classify_python_error(message: str) -> ErrorCategory:
    kind = message.split(":", 1)[0]
    if kind in ("PermissionError", "ImportError", "ModuleNotFoundError"):
        return ErrorCategory.UNSAFE_CODE
    if kind in ("SyntaxError", "IndentationError"):
        return ErrorCategory.SYNTAX_ERROR
    if kind == "KeyError":
        return ErrorCategory.UNKNOWN_COLUMN
    if kind == "TypeError":
        return ErrorCategory.TYPE_MISMATCH
    return ErrorCategory.RUNTIME_ERROR


def run_python(
    code: str,
    tables: Mapping[str, pd.DataFrame],
    *,
    row_limit: int = 5000,
    timeout_seconds: float = 30.0,
) -> ExecutionResult:
    started = time.perf_counter()

    def failure(message: str, category: ErrorCategory) -> ExecutionResult:
        elapsed = (time.perf_counter() - started) * 1000
        return ExecutionResult(False, error=message, category=category, elapsed_ms=elapsed)

    if not code.strip():
        return failure("Empty code.", ErrorCategory.NO_CODE)
    try:
        validate_code(code)
    except SyntaxError as error:
        return failure(f"SyntaxError: {error}", ErrorCategory.SYNTAX_ERROR)
    except UnsafeCodeError as error:
        return failure(str(error), ErrorCategory.UNSAFE_CODE)

    # A fresh isolated interpreter (-I: no cwd on sys.path, no PYTHON* variables) rather
    # than multiprocessing, whose spawn mode re-imports the caller's main script.
    job = pickle.dumps({"code": code, "tables": dict(tables), "row_limit": row_limit})
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", WORKER_COMMAND],
            input=job,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return failure(f"Code exceeded the {timeout_seconds:g}s time limit.", ErrorCategory.TIMEOUT)

    header_line, separator, body = completed.stdout.partition(b"\n")
    if completed.returncode != 0 or not separator:
        detail = completed.stderr.decode("utf-8", "replace").strip().splitlines()[-1:]
        message = "The sandbox process exited unexpectedly." + (f" {detail[0]}" if detail else "")
        return failure(message, ErrorCategory.RUNTIME_ERROR)
    header = json.loads(header_line)
    if header["status"] != "ok":
        return failure(header["message"], classify_python_error(header["message"]))
    elapsed = (time.perf_counter() - started) * 1000
    frame = arrow_to_frame(body)
    return ExecutionResult(True, frame=frame, truncated=header["truncated"], elapsed_ms=elapsed)
