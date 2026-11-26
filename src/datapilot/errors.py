from enum import StrEnum


class ErrorCategory(StrEnum):
    """Error taxonomy shared by the executors, the critic and the evaluation logs."""

    UNSAFE_CODE = "unsafe_code"
    SYNTAX_ERROR = "syntax_error"
    UNKNOWN_TABLE = "unknown_table"
    UNKNOWN_COLUMN = "unknown_column"
    TYPE_MISMATCH = "type_mismatch"
    TIMEOUT = "timeout"
    RUNTIME_ERROR = "runtime_error"
    NO_CODE = "no_code"
    EMPTY_RESULT = "empty_result"
    ALL_NULL_RESULT = "all_null_result"
    IMPOSSIBLE_VALUE = "impossible_value"
    ROW_EXPLOSION = "row_explosion"
    AGGREGATION_MISMATCH = "aggregation_mismatch"
    SEMANTIC_MISMATCH = "semantic_mismatch"


EXECUTION_ERRORS = frozenset(
    {
        ErrorCategory.UNSAFE_CODE,
        ErrorCategory.SYNTAX_ERROR,
        ErrorCategory.UNKNOWN_TABLE,
        ErrorCategory.UNKNOWN_COLUMN,
        ErrorCategory.TYPE_MISMATCH,
        ErrorCategory.TIMEOUT,
        ErrorCategory.RUNTIME_ERROR,
        ErrorCategory.NO_CODE,
    }
)
