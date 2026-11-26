from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

Task = Literal["classification", "regression"]
RANDOM_STATE = 42
MIN_ROWS = 30
SMALL_SAMPLE = 200
LEAKAGE_SCORE = 0.99
MAX_CLASSES = 10
IMPORTANCE_ROWS = 2000
LOWER_IS_BETTER = {"mae", "rmse"}


class MLSetupError(ValueError):
    pass


@dataclass(frozen=True)
class MLSetup:
    target: str
    features: list[str]
    task: Task
    id_column: str | None = None
    positive_label: Any = None
    dropped: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelScore:
    model: str
    cv_mean: float
    cv_std: float
    cv_metrics: dict[str, float]


@dataclass
class MLReport:
    task: Task
    target: str
    features: list[str]
    dropped: dict[str, str]
    n_rows: int
    n_train: int
    n_test: int
    primary_metric: str
    model_scores: list[ModelScore]
    best_model: str
    test_metrics: dict[str, float]
    feature_importance: pd.DataFrame
    warnings: list[str]
    pipeline_description: str
    positive_label: str | None = None
    class_balance: dict[str, float] = field(default_factory=dict)
    confusion_matrix: list[list[int]] | None = None
    labels: list[str] = field(default_factory=list)
    risk_scores: pd.Series | None = None

    def comparison_frame(self) -> pd.DataFrame:
        rows = []
        for score in self.model_scores:
            row: dict[str, Any] = {"model": score.model}
            row.update({f"cv_{name}": round(value, 4) for name, value in score.cv_metrics.items()})
            row[f"cv_{self.primary_metric}_std"] = round(score.cv_std, 4)
            row["selected"] = score.model == self.best_model
            rows.append(row)
        return pd.DataFrame(rows)

    def summary_text(self) -> str:
        task = self.task
        if task == "classification":
            task = (
                "binary classification" if len(self.labels) == 2 else "multi-class classification"
            )
        lines = [
            f"Task: {task} predicting '{self.target}' on {self.n_rows:,} rows "
            f"({self.n_train:,} train / {self.n_test:,} test).",
        ]
        if self.positive_label is not None:
            lines.append(f"Positive class: {self.positive_label}")
        if self.class_balance:
            balance = ", ".join(
                f"{label}: {share:.1%}" for label, share in self.class_balance.items()
            )
            lines.append(f"Class balance: {balance}")
        lines.append(f"Features used ({len(self.features)}): {', '.join(self.features)}")
        if self.dropped:
            lines.append(
                "Dropped: " + "; ".join(f"{name} ({why})" for name, why in self.dropped.items())
            )
        lines.append("Cross-validated scores (5-fold on the training set):")
        for score in self.model_scores:
            metrics = ", ".join(f"{name} {value:.3f}" for name, value in score.cv_metrics.items())
            lines.append(f"  {score.model}: {metrics} (std {score.cv_std:.3f})")
        test = ", ".join(f"{name} {value:.3f}" for name, value in self.test_metrics.items())
        lines.append(f"Best model: {self.best_model}. Held-out test set: {test}")
        top = self.feature_importance.head(5)
        if not top.empty:
            drivers = ", ".join(
                f"{row.feature} ({row.importance:.3f})" for row in top.itertuples(index=False)
            )
            lines.append(f"Top features by permutation importance: {drivers}")
        if self.warnings:
            lines.append("Warnings: " + " ".join(self.warnings))
        return "\n".join(lines)


def infer_task(target: pd.Series) -> Task:
    if not is_numeric_dtype(target) or is_bool_dtype(target):
        return "classification"
    return "classification" if target.nunique() <= MAX_CLASSES else "regression"


def positive_class(target: pd.Series) -> Any:
    values = list(target.dropna().unique())
    for candidate in (1, True, "1", "yes", "Yes", "true", "True", "churn", "churned"):
        if candidate in values:
            return candidate
    return target.value_counts().idxmin()


def looks_like_id(series: pd.Series, name: str) -> bool:
    lowered = name.lower()
    named_like_id = lowered == "id" or lowered.endswith(("_id", "id"))
    non_null = series.dropna()
    return named_like_id and len(non_null) > 0 and non_null.nunique() >= 0.95 * len(non_null)


def leakage_score(feature: pd.Series, target: pd.Series, task: Task, positive: Any) -> float:
    """How well this single column predicts the target on its own (1.0 = perfectly)."""
    mask = feature.notna() & target.notna()
    feature, target = feature[mask], target[mask]
    if len(feature) < 20 or feature.nunique() < 2:
        return 0.0
    if is_numeric_dtype(feature) and not is_bool_dtype(feature):
        if task == "regression":
            correlation = np.corrcoef(feature.astype(float), target.astype(float))[0, 1]
            return 0.0 if np.isnan(correlation) else abs(float(correlation))
        if target.nunique() == 2:
            auc = roc_auc_score(target == positive, feature.astype(float))
            return max(auc, 1 - auc)
        return 0.0
    if task == "classification" and feature.nunique() < 0.5 * len(feature):
        crosstab = pd.crosstab(feature, target)
        return float(crosstab.max(axis=1).sum() / len(feature))
    return 0.0


def validate_setup(
    frame: pd.DataFrame,
    target: str,
    features: Sequence[str] | None = None,
    id_column: str | None = None,
) -> MLSetup:
    """Check a model-proposed target/feature choice against the real data."""
    if target not in frame.columns:
        raise MLSetupError(f"Target column {target!r} is not in the table.")
    target_values = frame[target].dropna()
    if target_values.nunique() < 2:
        raise MLSetupError(f"Target column {target!r} has fewer than two distinct values.")
    task = infer_task(target_values)
    positive = positive_class(target_values) if task == "classification" else None

    dropped: dict[str, str] = {}
    warnings: list[str] = []
    kept = []
    for column in features or [column for column in frame.columns if column != target]:
        if column == target:
            continue
        if column not in frame.columns:
            dropped[column] = "not in the table"
            continue
        series = frame[column]
        text_like = not is_numeric_dtype(series) and not is_datetime64_any_dtype(series)
        if column == id_column or looks_like_id(series, column):
            dropped[column] = "identifier"
        elif series.nunique(dropna=True) <= 1:
            dropped[column] = "constant"
        elif series.isna().mean() > 0.6:
            dropped[column] = "mostly missing"
        elif text_like and series.nunique() > max(50, 0.05 * len(series)):
            dropped[column] = "high-cardinality text"
        elif leakage_score(series, frame[target], task, positive) >= LEAKAGE_SCORE:
            dropped[column] = "almost perfectly predicts the target (likely leakage)"
            warnings.append(
                f"'{column}' was excluded because on its own it almost perfectly predicts "
                f"'{target}', which usually means it is recorded after the outcome."
            )
        else:
            kept.append(column)
    if not kept:
        raise MLSetupError("None of the proposed feature columns are usable.")
    return MLSetup(target, kept, task, id_column, positive, dropped, warnings)


def expand_dates(features: pd.DataFrame) -> pd.DataFrame:
    """Models cannot use timestamps directly, so turn them into calendar parts."""
    features = features.copy()
    for column in list(features.columns):
        if is_datetime64_any_dtype(features[column]):
            values = features.pop(column)
            features[f"{column}_year"] = values.dt.year
            features[f"{column}_month"] = values.dt.month
            features[f"{column}_dayofweek"] = values.dt.dayofweek
    return features


def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    features = expand_dates(features)
    for column in features.columns:
        series = features[column]
        if is_bool_dtype(series):
            features[column] = series.astype(float)
        elif not is_numeric_dtype(series):
            # One-hot encoding needs one consistent type per column.
            features[column] = series.map(lambda value: np.nan if pd.isna(value) else str(value))
            features[column] = features[column].astype(object)
    return features


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = [column for column in features.columns if is_numeric_dtype(features[column])]
    categorical = [column for column in features.columns if column not in numeric]
    numeric_steps = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_steps, numeric), ("categorical", categorical_steps, categorical)]
    )


def candidate_models(task: Task, random_state: int) -> dict[str, Any]:
    if task == "classification":
        # Balanced class weights keep the minority class (e.g. churners) from being ignored.
        return {
            "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
            "Gradient Boosting": HistGradientBoostingClassifier(
                class_weight="balanced", random_state=random_state
            ),
        }
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=random_state
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=random_state),
    }


def scoring_for(task: Task, labels: Sequence[Any], positive: Any) -> tuple[dict[str, Any], str]:
    if task == "regression":
        scoring = {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        }
        return scoring, "rmse"
    if len(labels) == 2:
        scoring = {
            "roc_auc": "roc_auc",
            "f1": make_scorer(f1_score, pos_label=positive),
            "accuracy": "accuracy",
        }
        return scoring, "roc_auc"
    return {"f1_macro": "f1_macro", "accuracy": "accuracy"}, "f1_macro"


def test_metrics_for(
    task: Task, pipeline: Pipeline, features: pd.DataFrame, target: pd.Series, positive: Any
) -> dict[str, float]:
    predicted = pipeline.predict(features)
    if task == "regression":
        return {
            "mae": float(mean_absolute_error(target, predicted)),
            "rmse": float(root_mean_squared_error(target, predicted)),
            "r2": float(r2_score(target, predicted)),
        }
    if target.nunique() == 2:
        classes = list(pipeline.classes_)
        probability = pipeline.predict_proba(features)[:, classes.index(positive)]
        return {
            "roc_auc": float(roc_auc_score(target == positive, probability)),
            "f1": float(f1_score(target, predicted, pos_label=positive)),
            "precision": float(precision_score(target, predicted, pos_label=positive)),
            "recall": float(recall_score(target, predicted, pos_label=positive)),
            "accuracy": float(accuracy_score(target, predicted)),
        }
    return {
        "f1_macro": float(f1_score(target, predicted, average="macro")),
        "accuracy": float(accuracy_score(target, predicted)),
    }


def importance_frame(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    scoring: Any,
    random_state: int,
) -> pd.DataFrame:
    if len(features) > IMPORTANCE_ROWS:
        features = features.sample(IMPORTANCE_ROWS, random_state=random_state)
        target = target.loc[features.index]
    # Permuting raw columns (not one-hot outputs) keeps the importances readable.
    result = permutation_importance(
        pipeline, features, target, scoring=scoring, n_repeats=5, random_state=random_state
    )
    return (
        pd.DataFrame(
            {
                "feature": list(features.columns),
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run_model_comparison(
    frame: pd.DataFrame,
    setup: MLSetup,
    *,
    cv_folds: int = 5,
    test_size: float = 0.2,
    max_rows: int = 50_000,
    random_state: int = RANDOM_STATE,
) -> MLReport:
    """Train and compare the candidate models with a held-out test set and k-fold CV."""
    warnings = list(setup.warnings)
    data = frame.dropna(subset=[setup.target])
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=random_state)
        warnings.append(f"Trained on a random sample of {max_rows:,} rows to keep it fast.")

    target = data[setup.target]
    if setup.task == "classification":
        if is_numeric_dtype(target) and (target.dropna() % 1 == 0).all():
            target = target.astype(int)
        counts = target.value_counts()
        rare = counts[counts < cv_folds].index
        if len(rare):
            warnings.append(
                f"Dropped {len(rare)} classes with fewer than {cv_folds} rows: {list(rare)}."
            )
            data, target = data[~target.isin(rare)], target[~target.isin(rare)]
    if len(data) < MIN_ROWS:
        raise MLSetupError(f"Only {len(data)} usable rows; at least {MIN_ROWS} are needed.")
    if len(data) < SMALL_SAMPLE:
        warnings.append(
            f"Only {len(data)} rows: the scores are noisy and could change a lot with more data."
        )

    features = prepare_features(data[setup.features])
    is_classifier = setup.task == "classification"
    labels = sorted(target.unique().tolist()) if is_classifier else []
    positive = positive_class(target) if is_classifier and len(labels) == 2 else None
    balance = {}
    if is_classifier:
        balance = {str(k): float(v) for k, v in target.value_counts(normalize=True).items()}
        if min(balance.values()) < 0.1:
            warnings.append(
                "The classes are imbalanced (smallest class under 10%), so accuracy alone "
                "is misleading; look at F1 and ROC-AUC."
            )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target if is_classifier else None,
    )
    splitter = (
        StratifiedKFold(cv_folds, shuffle=True, random_state=random_state)
        if is_classifier
        else KFold(cv_folds, shuffle=True, random_state=random_state)
    )
    scoring, primary = scoring_for(setup.task, labels, positive)

    scores, pipelines = [], {}
    for name, model in candidate_models(setup.task, random_state).items():
        pipeline = Pipeline([("prepare", build_preprocessor(features)), ("model", model)])
        results = cross_validate(
            pipeline, x_train, y_train, cv=splitter, scoring=scoring, error_score="raise"
        )
        metrics = {}
        for metric in scoring:
            values = results[f"test_{metric}"]
            metrics[metric] = float(-values.mean() if metric in LOWER_IS_BETTER else values.mean())
        spread = float(results[f"test_{primary}"].std())
        scores.append(ModelScore(name, metrics[primary], spread, metrics))
        pipelines[name] = pipeline

    if primary in LOWER_IS_BETTER:
        best = min(scores, key=lambda score: score.cv_mean)
    else:
        best = max(scores, key=lambda score: score.cv_mean)
    best_pipeline = pipelines[best.model].fit(x_train, y_train)
    test_metrics = test_metrics_for(setup.task, best_pipeline, x_test, y_test, positive)

    headline = test_metrics.get("roc_auc", test_metrics.get("r2"))
    if headline is not None and headline >= LEAKAGE_SCORE:
        warnings.append(
            "The test score is almost perfect. Double-check that no feature is only known "
            "after the outcome (data leakage)."
        )

    importance_scoring = scoring[primary]
    importance = importance_frame(best_pipeline, x_test, y_test, importance_scoring, random_state)

    matrix = None
    if is_classifier:
        matrix = confusion_matrix(y_test, best_pipeline.predict(x_test), labels=labels).tolist()

    risk_scores = None
    if positive is not None:
        # Out-of-fold probabilities: every row is scored by a model that never saw it.
        probabilities = cross_val_predict(
            pipelines[best.model], features, target, cv=splitter, method="predict_proba"
        )
        risk_scores = pd.Series(
            probabilities[:, labels.index(positive)], index=data.index, name="risk_score"
        )

    return MLReport(
        task=setup.task,
        target=setup.target,
        features=list(setup.features),
        dropped=dict(setup.dropped),
        n_rows=len(data),
        n_train=len(x_train),
        n_test=len(x_test),
        primary_metric=primary,
        model_scores=scores,
        best_model=best.model,
        test_metrics=test_metrics,
        feature_importance=importance,
        warnings=warnings,
        pipeline_description=str(best_pipeline),
        positive_label=None if positive is None else str(positive),
        class_balance=balance,
        confusion_matrix=matrix,
        labels=[str(label) for label in labels],
        risk_scores=risk_scores,
    )
