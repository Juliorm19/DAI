from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TemporalSplits:
    train: pd.DataFrame
    optimization: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def temporal_group_split(
    frame: pd.DataFrame,
    fractions: tuple[float, float, float, float] = (0.60, 0.15, 0.10, 0.15),
) -> TemporalSplits:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to one")
    games = (
        frame[["game_id", "game_date"]]
        .drop_duplicates("game_id")
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )
    games_per_date = games.groupby("game_date").size().sort_index()
    cumulative = games_per_date.cumsum().to_numpy()
    targets = np.cumsum(np.asarray(fractions[:-1]) * len(games))
    cutoff_positions = np.searchsorted(cumulative, targets, side="left")
    cutoff_dates = [games_per_date.index[min(int(position), len(games_per_date) - 2)] for position in cutoff_positions]
    date_masks = [
        games["game_date"] <= cutoff_dates[0],
        (games["game_date"] > cutoff_dates[0]) & (games["game_date"] <= cutoff_dates[1]),
        (games["game_date"] > cutoff_dates[1]) & (games["game_date"] <= cutoff_dates[2]),
        games["game_date"] > cutoff_dates[2],
    ]
    game_groups = [games.loc[mask, "game_id"].to_numpy() for mask in date_masks]
    if any(len(group) == 0 for group in game_groups):
        raise ValueError("Not enough distinct dates for four temporal partitions")
    parts = [frame[frame["game_id"].isin(ids)].copy() for ids in game_groups]
    for before, after in zip(parts, parts[1:]):
        if before["game_date"].max() >= after["game_date"].min():
            raise ValueError("Temporal ordering failed")
        if set(before["game_id"]).intersection(after["game_id"]):
            raise ValueError("A game appears in multiple splits")
    return TemporalSplits(*parts)


def make_logistic_pipeline(c_value: float, random_seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    max_iter=1_000,
                    solver="lbfgs",
                    random_state=random_seed,
                ),
            ),
        ]
    )


@dataclass
class ProbabilityCalibrator:
    method: str
    estimator: Any

    def predict(self, pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
        if self.method == "sigmoid":
            scores = pipeline.decision_function(features).reshape(-1, 1)
            return self.estimator.predict_proba(scores)[:, 1]
        if self.method == "isotonic":
            raw = pipeline.predict_proba(features)[:, 1]
            return np.asarray(self.estimator.predict(raw), dtype=float)
        raise ValueError(f"Unknown calibration method: {self.method}")


def _fit_calibrator(method: str, pipeline: Pipeline, features: pd.DataFrame, target: pd.Series) -> ProbabilityCalibrator:
    y = np.asarray(target, dtype=int)
    if method == "sigmoid":
        scores = pipeline.decision_function(features).reshape(-1, 1)
        estimator = LogisticRegression(C=1_000_000, solver="lbfgs", max_iter=500, random_state=42)
        estimator.fit(scores, y)
        return ProbabilityCalibrator(method=method, estimator=estimator)
    if method == "isotonic":
        raw = pipeline.predict_proba(features)[:, 1]
        estimator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
        estimator.fit(raw, y)
        return ProbabilityCalibrator(method=method, estimator=estimator)
    raise ValueError(method)


def choose_and_fit_calibrator(
    pipeline: Pipeline,
    calibration_frame: pd.DataFrame,
    feature_names: list[str],
) -> tuple[ProbabilityCalibrator, dict[str, float]]:
    games = (
        calibration_frame[["game_id", "game_date"]]
        .drop_duplicates("game_id")
        .sort_values(["game_date", "game_id"])
    )
    midpoint = max(1, len(games) // 2)
    fit_ids = set(games.iloc[:midpoint]["game_id"])
    selection_ids = set(games.iloc[midpoint:]["game_id"])
    fit_frame = calibration_frame[calibration_frame["game_id"].isin(fit_ids)]
    selection_frame = calibration_frame[calibration_frame["game_id"].isin(selection_ids)]
    if selection_frame.empty:
        selection_frame = fit_frame

    scores: dict[str, float] = {}
    for method in ("sigmoid", "isotonic"):
        calibrator = _fit_calibrator(method, pipeline, fit_frame[feature_names], fit_frame["won"])
        probabilities = calibrator.predict(pipeline, selection_frame[feature_names])
        scores[method] = float(log_loss(selection_frame["won"], probabilities, labels=[0, 1]))
    selected_method = min(scores, key=scores.get)
    final_calibrator = _fit_calibrator(
        selected_method,
        pipeline,
        calibration_frame[feature_names],
        calibration_frame["won"],
    )
    return final_calibrator, scores


def expected_calibration_error(y_true: Iterable[int], probabilities: Iterable[float], bins: int = 10) -> float:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(probabilities), dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            error += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error)


def evaluate_probabilities(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    predicted = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_10_bins": expected_calibration_error(y, p, bins=10),
    }


@dataclass
class FittedProbabilityModel:
    name: str
    feature_names: list[str]
    pipeline: Pipeline
    calibrator: ProbabilityCalibrator
    metadata: dict[str, Any]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        missing = set(self.feature_names) - set(features.columns)
        if missing:
            raise ValueError(f"Missing inference features: {sorted(missing)}")
        return self.calibrator.predict(self.pipeline, features[self.feature_names])


def fit_probability_model(
    name: str,
    development_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    feature_names: list[str],
    c_value: float,
    random_seed: int = 42,
) -> tuple[FittedProbabilityModel, dict[str, float]]:
    pipeline = make_logistic_pipeline(c_value, random_seed=random_seed)
    pipeline.fit(development_frame[feature_names], development_frame["won"])
    calibrator, calibration_scores = choose_and_fit_calibrator(pipeline, calibration_frame, feature_names)
    model = FittedProbabilityModel(
        name=name,
        feature_names=list(feature_names),
        pipeline=pipeline,
        calibrator=calibrator,
        metadata={
            "regularization_c": float(c_value),
            "calibration_method": calibrator.method,
            "calibration_selection_scores": calibration_scores,
        },
    )
    return model, calibration_scores
