from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_COLUMN = "Person_ID"
TARGET_COLUMN = "Health_Score"
DIRECT_LEAKAGE_COLUMNS = ["Wellness_Category"]
TIME_COLUMNS = ["Wake_Up_Time", "Sleep_Time"]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace(" ", "_") for c in out.columns]
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_column_names(df)

    for col in TIME_COLUMNS:
        if col not in out.columns:
            continue
        parsed = pd.to_datetime(out[col].astype(str), format="%H:%M", errors="coerce")
        minutes = parsed.dt.hour * 60 + parsed.dt.minute
        out[f"{col}_sin"] = np.sin(2.0 * np.pi * minutes / 1440.0)
        out[f"{col}_cos"] = np.cos(2.0 * np.pi * minutes / 1440.0)
        out = out.drop(columns=[col])

    drop_columns = [ID_COLUMN, TARGET_COLUMN, *DIRECT_LEAKAGE_COLUMNS]
    out = out.drop(columns=[c for c in drop_columns if c in out.columns], errors="ignore")
    return out


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="if_binary",
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_residual_models(random_state: int = 2026) -> Dict[str, BaseEstimator]:
    return {
        "hist_residual": HistGradientBoostingRegressor(
            max_iter=320,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=10.0,
            random_state=random_state,
        ),
        "extra_residual": ExtraTreesRegressor(
            n_estimators=360,
            max_features=0.80,
            min_samples_leaf=8,
            n_jobs=-1,
            random_state=random_state,
        ),
        "gbr_residual": GradientBoostingRegressor(
            n_estimators=320,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=20,
            loss="huber",
            random_state=random_state,
        ),
    }


class ResidualStackRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        residual_weights: Optional[Dict[str, float]] = None,
        random_state: int = 2026,
        clip_min: float = 0.0,
        clip_max: float = 100.0,
    ) -> None:
        self.residual_weights = residual_weights
        self.random_state = random_state
        self.clip_min = clip_min
        self.clip_max = clip_max

    def fit(self, X: pd.DataFrame, y: Iterable[float]):
        X_engineered = engineer_features(X)
        self.input_columns_ = list(normalize_column_names(X).columns)
        self.feature_columns_ = list(X_engineered.columns)
        self.preprocessor_ = build_preprocessor(X_engineered)
        Z = self.preprocessor_.fit_transform(X_engineered)
        y_array = np.asarray(y, dtype=float)

        self.linear_model_ = Ridge(alpha=0.1)
        self.linear_model_.fit(Z, y_array)
        linear_train = self.linear_model_.predict(Z)
        residual_target = y_array - linear_train

        self.residual_models_ = make_residual_models(self.random_state)
        for model in self.residual_models_.values():
            model.fit(Z, residual_target)

        default_weights = {
            "hist_residual": 0.0,
            "extra_residual": 1.0,
            "gbr_residual": 0.0,
        }
        self.residual_weights_ = dict(default_weights)
        if self.residual_weights is not None:
            self.residual_weights_.update(self.residual_weights)
        return self

    def predict_components(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        X_engineered = engineer_features(X)
        missing = [c for c in self.feature_columns_ if c not in X_engineered.columns]
        if missing:
            raise ValueError(f"输入缺少训练所需字段: {missing}")
        X_engineered = X_engineered[self.feature_columns_]
        Z = self.preprocessor_.transform(X_engineered)
        linear = self.linear_model_.predict(Z)
        components = {"linear": linear}
        for name, model in self.residual_models_.items():
            components[name] = linear + model.predict(Z)
        return components

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_engineered = engineer_features(X)
        missing = [c for c in self.feature_columns_ if c not in X_engineered.columns]
        if missing:
            raise ValueError(f"输入缺少训练所需字段: {missing}")
        X_engineered = X_engineered[self.feature_columns_]
        Z = self.preprocessor_.transform(X_engineered)
        pred = self.linear_model_.predict(Z)
        for name, model in self.residual_models_.items():
            pred = pred + self.residual_weights_.get(name, 0.0) * model.predict(Z)
        return np.clip(pred, self.clip_min, self.clip_max)


@dataclass
class ValidationArtifacts:
    person_ids: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    fold_ids: np.ndarray

