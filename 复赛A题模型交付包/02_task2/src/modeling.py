from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_COLUMN = "Person_ID"
TARGET_COLUMN = "Productivity_Score"
TIME_COLUMNS = ("Wake_Up_Time", "Sleep_Time")

COMPOSITE_PROXY_COLUMNS = (
    "Health_Score",
    "Fitness_Level",
    "Healthy_Aging_Score",
    "Wellness_Category",
)

PREDICTION_EXCLUDED_COLUMNS = (
    *COMPOSITE_PROXY_COLUMNS,
    "Energy_Level_Score",
    "Fatigue_Level_Score",
    "Immune_Health_Score",
    "Mood_Score",
    "Anxiety_Score",
    "Depression_Risk_Score",
    "Focus_Concentration_Score",
    "Life_Satisfaction_Score",
)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip().replace(" ", "_") for col in out.columns]
    return out


def engineer_features(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    excluded_columns: Sequence[str] = PREDICTION_EXCLUDED_COLUMNS,
) -> pd.DataFrame:
    out = normalize_column_names(df)

    for col in TIME_COLUMNS:
        if col not in out.columns:
            continue
        parsed = pd.to_datetime(out[col].astype(str), format="%H:%M", errors="coerce")
        minutes = parsed.dt.hour * 60 + parsed.dt.minute
        out[f"{col}_sin"] = np.sin(2.0 * np.pi * minutes / 1440.0)
        out[f"{col}_cos"] = np.cos(2.0 * np.pi * minutes / 1440.0)
        out = out.drop(columns=[col])

    drop_columns = [ID_COLUMN, target_column, *excluded_columns]
    return out.drop(columns=[c for c in drop_columns if c in out.columns], errors="ignore")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
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


class TabularRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        *,
        target_column: str = TARGET_COLUMN,
        excluded_columns: Sequence[str] = PREDICTION_EXCLUDED_COLUMNS,
    ) -> None:
        self.estimator = estimator
        self.target_column = target_column
        self.excluded_columns = excluded_columns

    def fit(self, X: pd.DataFrame, y: Iterable[float]):
        raw = normalize_column_names(X)
        excluded = set(self.excluded_columns)
        self.used_input_columns_ = [
            col
            for col in raw.columns
            if col not in {ID_COLUMN, self.target_column} and col not in excluded
        ]
        engineered = engineer_features(
            raw[self.used_input_columns_],
            target_column=self.target_column,
            excluded_columns=self.excluded_columns,
        )
        self.feature_columns_ = list(engineered.columns)
        self.raw_feature_count_ = len(self.used_input_columns_)
        self.preprocessor_ = build_preprocessor(engineered)
        transformed = self.preprocessor_.fit_transform(engineered)
        self.transformed_feature_count_ = int(transformed.shape[1])
        base_estimator = self.estimator if self.estimator is not None else Ridge(alpha=10.0)
        self.estimator_ = clone(base_estimator)
        self.estimator_.fit(transformed, np.asarray(y, dtype=float))
        return self

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        raw = normalize_column_names(X)
        missing = [col for col in self.used_input_columns_ if col not in raw.columns]
        if missing:
            raise ValueError(f"输入缺少训练所需字段: {missing}")
        engineered = engineer_features(
            raw[self.used_input_columns_],
            target_column=self.target_column,
            excluded_columns=self.excluded_columns,
        )
        missing_engineered = [col for col in self.feature_columns_ if col not in engineered.columns]
        if missing_engineered:
            raise ValueError(f"特征工程后缺少字段: {missing_engineered}")
        return self.preprocessor_.transform(engineered[self.feature_columns_])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator_.predict(self._transform(X)), dtype=float)

    def feature_names_out(self) -> np.ndarray:
        return self.preprocessor_.get_feature_names_out()
