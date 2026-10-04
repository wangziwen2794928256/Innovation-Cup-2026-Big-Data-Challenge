from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HERE = Path(__file__).resolve().parents[1]


def make_model(config):
    features = config["features"]
    categorical = config["categorical_features"]
    numeric = [name for name in features if name not in categorical]

    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(
                strategy="constant", fill_value="__MISSING__", keep_empty_features=True
            )),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], remainder="drop")

    return Pipeline([
        ("preprocess", preprocess),
        ("regressor", Ridge(alpha=1.0, solver="svd", fit_intercept=True)),
    ])


def main():
    parser = argparse.ArgumentParser(description="复现任务一最终 Ridge 模型")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads((HERE / "model_config.json").read_text(encoding="utf-8"))
    x = pd.read_csv(
        HERE / "data" / "development_features.csv",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    y = pd.read_csv(
        HERE / "data" / "development_targets.csv",
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    if x["Person_ID"].astype(str).tolist() != y["Person_ID"].astype(str).tolist():
        raise ValueError("开发集特征与目标的 Person_ID 顺序不一致")

    features = config["features"]
    categorical = config["categorical_features"]
    numeric = [name for name in features if name not in categorical]

    data = x[features].copy()
    for name in numeric:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    for name in categorical:
        data[name] = data[name].replace("", np.nan)

    target = pd.to_numeric(y[config["target"]], errors="raise")
    model = make_model(config)
    model.fit(data, target)

    args.output.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, args.output / "model.joblib", compress=3)


if __name__ == "__main__":
    main()
