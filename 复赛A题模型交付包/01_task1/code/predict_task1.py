from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = HERE / "model" / "model.joblib"


def prepare_input(df: pd.DataFrame, model):
    if "Person_ID" not in df.columns:
        raise ValueError("输入缺少 Person_ID")

    features = list(model.feature_names_in_)
    missing = [name for name in features if name not in df.columns]
    if missing:
        raise ValueError("输入缺少模型字段: " + ", ".join(missing))

    preprocess = model.named_steps["preprocess"]
    numeric = list(preprocess.transformers_[0][2])
    categorical = list(preprocess.transformers_[1][2])

    x = df[features].copy()
    for name in numeric:
        x[name] = pd.to_numeric(x[name], errors="coerce")
    for name in categorical:
        x[name] = x[name].replace("", np.nan)
    return x


def main():
    parser = argparse.ArgumentParser(description="任务一睡眠质量评分预测")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False)
    x = prepare_input(df, model)
    pred = model.predict(x)

    result = pd.DataFrame({
        "Person_ID": df["Person_ID"].astype(str),
        "predicted_value": pred,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
