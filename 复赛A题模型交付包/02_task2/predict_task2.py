from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.modeling import ID_COLUMN, TARGET_COLUMN, normalize_column_names


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复赛 A 题任务二生产力评分预测")
    parser.add_argument("--input", required=True, help="待预测 CSV 路径")
    parser.add_argument("--output", required=True, help="预测结果 CSV 路径")
    parser.add_argument(
        "--model",
        default=str(ROOT / "models" / "task2_final_model.joblib"),
        help="Joblib 模型路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    model_path = Path(args.model).resolve()

    frame = normalize_column_names(pd.read_csv(input_path))
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"输入 CSV 必须包含 {ID_COLUMN}")
    model = joblib.load(model_path)
    prediction = np.asarray(model.predict(frame), dtype=float)

    result = pd.DataFrame(
        {
            ID_COLUMN: frame[ID_COLUMN].to_numpy(),
            "Predicted_Productivity_Score": np.round(prediction, 6),
        }
    )
    if TARGET_COLUMN in frame.columns:
        result.insert(1, "True_Productivity_Score", frame[TARGET_COLUMN].to_numpy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已写入 {len(result)} 条预测: {output_path}")


if __name__ == "__main__":
    main()
