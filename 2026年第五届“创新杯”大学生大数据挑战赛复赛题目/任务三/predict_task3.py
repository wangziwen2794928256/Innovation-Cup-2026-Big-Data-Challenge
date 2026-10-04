from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="复赛A题任务三 Health_Score 预测")
    parser.add_argument("--input", required=True, help="待预测CSV路径")
    parser.add_argument("--output", required=True, help="预测结果CSV路径")
    parser.add_argument("--model", default="models/task3_final_model.joblib", help="模型文件路径")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)
    df = pd.read_csv(input_path)
    normalized = {str(c).strip().replace(" ", "_"): c for c in df.columns}
    id_source = normalized.get("Person_ID")
    if id_source is None:
        raise ValueError("输入CSV必须包含 Person_ID 列")

    model = joblib.load(model_path)
    pred = model.predict(df)
    result = pd.DataFrame(
        {
            "Person_ID": df[id_source].astype(str),
            "Predicted_Health_Score": pred.round(6),
        }
    )
    if "Health_Score" in normalized:
        result.insert(1, "True_Health_Score", df[normalized["Health_Score"]].to_numpy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已生成 {len(result)} 条预测：{output_path}")


if __name__ == "__main__":
    main()

