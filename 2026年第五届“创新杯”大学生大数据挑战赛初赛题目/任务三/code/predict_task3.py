from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


CLASS_ORDER = ["Average", "Good", "Excellent"]
DIRECT = ["Health_Score", "Fitness_Level"]


def time_to_minutes(series: pd.Series) -> pd.Series:
    parts = series.astype(str).str.split(":", n=1, expand=True)
    return (
        pd.to_numeric(parts[0], errors="coerce") * 60
        + pd.to_numeric(parts[1], errors="coerce")
    )


def engineer(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    x = df.drop(columns=["Wellness_Category", "Person_ID"], errors="ignore").copy()
    if scope == "NO_DIRECT":
        x = x.drop(columns=DIRECT, errors="ignore")

    for column in ["Exercise_Type", "Workout_Intensity"]:
        if column in x:
            x[column] = x[column].fillna("None")
    if "Alcohol_Consumption" in x:
        x["Alcohol_Consumption_is_missing"] = x["Alcohol_Consumption"].isna().astype(int)
        x["Alcohol_Consumption"] = x["Alcohol_Consumption"].fillna("Unknown")

    wake = time_to_minutes(x["Wake_Up_Time"])
    sleep = time_to_minutes(x["Sleep_Time"])
    x["wake_sin"] = np.sin(2 * np.pi * wake / 1440)
    x["wake_cos"] = np.cos(2 * np.pi * wake / 1440)
    x["sleep_sin"] = np.sin(2 * np.pi * sleep / 1440)
    x["sleep_cos"] = np.cos(2 * np.pi * sleep / 1440)
    x["cross_midnight"] = (sleep > wake).astype(int)
    calculated_sleep = ((wake - sleep) % 1440) / 60
    x["calculated_sleep_duration"] = calculated_sleep
    x["sleep_duration_difference"] = (
        calculated_sleep - x["Sleep_Duration_Hours"]
    ).abs()
    x = x.drop(columns=["Wake_Up_Time", "Sleep_Time"])

    eps = 1e-6
    x["sleep_interruption_density"] = (
        x["Number_of_Night_Awakenings"] / (x["Sleep_Duration_Hours"] + eps)
    )
    x["screen_sleep_ratio"] = (
        x["Screen_Time_Before_Bed_Hours"] / (x["Sleep_Duration_Hours"] + eps)
    )
    x["sleep_duration_deviation"] = (x["Sleep_Duration_Hours"] - 8).abs()
    x["weekly_exercise_minutes"] = (
        x["Exercise_Frequency_Per_Week"] * x["Exercise_Duration_Minutes"]
    )
    x["steps_per_sitting_hour"] = (
        x["Daily_Steps"] / (x["Sitting_Hours_Per_Day"] + eps)
    )
    x["fruit_vegetable_total"] = (
        x["Fruit_Intake_Per_Day"] + x["Vegetable_Intake_Per_Day"]
    )
    x["sugary_fastfood_index"] = (
        x["Sugary_Drinks_Per_Week"] + x["Fast_Food_Meals_Per_Week"]
    )
    x["protein_per_kg"] = x["Protein_Intake_Grams"] / (x["Weight_kg"] + eps)
    x["water_per_kg"] = x["Water_Intake_Liters"] / (x["Weight_kg"] + eps)
    x["anxiety_depression_index"] = (
        x["Anxiety_Score"] + x["Depression_Risk_Score"]
    )
    x["mood_satisfaction_index"] = (
        x["Mood_Score"] + x["Life_Satisfaction_Score"]
    )
    x["stress_work_interaction"] = (
        x["Stress_Level"] * x["Working_Hours_Per_Day"]
    )
    x["energy_fatigue_difference"] = (
        x["Energy_Level_Score"] - x["Fatigue_Level_Score"]
    )
    x["pulse_pressure"] = x["Systolic_BP"] - x["Diastolic_BP"]
    x["mean_arterial_pressure"] = (
        x["Systolic_BP"] + 2 * x["Diastolic_BP"]
    ) / 3
    x["bmi_recalculated"] = (
        x["Weight_kg"] / ((x["Height_cm"] / 100) ** 2 + eps)
    )
    return x.replace([np.inf, -np.inf], np.nan)


def aligned_probability(model, x: pd.DataFrame) -> np.ndarray:
    probability = model.predict_proba(x)
    classifier = model.named_steps["clf"]
    classes = list(classifier.classes_)
    return probability[:, [classes.index(label) for label in CLASS_ORDER]]


def predict_full(df: pd.DataFrame, model_dir: Path) -> np.ndarray:
    model = joblib.load(model_dir / "FINAL_ExtraTrees_FULL_3C_all10000.joblib")
    return aligned_probability(model, engineer(df, "FULL"))


def predict_no_direct(df: pd.DataFrame, model_dir: Path) -> np.ndarray:
    manifest = json.loads(
        (model_dir / "final_model_manifest.json").read_text(encoding="utf-8")
    )
    weights = manifest["fusion_weights"]
    x = engineer(df, "NO_DIRECT")
    probability = np.zeros((len(df), len(CLASS_ORDER)), dtype=float)
    for name, weight in weights.items():
        if weight <= 0:
            continue
        model = joblib.load(
            model_dir / f"FINAL_{name}_NO_DIRECT_3C_all10000.joblib"
        )
        probability += float(weight) * aligned_probability(model, x)
    return probability


def main() -> None:
    parser = argparse.ArgumentParser(
        description="任务三综合健康类别三分类预测"
    )
    parser.add_argument("--input", required=True, help="输入CSV路径")
    parser.add_argument("--output", required=True, help="输出CSV路径")
    parser.add_argument(
        "--mode",
        choices=["competition", "explanation"],
        default="competition",
        help="competition使用FULL模型；explanation使用NO_DIRECT融合模型",
    )
    parser.add_argument(
        "--model-dir",
        default=str(Path(__file__).resolve().parents[1] / "models"),
        help="模型目录",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_dir = Path(args.model_dir)
    frame = pd.read_csv(input_path)
    person_id = (
        frame["Person_ID"].copy()
        if "Person_ID" in frame
        else pd.Series(np.arange(1, len(frame) + 1), name="Person_ID")
    )

    if args.mode == "competition":
        probability = predict_full(frame, model_dir)
    else:
        probability = predict_no_direct(frame, model_dir)

    predicted = np.asarray(CLASS_ORDER)[probability.argmax(axis=1)]
    result = pd.DataFrame(
        {
            "Person_ID": person_id,
            "Predicted_Wellness_Category": predicted,
            **{
                f"Probability_{label}": probability[:, index]
                for index, label in enumerate(CLASS_ORDER)
            },
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(result)} predictions to {output_path}")


if __name__ == "__main__":
    main()
