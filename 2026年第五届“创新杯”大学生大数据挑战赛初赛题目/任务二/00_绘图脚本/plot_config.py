from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


CM = 1 / 2.54

MODEL_COLORS = {
    "Dummy_Most_Frequent": "#B6BEC8",
    "Multinomial_Logistic": "#6D7F9B",
    "Ordinal_Cumulative": "#7556A3",
    "LightGBM_Tuned": "#3B9B8F",
    "Stacking": "#176B87",
}

MODEL_LABELS = {
    "Dummy_Most_Frequent": "多数类基线",
    "Multinomial_Logistic": "多项逻辑回归",
    "Ordinal_Cumulative": "Ordinal Logistic",
    "LightGBM_Tuned": "LightGBM",
    "Stacking": "Stacking融合",
}

CLASS_ORDER = ["Poor", "Average", "Good", "Excellent"]
CLASS_COLORS = {
    "Poor": "#B24745",
    "Average": "#E5A84B",
    "Good": "#4E91A8",
    "Excellent": "#398564",
}

MAIN_BLUE = "#376795"
TEAL = "#3B9B8F"
ORANGE = "#E07B39"
DARK_RED = "#B24745"
LIGHT_GREY = "#E8ECEF"
MID_GREY = "#AAB3BA"
DARK_INK = "#263238"


FEATURE_LABELS = {
    "BMI": "BMI",
    "Sleep_Quality_Score": "睡眠质量评分",
    "Fast_Food_Meals_Per_Week": "每周快餐次数",
    "Daily_Steps": "每日步数",
    "Exercise_Frequency_Per_Week": "每周运动频次",
    "Mood_Score": "情绪评分",
    "Stress_Level": "压力水平",
    "Energy_Level_Score": "精力评分",
    "Age": "年龄",
    "Sleep_Duration_Hours": "睡眠时长",
    "Number_of_Night_Awakenings": "夜间醒来次数",
    "Weekend_Sleep_Difference_Hours": "周末睡眠差异",
    "Screen_Time_Before_Bed_Hours": "睡前屏幕时间",
    "Exercise_Duration_Minutes": "运动时长",
    "Water_Intake_Liters": "饮水量",
    "Fruit_Intake_Per_Day": "每日水果摄入",
    "Vegetable_Intake_Per_Day": "每日蔬菜摄入",
    "Protein_Intake_Grams": "蛋白质摄入",
    "Breakfast_Regularity_Score": "早餐规律评分",
    "Smoking_Status_Current": "当前吸烟",
    "Smoking_Status_Former": "曾经吸烟",
    "Smoking_Status_Never": "从不吸烟",
    "Obesity_Risk_High": "高肥胖风险",
    "Obesity_Risk_Low": "低肥胖风险",
    "Alcohol_Consumption_Heavy": "重度饮酒",
    "Alcohol_Consumption_Moderate": "中度饮酒",
    "Morning_Workout_Yes": "晨练：是",
    "Morning_Workout_No": "晨练：否",
    "Hypertension_Risk_High": "高血压高风险",
    "Unhealthy_Diet_Exposure": "不健康饮食暴露",
    "Mental_Risk_Composite": "心理风险综合指标",
    "Weekly_Exercise_Minutes": "每周运动分钟数",
    "Immune_Health_Score": "免疫健康评分",
    "Fruit_Vegetable_Intake": "果蔬摄入",
    "Disease_Risk_Composite": "疾病风险综合指标",
    "Productivity_Score": "生产力评分",
    "Energy_Fatigue_Gap": "精力疲劳差值",
    "Steps_Per_Sitting_Hour": "每坐一小时步数",
    "Cardiovascular_Risk_Low": "心血管低风险",
    "Focus_Concentration_Score": "专注力评分",
    "Life_Satisfaction_Score": "生活满意度",
    "Depression_Risk_Score": "抑郁风险评分",
    "Anxiety_Score": "焦虑评分",
    "Fatigue_Level_Score": "疲劳水平评分",
    "Blood_Pressure": "血压",
    "Weight_kg": "体重",
}


def clean_feature(name: str) -> str:
    cleaned = str(name).replace("num__", "").replace("cat__", "")
    return FEATURE_LABELS.get(cleaned, cleaned.replace("_", " "))


def apply_style() -> None:
    mpl.rcParams.update(
        {
            # Matplotlib does not reliably fall back per glyph on Windows.
            # Use a CJK-capable family globally so Chinese never becomes tofu;
            # math text still uses STIX/Times-like glyphs below.
            "font.family": "Microsoft YaHei",
            "font.size": 9.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.edgecolor": "#5D6870",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.8,
            "legend.frameon": False,
            "grid.color": "#B9C1C7",
            "grid.alpha": 0.28,
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 600) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png, pdf]


def add_source_note(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.005, text, ha="left", va="bottom", fontsize=7.2, color="#66757F")


def quiet_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
