from __future__ import annotations

import json
import platform
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from src.modeling import (
    COMPOSITE_PROXY_COLUMNS,
    ID_COLUMN,
    PREDICTION_EXCLUDED_COLUMNS,
    TARGET_COLUMN,
    TabularRegressor,
    engineer_features,
    normalize_column_names,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "A_dataset.csv"
MODEL_DIR = ROOT / "models"
PRED_DIR = ROOT / "predictions"
FIG_DIR = ROOT / "figures"
RANDOM_STATE = 2026
HOLDOUT_SIZE = 0.20
N_SPLITS = 5

ENERGY_TARGET = "Energy_Level_Score"
ENERGY_STATE_PROXY_COLUMNS = (
    TARGET_COLUMN,
    *COMPOSITE_PROXY_COLUMNS,
    "Fatigue_Level_Score",
    "Immune_Health_Score",
    "Mood_Score",
    "Anxiety_Score",
    "Depression_Risk_Score",
    "Focus_Concentration_Score",
    "Life_Satisfaction_Score",
)

def adjusted_r2(r2: float, n: int, p: int) -> float:
    if n <= p + 1:
        raise ValueError(f"调整 R² 无法计算: n={n}, p={p}")
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))


def regression_metrics(y_true, y_pred, p_transformed: int, p_raw: int) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = float(r2_score(y_true, y_pred))
    return {
        "R2": r2,
        "Adjusted_R2_Transformed": adjusted_r2(r2, len(y_true), p_transformed),
        "Adjusted_R2_Raw_Groups": adjusted_r2(r2, len(y_true), p_raw),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def candidate_models() -> dict[str, TabularRegressor]:
    return {
        "Ridge α=0.1": TabularRegressor(Ridge(alpha=0.1)),
        "Ridge α=1": TabularRegressor(Ridge(alpha=1.0)),
        "Ridge α=10": TabularRegressor(Ridge(alpha=10.0)),
        "Ridge α=100": TabularRegressor(Ridge(alpha=100.0)),
        "ElasticNet": TabularRegressor(
            ElasticNet(
                alpha=0.002,
                l1_ratio=0.10,
                max_iter=20000,
                random_state=RANDOM_STATE,
            )
        ),
        "HistGradientBoosting": TabularRegressor(
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.04,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=5.0,
                random_state=RANDOM_STATE,
            )
        ),
        "ExtraTrees": TabularRegressor(
            ExtraTreesRegressor(
                n_estimators=300,
                max_features=0.80,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
    }


def evaluate_oof(
    name: str,
    prototype: TabularRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    started = perf_counter()
    oof = np.zeros(len(X), dtype=float)
    fold_ids = np.zeros(len(X), dtype=int)
    fold_models = []
    fold_rows = []
    p_transformed = 0
    p_raw = 0

    for fold_number, (train_idx, valid_idx) in enumerate(folds, start=1):
        model = clone(prototype)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[valid_idx])
        oof[valid_idx] = pred
        fold_ids[valid_idx] = fold_number
        p_transformed = max(p_transformed, model.transformed_feature_count_)
        p_raw = max(p_raw, model.raw_feature_count_)
        fold_models.append(model)
        fold_rows.append(
            {
                "Model": name,
                "Fold": fold_number,
                "N": int(len(valid_idx)),
                "P_Transformed": int(model.transformed_feature_count_),
                "P_Raw_Groups": int(model.raw_feature_count_),
                **regression_metrics(
                    y.iloc[valid_idx],
                    pred,
                    model.transformed_feature_count_,
                    model.raw_feature_count_,
                ),
            }
        )

    aggregate = regression_metrics(y, oof, p_transformed, p_raw)
    aggregate.update(
        {
            "Model": name,
            "N": int(len(X)),
            "P_Transformed": int(p_transformed),
            "P_Raw_Groups": int(p_raw),
            "Fit_Seconds": float(perf_counter() - started),
        }
    )
    return {
        "name": name,
        "prototype": prototype,
        "oof": oof,
        "fold_ids": fold_ids,
        "fold_models": fold_models,
        "fold_rows": fold_rows,
        "aggregate": aggregate,
    }


def grouped_permutation_importance(
    models: list[TabularRegressor],
    folds: list[tuple[np.ndarray, np.ndarray]],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_fold_rows = []
    used_columns = list(models[0].used_input_columns_)

    for fold_number, (model, (_, valid_idx)) in enumerate(zip(models, folds), start=1):
        X_valid = X.iloc[valid_idx].copy()
        y_valid = y.iloc[valid_idx]
        baseline = float(r2_score(y_valid, model.predict(X_valid)))
        for feature_number, feature in enumerate(used_columns):
            drops = []
            for repeat in range(repeats):
                rng = np.random.default_rng(
                    random_state + 10000 * fold_number + 100 * feature_number + repeat
                )
                shuffled = X_valid.copy()
                shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
                score = float(r2_score(y_valid, model.predict(shuffled)))
                drops.append(baseline - score)
            per_fold_rows.append(
                {
                    "Fold": fold_number,
                    "Feature": feature,
                    "Importance_Mean": float(np.mean(drops)),
                    "Importance_STD_Repeats": float(np.std(drops, ddof=1)),
                }
            )

    per_fold = pd.DataFrame(per_fold_rows)
    summary = (
        per_fold.groupby("Feature", as_index=False)
        .agg(
            Importance_Mean=("Importance_Mean", "mean"),
            Importance_STD_Folds=("Importance_Mean", "std"),
            Min_Fold_Importance=("Importance_Mean", "min"),
            Max_Fold_Importance=("Importance_Mean", "max"),
        )
        .sort_values("Importance_Mean", ascending=False)
    )
    return summary, per_fold


def raw_feature_from_transformed(name: str, used_columns: list[str]) -> str:
    if name.startswith("Wake_Up_Time_"):
        return "Wake_Up_Time"
    if name.startswith("Sleep_Time_"):
        return "Sleep_Time"
    for column in sorted(used_columns, key=len, reverse=True):
        if name == column or name.startswith(f"{column}_"):
            return column
    return name


def coefficient_stability(models: list[TabularRegressor]) -> pd.DataFrame:
    rows = []
    for fold_number, model in enumerate(models, start=1):
        if not hasattr(model.estimator_, "coef_"):
            continue
        grouped = defaultdict(list)
        for feature_name, value in zip(model.feature_names_out(), model.estimator_.coef_):
            raw_name = raw_feature_from_transformed(feature_name, model.used_input_columns_)
            grouped[raw_name].append(float(value))
        for feature, values in grouped.items():
            rows.append(
                {
                    "Fold": fold_number,
                    "Feature": feature,
                    "Coefficient_L2": float(np.sqrt(np.sum(np.square(values)))),
                    "Coefficient_Sum": float(np.sum(values)),
                }
            )
    per_fold = pd.DataFrame(rows)
    if per_fold.empty:
        return per_fold
    return (
        per_fold.groupby("Feature", as_index=False)
        .agg(
            Coefficient_L2_Mean=("Coefficient_L2", "mean"),
            Coefficient_L2_STD=("Coefficient_L2", "std"),
            Coefficient_Sum_Mean=("Coefficient_Sum", "mean"),
            Coefficient_Sum_STD=("Coefficient_Sum", "std"),
        )
        .sort_values("Coefficient_L2_Mean", ascending=False)
    )


def multicollinearity_audit(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized = normalize_column_names(X)
    normalized = normalized.drop(columns=list(PREDICTION_EXCLUDED_COLUMNS), errors="ignore")
    numeric = normalized.select_dtypes(include=np.number).copy()
    corr = numeric.corr()
    pairs = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = float(corr.loc[left, right])
            if abs(value) >= 0.75:
                pairs.append(
                    {
                        "Feature_A": left,
                        "Feature_B": right,
                        "Pearson_r": value,
                        "Abs_Pearson_r": abs(value),
                    }
                )
    pair_df = pd.DataFrame(pairs).sort_values("Abs_Pearson_r", ascending=False)

    engineered = engineer_features(normalized)
    numeric_engineered = engineered.select_dtypes(include=np.number).copy()
    numeric_engineered = numeric_engineered.loc[:, numeric_engineered.nunique() > 1]
    numeric_engineered = numeric_engineered.fillna(numeric_engineered.median())
    z = (numeric_engineered - numeric_engineered.mean()) / numeric_engineered.std(ddof=0)
    correlation = np.corrcoef(z.to_numpy(dtype=float), rowvar=False)
    inverse = np.linalg.pinv(correlation, rcond=1e-10)
    vif = pd.DataFrame(
        {
            "Feature": numeric_engineered.columns,
            "VIF": np.diag(inverse),
        }
    ).sort_values("VIF", ascending=False)
    return pair_df, vif


def configure_plots() -> None:
    from matplotlib import font_manager

    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            family = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = family
            break
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#3B4654",
            "text.color": "#1D2733",
            "axes.labelcolor": "#1D2733",
            "xtick.color": "#4D5967",
            "ytick.color": "#4D5967",
            "grid.color": "#D8DEE6",
            "grid.alpha": 0.65,
        }
    )


def save_figures(
    df: pd.DataFrame,
    comparison: pd.DataFrame,
    y_holdout: pd.Series,
    holdout_pred: np.ndarray,
    holdout_metric_values: dict[str, float],
    importance: pd.DataFrame,
    energy_importance: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    ablation: pd.DataFrame,
) -> None:
    configure_plots()
    navy, blue, gold, gray, pale = "#17365D", "#4B78A6", "#C59A3D", "#8A96A3", "#DDE7F0"

    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    sns.histplot(df[TARGET_COLUMN], bins=30, kde=True, color=blue, ax=ax)
    ax.axvline(df[TARGET_COLUMN].mean(), color=gold, lw=2, label=f"均值 {df[TARGET_COLUMN].mean():.2f}")
    ax.set(title="生产力评分分布", xlabel="Productivity Score（分）", ylabel="样本数")
    ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "Fig01_productivity_distribution.png", bbox_inches="tight")
    plt.close(fig)

    focus = [
        TARGET_COLUMN,
        "Energy_Level_Score",
        "Fatigue_Level_Score",
        "Mood_Score",
        "Sleep_Quality_Score",
        "Anxiety_Score",
        "Depression_Risk_Score",
        "Life_Satisfaction_Score",
        "Stress_Level",
        "Exercise_Frequency_Per_Week",
        "Daily_Steps",
        "Working_Hours_Per_Day",
        "Sitting_Hours_Per_Day",
    ]
    corr = df[focus].corr()
    fig, ax = plt.subplots(figsize=(10.0, 8.2), constrained_layout=True)
    sns.heatmap(
        corr,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.45,
        cbar_kws={"shrink": 0.78},
        ax=ax,
    )
    ax.set_title("生产力及主要连续指标的 Pearson 相关矩阵", pad=12)
    ax.tick_params(axis="x", rotation=48, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.savefig(FIG_DIR / "Fig02_multicollinearity_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    comp = comparison.sort_values("Adjusted_R2_Transformed")
    colors = [navy if value else gray for value in comp["Selected"]]
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    bars = ax.barh(comp["Model"], comp["Adjusted_R2_Transformed"], color=colors)
    ax.set_xlim(0, min(0.72, comp["Adjusted_R2_Transformed"].max() + 0.06))
    ax.set(title="候选模型五折 OOF 调整 R²", xlabel="调整 R²（按变换后维数）", ylabel="")
    ax.grid(axis="x")
    for bar, value in zip(bars, comp["Adjusted_R2_Transformed"]):
        ax.text(value + 0.006, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=9)
    fig.savefig(FIG_DIR / "Fig03_model_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.8), constrained_layout=True)
    ax.scatter(y_holdout, holdout_pred, s=16, alpha=0.38, color=blue, edgecolors="none")
    low = min(float(y_holdout.min()), float(holdout_pred.min()))
    high = max(float(y_holdout.max()), float(holdout_pred.max()))
    ax.plot([low, high], [low, high], color=gold, lw=2, label="理想预测线")
    metric = holdout_metric_values
    ax.text(
        0.035,
        0.955,
        f"R² = {metric['R2']:.4f}\n调整 R² = {metric['Adjusted_R2_Transformed']:.4f}\nRMSE = {metric['RMSE']:.4f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": "#CDD4DD"},
    )
    ax.set(title="锁定验证集真实值与预测值", xlabel="真实 Productivity Score", ylabel="预测 Productivity Score")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True)
    fig.savefig(FIG_DIR / "Fig04_true_vs_predicted.png", bbox_inches="tight")
    plt.close(fig)

    residual = y_holdout.to_numpy() - holdout_pred
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    axes[0].scatter(holdout_pred, residual, s=15, alpha=0.38, color=blue, edgecolors="none")
    axes[0].axhline(0, color=gold, lw=2)
    axes[0].set(title="残差与预测值", xlabel="预测 Productivity Score", ylabel="残差（真实值-预测值）")
    axes[0].grid(True)
    sns.histplot(residual, bins=30, kde=True, color=blue, ax=axes[1])
    axes[1].axvline(0, color=gold, lw=2)
    axes[1].set(title="残差分布", xlabel="残差", ylabel="样本数")
    fig.savefig(FIG_DIR / "Fig05_residual_diagnostics.png", bbox_inches="tight")
    plt.close(fig)

    top = importance.head(15).sort_values("Importance_Mean")
    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    ax.barh(
        top["Feature"],
        top["Importance_Mean"],
        xerr=top["Importance_STD_Folds"].fillna(0),
        color=blue,
        ecolor=gray,
        capsize=2,
    )
    ax.axvline(0, color="#3B4654", lw=1)
    ax.set(title="生产力模型的分组置换重要性", xlabel="验证 R² 平均下降量", ylabel="")
    ax.grid(axis="x")
    fig.savefig(FIG_DIR / "Fig06_productivity_permutation_importance.png", bbox_inches="tight")
    plt.close(fig)

    energy_top = energy_importance.head(15).sort_values("Importance_Mean")
    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    ax.barh(
        energy_top["Feature"],
        energy_top["Importance_Mean"],
        xerr=energy_top["Importance_STD_Folds"].fillna(0),
        color=navy,
        ecolor=gray,
        capsize=2,
    )
    ax.axvline(0, color="#3B4654", lw=1)
    ax.set(title="精力评分辅助模型的生活方式变量重要性", xlabel="验证 R² 平均下降量", ylabel="")
    ax.grid(axis="x")
    fig.savefig(FIG_DIR / "Fig07_energy_driver_importance.png", bbox_inches="tight")
    plt.close(fig)

    selected_folds = fold_metrics[fold_metrics["Selected"]].sort_values("Fold")
    fig, ax = plt.subplots(figsize=(7.8, 4.5), constrained_layout=True)
    bars = ax.bar(selected_folds["Fold"].astype(str), selected_folds["Adjusted_R2_Transformed"], color=blue)
    mean_value = selected_folds["Adjusted_R2_Transformed"].mean()
    ax.axhline(mean_value, color=gold, ls="--", lw=1.8, label=f"折均值 {mean_value:.4f}")
    for bar, value in zip(bars, selected_folds["Adjusted_R2_Transformed"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.4f}", ha="center", fontsize=8)
    ax.set_ylim(0, min(0.75, selected_folds["Adjusted_R2_Transformed"].max() + 0.08))
    ax.set(title="入选模型五折验证稳定性", xlabel="折次", ylabel="调整 R²")
    ax.legend(frameon=False)
    ax.grid(axis="y")
    fig.savefig(FIG_DIR / "Fig08_fold_stability.png", bbox_inches="tight")
    plt.close(fig)

    scope = ablation.sort_values("Adjusted_R2_Transformed")
    fig, ax = plt.subplots(figsize=(8.9, 4.6), constrained_layout=True)
    colors = [pale, blue, navy]
    bars = ax.barh(scope["Scope"], scope["Adjusted_R2_Transformed"], color=colors[: len(scope)])
    ax.set_xlim(0, min(0.75, scope["Adjusted_R2_Transformed"].max() + 0.07))
    ax.set(title="特征口径敏感性分析", xlabel="开发集 OOF 调整 R²", ylabel="")
    ax.grid(axis="x")
    for bar, value in zip(bars, scope["Adjusted_R2_Transformed"]):
        ax.text(value + 0.006, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=9)
    fig.savefig(FIG_DIR / "Fig09_scope_ablation.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for directory in [MODEL_DIR, PRED_DIR, FIG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    df = normalize_column_names(pd.read_csv(DATA_PATH))
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"训练数据必须包含目标列 {TARGET_COLUMN}")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Person_ID 必须唯一")

    train_indices, holdout_indices = train_test_split(
        np.arange(len(df)),
        test_size=HOLDOUT_SIZE,
        random_state=RANDOM_STATE,
    )
    development = df.iloc[train_indices].reset_index(drop=True)
    holdout = df.iloc[holdout_indices].reset_index(drop=True)
    X_development = development.drop(columns=[TARGET_COLUMN])
    y_development = development[TARGET_COLUMN]
    X_holdout = holdout.drop(columns=[TARGET_COLUMN])
    y_holdout = holdout[TARGET_COLUMN]

    folds = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(X_development))
    results = {}
    for name, prototype in candidate_models().items():
        print(f"正在评估 {name} ...", flush=True)
        results[name] = evaluate_oof(name, prototype, X_development, y_development, folds)

    comparison = pd.DataFrame([result["aggregate"] for result in results.values()])
    best_adjusted = float(comparison["Adjusted_R2_Transformed"].max())
    ridge_anchor = "Ridge α=10"
    ridge_adjusted = float(
        comparison.loc[comparison["Model"] == ridge_anchor, "Adjusted_R2_Transformed"].iloc[0]
    )
    selected_name = ridge_anchor if best_adjusted - ridge_adjusted <= 0.001 else str(
        comparison.sort_values("Adjusted_R2_Transformed", ascending=False).iloc[0]["Model"]
    )
    comparison["Selected"] = comparison["Model"].eq(selected_name)
    comparison = comparison.sort_values("Adjusted_R2_Transformed", ascending=False)

    fold_metrics = pd.DataFrame(
        [row for result in results.values() for row in result["fold_rows"]]
    )
    fold_metrics["Selected"] = fold_metrics["Model"].eq(selected_name)
    selected = results[selected_name]

    holdout_model = clone(selected["prototype"])
    holdout_model.fit(X_development, y_development)
    holdout_pred = holdout_model.predict(X_holdout)
    holdout_metrics = regression_metrics(
        y_holdout,
        holdout_pred,
        holdout_model.transformed_feature_count_,
        holdout_model.raw_feature_count_,
    )

    validation_predictions = pd.DataFrame(
        {
            ID_COLUMN: holdout[ID_COLUMN].to_numpy(),
            "True_Productivity_Score": y_holdout.to_numpy(),
            "Predicted_Productivity_Score": np.round(holdout_pred, 6),
            "Residual": np.round(y_holdout.to_numpy() - holdout_pred, 6),
        }
    )
    validation_predictions.to_csv(
        PRED_DIR / "task2_validation_predictions.csv", index=False, encoding="utf-8-sig"
    )

    oof_predictions = pd.DataFrame(
        {
            ID_COLUMN: development[ID_COLUMN].to_numpy(),
            "Fold": selected["fold_ids"],
            "True_Productivity_Score": y_development.to_numpy(),
            "OOF_Predicted_Productivity_Score": np.round(selected["oof"], 6),
            "Residual": np.round(y_development.to_numpy() - selected["oof"], 6),
        }
    )
    oof_predictions.to_csv(
        PRED_DIR / "task2_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    comparison.to_csv(PRED_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(PRED_DIR / "fold_metrics.csv", index=False, encoding="utf-8-sig")

    importance, importance_by_fold = grouped_permutation_importance(
        selected["fold_models"],
        folds,
        X_development,
        y_development,
        repeats=5,
        random_state=RANDOM_STATE,
    )
    importance.to_csv(
        PRED_DIR / "permutation_importance.csv", index=False, encoding="utf-8-sig"
    )
    importance_by_fold.to_csv(
        PRED_DIR / "permutation_importance_by_fold.csv", index=False, encoding="utf-8-sig"
    )

    coefficient = coefficient_stability(selected["fold_models"])
    coefficient.to_csv(
        PRED_DIR / "ridge_coefficient_stability.csv", index=False, encoding="utf-8-sig"
    )

    correlation_pairs, vif = multicollinearity_audit(X_development)
    correlation_pairs.to_csv(
        PRED_DIR / "multicollinearity_pairs.csv", index=False, encoding="utf-8-sig"
    )
    vif.to_csv(PRED_DIR / "numeric_vif.csv", index=False, encoding="utf-8-sig")

    ablation_specs = {
        "主口径：剔除综合与并发状态评分": PREDICTION_EXCLUDED_COLUMNS,
        "仅剔除综合回流字段": COMPOSITE_PROXY_COLUMNS,
        "全字段性能对照": (),
    }
    ablation_rows = []
    for scope, exclusions in ablation_specs.items():
        prototype = TabularRegressor(Ridge(alpha=10.0), excluded_columns=exclusions)
        result = evaluate_oof(scope, prototype, X_development, y_development, folds)
        ablation_rows.append({"Scope": scope, **result["aggregate"]})
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(PRED_DIR / "scope_ablation.csv", index=False, encoding="utf-8-sig")

    energy_y_development = development[ENERGY_TARGET]
    energy_X_development = development.drop(columns=[ENERGY_TARGET])
    energy_prototype = TabularRegressor(
        Ridge(alpha=10.0),
        target_column=ENERGY_TARGET,
        excluded_columns=ENERGY_STATE_PROXY_COLUMNS,
    )
    energy_result = evaluate_oof(
        "Energy Ridge α=10",
        energy_prototype,
        energy_X_development,
        energy_y_development,
        folds,
    )
    energy_importance, energy_importance_by_fold = grouped_permutation_importance(
        energy_result["fold_models"],
        folds,
        energy_X_development,
        energy_y_development,
        repeats=5,
        random_state=RANDOM_STATE + 5000,
    )
    energy_importance.to_csv(
        PRED_DIR / "energy_driver_importance.csv", index=False, encoding="utf-8-sig"
    )
    energy_importance_by_fold.to_csv(
        PRED_DIR / "energy_driver_importance_by_fold.csv", index=False, encoding="utf-8-sig"
    )

    joblib.dump(holdout_model, MODEL_DIR / "task2_holdout_model.joblib", compress=3)
    final_model = clone(selected["prototype"])
    final_model.fit(df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN])
    joblib.dump(final_model, MODEL_DIR / "task2_final_model.joblib", compress=3)

    example_input = df.drop(columns=[TARGET_COLUMN]).iloc[holdout_indices[:40]].copy()
    example_input.to_csv(
        ROOT / "data" / "task2_test_input_example.csv", index=False, encoding="utf-8-sig"
    )
    example_output = pd.DataFrame(
        {
            ID_COLUMN: example_input[ID_COLUMN].to_numpy(),
            "Predicted_Productivity_Score": np.round(final_model.predict(example_input), 6),
        }
    )
    example_output.to_csv(
        PRED_DIR / "task2_test_output_example.csv", index=False, encoding="utf-8-sig"
    )

    save_figures(
        df,
        comparison,
        y_holdout,
        holdout_pred,
        holdout_metrics,
        importance,
        energy_importance,
        fold_metrics,
        ablation,
    )

    selected_oof_metrics = selected["aggregate"]
    summary = {
        "task": "复赛 A 题任务二：Productivity_Score 回归",
        "target": TARGET_COLUMN,
        "id_column": ID_COLUMN,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "random_state": RANDOM_STATE,
        "holdout_size": HOLDOUT_SIZE,
        "cv_folds": N_SPLITS,
        "selected_model": selected_name,
        "selection_rule": "若 Ridge α=10 的 OOF 调整 R² 距最佳值不超过 0.001，则优先采用 Ridge；否则采用最佳候选。",
        "excluded_prediction_proxies": list(PREDICTION_EXCLUDED_COLUMNS),
        "adjusted_r2_definition": {
            "headline": "变换后设计矩阵维数 p",
            "formula": "1-(1-R2)*(n-1)/(n-p-1)",
            "sensitivity": "同时保存按原始变量组数 p 计算的结果",
        },
        "oof_metrics": {
            key: float(value)
            for key, value in selected_oof_metrics.items()
            if key in {"R2", "Adjusted_R2_Transformed", "Adjusted_R2_Raw_Groups", "RMSE", "MAE"}
        },
        "holdout_metrics": holdout_metrics,
        "energy_auxiliary_oof_metrics": {
            key: float(value)
            for key, value in energy_result["aggregate"].items()
            if key in {"R2", "Adjusted_R2_Transformed", "Adjusted_R2_Raw_Groups", "RMSE", "MAE"}
        },
        "top_productivity_dependencies": importance.head(10).to_dict(orient="records"),
        "top_energy_lifestyle_dependencies": energy_importance.head(10).to_dict(orient="records"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "joblib": joblib.__version__,
        },
    }
    (PRED_DIR / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
