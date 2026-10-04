from __future__ import annotations

import json
import math
import shutil
import time
import traceback
import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from scipy.stats import zscore
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from umap import UMAP

from plot_config import (
    CLASS_COLORS,
    CLASS_ORDER,
    CM,
    DARK_INK,
    DARK_RED,
    LIGHT_GREY,
    MAIN_BLUE,
    MID_GREY,
    MODEL_COLORS,
    MODEL_LABELS,
    ORANGE,
    TEAL,
    add_source_note,
    apply_style,
    clean_feature,
    quiet_spines,
    save_figure,
)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parents[1]
IMAGE_ROOT = PROJECT / "图片"
OUTPUT_ROOT = IMAGE_ROOT / "任务二正式图件"
DATA_OUT = IMAGE_ROOT / "00_绘图数据"
STRICT = PROJECT / "experiments" / "strict_no_proxy"
COMP = PROJECT / "experiments" / "competition_full"
RAW_PATH = PROJECT / "data" / "A题数据集.csv"
SPLIT_PATH = PROJECT / "data_audit" / "split_manifest.csv"

DIRS = {
    "data": OUTPUT_ROOT / "01_数据与标签",
    "tuning": OUTPUT_ROOT / "02_训练调参",
    "performance": OUTPUT_ROOT / "03_模型性能",
    "error": OUTPUT_ROOT / "04_误差与校准",
    "explain": OUTPUT_ROOT / "05_可解释性",
    "structure": OUTPUT_ROOT / "06_特征关系与降维",
    "route": OUTPUT_ROOT / "07_技术路线",
    "scope": OUTPUT_ROOT / "08_竞赛口径对照",
}

MODEL_ORDER = [
    "Dummy_Most_Frequent",
    "Multinomial_Logistic",
    "Ordinal_Cumulative",
    "LightGBM_Tuned",
    "Stacking",
]
MAIN_MODELS = ["Multinomial_Logistic", "Ordinal_Cumulative", "LightGBM_Tuned", "Stacking"]
THREE_MODELS = ["Ordinal_Cumulative", "LightGBM_Tuned", "Stacking"]

MANIFEST: list[dict[str, str]] = []


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def register(stem: str, title: str, purpose: str, source: str, folder: str, note: str = "") -> None:
    MANIFEST.append(
        {
            "file_stem": stem,
            "title": title,
            "purpose": purpose,
            "source_data": source,
            "folder": folder,
            "note": note,
        }
    )


def ship(fig, key: str, stem: str, title: str, purpose: str, source: str, note: str = "") -> None:
    add_source_note(fig, f"数据来源：{source}；主口径：strict_no_proxy；随机种子：2026")
    save_figure(fig, DIRS[key], stem)
    register(stem, title, purpose, source, DIRS[key].name, note)


def make_dirs() -> None:
    for path in [OUTPUT_ROOT, DATA_OUT, *DIRS.values()]:
        path.mkdir(parents=True, exist_ok=True)


def copy_plot_sources() -> None:
    sources = {
        "strict_model_comparison.csv": STRICT / "model_comparison.csv",
        "strict_sampling_comparison.csv": STRICT / "sampling_comparison.csv",
        "strict_tuning_history.csv": STRICT / "tuning_history.csv",
        "strict_validation_predictions_all_models.csv": STRICT / "validation_predictions_all_models.csv",
        "strict_oof_predictions.csv": STRICT / "oof_predictions.csv",
        "strict_confusion_matrix_long.csv": STRICT / "confusion_matrix_long.csv",
        "strict_class_metrics.csv": STRICT / "class_metrics.csv",
        "strict_grade_error_distribution.csv": STRICT / "grade_error_distribution.csv",
        "strict_label_distribution.csv": STRICT / "label_distribution.csv",
        "strict_fold_bins.csv": STRICT / "fold_bins.csv",
        "strict_calibration_metrics.csv": STRICT / "calibration_metrics.csv",
        "strict_shap_values_long_top20.csv": STRICT / "explanation_data" / "shap_values_long_top20.csv",
        "strict_shap_global_importance.csv": STRICT / "explanation_data" / "shap_global_importance.csv",
        "strict_ordinal_coefficients.csv": STRICT / "explanation_data" / "ordinal_coefficients.csv",
        "strict_feature_importance.csv": STRICT / "explanation_data" / "feature_importance.csv",
        "all_scopes_model_comparison.csv": PROJECT / "plot_data_all_scopes" / "model_comparison.csv",
        "feature_scope_comparison.csv": PROJECT / "plot_data_all_scopes" / "feature_scope_comparison.csv",
    }
    for name, source in sources.items():
        if source.exists():
            shutil.copy2(source, DATA_OUT / name)


def load_all() -> dict[str, pd.DataFrame]:
    raw = read_csv(RAW_PATH)
    raw["Person_ID"] = raw["Person_ID"].astype(str)
    split = read_csv(SPLIT_PATH)
    split["Person_ID"] = split["Person_ID"].astype(str)
    dfs = {
        "raw": raw,
        "split": split,
        "features": read_csv(STRICT / "feature_list.csv"),
        "labels": read_csv(STRICT / "label_distribution.csv"),
        "bins": read_csv(STRICT / "fold_bins.csv"),
        "models": read_csv(STRICT / "model_comparison.csv"),
        "sampling": read_csv(STRICT / "sampling_comparison.csv"),
        "tuning": read_csv(STRICT / "tuning_history.csv"),
        "preds": read_csv(STRICT / "validation_predictions_all_models.csv"),
        "confusion": read_csv(STRICT / "confusion_matrix_long.csv"),
        "class_metrics": read_csv(STRICT / "class_metrics.csv"),
        "grade_errors": read_csv(STRICT / "grade_error_distribution.csv"),
        "calibration": read_csv(STRICT / "plot_data" / "calibration_curve_data.csv"),
        "calibration_metrics": read_csv(STRICT / "calibration_metrics.csv"),
        "shap_long": read_csv(STRICT / "explanation_data" / "shap_values_long_top20.csv"),
        "shap_global": read_csv(STRICT / "explanation_data" / "shap_global_importance.csv"),
        "ordinal": read_csv(STRICT / "explanation_data" / "ordinal_coefficients.csv"),
        "importance": read_csv(STRICT / "explanation_data" / "feature_importance.csv"),
        "scope_models": read_csv(PROJECT / "plot_data_all_scopes" / "model_comparison.csv"),
        "scope_stack": read_csv(PROJECT / "plot_data_all_scopes" / "feature_scope_comparison.csv"),
    }
    dfs["preds"]["Person_ID"] = dfs["preds"]["Person_ID"].astype(str)
    dfs["shap_long"]["Person_ID"] = dfs["shap_long"]["Person_ID"].astype(str)
    return dfs


def plot_label_distribution(d: dict[str, pd.DataFrame]) -> None:
    df = to_numeric(d["labels"].copy(), ["count", "rate"])
    fig, ax = plt.subplots(figsize=(14 * CM, 8 * CM))
    x = np.arange(len(CLASS_ORDER))
    width = 0.36
    for offset, split_name, label, hatch in [(-width / 2, "outer_train", "训练集 (n=8000)", ""), (width / 2, "outer_validation", "外部验证集 (n=2000)", "//")]:
        sub = df[df["split"] == split_name].set_index("class").reindex(CLASS_ORDER)
        bars = ax.bar(
            x + offset,
            sub["count"],
            width,
            color=[CLASS_COLORS[c] for c in CLASS_ORDER],
            alpha=0.94 if not hatch else 0.62,
            hatch=hatch,
            edgecolor="#37474F",
            linewidth=0.6,
            label=label,
        )
        for bar, rate in zip(bars, sub["rate"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 45, f"{rate:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, CLASS_ORDER)
    ax.set_ylabel("样本数")
    ax.set_title("任务二健康等级标签分布")
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    quiet_spines(ax)
    ship(fig, "data", "FigT2_01_label_distribution", "任务二健康等级标签分布", "核对训练/验证集四级标签是否近似均衡", "strict_no_proxy/label_distribution.csv")


def plot_score_distribution(d: dict[str, pd.DataFrame]) -> None:
    raw = d["raw"].merge(d["split"][["Person_ID", "split"]], on="Person_ID", how="left")
    train = raw[raw["split"] == "train"].copy()
    score_col = "Health_Score"
    train[score_col] = pd.to_numeric(train[score_col], errors="coerce")
    bounds = to_numeric(d["bins"].copy(), ["q25", "q50", "q75"])
    outer = bounds[bounds["split"] == "outer_train"].iloc[0]
    q = [float(outer["q25"]), float(outer["q50"]), float(outer["q75"])]
    fig, ax = plt.subplots(figsize=(14 * CM, 8 * CM))
    sns.histplot(train[score_col].dropna(), bins=32, stat="density", color=MAIN_BLUE, alpha=0.34, edgecolor="white", ax=ax)
    sns.kdeplot(train[score_col].dropna(), color=MAIN_BLUE, linewidth=2.0, ax=ax)
    lo, hi = train[score_col].min(), train[score_col].max()
    spans = [(lo, q[0], "Poor"), (q[0], q[1], "Average"), (q[1], q[2], "Good"), (q[2], hi, "Excellent")]
    for left, right, cls in spans:
        ax.axvspan(left, right, color=CLASS_COLORS[cls], alpha=0.07)
    for value, label in zip(q, ["Q25", "Q50", "Q75"]):
        ax.axvline(value, color=DARK_INK, linestyle="--", linewidth=1.1)
        ax.text(value, ax.get_ylim()[1] * 0.93, f"{label}={value:.1f}", rotation=90, ha="right", va="top", fontsize=8.2, color=DARK_INK)
    ax.set_xlabel("Health Score（训练集）")
    ax.set_ylabel("密度")
    ax.set_title("训练集健康评分分布与四分位分箱边界")
    ax.grid(axis="y")
    quiet_spines(ax)
    ship(fig, "data", "FigT2_02_health_score_distribution_bins", "健康评分分布与四分位边界", "说明任务二四级标签的训练集内定义", "A题数据集.csv + strict_no_proxy/fold_bins.csv")


def plot_fold_bins(d: dict[str, pd.DataFrame]) -> None:
    df = to_numeric(d["bins"].copy(), ["q25", "q50", "q75"])
    df["折次"] = df["split"].replace({
        "cv_fold_1_train": "CV1", "cv_fold_2_train": "CV2", "cv_fold_3_train": "CV3", "cv_fold_4_train": "CV4", "outer_train": "Outer train"
    })
    fig, ax = plt.subplots(figsize=(14 * CM, 8 * CM))
    colors = [DARK_RED, ORANGE, TEAL]
    for col, label, color in zip(["q25", "q50", "q75"], ["Q25", "Q50", "Q75"], colors):
        ax.plot(df["折次"], df[col], marker="o", linewidth=1.8, markersize=5.2, color=color, label=label)
        spread = df[col].max() - df[col].min()
        ax.text(len(df) - 1, df[col].iloc[-1] + 0.08, f"范围 {spread:.2f}", color=color, fontsize=8, ha="right")
    ax.set_ylabel("Health Score 边界值")
    ax.set_title("交叉验证折内分箱边界稳定性")
    ax.grid(axis="y")
    ax.legend(ncol=3, loc="upper left")
    quiet_spines(ax)
    ship(fig, "data", "FigT2_03_fold_boundary_stability", "折内分箱边界稳定性", "证明所有分箱边界均由训练折独立计算且波动较小", "strict_no_proxy/fold_bins.csv")


def strict_raw_train(d: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    feature_names = d["features"]["feature"].astype(str).tolist()
    train_ids = set(d["split"].loc[d["split"]["split"] == "train", "Person_ID"].astype(str))
    raw = d["raw"][d["raw"]["Person_ID"].isin(train_ids)].copy()
    cols = [c for c in feature_names if c in raw.columns]
    return raw, cols


def top_raw_numeric_features(d: dict[str, pd.DataFrame], n: int = 12) -> list[str]:
    raw, cols = strict_raw_train(d)
    numeric = {c for c in cols if pd.api.types.is_numeric_dtype(raw[c])}
    sg = to_numeric(d["shap_global"].copy(), ["mean_abs_shap"])
    sg["raw_feature"] = sg["feature"].astype(str).str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
    ranked = sg.groupby("raw_feature", as_index=False)["mean_abs_shap"].mean().sort_values("mean_abs_shap", ascending=False)
    result = [f for f in ranked["raw_feature"] if f in numeric]
    return result[:n]


def plot_missing_rates(d: dict[str, pd.DataFrame]) -> None:
    raw, cols = strict_raw_train(d)
    rates = raw[cols].isna().mean().mul(100).sort_values(ascending=False)
    rates = rates[rates > 0].head(18).sort_values()
    fig, ax = plt.subplots(figsize=(14 * CM, 8 * CM))
    if rates.empty:
        ax.text(0.5, 0.5, "严格口径特征中未检测到缺失值", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    else:
        bars = ax.barh([clean_feature(x) for x in rates.index], rates.values, color=MAIN_BLUE, edgecolor="#294B67", linewidth=0.5)
        for bar, val in zip(bars, rates.values):
            ax.text(val + max(rates.max() * 0.015, 0.1), bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8)
        ax.set_xlabel("训练集缺失率（%）")
        ax.set_title("任务二严格口径特征缺失率（仅显示非零字段）")
        ax.grid(axis="x")
        quiet_spines(ax)
    ship(fig, "data", "FigT2_04_missing_rate", "任务二特征缺失率", "定位真正存在缺失的字段并避免展示空列", "A题数据集.csv + strict_no_proxy/feature_list.csv")


def plot_standardized_boxplots(d: dict[str, pd.DataFrame]) -> None:
    raw, _ = strict_raw_train(d)
    features = top_raw_numeric_features(d, 10)
    z = raw[features].apply(pd.to_numeric, errors="coerce")
    z = z.fillna(z.median())
    z = z.apply(lambda s: np.clip(zscore(s, nan_policy="omit"), -4.5, 4.5))
    long = z.rename(columns={c: clean_feature(c) for c in features}).melt(var_name="特征", value_name="标准化值")
    fig, ax = plt.subplots(figsize=(14 * CM, 9 * CM))
    sns.boxplot(data=long, x="标准化值", y="特征", color="#BBD5E5", fliersize=1.2, linewidth=0.8, ax=ax)
    ax.axvline(0, color=MID_GREY, linewidth=0.8)
    ax.set_title("关键数值特征标准化箱线图")
    ax.set_xlabel("标准化值（截断至 ±4.5 仅用于显示）")
    ax.set_ylabel("")
    ax.grid(axis="x")
    quiet_spines(ax)
    ship(fig, "data", "FigT2_05_standardized_boxplots", "关键数值特征箱线图", "比较不同量纲变量的分布、偏态和离群点", "A题数据集.csv + strict_no_proxy/shap_global_importance.csv")


def plot_correlation_heatmap(d: dict[str, pd.DataFrame]) -> None:
    raw, _ = strict_raw_train(d)
    features = top_raw_numeric_features(d, 11)
    cols = [*features, "Health_Score"]
    corr = raw[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    labels = [clean_feature(c) if c != "Health_Score" else "健康评分（目标）" for c in corr.columns]
    corr.index = labels
    corr.columns = labels
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(16 * CM, 10 * CM))
    sns.heatmap(corr, mask=mask, cmap="vlag", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, square=True, linewidths=0.35, cbar_kws={"label": "Spearman ρ", "shrink": 0.78}, ax=ax)
    ax.set_title("关键数值特征与健康评分的 Spearman 相关结构")
    ax.tick_params(axis="x", rotation=55, labelsize=8.0)
    ax.tick_params(axis="y", rotation=0)
    fig.subplots_adjust(bottom=0.28, left=0.20)
    ship(fig, "data", "FigT2_06_correlation_heatmap", "关键特征相关性热力图", "展示关键数值特征的线性单调关系及与目标的关联", "A题数据集.csv + strict_no_proxy/shap_global_importance.csv")


def plot_tuning_history(d: dict[str, pd.DataFrame]) -> None:
    df = d["tuning"].copy()
    num_cols = ["trial_number", "objective", "mean_accuracy", "mean_kappa", "mean_macro_f1", "C"]
    df = to_numeric(df, num_cols)
    lgb = df[df["trial_number"].notna()].sort_values("trial_number")
    ord_df = df[df["model"] == "Ordinal_Cumulative"].sort_values("C")
    fig, axes = plt.subplots(1, 2, figsize=(16 * CM, 7.5 * CM))
    ax = axes[0]
    ax.plot(lgb["trial_number"], lgb["objective"], color=TEAL, marker="o", linewidth=1.2, label="单次目标值")
    ax.plot(lgb["trial_number"], lgb["objective"].cummax(), color=ORANGE, linewidth=2.0, label="历史最优")
    best = lgb.loc[lgb["objective"].idxmax()]
    ax.scatter([best["trial_number"]], [best["objective"]], color=ORANGE, s=58, zorder=5)
    ax.annotate(f"最优试验 #{int(best['trial_number'])}\n{best['objective']:.4f}", (best["trial_number"], best["objective"]), xytext=(8, -30), textcoords="offset points", fontsize=8)
    ax.set_title("LightGBM 贝叶斯调参轨迹")
    ax.set_xlabel("试验编号")
    ax.set_ylabel("综合目标值")
    ax.grid(True)
    ax.legend()
    quiet_spines(ax)
    ax = axes[1]
    for metric, label, color, marker in [("mean_accuracy", "Accuracy", MAIN_BLUE, "o"), ("mean_macro_f1", "Macro-F1", TEAL, "s")]:
        ax.plot(ord_df["C"], ord_df[metric], marker=marker, color=color, linewidth=1.7, label=label)
    ax.set_xscale("log")
    ax.set_title("Ordinal Logistic 正则强度搜索")
    ax.set_xlabel("C（对数轴）")
    ax.set_ylabel("四折交叉验证均值")
    ax.grid(True)
    ax.legend()
    quiet_spines(ax)
    fig.suptitle("任务二超参数调优过程", y=1.01)
    fig.tight_layout()
    ship(fig, "tuning", "FigT2_07_tuning_history", "任务二超参数调优过程", "保留LightGBM和Ordinal Logistic的搜索轨迹及最优位置", "strict_no_proxy/tuning_history.csv")


def plot_sampling_comparison(d: dict[str, pd.DataFrame]) -> None:
    df = to_numeric(d["sampling"].copy(), ["accuracy", "macro_f1", "quadratic_weighted_kappa"])
    summary = df.groupby("strategy")[["accuracy", "macro_f1", "quadratic_weighted_kappa"]].agg(["mean", "std"])
    strategies = ["none", "class_weight_balanced", "SMOTE_after_onehot"]
    labels = ["无采样", "类别权重", "SMOTE"]
    metrics = [("accuracy", "Accuracy", MAIN_BLUE), ("macro_f1", "Macro-F1", TEAL), ("quadratic_weighted_kappa", "加权Kappa", ORANGE)]
    fig, ax = plt.subplots(figsize=(14 * CM, 8 * CM))
    x = np.arange(len(strategies))
    width = 0.23
    for idx, (metric, label, color) in enumerate(metrics):
        means = [summary.loc[s, (metric, "mean")] for s in strategies]
        stds = [summary.loc[s, (metric, "std")] for s in strategies]
        bars = ax.bar(x + (idx - 1) * width, means, width, yerr=stds, capsize=2.5, color=color, edgecolor="#455A64", linewidth=0.45, label=label)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=7.3, rotation=90)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x, labels)
    ax.set_ylabel("四折交叉验证均值")
    ax.set_title("采样与类别平衡策略对比（均值±标准差）")
    ax.grid(axis="y")
    ax.legend(ncol=3, loc="upper left")
    quiet_spines(ax)
    ship(fig, "tuning", "FigT2_08_sampling_comparison", "采样策略对比", "比较无采样、类别权重和SMOTE的真实增益", "strict_no_proxy/sampling_comparison.csv")


def plot_model_performance(d: dict[str, pd.DataFrame]) -> None:
    df = to_numeric(d["models"].copy(), ["accuracy", "macro_f1", "quadratic_weighted_kappa"])
    df = df.set_index("model").reindex(MODEL_ORDER).dropna(how="all")
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("quadratic_weighted_kappa", "加权Kappa")]
    fig, ax = plt.subplots(figsize=(16 * CM, 8.5 * CM))
    x = np.arange(len(df))
    width = 0.23
    alphas = [0.92, 0.70, 0.48]
    for idx, (metric, label) in enumerate(metrics):
        bars = ax.bar(x + (idx - 1) * width, df[metric], width, color=[MODEL_COLORS[m] for m in df.index], alpha=alphas[idx], edgecolor="#37474F", linewidth=0.45, label=label)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=7.1, rotation=90)
    ax.set_ylim(0, 1.03)
    ax.set_xticks(x, [MODEL_LABELS[m] for m in df.index], rotation=12)
    ax.set_ylabel("外部验证集指标")
    ax.set_title("任务二主要模型性能对比（n=2000）")
    ax.grid(axis="y")
    ax.legend(ncol=3, loc="upper left")
    quiet_spines(ax)
    fig.subplots_adjust(bottom=0.22)
    ship(fig, "performance", "FigT2_09_model_performance", "主要模型性能对比", "比较基线、单模型与最终Stacking融合模型", "strict_no_proxy/model_comparison.csv")


def plot_class_metrics(d: dict[str, pd.DataFrame]) -> None:
    df = to_numeric(d["class_metrics"].copy(), ["precision", "recall", "f1-score", "support"])
    df = df[df["model"] == "Stacking"].set_index("class").reindex(CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(14 * CM, 8 * CM))
    x = np.arange(4)
    width = 0.24
    for idx, (metric, label, color) in enumerate([("precision", "Precision", MAIN_BLUE), ("recall", "Recall", TEAL), ("f1-score", "F1", ORANGE)]):
        bars = ax.bar(x + (idx - 1) * width, df[metric], width, color=color, alpha=0.90, edgecolor="#455A64", linewidth=0.45, label=label)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.014, f"{val:.3f}", ha="center", fontsize=7.3)
    for i, support in enumerate(df["support"]):
        ax.text(i, 0.035, f"n={int(support)}", ha="center", fontsize=7.5, color="#607D8B")
    ax.set_ylim(0, 1.04)
    ax.set_xticks(x, CLASS_ORDER)
    ax.set_ylabel("指标值")
    ax.set_title("Stacking模型分等级分类指标")
    ax.grid(axis="y")
    ax.legend(ncol=3, loc="upper left")
    quiet_spines(ax)
    ship(fig, "performance", "FigT2_10_class_metrics", "分等级分类指标", "检查Poor、Average、Good、Excellent各等级的识别质量", "strict_no_proxy/class_metrics.csv")


def confusion_matrices(df: pd.DataFrame, model: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df[df["model"] == model]
    count = sub.pivot(index="true_label", columns="predicted_label", values="count").reindex(index=CLASS_ORDER, columns=CLASS_ORDER).fillna(0).astype(int).to_numpy()
    rate = sub.pivot(index="true_label", columns="predicted_label", values="row_rate").reindex(index=CLASS_ORDER, columns=CLASS_ORDER).fillna(0).astype(float).to_numpy()
    return count, rate


def draw_confusion(ax, count: np.ndarray, rate: np.ndarray, title: str, cbar: bool = False, annot_size: float = 8.5) -> None:
    annot = np.empty_like(count, dtype=object)
    for i in range(count.shape[0]):
        for j in range(count.shape[1]):
            annot[i, j] = f"{count[i, j]}\n{rate[i, j]:.1%}"
    sns.heatmap(rate, annot=annot, fmt="", annot_kws={"size": annot_size}, cmap="Blues", vmin=0, vmax=1, square=True, linewidths=0.6, cbar=cbar, cbar_kws={"label": "行归一化比例"}, xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)


def plot_confusions(d: dict[str, pd.DataFrame]) -> None:
    df = to_numeric(d["confusion"].copy(), ["count", "row_rate"])
    count, rate = confusion_matrices(df, "Stacking")
    fig, ax = plt.subplots(figsize=(10 * CM, 8.5 * CM))
    draw_confusion(ax, count, rate, "Stacking模型混淆矩阵", cbar=True)
    ship(fig, "error", "FigT2_11_stacking_confusion_matrix", "Stacking混淆矩阵", "同时展示样本数和行归一化比例，观察相邻等级误判", "strict_no_proxy/confusion_matrix_long.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16 * CM, 6.2 * CM), sharex=True, sharey=True)
    for ax, model in zip(axes, THREE_MODELS):
        count, rate = confusion_matrices(df, model)
        draw_confusion(ax, count, rate, MODEL_LABELS[model], cbar=False, annot_size=5.6)
        ax.set_ylabel("真实类别" if ax is axes[0] else "")
        ax.tick_params(axis="both", labelsize=6.5)
    fig.suptitle("主要模型错误结构对比", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.27)
    ship(fig, "error", "FigT2_12_confusion_matrix_models", "主要模型混淆矩阵对比", "比较Ordinal、LightGBM和Stacking的等级错误结构", "strict_no_proxy/confusion_matrix_long.csv")


def probability_columns() -> dict[str, str]:
    return {c: f"Probability_{c}" for c in CLASS_ORDER}


def plot_roc_pr(d: dict[str, pd.DataFrame]) -> None:
    df = d["preds"].copy()
    for col in probability_columns().values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    model_lines = [(m, MODEL_LABELS[m], MODEL_COLORS[m]) for m in MAIN_MODELS]
    for kind in ["roc", "pr"]:
        fig, axes = plt.subplots(2, 2, figsize=(16 * CM, 11 * CM), sharex=False, sharey=False)
        records = []
        for ax, cls in zip(axes.flat, CLASS_ORDER):
            for model, label, color in model_lines:
                sub = df[df["Model"] == model]
                y = (sub["True_Label"] == cls).astype(int).to_numpy()
                score = sub[f"Probability_{cls}"].to_numpy()
                if kind == "roc":
                    x, yv, _ = roc_curve(y, score)
                    metric = auc(x, yv)
                    ax.plot(x, yv, color=color, linewidth=1.6, label=f"{label}  AUC={metric:.3f}")
                    records.extend({"model": model, "class": cls, "fpr": xx, "tpr": yy, "auc": metric} for xx, yy in zip(x, yv))
                else:
                    precision, recall, _ = precision_recall_curve(y, score)
                    metric = average_precision_score(y, score)
                    ax.plot(recall, precision, color=color, linewidth=1.6, label=f"{label}  AP={metric:.3f}")
                    records.extend({"model": model, "class": cls, "recall": xx, "precision": yy, "ap": metric} for xx, yy in zip(recall, precision))
            if kind == "roc":
                ax.plot([0, 1], [0, 1], color="#7A858C", linestyle="--", linewidth=0.9)
                ax.set_xlabel("假阳性率 FPR")
                ax.set_ylabel("真阳性率 TPR")
            else:
                prevalence = (df[df["Model"] == "Stacking"]["True_Label"] == cls).mean()
                ax.axhline(prevalence, color="#7A858C", linestyle="--", linewidth=0.9, label=f"类别基准={prevalence:.3f}")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_title(cls, color=CLASS_COLORS[cls])
            ax.grid(True)
            ax.legend(fontsize=7.0, loc="best")
            quiet_spines(ax)
        if kind == "roc":
            stem, title, purpose = "FigT2_13_multiclass_roc", "多模型多类别ROC曲线", "按类别比较主要模型的一对其余判别能力"
            pd.DataFrame(records).to_csv(DATA_OUT / "derived_roc_curve_all_models.csv", index=False, encoding="utf-8-sig")
        else:
            stem, title, purpose = "FigT2_14_multiclass_pr", "多模型多类别PR曲线", "在各等级样本比例下比较主要模型的查准率与召回率"
            pd.DataFrame(records).to_csv(DATA_OUT / "derived_pr_curve_all_models.csv", index=False, encoding="utf-8-sig")
        fig.suptitle(f"任务二{title}（OvR，外部验证集 n=2000）", y=1.01)
        fig.tight_layout()
        ship(fig, "performance", stem, title, purpose, "strict_no_proxy/validation_predictions_all_models.csv")


def plot_calibration(d: dict[str, pd.DataFrame]) -> None:
    df = to_numeric(d["calibration"].copy(), ["mean_predicted_probability", "observed_frequency"])
    metric = to_numeric(d["calibration_metrics"].copy(), ["value"])
    fig, axes = plt.subplots(2, 2, figsize=(14 * CM, 10.5 * CM), sharex=True, sharey=True)
    for ax, cls in zip(axes.flat, CLASS_ORDER):
        sub = df[df["class"] == cls].sort_values("mean_predicted_probability")
        ax.plot([0, 1], [0, 1], color="#7A858C", linestyle="--", linewidth=1.0, label="理想校准")
        ax.plot(sub["mean_predicted_probability"], sub["observed_frequency"], color=CLASS_COLORS[cls], marker="o", linewidth=1.8, label="Stacking")
        brier = metric[(metric["class"] == cls) & (metric["metric"] == "one_vs_rest_brier")]["value"]
        suffix = f"（Brier={float(brier.iloc[0]):.3f}）" if len(brier) else ""
        ax.set_title(cls + suffix, color=CLASS_COLORS[cls])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("平均预测概率")
        ax.set_ylabel("实际发生比例")
        ax.grid(True)
        quiet_spines(ax)
    ece = metric[metric["metric"] == "multiclass_confidence_ECE"]["value"]
    ece_text = f"，置信度ECE={float(ece.iloc[0]):.3f}" if len(ece) else ""
    fig.suptitle(f"Stacking模型可靠性曲线（10 bins{ece_text}）", y=1.01)
    fig.tight_layout()
    ship(fig, "error", "FigT2_15_reliability_curve", "Stacking可靠性曲线", "判断四个等级概率是否过度自信并报告ECE/Brier", "strict_no_proxy/plot_data/calibration_curve_data.csv + calibration_metrics.csv")


def plot_grade_errors(d: dict[str, pd.DataFrame]) -> None:
    df = to_numeric(d["grade_errors"].copy(), ["absolute_grade_error", "rate"])
    models = MAIN_MODELS
    fig, ax = plt.subplots(figsize=(15 * CM, 8 * CM))
    x = np.arange(4)
    width = 0.18
    for idx, model in enumerate(models):
        sub = df[df["model"] == model].set_index("absolute_grade_error").reindex(range(4)).fillna(0)
        ax.bar(x + (idx - 1.5) * width, sub["rate"], width, color=MODEL_COLORS[model], edgecolor="#455A64", linewidth=0.4, label=MODEL_LABELS[model])
    ax.set_xticks(x, ["完全正确", "相差1级", "相差2级", "相差3级"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("外部验证集占比")
    ax.set_title("主要模型绝对等级误差分布")
    ax.grid(axis="y")
    ax.legend(ncol=2)
    quiet_spines(ax)
    ship(fig, "error", "FigT2_16_grade_error_distribution", "绝对等级误差分布", "突出相邻等级误差与跨两级以上严重误差", "strict_no_proxy/grade_error_distribution.csv")


def plot_prediction_distribution(d: dict[str, pd.DataFrame]) -> None:
    df = d["preds"].copy()
    rows = []
    stack = df[df["Model"] == "Stacking"]
    for cls in CLASS_ORDER:
        rows.append({"series": "真实分布", "class": cls, "rate": (stack["True_Label"] == cls).mean()})
    for model in MAIN_MODELS:
        sub = df[df["Model"] == model]
        for cls in CLASS_ORDER:
            rows.append({"series": MODEL_LABELS[model], "class": cls, "rate": (sub["Predicted_Label"] == cls).mean()})
    out = pd.DataFrame(rows)
    out.to_csv(DATA_OUT / "derived_prediction_distribution.csv", index=False, encoding="utf-8-sig")
    series = ["真实分布", *[MODEL_LABELS[m] for m in MAIN_MODELS]]
    colors = [DARK_INK, *[MODEL_COLORS[m] for m in MAIN_MODELS]]
    fig, ax = plt.subplots(figsize=(15 * CM, 8 * CM))
    x = np.arange(4)
    width = 0.15
    for idx, (name, color) in enumerate(zip(series, colors)):
        sub = out[out["series"] == name].set_index("class").reindex(CLASS_ORDER)
        ax.bar(x + (idx - 2) * width, sub["rate"], width, color=color, alpha=0.9, edgecolor="#455A64", linewidth=0.35, label=name)
    ax.set_xticks(x, CLASS_ORDER)
    ax.set_ylabel("占比")
    ax.set_ylim(0, 0.38)
    ax.set_title("真实等级与主要模型预测等级分布")
    ax.grid(axis="y")
    ax.legend(ncol=3, loc="upper center")
    quiet_spines(ax)
    ship(fig, "error", "FigT2_17_prediction_distribution", "预测等级分布", "检查模型是否系统性高估或低估某一健康等级", "strict_no_proxy/validation_predictions_all_models.csv")


def normalize_feature_color(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(arr).any():
        return np.full_like(arr, 0.5)
    lo, hi = np.nanpercentile(arr, [5, 95])
    if hi <= lo:
        return np.full_like(arr, 0.5)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def plot_shap_beeswarm(d: dict[str, pd.DataFrame]) -> None:
    df = to_numeric(d["shap_long"].copy(), ["shap_value", "feature_value_encoded"])
    fig, axes = plt.subplots(2, 1, figsize=(15 * CM, 14 * CM), sharex=False)
    rng = np.random.default_rng(2026)
    for ax, cls in zip(axes, ["Poor", "Excellent"]):
        sub = df[df["class"] == cls].copy()
        rank = sub.groupby("feature")["shap_value"].apply(lambda s: s.abs().mean()).sort_values(ascending=False).head(15).index.tolist()
        for yi, feature in enumerate(rank):
            pts = sub[sub["feature"] == feature]
            jitter = rng.normal(0, 0.10, len(pts))
            color = normalize_feature_color(pts["feature_value_encoded"])
            ax.scatter(pts["shap_value"], yi + jitter, c=color, cmap="coolwarm", vmin=0, vmax=1, s=8, alpha=0.62, linewidths=0)
        ax.axvline(0, color="#68757D", linewidth=0.8)
        ax.set_yticks(range(len(rank)), [clean_feature(x) for x in rank])
        ax.invert_yaxis()
        ax.set_xlabel("SHAP值（对该等级的贡献）")
        ax.set_title(cls, color=CLASS_COLORS[cls])
        ax.grid(axis="x")
        quiet_spines(ax)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.024, pad=0.018)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["特征值低", "特征值高"])
    fig.suptitle("任务二SHAP蜂群图：Poor与Excellent", y=0.98)
    fig.subplots_adjust(hspace=0.55, left=0.30, right=0.90, top=0.90, bottom=0.15)
    ship(fig, "explain", "FigT2_18_shap_beeswarm", "SHAP蜂群图", "展示关键特征对Poor和Excellent等级预测的方向与强度", "strict_no_proxy/explanation_data/shap_values_long_top20.csv")


def plot_shap_global_heatmap(d: dict[str, pd.DataFrame]) -> None:
    df = to_numeric(d["shap_global"].copy(), ["mean_abs_shap"])
    totals = df.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False).head(15)
    pivot = df[df["feature"].isin(totals.index)].pivot(index="feature", columns="class", values="mean_abs_shap").reindex(index=totals.index, columns=CLASS_ORDER).fillna(0)
    pivot.index = [clean_feature(x) for x in pivot.index]
    fig, ax = plt.subplots(figsize=(12.5 * CM, 9 * CM))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.35, cbar_kws={"label": "平均|SHAP|"}, ax=ax)
    ax.set_xlabel("健康等级")
    ax.set_ylabel("")
    ax.set_title("各等级全局SHAP重要性热力图（Top 15）")
    ax.tick_params(axis="x", rotation=0)
    fig.subplots_adjust(bottom=0.21, left=0.24)
    ship(fig, "explain", "FigT2_19_shap_global_heatmap", "全局SHAP重要性热力图", "比较同一特征在四个等级中的贡献强度", "strict_no_proxy/explanation_data/shap_global_importance.csv")


def plot_shap_dependence(d: dict[str, pd.DataFrame]) -> None:
    df = to_numeric(d["shap_long"].copy(), ["shap_value", "feature_value_encoded"])
    sg = to_numeric(d["shap_global"].copy(), ["mean_abs_shap"])
    numeric_features = [f for f in sg.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False).index if str(f).startswith("num__")][:4]
    sub_cls = df[df["class"] == "Excellent"]
    fig, axes = plt.subplots(2, 2, figsize=(14 * CM, 10 * CM))
    for ax, feature in zip(axes.flat, numeric_features):
        sub = sub_cls[sub_cls["feature"] == feature].dropna(subset=["feature_value_encoded", "shap_value"]).copy()
        ax.scatter(sub["feature_value_encoded"], sub["shap_value"], s=10, alpha=0.28, color=CLASS_COLORS["Excellent"], linewidths=0)
        if len(sub) >= 20:
            sub["bin"] = pd.qcut(sub["feature_value_encoded"], q=min(12, sub["feature_value_encoded"].nunique()), duplicates="drop")
            med = sub.groupby("bin", observed=True).agg(x=("feature_value_encoded", "median"), y=("shap_value", "median"))
            ax.plot(med["x"], med["y"], color=ORANGE, linewidth=2.0)
        ax.axhline(0, color="#7A858C", linewidth=0.8, linestyle="--")
        ax.set_title(clean_feature(feature))
        ax.set_xlabel("编码/标准化后的特征值")
        ax.set_ylabel("Excellent类SHAP值")
        ax.grid(True)
        quiet_spines(ax)
    fig.suptitle("关键数值特征SHAP依赖关系（Excellent类）", y=1.01)
    fig.tight_layout()
    ship(fig, "explain", "FigT2_20_shap_dependence", "SHAP依赖图", "展示关键连续特征的非线性阈值及贡献方向", "strict_no_proxy/explanation_data/shap_values_long_top20.csv")


def plot_ordinal_coefficients(d: dict[str, pd.DataFrame]) -> None:
    df = to_numeric(d["ordinal"].copy(), ["coefficient", "ci95_lower_approx", "ci95_upper_approx"])
    sub = df[(df["threshold"] == "P(y>1)") & (df["feature"] != "intercept")].copy()
    sub = sub[np.isfinite(sub["coefficient"]) & np.isfinite(sub["ci95_lower_approx"]) & np.isfinite(sub["ci95_upper_approx"])]
    sub = sub.loc[sub["coefficient"].abs().nlargest(15).index].sort_values("coefficient")
    y = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(14 * CM, 9 * CM))
    colors = [TEAL if v > 0 else DARK_RED for v in sub["coefficient"]]
    xerr = np.vstack([sub["coefficient"] - sub["ci95_lower_approx"], sub["ci95_upper_approx"] - sub["coefficient"]])
    ax.errorbar(sub["coefficient"], y, xerr=xerr, fmt="none", ecolor="#76858E", elinewidth=1.0, capsize=2.2)
    ax.scatter(sub["coefficient"], y, c=colors, s=34, zorder=3)
    ax.axvline(0, color="#59656D", linewidth=0.9, linestyle="--")
    ax.set_yticks(y, [clean_feature(f) for f in sub["feature"]])
    ax.set_xlabel("标准化系数（正值→更高健康等级）")
    ax.set_title("Ordinal Logistic中间阈值系数及近似95%区间")
    ax.grid(axis="x")
    quiet_spines(ax)
    fig.subplots_adjust(bottom=0.18, left=0.28)
    ship(fig, "explain", "FigT2_21_ordinal_coefficients", "Ordinal Logistic系数图", "解释特征升高对进入较高健康等级的方向与不确定性", "strict_no_proxy/explanation_data/ordinal_coefficients.csv", "区间为正则化Hessian近似区间")


def plot_feature_importance(d: dict[str, pd.DataFrame]) -> None:
    df = to_numeric(d["importance"].copy(), ["gain_importance_normalized"])
    sub = df.nlargest(20, "gain_importance_normalized").sort_values("gain_importance_normalized")
    fig, ax = plt.subplots(figsize=(14 * CM, 9.5 * CM))
    bars = ax.barh([clean_feature(x) for x in sub["feature"]], sub["gain_importance_normalized"], color=TEAL, edgecolor="#2D6F68", linewidth=0.45)
    for bar, val in zip(bars, sub["gain_importance_normalized"]):
        ax.text(val + sub["gain_importance_normalized"].max() * 0.012, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=7.5)
    ax.set_xlabel("归一化Gain重要性")
    ax.set_title("LightGBM特征重要性（Top 20）")
    ax.grid(axis="x")
    quiet_spines(ax)
    fig.subplots_adjust(bottom=0.18, left=0.27)
    ship(fig, "explain", "FigT2_22_lightgbm_feature_importance", "LightGBM特征重要性", "从树模型角度补充全局特征贡献排序", "strict_no_proxy/explanation_data/feature_importance.csv")


def sankey_static(pivot: pd.DataFrame, out_dir: Path, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(16 * CM, 8.5 * CM))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    source_specs = [("Ordinal_Cumulative", 0.14, "Ordinal Logistic"), ("LightGBM_Tuned", 0.14, "LightGBM")]
    target_x = 0.84
    class_y = {cls: 0.82 - i * 0.21 for i, cls in enumerate(CLASS_ORDER)}
    for model, sx, title in source_specs:
        offset = 0.035 if model == "Ordinal_Cumulative" else -0.035
        for cls in CLASS_ORDER:
            sy = class_y[cls] + offset
            ax.add_patch(FancyBboxPatch((sx - 0.065, sy - 0.025), 0.13, 0.05, boxstyle="round,pad=0.01", facecolor=CLASS_COLORS[cls], edgecolor="white", alpha=0.86))
            ax.text(sx, sy, f"{title}\n{cls}", ha="center", va="center", fontsize=7, color="white")
            counts = pivot.groupby([model, "Stacking"]).size() if False else None
    for cls in CLASS_ORDER:
        ty = class_y[cls]
        ax.add_patch(FancyBboxPatch((target_x - 0.07, ty - 0.035), 0.14, 0.07, boxstyle="round,pad=0.01", facecolor=CLASS_COLORS[cls], edgecolor="white", alpha=0.95))
        ax.text(target_x, ty, f"Stacking\n{cls}", ha="center", va="center", fontsize=8, color="white")
    max_count = 1
    links = []
    for model, sx, _ in source_specs:
        for scls in CLASS_ORDER:
            for tcls in CLASS_ORDER:
                count = int(((pivot[model] == scls) & (pivot["Stacking"] == tcls)).sum())
                links.append((model, sx, scls, tcls, count))
                max_count = max(max_count, count)
    for model, sx, scls, tcls, count in links:
        if count == 0:
            continue
        offset = 0.035 if model == "Ordinal_Cumulative" else -0.035
        sy, ty = class_y[scls] + offset, class_y[tcls]
        path = MplPath([(sx + 0.07, sy), (0.45, sy), (0.58, ty), (target_x - 0.08, ty)], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=CLASS_COLORS[tcls], linewidth=0.3 + 5.2 * count / max_count, alpha=0.18))
    ax.text(0.14, 0.96, "一级模型预测等级", ha="center", fontsize=10.5, weight="bold", color=DARK_INK)
    ax.text(target_x, 0.96, "融合模型最终等级", ha="center", fontsize=10.5, weight="bold", color=DARK_INK)
    ax.set_title("一级模型预测等级到Stacking最终判断的流向", pad=8)
    add_source_note(fig, "数据来源：strict_no_proxy/validation_predictions_all_models.csv；每个样本分别计入两个一级模型流向")
    save_figure(fig, out_dir, stem)


def plot_sankey(d: dict[str, pd.DataFrame]) -> None:
    preds = d["preds"]
    pivot = preds[preds["Model"].isin(["Ordinal_Cumulative", "LightGBM_Tuned", "Stacking"])].pivot(index="Person_ID", columns="Model", values="Predicted_Label").dropna()
    html_path = DIRS["performance"] / "FigT2_23_stacking_sankey.html"
    image_ok = False
    try:
        import plotly.graph_objects as go

        node_labels = []
        node_colors = []
        node_ids: dict[tuple[str, str], int] = {}
        for model in ["Ordinal_Cumulative", "LightGBM_Tuned", "Stacking"]:
            for cls in CLASS_ORDER:
                node_ids[(model, cls)] = len(node_labels)
                node_labels.append(f"{MODEL_LABELS[model]} · {cls}")
                node_colors.append(CLASS_COLORS[cls])
        sources, targets, values, link_colors = [], [], [], []
        for model in ["Ordinal_Cumulative", "LightGBM_Tuned"]:
            for scls in CLASS_ORDER:
                for tcls in CLASS_ORDER:
                    count = int(((pivot[model] == scls) & (pivot["Stacking"] == tcls)).sum())
                    if count:
                        sources.append(node_ids[(model, scls)])
                        targets.append(node_ids[("Stacking", tcls)])
                        values.append(count)
                        rgba = mcolors.to_rgba(CLASS_COLORS[tcls], 0.28)
                        link_colors.append(f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]:.2f})")
        fig = go.Figure(go.Sankey(arrangement="snap", node=dict(label=node_labels, color=node_colors, pad=13, thickness=15, line=dict(color="white", width=0.6)), link=dict(source=sources, target=targets, value=values, color=link_colors)))
        fig.update_layout(title="任务二一级模型预测等级到Stacking最终判断的流向", font=dict(family="Microsoft YaHei, Times New Roman", size=12, color=DARK_INK), width=1200, height=680, paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=20, r=20, t=65, b=25))
        fig.write_html(html_path, include_plotlyjs="directory")
        try:
            fig.write_image(DIRS["performance"] / "FigT2_23_stacking_sankey.png", width=1600, height=900, scale=2)
            fig.write_image(DIRS["performance"] / "FigT2_23_stacking_sankey.pdf", width=1200, height=680)
            image_ok = True
        except Exception:
            image_ok = False
    except Exception:
        image_ok = False
    if not image_ok:
        sankey_static(pivot, DIRS["performance"], "FigT2_23_stacking_sankey")
    register("FigT2_23_stacking_sankey", "Stacking等级流向桑基图", "展示两个一级模型预测等级与最终融合判断的流向", "strict_no_proxy/validation_predictions_all_models.csv", DIRS["performance"].name, "HTML为Plotly交互版；每个样本分别计入两个一级模型流向")


def prepare_validation_numeric(d: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    raw = d["raw"].copy()
    labels = d["preds"][d["preds"]["Model"] == "Stacking"][["Person_ID", "True_Label"]].drop_duplicates()
    merged = raw.merge(labels, on="Person_ID", how="inner")
    feature_names = d["features"]["feature"].astype(str).tolist()
    numeric = [c for c in feature_names if c in merged.columns and pd.api.types.is_numeric_dtype(merged[c])]
    X = merged[numeric].apply(pd.to_numeric, errors="coerce")
    return X, merged["True_Label"], merged["Person_ID"]


def plot_tsne_umap(d: dict[str, pd.DataFrame]) -> None:
    X, labels, ids = prepare_validation_numeric(d)
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_std = StandardScaler().fit_transform(X_imp)
    n_comp = max(2, min(30, X_std.shape[1]))
    X_pca = PCA(n_components=n_comp, random_state=2026).fit_transform(X_std)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", max_iter=1000, random_state=2026)
    tsne_xy = tsne.fit_transform(X_pca)
    umap_xy = UMAP(n_components=2, n_neighbors=30, min_dist=0.15, metric="euclidean", random_state=2026).fit_transform(X_pca)
    label_code = pd.Categorical(labels, categories=CLASS_ORDER, ordered=True).codes
    sil_tsne = silhouette_score(tsne_xy, label_code)
    sil_umap = silhouette_score(umap_xy, label_code)
    emb = pd.DataFrame({"Person_ID": ids, "True_Label": labels, "tsne_x": tsne_xy[:, 0], "tsne_y": tsne_xy[:, 1], "umap_x": umap_xy[:, 0], "umap_y": umap_xy[:, 1]})
    emb.to_csv(DATA_OUT / "derived_embedding_2d.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(16 * CM, 7.8 * CM))
    for ax, xcol, ycol, title, sil in [
        (axes[0], "tsne_x", "tsne_y", "t-SNE", sil_tsne),
        (axes[1], "umap_x", "umap_y", "UMAP", sil_umap),
    ]:
        for cls in CLASS_ORDER:
            sub = emb[emb["True_Label"] == cls]
            ax.scatter(sub[xcol], sub[ycol], s=9, alpha=0.50, color=CLASS_COLORS[cls], label=cls, linewidths=0)
        ax.set_title(f"{title}（轮廓系数={sil:.3f}）")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=CLASS_COLORS[c], markersize=6, label=c) for c in CLASS_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("任务二外部验证集数值特征二维分布", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    ship(fig, "structure", "FigT2_24_tsne_umap", "t-SNE与UMAP二维分布", "观察四个健康等级在严格口径数值特征空间中的分离与重叠", "A题数据集.csv + strict_no_proxy/validation_predictions_all_models.csv", "仅使用严格口径数值特征；未输入Health Score及代理变量")


def feature_group(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["sleep", "night", "nap", "bed"]):
        return "睡眠"
    if any(k in n for k in ["exercise", "steps", "workout", "gym"]):
        return "运动"
    if any(k in n for k in ["food", "intake", "diet", "breakfast", "water", "fruit", "vegetable", "protein", "sugary"]):
        return "营养"
    if any(k in n for k in ["stress", "mood", "anxiety", "energy", "social", "meditation"]):
        return "心理"
    if any(k in n for k in ["bmi", "blood", "pressure", "sugar", "risk", "age", "weight", "height"]):
        return "生理"
    return "其他"


def plot_mi_network(d: dict[str, pd.DataFrame]) -> None:
    raw, cols = strict_raw_train(d)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(raw[c])]
    X = raw[numeric].apply(pd.to_numeric, errors="coerce")
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    outer = to_numeric(d["bins"].copy(), ["q25", "q50", "q75"])
    b = outer[outer["split"] == "outer_train"].iloc[0]
    y = pd.cut(pd.to_numeric(raw["Health_Score"], errors="coerce"), bins=[-np.inf, b["q25"], b["q50"], b["q75"], np.inf], labels=False, include_lowest=True).astype(int)
    mi = mutual_info_classif(X_imp, y, random_state=2026)
    scores = pd.Series(mi, index=numeric).sort_values(ascending=False).head(15)
    selected = scores.index.tolist()
    corr = raw[selected].apply(pd.to_numeric, errors="coerce").corr(method="spearman").abs()
    candidate = []
    for i, a in enumerate(selected):
        for bname in selected[i + 1 :]:
            candidate.append((a, bname, float(corr.loc[a, bname])))
    candidate.sort(key=lambda x: x[2], reverse=True)
    edges = [e for e in candidate if e[2] >= 0.25][:28]
    if len(edges) < 14:
        edges = candidate[:20]
    out_scores = pd.DataFrame({"feature": scores.index, "mutual_information": scores.values, "group": [feature_group(x) for x in scores.index]})
    out_edges = pd.DataFrame(edges, columns=["source", "target", "abs_spearman"])
    out_scores.to_csv(DATA_OUT / "derived_mi_feature_scores.csv", index=False, encoding="utf-8-sig")
    out_edges.to_csv(DATA_OUT / "derived_mi_network_edges.csv", index=False, encoding="utf-8-sig")
    graph = nx.Graph()
    for f, value in scores.items():
        graph.add_node(f, mi=float(value), group=feature_group(f))
    for a, bname, weight in edges:
        graph.add_edge(a, bname, weight=weight)
    # A circular layout keeps all 15 labels separable in an A4-width figure;
    # edge thickness still carries the association strength.
    pos = nx.circular_layout(graph, scale=0.78)
    group_colors = {"睡眠": MAIN_BLUE, "运动": TEAL, "营养": ORANGE, "心理": DARK_RED, "生理": "#7556A3", "其他": MID_GREY}
    sizes = 280 + 1850 * (scores / max(scores.max(), 1e-9))
    fig, ax = plt.subplots(figsize=(16 * CM, 11 * CM))
    nx.draw_networkx_edges(graph, pos, width=[0.5 + 4 * graph[u][v]["weight"] for u, v in graph.edges], edge_color="#87969F", alpha=0.35, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_size=[sizes[n] for n in graph.nodes], node_color=[group_colors[graph.nodes[n]["group"]] for n in graph.nodes], edgecolors="white", linewidths=0.8, alpha=0.92, ax=ax)
    label_artists = nx.draw_networkx_labels(graph, pos, labels={n: clean_feature(n) for n in graph.nodes}, font_family="Microsoft YaHei", font_size=6.8, font_color=DARK_INK, ax=ax)
    for artist in label_artists.values():
        artist.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 0.4})
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markersize=7, label=g) for g, c in group_colors.items()]
    ax.legend(handles=handles, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    ax.set_title("任务二关键数值特征MI—相关网络（Top 15）")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.04, 1.04)
    ax.axis("off")
    ship(fig, "structure", "FigT2_25_mi_feature_network", "MI特征网络图", "节点大小表示与等级标签的互信息，连线粗细表示特征间Spearman关联", "A题数据集.csv + strict_no_proxy/fold_bins.csv", "这是MI—相关网络，不冒充需要专门MIC算法计算的MIC网络")


def draw_box(ax, x, y, w, h, text, face, edge="#FFFFFF", fontsize=8.5) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", facecolor=face, edgecolor=edge, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="white", weight="bold")


def plot_pipeline_route() -> None:
    fig, ax = plt.subplots(figsize=(16 * CM, 9 * CM))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.03, 0.69, 0.15, 0.17, "原始健康数据\n10000×64", MAIN_BLUE),
        (0.23, 0.69, 0.16, 0.17, "泄漏审查与划分\n严格去代理变量", "#496F8A"),
        (0.44, 0.69, 0.16, 0.17, "训练折内四分位分箱\nPoor→Excellent", ORANGE),
        (0.65, 0.69, 0.15, 0.17, "预处理Pipeline\n缺失/编码/标准化", TEAL),
        (0.19, 0.34, 0.19, 0.17, "Ordinal Logistic\n有序线性关系", "#7556A3"),
        (0.44, 0.34, 0.19, 0.17, "LightGBM\n非线性与高阶交互", TEAL),
        (0.69, 0.34, 0.18, 0.17, "OOF概率Stacking\n逻辑回归元学习器", "#176B87"),
        (0.31, 0.06, 0.18, 0.15, "多指标评价\nACC / F1 / Kappa", MAIN_BLUE),
        (0.56, 0.06, 0.18, 0.15, "可解释性\nSHAP / 系数 / 误差", ORANGE),
        (0.80, 0.06, 0.16, 0.15, "保存成果\nCSV / 模型 / 图件", DARK_RED),
    ]
    for x, y, w, h, text, color in boxes:
        draw_box(ax, x, y, w, h, text, color, fontsize=7.5 if y > 0.6 else 8.1)
    arrows = [
        ((0.18, 0.775), (0.23, 0.775)), ((0.39, 0.775), (0.44, 0.775)), ((0.60, 0.775), (0.65, 0.775)),
        ((0.725, 0.69), (0.30, 0.51)), ((0.725, 0.69), (0.535, 0.51)),
        ((0.38, 0.425), (0.69, 0.425)), ((0.63, 0.425), (0.69, 0.425)),
        ((0.78, 0.34), (0.40, 0.21)), ((0.78, 0.34), (0.65, 0.21)),
        ((0.49, 0.135), (0.56, 0.135)), ((0.74, 0.135), (0.80, 0.135)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#6E7B83", connectionstyle="arc3,rad=0.0"))
    ax.text(0.5, 0.95, "任务二：基于有序学习与Stacking的健康等级预测技术路线", ha="center", va="center", fontsize=13, weight="bold", color=DARK_INK)
    ax.text(0.50, 0.29, "一级模型仅向元学习器提供折外（OOF）概率，避免信息泄漏", ha="center", fontsize=8.2, color="#607D8B")
    add_source_note(fig, "依据：task2_pipeline.py、严格去代理变量实验与已保存训练产物")
    save_figure(fig, DIRS["route"], "FigT2_26_task2_pipeline")
    register("FigT2_26_task2_pipeline", "任务二总体技术路线图", "概括数据、分箱、双模型、OOF融合、评价和输出链路", "code/task2_pipeline.py + strict_no_proxy实验", DIRS["route"].name)


def plot_scope_comparison(d: dict[str, pd.DataFrame]) -> None:
    df = to_numeric(d["scope_stack"].copy(), ["accuracy", "macro_f1", "quadratic_weighted_kappa"])
    scopes = ["strict_no_proxy", "competition_full"]
    labels = ["严格去代理变量", "竞赛全特征"]
    colors = [MAIN_BLUE, ORANGE]
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("quadratic_weighted_kappa", "加权Kappa")]
    fig, ax = plt.subplots(figsize=(14 * CM, 8 * CM))
    x = np.arange(len(metrics))
    width = 0.32
    for idx, (scope, label, color) in enumerate(zip(scopes, labels, colors)):
        row = df[df["scope"] == scope].iloc[0]
        vals = [row[m] for m, _ in metrics]
        bars = ax.bar(x + (idx - 0.5) * width, vals, width, color=color, edgecolor="#455A64", linewidth=0.45, label=label)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.014, f"{val:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.03)
    ax.set_xticks(x, [x[1] for x in metrics])
    ax.set_ylabel("外部验证集指标")
    ax.set_title("Stacking模型两种特征口径对照")
    ax.grid(axis="y")
    ax.legend()
    quiet_spines(ax)
    ship(fig, "scope", "FigT2_27_feature_scope_comparison", "两种特征口径对照", "量化代理变量对竞赛准确率和可解释性口径的影响", "plot_data_all_scopes/feature_scope_comparison.csv", "正文建议使用严格口径；竞赛全特征口径作为对照或附录")


def plot_radar(d: dict[str, pd.DataFrame]) -> None:
    df = to_numeric(d["models"].copy(), ["accuracy", "macro_f1", "quadratic_weighted_kappa", "grade_mae", "severe_error_rate"])
    df = df[df["model"].isin(MAIN_MODELS)].set_index("model").reindex(MAIN_MODELS)
    radar = pd.DataFrame(index=df.index)
    radar["Accuracy"] = df["accuracy"]
    radar["Macro-F1"] = df["macro_f1"]
    radar["Kappa"] = df["quadratic_weighted_kappa"]
    radar["1−MAE/3"] = 1 - df["grade_mae"] / 3
    radar["1−严重误差率"] = 1 - df["severe_error_rate"]
    radar.to_csv(DATA_OUT / "derived_radar_metrics.csv", encoding="utf-8-sig")
    labels = radar.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(11 * CM, 10.5 * CM), subplot_kw={"polar": True})
    for model in radar.index:
        values = radar.loc[model].tolist() + [radar.loc[model].iloc[0]]
        ax.plot(angles, values, color=MODEL_COLORS[model], linewidth=1.7, label=MODEL_LABELS[model])
        ax.fill(angles, values, color=MODEL_COLORS[model], alpha=0.04)
    ax.set_xticks(angles[:-1], labels)
    ax.set_ylim(0.65, 1.0)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.7", "0.8", "0.9", "1.0"], fontsize=7)
    ax.grid(alpha=0.30)
    ax.set_title("主要模型综合性能雷达图", pad=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    fig.subplots_adjust(bottom=0.20, top=0.88)
    ship(fig, "performance", "FigT2_28_model_radar", "主要模型综合性能雷达图", "综合比较分类性能、等级一致性与严重误差控制", "strict_no_proxy/model_comparison.csv", "雷达轴均统一为越大越好；径向范围0.65–1.00并明确标注")


def plot_local_shap(d: dict[str, pd.DataFrame]) -> None:
    preds = d["preds"][d["preds"]["Model"] == "Stacking"].copy()
    for cls in ["Excellent", "Poor"]:
        preds[f"Probability_{cls}"] = pd.to_numeric(preds[f"Probability_{cls}"], errors="coerce")
    candidates = preds[(preds["True_Label"] == preds["Predicted_Label"]) & (preds["Predicted_Label"].isin(["Excellent", "Poor"]))].copy()
    candidates["confidence"] = candidates.apply(lambda r: r[f"Probability_{r['Predicted_Label']}"] if r["Predicted_Label"] in ["Excellent", "Poor"] else np.nan, axis=1)
    candidates = candidates.sort_values("confidence", ascending=False)
    shap = to_numeric(d["shap_long"].copy(), ["shap_value"])
    chosen = None
    for _, row in candidates.iterrows():
        sub = shap[(shap["Person_ID"] == row["Person_ID"]) & (shap["class"] == row["Predicted_Label"])]
        if len(sub) >= 8:
            chosen = (row, sub)
            break
    if chosen is None:
        return
    row, sub = chosen
    sub = sub.assign(abs_shap=sub["shap_value"].abs()).nlargest(15, "abs_shap").sort_values("shap_value")
    fig, ax = plt.subplots(figsize=(14 * CM, 8.5 * CM))
    colors = [TEAL if v > 0 else DARK_RED for v in sub["shap_value"]]
    bars = ax.barh([clean_feature(x) for x in sub["feature"]], sub["shap_value"], color=colors, edgecolor="#455A64", linewidth=0.4)
    ax.axvline(0, color="#59656D", linewidth=0.9)
    for bar, val in zip(bars, sub["shap_value"]):
        ax.text(val + (0.02 if val >= 0 else -0.02), bar.get_y() + bar.get_height() / 2, f"{val:+.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=7.4)
    ax.set_xlabel(f"对{row['Predicted_Label']}类的局部SHAP贡献")
    ax.set_title(f"代表性样本局部SHAP贡献：{row['Person_ID']}（置信度={row['confidence']:.3f}）")
    ax.grid(axis="x")
    quiet_spines(ax)
    fig.subplots_adjust(bottom=0.18, left=0.25)
    ship(fig, "explain", "FigT2_29_local_shap_contributions", "代表性样本局部SHAP贡献", "解释一个高置信度且预测正确样本的主要正负贡献", "strict_no_proxy/explanation_data/shap_values_long_top20.csv + validation_predictions_all_models.csv", "未保存SHAP基线值，因此使用真实局部贡献条形图而不伪造瀑布终点")


def write_manifest() -> None:
    frame = pd.DataFrame(MANIFEST)
    frame.to_csv(IMAGE_ROOT / "任务二图件清单.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 任务二正式图件使用说明",
        "",
        "本批图件由 `00_绘图脚本/plot_task2_all.py` 生成。正文优先使用 `strict_no_proxy` 严格去代理变量口径；`08_竞赛口径对照` 仅用于说明竞赛全特征口径。每张 Matplotlib 图均同时导出 600 dpi PNG 与 PDF。",
        "",
        "## 推荐正文图",
        "",
        "- 标签与分箱：FigT2_01、FigT2_02、FigT2_03。",
        "- 主结果：FigT2_09、FigT2_11、FigT2_13、FigT2_14、FigT2_15。",
        "- 等级误差：FigT2_16、FigT2_17。",
        "- 可解释性：FigT2_18、FigT2_20、FigT2_21、FigT2_22。",
        "- 融合与路线：FigT2_23、FigT2_26。",
        "",
        "## 图件明细",
        "",
    ]
    for item in MANIFEST:
        extra = f"；说明：{item['note']}" if item["note"] else ""
        lines.append(f"- `{item['file_stem']}`（{item['folder']}）：{item['purpose']}。数据：{item['source_data']}{extra}")
    lines += [
        "",
        "## 重新运行",
        "",
        "```powershell",
        r"& 'C:\Users\luo29\Desktop\任务二\.venv_plot\Scripts\python.exe' 'C:\Users\luo29\Desktop\任务二\图片\00_绘图脚本\plot_task2_all.py'",
        "```",
        "",
        "若只需要论文插图，优先使用 PDF；WPS兼容性不佳时使用同名PNG。",
    ]
    (IMAGE_ROOT / "README_任务二正式图件.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.time()
    # Seaborn resets font.family to sans-serif, so apply the paper-wide font
    # settings after the Seaborn theme rather than before it.
    sns.set_theme(style="whitegrid")
    apply_style()
    make_dirs()
    copy_plot_sources()
    d = load_all()
    tasks = [
        ("FigT2_01", lambda: plot_label_distribution(d)),
        ("FigT2_02", lambda: plot_score_distribution(d)),
        ("FigT2_03", lambda: plot_fold_bins(d)),
        ("FigT2_04", lambda: plot_missing_rates(d)),
        ("FigT2_05", lambda: plot_standardized_boxplots(d)),
        ("FigT2_06", lambda: plot_correlation_heatmap(d)),
        ("FigT2_07", lambda: plot_tuning_history(d)),
        ("FigT2_08", lambda: plot_sampling_comparison(d)),
        ("FigT2_09", lambda: plot_model_performance(d)),
        ("FigT2_10", lambda: plot_class_metrics(d)),
        ("FigT2_11_12", lambda: plot_confusions(d)),
        ("FigT2_13_14", lambda: plot_roc_pr(d)),
        ("FigT2_15", lambda: plot_calibration(d)),
        ("FigT2_16", lambda: plot_grade_errors(d)),
        ("FigT2_17", lambda: plot_prediction_distribution(d)),
        ("FigT2_18", lambda: plot_shap_beeswarm(d)),
        ("FigT2_19", lambda: plot_shap_global_heatmap(d)),
        ("FigT2_20", lambda: plot_shap_dependence(d)),
        ("FigT2_21", lambda: plot_ordinal_coefficients(d)),
        ("FigT2_22", lambda: plot_feature_importance(d)),
        ("FigT2_23", lambda: plot_sankey(d)),
        ("FigT2_24", lambda: plot_tsne_umap(d)),
        ("FigT2_25", lambda: plot_mi_network(d)),
        ("FigT2_26", plot_pipeline_route),
        ("FigT2_27", lambda: plot_scope_comparison(d)),
        ("FigT2_28", lambda: plot_radar(d)),
        ("FigT2_29", lambda: plot_local_shap(d)),
    ]
    log = []
    for name, task in tasks:
        t0 = time.time()
        try:
            task()
            msg = {"task": name, "status": "ok", "seconds": round(time.time() - t0, 2)}
        except Exception as exc:
            msg = {"task": name, "status": "failed", "seconds": round(time.time() - t0, 2), "error": str(exc), "traceback": traceback.format_exc()}
        log.append(msg)
        print(json.dumps(msg, ensure_ascii=False), flush=True)
    write_manifest()
    summary = {
        "started": started,
        "seconds": round(time.time() - started, 2),
        "figures_registered": len(MANIFEST),
        "png_count": len(list(OUTPUT_ROOT.rglob("*.png"))),
        "pdf_count": len(list(OUTPUT_ROOT.rglob("*.pdf"))),
        "html_count": len(list(OUTPUT_ROOT.rglob("*.html"))),
        "tasks": log,
    }
    (IMAGE_ROOT / "plot_generation_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
