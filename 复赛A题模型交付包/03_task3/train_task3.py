from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from src.modeling import (
    DIRECT_LEAKAGE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    ResidualStackRegressor,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "A_dataset.csv"
MODEL_DIR = ROOT / "models"
PRED_DIR = ROOT / "predictions"
FIG_DIR = ROOT / "figures"
RANDOM_STATE = 2026
HOLDOUT_SIZE = 0.20
N_SPLITS = 5


def metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def fit_oof_weights(X, y):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    names = ["linear", "hist_residual", "extra_residual", "gbr_residual"]
    linear_oof = np.zeros(len(X), dtype=float)
    corrections = {name: np.zeros(len(X), dtype=float) for name in names[1:]}
    fold_ids = np.zeros(len(X), dtype=int)
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        model = ResidualStackRegressor(random_state=RANDOM_STATE + fold)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        comp = model.predict_components(X.iloc[va_idx])
        linear_oof[va_idx] = comp["linear"]
        for name in names[1:]:
            corrections[name][va_idx] = comp[name] - comp["linear"]
        fold_ids[va_idx] = fold

    corr_matrix = np.column_stack([corrections[n] for n in names[1:]])

    def objective(w):
        pred = np.clip(linear_oof + corr_matrix @ w, 0.0, 100.0)
        return np.mean((y.to_numpy() - pred) ** 2)

    result = minimize(
        objective,
        x0=np.array([0.3, 0.7, 0.3]),
        method="SLSQP",
        bounds=[(0.0, 1.2)] * 3,
        options={"maxiter": 300, "ftol": 1e-12},
    )
    weights = {name: float(w) for name, w in zip(names[1:], result.x)}
    final_oof = np.clip(linear_oof + corr_matrix @ result.x, 0.0, 100.0)

    candidates = {"Ridge线性主干": linear_oof}
    for name_cn, name in [
        ("线性+HistGB残差", "hist_residual"),
        ("线性+ExtraTrees残差", "extra_residual"),
        ("线性+GBR残差", "gbr_residual"),
    ]:
        candidates[name_cn] = np.clip(linear_oof + corrections[name], 0.0, 100.0)
    candidates["OOF加权残差集成"] = final_oof

    comparison = []
    for model_name, pred in candidates.items():
        row = {"Model": model_name, **metrics(y, pred)}
        comparison.append(row)
    comparison_df = pd.DataFrame(comparison).sort_values("R2", ascending=False)

    for fold in range(1, N_SPLITS + 1):
        mask = fold_ids == fold
        fold_rows.append({"Fold": fold, **metrics(y.iloc[mask], final_oof[mask])})
    fold_df = pd.DataFrame(fold_rows)
    return weights, final_oof, fold_ids, comparison_df, fold_df


def grouped_permutation_importance(model, X, y, repeats=5):
    rng = np.random.default_rng(RANDOM_STATE)
    baseline = r2_score(y, model.predict(X))
    rows = []
    for col in X.columns:
        if col in {ID_COLUMN, TARGET_COLUMN, *DIRECT_LEAKAGE_COLUMNS}:
            continue
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            drops.append(baseline - r2_score(y, model.predict(shuffled)))
        rows.append(
            {
                "Feature": col,
                "Importance_Mean": float(np.mean(drops)),
                "Importance_STD": float(np.std(drops, ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values("Importance_Mean", ascending=False)


def configure_plots():
    from matplotlib import font_manager

    local_fonts = [
        ROOT / "assets" / "NotoSansCJKsc-Regular.otf",
        ROOT.parents[1] / "work" / "fonts" / "NotoSansCJKsc-Regular.otf",
    ]
    for font in local_fonts:
        if font.exists():
            font_manager.fontManager.addfont(str(font))
            plt.rcParams["font.family"] = "Noto Sans CJK SC"
            break
    else:
        available = {item.name for item in font_manager.fontManager.ttflist}
        for family in ["Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei", "Arial Unicode MS"]:
            if family in available:
                plt.rcParams["font.family"] = family
                break
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#333333",
            "text.color": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "grid.color": "#D9DEE7",
            "grid.alpha": 0.55,
        }
    )


def save_figures(df, y_holdout, holdout_pred, comparison, folds, importance, ablation):
    configure_plots()
    navy, blue, gold, gray = "#17365D", "#3F6F9F", "#C59A3D", "#8A96A3"

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    sns.histplot(df[TARGET_COLUMN], bins=32, kde=True, color=blue, ax=axes[0])
    axes[0].axvline(df[TARGET_COLUMN].mean(), color=gold, lw=2, label=f"均值 {df[TARGET_COLUMN].mean():.2f}")
    axes[0].set(title="Health Score 分布", xlabel="综合健康评分", ylabel="样本数")
    axes[0].legend(frameon=False)
    order = ["Poor", "Average", "Good", "Excellent"]
    tmp = df.groupby("Wellness_Category")[TARGET_COLUMN].agg(["min", "max", "mean", "count"]).reindex(order)
    axes[1].bar(tmp.index, tmp["mean"], color=[gray, "#6F8195", blue, navy])
    for i, (_, row) in enumerate(tmp.iterrows()):
        axes[1].text(i, row["mean"] + 1.3, f"{row['min']:.1f}-{row['max']:.1f}", ha="center", fontsize=8)
    axes[1].set(title="目标分档字段的泄露证据", xlabel="Wellness Category", ylabel="组内 Health Score 均值", ylim=(0, 105))
    fig.savefig(FIG_DIR / "Fig01_target_and_leakage.png", bbox_inches="tight")
    plt.close(fig)

    numeric = df.select_dtypes(include=np.number)
    top = numeric.corrwith(df[TARGET_COLUMN]).abs().sort_values(ascending=False).head(13).index
    corr = numeric[top].corr()
    fig, ax = plt.subplots(figsize=(9.2, 7.4), constrained_layout=True)
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, square=True, linewidths=0.45, cbar_kws={"shrink": 0.78}, ax=ax)
    ax.set_title("高相关连续变量 Pearson 相关矩阵", pad=12)
    ax.tick_params(axis="x", rotation=48, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.savefig(FIG_DIR / "Fig02_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    comp = comparison.sort_values("R2")
    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    bars = ax.barh(comp["Model"], comp["R2"], color=[gray] * (len(comp) - 1) + [navy])
    ax.set_xlim(max(0.90, comp["R2"].min() - 0.006), min(1.0, comp["R2"].max() + 0.004))
    ax.set_xlabel("五折 OOF R²")
    ax.set_title("候选模型与残差集成性能比较")
    ax.grid(axis="x")
    for bar, value in zip(bars, comp["R2"]):
        ax.text(value + 0.00025, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=9)
    fig.savefig(FIG_DIR / "Fig03_model_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.8), constrained_layout=True)
    ax.scatter(y_holdout, holdout_pred, s=16, alpha=0.42, color=blue, edgecolors="none")
    lo = min(y_holdout.min(), holdout_pred.min())
    hi = max(y_holdout.max(), holdout_pred.max())
    ax.plot([lo, hi], [lo, hi], color=gold, lw=2, label="理想预测线")
    m = metrics(y_holdout, holdout_pred)
    ax.text(0.035, 0.955, f"R² = {m['R2']:.4f}\nRMSE = {m['RMSE']:.4f}\nMAE = {m['MAE']:.4f}", transform=ax.transAxes, va="top", bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": "#CDD4DD"})
    ax.set(title="锁定验证集：真实值与预测值", xlabel="真实 Health Score", ylabel="预测 Health Score")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True)
    fig.savefig(FIG_DIR / "Fig04_true_vs_predicted.png", bbox_inches="tight")
    plt.close(fig)

    residual = y_holdout.to_numpy() - holdout_pred
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    axes[0].scatter(holdout_pred, residual, s=15, alpha=0.40, color=blue, edgecolors="none")
    axes[0].axhline(0, color=gold, lw=2)
    axes[0].set(title="残差-预测值图", xlabel="预测 Health Score", ylabel="残差（真实-预测）")
    axes[0].grid(True)
    sns.histplot(residual, bins=32, kde=True, color=blue, ax=axes[1])
    axes[1].axvline(0, color=gold, lw=2)
    axes[1].set(title="残差分布", xlabel="残差", ylabel="样本数")
    fig.savefig(FIG_DIR / "Fig05_residual_diagnostics.png", bbox_inches="tight")
    plt.close(fig)

    imp = importance.head(15)
    leader = imp.iloc[[0]]
    rest = imp.iloc[1:].sort_values("Importance_Mean")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.8), gridspec_kw={"width_ratios": [0.9, 2.5]}, constrained_layout=True)
    axes[0].barh(leader["Feature"], leader["Importance_Mean"], xerr=leader["Importance_STD"], color=navy, ecolor=gray, capsize=2)
    axes[0].set(title="首要代理指标", xlabel="R² 平均下降量", ylabel="")
    axes[0].grid(axis="x")
    axes[1].barh(rest["Feature"], rest["Importance_Mean"], xerr=rest["Importance_STD"], color=blue, ecolor=gray, capsize=2)
    axes[1].set(title="其余关键驱动因素（放大）", xlabel="R² 平均下降量", ylabel="")
    axes[1].grid(axis="x")
    fig.suptitle("锁定验证集分组置换重要性", fontweight="bold")
    fig.savefig(FIG_DIR / "Fig06_permutation_importance.png", bbox_inches="tight")
    plt.close(fig)

    abl = ablation.sort_values("R2")
    fig, ax = plt.subplots(figsize=(8.8, 4.5), constrained_layout=True)
    bars = ax.barh(abl["Scope"], abl["R2"], color=[gray] * (len(abl) - 1) + [navy])
    ax.set_xlim(max(0.0, abl["R2"].min() - 0.025), min(1.0, abl["R2"].max() + 0.008))
    ax.set(title="特征口径消融实验", xlabel="锁定验证集 R²")
    ax.grid(axis="x")
    for b, value in zip(bars, abl["R2"]):
        ax.text(value + 0.001, b.get_y() + b.get_height()/2, f"{value:.4f}", va="center", fontsize=9)
    fig.savefig(FIG_DIR / "Fig07_ablation.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.5), constrained_layout=True)
    ax.errorbar(folds["Fold"], folds["R2"], yerr=None, marker="o", ms=7, lw=2, color=navy)
    ax.axhline(folds["R2"].mean(), color=gold, ls="--", lw=1.8, label=f"均值 {folds['R2'].mean():.4f}")
    for x, value in zip(folds["Fold"], folds["R2"]):
        ax.text(x, value + 0.0008, f"{value:.4f}", ha="center", fontsize=8)
    ax.set_xticks(folds["Fold"])
    ax.set(title="五折 OOF 稳定性", xlabel="折次", ylabel="R²", ylim=(folds["R2"].min() - 0.004, folds["R2"].max() + 0.004))
    ax.legend(frameon=False)
    ax.grid(axis="y")
    fig.savefig(FIG_DIR / "Fig08_fold_stability.png", bbox_inches="tight")
    plt.close(fig)


def main():
    for directory in [MODEL_DIR, PRED_DIR, FIG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"训练文件必须包含目标列 {TARGET_COLUMN}")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Person_ID 必须唯一")

    train_idx, holdout_idx = train_test_split(
        np.arange(len(df)),
        test_size=HOLDOUT_SIZE,
        random_state=RANDOM_STATE,
    )
    dev = df.iloc[train_idx].reset_index(drop=True)
    holdout = df.iloc[holdout_idx].reset_index(drop=True)
    X_dev, y_dev = dev.drop(columns=[TARGET_COLUMN]), dev[TARGET_COLUMN]
    X_holdout, y_holdout = holdout.drop(columns=[TARGET_COLUMN]), holdout[TARGET_COLUMN]

    weights, oof_pred, fold_ids, comparison, fold_metrics = fit_oof_weights(X_dev, y_dev)
    holdout_model = ResidualStackRegressor(residual_weights=weights, random_state=RANDOM_STATE)
    holdout_model.fit(X_dev, y_dev)
    holdout_pred = holdout_model.predict(X_holdout)
    holdout_metrics = metrics(y_holdout, holdout_pred)

    holdout_predictions = pd.DataFrame(
        {
            ID_COLUMN: holdout[ID_COLUMN].to_numpy(),
            "True_Health_Score": y_holdout.to_numpy(),
            "Predicted_Health_Score": np.round(holdout_pred, 6),
            "Residual": np.round(y_holdout.to_numpy() - holdout_pred, 6),
        }
    )
    holdout_predictions.to_csv(PRED_DIR / "task3_validation_predictions.csv", index=False, encoding="utf-8-sig")

    oof_predictions = pd.DataFrame(
        {
            ID_COLUMN: dev[ID_COLUMN].to_numpy(),
            "Fold": fold_ids,
            "True_Health_Score": y_dev.to_numpy(),
            "OOF_Predicted_Health_Score": np.round(oof_pred, 6),
            "Residual": np.round(y_dev.to_numpy() - oof_pred, 6),
        }
    )
    oof_predictions.to_csv(PRED_DIR / "task3_oof_predictions.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(PRED_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(PRED_DIR / "fold_metrics.csv", index=False, encoding="utf-8-sig")

    importance = grouped_permutation_importance(holdout_model, X_holdout, y_holdout, repeats=5)
    importance.to_csv(PRED_DIR / "permutation_importance.csv", index=False, encoding="utf-8-sig")

    ablation_rows = [{"Scope": "全部有效特征（剔除直接泄露）", **holdout_metrics}]
    for label, drops in [
        ("去除 Healthy Aging Score", ["Healthy_Aging_Score"]),
        ("去除健康衍生评分", ["Healthy_Aging_Score", "Fitness_Level"]),
    ]:
        reduced_dev = X_dev.drop(columns=drops, errors="ignore")
        reduced_holdout = X_holdout.drop(columns=drops, errors="ignore")
        reduced_model = ResidualStackRegressor(residual_weights=weights, random_state=RANDOM_STATE)
        reduced_model.fit(reduced_dev, y_dev)
        reduced_pred = reduced_model.predict(reduced_holdout)
        ablation_rows.append({"Scope": label, **metrics(y_holdout, reduced_pred)})
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(PRED_DIR / "ablation_results.csv", index=False, encoding="utf-8-sig")

    save_figures(df, y_holdout, holdout_pred, comparison, fold_metrics, importance, ablation)

    joblib.dump(holdout_model, MODEL_DIR / "task3_holdout_model.joblib", compress=3)
    final_model = ResidualStackRegressor(residual_weights=weights, random_state=RANDOM_STATE)
    final_model.fit(df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN])
    joblib.dump(final_model, MODEL_DIR / "task3_final_model.joblib", compress=3)

    summary = {
        "task": "复赛A题任务三：Health_Score回归",
        "target": TARGET_COLUMN,
        "id_column": ID_COLUMN,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "random_state": RANDOM_STATE,
        "holdout_size": HOLDOUT_SIZE,
        "cv_folds": N_SPLITS,
        "removed_direct_leakage": DIRECT_LEAKAGE_COLUMNS,
        "residual_weights": weights,
        "oof_metrics": metrics(y_dev, oof_pred),
        "holdout_metrics": holdout_metrics,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (PRED_DIR / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
