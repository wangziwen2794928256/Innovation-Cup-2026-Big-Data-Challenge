"""基于模型包中的固定 CSV 图源，生成论文修订版矢量图。

输出规则：
1. 仅输出 SVG；
2. 全部图内文字使用中文；
3. 统一字体、字号、配色和线宽；
4. 不绘制脚注、说明性小字或子图编号。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd
import seaborn as sns


NAVY = "#183B5B"
BLUE = "#3E7CB1"
TEAL = "#2A9D8F"
GOLD = "#E9A23B"
CORAL = "#E76F51"
INK = "#1E2A36"
MID = "#657483"
GRID = "#D9E3EA"
PALE_BLUE = "#EAF2F8"
PALE_TEAL = "#E7F4F1"
PALE_GOLD = "#FFF4DE"
PALE_GRAY = "#F3F6F8"

CMAP_BLUE = LinearSegmentedColormap.from_list(
    "论文蓝", ["#F4F8FB", "#C7DBEA", "#6FA1C7", NAVY]
)
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "论文相关", [NAVY, "#B8D1E4", "#F7F8F8", "#F3C8A5", CORAL]
)

LABELS = {
    "Age": "年龄",
    "Weight_kg": "体重",
    "BMI": "体重指数",
    "Exercise_Frequency_Per_Week": "每周运动频次",
    "Stress_Level": "压力水平",
    "Mood_Score": "情绪评分",
    "Fatigue_Level_Score": "疲劳评分",
    "Anxiety_Score": "焦虑评分",
    "Productivity_Score": "生产力评分",
    "Focus_Concentration_Score": "专注力评分",
    "Meditation_Practice": "冥想习惯",
    "Depression_Risk_Score": "抑郁风险评分",
    "Energy_Level_Score": "精力评分",
    "Social_Interaction_Score": "社交互动评分",
    "Sleep_Quality_Score": "睡眠质量评分",
    "Early_Waker": "早起习惯",
    "Morning_Workout": "晨练习惯",
    "Sleep_Time": "入睡时刻",
    "Wake_Up_Time": "起床时刻",
    "Sleep_Duration_Hours": "睡眠时长",
    "Obesity_Risk": "肥胖风险",
    "Diabetes_Risk": "糖尿病风险",
    "Sleep_Disorder_Risk": "睡眠障碍风险",
    "Daily_Steps": "每日步数",
    "Systolic_BP": "收缩压",
    "Water_Intake_Liters": "每日饮水量",
    "Working_Hours_Per_Day": "每日工作时长",
    "Sitting_Hours_Per_Day": "每日久坐时长",
    "Screen_Time_Before_Bed_Hours": "睡前屏幕时长",
    "Healthy_Aging_Score": "健康老龄化评分",
    "Cardiovascular_Risk": "心血管风险",
    "Fitness_Level": "体能水平",
    "Fast_Food_Meals_Per_Week": "每周快餐频次",
    "Smoking_Status": "吸烟状态",
    "Vegetable_Intake_Per_Day": "每日蔬菜摄入",
    "Health_Score": "综合健康评分",
    "Life_Satisfaction_Score": "生活满意度评分",
}

TASK1_MODEL_NAMES = {
    "ridge_alpha1": "岭回归 α=1（入选）",
    "ridge_001": "岭回归 α=0.01",
    "hgb_15": "直方图梯度提升（15叶）",
    "ridge_1000": "岭回归 α=1000",
    "rf_128": "随机森林（128叶）",
    "extra_128": "极端随机树（128叶）",
}

TASK2_MODEL_NAMES = {
    "ElasticNet": "弹性网络",
    "Ridge α=10": "岭回归 α=10（入选）",
    "Ridge α=100": "岭回归 α=100",
    "Ridge α=1": "岭回归 α=1",
    "Ridge α=0.1": "岭回归 α=0.1",
    "HistGradientBoosting": "直方图梯度提升",
    "ExtraTrees": "极端随机树",
}

TASK3_MODEL_NAMES = {
    "OOF加权残差集成": "折外加权残差集成",
    "线性+GBR残差": "线性主干＋梯度提升残差",
    "线性+HistGB残差": "线性主干＋直方图提升残差",
    "线性+ExtraTrees残差": "线性主干＋极端随机树残差",
    "Ridge线性主干": "岭回归线性主干",
}


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(1 - np.square(y_true - y_pred).sum() / np.square(y_true - y_true.mean()).sum())


def mean_squared_error(y_true, y_pred) -> float:
    return float(np.square(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)).mean())


def mean_absolute_error(y_true, y_pred) -> float:
    return float(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)).mean())


def configure_style() -> None:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    font_path = next((p for p in candidates if p.exists()), None)
    if font_path is None:
        for family_name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
            try:
                font_path = Path(fm.findfont(family_name, fallback_to_default=False))
                break
            except ValueError:
                continue
    if font_path is None:
        raise FileNotFoundError("未找到可用的中文字体。")
    fm.fontManager.addfont(str(font_path))
    family = fm.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": MID,
            "axes.linewidth": 0.9,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": MID,
            "ytick.color": MID,
            "text.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "legend.fontsize": 10.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def clean_axis(ax: plt.Axes, grid: str | None = None) -> None:
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid)


def save_svg(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.12,
        metadata={"Date": None},
    )
    plt.close(fig)
    text = path.read_text(encoding="utf-8")
    if "<image" in text:
        raise RuntimeError(f"{path.name} 含有栅格图像节点，不符合纯矢量要求。")


def metric_box(ax: plt.Axes, rows: list[str], loc=(0.04, 0.95)) -> None:
    ax.text(
        loc[0],
        loc[1],
        "\n".join(rows),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={
            "boxstyle": "round,pad=0.48,rounding_size=0.18",
            "fc": "white",
            "ec": GRID,
            "lw": 1,
            "alpha": 0.95,
        },
    )


def display_sample(values, limit: int = 1400):
    """为矢量散点图做确定性抽样，统计量仍使用完整数据计算。"""
    if len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, limit, dtype=int)
    if hasattr(values, "iloc"):
        return values.iloc[indices]
    return np.asarray(values)[indices]


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    face: str,
    edge: str,
    fontsize: float = 12,
    weight: str = "normal",
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=INK,
        linespacing=1.35,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color=MID,
            shrinkA=2,
            shrinkB=2,
            connectionstyle="arc3,rad=0",
        )
    )


def draw_flowcharts(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("三项任务统一建模技术路线", loc="left", pad=12)

    ax.add_patch(
        FancyBboxPatch(
            (0.025, 0.56),
            0.95,
            0.34,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            facecolor=PALE_GRAY,
            edgecolor="none",
        )
    )
    ax.text(0.045, 0.86, "数据治理与验证", color=NAVY, fontsize=12.5, fontweight="bold")
    xs = [0.055, 0.285, 0.515, 0.745]
    texts = [
        "原始数据\n样本与字段审计",
        "数据治理\n缺失、类型与边界",
        "折内预处理\n周期编码与标准化",
        "固定验证\n五折选模与锁定评估",
    ]
    faces = [PALE_BLUE, PALE_TEAL, PALE_BLUE, PALE_GOLD]
    edges = [BLUE, TEAL, BLUE, GOLD]
    for x, label, face, edge in zip(xs, texts, faces, edges):
        rounded_box(ax, x, 0.64, 0.20, 0.15, label, face, edge, fontsize=11.5)
    for left, right in zip(xs[:-1], xs[1:]):
        arrow(ax, (left + 0.20, 0.715), (right, 0.715))

    ax.text(0.045, 0.49, "三项任务建模", color=NAVY, fontsize=12.5, fontweight="bold")
    task_x = [0.055, 0.38, 0.705]
    task_text = [
        "任务一\n睡眠质量回归",
        "任务二\n生产力与精力分析",
        "任务三\n综合健康预测",
    ]
    task_faces = [PALE_BLUE, PALE_TEAL, PALE_GOLD]
    task_edges = [BLUE, TEAL, GOLD]
    for x, label, face, edge in zip(task_x, task_text, task_faces, task_edges):
        rounded_box(ax, x, 0.29, 0.24, 0.14, label, face, edge, fontsize=12, weight="bold")
        arrow(ax, (0.845, 0.64), (x + 0.12, 0.43))

    rounded_box(
        ax,
        0.30,
        0.06,
        0.40,
        0.12,
        "统一输出\n性能评估、变量重要性与误差诊断",
        "white",
        TEAL,
        fontsize=11.8,
        weight="bold",
    )
    for x in task_x:
        arrow(ax, (x + 0.12, 0.29), (0.50, 0.18))
    save_svg(fig, out / "图01_三任务统一建模技术路线.svg")

    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("任务一特征边界与固定验证流程", loc="left", pad=12)
    for x, heading, face in [
        (0.035, "输入边界", PALE_BLUE),
        (0.515, "固定划分与模型评估", PALE_TEAL),
    ]:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.06),
                0.45,
                0.84,
                boxstyle="round,pad=0.015,rounding_size=0.03",
                facecolor=face,
                edgecolor="none",
            )
        )
        ax.text(x + 0.025, 0.845, heading, fontsize=13, fontweight="bold", color=NAVY)

    rounded_box(ax, 0.085, 0.66, 0.35, 0.12, "原始数据\n10,000 人，64 列", "white", BLUE, 12)
    rounded_box(ax, 0.085, 0.43, 0.35, 0.14, "模型输入\n34 个数值字段＋14 个类别字段", "white", TEAL, 12)
    rounded_box(ax, 0.085, 0.18, 0.35, 0.15, "明确排除字段\n8 个睡眠字段＋2 个边界字段\n＋4 个综合字段＋编号＋目标", "white", MID, 11.5)
    arrow(ax, (0.26, 0.66), (0.26, 0.57))
    arrow(ax, (0.26, 0.43), (0.26, 0.33))

    rounded_box(ax, 0.565, 0.70, 0.35, 0.10, "按样本编号哈希划分", "white", TEAL, 12, "bold")
    rounded_box(ax, 0.55, 0.52, 0.18, 0.10, "开发集\n8,000 人", "white", BLUE, 11.5)
    rounded_box(ax, 0.75, 0.52, 0.18, 0.10, "锁定验证集\n2,000 人", "white", GOLD, 11.5)
    rounded_box(ax, 0.55, 0.34, 0.18, 0.10, "开发集五折比较\n仅在训练折拟合", "white", BLUE, 11)
    rounded_box(ax, 0.55, 0.16, 0.18, 0.10, "固定岭回归 α=1\n完整开发集重拟合", "white", BLUE, 11)
    rounded_box(ax, 0.75, 0.16, 0.18, 0.10, "一次锁定评估\n保留审计结果", "white", GOLD, 11)
    arrow(ax, (0.74, 0.70), (0.64, 0.62))
    arrow(ax, (0.74, 0.70), (0.84, 0.62))
    arrow(ax, (0.64, 0.52), (0.64, 0.44))
    arrow(ax, (0.64, 0.34), (0.64, 0.26))
    arrow(ax, (0.73, 0.21), (0.75, 0.21))
    arrow(ax, (0.84, 0.52), (0.84, 0.26))
    save_svg(fig, out / "图02_任务一特征边界与固定验证流程.svg")


def draw_task1(root: Path, out: Path) -> None:
    src = root / "01_task1" / "plotting" / "source_data"
    comp = pd.read_csv(src / "model_comparison.csv")
    dev = pd.read_csv(src / "development_oof.csv")
    locked = pd.read_csv(src / "locked_test_diagnostics.csv")
    imp = pd.read_csv(src / "permutation_importance_all48.csv")

    ordered = comp.sort_values("fold_r2_mean", ascending=True).copy()
    names = [TASK1_MODEL_NAMES[x] for x in ordered["candidate"]]
    colors = [GOLD if bool(x) else TEAL for x in ordered["selected"]]
    fig, ax = plt.subplots(figsize=(9.6, 6.2), constrained_layout=True)
    y = np.arange(len(ordered))
    ax.errorbar(
        ordered["fold_r2_mean"],
        y,
        xerr=ordered["fold_r2_std_ddof1"],
        fmt="none",
        ecolor=INK,
        elinewidth=2,
        capsize=4,
        zorder=2,
    )
    ax.scatter(
        ordered["fold_r2_mean"], y, s=90, c=colors, edgecolor="white", linewidth=1.2, zorder=3
    )
    for yi, val in zip(y, ordered["fold_r2_mean"]):
        ax.text(val + 0.003, yi, f"{val:.4f}", va="center", fontsize=10.5)
    ax.set_yticks(y, names)
    ax.set_xlim(0.55, 0.69)
    ax.set_xlabel("五折验证决定系数 R²")
    ax.set_title("任务一开发集五折模型比较", loc="left")
    clean_axis(ax, "x")
    save_svg(fig, out / "图03_任务一五折模型比较.svg")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)
    for ax, df, heading, color in [
        (axes[0], dev, "开发集折外预测", BLUE),
        (axes[1], locked, "锁定验证集预测", TEAL),
    ]:
        shown = display_sample(df)
        ax.scatter(
            shown["true_value"],
            shown["predicted_value"],
            s=10,
            alpha=0.24,
            color=color,
            edgecolors="none",
        )
        lo = min(df["true_value"].min(), df["predicted_value"].min())
        hi = max(df["true_value"].max(), df["predicted_value"].max())
        ax.plot([lo, hi], [lo, hi], color=GOLD, lw=2.2, label="理想预测线")
        r2 = r2_score(df["true_value"], df["predicted_value"])
        rmse = mean_squared_error(df["true_value"], df["predicted_value"]) ** 0.5
        metric_box(ax, [f"样本量：{len(df):,} 人", f"R²：{r2:.4f}", f"均方根误差：{rmse:.4f}"])
        ax.set_title(heading, color=color)
        ax.set_xlabel("真实睡眠质量评分")
        ax.set_ylabel("预测睡眠质量评分")
        ax.legend(frameon=False, loc="lower right")
        clean_axis(ax, "both")
    fig.suptitle("任务一真实值与预测值对照", x=0.01, ha="left", fontsize=16, fontweight="bold")
    save_svg(fig, out / "图04a_任务一预测对照.svg")

    residual = locked["residual_true_minus_pred"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), constrained_layout=True)
    shown = display_sample(locked)
    axes[0].scatter(
        shown["predicted_value"], shown["residual_true_minus_pred"], s=14, alpha=0.38, color=BLUE, edgecolors="none"
    )
    axes[0].axhline(0, color=GOLD, lw=2)
    axes[0].set_title("残差与预测值")
    axes[0].set_xlabel("预测睡眠质量评分")
    axes[0].set_ylabel("残差（真实值－预测值）")
    clean_axis(axes[0], "both")
    sns.histplot(residual, bins=34, kde=True, color=TEAL, ax=axes[1])
    axes[1].axvline(0, color=GOLD, lw=2, label="零残差")
    axes[1].set_title("残差分布")
    axes[1].set_xlabel("残差")
    axes[1].set_ylabel("样本数")
    axes[1].legend(frameon=False)
    clean_axis(axes[1], "y")
    fig.suptitle("任务一锁定验证残差诊断", x=0.01, ha="left", fontsize=16, fontweight="bold")
    save_svg(fig, out / "图04b_任务一锁定验证残差诊断.svg")

    top = imp.head(12).sort_values("mean_r2_drop")
    fig, ax = plt.subplots(figsize=(9.8, 6.5), constrained_layout=True)
    ypos = np.arange(len(top))
    bars = ax.barh(ypos, top["mean_r2_drop"], color=[TEAL] * 9 + [BLUE] * 3, height=0.66)
    ax.errorbar(
        top["mean_r2_drop"], ypos, xerr=top["sd_five_fold_means"], fmt="none", ecolor=INK, capsize=3
    )
    ax.set_yticks(ypos, [LABELS.get(x, x) for x in top["feature"]])
    ax.set_xlabel("置换后 R² 平均下降量")
    ax.set_title("任务一开发集五折置换重要性", loc="left")
    clean_axis(ax, "x")
    for bar, value in zip(bars, top["mean_r2_drop"]):
        ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center")
    save_svg(fig, out / "图05a_任务一置换重要性.svg")

    display = pd.read_csv(src / "correlation_display_fields.csv").sort_values("display_order")["feature"].tolist()
    long = pd.read_csv(src / "development_numeric_correlations_full.csv")
    corr = long.pivot(index="feature_x", columns="feature_y", values="pearson_r").loc[display, display]
    corr.index = [LABELS.get(x, x) for x in corr.index]
    corr.columns = [LABELS.get(x, x) for x in corr.columns]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10.0, 8.0), constrained_layout=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=CMAP_DIVERGING,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.6,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 9.5},
        cbar=False,
        ax=ax,
    )
    ax.set_title("任务一主要数值字段相关结构", loc="left", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=42)
    ax.tick_params(axis="y", rotation=0)
    ax.legend(
        handles=[Patch(facecolor=NAVY, label="负相关"), Patch(facecolor="#F7F8F8", edgecolor=GRID, label="接近零"), Patch(facecolor=CORAL, label="正相关")],
        frameon=False,
        ncol=3,
        loc="upper right",
    )
    save_svg(fig, out / "图05b_任务一数值字段相关结构.svg")


def importance_bar(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    value_col: str,
    std_col: str,
    color: str,
    n: int = 12,
) -> None:
    data = df.head(n).sort_values(value_col)
    fig, ax = plt.subplots(figsize=(9.8, 6.4), constrained_layout=True)
    ypos = np.arange(len(data))
    bars = ax.barh(ypos, data[value_col], color=color, height=0.66)
    ax.errorbar(data[value_col], ypos, xerr=data[std_col], fmt="none", ecolor=INK, capsize=3)
    ax.set_yticks(ypos, [LABELS.get(x, x) for x in data["Feature"]])
    ax.set_xlabel("置换后 R² 平均下降量")
    ax.set_title(title, loc="left")
    clean_axis(ax, "x")
    pad = max(data[value_col].max() * 0.025, 0.002)
    for bar, value in zip(bars, data[value_col]):
        ax.text(value + pad, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center")
    save_svg(fig, out_path)


def draw_task2(root: Path, out: Path) -> None:
    src = root / "02_task2" / "plotting" / "source_data"
    df = pd.read_csv(src / "A_dataset.csv")
    comp = pd.read_csv(src / "model_comparison.csv")
    val = pd.read_csv(src / "validation_predictions.csv")
    prod_imp = pd.read_csv(src / "permutation_importance.csv")
    energy_imp = pd.read_csv(src / "energy_driver_importance.csv")
    folds = pd.read_csv(src / "fold_metrics.csv")
    ablation = pd.read_csv(src / "scope_ablation.csv")

    target = df["Productivity_Score"]
    fig, ax = plt.subplots(figsize=(9.4, 5.6), constrained_layout=True)
    sns.histplot(target, bins=32, kde=True, color=BLUE, edgecolor="white", linewidth=0.5, ax=ax)
    ax.axvline(target.mean(), color=GOLD, lw=2.4, label=f"均值：{target.mean():.2f} 分")
    ax.set_title("生产力评分分布", loc="left")
    ax.set_xlabel("生产力评分（分）")
    ax.set_ylabel("样本数")
    ax.legend(frameon=False)
    clean_axis(ax, "y")
    save_svg(fig, out / "图06a_生产力评分分布.svg")

    focus = [
        "Productivity_Score",
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
    corr.index = [LABELS[x] for x in corr.index]
    corr.columns = [LABELS[x] for x in corr.columns]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10.3, 8.6), constrained_layout=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=CMAP_DIVERGING,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.55,
        linecolor="white",
        cbar=False,
        ax=ax,
    )
    ax.set_title("生产力及主要连续指标相关结构", loc="left", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=43)
    ax.tick_params(axis="y", rotation=0)
    ax.legend(
        handles=[Patch(facecolor=NAVY, label="负相关"), Patch(facecolor="#F7F8F8", edgecolor=GRID, label="接近零"), Patch(facecolor=CORAL, label="正相关")],
        frameon=False,
        ncol=3,
        loc="upper right",
    )
    save_svg(fig, out / "图06b_生产力连续指标相关结构.svg")

    data = comp.sort_values("Adjusted_R2_Transformed")
    fig, ax = plt.subplots(figsize=(9.8, 6.0), constrained_layout=True)
    colors = [GOLD if bool(v) else BLUE for v in data["Selected"]]
    bars = ax.barh(
        [TASK2_MODEL_NAMES[x] for x in data["Model"]],
        data["Adjusted_R2_Transformed"],
        color=colors,
        height=0.66,
    )
    ax.set_xlim(0, 0.67)
    ax.set_xlabel("调整后折外决定系数 R²")
    ax.set_title("任务二候选模型比较", loc="left")
    clean_axis(ax, "x")
    for bar, value in zip(bars, data["Adjusted_R2_Transformed"]):
        ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center")
    save_svg(fig, out / "图07a_任务二候选模型比较.svg")

    scope = ablation.sort_values("Adjusted_R2_Transformed")
    fig, ax = plt.subplots(figsize=(9.6, 4.9), constrained_layout=True)
    bars = ax.barh(scope["Scope"], scope["Adjusted_R2_Transformed"], color=[MID, BLUE, TEAL], height=0.62)
    ax.set_xlim(0, 0.67)
    ax.set_xlabel("调整后折外决定系数 R²")
    ax.set_title("任务二特征口径消融", loc="left")
    clean_axis(ax, "x")
    for bar, value in zip(bars, scope["Adjusted_R2_Transformed"]):
        ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center")
    save_svg(fig, out / "图07b_任务二特征口径消融.svg")

    y_true = val["true_value"].to_numpy()
    y_pred = val["predicted_value"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.8, 6.2), constrained_layout=True)
    shown_idx = np.linspace(0, len(y_true) - 1, min(len(y_true), 1400), dtype=int)
    ax.scatter(y_true[shown_idx], y_pred[shown_idx], s=11, alpha=0.25, color=TEAL, edgecolors="none")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color=GOLD, lw=2.2, label="理想预测线")
    metric_box(
        ax,
        [
            f"R²：{r2_score(y_true, y_pred):.4f}",
            f"均方根误差：{mean_squared_error(y_true, y_pred) ** 0.5:.4f}",
            f"平均绝对误差：{mean_absolute_error(y_true, y_pred):.4f}",
        ],
    )
    ax.set_title("任务二锁定验证集真实值与预测值", loc="left")
    ax.set_xlabel("真实生产力评分")
    ax.set_ylabel("预测生产力评分")
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax, "both")
    save_svg(fig, out / "图08a_任务二锁定验证预测.svg")

    residual = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), constrained_layout=True)
    axes[0].scatter(y_pred[shown_idx], residual[shown_idx], s=14, alpha=0.36, color=TEAL, edgecolors="none")
    axes[0].axhline(0, color=GOLD, lw=2)
    axes[0].set_title("残差与预测值")
    axes[0].set_xlabel("预测生产力评分")
    axes[0].set_ylabel("残差（真实值－预测值）")
    clean_axis(axes[0], "both")
    sns.histplot(residual, bins=34, kde=True, color=BLUE, ax=axes[1])
    axes[1].axvline(0, color=GOLD, lw=2, label="零残差")
    axes[1].set_title("残差分布")
    axes[1].set_xlabel("残差")
    axes[1].set_ylabel("样本数")
    axes[1].legend(frameon=False)
    clean_axis(axes[1], "y")
    fig.suptitle("任务二锁定验证残差诊断", x=0.01, ha="left", fontsize=16, fontweight="bold")
    save_svg(fig, out / "图08b_任务二锁定验证残差诊断.svg")

    importance_bar(
        prod_imp,
        out / "图09a_生产力模型变量重要性.svg",
        "生产力模型变量重要性",
        "Importance_Mean",
        "Importance_STD_Folds",
        BLUE,
        15,
    )
    importance_bar(
        energy_imp,
        out / "图09b_精力辅助模型变量重要性.svg",
        "精力辅助模型变量重要性",
        "Importance_Mean",
        "Importance_STD_Folds",
        TEAL,
        15,
    )

    selected = folds[folds["Selected"]].sort_values("Fold")
    values = selected["Adjusted_R2_Transformed"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    ax.plot(selected["Fold"], values, color=NAVY, lw=2.2, marker="o", markersize=7)
    ax.axhline(values.mean(), color=GOLD, lw=1.8, ls="--", label=f"五折均值：{values.mean():.4f}")
    for x, yv in zip(selected["Fold"], values):
        ax.text(x, yv + 0.004, f"{yv:.4f}", ha="center")
    ax.set_xticks(selected["Fold"], [f"第{x}折" for x in selected["Fold"]])
    ax.set_ylim(values.min() - 0.025, values.max() + 0.025)
    ax.set_xlabel("验证折")
    ax.set_ylabel("调整后决定系数 R²")
    ax.set_title("任务二入选模型五折稳定性", loc="left")
    ax.legend(frameon=False)
    clean_axis(ax, "y")
    save_svg(fig, out / "图10_任务二五折验证稳定性.svg")


def draw_task3(root: Path, out: Path) -> None:
    src = root / "03_task3" / "plotting" / "source_data"
    df = pd.read_csv(src / "A_dataset.csv")
    val = pd.read_csv(src / "validation_predictions.csv")
    comp = pd.read_csv(src / "model_comparison.csv")
    folds = pd.read_csv(src / "fold_metrics.csv")
    imp = pd.read_csv(src / "permutation_importance.csv")
    ablation = pd.read_csv(src / "ablation_results.csv")

    target = df["Health_Score"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), constrained_layout=True)
    sns.histplot(target, bins=34, kde=True, color=BLUE, edgecolor="white", linewidth=0.5, ax=axes[0])
    axes[0].axvline(target.mean(), color=GOLD, lw=2.2, label=f"均值：{target.mean():.2f} 分")
    axes[0].set_title("综合健康评分分布")
    axes[0].set_xlabel("综合健康评分（分）")
    axes[0].set_ylabel("样本数")
    axes[0].legend(frameon=False)
    clean_axis(axes[0], "y")

    order = ["Poor", "Average", "Good", "Excellent"]
    cn = ["较差", "一般", "良好", "优秀"]
    grouped = df.groupby("Wellness_Category")["Health_Score"].agg(["min", "max", "mean"]).reindex(order)
    bars = axes[1].bar(cn, grouped["mean"], color=[MID, "#7F9AB5", BLUE, NAVY], width=0.68)
    for bar, (_, row) in zip(bars, grouped.iterrows()):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            row["mean"] + 2.1,
            f"{row['min']:.1f}～{row['max']:.1f}",
            ha="center",
            fontsize=10.5,
        )
    axes[1].set_title("目标分档字段的确定性边界")
    axes[1].set_xlabel("健康等级")
    axes[1].set_ylabel("组内综合健康评分均值")
    axes[1].set_ylim(0, 108)
    clean_axis(axes[1], "y")
    fig.suptitle("任务三目标分布与分档泄露", x=0.01, ha="left", fontsize=16, fontweight="bold")
    save_svg(fig, out / "图11a_任务三目标分布与分档泄露.svg")

    numeric = df.select_dtypes(include=np.number)
    top = numeric.corrwith(df["Health_Score"]).abs().sort_values(ascending=False).head(13).index.tolist()
    corr = numeric[top].corr()
    corr.index = [LABELS.get(x, x) for x in corr.index]
    corr.columns = [LABELS.get(x, x) for x in corr.columns]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10.5, 8.7), constrained_layout=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=CMAP_DIVERGING,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.55,
        linecolor="white",
        cbar=False,
        ax=ax,
    )
    ax.set_title("综合健康评分及高相关连续变量结构", loc="left", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=43)
    ax.tick_params(axis="y", rotation=0)
    ax.legend(
        handles=[Patch(facecolor=NAVY, label="负相关"), Patch(facecolor="#F7F8F8", edgecolor=GRID, label="接近零"), Patch(facecolor=CORAL, label="正相关")],
        frameon=False,
        ncol=3,
        loc="upper right",
    )
    save_svg(fig, out / "图11b_任务三高相关连续变量结构.svg")

    data = comp.sort_values("R2")
    fig, ax = plt.subplots(figsize=(9.8, 5.6), constrained_layout=True)
    colors = [GOLD if name == "OOF加权残差集成" else BLUE for name in data["Model"]]
    bars = ax.barh([TASK3_MODEL_NAMES[x] for x in data["Model"]], data["R2"], color=colors, height=0.64)
    ax.set_xlim(data["R2"].min() - 0.003, data["R2"].max() + 0.0015)
    ax.set_xlabel("五折折外决定系数 R²")
    ax.set_title("任务三候选残差模型与融合性能", loc="left")
    clean_axis(ax, "x")
    for bar, value in zip(bars, data["R2"]):
        ax.text(value + 0.00018, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center")
    save_svg(fig, out / "图12_任务三候选残差模型比较.svg")

    y_true = val["true_value"].to_numpy()
    y_pred = val["predicted_value"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.8, 6.3), constrained_layout=True)
    shown_idx = np.linspace(0, len(y_true) - 1, min(len(y_true), 1400), dtype=int)
    ax.scatter(y_true[shown_idx], y_pred[shown_idx], s=11, alpha=0.24, color=BLUE, edgecolors="none")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color=GOLD, lw=2.2, label="理想预测线")
    metric_box(
        ax,
        [
            f"R²：{r2_score(y_true, y_pred):.4f}",
            f"均方根误差：{mean_squared_error(y_true, y_pred) ** 0.5:.4f}",
            f"平均绝对误差：{mean_absolute_error(y_true, y_pred):.4f}",
        ],
    )
    ax.set_title("任务三锁定验证集真实值与预测值", loc="left")
    ax.set_xlabel("真实综合健康评分")
    ax.set_ylabel("预测综合健康评分")
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax, "both")
    save_svg(fig, out / "图13a_任务三锁定验证预测.svg")

    residual = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), constrained_layout=True)
    axes[0].scatter(y_pred[shown_idx], residual[shown_idx], s=14, alpha=0.35, color=BLUE, edgecolors="none")
    axes[0].axhline(0, color=GOLD, lw=2)
    axes[0].set_title("残差与预测值")
    axes[0].set_xlabel("预测综合健康评分")
    axes[0].set_ylabel("残差（真实值－预测值）")
    clean_axis(axes[0], "both")
    sns.histplot(residual, bins=34, kde=True, color=TEAL, ax=axes[1])
    axes[1].axvline(0, color=GOLD, lw=2, label="零残差")
    axes[1].set_title("残差分布")
    axes[1].set_xlabel("残差")
    axes[1].set_ylabel("样本数")
    axes[1].legend(frameon=False)
    clean_axis(axes[1], "y")
    fig.suptitle("任务三锁定验证残差诊断", x=0.01, ha="left", fontsize=16, fontweight="bold")
    save_svg(fig, out / "图13b_任务三锁定验证残差诊断.svg")

    leader = imp.iloc[0]
    fig, ax = plt.subplots(figsize=(8.6, 3.6), constrained_layout=True)
    ax.barh([LABELS[leader["Feature"]]], [leader["Importance_Mean"]], xerr=[leader["Importance_STD"]], color=NAVY, height=0.38, capsize=4)
    ax.text(leader["Importance_Mean"] + 0.025, 0, f"{leader['Importance_Mean']:.4f}", va="center", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("置换后 R² 平均下降量")
    ax.set_title("任务三首要代理指标重要性", loc="left")
    clean_axis(ax, "x")
    save_svg(fig, out / "图14a_任务三首要代理指标重要性.svg")

    rest = imp.iloc[1:15].sort_values("Importance_Mean")
    fig, ax = plt.subplots(figsize=(9.8, 7.0), constrained_layout=True)
    ypos = np.arange(len(rest))
    bars = ax.barh(ypos, rest["Importance_Mean"], color=BLUE, height=0.65)
    ax.errorbar(rest["Importance_Mean"], ypos, xerr=rest["Importance_STD"], fmt="none", ecolor=INK, capsize=3)
    ax.set_yticks(ypos, [LABELS.get(x, x) for x in rest["Feature"]])
    ax.set_xlabel("置换后 R² 平均下降量")
    ax.set_title("任务三其他关键驱动因素重要性", loc="left")
    clean_axis(ax, "x")
    for bar, value in zip(bars, rest["Importance_Mean"]):
        ax.text(value + 0.0012, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center")
    save_svg(fig, out / "图14b_任务三其他关键驱动因素重要性.svg")

    scopes = ablation.sort_values("R2")
    fig, ax = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)
    bars = ax.barh(scopes["Scope"].str.replace("Healthy Aging Score", "健康老龄化评分", regex=False), scopes["R2"], color=[CORAL, BLUE, TEAL], height=0.60)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("锁定验证决定系数 R²")
    ax.set_title("任务三特征口径消融", loc="left")
    clean_axis(ax, "x")
    for bar, value in zip(bars, scopes["R2"]):
        ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center")
    save_svg(fig, out / "图15a_任务三特征口径消融.svg")

    values = folds.sort_values("Fold")
    mean_r2 = values["R2"].mean()
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    ax.plot(values["Fold"], values["R2"], color=NAVY, lw=2.2, marker="o", markersize=7)
    ax.axhline(mean_r2, color=GOLD, lw=1.8, ls="--", label=f"五折均值：{mean_r2:.4f}")
    for x, yv in zip(values["Fold"], values["R2"]):
        ax.text(x, yv + 0.00018, f"{yv:.4f}", ha="center")
    ax.set_xticks(values["Fold"], [f"第{x}折" for x in values["Fold"]])
    ax.set_ylim(values["R2"].min() - 0.0012, values["R2"].max() + 0.0012)
    ax.set_xlabel("验证折")
    ax.set_ylabel("决定系数 R²")
    ax.set_title("任务三五折验证稳定性", loc="left")
    ax.legend(frameon=False)
    clean_axis(ax, "y")
    save_svg(fig, out / "图15b_任务三五折验证稳定性.svg")


def verify_outputs(out: Path) -> None:
    svgs = sorted(out.glob("*.svg"))
    if len(svgs) != 25:
        raise RuntimeError(f"应生成 25 个 SVG，实际生成 {len(svgs)} 个。")
    for path in svgs:
        raw = path.read_text(encoding="utf-8")
        if "<svg" not in raw or "<image" in raw:
            raise RuntimeError(f"SVG 校验失败：{path.name}")
        if path.stat().st_size < 2_000:
            raise RuntimeError(f"SVG 文件异常偏小：{path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_style()
    args.output.mkdir(parents=True, exist_ok=True)
    draw_flowcharts(args.output)
    draw_task1(args.model_root, args.output)
    draw_task2(args.model_root, args.output)
    draw_task3(args.model_root, args.output)
    verify_outputs(args.output)
    print(f"已生成 {len(list(args.output.glob('*.svg')))} 个纯矢量 SVG：{args.output}")


if __name__ == "__main__":
    main()
