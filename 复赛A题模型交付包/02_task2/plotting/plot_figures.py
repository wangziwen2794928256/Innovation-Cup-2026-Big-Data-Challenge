from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
TARGET_COLUMN = "Productivity_Score"

def configure_plots():
    from matplotlib import font_manager
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            family = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = family
            break
    plt.rcParams.update({
        "figure.dpi":150, "savefig.dpi":300, "savefig.facecolor":"white",
        "axes.unicode_minus":False, "axes.spines.top":False, "axes.spines.right":False,
        "axes.edgecolor":"#3B4654", "text.color":"#1D2733", "axes.labelcolor":"#1D2733",
        "xtick.color":"#4D5967", "ytick.color":"#4D5967", "grid.color":"#D8DEE6",
        "grid.alpha":0.65,
    })

def adjusted_r2(r2, n, p):
    return 1 - (1-r2)*(n-1)/(n-p-1)

def main():
    configure_plots()
    df = pd.read_csv(SRC/"A_dataset.csv")
    comparison = pd.read_csv(SRC/"model_comparison.csv")
    val = pd.read_csv(SRC/"validation_predictions.csv")
    importance = pd.read_csv(SRC/"permutation_importance.csv")
    energy_importance = pd.read_csv(SRC/"energy_driver_importance.csv")
    fold_metrics = pd.read_csv(SRC/"fold_metrics.csv")
    ablation = pd.read_csv(SRC/"scope_ablation.csv")
    y_holdout = val["true_value"]
    holdout_pred = val["predicted_value"].to_numpy()
    r2 = r2_score(y_holdout, holdout_pred)
    p_transformed = int(comparison.loc[comparison["Selected"], "P_Transformed"].iloc[0])
    metric = {
        "R2": r2,
        "Adjusted_R2_Transformed": adjusted_r2(r2, len(y_holdout), p_transformed),
        "RMSE": mean_squared_error(y_holdout, holdout_pred) ** 0.5,
    }

    navy, blue, gold, gray, pale = "#17365D", "#4B78A6", "#C59A3D", "#8A96A3", "#DDE7F0"

    fig, ax = plt.subplots(figsize=(8.6,4.8), constrained_layout=True)
    sns.histplot(df[TARGET_COLUMN], bins=30, kde=True, color=blue, ax=ax)
    ax.axvline(df[TARGET_COLUMN].mean(), color=gold, lw=2, label=f"均值 {df[TARGET_COLUMN].mean():.2f}")
    ax.set(title="生产力评分分布", xlabel="Productivity Score（分）", ylabel="样本数"); ax.legend(frameon=False)
    fig.savefig(OUT/"Fig01_productivity_distribution.png", bbox_inches="tight"); plt.close(fig)

    focus=[TARGET_COLUMN,"Energy_Level_Score","Fatigue_Level_Score","Mood_Score","Sleep_Quality_Score",
           "Anxiety_Score","Depression_Risk_Score","Life_Satisfaction_Score","Stress_Level",
           "Exercise_Frequency_Per_Week","Daily_Steps","Working_Hours_Per_Day","Sitting_Hours_Per_Day"]
    corr=df[focus].corr()
    fig,ax=plt.subplots(figsize=(10.0,8.2), constrained_layout=True)
    sns.heatmap(corr,cmap="vlag",vmin=-1,vmax=1,center=0,square=True,linewidths=.45,cbar_kws={"shrink":.78},ax=ax)
    ax.set_title("生产力及主要连续指标的 Pearson 相关矩阵",pad=12); ax.tick_params(axis="x",rotation=48,labelsize=8); ax.tick_params(axis="y",rotation=0,labelsize=8)
    fig.savefig(OUT/"Fig02_multicollinearity_heatmap.png",bbox_inches="tight"); plt.close(fig)

    comp=comparison.sort_values("Adjusted_R2_Transformed")
    colors=[navy if value else gray for value in comp["Selected"]]
    fig,ax=plt.subplots(figsize=(9.2,5.2),constrained_layout=True)
    bars=ax.barh(comp["Model"],comp["Adjusted_R2_Transformed"],color=colors)
    ax.set_xlim(0,min(.72,comp["Adjusted_R2_Transformed"].max()+.06)); ax.set(title="候选模型五折 OOF 调整 R²",xlabel="调整 R²（按变换后维数）",ylabel=""); ax.grid(axis="x")
    for bar,value in zip(bars,comp["Adjusted_R2_Transformed"]): ax.text(value+.006,bar.get_y()+bar.get_height()/2,f"{value:.4f}",va="center",fontsize=9)
    fig.savefig(OUT/"Fig03_model_comparison.png",bbox_inches="tight"); plt.close(fig)

    fig,ax=plt.subplots(figsize=(6.5,5.8),constrained_layout=True)
    ax.scatter(y_holdout,holdout_pred,s=16,alpha=.38,color=blue,edgecolors="none")
    low=min(float(y_holdout.min()),float(holdout_pred.min())); high=max(float(y_holdout.max()),float(holdout_pred.max()))
    ax.plot([low,high],[low,high],color=gold,lw=2,label="理想预测线")
    ax.text(.035,.955,f"R² = {metric['R2']:.4f}\n调整 R² = {metric['Adjusted_R2_Transformed']:.4f}\nRMSE = {metric['RMSE']:.4f}",
            transform=ax.transAxes,va="top",bbox={"boxstyle":"round,pad=0.45","fc":"white","ec":"#CDD4DD"})
    ax.set(title="锁定验证集真实值与预测值",xlabel="真实 Productivity Score",ylabel="预测 Productivity Score"); ax.legend(frameon=False,loc="lower right"); ax.grid(True)
    fig.savefig(OUT/"Fig04_true_vs_predicted.png",bbox_inches="tight"); plt.close(fig)

    residual=y_holdout.to_numpy()-holdout_pred
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.3),constrained_layout=True)
    axes[0].scatter(holdout_pred,residual,s=15,alpha=.38,color=blue,edgecolors="none"); axes[0].axhline(0,color=gold,lw=2)
    axes[0].set(title="残差与预测值",xlabel="预测 Productivity Score",ylabel="残差（真实值-预测值）"); axes[0].grid(True)
    sns.histplot(residual,bins=30,kde=True,color=blue,ax=axes[1]); axes[1].axvline(0,color=gold,lw=2); axes[1].set(title="残差分布",xlabel="残差",ylabel="样本数")
    fig.savefig(OUT/"Fig05_residual_diagnostics.png",bbox_inches="tight"); plt.close(fig)

    top=importance.head(15).sort_values("Importance_Mean")
    fig,ax=plt.subplots(figsize=(9.5,6.2),constrained_layout=True)
    ax.barh(top["Feature"],top["Importance_Mean"],xerr=top["Importance_STD_Folds"].fillna(0),color=blue,ecolor=gray,capsize=2)
    ax.axvline(0,color="#3B4654",lw=1); ax.set(title="生产力模型的分组置换重要性",xlabel="验证 R² 平均下降量",ylabel=""); ax.grid(axis="x")
    fig.savefig(OUT/"Fig06_productivity_permutation_importance.png",bbox_inches="tight"); plt.close(fig)

    energy_top=energy_importance.head(15).sort_values("Importance_Mean")
    fig,ax=plt.subplots(figsize=(9.5,6.2),constrained_layout=True)
    ax.barh(energy_top["Feature"],energy_top["Importance_Mean"],xerr=energy_top["Importance_STD_Folds"].fillna(0),color=navy,ecolor=gray,capsize=2)
    ax.axvline(0,color="#3B4654",lw=1); ax.set(title="精力评分辅助模型的生活方式变量重要性",xlabel="验证 R² 平均下降量",ylabel=""); ax.grid(axis="x")
    fig.savefig(OUT/"Fig07_energy_driver_importance.png",bbox_inches="tight"); plt.close(fig)

    selected_folds=fold_metrics[fold_metrics["Selected"]].sort_values("Fold")
    fig,ax=plt.subplots(figsize=(7.8,4.5),constrained_layout=True)
    bars=ax.bar(selected_folds["Fold"].astype(str),selected_folds["Adjusted_R2_Transformed"],color=blue)
    mean_value=selected_folds["Adjusted_R2_Transformed"].mean(); ax.axhline(mean_value,color=gold,ls="--",lw=1.8,label=f"折均值 {mean_value:.4f}")
    for bar,value in zip(bars,selected_folds["Adjusted_R2_Transformed"]): ax.text(bar.get_x()+bar.get_width()/2,value+.006,f"{value:.4f}",ha="center",fontsize=8)
    ax.set_ylim(0,min(.75,selected_folds["Adjusted_R2_Transformed"].max()+.08)); ax.set(title="入选模型五折验证稳定性",xlabel="折次",ylabel="调整 R²"); ax.legend(frameon=False); ax.grid(axis="y")
    fig.savefig(OUT/"Fig08_fold_stability.png",bbox_inches="tight"); plt.close(fig)

    scope=ablation.sort_values("Adjusted_R2_Transformed")
    fig,ax=plt.subplots(figsize=(8.9,4.6),constrained_layout=True)
    colors=[pale,blue,navy]; bars=ax.barh(scope["Scope"],scope["Adjusted_R2_Transformed"],color=colors[:len(scope)])
    ax.set_xlim(0,min(.75,scope["Adjusted_R2_Transformed"].max()+.07)); ax.set(title="特征口径敏感性分析",xlabel="开发集 OOF 调整 R²",ylabel=""); ax.grid(axis="x")
    for bar,value in zip(bars,scope["Adjusted_R2_Transformed"]): ax.text(value+.006,bar.get_y()+bar.get_height()/2,f"{value:.4f}",va="center",fontsize=9)
    fig.savefig(OUT/"Fig09_scope_ablation.png",bbox_inches="tight"); plt.close(fig)
    print("Task 2 figures generated:", len(list(OUT.glob("*.png"))))

if __name__=="__main__":
    main()
