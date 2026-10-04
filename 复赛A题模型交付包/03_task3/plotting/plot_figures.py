from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT=Path(__file__).resolve().parent
SRC=ROOT/"source_data"
OUT=ROOT/"figures"
OUT.mkdir(exist_ok=True)
TARGET_COLUMN="Health_Score"

def metrics(y_true,y_pred):
    return {"R2":r2_score(y_true,y_pred),"RMSE":mean_squared_error(y_true,y_pred)**0.5,"MAE":mean_absolute_error(y_true,y_pred)}

def configure_plots():
    from matplotlib import font_manager
    candidates=[
        Path("C:/Windows/Fonts/msyh.ttc"),Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")]
    for font in candidates:
        if font.exists():
            font_manager.fontManager.addfont(str(font)); plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(font)).get_name(); break
    plt.rcParams.update({"figure.dpi":150,"savefig.dpi":300,"axes.unicode_minus":False,"axes.spines.top":False,
        "axes.spines.right":False,"axes.titleweight":"bold","axes.edgecolor":"#333333","text.color":"#222222",
        "axes.labelcolor":"#222222","xtick.color":"#444444","ytick.color":"#444444","grid.color":"#D9DEE7","grid.alpha":.55})

def main():
    configure_plots()
    df=pd.read_csv(SRC/"A_dataset.csv")
    val=pd.read_csv(SRC/"validation_predictions.csv")
    comparison=pd.read_csv(SRC/"model_comparison.csv")
    folds=pd.read_csv(SRC/"fold_metrics.csv")
    importance=pd.read_csv(SRC/"permutation_importance.csv")
    ablation=pd.read_csv(SRC/"ablation_results.csv")
    y_holdout=val["true_value"]; holdout_pred=val["predicted_value"].to_numpy()
    navy,blue,gold,gray="#17365D","#3F6F9F","#C59A3D","#8A96A3"

    fig,axes=plt.subplots(1,2,figsize=(10.8,4.2),constrained_layout=True)
    sns.histplot(df[TARGET_COLUMN],bins=32,kde=True,color=blue,ax=axes[0])
    axes[0].axvline(df[TARGET_COLUMN].mean(),color=gold,lw=2,label=f"均值 {df[TARGET_COLUMN].mean():.2f}")
    axes[0].set(title="Health Score 分布",xlabel="综合健康评分",ylabel="样本数"); axes[0].legend(frameon=False)
    order=["Poor","Average","Good","Excellent"]; tmp=df.groupby("Wellness_Category")[TARGET_COLUMN].agg(["min","max","mean","count"]).reindex(order)
    axes[1].bar(tmp.index,tmp["mean"],color=[gray,"#6F8195",blue,navy])
    for i,(_,row) in enumerate(tmp.iterrows()): axes[1].text(i,row["mean"]+1.3,f"{row['min']:.1f}-{row['max']:.1f}",ha="center",fontsize=8)
    axes[1].set(title="目标分档字段的泄露证据",xlabel="Wellness Category",ylabel="组内 Health Score 均值",ylim=(0,105))
    fig.savefig(OUT/"Fig01_target_and_leakage.png",bbox_inches="tight"); plt.close(fig)

    numeric=df.select_dtypes(include=np.number); top=numeric.corrwith(df[TARGET_COLUMN]).abs().sort_values(ascending=False).head(13).index; corr=numeric[top].corr()
    fig,ax=plt.subplots(figsize=(9.2,7.4),constrained_layout=True)
    sns.heatmap(corr,cmap="vlag",center=0,vmin=-1,vmax=1,square=True,linewidths=.45,cbar_kws={"shrink":.78},ax=ax)
    ax.set_title("高相关连续变量 Pearson 相关矩阵",pad=12); ax.tick_params(axis="x",rotation=48,labelsize=8); ax.tick_params(axis="y",rotation=0,labelsize=8)
    fig.savefig(OUT/"Fig02_correlation_heatmap.png",bbox_inches="tight"); plt.close(fig)

    comp=comparison.sort_values("R2")
    fig,ax=plt.subplots(figsize=(9.4,4.8),constrained_layout=True)
    bars=ax.barh(comp["Model"],comp["R2"],color=[gray]*(len(comp)-1)+[navy]); ax.set_xlim(max(.90,comp["R2"].min()-.006),min(1.0,comp["R2"].max()+.004))
    ax.set_xlabel("五折 OOF R²"); ax.set_title("候选模型与残差集成性能比较"); ax.grid(axis="x")
    for bar,value in zip(bars,comp["R2"]): ax.text(value+.00025,bar.get_y()+bar.get_height()/2,f"{value:.4f}",va="center",fontsize=9)
    fig.savefig(OUT/"Fig03_model_comparison.png",bbox_inches="tight"); plt.close(fig)

    fig,ax=plt.subplots(figsize=(6.6,5.8),constrained_layout=True)
    ax.scatter(y_holdout,holdout_pred,s=16,alpha=.42,color=blue,edgecolors="none")
    lo=min(y_holdout.min(),holdout_pred.min()); hi=max(y_holdout.max(),holdout_pred.max()); ax.plot([lo,hi],[lo,hi],color=gold,lw=2,label="理想预测线")
    m=metrics(y_holdout,holdout_pred); ax.text(.035,.955,f"R² = {m['R2']:.4f}\nRMSE = {m['RMSE']:.4f}\nMAE = {m['MAE']:.4f}",transform=ax.transAxes,va="top",bbox={"boxstyle":"round,pad=0.45","fc":"white","ec":"#CDD4DD"})
    ax.set(title="锁定验证集：真实值与预测值",xlabel="真实 Health Score",ylabel="预测 Health Score"); ax.legend(frameon=False,loc="lower right"); ax.grid(True)
    fig.savefig(OUT/"Fig04_true_vs_predicted.png",bbox_inches="tight"); plt.close(fig)

    residual=y_holdout.to_numpy()-holdout_pred
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.2),constrained_layout=True)
    axes[0].scatter(holdout_pred,residual,s=15,alpha=.40,color=blue,edgecolors="none"); axes[0].axhline(0,color=gold,lw=2)
    axes[0].set(title="残差-预测值图",xlabel="预测 Health Score",ylabel="残差（真实-预测）"); axes[0].grid(True)
    sns.histplot(residual,bins=32,kde=True,color=blue,ax=axes[1]); axes[1].axvline(0,color=gold,lw=2); axes[1].set(title="残差分布",xlabel="残差",ylabel="样本数")
    fig.savefig(OUT/"Fig05_residual_diagnostics.png",bbox_inches="tight"); plt.close(fig)

    imp=importance.head(15); leader=imp.iloc[[0]]; rest=imp.iloc[1:].sort_values("Importance_Mean")
    fig,axes=plt.subplots(1,2,figsize=(10.8,5.8),gridspec_kw={"width_ratios":[.9,2.5]},constrained_layout=True)
    axes[0].barh(leader["Feature"],leader["Importance_Mean"],xerr=leader["Importance_STD"],color=navy,ecolor=gray,capsize=2); axes[0].set(title="首要代理指标",xlabel="R² 平均下降量",ylabel=""); axes[0].grid(axis="x")
    axes[1].barh(rest["Feature"],rest["Importance_Mean"],xerr=rest["Importance_STD"],color=blue,ecolor=gray,capsize=2); axes[1].set(title="其余关键驱动因素（放大）",xlabel="R² 平均下降量",ylabel=""); axes[1].grid(axis="x")
    fig.suptitle("锁定验证集分组置换重要性",fontweight="bold"); fig.savefig(OUT/"Fig06_permutation_importance.png",bbox_inches="tight"); plt.close(fig)

    abl=ablation.sort_values("R2")
    fig,ax=plt.subplots(figsize=(8.8,4.5),constrained_layout=True)
    bars=ax.barh(abl["Scope"],abl["R2"],color=[gray]*(len(abl)-1)+[navy]); ax.set_xlim(max(0.0,abl["R2"].min()-.025),min(1.0,abl["R2"].max()+.008))
    ax.set(title="特征口径消融实验",xlabel="锁定验证集 R²"); ax.grid(axis="x")
    for b,value in zip(bars,abl["R2"]): ax.text(value+.001,b.get_y()+b.get_height()/2,f"{value:.4f}",va="center",fontsize=9)
    fig.savefig(OUT/"Fig07_ablation.png",bbox_inches="tight"); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.8,4.5),constrained_layout=True)
    ax.errorbar(folds["Fold"],folds["R2"],marker="o",ms=7,lw=2,color=navy); ax.axhline(folds["R2"].mean(),color=gold,ls="--",lw=1.8,label=f"均值 {folds['R2'].mean():.4f}")
    for x,value in zip(folds["Fold"],folds["R2"]): ax.text(x,value+.0008,f"{value:.4f}",ha="center",fontsize=8)
    ax.set_xticks(folds["Fold"]); ax.set(title="五折 OOF 稳定性",xlabel="折次",ylabel="R²",ylim=(folds["R2"].min()-.004,folds["R2"].max()+.004)); ax.legend(frameon=False); ax.grid(axis="y")
    fig.savefig(OUT/"Fig08_fold_stability.png",bbox_inches="tight"); plt.close(fig)
    print("Task 3 figures generated:",len(list(OUT.glob("*.png"))))

if __name__=="__main__":
    main()
