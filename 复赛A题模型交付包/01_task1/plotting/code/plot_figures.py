"""从固定 CSV 图源生成任务一科研图。

蓝绿配色沿用初赛 plot_config.py。中文字体可通过 --font 指定；
Windows 通常可使用 C:/Windows/Fonts/msyh.ttc。
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import warnings
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.text import Text
from matplotlib.ticker import NullLocator
from PIL import Image
from pypdf import PdfReader, PdfWriter

HERE=Path(__file__).resolve().parents[1]
BLUE="#376795"; TEAL="#3B9B8F"; INK="#263238"; GREY="#AAB3BA"; LIGHT="#E8ECEF"
LABELS={
    "Exercise_Frequency_Per_Week":"每周运动频次", "Stress_Level":"压力水平", "Mood_Score":"情绪评分",
    "Fatigue_Level_Score":"疲劳水平评分", "Anxiety_Score":"焦虑评分", "Productivity_Score":"生产力评分",
    "Focus_Concentration_Score":"专注力评分", "Meditation_Practice":"冥想习惯", "Depression_Risk_Score":"抑郁风险评分",
    "Energy_Level_Score":"精力评分", "Social_Interaction_Score":"社交互动评分", "Weight_kg":"体重",
    "Sleep_Quality_Score":"睡眠质量评分",
}
NAMES={"ridge_alpha1":"Ridge α=1（选定）", "ridge_001":"Ridge α=0.01",
       "hgb_15":"HGB（15叶）", "ridge_1000":"Ridge α=1000", "rf_128":"RF（128叶）", "extra_128":"ExtraTrees（128叶）"}


def read(name):
    return pd.read_csv(HERE/"source_data"/name,keep_default_na=False)


def atomic_bytes(path,data):
    fd,temporary=tempfile.mkstemp(prefix=".pending_",dir=path.parent)
    with os.fdopen(fd,"wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    assert Path(temporary).read_bytes()==data
    os.replace(temporary,path)
    assert path.read_bytes()==data


def title(fig,heading,sub):
    fig.text(.055,.956,heading,fontsize=19,color=INK,ha="left",va="top")
    fig.text(.055,.902,sub,fontsize=10.7,color="#61717B",ha="left",va="top")


def foot(fig,text):
    fig.text(.055,.022,text,fontsize=9.2,color="#61717B",ha="left",va="bottom")


def quiet(ax,grid="x"):
    ax.spines[["top","right"]].set_visible(False)
    if grid: ax.grid(axis=grid,color=LIGHT,lw=.8,zorder=0)
    ax.set_axisbelow(True)


def setup(font):
    if not font:
        for name in ["Microsoft YaHei","Noto Sans CJK SC","SimHei","SimSun"]:
            try:
                font=Path(fm.findfont(fm.FontProperties(family=name),fallback_to_default=False)); break
            except ValueError: pass
    if not font or not font.exists():
        raise ValueError("缺少中文字体，请传入--font，例如C:/Windows/Fonts/msyh.ttc")
    fm.fontManager.addfont(str(font))
    family=fm.FontProperties(fname=str(font)).get_name()
    matplotlib.rcParams.update({"font.family":["DejaVu Sans",family],"font.size":12,"axes.labelsize":12,
        "axes.titlesize":13,"xtick.labelsize":10.5,"ytick.labelsize":11,"legend.fontsize":10,
        "axes.edgecolor":"#65747C","axes.linewidth":.8,"axes.unicode_minus":False,"mathtext.fontset":"stix",
        "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white","legend.frameon":False,
        "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"path","svg.hashsalt":"DATA2600655_TASK1"})
    return {"font_basename":font.name,"font_family":family,"font_sha256":hashlib.sha256(font.read_bytes()).hexdigest(),
            "note":"Font is not redistributed as a standalone font; PDF embeds subsets, SVG outlines glyphs. Regeneration requires a CJK-capable local font."}


def export(fig,stem,records,checks):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    outside=[]
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text(): continue
        box=text.get_window_extent(renderer)
        if box.x0 < -2 or box.y0 < -2 or box.x1 > fig.bbox.width+2 or box.y1 > fig.bbox.height+2:
            outside.append(text.get_text())
    if outside:
        raise ValueError(f"文本超出画布 {stem}: {outside}")
    paths={}
    for fmt,folder in [("png","png_600dpi"),("pdf","pdf"),("svg","svg")]:
        destination=HERE/"figures"/folder; destination.mkdir(parents=True,exist_ok=True)
        p=destination/(stem+"."+fmt)
        meta={"Software":"DATA2600655 task1 figures"} if fmt=="png" else (
             {"Creator":"DATA2600655 task1 figures","CreationDate":datetime(2026,8,28,tzinfo=timezone.utc),"ModDate":datetime(2026,8,28,tzinfo=timezone.utc)} if fmt=="pdf" else
             {"Creator":"DATA2600655 task1 figures","Date":"2026-08-28"})
        buffer=io.BytesIO()
        fig.savefig(buffer,format=fmt,dpi=600,bbox_inches="tight",pad_inches=.16,metadata=meta)
        data=buffer.getvalue()
        if fmt=="svg": ET.fromstring(data)
        elif fmt=="pdf": assert len(PdfReader(io.BytesIO(data)).pages)==1
        else:
            with Image.open(io.BytesIO(data)) as test_image:test_image.verify()
        atomic_bytes(p,data)
        paths[fmt]=p.relative_to(HERE).as_posix()
    im=Image.open(HERE/paths["png"])
    assert all(abs(v-600)<.1 for v in im.info["dpi"])
    records.append({"figure_id":stem,"files":paths,"png_pixels":list(im.size),"png_dpi":list(im.info["dpi"]),"text_bounds":"PASS",**checks})
    im.close()
    plt.close(fig)
    print("EXPORTED",stem,flush=True)


def workflow(records):
    fig=plt.figure(figsize=(11.2,6.6)); ax=fig.add_axes([.045,.09,.91,.77]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    title(fig,"特征边界与验证流程","任务一：非睡眠类输入预测睡眠质量；模型与参数由开发集五折结果确定")
    def box(x,y,w,h,text,color=BLUE,fill="#F4F8FB",size=12):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.008,rounding_size=0.018",ec=color,fc=fill,lw=1.1))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=size,color=INK,linespacing=1.6)
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=13,color="#73858F",lw=1.1))
    ax.text(.02,.95,"A  输入契约",fontsize=15,color=BLUE)
    box(.025,.71,.39,.17,"原始数据：10,000人 × 64列")
    box(.025,.42,.39,.18,"模型输入：48列\n34个数值字段 + 14个类别字段",TEAL,"#EFF8F6")
    arrow((.22,.70),(.22,.61))
    box(.025,.075,.39,.25,"移出模型输入：16列\n睡眠字段8 + 边界字段2 + 综合字段4\nPerson_ID 1列 + 目标1列",GREY,"#FAFBFC",11.5)
    ax.text(.455,.95,"B  固定划分与模型评价",fontsize=15,color=BLUE)
    box(.465,.73,.51,.13,"按Person_ID哈希划分；不按目标划分")
    box(.465,.51,.245,.13,"开发集：8,000人",BLUE)
    box(.75,.51,.225,.13,"锁定验证：2,000人",TEAL,"#EFF8F6")
    arrow((.61,.72),(.585,.65)); arrow((.83,.72),(.86,.65))
    box(.465,.285,.245,.14,"固定五折模型比较\n仅开发集选择参数",BLUE,size=11.8)
    arrow((.585,.50),(.585,.435))
    box(.465,.055,.245,.14,"选定Ridge，完整开发集拟合\n48个原始字段 → 107编码列",BLUE,size=10.5)
    arrow((.585,.275),(.585,.205))
    box(.75,.055,.225,.14,"方案确定后独立评价\n锁定验证结果已保存",TEAL,"#EFF8F6",11)
    arrow((.72,.125),(.74,.125)); arrow((.862,.50),(.862,.205))
    foot(fig,"固定字段与固定划分；预处理只在相应训练折拟合；锁定验证结果不用于调参。")
    export(fig,"T1_F01_feature_and_validation_workflow",records,{"source_csv":"feature_contract_counts.csv","input_count":48,"excluded_from_input":16})


def comparison(records):
    df=read("model_comparison.csv").sort_values("fold_r2_mean",ascending=False); folds=read("model_fold_scores.csv")
    fig,ax=plt.subplots(figsize=(10.2,6.4)); fig.subplots_adjust(left=.26,right=.94,top=.81,bottom=.18)
    title(fig,"开发集五折模型比较","相同48个输入字段、相同五折；Ridge α=1仍是已考察配置中的最高均值")
    for i,row in enumerate(df.itertuples()):
        vals=folds.loc[folds.candidate==row.candidate,"r2"].to_numpy()
        color=BLUE if row.selected else TEAL
        ax.scatter(vals,i+np.linspace(-.13,.13,5),s=30,facecolors="white",edgecolors=color,lw=.9,zorder=3)
        ax.errorbar(row.fold_r2_mean,i,xerr=row.fold_r2_std_ddof1,fmt="D",ms=6.5,color=color,capsize=4,lw=1.5,zorder=4)
        ax.text(.690,i,f"{row.fold_r2_mean:.6f}",ha="right",va="center",fontsize=11,color=INK)
    ax.set_yticks(np.arange(len(df)),[NAMES[n] for n in df.candidate]); ax.invert_yaxis()
    ax.set_xlim(.56,.693); ax.set_xlabel("验证折 R²（横轴为局部区间）")
    ax.set_xticks(np.arange(.56,.681,.02)); quiet(ax)
    foot(fig,"空心点：各折结果；实心点：五折均值；误差线：折间样本标准差，非置信区间。两种小α配置差异不作显著性结论。")
    export(fig,"T1_F02_development_model_comparison",records,{"source_csv":"model_comparison.csv;model_fold_scores.csv","candidates":6,"fold_points":30})


def parity(records):
    dev=read("development_oof.csv"); test=read("locked_test_diagnostics.csv"); ms=read("evaluation_metrics.csv").set_index("split")
    values=np.concatenate([d[c].to_numpy() for d in [dev,test] for c in ["true_value","predicted_value"]])
    low=float(np.floor(values.min())); high=float(np.ceil(values.max()))
    fig,axes=plt.subplots(1,2,figsize=(11.6,6.4)); fig.subplots_adjust(left=.075,right=.90,bottom=.20,top=.79,wspace=.49)
    title(fig,"预测值与真实评分对照","开发集折外结果与锁定验证分别展示；使用全部样本，未抽样或裁剪预测")
    cmap=LinearSegmentedColormap.from_list("blue_density",["#EDF5FA",BLUE,"#173C5C"])
    sums=[]
    for ax,df,label,key in zip(axes,[dev,test],["A  开发集折外预测","B  锁定验证预测"],["development_oof","locked_test"]):
        h=ax.hexbin(df.true_value,df.predicted_value,gridsize=34,mincnt=1,bins="log",extent=(low,high,low,high),cmap=cmap,linewidths=.12)
        counts=int(h.get_array().sum()); assert counts==len(df); sums.append(counts)
        ax.plot([low,high],[low,high],ls="--",lw=1.15,color=TEAL,label="理想预测线")
        ax.set(xlim=(low,high),ylim=(low,high),xlabel="真实睡眠质量评分",ylabel="预测睡眠质量评分")
        ax.set_aspect("equal",adjustable="box"); ax.set_title(label,loc="left",pad=13)
        m=ms.loc[key]
        ax.text(.04,.95,f"n = {len(df):,}\nR² = {m.r2:.4f}\nRMSE = {m.rmse:.4f}",transform=ax.transAxes,va="top",fontsize=11,
                bbox={"boxstyle":"round,pad=.38","facecolor":"white","edgecolor":"none","alpha":.92})
        c=fig.colorbar(h,ax=ax,fraction=.044,pad=.03)
        ticks=[v for v in [1,2,5,10,20,50,100,200,500,1000] if v<=h.get_array().max()]
        c.set_ticks(ticks,labels=[str(v) for v in ticks]); c.ax.yaxis.set_minor_locator(NullLocator())
        c.set_label("单元样本数（对数色阶）",fontsize=10); c.ax.tick_params(labelsize=9)
        c.solids.set_rasterized(False)
        c.solids.set_edgecolor("face")
        quiet(ax,None)
    foot(fig,"六边形颜色表示样本密度，不改变坐标或预测值。开发集折外n=8,000；锁定测试n=2,000。两个面板坐标范围一致。")
    export(fig,"T1_F03_observed_vs_predicted",records,{"source_csv":"development_oof.csv;locked_test_diagnostics.csv","hexbin_count_sums":sums,"axis_limits":[low,high]})


def residuals(records):
    df=read("locked_test_diagnostics.csv"); s=read("residual_summary.csv").iloc[0]
    e=df.residual_true_minus_pred.to_numpy(); ae=np.sort(np.abs(e)); n=len(df)
    fig,axes=plt.subplots(1,3,figsize=(12.3,5.5)); fig.subplots_adjust(left=.065,right=.97,top=.79,bottom=.21,wspace=.33)
    title(fig,"锁定验证集误差诊断","仅分析已保存的2,000条锁定验证预测；残差结果不用于新的模型选择")
    ax=axes[0]; ax.scatter(df.predicted_value,e,s=8,alpha=.30,color=BLUE,rasterized=False,linewidths=0)
    ax.axhline(0,color=TEAL,lw=1.4,ls="--"); ax.set(xlabel="预测评分",ylabel="残差（真实值 - 预测值）"); ax.set_title("A  残差与预测值",loc="left")
    bound=float(np.ceil(np.max(np.abs(e))))
    ax.set_ylim(-bound,bound); ax.set_yticks(np.arange(-bound,bound+.01,2))
    ax.set_xlim(1,10); ax.set_xticks([2,4,6,8,10])
    quiet(ax,"y")
    ax=axes[1]; hist,edges,_=ax.hist(e,bins=30,color=TEAL,edgecolor="white",lw=.5,alpha=.9)
    assert int(hist.sum())==2000
    ax.axvline(0,color=BLUE,lw=1.4,ls="--"); ax.set(xlabel="残差（真实值 - 预测值）",ylabel="样本数"); ax.set_title("B  残差分布",loc="left")
    ax.set_xlim(-bound,bound); ax.set_xticks(np.arange(-bound,bound+.01,2))
    ax.set_ylim(0,float(np.ceil(hist.max()/50)*50))
    ax.text(.96,.96,f"均值 = {s.mean_residual:.4f}\n标准差 = {s.residual_sd_ddof1:.4f}",transform=ax.transAxes,ha="right",va="top",fontsize=10)
    quiet(ax,"y")
    ax=axes[2]; ax.step(ae,np.arange(1,n+1)/n,where="post",color=BLUE,lw=1.6)
    ax.axhline(.9,color=GREY,lw=.9,ls=":"); ax.axvline(s.absolute_error_p90,color=TEAL,lw=1.2,ls="--")
    ax.text(.96,.15,f"MAE = {s.mae:.4f}\n90%误差分位 = {s.absolute_error_p90:.4f}",transform=ax.transAxes,ha="right",fontsize=10)
    ax.set(xlabel="绝对误差",ylabel="累计样本比例",ylim=(0,1.04),xlim=(0,bound)); ax.set_title("C  绝对误差累计分布",loc="left")
    ax.set_xticks(np.arange(0,bound+.01,1)); ax.set_yticks(np.arange(0,1.01,.2))
    quiet(ax,"y")
    foot(fig,"残差正值表示模型低估，负值表示高估。90%误差分位是本次测试的描述性统计，不是新个体的预测区间。")
    export(fig,"T1_F04_locked_test_residual_diagnostics",records,{"source_csv":"locked_test_diagnostics.csv;residual_summary.csv","scatter_rows":2000,"histogram_count":int(hist.sum()),"ecdf_rows":2000})


def importance(records):
    df=read("permutation_importance_all48.csv").head(12)
    fig,ax=plt.subplots(figsize=(10.4,7.8)); fig.subplots_adjust(left=.23,right=.94,top=.82,bottom=.17)
    title(fig,"开发集验证折的特征置换重要性","按原始字段整列置换；48个字段全部计算，本图展示均值最高的12个")
    positions=np.arange(len(df)); colors=[BLUE if i<3 else TEAL for i in positions]
    ax.barh(positions,df.mean_r2_drop,color=colors,height=.64,zorder=2)
    ax.errorbar(df.mean_r2_drop,positions,xerr=df.sd_five_fold_means,fmt="none",ecolor=INK,capsize=3,lw=1,zorder=3)
    for i,row in enumerate(df.itertuples()):
        ax.text(row.mean_r2_drop+row.sd_five_fold_means+.009,i,f"{row.mean_r2_drop:.4f}",va="center",fontsize=10.6,color=INK)
    ax.set_yticks(positions,[LABELS[n] for n in df.feature]); ax.invert_yaxis()
    ax.set_xlim(0,.58); ax.set_xlabel("置换后验证折 R² 下降量（越大表示模型越依赖该字段）")
    quiet(ax)
    foot(fig,"每折重复10次，先取折内均值，再汇总5折；误差线为5个折均值的样本标准差，非置信区间。\n这是固定模型的预测依赖分析，不是因果贡献；相关特征可能分担重要性。未据此删字段或重新训练。")
    export(fig,"T1_F05_development_permutation_importance",records,{"source_csv":"permutation_importance_all48.csv","shown_features":12,"computed_features":48,"permutations":2400,"test_used_for_importance":False})


def correlations(records):
    order=list(read("correlation_display_fields.csv").feature)
    full=read("development_numeric_correlations_full.csv").pivot(index="feature_x",columns="feature_y",values="pearson_r")
    values=full.loc[order,order].to_numpy()
    masked=np.ma.array(values,mask=np.triu(np.ones_like(values,dtype=bool),1))
    fig,ax=plt.subplots(figsize=(10.4,9.0)); fig.subplots_adjust(left=.22,right=.89,top=.81,bottom=.25)
    title(fig,"主要数值特征的相关性","开发集n=8,000；按开发集置换重要性选取前10个数值字段，附目标用于描述性对照")
    cmap=LinearSegmentedColormap.from_list("blue_white_teal",[BLUE,"#FAFCFD",TEAL])
    edges=np.arange(len(order)+1)-.5
    image=ax.pcolormesh(edges,edges,masked,vmin=-1,vmax=1,cmap=cmap,shading="flat",rasterized=False)
    ax.set_xlim(-.5,len(order)-.5); ax.set_ylim(len(order)-.5,-.5); ax.set_aspect("equal")
    for i in range(len(order)):
        for j in range(i+1):
            ax.text(j,i,f"{values[i,j]:.2f}",ha="center",va="center",fontsize=9.7,color="white" if abs(values[i,j])>.72 else INK)
    labels=[LABELS[n] for n in order]
    ax.set_xticks(np.arange(len(order)),labels,rotation=45,ha="right",rotation_mode="anchor",fontsize=10.3)
    ax.set_yticks(np.arange(len(order)),labels,fontsize=10.3)
    ax.set_xticks(np.arange(-.5,len(order),1),minor=True); ax.set_yticks(np.arange(-.5,len(order),1),minor=True)
    ax.grid(which="minor",color="white",lw=1.2); ax.tick_params(which="both",length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    cax=fig.add_axes([.915,.30,.018,.46]); cb=fig.colorbar(image,cax=cax); cb.set_label("Pearson相关系数",fontsize=11)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")
    foot(fig,"睡眠质量评分只用于本图相关性对照，从未作为模型输入。完整34个数值特征与目标的相关矩阵另存CSV。\n线性相关不等于因果关系，也不能单凭低相关断言不存在语义泄露；本图不触发任何新特征筛选。")
    export(fig,"T1_F06_development_numeric_correlations",records,{"source_csv":"development_numeric_correlations_full.csv;correlation_display_fields.csv","display_matrix_shape":[11,11],"development_rows":8000,"target_is_predictor":False})


def main(font):
    info=setup(font)
    report=json.loads((HERE/"analysis_verification.json").read_text())
    for name,expected in report["outputs"].items():
        assert hashlib.sha256((HERE/"source_data"/name).read_bytes()).hexdigest()==expected
    records=[]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for function in [workflow,comparison,parity,residuals,importance,correlations]:function(records)
    glyph=[str(w.message) for w in captured if "Glyph" in str(w.message)]
    if glyph: raise ValueError("中文字体缺字："+"\n".join(sorted(set(glyph))))
    writer=PdfWriter()
    for r in records:
        reader=PdfReader(HERE/r["files"]["pdf"]); assert len(reader.pages)==1
        writer.add_page(reader.pages[0])
    writer.add_metadata({"/Title":"任务一图件总览","/Author":"DATA2600655","/CreationDate":"D:20260828000000Z"})
    buffer=io.BytesIO();writer.write(buffer)
    assert len(PdfReader(io.BytesIO(buffer.getvalue())).pages)==6
    atomic_bytes(HERE/"任务一图件总览.pdf",buffer.getvalue())
    result={"status":"PASS","figures":records,"font":info,"matplotlib_version":matplotlib.__version__,
            "warnings":[str(w.message) for w in captured],"glyph_warnings":0,"overview_pages":6,
            "needs_visual_review":True,"source_hash_verification":"PASS"}
    atomic_bytes(HERE/"plot_verification.json",(json.dumps(result,ensure_ascii=False,indent=2)+"\n").encode())
    print(json.dumps({"status":"PASS","figure_groups":len(records),"overview_pages":6,"glyph_warnings":0},ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--font",type=Path)
    main(parser.parse_args().font)
