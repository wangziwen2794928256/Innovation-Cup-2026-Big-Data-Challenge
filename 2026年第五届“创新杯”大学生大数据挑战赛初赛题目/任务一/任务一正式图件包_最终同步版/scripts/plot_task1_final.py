from __future__ import annotations

import json
import shutil
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score, log_loss
)
from PIL import Image, ImageOps, ImageDraw

SRC = Path('/mnt/data/task1_v4/任务一_完成交付包_V4')
OUTROOT = Path('/mnt/data/任务一正式图件包_最终同步版')
if OUTROOT.exists():
    shutil.rmtree(OUTROOT)
DATA = OUTROOT / 'data'
FIG = OUTROOT / 'figures'
SCRIPT = OUTROOT / 'scripts'
for p in (DATA, FIG, SCRIPT):
    p.mkdir(parents=True, exist_ok=True)

# ---------- style ----------
DPI = 600
CM = 1 / 2.54
COLORS = {
    'rf': '#3B9B8F',
    'xgb': '#376795',
    'lr': '#7556A3',
    'full': '#176B87',
    'strict': '#E07B39',
    'grid': '#D9DEE3',
    'text': '#262626',
    'gray': '#8C939A',
}
font_regular = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
font_bold = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
plt.rcParams.update({
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.85,
    'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'],
    'text.color': COLORS['text'],
    'savefig.facecolor': 'white',
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

def clean_name(s: str) -> str:
    return str(s).replace('_', ' ')

def style_axis(ax, grid_axis='y'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis=grid_axis, color=COLORS['grid'], linewidth=0.55, alpha=0.68, linestyle='--')
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9.0, width=0.75, length=3.2)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(font_regular)

def save_all(fig, stem: str, pad=0.06):
    for ext in ('png', 'pdf', 'svg'):
        kwargs = dict(bbox_inches='tight', pad_inches=pad)
        if ext == 'png':
            kwargs['dpi'] = DPI
        fig.savefig(FIG / f'{stem}.{ext}', **kwargs)
    plt.close(fig)

# ---------- read exact V4 outputs ----------
full_dir = SRC / 'experiments' / 'competition_full'
strict_dir = SRC / 'experiments' / 'no_time_no_composite'
minimal_dir = SRC / 'experiments' / 'minimal_rule_tree'

full_pred = pd.read_csv(full_dir / 'validation_predictions.csv', encoding='utf-8-sig')
strict_pred = pd.read_csv(strict_dir / 'validation_predictions.csv', encoding='utf-8-sig')
full_cmp = pd.read_csv(full_dir / 'model_comparison.csv', encoding='utf-8-sig')
strict_cmp = pd.read_csv(strict_dir / 'model_comparison.csv', encoding='utf-8-sig')
full_metrics = json.loads((full_dir / 'metrics.json').read_text(encoding='utf-8'))
strict_metrics = json.loads((strict_dir / 'metrics.json').read_text(encoding='utf-8'))
minimal_metrics = json.loads((minimal_dir / 'metrics.json').read_text(encoding='utf-8'))
full_shap_global = pd.read_csv(full_dir / 'explanation_data' / 'shap_global_importance.csv', encoding='utf-8-sig')
strict_shap_global = pd.read_csv(strict_dir / 'explanation_data' / 'shap_global_importance.csv', encoding='utf-8-sig')
full_shap_long = pd.read_csv(full_dir / 'explanation_data' / 'shap_values_long_top20.csv', encoding='utf-8-sig')
strict_shap_long = pd.read_csv(strict_dir / 'explanation_data' / 'shap_values_long_top20.csv', encoding='utf-8-sig')
full_cal = pd.read_csv(full_dir / 'plot_data' / 'calibration_curve_data.csv', encoding='utf-8-sig')
strict_cal = pd.read_csv(strict_dir / 'plot_data' / 'calibration_curve_data.csv', encoding='utf-8-sig')
raw_data = pd.read_csv(SRC / 'data' / 'A题数据集.csv', encoding='utf-8-sig')

# Store corrected source tables in a simple, backwards-compatible structure.
cv_rows = []
for label, df in [('FULL', full_cmp), ('STRICT', strict_cmp)]:
    for _, r in df.iterrows():
        cv_rows.append({
            'Feature_Set': label,
            'Model': r['model'],
            'CV_Accuracy_Mean': r['repeated_cv_accuracy_mean'],
            'CV_Accuracy_Std': r['repeated_cv_accuracy_std'],
            'CV_F1_Mean': r['repeated_cv_f1_mean'],
            'CV_F1_Std': r['repeated_cv_f1_std'],
            'CV_ROC_AUC_Mean': r['repeated_cv_roc_auc_mean'],
            'CV_ROC_AUC_Std': r['repeated_cv_roc_auc_std'],
            'Total_CV_Time_Seconds': r['total_seconds'],
        })
cv = pd.DataFrame(cv_rows)
cv.to_csv(DATA / 'task1_final_cv_summary.csv', index=False, encoding='utf-8-sig')

pred_rows = []
for label, df in [('FULL', full_pred), ('STRICT', strict_pred)]:
    temp = pd.DataFrame({
        'Person_ID': df['Person_ID'],
        'Feature_Set': label,
        'Actual_Early_Waker': df['true_label'],
        'Predicted_Early_Waker': df['predicted_label'],
        'Probability_Yes': df['probability_yes'],
        'Correct': df['correct'],
        'Model': df['model'],
    })
    pred_rows.append(temp)
pred = pd.concat(pred_rows, ignore_index=True)
pred.to_csv(DATA / 'task1_final_predictions.csv', index=False, encoding='utf-8-sig')

summary = pd.DataFrame([
    {
        'Feature_Set': 'FULL', 'Best_Model': full_metrics['selected_model'],
        'Holdout_Accuracy': full_metrics['holdout_accuracy'],
        'Holdout_Precision': full_metrics['holdout_precision'],
        'Holdout_Recall': full_metrics['holdout_recall'],
        'Holdout_F1': full_metrics['holdout_f1'],
        'Holdout_ROC_AUC': full_metrics['holdout_roc_auc'],
        'Holdout_Average_Precision': full_metrics['holdout_average_precision'],
        'Holdout_Log_Loss': full_metrics['holdout_log_loss'],
        'TN': full_metrics['confusion_matrix'][0][0], 'FP': full_metrics['confusion_matrix'][0][1],
        'FN': full_metrics['confusion_matrix'][1][0], 'TP': full_metrics['confusion_matrix'][1][1],
        'CV_Accuracy_Mean': full_metrics['repeated_cv_accuracy_mean'],
        'CV_Accuracy_Std': full_metrics['repeated_cv_accuracy_std'],
        'CV_F1_Mean': full_metrics['repeated_cv_f1_mean'],
        'CV_ROC_AUC_Mean': full_metrics['repeated_cv_roc_auc_mean'],
    },
    {
        'Feature_Set': 'STRICT', 'Best_Model': strict_metrics['selected_model'],
        'Holdout_Accuracy': strict_metrics['holdout_accuracy'],
        'Holdout_Precision': strict_metrics['holdout_precision'],
        'Holdout_Recall': strict_metrics['holdout_recall'],
        'Holdout_F1': strict_metrics['holdout_f1'],
        'Holdout_ROC_AUC': strict_metrics['holdout_roc_auc'],
        'Holdout_Average_Precision': strict_metrics['holdout_average_precision'],
        'Holdout_Log_Loss': strict_metrics['holdout_log_loss'],
        'TN': strict_metrics['confusion_matrix'][0][0], 'FP': strict_metrics['confusion_matrix'][0][1],
        'FN': strict_metrics['confusion_matrix'][1][0], 'TP': strict_metrics['confusion_matrix'][1][1],
        'CV_Accuracy_Mean': strict_metrics['repeated_cv_accuracy_mean'],
        'CV_Accuracy_Std': strict_metrics['repeated_cv_accuracy_std'],
        'CV_F1_Mean': strict_metrics['repeated_cv_f1_mean'],
        'CV_ROC_AUC_Mean': strict_metrics['repeated_cv_roc_auc_mean'],
    },
])
summary.to_csv(DATA / 'task1_final_summary.csv', index=False, encoding='utf-8-sig')

settings = {
    'source_package': '任务一_完成交付包_V4',
    'holdout_seed': 42,
    'holdout_test_size': 0.2,
    'repeated_cv': '3 random seeds × 5 folds',
    'full_definition': '删除 Person_ID 与四个跨任务综合字段；保留 Wake Up Time 及周期派生特征',
    'strict_definition': '在 FULL 基础上进一步删除 Wake Up Time、Sleep Time、Sleep Duration Hours 及其派生信息',
    'full_final_model': 'Random_Forest',
    'strict_final_model': 'Logistic_Regression',
    'verified': True,
}
(DATA / 'task1_final_settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
raw_data.to_csv(DATA / 'A题数据集.csv', index=False, encoding='utf-8-sig')
full_shap_global.to_csv(DATA / 'shap_global_full.csv', index=False, encoding='utf-8-sig')
strict_shap_global.to_csv(DATA / 'shap_global_strict.csv', index=False, encoding='utf-8-sig')
full_shap_long.to_csv(DATA / 'shap_values_full_top20.csv', index=False, encoding='utf-8-sig')
strict_shap_long.to_csv(DATA / 'shap_values_strict_top20.csv', index=False, encoding='utf-8-sig')
full_cal.to_csv(DATA / 'calibration_full.csv', index=False, encoding='utf-8-sig')
strict_cal.to_csv(DATA / 'calibration_strict.csv', index=False, encoding='utf-8-sig')

# ---------- Figure 0: target distribution ----------
counts = raw_data['Early_Waker'].value_counts().reindex(['No','Yes'])
fig, ax = plt.subplots(figsize=(15.2*CM, 8.8*CM))
bars = ax.bar(['No','Yes'], counts.values, width=0.55, color=['#4E91A8','#E07B39'], edgecolor='white')
ax.set_ylabel('样本数', fontproperties=font_bold, fontsize=11)
style_axis(ax)
ax.set_ylim(0, counts.max()*1.16)
for b, label, n in zip(bars, ['No','Yes'], counts.values):
    pct = n/counts.sum()*100
    ax.text(b.get_x()+b.get_width()/2, n+counts.max()*0.025, f'{n:,}\n({pct:.2f}%)', ha='center', va='bottom', fontproperties=font_bold, fontsize=9)
fig.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.92)
save_all(fig, 'FigT1_00_target_distribution', pad=0.08)

# ---------- Figure 1: exact repeated CV model comparison ----------
order = [('FULL','Random_Forest'),('FULL','XGBoost'),('FULL','Logistic_Regression'),
         ('STRICT','Logistic_Regression'),('STRICT','XGBoost'),('STRICT','Random_Forest')]
sel = pd.DataFrame([cv[(cv.Feature_Set==fs)&(cv.Model==m)].iloc[0] for fs,m in order])
labels = ['Random\nForest','XGBoost','Logistic\nRegression','Logistic\nRegression','XGBoost','Random\nForest']
colors = [COLORS['rf'],COLORS['xgb'],COLORS['lr'],COLORS['lr'],COLORS['xgb'],COLORS['rf']]
fig, ax = plt.subplots(figsize=(16.4*CM,9.2*CM))
x=np.arange(6)
bars=ax.bar(x,sel.CV_Accuracy_Mean,yerr=sel.CV_Accuracy_Std,capsize=2.8,width=0.56,color=colors,edgecolor='white',linewidth=0.65,error_kw={'elinewidth':0.85,'capthick':0.85})
ax.set_xlim(-0.75,5.75); ax.set_ylim(0,1.105)
ax.set_ylabel('重复 5 折交叉验证准确率',fontproperties=font_bold,fontsize=11)
ax.set_xticks(x,labels)
style_axis(ax)
ax.axvline(2.5,color='#AAB2B8',ls='--',lw=0.85)
for b,v,s in zip(bars,sel.CV_Accuracy_Mean,sel.CV_Accuracy_Std):
    text=f'{v:.4f}' if s==0 else f'{v:.4f}\n±{s:.4f}'
    ax.text(b.get_x()+b.get_width()/2,v+0.019,text,ha='center',va='bottom',fontproperties=font_regular,fontsize=7.7)
trans=ax.get_xaxis_transform()
for left,right,name in [(-0.35,2.35,'FULL'),(2.65,5.35,'NO_TIME_NO_COMPOSITE')]:
    y=-0.21
    ax.plot([left,right],[y,y],transform=trans,color=COLORS['text'],lw=0.75,clip_on=False)
    ax.plot([left,left],[y,y+0.035],transform=trans,color=COLORS['text'],lw=0.75,clip_on=False)
    ax.plot([right,right],[y,y+0.035],transform=trans,color=COLORS['text'],lw=0.75,clip_on=False)
    ax.text((left+right)/2,y-0.06,name,transform=trans,ha='center',va='top',fontproperties=font_bold,fontsize=8.5)
fig.subplots_adjust(left=0.105,right=0.975,top=0.925,bottom=0.31)
save_all(fig,'FigT1_01_model_cv_accuracy',pad=0.08)

# ---------- Figures 2-3: exact confusion matrices ----------
def plot_cm(df, scope_label, stem, accent):
    y_true=(df.true_label=='Yes').astype(int)
    y_pred=(df.predicted_label=='Yes').astype(int)
    cm=confusion_matrix(y_true,y_pred,labels=[0,1])
    pct=cm/cm.sum(axis=1,keepdims=True)*100
    fig,ax=plt.subplots(figsize=(7.2*CM,5.5*CM))
    im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(cm.max(),1),aspect='equal')
    for i in range(2):
        for j in range(2):
            color='white' if cm[i,j] > cm.max()*0.48 else COLORS['text']
            ax.text(j,i,f'{cm[i,j]}\n({pct[i,j]:.1f}%)',ha='center',va='center',fontproperties=font_bold,fontsize=8,color=color)
    ax.set_xticks([0,1],['No','Yes']); ax.set_yticks([0,1],['No','Yes'])
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontproperties(font_regular); t.set_fontsize(8.6)
    ax.set_xlabel('预测类别',fontproperties=font_bold,fontsize=9.5)
    ax.set_ylabel('真实类别',fontproperties=font_bold,fontsize=9.5)
    ax.text(0.01,1.035,scope_label,transform=ax.transAxes,fontproperties=font_bold,fontsize=8.2,color=accent)
    ax.set_xticks(np.arange(-.5,2,1),minor=True); ax.set_yticks(np.arange(-.5,2,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=0.7); ax.tick_params(which='minor',bottom=False,left=False)
    fig.subplots_adjust(left=0.21,right=0.96,bottom=0.19,top=0.91)
    save_all(fig,stem,pad=0.04)

plot_cm(full_pred,'FULL','FigT1_02_full_confusion_matrix',COLORS['full'])
plot_cm(strict_pred,'NO_TIME_NO_COMPOSITE','FigT1_03_strict_confusion_matrix',COLORS['strict'])

# ---------- Figure 4: exact ROC ----------
fig,ax=plt.subplots(figsize=(15.2*CM,8.8*CM))
for label,df,color in [('FULL',full_pred,COLORS['full']),('NO_TIME_NO_COMPOSITE',strict_pred,COLORS['strict'])]:
    yt=(df.true_label=='Yes').astype(int).to_numpy(); p=df.probability_yes.to_numpy()
    fpr,tpr,_=roc_curve(yt,p); aucv=roc_auc_score(yt,p)
    ax.plot(fpr,tpr,lw=1.45,color=color,label=f'{label} (AUC={aucv:.4f})')
ax.plot([0,1],[0,1],ls='--',lw=0.85,color=COLORS['gray'],label='随机分类')
ax.set_xlim(-0.025,1.025); ax.set_ylim(-0.025,1.055)
ax.set_xlabel('假阳性率',fontproperties=font_bold,fontsize=11); ax.set_ylabel('真阳性率',fontproperties=font_bold,fontsize=11)
style_axis(ax,'both'); ax.grid(True,color=COLORS['grid'],ls='--',lw=0.55,alpha=.68)
ax.legend(frameon=False,loc='lower right',prop=font_regular,fontsize=8.4)
fig.subplots_adjust(left=.125,right=.96,bottom=.16,top=.92)
save_all(fig,'FigT1_04_roc_curves',pad=.08)

# ---------- Figure 5: exact PR ----------
fig,ax=plt.subplots(figsize=(15.2*CM,8.8*CM))
for label,df,color in [('FULL',full_pred,COLORS['full']),('NO_TIME_NO_COMPOSITE',strict_pred,COLORS['strict'])]:
    yt=(df.true_label=='Yes').astype(int).to_numpy(); p=df.probability_yes.to_numpy()
    precision,recall,_=precision_recall_curve(yt,p); ap=average_precision_score(yt,p)
    ax.plot(recall,precision,lw=1.45,color=color,label=f'{label} (AP={ap:.4f})')
base=(full_pred.true_label=='Yes').mean()
ax.axhline(base,ls='--',lw=.85,color=COLORS['gray'],label=f'正类基线={base:.4f}')
ax.set_xlim(-.025,1.025); ax.set_ylim(-.025,1.055)
ax.set_xlabel('召回率',fontproperties=font_bold,fontsize=11); ax.set_ylabel('精确率',fontproperties=font_bold,fontsize=11)
style_axis(ax,'both'); ax.grid(True,color=COLORS['grid'],ls='--',lw=.55,alpha=.68)
ax.legend(frameon=False,loc='lower left',prop=font_regular,fontsize=8.4)
fig.subplots_adjust(left=.125,right=.96,bottom=.16,top=.92)
save_all(fig,'FigT1_05_precision_recall',pad=.08)

# ---------- Figure 6: exact reliability curves ----------
fig,ax=plt.subplots(figsize=(15.2*CM,8.8*CM))
for label,dcal,metric,color in [
    ('FULL',full_cal,full_metrics,COLORS['full']),
    ('NO_TIME_NO_COMPOSITE',strict_cal,strict_metrics,COLORS['strict'])]:
    ax.plot(dcal.mean_predicted_probability,dcal.observed_positive_rate,marker='o',ms=3.6,lw=1.4,color=color,label=f"{label} (Log Loss={metric['holdout_log_loss']:.4f})")
ax.plot([0,1],[0,1],ls='--',lw=.85,color=COLORS['gray'],label='理想校准')
ax.set_xlim(-.035,1.035); ax.set_ylim(-.035,1.035)
ax.set_xlabel('平均预测概率',fontproperties=font_bold,fontsize=11); ax.set_ylabel('实际正类频率',fontproperties=font_bold,fontsize=11)
style_axis(ax,'both'); ax.grid(True,color=COLORS['grid'],ls='--',lw=.55,alpha=.68)
ax.legend(frameon=False,loc='upper left',prop=font_regular,fontsize=8.2)
fig.subplots_adjust(left=.125,right=.96,bottom=.16,top=.92)
save_all(fig,'FigT1_06_reliability_curve',pad=.08)

# ---------- Figure 7: exact global SHAP importance, FULL vs STRICT ----------
fig,axes=plt.subplots(1,2,figsize=(18.2*CM,9.2*CM))
for ax,df,label,color in [
    (axes[0],full_shap_global.head(12),'FULL',COLORS['xgb']),
    (axes[1],strict_shap_global.head(12),'NO_TIME_NO_COMPOSITE',COLORS['rf'])]:
    d=df.head(12).iloc[::-1].copy()
    ax.barh(np.arange(len(d)),d.mean_abs_shap,color=color,height=.62,alpha=.92)
    ax.set_yticks(np.arange(len(d)),[clean_name(x) for x in d.feature])
    for t in ax.get_yticklabels(): t.set_fontproperties(font_regular); t.set_fontsize(6.7)
    ax.set_xlabel('Mean |SHAP value|',fontproperties=font_bold,fontsize=9.2)
    ax.text(.02,1.03,label,transform=ax.transAxes,fontproperties=font_bold,fontsize=9.4,color=color)
    style_axis(ax,'x')
    ax.grid(axis='x',color=COLORS['grid'],ls='--',lw=.5,alpha=.65)
    ax.margins(x=.07)
fig.subplots_adjust(left=.18,right=.985,bottom=.15,top=.91,wspace=.88)
save_all(fig,'Fig07_task1_shap_summary',pad=.06)

# ---------- Figure 7b: exact strict SHAP beeswarm-like plot ----------
top_feats=strict_shap_global.head(15).feature.tolist()
d=strict_shap_long[strict_shap_long.feature.isin(top_feats)].copy()
fig,ax=plt.subplots(figsize=(10.2*CM,9.0*CM))
rng=np.random.default_rng(2026)
all_vals=d.feature_value_encoded.to_numpy()
vmin,vmax=np.nanpercentile(all_vals,[3,97])
norm=Normalize(vmin=vmin,vmax=vmax,clip=True)
cmap=plt.get_cmap('coolwarm')
for yi,feat in enumerate(top_feats[::-1]):
    sub=d[d.feature==feat]
    jitter=rng.normal(0,.085,len(sub))
    ax.scatter(sub.shap_value,yi+jitter,c=sub.feature_value_encoded,cmap=cmap,norm=norm,s=7.0,alpha=.68,edgecolors='none',rasterized=True)
ax.axvline(0,color=COLORS['gray'],lw=.75,ls='--')
ax.set_yticks(np.arange(len(top_feats)),[clean_name(x) for x in top_feats[::-1]])
for t in ax.get_yticklabels(): t.set_fontproperties(font_regular); t.set_fontsize(7.1)
ax.set_xlabel('SHAP value',fontproperties=font_bold,fontsize=9.5)
style_axis(ax,'x'); ax.grid(axis='x',color=COLORS['grid'],ls='--',lw=.45,alpha=.6)
sm=ScalarMappable(norm=norm,cmap=cmap); sm.set_array([])
cbar=fig.colorbar(sm,ax=ax,pad=.025,fraction=.035)
cbar.set_label('特征值（编码后）',fontproperties=font_bold,fontsize=7.2)
cbar.set_ticks([vmin,vmax]); cbar.set_ticklabels(['低','高'])
for t in cbar.ax.get_yticklabels(): t.set_fontproperties(font_regular); t.set_fontsize(7)
fig.subplots_adjust(left=.39,right=.92,bottom=.12,top=.97)
save_all(fig,'Fig07b_task1_shap_beeswarm_strict',pad=.04)

# ---------- Figure 8: exact FULL SHAP dependence from archived SHAP values ----------
features=[('Wake_Up_Minutes','起床时间（分钟）'),('Sleep_Duration_Hours','睡眠时长（小时）')]
fig,axes=plt.subplots(2,1,figsize=(7.2*CM,7.5*CM))
for ax,(feat,xlabel) in zip(axes,features):
    sub=full_shap_long[full_shap_long.feature==feat].sort_values('feature_value_encoded')
    x=sub.feature_value_encoded.to_numpy(); yv=sub.shap_value.to_numpy()
    ax.scatter(x,yv,s=7,alpha=.40,color='#376795',edgecolors='none')
    # Quantile-bin median trend, using exact archived values.
    try:
        q=pd.qcut(x,q=min(14,max(5,len(np.unique(x))//4)),duplicates='drop')
        trend=pd.DataFrame({'x':x,'y':yv,'q':q}).groupby('q',observed=True).median(numeric_only=True)
        ax.plot(trend.x,trend.y,color=COLORS['strict'],lw=1.15)
    except Exception:
        pass
    ax.axhline(0,color=COLORS['gray'],lw=.65,ls='--')
    ax.set_xlabel(xlabel,fontproperties=font_bold,fontsize=8.1)
    ax.set_ylabel('SHAP value',fontproperties=font_bold,fontsize=8.1)
    style_axis(ax,'both'); ax.grid(True,color=COLORS['grid'],ls='--',lw=.45,alpha=.60)
    ax.margins(x=.05,y=.10)
fig.subplots_adjust(left=.22,right=.97,bottom=.12,top=.98,hspace=.56)
save_all(fig,'Fig08_task1_shap_dependence',pad=.04)

# ---------- concise results sheet image ----------
fig,ax=plt.subplots(figsize=(18.2*CM,6.2*CM)); ax.axis('off')
columns=['口径','最终模型','ACC','Precision','Recall','F1','ROC-AUC']
rows=[
    ['FULL','Random Forest','1.0000','1.0000','1.0000','1.0000','1.0000'],
    ['NO_TIME_NO_COMPOSITE','Logistic Regression','0.7685','0.7418','0.6803','0.7097','0.8355'],
]
table=ax.table(cellText=rows,colLabels=columns,cellLoc='center',colLoc='center',loc='center',colWidths=[.27,.20,.09,.11,.09,.09,.11])
table.auto_set_font_size(False); table.set_fontsize(8.2); table.scale(1,1.8)
for (r,c),cell in table.get_celld().items():
    cell.set_edgecolor('#8A969E'); cell.set_linewidth(.6)
    if r==0:
        cell.set_facecolor('#176B87'); cell.get_text().set_color('white'); cell.get_text().set_fontproperties(font_bold)
    else:
        cell.set_facecolor('#F5F8FA' if r%2==0 else 'white'); cell.get_text().set_fontproperties(font_regular)
fig.subplots_adjust(left=.02,right=.98,bottom=.08,top=.92)
save_all(fig,'FigT1_09_final_results_table',pad=.04)

# ---------- overview montage ----------
pngs=[
    'FigT1_00_target_distribution.png','FigT1_01_model_cv_accuracy.png',
    'FigT1_02_full_confusion_matrix.png','FigT1_03_strict_confusion_matrix.png',
    'FigT1_04_roc_curves.png','FigT1_05_precision_recall.png',
    'FigT1_06_reliability_curve.png','Fig07_task1_shap_summary.png',
    'Fig07b_task1_shap_beeswarm_strict.png','Fig08_task1_shap_dependence.png',
    'FigT1_09_final_results_table.png'
]
thumb_w,thumb_h=760,500
cols=3; rows_n=(len(pngs)+cols-1)//cols
canvas=Image.new('RGB',(cols*thumb_w,rows_n*(thumb_h+42)),'white')
draw=ImageDraw.Draw(canvas)
for i,name in enumerate(pngs):
    im=Image.open(FIG/name).convert('RGB')
    im.thumbnail((thumb_w-34,thumb_h-34),Image.Resampling.LANCZOS)
    cell=Image.new('RGB',(thumb_w,thumb_h),'white')
    x0=(thumb_w-im.width)//2; y0=(thumb_h-im.height)//2
    cell.paste(im,(x0,y0))
    cell=ImageOps.expand(cell,border=1,fill='#D9DEE3')
    x=(i%cols)*thumb_w; y=(i//cols)*(thumb_h+42)
    canvas.paste(cell,(x,y))
    draw.text((x+16,y+thumb_h+9),name,fill='#262626')
canvas.save(OUTROOT/'任务一图件总览.png',quality=95)

# ---------- README, correction note, script ----------
readme=f'''# 任务一正式图件包（最终同步版）\n\n本包已以 `任务一_完成交付包_V4` 的真实预测文件、交叉验证汇总、SHAP 数据和校准数据为唯一来源重新生成。\n\n## 最终指标\n\n- FULL / Random Forest：ACC=1.0000，F1=1.0000，ROC-AUC=1.0000；\n- NO_TIME_NO_COMPOSITE / Logistic Regression：ACC=0.7685，Precision=0.7418，Recall=0.6803，F1=0.7097，ROC-AUC=0.8355；\n- 严格口径混淆矩阵：TN=971，FP=197，FN=266，TP=566；\n- 严格口径 AP=0.7913，Log Loss=0.4927。\n\n## 修正内容\n\n旧版图件包中的严格口径 ACC=0.7695、TN=973、FP=195 等数值来自较早实验。本版已全部同步为 V4 验证通过的最终结果。\n\n## 文件结构\n\n- `figures/`：PDF、SVG、600 dpi PNG；\n- `data/`：所有图件对应的最终数据；\n- `scripts/plot_task1_final.py`：一键重绘脚本；\n- `任务一图件总览.png`：快速预览；\n- `manifest.json`：来源与哈希记录。\n\n## 使用原则\n\n论文和群内交接均以本包为任务一最终图件版本。模型工程包本身不需要重训或替换。\n'''
(OUTROOT/'README.md').write_text(readme,encoding='utf-8')
(OUTROOT/'修正说明.md').write_text('''# 修正说明\n\n本次仅同步图件数据与版式，不重新训练模型。\n\n需要在总论文中替换的旧图主要包括：候选模型 CV 比较、严格口径混淆矩阵、ROC、PR、可靠性以及任务一 SHAP 图。\n\n最终数字固定为：FULL ACC=1.0000；严格口径 ACC=0.7685、F1=0.7097、AUC=0.8355、TN/FP/FN/TP=971/197/266/566。\n''',encoding='utf-8')
# Copy this generator as reproducible plotting script.
shutil.copy2(Path(__file__), SCRIPT/'plot_task1_final.py')

# ---------- manifest ----------
manifest={
    'package_name':'任务一正式图件包_最终同步版',
    'source':'任务一交付包_.zip / 任务一_完成交付包_V4',
    'verified_metrics':{
        'FULL':{k:full_metrics[k] for k in ['selected_model','holdout_accuracy','holdout_precision','holdout_recall','holdout_f1','holdout_roc_auc','holdout_average_precision','holdout_log_loss','confusion_matrix']},
        'NO_TIME_NO_COMPOSITE':{k:strict_metrics[k] for k in ['selected_model','holdout_accuracy','holdout_precision','holdout_recall','holdout_f1','holdout_roc_auc','holdout_average_precision','holdout_log_loss','confusion_matrix']},
    },
    'files':[]
}
for p in sorted(OUTROOT.rglob('*')):
    if p.is_file():
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        manifest['files'].append({'path':str(p.relative_to(OUTROOT)),'sha256':h,'size_bytes':p.stat().st_size})
(OUTROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

print(f'Created {OUTROOT}')
print('PNG count:',len(list(FIG.glob('*.png'))),'PDF:',len(list(FIG.glob('*.pdf'))),'SVG:',len(list(FIG.glob('*.svg'))))
