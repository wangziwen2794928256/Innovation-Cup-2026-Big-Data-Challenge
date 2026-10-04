# 任务二正式图件使用说明

本批图件由 `00_绘图脚本/plot_task2_all.py` 生成。正文优先使用 `strict_no_proxy` 严格去代理变量口径；`08_竞赛口径对照` 仅用于说明竞赛全特征口径。每张 Matplotlib 图均同时导出 600 dpi PNG 与 PDF。

## 推荐正文图

- 标签与分箱：FigT2_01、FigT2_02、FigT2_03。
- 主结果：FigT2_09、FigT2_11、FigT2_13、FigT2_14、FigT2_15。
- 等级误差：FigT2_16、FigT2_17。
- 可解释性：FigT2_18、FigT2_20、FigT2_21、FigT2_22。
- 融合与路线：FigT2_23、FigT2_26。

## 图件明细

- `FigT2_01_label_distribution`（01_数据与标签）：核对训练/验证集四级标签是否近似均衡。数据：strict_no_proxy/label_distribution.csv
- `FigT2_02_health_score_distribution_bins`（01_数据与标签）：说明任务二四级标签的训练集内定义。数据：A题数据集.csv + strict_no_proxy/fold_bins.csv
- `FigT2_03_fold_boundary_stability`（01_数据与标签）：证明所有分箱边界均由训练折独立计算且波动较小。数据：strict_no_proxy/fold_bins.csv
- `FigT2_04_missing_rate`（01_数据与标签）：定位真正存在缺失的字段并避免展示空列。数据：A题数据集.csv + strict_no_proxy/feature_list.csv
- `FigT2_05_standardized_boxplots`（01_数据与标签）：比较不同量纲变量的分布、偏态和离群点。数据：A题数据集.csv + strict_no_proxy/shap_global_importance.csv
- `FigT2_06_correlation_heatmap`（01_数据与标签）：展示关键数值特征的线性单调关系及与目标的关联。数据：A题数据集.csv + strict_no_proxy/shap_global_importance.csv
- `FigT2_07_tuning_history`（02_训练调参）：保留LightGBM和Ordinal Logistic的搜索轨迹及最优位置。数据：strict_no_proxy/tuning_history.csv
- `FigT2_08_sampling_comparison`（02_训练调参）：比较无采样、类别权重和SMOTE的真实增益。数据：strict_no_proxy/sampling_comparison.csv
- `FigT2_09_model_performance`（03_模型性能）：比较基线、单模型与最终Stacking融合模型。数据：strict_no_proxy/model_comparison.csv
- `FigT2_10_class_metrics`（03_模型性能）：检查Poor、Average、Good、Excellent各等级的识别质量。数据：strict_no_proxy/class_metrics.csv
- `FigT2_11_stacking_confusion_matrix`（04_误差与校准）：同时展示样本数和行归一化比例，观察相邻等级误判。数据：strict_no_proxy/confusion_matrix_long.csv
- `FigT2_12_confusion_matrix_models`（04_误差与校准）：比较Ordinal、LightGBM和Stacking的等级错误结构。数据：strict_no_proxy/confusion_matrix_long.csv
- `FigT2_13_multiclass_roc`（03_模型性能）：按类别比较主要模型的一对其余判别能力。数据：strict_no_proxy/validation_predictions_all_models.csv
- `FigT2_14_multiclass_pr`（03_模型性能）：在各等级样本比例下比较主要模型的查准率与召回率。数据：strict_no_proxy/validation_predictions_all_models.csv
- `FigT2_15_reliability_curve`（04_误差与校准）：判断四个等级概率是否过度自信并报告ECE/Brier。数据：strict_no_proxy/plot_data/calibration_curve_data.csv + calibration_metrics.csv
- `FigT2_16_grade_error_distribution`（04_误差与校准）：突出相邻等级误差与跨两级以上严重误差。数据：strict_no_proxy/grade_error_distribution.csv
- `FigT2_17_prediction_distribution`（04_误差与校准）：检查模型是否系统性高估或低估某一健康等级。数据：strict_no_proxy/validation_predictions_all_models.csv
- `FigT2_18_shap_beeswarm`（05_可解释性）：展示关键特征对Poor和Excellent等级预测的方向与强度。数据：strict_no_proxy/explanation_data/shap_values_long_top20.csv
- `FigT2_19_shap_global_heatmap`（05_可解释性）：比较同一特征在四个等级中的贡献强度。数据：strict_no_proxy/explanation_data/shap_global_importance.csv
- `FigT2_20_shap_dependence`（05_可解释性）：展示关键连续特征的非线性阈值及贡献方向。数据：strict_no_proxy/explanation_data/shap_values_long_top20.csv
- `FigT2_21_ordinal_coefficients`（05_可解释性）：解释特征升高对进入较高健康等级的方向与不确定性。数据：strict_no_proxy/explanation_data/ordinal_coefficients.csv；说明：区间为正则化Hessian近似区间
- `FigT2_22_lightgbm_feature_importance`（05_可解释性）：从树模型角度补充全局特征贡献排序。数据：strict_no_proxy/explanation_data/feature_importance.csv
- `FigT2_23_stacking_sankey`（03_模型性能）：展示两个一级模型预测等级与最终融合判断的流向。数据：strict_no_proxy/validation_predictions_all_models.csv；说明：HTML为Plotly交互版；每个样本分别计入两个一级模型流向
- `FigT2_24_tsne_umap`（06_特征关系与降维）：观察四个健康等级在严格口径数值特征空间中的分离与重叠。数据：A题数据集.csv + strict_no_proxy/validation_predictions_all_models.csv；说明：仅使用严格口径数值特征；未输入Health Score及代理变量
- `FigT2_25_mi_feature_network`（06_特征关系与降维）：节点大小表示与等级标签的互信息，连线粗细表示特征间Spearman关联。数据：A题数据集.csv + strict_no_proxy/fold_bins.csv；说明：这是MI—相关网络，不冒充需要专门MIC算法计算的MIC网络
- `FigT2_26_task2_pipeline`（07_技术路线）：概括数据、分箱、双模型、OOF融合、评价和输出链路。数据：code/task2_pipeline.py + strict_no_proxy实验
- `FigT2_27_feature_scope_comparison`（08_竞赛口径对照）：量化代理变量对竞赛准确率和可解释性口径的影响。数据：plot_data_all_scopes/feature_scope_comparison.csv；说明：正文建议使用严格口径；竞赛全特征口径作为对照或附录
- `FigT2_28_model_radar`（03_模型性能）：综合比较分类性能、等级一致性与严重误差控制。数据：strict_no_proxy/model_comparison.csv；说明：雷达轴均统一为越大越好；径向范围0.65–1.00并明确标注
- `FigT2_29_local_shap_contributions`（05_可解释性）：解释一个高置信度且预测正确样本的主要正负贡献。数据：strict_no_proxy/explanation_data/shap_values_long_top20.csv + validation_predictions_all_models.csv；说明：未保存SHAP基线值，因此使用真实局部贡献条形图而不伪造瀑布终点

## 重新运行

```powershell
& 'C:\Users\luo29\Desktop\任务二\.venv_plot\Scripts\python.exe' 'C:\Users\luo29\Desktop\任务二\图片\00_绘图脚本\plot_task2_all.py'
```

若只需要论文插图，优先使用 PDF；WPS兼容性不佳时使用同名PNG。