# 任务一正式图件包（最终同步版）

本包已以 `任务一_完成交付包_V4` 的真实预测文件、交叉验证汇总、SHAP 数据和校准数据为唯一来源重新生成。

## 最终指标

- FULL / Random Forest：ACC=1.0000，F1=1.0000，ROC-AUC=1.0000；
- NO_TIME_NO_COMPOSITE / Logistic Regression：ACC=0.7685，Precision=0.7418，Recall=0.6803，F1=0.7097，ROC-AUC=0.8355；
- 严格口径混淆矩阵：TN=971，FP=197，FN=266，TP=566；
- 严格口径 AP=0.7913，Log Loss=0.4927。

## 修正内容

旧版图件包中的严格口径 ACC=0.7695、TN=973、FP=195 等数值来自较早实验。本版已全部同步为 V4 验证通过的最终结果。

## 文件结构

- `figures/`：PDF、SVG、600 dpi PNG；
- `data/`：所有图件对应的最终数据；
- `scripts/plot_task1_final.py`：一键重绘脚本；
- `任务一图件总览.png`：快速预览；
- `manifest.json`：来源与哈希记录。

## 使用原则

论文和群内交接均以本包为任务一最终图件版本。模型工程包本身不需要重训或替换。
