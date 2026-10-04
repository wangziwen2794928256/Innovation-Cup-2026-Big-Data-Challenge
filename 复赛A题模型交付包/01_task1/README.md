# 任务一：睡眠质量评分回归预测

> 队伍编号：DATA2600655　|　目标变量：`Sleep_Quality_Score`　|　主要指标：R²

<div align="right">

[← 交付包总览](../README.md) · [↑ 仓库根 README](../../README.md)

</div>

本目录为复赛 A 题任务一的完整工程，包含建模报告、已训练模型、训练与预测代码、固定验证预测、模型比较结果及配套图件。

## 一、主要结果

| 评价阶段 | R² | RMSE | MAE |
| --- | --- | --- | --- |
| 开发集五折 OOF | 0.651667 | — | — |
| 2,000 条固定锁定验证 | 0.634460 | 0.944974 | 0.748283 |

**最终模型**：Ridge 回归（alpha = 1），按既定 48 字段口径拟合 8,000 条开发记录；严格仅使用非睡眠类特征，睡眠时刻类字段经周期（正弦/余弦）编码。

**验证口径**：全部 10,000 条记录固定划分为 8,000 条开发集与 2,000 条锁定验证集；候选模型与参数仅在开发集上经五折 OOF 比较确定，锁定验证集不参与训练与调参，仅用于一次独立评价。

## 二、目录结构

| 路径 | 说明 |
| --- | --- |
| `task1_model_report.pdf` | 完整建模、验证结果、运行环境与模型调用说明 |
| `model/model.joblib` | 8,000 条开发记录拟合的最终任务一流水线（权重不入库，可复现） |
| `model/model_metadata.json` | 模型参数、输入字段、训练行数与环境记录 |
| `model_config.json` | 输入字段、类别字段、8,000/2,000 划分与模型参数 |
| `code/predict_task1.py` | 新数据预测入口 |
| `code/train_task1.py` | 按既定 48 字段与 Ridge alpha=1 复现开发集拟合 |
| `data/development_features.csv` | 8,000 条开发集特征 |
| `data/development_targets.csv` | 8,000 条开发集目标 |
| `data/locked_test_inputs.csv` | 2,000 条固定验证输入 |
| `predictions/development_oof_predictions.csv` | 8,000 条开发集五折 OOF 预测 |
| `predictions/validation_predictions.csv` | 2,000 条固定验证预测 |
| `results/model_comparison.csv` | 候选模型比较 |
| `results/model_fold_scores.csv` | 五折逐折指标 |
| `plotting/` | 绘图脚本、固定图源与校验记录 |
| `figures/` | Python 生成的建模报告配套图件（6 张 PNG） |
| `requirements.txt` | 运行依赖 |

## 三、运行方法

### 3.1 直接预测

```bash
python -m pip install -r requirements.txt
python code/predict_task1.py --input data/locked_test_inputs.csv --output my_predictions.csv
```

### 3.2 复现拟合

```bash
python code/train_task1.py --output model_rebuild
```

---

*原始交付说明见 `README.txt`，两者内容一致；本文件为规范化文档版本。*
