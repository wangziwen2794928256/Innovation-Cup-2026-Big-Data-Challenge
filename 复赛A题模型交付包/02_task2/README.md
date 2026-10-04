# 任务二：工作效率评分回归与驱动因素分析

> 队伍编号：DATA2600655　|　目标变量：`Productivity_Score`　|　主要指标：调整 R²

<div align="right">

[← 交付包总览](../README.md) · [↑ 仓库根 README](../../README.md)

</div>

本目录为复赛 A 题任务二的完整工程，包含建模报告、已训练模型、训练与预测代码、固定验证预测、模型比较与驱动因素分析结果及配套图件。

## 一、主要结果

| 评价阶段 | R² | 调整 R² | RMSE | MAE |
| --- | --- | --- | --- | --- |
| 开发集五折 OOF | 0.620693 | 0.615989 | 0.705211 | 0.561556 |
| 2,000 条固定锁定验证 | 0.614099 | 0.594205 | 0.718606 | 0.576399 |

**最终模型**：Ridge 回归（alpha = 10）。任务二剔除综合健康评分及精力、疲劳、情绪、焦虑、抑郁、专注等并发状态评分，采用周期时间编码与 L2 正则化处理多重共线性，避免循环解释；除生产力主模型外，另设精力辅助模型用于驱动因素分析。

**验证口径**：与任务一一致的固定 8,000/2,000 划分与开发集五折 OOF 选模协议，锁定验证集仅用于一次独立评价。

## 二、目录结构

| 路径 | 说明 |
| --- | --- |
| `task2_model_report.pdf` | 完整建模小报告 |
| `models/task2_final_model.joblib` | 全量训练的最终推理模型（权重不入库，可复现） |
| `predict_task2.py` | 新数据预测入口 |
| `train_task2.py` | 训练、验证、指标与图件复现入口 |
| `src/modeling.py` | 预处理、时间编码与回归流水线定义 |
| `data/A_dataset.csv` | 训练数据副本 |
| `data/task2_test_input_example.csv` | 无目标列调用示例 |
| `predictions/validation_predictions.csv` | 2,000 条固定验证预测 |
| `results/run_summary.json` | 主要指标与模型口径 |
| `results/model_comparison.csv` | 候选模型比较 |
| `results/permutation_importance.csv` | 生产力模型置换重要性 |
| `results/energy_driver_importance.csv` | 精力辅助模型置换重要性 |
| `results/multicollinearity_pairs.csv` / `numeric_vif.csv` | 多重共线性诊断 |
| `plotting/` | 绘图脚本与固定图源 |
| `figures/` | 建模报告配套图件（9 张 PNG） |
| `requirements.txt` | 运行依赖 |

## 三、运行方法

### 3.1 直接预测

```bash
python -m pip install -r requirements.txt
python predict_task2.py --input data/task2_test_input_example.csv --output task2_predictions.csv
```

### 3.2 完整复现

```bash
python train_task2.py
```

---

*原始交付说明见 `README.txt`，两者内容一致；本文件为规范化文档版本。*
