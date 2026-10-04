# 任务三：综合健康评分回归预测

> 队伍编号：DATA2600655　|　目标变量：`Health_Score`　|　主要指标：R²

<div align="right">

[← 交付包总览](../README.md) · [↑ 仓库根 README](../../README.md)

</div>

本目录为复赛 A 题任务三的完整工程，包含建模报告、已训练模型、训练与预测代码、固定验证预测、消融实验、置换重要性及配套图件。

## 一、主要结果

| 评价阶段 | R² | RMSE | MAE |
| --- | --- | --- | --- |
| 开发集五折 OOF | 0.983168 | 1.588977 | 1.259917 |
| 2,000 条固定锁定验证 | 0.982881 | 1.613125 | 1.286548 |

**最终模型**：**Ridge 线性主干 + 三路非线性残差加权集成**。先由 Ridge 恢复综合健康评分的线性主结构，再由非线性模型学习残差并按 OOF 预测加权融合；严格折外验证证实残差集成带来稳定非线性增益。

**泄露控制**：由 `Health_Score` 分档直接生成的 `Wellness_Category` 属于直接目标派生标签，**不进入训练与推理**；对高相关代理 `Healthy Aging Score` 采取主结果与口径消融并列报告策略。

**验证口径**：与任务一、二一致的固定 8,000/2,000 划分与开发集五折 OOF 选模协议，锁定验证集仅用于一次独立评价。

## 二、目录结构

| 路径 | 说明 |
| --- | --- |
| `task3_model_report.pdf` | 完整建模小报告 |
| `models/task3_final_model.joblib` | 全量训练的最终推理模型（权重不入库，可复现） |
| `predict_task3.py` | 新数据预测入口 |
| `train_task3.py` | 训练、验证、消融与图件复现入口 |
| `src/modeling.py` | 特征工程与残差集成回归器定义 |
| `data/A_dataset.csv` | 训练数据副本 |
| `data/task3_test_input_example.csv` | 无目标列调用示例 |
| `predictions/validation_predictions.csv` | 2,000 条固定验证预测 |
| `results/run_summary.json` | 主要指标、环境与残差权重 |
| `results/model_comparison.csv` | 候选模型比较 |
| `results/ablation_results.csv` | 消融实验结果 |
| `results/fold_metrics.csv` | 五折性能 |
| `results/permutation_importance.csv` | 分组置换重要性 |
| `plotting/` | 绘图脚本与固定图源 |
| `figures/` | 建模报告配套图件（8 张 PNG） |
| `requirements.txt` | 运行依赖 |

## 三、运行方法

### 3.1 直接预测

```bash
python -m pip install -r requirements.txt
python predict_task3.py --input data/task3_test_input_example.csv --output task3_predictions.csv
```

### 3.2 完整复现

```bash
python train_task3.py
```

---

*原始交付说明见 `README.txt`，两者内容一致；本文件为规范化文档版本。*
