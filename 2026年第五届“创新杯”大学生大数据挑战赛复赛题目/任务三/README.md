# 复赛A题任务三：综合健康评分预测

本目录包含任务三的训练模型、完整源代码、验证预测、实验图件与建模报告。预测目标为连续变量 `Health_Score`，评价指标为决定系数 R²。

## 目录说明

- `Introduction.pdf`：任务三建模报告。
- `models/task3_final_model.joblib`：使用全部 10,000 条记录重训的最终模型，用于新测试集预测。
- `models/task3_holdout_model.joblib`：仅使用 80% 开发集训练的复核模型，用于重现报告中的锁定验证结果。
- `predict_task3.py`：模型调用入口。
- `train_task3.py`：完整训练、五折 OOF、锁定验证、消融、重要性和绘图流程。
- `src/modeling.py`：特征工程与残差集成模型定义。
- `predictions/task3_validation_predictions.csv`：2,000 条锁定验证样本的 `Person_ID`、真值、预测值与残差。
- `predictions/task3_oof_predictions.csv`：8,000 条开发样本的五折 OOF 预测。
- `data/task3_test_input_example.csv`：不含目标列的调用示例。
- `figures/`：报告所用的真实实验图件。

## 运行环境

推荐 Python 3.12。安装固定依赖：

```bash
python -m pip install -r requirements.txt
```

## 直接预测

在本目录执行：

```bash
python predict_task3.py \
  --input data/task3_test_input_example.csv \
  --output predictions/task3_test_output_example.csv
```

默认加载 `models/task3_final_model.joblib`。输入 CSV 必须包含 `Person_ID` 及训练所需原始特征，可以不含 `Health_Score` 和 `Wellness_Category`。输出至少包含：

- `Person_ID`
- `Predicted_Health_Score`

若输入文件包含真值，输出会同时保留 `True_Health_Score`。

## 完整复现

原始数据文件置于 `data/A题数据集.csv` 后执行：

```bash
python train_task3.py
```

脚本使用固定随机种子 2026，重新生成模型、OOF预测、锁定验证预测、指标表与全部图件。训练和推理阶段均自动剔除由目标分档得到的直接泄露字段 `Wellness_Category`。

## 已锁定结果

- 五折 OOF：R² = 0.983168，RMSE = 1.588977，MAE = 1.259917。
- 20% 锁定验证集：R² = 0.982881，RMSE = 1.613125，MAE = 1.286548。

