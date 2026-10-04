# 任务三：综合健康类别预测

本目录包含任务三的训练模型、完整源代码、运行环境、模型调用说明、预测结果、评价表与建模报告。

## 目录

- `Introduction.pdf`：任务三设计思路、实验方法、环境与模型调用说明。
- `code/train_task3.py`：数据审计、五折折外训练、概率融合、锁定留出评价、消融、绘图及全量重训。
- `code/predict_task3.py`：载入固化模型并输出三分类预测。
- `models/`：FULL竞赛模型、NO_DIRECT解释模型及融合清单。
- `data/A题数据集.csv`：任务三训练数据。
- `results/`：折外预测、锁定留出预测、评价指标与最终预测结果。
- `figures/`：建模报告中的真实实验图件。
- `configs/task3_config.json`：标签映射、随机种子、划分与融合参数。
- `requirements.txt`：Python依赖版本。

## 环境

- Python 3.11
- NumPy 2.3.5
- pandas 2.2.3
- scikit-learn 1.8.0
- Matplotlib 3.10.8
- seaborn 0.13.2
- joblib 1.5.3

安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 直接预测

在`任务三`目录执行：

```bash
python code/predict_task3.py \
  --input data/A题数据集.csv \
  --output results/task3_predictions.csv \
  --mode competition
```

输出字段为：

- `Person_ID`
- `Predicted_Wellness_Category`
- `Probability_Average`
- `Probability_Good`
- `Probability_Excellent`

`competition`模式调用FULL三分类ExtraTrees模型。`explanation`模式调用删除`Health_Score`与`Fitness_Level`后的折外加权融合模型。

## 完整复现

```bash
python code/train_task3.py
```

重新训练的模型、表格、预测与图件统一写入`reproduce_results/`。随机种子固定为2026，开发集与锁定留出集比例为80%/20%，开发集采用五折分层折外预测。
