任务一：睡眠质量评分回归预测
队伍编号：DATA2600655
目标变量：Sleep_Quality_Score
主要指标：R²

主要文件
task1_model_report.pdf                   完整建模、验证结果、运行环境与模型调用说明。
model/model.joblib                       8,000 条开发记录拟合的最终任务一流水线。
model/model_metadata.json                模型参数、输入字段、训练行数与环境记录。
model_config.json                        输入字段、类别字段、8,000/2,000 划分与模型参数。
code/predict_task1.py                    新数据预测入口。
code/train_task1.py                      按既定 48 字段与 Ridge alpha=1 复现 8,000 条开发集拟合。
data/development_features.csv            8,000 条开发集特征。
data/development_targets.csv             8,000 条开发集目标。
data/locked_test_inputs.csv              2,000 条固定验证输入。
predictions/development_oof_predictions.csv 8,000 条开发集五折 OOF 预测。
predictions/validation_predictions.csv   2,000 条固定验证预测。
results/model_comparison.csv             候选模型比较。
results/model_fold_scores.csv            五折逐折指标。
figures/                                 Python 生成的建模报告配套图件（6 张 PNG）。
requirements.txt                         运行依赖。

直接预测
python -m pip install -r requirements.txt
python code/predict_task1.py --input data/locked_test_inputs.csv --output my_predictions.csv

复现拟合
python code/train_task1.py --output model_rebuild

模型口径
全部 10,000 条记录固定划分为 8,000 条开发集和 2,000 条锁定验证集。
候选模型与参数仅在 8,000 条开发记录上通过五折 OOF 比较确定。
最终任务一模型使用完整 8,000 条开发记录拟合；2,000 条锁定验证记录不参与训练和调参，仅用于一次独立评价。

主要结果
开发集整体折外 R²：0.651667
固定验证 R²：0.634460
固定验证 RMSE：0.944974
固定验证 MAE：0.748283
