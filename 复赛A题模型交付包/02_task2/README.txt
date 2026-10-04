任务二：工作效率评分回归与驱动因素分析
队伍编号：DATA2600655
目标变量：Productivity_Score
主要指标：调整 R²

主要文件
task2_model_report.pdf                    完整建模小报告。
models/task2_final_model.joblib           全量训练的最终推理模型。
predict_task2.py                          新数据预测入口。
train_task2.py                            训练、验证、指标与图件复现入口。
src/modeling.py                           预处理、时间编码与回归流水线定义。
data/A_dataset.csv                        训练数据副本。
data/task2_test_input_example.csv         无目标列调用示例。
predictions/validation_predictions.csv    2,000 条固定验证预测。
results/run_summary.json                  主要指标与模型口径。
results/model_comparison.csv              候选模型比较。
results/permutation_importance.csv        生产力模型置换重要性。
results/energy_driver_importance.csv      精力辅助模型置换重要性。
figures/                                  建模报告配套图件（9 张 PNG）。

直接预测
python -m pip install -r requirements.txt
python predict_task2.py --input data/task2_test_input_example.csv --output task2_predictions.csv

完整复现
python train_task2.py

主要结果
五折 OOF：R²=0.620693，调整 R²=0.615989，RMSE=0.705211，MAE=0.561556。
固定验证：R²=0.614099，调整 R²=0.594205，RMSE=0.718606，MAE=0.576399。
最终模型：Ridge，alpha=10。
