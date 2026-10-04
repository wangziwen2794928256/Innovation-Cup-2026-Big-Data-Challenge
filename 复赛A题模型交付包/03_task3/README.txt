任务三：综合健康评分回归预测
队伍编号：DATA2600655
目标变量：Health_Score
主要指标：R²

主要文件
task3_model_report.pdf                    完整建模小报告。
models/task3_final_model.joblib           全量训练的最终推理模型。
predict_task3.py                          新数据预测入口。
train_task3.py                            训练、验证、消融与图件复现入口。
src/modeling.py                           特征工程与残差集成回归器定义。
data/A_dataset.csv                        训练数据副本。
data/task3_test_input_example.csv         无目标列调用示例。
predictions/validation_predictions.csv    2,000 条固定验证预测。
results/run_summary.json                  主要指标、环境与残差权重。
results/model_comparison.csv              候选模型比较。
results/ablation_results.csv              消融实验结果。
results/fold_metrics.csv                  五折性能。
results/permutation_importance.csv        分组置换重要性。
figures/                                  建模报告配套图件（8 张 PNG）。

直接预测
python -m pip install -r requirements.txt
python predict_task3.py --input data/task3_test_input_example.csv --output task3_predictions.csv

完整复现
python train_task3.py

主要结果
五折 OOF：R²=0.983168，RMSE=1.588977，MAE=1.259917。
固定验证：R²=0.982881，RMSE=1.613125，MAE=1.286548。
最终模型：Ridge 线性主干 + 三路非线性残差加权集成；Wellness_Category 不进入训练与推理。
