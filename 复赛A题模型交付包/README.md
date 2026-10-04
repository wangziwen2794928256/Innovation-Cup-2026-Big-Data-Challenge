# 复赛 A 题模型交付包（DATA2600655）

> 2026 年第五届"创新杯"大学生大数据挑战赛 · 复赛 A 题模型支撑材料

本目录为按赛事要求提交的官方模型支撑材料完整源码（对应仓库根目录 `ADATA2600655model.zip` 的解包版本）。

## 一、目录构成

| 子目录 | 内容 |
| --- | --- |
| [`01_task1/`](01_task1/README.md) | 睡眠质量评分回归预测（Ridge，锁定验证 R² = 0.634460） |
| [`02_task2/`](02_task2/README.md) | 工作效率评分回归与驱动因素分析（Ridge α=10，锁定验证调整 R² = 0.594205） |
| [`03_task3/`](03_task3/README.md) | 综合健康评分回归预测（Ridge 主干 + 三路残差加权集成，锁定验证 R² = 0.982881） |
| [`04_paper_figures/`](04_paper_figures/README.md) | 正式论文修订版 25 张纯矢量 SVG 与统一制图脚本 |
| `DATA2600655 AI 使用说明.pdf` | AI 工具使用详情说明 |

每个任务均包含：

1. `task*_model_report.pdf`：建模思路、验证结果、运行环境与模型调用说明；
2. 已训练最终模型（`*.joblib`，不入版本库，可经训练脚本原位重建）；
3. 训练与预测代码；
4. `requirements.txt` 固定依赖；
5. 固定验证预测、模型比较与必要结果文件；
6. `figures/`：Python 生成的建模报告配套 PNG 图件。

## 二、运行方法

进入对应任务目录，先安装 `requirements.txt`，再按该目录 `README.md`（或原始 `README.txt`、`task*_model_report.pdf`）中的命令运行。例如任务一：

```bash
cd 01_task1
python -m pip install -r requirements.txt
python code/predict_task1.py --input data/locked_test_inputs.csv --output my_predictions.csv   # 直接预测
python code/train_task1.py --output model_rebuild                                             # 复现拟合
```

## 三、绘图复现

三个任务均保留 `plotting/` 目录，包含建模报告图件的绘图脚本与固定图源；`04_paper_figures/` 另外保存正式论文修订版 SVG 及统一制图脚本，详见其目录内说明。

---

*原始交付说明见 `README.txt`，两者内容一致；本文件为规范化文档版本。*
