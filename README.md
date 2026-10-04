<div align="center">

# 2026 年第五届"创新杯"大学生大数据挑战赛参赛项目

### 基于正则化回归与残差集成的健康生活方式多任务评分预测研究

**队伍编号：DATA2600655**　|　**赛题：初赛 A 题 / 复赛 A 题**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Last Commit](https://img.shields.io/github/last-commit/wangziwen2794928256/Innovation-Cup-2026-Big-Data-Challenge)](https://github.com/wangziwen2794928256/Innovation-Cup-2026-Big-Data-Challenge/commits/main)
[![Repo Size](https://img.shields.io/github/repo-size/wangziwen2794928256/Innovation-Cup-2026-Big-Data-Challenge)](https://github.com/wangziwen2794928256/Innovation-Cup-2026-Big-Data-Challenge)

[📄 复赛论文](ADATA260065.pdf) · [📄 初赛论文](ADATA2600655初赛.pdf) · [📦 复赛模型交付包](复赛A题模型交付包/README.md) · [🚀 复现指南](#五复现指南) · [📊 核心结果](#核心结果总览)

</div>

---

## 目录

- [一、项目简介](#一项目简介)
- [二、效果一览（核心图件）](#二效果一览核心图件)
- [三、核心方法框架](#三核心方法框架)
- [四、核心结果总览](#四核心结果总览)
- [五、目录结构](#五目录结构)
- [六、复现指南](#六复现指南)
- [七、主要结论与边界声明](#七主要结论与边界声明)
- [八、引用与许可](#八引用与许可)

---

## 一、项目简介

本项目为 2026 年第五届"创新杯"大学生大数据挑战赛的完整参赛工程，覆盖**初赛 A 题**与**复赛 A 题**两个阶段的全部建模工作。项目围绕一份 **10,000 条个体记录、64 个字段**的健康生活方式数据集展开，研究对象涵盖人口统计、睡眠作息、运动行为、饮食习惯、工作与社会行为、心理状态、生理指标、疾病风险及综合评分等维度。

项目将各阶段任务统一表述为监督学习问题，并以**信息边界审计为前提、固定折外验证为主线**，构建了"性能—稳定性—机理边界"三层证据体系：

- **初赛 A 题**：完成二分类判别建模（含 FULL 全特征口径与剔除时间/复合特征的严格口径两套方案，配合 SHAP 可解释性分析）与三分类综合健康类别预测（FULL 竞赛模型与 NO_DIRECT 解释模型双轨方案）；
- **复赛 A 题**：完成三项回归任务——非睡眠特征驱动的**睡眠质量预测**、剔除并发状态代理的**工作效率预测**、以及剔除目标派生泄露后的**综合健康评分高精度预测**（线性主干 + 残差集成）。

核心论文见仓库根目录 [`ADATA260065.pdf`](ADATA260065.pdf)（复赛论文，27 页，含完整方法、实验与稳健性论证）。

---

## 二、效果一览（核心图件）

<table>
<tr>
<td width="50%" align="center"><b>初赛任务一 · SHAP 蜂群解释（严格口径）</b></td>
<td width="50%" align="center"><b>复赛任务一 · 睡眠质量预测对照</b></td>
</tr>
<tr>
<td><img src="2026年第五届“创新杯”大学生大数据挑战赛初赛题目/任务一/任务一正式图件包_最终同步版/figures/Fig07b_task1_shap_beeswarm_strict.png" width="100%"></td>
<td><img src="复赛A题模型交付包/01_task1/figures/T1_F03_observed_vs_predicted.png" width="100%"></td>
</tr>
<tr>
<td align="center"><b>复赛任务二 · 工作效率预测对照</b></td>
<td align="center"><b>复赛任务三 · 综合健康评分预测对照</b></td>
</tr>
<tr>
<td><img src="复赛A题模型交付包/02_task2/figures/Fig04_true_vs_predicted.png" width="100%"></td>
<td><img src="复赛A题模型交付包/03_task3/figures/Fig04_true_vs_predicted.png" width="100%"></td>
</tr>
</table>

> 全部图件（PDF / SVG / 600 dpi PNG）与一键重绘脚本随代码一并入库，见各任务 `figures/` 与 `plotting/` 目录。

---

## 三、核心方法框架

### 3.1 数据契约与泄露审计

建模前对全部 64 个字段实施语义级审计，区分**结构性缺失、普通缺失、标识字段、目标字段、直接派生标签与潜在并发代理**四类边界：

| 字段 | 缺失数 | 缺失率 | 处理方式 |
| --- | --- | --- | --- |
| Exercise Type | 824 | 8.24% | 与运动频率为 0 高度重合的结构性缺失，保留 `None` 类别 |
| Workout Intensity | 824 | 8.24% | 同上，保留 `None` 类别 |
| Alcohol Consumption | 3,014 | 30.14% | 题包无法区分"不饮酒"与"未记录"，训练折内作为显式未知类别 |

- **任务一（复赛）**：按题面排除睡眠时刻、睡眠时长、夜间醒来、午睡、睡眠障碍等全部睡眠类字段；
- **任务二（复赛）**：剔除综合健康评分及精力、疲劳、情绪、焦虑、抑郁、专注等并发状态评分，避免循环解释；
- **任务三（复赛/初赛）**：强制剔除由 `Health_Score` 分档直接生成的 `Wellness_Category` 泄露字段；对 `Healthy Aging Score`（与目标 Pearson 相关约 0.9388）采取"主结果 + 去代理敏感性结果"并列报告策略。

### 3.2 统一预处理与验证协议

1. **周期时间编码**：入睡/起床时刻位于 24 小时周期，直接以十进制小时建模会把 23:59 与 00:01 错误地视为相距近 24 小时。项目先将时刻转换为分钟 `m`，再构造正弦/余弦分量完成周期嵌入；
2. **折内拟合原则**：所有插补、编码与标准化统计量仅在训练折内估计，杜绝验证信息回流；
3. **固定划分协议**：全体样本固定划分为 8,000 条开发记录与 2,000 条锁定验证记录（80% / 20%），开发集内部采用五折分层折外（OOF）预测选模，**锁定集全程不参与字段选择、模型比较或超参数确定**；
4. **随机种子固定为 2026**，保证全流程可复现。

### 3.3 逐步增强的模型策略

| 任务 | 主模型 | 选择依据 |
| --- | --- | --- |
| 复赛任务一（睡眠质量） | Ridge 回归 | 高维独热编码下稳定处理，开发集 OOF R² = 0.6517，锁定验证 R² = 0.6345（RMSE = 0.9450） |
| 复赛任务二（工作效率） | Ridge（α = 10）+ 周期时间编码 | 处理多重共线性，OOF 调整 R² = 0.6160，锁定验证调整 R² = 0.5942（RMSE = 0.7186）；精力辅助模型 OOF 调整 R² = 0.7080 |
| 复赛任务三（综合健康评分） | **Ridge 线性主干 + 残差集成**（非线性模型学习残差并按 OOF 预测加权融合） | 严格折外验证确认存在非线性增益后引入，OOF R² = 0.9832，锁定验证 R² = 0.9829（RMSE = 1.6131） |
| 初赛任务一（二分类判别） | FULL 口径 Random Forest；严格口径 Logistic Regression | FULL：ACC = F1 = ROC-AUC = 1.0000；严格口径：ACC = 0.7685，ROC-AUC = 0.8355（TN = 971, FP = 197, FN = 266, TP = 566），配合 SHAP 蜂群图与依赖图解释 |
| 初赛任务三（三分类健康类别） | FULL 竞赛模型 ExtraTrees；NO_DIRECT 解释模型（折外加权融合 Logistic / HistGB / ExtraTrees / MLP / GradientBoosting） | `competition` 模式输出 Average / Good / Excellent 三分类概率；`explanation` 模式剔除 `Health_Score` 与 `Fitness_Level` 后评估真实增量 |

模型解释层面，项目综合**置换重要性（原始字段分组置换、五折汇总均值与标准差）、相关矩阵、残差诊断、折间稳定性与口径消融实验**交叉验证模型依赖；全部重要性排序仅反映预测依赖，不直接等同于因果干预效应。

---

## 四、核心结果总览

<table>
<tr><th>阶段 · 任务</th><th>主指标（锁定验证）</th><th>模型</th></tr>
<tr><td>初赛 · 任务一 FULL 口径</td><td>ACC = F1 = ROC-AUC = 1.0000</td><td>Random Forest</td></tr>
<tr><td>初赛 · 任务一严格口径</td><td>ACC = 0.7685，ROC-AUC = 0.8355</td><td>Logistic Regression + SHAP</td></tr>
<tr><td>初赛 · 任务三</td><td>三分类概率输出（Average / Good / Excellent）</td><td>ExtraTrees FULL / NO_DIRECT 融合</td></tr>
<tr><td>复赛 · 任务一（睡眠）</td><td>R² = 0.6345，RMSE = 0.9450</td><td>Ridge（α = 1）</td></tr>
<tr><td>复赛 · 任务二（效率）</td><td>调整 R² = 0.5942，RMSE = 0.7186</td><td>Ridge（α = 10）</td></tr>
<tr><td>复赛 · 任务三（综合健康）</td><td><b>R² = 0.9829</b>，RMSE = 1.6131</td><td>Ridge 主干 + 三路残差加权集成</td></tr>
</table>

---

## 五、目录结构

```
.
├── README.md / .gitignore / LICENSE   # 项目总览、版本控制规则与开源许可
├── ADATA260065.pdf                 # 复赛论文：基于正则化回归与残差集成的
│                                   #   健康生活方式多任务评分预测研究
├── ADATA2600655初赛.pdf/.docx      # 初赛论文（含中文摘要版）
│
├── 2026年第五届"创新杯"大学生大数据挑战赛初赛题目/
│   ├── A题/  B题/                  # 赛题 PDF 与数据集
│   ├── 任务一/                     # 二分类判别建模
│   │   └── 任务一正式图件包_最终同步版/
│   │       ├── figures/            # PDF / SVG / 600 dpi PNG 正式图件
│   │       ├── data/               # 图件对应的最终数据（CV 汇总、SHAP、校准）
│   │       ├── scripts/plot_task1_final.py   # 一键重绘脚本
│   │       └── manifest.json       # 来源与哈希记录
│   └── 任务三/                     # 三分类综合健康类别预测
│       ├── code/                   # train_task3.py / predict_task3.py
│       ├── models/                 # FULL 与 NO_DIRECT 模型（*.joblib，不入库，可复现）
│       ├── results/                # 折外预测、锁定留出预测、评价指标
│       ├── figures/                # 实验图件
│       ├── configs/                # 标签映射、随机种子、划分与融合参数
│       └── docs/                   # 建模报告
│
├── 2026年第五届"创新杯"大学生大数据挑战赛复赛题目/
│   ├── 复赛A题/  复赛B题/          # 复赛赛题 PDF 与数据集
│   └── 任务三/                     # 综合健康评分回归（Health_Score，指标 R²）
│       ├── train_task3.py / predict_task3.py   # 训练与推理入口
│       ├── src/modeling.py         # 特征工程与残差集成模型定义
│       ├── models/                 # 最终模型与锁定验证复核模型（不入库，可复现）
│       ├── predictions/            # OOF 预测与锁定验证预测
│       ├── figures/                # 报告所用真实实验图件
│       └── results/                # 消融、折间指标、置换重要性
│
└── 复赛A题模型交付包/              # DATA2600655 复赛 A 题官方模型支撑材料（完整源码）
    ├── README.md                   # 交付包总览（含各任务导航）
    ├── 01_task1/                 # 睡眠质量评分回归：代码、模型、预测、图件、建模报告
    ├── 02_task2/                 # 工作效率评分回归与驱动因素分析：同上结构
    ├── 03_task3/                 # 综合健康评分回归：同上结构
    ├── 04_paper_figures/         # 正式论文修订版 25 张纯矢量 SVG 与统一制图脚本
    └── DATA2600655 AI 使用说明.pdf   # AI 工具使用详情说明
```

> **说明**：受 GitHub 单文件 100 MB 限制，所有 `*.joblib` 模型权重未纳入版本库；每个任务目录均提供完整训练脚本，按"复现指南"执行即可原位重新生成。竞赛承诺书（含个人签名）亦未公开。

---

## 六、复现指南

### 6.1 环境

- 初赛任务三：Python 3.11，NumPy 2.3.5、pandas 2.2.3、scikit-learn 1.8.0、Matplotlib 3.10.8、seaborn 0.13.2、joblib 1.5.3；
- 复赛任务三：推荐 Python 3.12，依赖见对应目录 `requirements.txt`。

```bash
python -m pip install -r requirements.txt
```

### 6.2 直接推理

**复赛任务三**（综合健康评分回归）：

```bash
cd "2026年第五届"创新杯"大学生大数据挑战赛复赛题目/任务三"
python predict_task3.py \
  --input data/task3_test_input_example.csv \
  --output predictions/task3_test_output_example.csv
```

输入 CSV 需包含 `Person_ID` 及训练所需原始特征，可不含 `Health_Score` 与 `Wellness_Category`；输出至少包含 `Person_ID` 与 `Predicted_Health_Score`，若输入含真值则同时保留 `True_Health_Score`。

**初赛任务三**（三分类健康类别）：

```bash
cd "2026年第五届"创新杯"大学生大数据挑战赛初赛题目/任务三"
python code/predict_task3.py \
  --input data/A题数据集.csv \
  --output results/task3_predictions.csv \
  --mode competition        # competition 为 FULL 竞赛模型；explanation 为去泄露解释模型
```

### 6.3 完整复现

将原始数据 `A题数据集.csv` 置于对应任务目录的 `data/` 下，执行：

```bash
python train_task3.py       # 或 code/train_task3.py（初赛任务三）
```

脚本以固定随机种子 2026 重新生成模型、OOF 预测、锁定验证预测、指标表与全部图件；重新训练产物统一写入 `reproduce_results/`（已被 `.gitignore` 排除）。

### 6.4 复赛 A 题模型交付包（完整源码）

[`复赛A题模型交付包/`](复赛A题模型交付包/README.md) 为按赛事要求提交的官方模型支撑材料（对应仓库根目录压缩包 `ADATA2600655model.zip` 的解包版本），包含复赛三项任务各自的完整工程：`task*_model_report.pdf` 建模报告、已训练模型、训练与预测代码、`requirements.txt`、固定验证预测、模型比较结果及配套 PNG 图件。每个任务目录下：

```bash
python -m pip install -r requirements.txt          # 安装该任务固定依赖
python code/predict_task1.py --input data/locked_test_inputs.csv --output my_predictions.csv   # 直接预测
python code/train_task1.py --output model_rebuild  # 复现拟合（任务一示例）
```

三个任务均保留 `plotting/` 目录（绘图脚本 + 固定图源），`04_paper_figures/` 另存正式论文修订版 25 张纯矢量 SVG 及统一制图脚本 `generate_revised_figures.py`；新版图件仅基于既有固定图源重新排版，不改变模型、样本划分、预测结果或评价指标。交付包内 `*.joblib` 模型权重同样遵循版本控制排除规则，可经上述训练脚本原位重建。

---

## 七、主要结论与边界声明

1. **正则化模型适合可加结构任务**：在严格验证边界下，Ridge 回归已能稳定捕捉睡眠质量与工作效率的主结构；
2. **残差集成可提取非线性增益**：综合健康评分中存在超出线性主结构的稳定非线性信息，经残差学习后锁定验证 R² 达 0.9829；
3. **预测依赖不等于因果效应**：置换重要性排序反映模型依赖结构（如运动频率、压力、情绪之于睡眠；健康老龄化评分、心血管风险、BMI 之于综合健康），不构成健康管理干预的因果证据；
4. **评价属于内部泛化估计**：开发集与锁定验证集来自同一抽样总体，结论不外推为时间序列预测或健康管理效果。

---

## 八、引用与许可

- 论文：[`ADATA260065.pdf`](ADATA260065.pdf)（复赛）、[`ADATA2600655初赛.pdf`](ADATA2600655初赛.pdf)（初赛）；
- 本仓库以 [MIT License](LICENSE) 开源，供学习交流使用；
- 数据集为赛事主办方提供的竞赛数据，仅限非商业的学术交流用途。

<div align="center">

**If this repository helps your research, please consider giving it a ⭐**

</div>
