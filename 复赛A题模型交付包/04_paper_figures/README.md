# 04_paper_figures：正式论文修订版矢量图

> 队伍编号：DATA2600655

<div align="right">

[← 交付包总览](../README.md) · [↑ 仓库根 README](../../README.md)

</div>

本目录保存正式论文修订版的 **25 张纯矢量 SVG** 图件与统一制图脚本。

## 一、图源说明

新版图件读取 `01_task1`、`02_task2`、`03_task3` 各自 `plotting/source_data/` 中的固定 CSV 图源，仅调整字体、配色、布局和中文标签；**不重新训练模型，也不改变样本划分、预测结果或评价指标**。

三个任务原有 `figures/` 目录继续对应各自的建模报告；本目录中的 SVG 用于正式论文修订版，两套图件的数据与指标口径完全一致。

## 二、目录内容

| 路径 | 说明 |
| --- | --- |
| `revised_svg/` | 正式论文修订版的 25 张纯矢量 SVG |
| `generate_revised_figures.py` | 统一制图脚本 |
| `requirements.txt` | 制图依赖 |

## 三、运行方法

1. Windows 环境需安装 Microsoft YaHei、SimHei 等脚本支持的中文字体；
2. 在模型包根目录执行：

```bash
python -m pip install -r 04_paper_figures/requirements.txt
python 04_paper_figures/generate_revised_figures.py --model-root . --output 04_paper_figures/rebuilt_svg
```

3. 正常结束时应生成 25 张不含栅格图像节点的 SVG。

---

*原始交付说明见 `README.txt`，两者内容一致；本文件为规范化文档版本。*
