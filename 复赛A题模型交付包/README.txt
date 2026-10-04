DATA2600655 复赛 A 题模型支撑材料

目录
01_task1  睡眠质量评分回归预测
02_task2  工作效率评分回归与驱动因素分析
03_task3  综合健康评分回归预测
04_paper_figures  正式论文修订版矢量图与统一制图脚本
DATA2600655 AI 使用说明.pdf  AI 工具使用详情说明

每个任务均包含：
1. task*_model_report.pdf：建模思路、验证结果、运行环境与模型调用说明；
2. 已训练最终模型；
3. 训练与预测代码；
4. requirements.txt；
5. 固定验证预测、模型比较与必要结果文件；
6. figures/：Python 生成的建模报告配套 PNG 图件。

运行方法
进入对应任务目录，先安装 requirements.txt，再按该目录 README.txt 或 task*_model_report.pdf 中的命令运行。

绘图复现
三个任务均保留 plotting/ 目录，包含建模报告图件的绘图脚本与固定图源。
04_paper_figures/ 另外保存正式论文修订版的 25 张纯矢量 SVG 及统一制图脚本；新版图件基于三个任务现有固定图源重新排版，不改变模型、样本划分、预测结果或评价指标。
