任务一绘图复现目录

code/plot_figures.py：从固定 source_data/ 重新生成任务一 6 张科研图。
source_data/：绘图所需固定 CSV 图源。
运行：python code/plot_figures.py --font C:/Windows/Fonts/msyh.ttc
脚本运行后输出到本目录 figures/；上一级 ../figures/ 保存建模报告采用的固定 PNG 图件。
模型训练与预测不依赖本目录。
