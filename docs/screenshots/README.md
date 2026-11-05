# 系统截图索引

| 截图 | 内容 |
| --- | --- |
| `01_detection_single.png` | 单张图像检测：原图、标注图、7 处病灶、红色预警、检测明细 |
| `02_detection_batch.png` | 批量检测：3 张图像、类别分布柱状图、逐文件结果、ZIP 下载 |
| `03_detection_video.png` | 视频流检测：H.264 标注视频、逐秒曲线、峰值病灶与类别统计 |
| `04_records.png` | 数据查询：筛选条件、KPI、趋势/类别/置信度/终端图表、记录表、预警通报 |
| `05_assistant_context.png` | AI 助手：任务上下文（类别、置信度、严重度、标注缩略图）、模型状态 |
| `06_assistant_answer.png` | 本地 DeepSeek-R1-14B 的 RAG 回答与右侧知识库来源 |
| `07_assistant_report.png` | 一键生成的病害诊断与防治报告（Markdown） |

实验图表位于 `reports/figures/`：

| 图表 | 内容 |
| --- | --- |
| `training_curves.png` | 400 轮 mAP 与损失曲线 |
| `test_BoxPR_curve.png` | 测试集 P-R 曲线（mAP@0.5） |
| `test_BoxF1_curve.png` | F1-置信度曲线（峰值 0.91 @ 0.573） |
| `test_confusion_matrix.png` | 混淆矩阵 |
| `dataset_distribution.png` | 数据划分与类别分布 |
| `docs/figures/DFD.png` | 论文系统分层数据流图（对照） |
| `docs/figures/structure.png` | 论文系统功能结构图（对照） |
