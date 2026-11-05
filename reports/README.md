# 复现结果文件

| 文件 | 内容 |
| --- | --- |
| `metrics_test.json` | 测试集总体与逐类指标（conf=0.001 与 conf=0.25 两种口径） |
| `per_class_test.csv` | 逐类 Precision / Recall / mAP@0.5 / mAP@0.5:0.95 |
| `benchmark.json` | 参数量、GFLOPs、模型文件大小、前后处理耗时、FPS、batch 吞吐 |
| `dataset_stats.json` | 2723 张数据的划分、类别均衡、标注质量与跨集重复检查 |
| `train_meta.json` | 训练配置、轮数、最佳轮次与验证指标 |
| `figures/` | 训练曲线、P-R/F1/P/R 曲线、混淆矩阵、数据分布 |
| `llm_test_answer.md` | 本地 14B RAG 的一次完整回答样例 |
| `demo_tasks.json` | 演示数据库中的单张任务 ID |

> 完整的 400 轮训练日志在 `runs/detect/yolo12n_tea_400/results.csv`。
