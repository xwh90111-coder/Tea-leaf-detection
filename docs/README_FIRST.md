# 验收先看这里（TL;DR）

## 1. 启动系统

```bash
cd Tea-leaf-detection   # 仓库根目录
python src/app.py
```

浏览器打开 <http://127.0.0.1:8050>。若已启动，直接访问即可：

## 2. 三个页面

| 页面 | 地址 | 看什么 |
| --- | --- | --- |
| 病害检测 | `/detection` | 单张 / 批量 / 视频三种模式；标注图、置信度、严重度、预警；结果写 D1 |
| 数据查询 | `/records` | 5 条演示任务、18 处病灶、趋势/类别/置信度/终端图表、Excel 导出、预警列表 |
| AI 助手 | `/assistant?task=d6216aedeed1` | 载入检测任务；本地 DeepSeek-R1-14B 问答；知识库来源；历史会话；一键报告 |

演示数据（已写入 D1）：

- 单张：`sample_severe.jpg`，5 处病灶，中度 / 橙色预警
- 批量：3 张，8 处病灶
- 视频：`tea_garden_demo.mp4`（8 秒），峰值 5 处
- AI 会话：1 次真实 14B 问答 + 1 份完整诊断报告

## 3. 复现结果（测试集 273 张）

| 指标 | 论文 | 本复现 |
| --- | --- | --- |
| mAP@0.5 | 0.955 | **0.9738** |
| mAP@0.5:0.95 | 0.8457 | **0.9141** |
| 参数量 | 2,557,313 | **2,557,313** |
| GFLOPs | 6.3 | **6.32** |
| 模型文件 | 5.3 MB | **5.36 MB** |
| FPS（batch=1） | 189.30 | 70.76 |
| 吞吐（batch=4） | — | 349.77 img/s |

精度高于论文，参数/计算量/文件大小完全对齐；单流 FPS 低于论文，但批处理吞吐更高，原因见 `docs/reproduction_report.md` 第 6 节。

## 4. 证据文件

| 文件 | 内容 |
| --- | --- |
| `docs/reproduction_report.md` | 完整复现报告（含论文表格差异分析） |
| `docs/screenshots/` | 01 单张 / 02 批量 / 03 视频 / 04 查询 / 05 上下文 / 06 问答 / 07 报告 |
| `reports/metrics_test.json` | 测试集总体 + 逐类指标（conf=0.001 与 0.25 两种口径） |
| `reports/benchmark.json` | 参数量、GFLOPs、文件大小、FPS、批处理吞吐 |
| `reports/dataset_stats.json` | 2723 张划分、类别均衡、标注质量、无跨集重复 |
| `reports/figures/` | 训练曲线、PR/F1/混淆矩阵、数据分布 |
| `runs/detect/yolo12n_tea_400/results.csv` | 400 轮训练日志 |
| `models/best.pt` | 复现部署权重（5.36 MB，已剥离优化器） |
| `models/best.pt` | 部署权重（已剥离优化器，5.36 MB） |
| `docs/system_design.md` | 系统架构、D1/D2、RAG、与论文逐条对应 |
| `docs/acceptance_checklist.md` | 逐项验收步骤与已知限制 |

## 5. 快速复核命令

```bash
python scripts/self_check.py                 # 11 项自检
python src/evaluate.py --split test          # 测试集评估
python src/benchmark.py --weights models/best.pt
python src/dataset_report.py
python src/plot_training.py
```

## 6. 说明

- **硬件未落地**：Basler 相机、GigE、IP65/67、光源属于硬件实施；系统用 `CAM-A2A5328-01` 模拟终端，输入为上传图像/视频。
- **本地大模型**：DeepSeek-R1-Distill-Qwen-14B Q4_K_M（8.37 GB），llama.cpp CPU 推理，约 4～6 token/s，单次回答 1～4 分钟；完全离线。
- **论文表 4 差异**：论文表 4 的全类 mAP@0.5=0.9120 与摘要 0.955 不一致；复现发现 conf=0.25 时得到相近的 0.9456，详见报告 5.3 节。
- **训练配置**：400 epochs、batch 16（论文表格写 300/16，图注写 400；作者原始 args 为 400/8）。
