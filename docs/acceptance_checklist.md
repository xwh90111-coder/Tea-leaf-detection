# 验收清单（次日早晨）

> 建议按顺序检查，每项均给出预期结果。所有路径相对于 `reproduction/`。

## 0. 一键启动

```bash
cd reproduction
python src/app.py
```

- 浏览器访问 <http://127.0.0.1:8050>。
- 正常表现：左侧深绿色导航栏，顶部标题“病害检测”，四个 KPI 卡片，三个功能页可切换。
- 健康检查：<http://127.0.0.1:8050/health> 返回 `{"status":"ok", ...}`。
- 若 8050 端口被占用：`set TEA_PORT=8051` 后重启。

## 1. 检测功能页 `/detection`

### 1.1 单张图像
1. 点击“单张图像”标签。
2. 上传 `data/samples/sample_severe.jpg`。
3. 置信度保持 0.25，点击“开始检测”。

预期：
- 左侧显示原图，右侧显示带中文类别与置信度的标注图。
- 顶部出现“红色预警 / 病灶 7 处 / 严重程度：重度”。
- 检测明细表列出类别、置信度、边界框、面积占比。
- 状态栏显示“检测完成 · 任务 xxxx”。
- 点击“生成 AI 诊断报告 →”跳转 AI 助手并自动载入该任务。

### 1.2 批量处理
1. 切换到“批量处理”。
2. 多选 `data/samples/sample_batch_1.jpg`、`sample_batch_2.jpg`、`sample_mild.jpg`。
3. 点击“开始批量检测”。

预期：汇总卡片显示 3 张、约 10 处病灶；类别分布柱状图；逐文件表格；“下载标注结果 ZIP”可用。

### 1.3 视频流检测
1. 切换到“视频流检测”。
2. 上传 `data/samples/tea_garden_demo.mp4`（8 秒）。
3. 点击“开始视频检测”。

预期：生成标注视频（可播放，H.264）、峰值病灶 7 处、逐秒曲线、类别峰值表；“下载标注视频”可用。

## 2. 数据查询页 `/records`

预期：
- KPI：检测任务、检出病灶、平均置信度、红色预警。
- 四张图：近 14 天趋势、类别分布、置信度直方图、终端分布。
- 检测记录表（时间、任务、类型、文件、类别、病灶数、最高置信度、严重度、预警、耗时）。
- 可翻页；点击任务 ID 跳转 AI 助手。
- “导出 Excel”下载 `.xlsx`，包含“任务汇总”“检测明细”两个工作表。
- 底部“预警通报”列出按严重程度生成的预警。

## 3. AI 助手页 `/assistant`

### 3.1 任务上下文
- 从检测页跳转时自动载入任务；也可在左侧下拉框选择任务并点击“载入任务”。
- 预期：左侧显示任务 ID、文件、类别、严重度、置信度、标注缩略图。

### 3.2 问答（本地 14B）
1. 输入“这株茶树病害严重吗？应该怎么防治？”。
2. 点击“发送”。

预期：
- 出现加载动画，提示“本地 DeepSeek-R1-14B 推理中可能需要 1～4 分钟”。
- 返回 Markdown 格式的诊断与防治建议，正文引用知识库片段 [1][2]。
- 右侧“参考来源”显示 4～5 条带相似度的片段。
- 底部“历史对话”新增一条会话，可点击回看。

> 若模型尚未加载，首次请求会额外等待约 8～20 秒（加载 8.37 GB GGUF）。

### 3.3 报告生成
点击“生成综合分析报告”。

预期：返回包含“检测概况 / 结果解读 / 病害诊断 / 防治建议 / 预警与后续监测 / 参考依据”的报告，并下载 `.md` 文件到 `data/reports/`。

### 3.4 文件理解 & 趋势分析
- 上传任意 TXT/CSV/JSON/MD 检测报告，助手会结合知识库解读。
- 点击“最近一周趋势”，助手汇总 D1 最近 7 天数据后给出趋势与监测建议。

## 4. 实验复现证据

| 文件 | 内容 |
| --- | --- |
| `runs/detect/yolo12n_tea_400/results.csv` | 400 轮训练日志 |
| `models/best.pt` | 复现部署权重（5.36 MB） |
| `reports/metrics_test.json` | 测试集总体 + 逐类指标 |
| `reports/per_class_test.csv` | 逐类 P/R/F1/mAP |
| `reports/benchmark.json` | 参数量、GFLOPs、文件大小、FPS |
| `reports/dataset_stats.json` | 数据划分、类别均衡、标注质量、泄漏检查 |
| `reports/figures/` | 训练曲线、PR/F1/混淆矩阵、数据分布 |
| `docs/screenshots/` | 单张/批量/视频/查询/助手截图 |
| `docs/reproduction_report.md` | 完整复现报告 |

命令复核：

```bash
python src/evaluate.py --split test
python src/benchmark.py
python src/dataset_report.py
python src/plot_training.py
```

## 5. 数据库检查

```bash
python -c "import sqlite3;c=sqlite3.connect('data/tea_system.db');print(c.execute('select count(*) from tasks').fetchone());print(c.execute('select count(*) from detections').fetchone())"
```

预期：任务数 ≥ 5，检测明细 ≥ 15（演示数据为 1 张单图 + 3 张批量 + 1 段视频，共 18 处病灶）。

## 6. 已知限制

1. **硬件未复现**：Basler 相机、IP65/67 防护、GigE 采集、均匀光源属于硬件实施；系统用终端编号模拟，输入为上传图像/视频。
2. **大模型为 CPU 推理**：14B Q4_K_M 约 4～6 token/s，单次回答 1～4 分钟；如需更快，可切换 Ollama/GPU 后端。
3. **视频流为文件流**：支持上传视频逐帧检测，未接入实时 RTSP/相机流（属硬件/部署环节）。
4. **严重程度规则**：基于病灶数与面积占比的系统规则，不等同于农学病情指数。
5. **标注复核**：论文中“多人交叉验证、剔除 IoU<0.85 标注框”依赖人工，复现中只能做格式、范围、重复与缺失检查（结果见 `reports/dataset_stats.json`，无异常）。
6. **训练轮数**：论文表格 300、图注 400，本复现按 400 执行；batch 采用论文表格的 16。

## 7. 常见问题

- **页面空白 / 回调报错**：查看服务端控制台输出；本工程已关闭 `debug`，错误会打印到终端。
- **图片不显示**：确认 `data/outputs/` 下文件存在，且通过 `/media/outputs/...` 访问。
- **大模型加载慢或失败**：确认 `models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf` 完整（8,988,109,984 字节）；内存不足时关闭其他程序。
- **端口占用**：`set TEA_PORT=8051 & python src/app.py`。
- **训练续跑**：`python src/train.py --resume --name yolo12n_tea_400`。
