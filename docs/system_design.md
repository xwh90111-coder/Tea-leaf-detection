# 系统设计与论文对应关系

## 1. 总体架构

```
┌──────────────────────────────────────────────────────────────┐
│                    Dash Web 前端（单页三模块）                │
│  检测功能页        数据查询页          AI 助手分析页          │
└──────────────┬───────────────┬────────────────┬──────────────┘
               │               │                │
       ┌───────▼──────┐ ┌──────▼───────┐ ┌──────▼──────────┐
       │ YOLOv12n 推理 │ │ D1 查询/统计  │ │ RAG 编排          │
       │ 单张/批量/视频 │ │ Excel 导出    │ │ 检索 + 提示词      │
       └───────┬──────┘ └──────┬───────┘ └──────┬──────────┘
               │               │                │
       ┌───────▼───────────────▼──────┐  ┌──────▼──────────┐
       │ D1 病害数据库（SQLite）        │  │ D2 茶叶知识库     │
       │ tasks / detections / alerts   │  │ 58 片段向量索引   │
       │ chat_messages                 │  │ BGE-small-zh     │
       └───────────────────────────────┘  └──────┬──────────┘
                                                 │
                                        ┌────────▼─────────┐
                                        │ DeepSeek-R1-14B   │
                                        │ 本地 GGUF/llama.cpp│
                                        └──────────────────┘
```

## 2. 与论文的对应

| 论文描述 | 复现实现 |
| --- | --- |
| YOLOv12 轻量化检测核心 | `ultralytics` YOLOv12n，640×640，3 类，参数量 2,557,313，6.3 GFLOPs |
| 单张 / 批量 / 视频流检测 | `teasys/detector.py` + `pages/detection.py` |
| 检测结果与预警通报 | `detect_image()` 生成标注图、严重程度、预警等级并写 D1 |
| D1 病害数据库（结构化） | SQLite：`tasks`、`detections`、`alerts`、`chat_messages` |
| D2 茶叶知识库（向量存储） | `knowledge/` 文档 → 分块 → BGE 向量 → 本地索引；BM25 + 向量 RRF 融合 |
| 本地 DeepSeek-R1-14B | `teasys/llm.py`，默认加载 Q4_K_M GGUF（llama.cpp），完全离线 |
| 检索增强生成（RAG） | `teasys/assistant.py`：检索 → 提示词 → 生成 → 来源引用 |
| 数据查询与图表分析 | 时间/类别/终端/类型/置信度/关键词筛选，趋势、分布、直方图 |
| Excel 报表导出 | `db.export_excel()`，任务汇总 + 检测明细双工作表 |
| Dash 交互平台 | Dash 4.5 + 自定义 CSS，三页路由（`/detection`、`/records`、`/assistant`） |
| IP65/67 Basler 工业相机 | **不在复现范围（硬件落地）**；终端 ID 模拟为 `CAM-A2A5328-01`，图像通过上传/数据集输入 |
| 模型训练与评估 | `src/train.py`、`src/evaluate.py`、`src/benchmark.py`、`src/plot_training.py` |
| 数据集 8:1:1 | 原始划分为 2178/272/273；`src/split_dataset.py` 重新分层抽样得到 2179/272/272，验证可复现性 |

## 3. D1 数据库

### tasks（任务）
| 字段 | 说明 |
| --- | --- |
| id | 12 位任务 ID |
| batch_id | 批量任务分组 |
| task_type | single / batch / video |
| source_name | 原始文件名 |
| terminal_id | 采集终端（模拟相机编号） |
| total_objects | 病灶数（视频取峰值帧） |
| class_counts | 各类别数量（JSON） |
| max_confidence / avg_confidence | 置信度统计 |
| severity | 无 / 轻度 / 中度 / 重度 |
| alert_level | none / info / warning / danger |
| alert_message | 预警文本 |
| elapsed_ms | 端到端耗时 |
| annotated_path | 标注图/视频路径 |
| created_at | 检测时间 |

### detections（病灶明细）
`task_id`、`class_id`、`class_name`、`confidence`、`x1/y1/x2/y2`、`area_ratio`、`created_at`。

### alerts（预警）
`task_id`、`level`、`disease`、`message`、`acknowledged`、`created_at`。

### chat_messages（助手历史）
`session_id`、`role`、`content`、`sources`（JSON）、`task_id`、`created_at`。

## 4. D2 知识库与 RAG

1. **语料**：`knowledge/*.md`，11 篇文档，覆盖三种病害的症状、病原、发病规律、农业防治、化学防治、监测测报、气象风险、图像采集与系统规则。
2. **分块**：按二级标题分节，按句切分，块大小 420 字符、重叠 80 字符，共 58 个片段。
3. **向量化**：`BAAI/bge-small-zh-v1.5`（512 维，CLS pooling，L2 归一化），本地推理。
4. **索引**：`data/knowledge_index/{chunks.json, embeddings.npy, index_meta.json}`，文档变更通过指纹自动重建。
5. **检索**：向量余弦 Top-20 + BM25（jieba 分词）Top-20，RRF（k=60）融合，取 Top-4/5。
6. **生成**：系统提示词约束“只依据检测结果与知识片段、不得编造、标注来源编号、区分系统严重度与农学病情指数”。

## 5. 本地大模型部署

- 模型：`DeepSeek-R1-Distill-Qwen-14B`，Q4_K_M GGUF，8.37 GB。
- 运行时：`llama-cpp-python`（CPU，20 线程，n_ctx=8192，mmap）。
- 数据安全：模型文件、知识库、数据库、检测图像全部位于本地，推理不访问外部网络。
- 推理速度：CPU 约 4～6 token/s，单次问答 1～4 分钟；系统界面提供加载提示。
- 兼容后端：可通过环境变量切换 Ollama / OpenAI 兼容服务（`TEA_LLM_PROVIDER`）。

## 6. 严重程度与预警规则

| 条件 | 严重程度 | 预警等级 |
| --- | --- | --- |
| 无病灶 | 无 | none |
| 病灶 ≤2 且面积占比 <3% | 轻度 | info（黄色提示） |
| 病灶 ≤5 且面积占比 <12% | 中度 | warning（橙色预警） |
| 病灶 >5 或面积占比 ≥12% | 重度 | danger（红色预警） |

视频以单帧峰值病灶数判定；批量按文件逐个判定，并汇总批次统计。

## 7. 运行入口

| 路径 | 功能 |
| --- | --- |
| `/detection` | 检测功能页（三个标签页） |
| `/records` | 数据查询页 |
| `/assistant?task=<id>` | AI 助手，自动载入指定任务 |
| `/media/uploads/<file>` | 上传文件访问 |
| `/media/outputs/<file>` | 标注图/视频访问 |
| `/health` | 健康检查 JSON |

## 8. 硬件落地说明

论文中的 Basler ace2 a2A5328-4gmIP67 工业相机、IP65/67 防护、GigE 采集与均匀光源属于硬件实施，不在本次软件复现范围内。系统以 `terminal_id` 模拟采集终端，保留图像/视频输入接口；若接入真实相机，只需将采集帧写入 `data/uploads/` 或直接调用 `detect_image()`/`detect_video()`。
