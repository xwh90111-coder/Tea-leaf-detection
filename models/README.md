# 模型文件说明

| 文件 | 是否包含 | 大小 | 说明 |
| --- | --- | --- | --- |
| `models/best.pt` | ✅ 是 | 5.36 MB | 复现训练的 YOLOv12n 部署权重（已剥离优化器） |
| `models/bge-small-zh-v1.5/` | ❌ 否 | ~96 MB | D2 知识库中文向量模型 |
| `models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf` | ❌ 否 | 8.37 GB | 本地大模型（GitHub 无法托管） |

## 下载 BGE 向量模型

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir models/bge-small-zh-v1.5
```

或使用 ModelScope：

```bash
modelscope download --model BAAI/bge-small-zh-v1.5 --local_dir models/bge-small-zh-v1.5
```

## 下载 DeepSeek-R1-14B GGUF

```bash
curl -L -C - \
  -o models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf \
  "https://modelscope.cn/models/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF/resolve/master/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"
```

校验：文件大小应为 **8,988,109,984 字节**（8.37 GiB）。

## YOLOv12n 预训练权重

`yolo12n.pt`（COCO 预训练）会在首次训练时由 Ultralytics 自动下载；也可手动放到仓库根目录。
复现得到的最终权重已包含在 `models/best.pt`，推理/评估无需预训练权重。
