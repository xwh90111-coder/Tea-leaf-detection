# 数据集说明

本仓库不包含茶叶病害图像（体积与版权原因），请自行准备数据集。

## 期望结构

```
tea-leaf-disease/
├─ train/images/*.jpg
├─ train/labels/*.txt
├─ valid/images/*.jpg
├─ valid/labels/*.txt
├─ test/images/*.jpg
└─ test/labels/*.txt
```

论文配置：2723 张图像，训练/验证/测试 = 2178/272/273，3 类：

| id | 英文名 | 中文名 |
| --- | --- | --- |
| 0 | algal-leaf-spot | 茶藻斑病 |
| 1 | brown-blight | 茶褐枯病 |
| 2 | grey-blight | 茶灰枯病 |

## 配置

编辑 `data/tea.yaml` 的 `path`，指向数据集根目录；或把数据集放到仓库上一级的
`datasets/tea-leaf-disease`，保持默认相对路径即可。

```bash
python src/dataset_report.py   # 检查划分、类别均衡、标注质量
```

`data/samples/` 内含 4 张演示图像和 1 段 8 秒演示视频，仅用于 UI 演示。
