# datasets

数据集目录。**数据内容不进 Git**，只保留本说明。

建议结构：

```text
datasets/
  helmet/
    images/train|val|test/
    labels/train|val|test/
    data.yaml
```

数据来源、许可证、类别统计和校验值请记录在 `docs/` 或团队网盘说明中。

当前团队现有的 `Hard Hat Workers v14` 数据只有 `head` 一类，不能直接训练安全帽合规模型；具体见：

- [YOLO11 安全帽模型训练指南](../docs/YOLO11安全帽模型训练指南.md)
