# models

模型版本登记目录。**只放元数据，不放权重文件。**

`manifest.json` 记录：

- 模型版本；
- 对应代码 commit；
- 类别；
- 来源 run；
- 指标入口；
- SHA256；
- `artifacts/` 路径。

当前已登记 `v0.1.0`：

- 权重：`artifacts/weights/helmet_v0.1.0.pt`
- 评测结果：`artifacts/latest/runs/`
- 类别：`head` / `helmet`
- 校验：`python scripts/sync_artifacts.py`

指标不在此文件重复编造，统一看 `artifacts/latest/runs` 和 `docs/实验记录.md`。
