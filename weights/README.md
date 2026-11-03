# weights

本地权重缓存目录。**权重不直接进 Git**，共享权重放在 `artifacts/weights/`（Git LFS）。

使用方式：

1. `git lfs install`（首次）；
2. `git pull && git lfs pull`；
3. 从 `artifacts/weights/` 复制需要的权重到本目录，例如：
   ```text
   weights/helmet_v0.1.0.pt
   ```
4. 用 `python scripts/sync_artifacts.py` 校验 SHA256；
5. 应用通过配置读取权重路径。

共享版本和 SHA256 见 `models/manifest.json`。不要提交 `*.pt`、`*.onnx`、`*.engine` 到普通 Git。
