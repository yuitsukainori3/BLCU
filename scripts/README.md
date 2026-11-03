# scripts

项目脚本目录。

## 已有脚本

- `sync_artifacts.py`：检查 `artifacts/` 是否还是 Git LFS 指针；必要时执行 `git lfs pull`，并校验模型 SHA256。

## 后续可放

- `download_data.py`：从团队网盘下载数据集；
- `export_onnx.py`：模型导出；
- `check_dataset.py`：数据标签检查；
- `publish_artifacts.py`：把最新一次评测和模型复制到 `artifacts/latest/`；
- `run_demo.py`：本地演示启动。

脚本尽量使用项目根目录相对路径，不要写死某台电脑的盘符。
