# scripts

项目脚本目录。写法一律遵循 [docs/脚本规范.md](../docs/脚本规范.md)。

## 公共件

- `_ui.py`：配置回显 `echo_config`、进度条 `progress`（与 ultralytics 同款格式，未装 tqdm 自动降级）、
  日志 `Tee`、报错 `die`。新脚本直接 `from _ui import ...`，别各写一套。
- `templates/script_template.py`：新脚本模板，复制后只改头注释、可调参数区、`run_one()`。

## 已有脚本

| 脚本 | 干什么 |
|---|---|
| `sync_artifacts.py` | 检查 `artifacts/` 是否还是 Git LFS 指针；必要时 `git lfs pull`，并校验模型 SHA256 |
| `train_detect.py` | 检测训练入口（`--dry-run` 先看配置；训练必须由本人明确指令触发） |
| `eval_report.py` | val/test 评测，出混淆矩阵与必报指标 |
| `error_analysis.py` | 逐框匹配的错误分解（FP/FN 分类） |
| `video_regression.py` | 跑回归视频，出「手部 ROI 帧占比 / head 检出 / 单帧耗时」三数 |
| `mine_hard_negatives.py` | 从回归视频挖已确认的误检贴片 |
| `build_negative_aug.py` | 把负样本贴片合成进训练图（`datasets/shwd_neg`） |
| `build_head_crops.py` | 从检测数据集裁头部裁片，供两阶段分类器（该方向已关闭） |

## 后续可放

- `download_data.py`：从团队网盘下载数据集；
- `export_onnx.py`：模型导出；
- `check_dataset.py`：数据标签检查；
- `publish_artifacts.py`：把最新一次评测和模型复制到 `artifacts/latest/`；
- `run_demo.py`：本地演示启动。

脚本尽量使用项目根目录相对路径（`ROOT / ...`），不要写死某台电脑的盘符。
