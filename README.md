# 安全帽智检

基于 YOLO11 的工地安全帽佩戴检测系统。仓库用于同步项目代码、模型版本、应用框架和文档。

> 当前状态：方案/原型阶段。应用框架尚未选型，`backend/`、`frontend/`、`ml/`、`configs/` 等目录先预留位置；Ultralytics 源码暂时保留在 `ultralytics-8.3.163/ultralytics-8.3.163/`，后续再决定是否改为 PyPI 依赖或 `vendor/`。

## 目录结构

```text
BLCU/
├─ README.md
├─ AGENTS.md
├─ .gitignore
├─ backend/              # 预留：后端 API / 推理服务
├─ frontend/             # 预留：可视化界面
├─ ml/                   # 预留：训练、评估、推理、跟踪、业务规则
├─ configs/              # 预留：数据/模型/阈值配置
├─ docs/                 # 项目文档、训练指南、数据分析
├─ models/               # 模型登记表（不放权重）
├─ datasets/             # 数据目录（内容不进 Git，仅保留 README）
├─ weights/              # 权重目录（内容不进 Git，仅保留 README）
├─ scripts/              # 预留：下载、转换、部署脚本
├─ tests/                # 预留：跨模块测试
├─ samples/              # 预留：小体积示例图/视频
├─ runs/                 # 训练输出（不进 Git）
└─ ultralytics-8.3.163/  # 目前保留的上游源码
```

## 快速开始

```bash
git clone https://github.com/yuitsukainori3/BLCU.git
cd BLCU
python -m venv .venv
```

Windows：

```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS：

```bash
source .venv/bin/activate
```

先按自己的显卡安装 PyTorch（CUDA/CPU/MPS 以 [PyTorch 官网](https://pytorch.org/get-started/locally/) 为准），再安装 Ultralytics 源码：

```bash
pip install -e ultralytics-8.3.163/ultralytics-8.3.163
```

验证：

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
yolo checks
```

后端、前端和 `ml/` 的依赖等框架确定后再补；目前不要假设已经能启动完整应用。

## 数据和权重

数据集和模型权重**不进 Git**：

- 数据集放在 `datasets/`，说明见 [datasets/README.md](datasets/README.md)；
- 权重放在 `weights/`，说明见 [weights/README.md](weights/README.md)；
- 模型版本、指标、下载地址登记在 [models/manifest.json](models/manifest.json)；
- 大文件通过团队网盘或 GitHub Releases 共享。

当前 `Train.py` 仍在：

```text
ultralytics-8.3.163/ultralytics-8.3.163/Train.py
```

它默认读取同目录的 `data.yaml`。正式应用框架确定后，训练脚本建议迁到 `ml/train.py`，不要在第三方源码里长期堆业务代码。

## 文档

- [项目设计文档](docs/安全帽智检_项目文档_V0.1.docx)
- [标签规则](docs/标签规则.md)（head / helmet / person 标注规范，v1.1）
- [YOLO11 安全帽模型训练指南](docs/YOLO11安全帽模型训练指南.md)
- [通用版训练指南](docs/YOLO11安全帽模型训练指南_通用版.md)
- [获奖作品选题分析](docs/人工智能应用方向获奖分析与选题建议.md)
- [筛选说明](docs/筛选说明.md)

## 协作约定

- `main` 保持可演示；功能开发开 `feature/*` 分支；
- 提交前确认没有把 `datasets/`、`weights/`、`runs/`、`*.pt`、`*.zip` 加进 Git；
- 模型版本和代码版本要能对应，发布时打 tag、写 Release；
- 不编造指标；没有实测就写“待实测”；
- 上游 Ultralytics 代码尽量少改，业务定制优先放在应用目录。

## 许可证

Ultralytics 源码遵循其仓库内的 `LICENSE`（AGPL-3.0）。使用、修改和分发前请核对许可证要求。应用代码的许可证待团队确定。
