# 安全帽智检

基于 YOLO11 的工地安全帽佩戴检测系统。仓库用于同步项目代码、模型版本、应用框架和文档。

> 当前状态：YOLO11 训练、评测路线已建立；本地视频展示原型和时序/分区风险逻辑已实现。当前没有实时摄像头 API、持久化数据库或消息推送。

## 目录结构

```text
BLCU/
├─ README.md
├─ AGENTS.md
├─ .gitignore
├─ backend/              # 后续 API 服务
├─ frontend/             # Streamlit 本地演示入口 app.py
├─ ml/                   # 视频检测、跟踪关联、事件聚合与风险计算
├─ configs/              # 数据/模型/阈值配置
├─ docs/                 # 项目文档、训练指南、数据分析
├─ models/               # 模型登记表（不放权重）
├─ artifacts/            # 共享评测产物和模型（Git LFS）
├─ datasets/             # 数据目录（内容不进 Git，仅保留 README）
├─ weights/              # 权重目录（内容不进 Git，仅保留 README）
├─ scripts/              # 预留：下载、转换、部署脚本
├─ tests/                # 预留：跨模块测试
├─ samples/              # 预留：小体积示例图/视频
├─ runs/                 # 训练输出（不进 Git）
└─ ultralytics-8.3.163/  # 目前保留的上游源码
```

## 快速开始

本仓库的 `artifacts/` 使用 Git LFS 保存评测图和模型。**第一次 clone 前先安装 Git LFS**：

```bash
git lfs install
git clone https://github.com/yuitsukainori3/BLCU.git
cd BLCU
git lfs pull
python -m venv .venv
```

如果已经 clone 过，执行：

```bash
git pull
git lfs pull
python scripts/sync_artifacts.py
```

Windows：

```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS：

```bash
source .venv/bin/activate
```

先按自己的显卡安装 PyTorch（CUDA/CPU/MPS 以 [PyTorch 官网](https://pytorch.org/get-started/locally/) 为准），再安装项目依赖：

```bash
python -m pip install -r requirements.txt
```

应用和训练脚本会优先从仓库内 `ultralytics-8.3.163/ultralytics-8.3.163/` 加载固定版本。

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
yolo checks
```

## 视频展示原型

在 Python 环境中安装仓库根目录的 `requirements.txt`，准备 SHWD 两类数据和有效的 YOLO11 权重后：

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

页面支持上传本地视频、960 高分辨率 head/helmet 推理、COCO 人体跟踪、滑窗事件去抖、疑似事件列表和分区风险摘要。视频事件只作为人工复核候选。

改进训练入口：

```powershell
python scripts/train_improved.py --data datasets/shwd/data.yaml --model yolo11s.pt --imgsz 960 --epochs 120
```

如需试验裸头类重采样（只作用于训练图片，不复制或调整验证/测试集）：

```powershell
python scripts/build_balanced_dataset.py --data datasets/shwd/data.yaml --repeat 2
python scripts/train_improved.py --data runs/data/head_repeated/helmet_head_repeated.yaml --name helmet_head_repeated
```

每项训练改动需与当前基线在相同验证/测试划分、相同阈值口径下对比；重复采样可能过拟合，不能预设它一定提升。

## 数据和权重

数据集和原始训练输出**不进 Git**：

- 数据集放在 `datasets/`，说明见 [datasets/README.md](datasets/README.md)；
- 原始 `runs/` 和 `weights/` 不提交，避免仓库膨胀；
- 要共享的评测图和模型放在 `artifacts/`，由 Git LFS 管理，说明见 [artifacts/README.md](artifacts/README.md)；
- 模型版本、指标和 SHA256 登记在 [models/manifest.json](models/manifest.json)。

当前共享版本：

- 模型：`artifacts/weights/helmet_v0.1.0.pt`
- 评测结果：`artifacts/latest/runs/`
- 校验脚本：`python scripts/sync_artifacts.py`

如果 `artifacts/` 下的文件只有约 130 字节，说明拿到的是 LFS 指针，执行：

```bash
git lfs pull
```

## 文档

- [项目设计文档](docs/安全帽智检_项目文档_V0.1.docx)
- [项目初期方案](docs/项目初期方案.md)（给指导老师的初期方案，简版；[Word 版](docs/安全帽智检项目初期方案.docx)）
- [标签规则](docs/标签规则.md)（head / helmet 两类标注规范，v1.2）
- [YOLO11 安全帽模型训练指南](docs/YOLO11安全帽模型训练指南.md)
- [通用版训练指南](docs/YOLO11安全帽模型训练指南_通用版.md)
- [两路检测方案](docs/两路检测方案.md)（COCO 人体 + 自训头部）
- [展示系统使用说明](docs/展示系统使用说明.md)
- [本地小样本试跑记录](docs/本地小样本试跑记录.md)（仅流程验证，不是 SHWD 新指标）
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
