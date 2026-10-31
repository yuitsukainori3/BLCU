# YOLO11 安全帽模型训练指南（通用版）

> 适用对象：团队队友，不限操作系统和显卡型号。  
> 覆盖环境：Windows / Linux / macOS；NVIDIA GPU / Apple Silicon / CPU。  
> 说明：本文不绑定某台电脑，命令中的 `<PROJECT_ROOT>` 请替换成你自己的项目根目录。所有指标以实际训练和测试结果为准。

## 0. 先看结论

训练一个“自己的 YOLO11 安全帽模型”，最短路径是：

```text
准备 helmet/head/person 数据
  → 安装 Python + PyTorch + Ultralytics
  → 写 data.yaml
  → 检查图片、标签、类别
  → 用 YOLO11 预训练权重微调
  → 在测试集上评估
  → 图片/视频/摄像头推理
  → 导出 ONNX
  → 接入业务逻辑
```

三个最容易踩的坑：

1. **数据只有 `head` 一类**：只能训练“头检测器”，不能判断有没有戴安全帽。
2. **把 `head` 改名成 `helmet`**：模型会把裸头也当成安全帽，结果看起来有框，实际是错的。
3. **环境没装对**：`torch.cuda.is_available()` 为 False，训练实际跑在 CPU 上，速度慢很多。

---

## 1. 先确认数据：团队现有数据能不能用

团队当前的 `Hard Hat Workers v14` 数据，据压缩包统计：

| 划分 | 图片数 | head 框数 |
|---|---:|---:|
| train | 4,916 | 18,531 |
| valid | 1,413 | 5,252 |
| test | 706 | 2,641 |
| **合计** | **7,035** | **26,424** |

它的 `data.yaml` 是：

```yaml
nc: 1
names: ['head']
```

所有标签都是 `class 0 = head`。这意味着：

- 可以用来跑通 YOLO11 训练流程；
- **不能**直接训练“安全帽佩戴合规”模型；
- **不要**把 `head` 改名成 `helmet`，否则裸头也会被判为已佩戴。

需要的数据至少是：

```yaml
names:
  0: head
  1: helmet
```

更推荐三类：

```yaml
names:
  0: head
  1: helmet
  2: person
```

其中 `head` 指**未戴安全帽的裸头**，不是“所有头”。

如果队友手里有 `helmet/head/person` 的数据，直接用那份；如果只有当前 `head` 数据，先用来练手，同时找/补一份合规数据。

---

## 2. 环境安装

### 2.1 Python 版本

建议 Python **3.10—3.12**。不要用太新的版本做第一次训练，PyTorch 和相关依赖的轮子可能还没跟上。

Windows：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux / macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

如果 PowerShell 提示禁止运行脚本，只对当前窗口放开：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 2.2 安装 PyTorch

不要照抄别人的 CUDA 版本。先打开 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)，按你的系统、包管理器、显卡选择命令，再复制。

常见情况：

- **NVIDIA GPU**：选择对应的 CUDA 版本。较新的 RTX 50 系需要较新的 CUDA 轮子，旧版 cu118 可能报 `no kernel image is available`。
- **Apple Silicon**：选择 macOS / Pip，安装支持 MPS 的版本。
- **只有 CPU**：选择 CPU 版本，能跑但训练慢。
- **AMD GPU**：Linux 可查 ROCm 版 PyTorch；Windows 上通常先用 CPU。

示例（仅示意，具体以官网为准）：

```bash
# NVIDIA CUDA 示例
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CPU 示例
pip install torch torchvision
```

### 2.3 安装 Ultralytics

```bash
pip install ultralytics
```

如果团队使用源码版：

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -e .
```

第一次用建议直接用 PyPI 版，依赖更简单。

### 2.4 验证环境

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

NVIDIA 用户期望看到 `True`。Apple Silicon 用户检查：

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

再检查 Ultralytics：

```bash
yolo checks
```

如果 `yolo` 命令找不到，通常是虚拟环境没激活，或 `Scripts/bin` 不在 PATH；直接用 `python` 调用 Ultralytics API 也可以。

---

## 3. 数据准备

### 3.1 类别定义

推荐三类：

| ID | 名称 | 标注对象 | 不要标 |
|---:|---|---|---|
| 0 | `head` | **未戴安全帽的裸头** | 戴了安全帽的头、严重遮挡看不清的头 |
| 1 | `helmet` | 戴在头上的安全帽 | 拿在手里、放在地上、挂在墙上的安全帽 |
| 2 | `person` | 画面中可见的施工人员 | 衣服、人偶、照片里的人 |

关键：`head` 是裸头，不是所有头。所有头都标 `head`，模型仍然分不出戴没戴。

如果只做两类，可以：

```yaml
names:
  0: head
  1: helmet
```

但没有 `person` 时，后处理只能按“头”统计，无法稳定地把头盔关联到具体人员。比赛项目建议三类。

### 3.2 目录结构

```text
<PROJECT_ROOT>/
  datasets/
    helmet/
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/
      data.yaml
```

图片和标签必须同名：

```text
images/train/0001.jpg
labels/train/0001.txt
```

### 3.3 标签格式

YOLO txt，每行一个目标：

```text
class_id x_center y_center width height
```

坐标是 0—1 的归一化值。例如：

```text
0 0.512 0.433 0.120 0.180
1 0.480 0.410 0.090 0.120
```

### 3.4 data.yaml

```yaml
path: <PROJECT_ROOT>/datasets/helmet
train: images/train
val: images/val
test: images/test

names:
  0: head
  1: helmet
  2: person
```

注意：

- 路径用 `/`，不要用单个 `\`。
- `path` 写绝对路径，或确保相对路径从正确目录解析。
- `names` 顺序必须和标签 class_id 一致：本项目固定 `0 head`、`1 helmet`、`2 person`。
- 路径有空格时用引号包起来。

### 3.5 数据检查脚本

在项目根目录运行：

```python
from pathlib import Path
import yaml

root = Path("datasets/helmet")
data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
names = data["names"]
print("classes:", names)

for split in ("train", "val", "test"):
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    imgs = [p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    lbls = list(lbl_dir.glob("*.txt"))
    boxes = 0
    bad = 0
    for lbl in lbls:
        for line in lbl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                bad += 1
                continue
            cid = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            if not (0 <= cid < len(names)):
                bad += 1
            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                bad += 1
            boxes += 1
    img_stems = {p.stem for p in imgs}
    lbl_stems = {p.stem for p in lbls}
    print(f"{split}: images={len(imgs)} labels={len(lbls)} boxes={boxes} bad={bad} "
          f"missing_labels={len(img_stems - lbl_stems)} extra_labels={len(lbl_stems - img_stems)}")
```

`missing_labels` 多，说明图片/标签没对应上；`bad` 不为 0，先修标签再训练。

### 3.6 数据划分

- 按视频/场景划分，不要按连续帧随机切。
- 同一段视频的相邻帧同时出现在训练和测试，指标会虚高。
- 测试集来自未参与训练的场景或日期。
- 公开数据自带 split 可以先用，但正式报告前检查是否有同场景泄漏。

---

## 4. 选择模型和训练参数

### 4.1 按设备选起点

以下是起点建议，不是硬性规定；显存不够就降。

| 设备情况 | 建议模型 | imgsz | batch 起点 | 说明 |
|---|---|---|---:|---|
| 只有 CPU | `yolo11n.pt` | 512/640 | 4—8 | 能跑通，训练慢 |
| NVIDIA 4—6 GB | `yolo11n/s.pt` | 640 | 4—8 | 开 AMP，OOM 降 batch |
| NVIDIA 8—12 GB | `yolo11s.pt` | 640 | 8—16 | 常用起点 |
| NVIDIA 16 GB+ | `yolo11s/m.pt` | 640/960 | 16—32 | 960 更吃显存 |
| Apple Silicon | `yolo11n/s.pt` | 640 | 4—8 | `device=mps`，部分算子回退 CPU |
| AMD Linux | `yolo11n/s.pt` | 640 | 4—8 | 需 ROCm 版 PyTorch |

### 4.2 设备参数

| 设备 | `device` 写法 |
|---|---|
| 单张 NVIDIA | `device=0` |
| 多张 NVIDIA | `device=0,1` |
| Apple Silicon | `device=mps` |
| CPU | `device=cpu` |

### 4.3 训练命令

在项目根目录执行：

```bash
yolo detect train model=yolo11s.pt data=datasets/helmet/data.yaml epochs=100 imgsz=640 batch=8 device=0 workers=4 project=runs/train name=helmet_v1 pretrained=True patience=30 amp=True seed=42
```

显存不够时：

```bash
yolo detect train model=yolo11n.pt data=datasets/helmet/data.yaml epochs=100 imgsz=640 batch=4 device=0 workers=4 project=runs/train name=helmet_v1_n
```

Windows / macOS 如果 DataLoader 报错，先把 `workers=4` 改成 `workers=0`。

### 4.4 Python API

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.train(
    data="datasets/helmet/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    workers=4,
    project="runs/train",
    name="helmet_v1",
    pretrained=True,
    patience=30,
    amp=True,
    seed=42,
)
```

### 4.5 断点续训

```bash
yolo detect train resume model=runs/train/helmet_v1/weights/last.pt
```

不要手动改 `last.pt` 的名字，`resume` 依赖训练目录里的配置。

---

## 5. 训练时看什么

输出目录：

```text
runs/train/helmet_v1/
  weights/best.pt
  weights/last.pt
  results.csv
  results.png
  confusion_matrix.png
  val_batch0_pred.jpg
```

重点看：

- `results.csv` 里的 `metrics/mAP50(B)`、`metrics/mAP50-95(B)`、`val/box_loss`。
- `val_batch0_pred.jpg` 里的框是否合理。
- 训练集高、验证集低：先怀疑过拟合或数据泄漏。
- 验证集一直差：先检查标签，不要盲目加 epoch。

---

## 6. 评估

### 6.1 测试集验证

```bash
yolo detect val model=runs/train/helmet_v1/weights/best.pt data=datasets/helmet/data.yaml split=test imgsz=640 batch=8 device=0
```

记录每一类：

- Precision / Recall / F1
- mAP50 / mAP50-95
- 混淆矩阵
- 误报/漏报的典型图片

### 6.2 业务指标

| 指标 | 含义 |
|---|---|
| 违规召回率 | 真实未戴安全帽的人里，系统报出了多少 |
| 误报率 | 正常/合规画面里，被误报为违规的比例 |
| 告警延迟 | 从违规出现到看板出现事件的时间 |
| FPS | 每秒处理帧数 |

这些要用视频和事件级统计，不要只用单帧 mAP 代替。

### 6.3 错误样本

至少检查：

- 手持安全帽
- 安全帽放在地上/挂在墙上
- 逆光、夜间
- 远处小目标
- 多人遮挡
- 背对镜头
- 黄色安全帽和黄色物体混淆

把这些样本单独整理成“难例集”，下一轮补数据或调阈值。

---

## 7. 推理

### 7.1 图片

```bash
yolo detect predict model=runs/train/helmet_v1/weights/best.pt source=datasets/helmet/images/test conf=0.25 imgsz=640 save=True project=runs/predict name=helmet_test
```

### 7.2 视频

```bash
yolo detect predict model=runs/train/helmet_v1/weights/best.pt source=test.mp4 conf=0.25 imgsz=640 save=True project=runs/predict name=helmet_video
```

### 7.3 摄像头 / RTSP

```bash
yolo detect predict model=runs/train/helmet_v1/weights/best.pt source="rtsp://用户名:密码@摄像头IP:554/路径" conf=0.25 imgsz=640 show=True
```

### 7.4 Python 调用

```python
from ultralytics import YOLO

model = YOLO("runs/train/helmet_v1/weights/best.pt")
results = model.predict(source="test.jpg", conf=0.25, imgsz=640)

for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(result.names[cls_id], conf, x1, y1, x2, y2)
```

检测模型只输出框和类别；“是否违规”要在业务层做：

```text
person 框
  → 取头部区域
  → 匹配 helmet / head
  → 连续帧投票
  → 告警/忽略
```

---

## 8. 导出 ONNX

训练和验证稳定后：

```bash
yolo export model=runs/train/helmet_v1/weights/best.pt format=onnx imgsz=640 opset=12 simplify=True
```

用同一批图片对比 PyTorch 和 ONNX：

```bash
yolo detect predict model=runs/train/helmet_v1/weights/best.onnx source=datasets/helmet/images/test conf=0.25 imgsz=640 save=True
```

TensorRT 可以后面再做，Windows 上安装和版本匹配较麻烦；先把 PyTorch/ONNX 跑通。

---

## 9. 跨平台注意事项

| 系统 | 虚拟环境激活 | 设备参数 | 注意 |
|---|---|---|---|
| Windows | `.venv\Scripts\Activate.ps1` | `device=0` | 路径有空格要加引号；DataLoader 报错先 `workers=0` |
| Linux | `source .venv/bin/activate` | `device=0` | 可用 `workers=4/8` |
| macOS | `source .venv/bin/activate` | `device=mps` | 部分算子不支持 MPS，可能回退 CPU |
| CPU | 同系统 | `device=cpu` | 训练慢，先用小模型/小尺寸跑通 |
| AMD Linux | 同 Linux | `device=0` | 需要 ROCm 版 PyTorch |

Windows 路径建议用 `F:/project/...` 或 `C:/project/...`，不要用单个反斜杠。共享给队友时，尽量用相对路径，避免每台电脑路径不同。

---

## 10. 常见问题

| 现象 | 常见原因 | 处理 |
|---|---|---|
| `torch.cuda.is_available()` 为 False | 装了 CPU 版 PyTorch，或 CUDA 版本不匹配 | 按官网选择正确的 CUDA 版本重装 |
| `no kernel image is available` | 新显卡用了旧 CUDA 轮子 | 换更新的 CUDA 版 PyTorch |
| `Dataset not found` | `data.yaml` 的 `path`/`train` 路径错 | 用绝对路径；检查斜杠 |
| `No labels found` | 图片和标签不同名，或目录结构错 | 检查 `images/train` 与 `labels/train` |
| `Class index out of range` | 标签 class_id 超过 `names` 数量 | 检查 `names` 顺序和标签 |
| `CUDA out of memory` | batch/imgsz 太大 | batch 减半，imgsz 降 512，或换 yolo11n |
| mAP 一直是 0 | 验证集为空、标签路径错、类别不匹配 | 先跑数据检查脚本 |
| DataLoader worker 报错 | Windows/macOS 多进程问题 | `workers=0` |
| `yolo` 不是命令 | 虚拟环境没激活 | 激活 `.venv`，或用 Python API |
| loss 下降但 mAP 很低 | 标签噪声、过拟合、数据泄漏 | 看预测图，检查标签和划分 |
| 训练很慢 | 用了 CPU，或 `device` 没设对 | `device=0/mps`；`yolo checks` 确认 |
| MPS 报错 | 某个算子不支持 | 升 PyTorch；必要时 `device=cpu` |

---

## 11. 团队协作与交付

### 11.1 不要把这些提交到 Git

在 `.gitignore` 中加入：

```text
.venv/
datasets/
runs/
*.pt
*.onnx
*.engine
```

数据、权重、训练输出通常很大，提交到 Git 会让仓库膨胀。可以提交：

- `data.yaml`
- 训练配置/命令
- `results.csv`
- 测试报告
- 代码和 README

`best.pt`、数据集、演示视频可以放团队云盘/移动硬盘，并在 README 里写清楚存放位置和校验值。

### 11.2 交接时记录

- 代码 commit 号
- 数据集版本/来源/类别统计
- 训练命令和随机种子
- `pip freeze > environment.txt`
- 模型文件名、输入尺寸、阈值
- 测试集指标和已知失败场景

### 11.3 复现检查

队友拿到仓库后，应该能按 README 完成：

```bash
python -m venv .venv
# 激活环境
pip install -r requirements.txt
python train.py
```

如果只有你能跑，说明环境或路径没有记录清楚。

---

## 12. 7 天跑通计划

| 天数 | 任务 | 产出 |
|---|---|---|
| 第 1 天 | 建 venv、装 PyTorch/Ultralytics、确认设备 | `torch.cuda.is_available()` 或 MPS 为 True |
| 第 2 天 | 确认类别，准备 helmet/head/person 数据 | 数据目录 + `data.yaml` |
| 第 3 天 | 跑数据检查，训练 YOLO11 基线 | `best.pt`、`results.csv` |
| 第 4 天 | 看验证集、混淆矩阵、错误样本 | 问题清单 |
| 第 5 天 | 补难例/修标签，重训一版 | 第二版模型 |
| 第 6 天 | 视频/摄像头推理，测 FPS | 演示视频 |
| 第 7 天 | 导出 ONNX，整理指标和文档 | 可交付模型 + 记录 |

如果只是做比赛原型，先把第 1—4 天跑通；第 5 天之后再迭代。

---

## 13. 验收清单

- [ ] Python 3.10—3.12，虚拟环境可用。
- [ ] PyTorch 安装正确，GPU/MPS 可用。
- [ ] `ultralytics` 安装成功，`yolo checks` 通过。
- [ ] 数据类别不是只有 `head`，而是 `helmet` + `head`（最好加 `person`）。
- [ ] `data.yaml` 的 `path`、`train`、`val`、`test` 都能找到文件。
- [ ] 每张图片都有同名标签，class_id 在范围内。
- [ ] 训练完成，保存了 `best.pt` 和 `last.pt`。
- [ ] 测试集指标已记录，不只记录训练 loss。
- [ ] 至少跑过一段视频/摄像头，确认不是只在图片上有效。
- [ ] 导出 ONNX 后，用同一批图片复测过。
- [ ] 没有把公开数据说成现场数据，没有编造指标。
- [ ] 数据、权重、训练输出没有误提交到 Git。
- [ ] 代码、配置、权重、数据版本已归档，队友能复现。

---

## 14. 数据、许可证与安全

- 使用 Roboflow、SHWD 等公开数据前，核对 License 和类别定义。
- Ultralytics YOLO11 的许可证以其官方 LICENSE 为准；商业化前评估 AGPL-3.0 等条款。
- 工地视频需要管理方书面许可；不做人脸识别，演示素材中的面部打码。
- 不把现场视频、模型权重随意上传公共网盘或公共仓库。
- 对外材料不写“100% 准确”“完全替代人工”，只写实测指标和适用范围。

---

## 15. 参考

- Ultralytics YOLO11 文档：https://docs.ultralytics.com/models/yolo11/
- PyTorch 安装：https://pytorch.org/get-started/locally/
- Roboflow Universe：https://universe.roboflow.com/
- SHWD 安全帽数据：https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset
