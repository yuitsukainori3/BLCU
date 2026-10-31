# YOLO11 安全帽模型训练指南

> 适用环境：Windows + NVIDIA RTX 5060 Laptop（8 GB 显存）  
> 目标：从 COCO 预训练的 YOLO11 出发，用自己的安全帽数据微调一个可用的检测模型。  
> 说明：本文只讲“怎么做、怎么检查”，不预设任何准确率；所有指标以你实际训练和测试的结果为准。

## 0. 先说结论：你现在这份数据不能直接训练“安全帽佩戴”模型

我检查了工作区里的：

`Hard Hat Workers.v14-raw_headclassonly.yolov11.zip`

它的 `data.yaml` 是：

```yaml
nc: 1
names: ['head']
```

数据统计如下：

| 划分 | 图片数 | head 框数 |
|---|---:|---:|
| train | 4,916 | 18,531 |
| valid | 1,413 | 5,252 |
| test | 706 | 2,641 |
| **合计** | **7,035** | **26,424** |

所有标签都是 `class 0 = head`，没有空标签，也没有格式错误。这个数据量足够用来跑通 YOLO11 的训练流程，但它**只有一个 head 类**，无法区分“戴了安全帽的头”和“没戴安全帽的头”。

因此：

- 直接训练这份数据，得到的是 **head 检测器**，不是安全帽合规模型。
- **不要**把 `head` 简单改名成 `helmet`。这样模型会把所有头都当成安全帽，裸头也会被判为合规。
- 想完成“安全帽佩戴检测”，需要一份至少包含 `helmet` 和 `head`（裸头）两类，最好再加 `person` 的数据。

三条路线：

| 路线 | 做法 | 适合情况 |
|---|---|---|
| A. 换数据集（推荐） | 找 `helmet / head / person` 三类的安全帽数据集，直接训练 | 想尽快做出合规模型 |
| B. 自己补标 | 在当前 7,035 张图上补 `helmet` 框，把裸头标成 `head` | 有标注人力、想用现成图片 |
| C. 两阶段（备选） | 先用当前数据训练 head 检测器，再裁头训练 helmet/no_helmet 分类器 | 不想重标整图，但能接受更复杂的管线 |

路线 A 的数据来源：到 [Roboflow Universe](https://universe.roboflow.com/) 搜索 `safety helmet` / `hard hat`，下载前先看三点：

1. Classes 是否包含 `helmet`、`head`（最好还有 `person`）；
2. 是否提供 YOLO11 / YOLOv8 格式；
3. License 是否允许你的参赛/展示用途。

也可以考虑 SHWD 等公开安全帽数据集，但同样要先核对类别定义和许可证，不要直接假设它的标签和你的规则一致。

下面按“路线 A”讲完整流程；路线 B 只是把数据换掉，步骤一样。路线 C 在最后简单说明。

---

## 1. 整体流程

```text
确认类别
  → 安装 Python/PyTorch/Ultralytics
  → 准备 helmet/head 数据
  → 写 data.yaml
  → 检查图片、标签、类别
  → YOLO11 微调训练
  → 验证集/测试集评估
  → 图片/视频/摄像头推理
  → 导出 ONNX
  → 接入项目业务逻辑
```

先跑通“能训练、能评估、能推理”的闭环，再谈涨点。不要一上来就改网络结构。

---

## 2. 环境安装

### 2.1 当前机器情况

我检查到的环境：

- GPU：`NVIDIA GeForce RTX 5060 Laptop GPU`，8 GB 显存，驱动 `621.01`。
- 当前 Python 环境里还没有安装 `torch`、`ultralytics`、`opencv`。
- 工作区里有 `ultralytics-8.3.163` 源码和 `yolo11n.pt`，但建议新建独立虚拟环境，避免污染系统 Python。

### 2.2 安装 Python 3.11 或 3.12

不建议用 Python 3.14 做第一次训练：PyTorch 和部分依赖的轮子可能还没跟上。用 3.11/3.12 更稳。

在 PowerShell 中：

```powershell
py -3.12 -m venv F:\Competition\.venv
F:\Competition\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果提示“禁止运行脚本”，只对当前窗口放开：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 2.3 安装 PyTorch（RTX 5060 注意 CUDA 版本）

RTX 5060 是较新的 Blackwell 架构，**不要装 cu118 的旧轮子**，否则可能出现“检测不到 GPU”或“no kernel image is available”。建议使用 CUDA 12.8 或更高版本的官方轮子。

先到 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择 Windows / Pip / 你的 CUDA 版本，复制它给出的命令。一个常见形式是：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

如果这个地址在你的网络下不可用，就以官网当前给出的 CUDA 版本为准，不要自己猜版本号。

### 2.4 安装 Ultralytics

```powershell
pip install ultralytics
```

工作区里已有 `ultralytics-8.3.163` 源码，也可以选择从源码安装：

```powershell
pip install -e "F:\Competition\ultralytics-8.3.163\ultralytics-8.3.163"
```

第一次用建议直接用 PyPI 版，依赖关系更简单。

### 2.5 验证 GPU

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

期望看到：

```text
True
NVIDIA GeForce RTX 5060 Laptop GPU
```

如果 `torch.cuda.is_available()` 是 `False`：

1. 先确认装的是 CUDA 版，不是 CPU 版；
2. 卸载后按官网命令重装；
3. 更新 NVIDIA 驱动；
4. 仍不行再检查是否被其他环境变量/代理影响。

### 2.6 验证 YOLO

```powershell
yolo checks
```

或者：

```powershell
python -c "from ultralytics import YOLO; print(YOLO('yolo11n.pt'))"
```

第一次会自动下载 `yolo11n.pt`。如果 `yolo` 命令找不到，通常是没有激活虚拟环境，或 `Scripts` 不在 PATH；直接用 `python` 调用也可以。

---

## 3. 准备数据

### 3.1 推荐类别

建议三类：

| ID | 名称 | 标注对象 | 不要标 |
|---:|---|---|---|
| 0 | `person` | 画面中可见的施工人员 | 衣服、人偶、照片里的人 |
| 1 | `helmet` | 戴在头上的安全帽 | 拿在手里、放在地上、挂在墙上的安全帽 |
| 2 | `head` | **未戴安全帽的裸头** | 戴了安全帽的头、严重遮挡看不清的头 |

关键点：`head` 是“裸头”，不是“所有头”。如果所有头都标成 `head`，模型还是分不出戴没戴。

如果只想先做两类，也可以：

```yaml
names:
  0: helmet
  1: head
```

但没有 `person` 时，后处理只能按“头”做统计，无法稳定地把头盔关联到具体人员。比赛项目建议三类。

另外，工作区里的 `D-Fire` 是另一项任务的数据，不建议和安全帽数据混在一起；除非你明确要做多任务模型，否则分开训练。

### 3.2 目录结构

把数据整理成 Ultralytics 习惯的结构：

```text
F:\Competition\datasets\helmet\
  images\
    train\
    val\
    test\
  labels\
    train\
    val\
    test\
  data.yaml
```

图片和标签必须**同名**，例如：

```text
images/train/0001.jpg
labels/train/0001.txt
```

标签文件是 YOLO txt，每行一个目标：

```text
class_id x_center y_center width height
```

坐标都是 0—1 的归一化值。例如：

```text
1 0.512 0.433 0.120 0.180
2 0.480 0.410 0.090 0.120
```

### 3.3 data.yaml

在 `F:\Competition\datasets\helmet\data.yaml` 写：

```yaml
path: F:/Competition/datasets/helmet
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: helmet
  2: head
```

注意：

- 路径用 `/` 或双反斜杠，不要用单个 `\`。
- `path` 建议写绝对路径，最不容易出错。
- `names` 的顺序必须和标签里的 class_id 一致。
- 不要把 `nc` 和 `names` 写冲突；Ultralytics 会从 `names` 推断类别数。

### 3.4 当前 head 数据怎么处理

如果只是想先跑通训练流程，可以把当前 zip 解压到：

```text
F:\Competition\datasets\hardhat_head\
```

然后把它原来的 `data.yaml` 改成：

```yaml
path: F:/Competition/datasets/hardhat_head
train: train/images
val: valid/images
test: test/images

names: ['head']
```

原来的 `train: ../train/images` 是相对路径，容易指到数据集外面；改成上面的写法更稳。

然后可以训练一个 **head 检测器** 练手：

```powershell
yolo detect train model=yolo11s.pt data=F:/Competition/datasets/hardhat_head/data.yaml epochs=100 imgsz=640 batch=8 device=0 workers=4 project=F:/Competition/runs/train name=head_practice pretrained=True patience=30 amp=True seed=42
```

但请记住：这个模型只能检测“头”，不能判断“有没有戴安全帽”，不能直接作为比赛的安全帽合规模型。

### 3.5 数据检查脚本

训练前先跑一遍检查，避免“标签路径不对/类别越界/坐标为 0”这类低级问题：

```python
from pathlib import Path
import yaml

root = Path(r"F:/Competition/datasets/helmet")
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

如果 `missing_labels` 很多，说明图片和标签没有一一对应；如果 `bad` 不为 0，先修标签再训练。

### 3.6 数据划分

- 训练/验证/测试建议按 **视频或场景** 划分，不要按连续帧随机切。
- 同一段视频的相邻帧同时出现在训练和测试，指标会虚高。
- 测试集最好来自未参与训练的场景或日期。
- 当前 Roboflow 数据自带 train/valid/test，先用它跑通可以，但正式报告前要检查是否有同场景泄漏。

---

## 4. 开始训练

### 4.1 推荐起点

- 模型：`yolo11s.pt`（8 GB 显存优先从 s 开始；跑不动再换 n）
- 输入：`640`
- batch：`8`（OOM 就 4）
- epochs：`100`（配合 early stopping）
- patience：`30`
- AMP：开启
- 预训练：COCO 权重，不从零训练

### 4.2 命令行训练

```powershell
yolo detect train model=yolo11s.pt data=F:/Competition/datasets/helmet/data.yaml epochs=100 imgsz=640 batch=8 device=0 workers=4 project=F:/Competition/runs/train name=helmet_v1 pretrained=True patience=30 amp=True seed=42
```

如果显存不够：

```powershell
yolo detect train model=yolo11n.pt data=F:/Competition/datasets/helmet/data.yaml epochs=100 imgsz=640 batch=4 device=0 workers=4 project=F:/Competition/runs/train name=helmet_v1_n
```

如果 Windows 下 DataLoader 报错，先把 `workers=4` 改成 `workers=0`。

### 4.3 Python API 训练

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.train(
    data=r"F:/Competition/datasets/helmet/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    workers=4,
    project=r"F:/Competition/runs/train",
    name="helmet_v1",
    pretrained=True,
    patience=30,
    amp=True,
    seed=42,
)
```

### 4.4 断点续训

训练中断后：

```powershell
yolo detect train resume model=F:/Competition/runs/train/helmet_v1/weights/last.pt
```

不要手动改 `last.pt` 的名字，`resume` 依赖训练目录里的配置。

### 4.5 训练时看什么

训练输出目录：

```text
F:\Competition\runs\train\helmet_v1\
  weights\best.pt
  weights\last.pt
  results.csv
  results.png
  confusion_matrix.png
  val_batch0_pred.jpg
```

重点看：

- `results.csv` 里的 `metrics/mAP50(B)`、`metrics/mAP50-95(B)`、`val/box_loss`。
- `val_batch0_pred.jpg` 里的框是否合理，不要只看 loss。
- 如果训练集指标很高、验证集很低，优先怀疑过拟合或数据泄漏。
- 如果验证集一直很差，优先检查标签，而不是继续加 epoch。

---

## 5. 评估模型

### 5.1 在测试集上验证

```powershell
yolo detect val model=F:/Competition/runs/train/helmet_v1/weights/best.pt data=F:/Competition/datasets/helmet/data.yaml split=test imgsz=640 batch=8 device=0
```

记录每一类的：

- Precision / Recall / F1
- mAP50 / mAP50-95
- 混淆矩阵
- 误报和漏报的典型图片

不要只报一个总体 mAP。安全帽场景更关心：

| 业务指标 | 含义 |
|---|---|
| 违规召回率 | 真实未戴安全帽的人里，系统报出了多少 |
| 误报率 | 正常/合规画面里，被误报为违规的比例 |
| 告警延迟 | 从违规出现到看板出现事件的时间 |
| FPS | 每秒处理帧数 |

这些指标要用视频和事件级统计，不要只用单帧 mAP 代替。

### 5.2 看错误样本

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

## 6. 推理

### 6.1 图片

```powershell
yolo detect predict model=F:/Competition/runs/train/helmet_v1/weights/best.pt source=F:/Competition/datasets/helmet/images/test conf=0.25 imgsz=640 save=True project=F:/Competition/runs/predict name=helmet_test
```

### 6.2 视频

```powershell
yolo detect predict model=F:/Competition/runs/train/helmet_v1/weights/best.pt source=F:/Competition/test.mp4 conf=0.25 imgsz=640 save=True project=F:/Competition/runs/predict name=helmet_video
```

### 6.3 摄像头 / RTSP

```powershell
yolo detect predict model=F:/Competition/runs/train/helmet_v1/weights/best.pt source="rtsp://用户名:密码@摄像头IP:554/路径" conf=0.25 imgsz=640 show=True
```

### 6.4 Python 调用

```python
from ultralytics import YOLO

model = YOLO(r"F:/Competition/runs/train/helmet_v1/weights/best.pt")
results = model.predict(source=r"F:/Competition/test.jpg", conf=0.25, imgsz=640)

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

这部分对应项目文档里的 5.4 节。

---

## 7. 导出 ONNX

训练和验证稳定后，再导出：

```powershell
yolo export model=F:/Competition/runs/train/helmet_v1/weights/best.pt format=onnx imgsz=640 opset=12 simplify=True
```

导出后一定用同一批图片对比 PyTorch 和 ONNX 的结果：

```powershell
yolo detect predict model=F:/Competition/runs/train/helmet_v1/weights/best.onnx source=F:/Competition/datasets/helmet/images/test conf=0.25 imgsz=640 save=True
```

TensorRT 可以后面再做。Windows 上 TensorRT 安装和版本匹配比较麻烦，先把 PyTorch/ONNX 跑通，不要一开始就上 TensorRT。

---

## 8. 常见问题

| 现象 | 常见原因 | 处理 |
|---|---|---|
| `torch.cuda.is_available()` 为 False | 装了 CPU 版 PyTorch，或 CUDA 版本不匹配 | 卸载后按官网 CUDA 12.8+ 命令重装 |
| `no kernel image is available` | RTX 50 系用了旧 CUDA 轮子 | 换 CUDA 12.8 或更高版本的 PyTorch |
| `Dataset not found` | `data.yaml` 的 `path`/`train` 路径错 | 用绝对路径；检查斜杠方向 |
| `No labels found` | 图片和标签不同名，或目录结构错 | 检查 `images/train` 与 `labels/train` |
| `Class index out of range` | 标签 class_id 超过 `names` 数量 | 检查 `names` 顺序和标签 |
| `CUDA out of memory` | batch/imgsz 太大 | batch 减半，imgsz 降 512，或换 yolo11n |
| mAP 一直是 0 | 验证集为空、标签路径错、类别不匹配 | 先跑数据检查脚本 |
| DataLoader worker 报错 | Windows 多进程问题 | `workers=0` |
| `yolo` 不是命令 | 虚拟环境没激活 | 用 `python -m` 或激活 `.venv` |
| loss 下降但 mAP 很低 | 标签噪声、过拟合、数据泄漏 | 看 `val_batch*_pred.jpg`，检查标签和划分 |
| 训练很慢 | 用了 CPU，或 `device` 没设对 | `device=0`；`yolo checks` 确认 GPU |

---

## 9. 7 天跑通计划

| 天数 | 任务 | 产出 |
|---|---|---|
| 第 1 天 | 建 venv、装 PyTorch/Ultralytics、确认 GPU | `torch.cuda.is_available() == True` |
| 第 2 天 | 确定类别，准备 helmet/head/person 数据 | 数据目录 + `data.yaml` |
| 第 3 天 | 跑数据检查，训练 YOLO11s 基线 | `best.pt`、`results.csv` |
| 第 4 天 | 看验证集、混淆矩阵、错误样本 | 问题清单 |
| 第 5 天 | 补难例/修标签，重训一版 | 第二版模型 |
| 第 6 天 | 视频/摄像头推理，测 FPS | 演示视频 |
| 第 7 天 | 导出 ONNX，整理指标和文档 | 可交付模型 + 记录 |

如果只是做比赛原型，先把第 1—4 天跑通；第 5 天之后再迭代。

---

## 10. 验收清单

- [ ] `torch.cuda.is_available()` 为 True，且识别到 RTX 5060。
- [ ] 数据类别不是只有 `head`，而是 `helmet` + `head`（最好加 `person`）。
- [ ] `data.yaml` 的 `path`、`train`、`val`、`test` 都能找到文件。
- [ ] 每张图片都有同名标签，class_id 在范围内。
- [ ] 训练完成，保存了 `best.pt` 和 `last.pt`。
- [ ] 测试集指标已记录，不只记录训练 loss。
- [ ] 至少跑过一段视频/摄像头，确认不是只在图片上有效。
- [ ] 导出 ONNX 后，用同一批图片复测过。
- [ ] 没有把公开数据说成现场数据，没有编造指标。
- [ ] 代码、配置、权重、数据版本已提交/归档。
- [ ] `.venv/`、`datasets/`、`runs/`、`*.pt` 已加入 `.gitignore`，避免把大数据和权重误提交到 Git。

---

## 11. 关于当前数据的建议

如果这周就要开始训练，我的建议是：

1. **不要**把当前 `head` 数据直接当成安全帽数据训练比赛模型。
2. 先用它跑通环境、训练、评估、推理的完整流程，确认 GPU 和代码没问题。
3. 同时找一份 `helmet / head / person` 的数据；找不到就先用 500—1,000 张图试标，确认类别定义可行，再扩到 3,000 张以上。
4. 如果不想重标整图，可以考虑路线 C：当前 head 检测器裁头 → 人工给裁出的头标 `helmet`/`no_helmet` → 训练分类器。但这条路线多了目标检测 + 分类两级，答辩时要解释清楚误差会累积，不如直接做三类检测干净。

---

## 12. 数据、许可证与安全

- 当前 `Hard Hat Workers` 数据在 `data.yaml` 中标注为 `Public Domain`，使用前仍建议核对 Roboflow 页面说明。
- Ultralytics YOLO11 的许可证以其官方 LICENSE 为准；如未来商业化，需要评估 AGPL-3.0 等条款。
- 工地视频需要管理方书面许可；不做人脸识别，演示素材中的面部打码。
- 不把现场视频、模型权重随意上传公共网盘或公共仓库。
- 对外材料不写“100% 准确”“完全替代人工”，只写实测指标和适用范围。

---

## 13. 参考

- Ultralytics YOLO11 文档：https://docs.ultralytics.com/models/yolo11/
- PyTorch 安装：https://pytorch.org/get-started/locally/
- 当前数据页面：https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/14
- 项目设计文档：`安全帽智检_项目文档_V0.1.docx`
