#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从检测数据集裁出“头部裁片”，用于训练两阶段里的复核分类器。

做什么
------
读 SHWD 两类标注（0 head / 1 helmet），把每个 GT 框外扩 --pad 倍后裁出来，
统一缩放到 --imgsz，按 Ultralytics 分类任务的目录结构落盘：

    datasets/head_cls/
        train/head/*.jpg
        train/helmet/*.jpg
        val/head/*.jpg
        val/helmet/*.jpg

几个刻意的设计
--------------
1. **抖动**：训练裁片随机平移/缩放 ±--jitter，模拟“推理时用预测框而不是 GT 框”的偏差。
   不加这个，分类器在真实推理上会掉点。
2. **head 过采样**：head 只有 966 个、helmet 有 6,421 个，默认把 head 复制 --head-repeat 份，
   每份用不同的抖动，缓解 1:6.6 的不平衡。
3. **小框过滤**：原始框短边 < --min-px 的不裁（放大到 224 也只是一团糊），并统计数量。

用法
----
python scripts/build_head_crops.py                        # 默认参数，写到 datasets/head_cls
python scripts/build_head_crops.py --pad 1.5 --jitter 0.15
python scripts/build_head_crops.py --dry-run              # 只统计，不写文件
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _ui import die, echo_config, progress  # noqa: E402

# ==================== 可调参数（只改这里） ====================
DATA_YAML = (ROOT / "ultralytics-8.3.163" / "ultralytics-8.3.163"
             / "ultralytics" / "cfg" / "datasets" / "SHWD.yaml")  # 原始 SHWD 划分
OUT_DIR = ROOT / "datasets" / "head_cls"   # 裁片输出（Ultralytics 分类目录结构）
PAD = 1.3             # GT 框外扩倍数（1.3 = 四周各留 15%）
JITTER = 0.1          # 训练裁片随机平移/缩放比例，模拟预测框偏差
CROP_SIZE = 224       # 裁片统一边长
MIN_PX = 10.0         # GT 框短边小于该值不裁（放大也是糊）
HEAD_REPEAT = 2       # head 类过采样倍数（只作用于 train，缓解 1:6.6 不平衡）
DEGRADE_PROB = 0.0    # 训练裁片随机降质概率，0 = 不做
BLUR_MAX = 2.0        # 高斯模糊半径上限
DET_IMGSZ = 960       # --from-pred 时检测器的输入尺寸
DET_CONF = 0.25       # --from-pred 时预测框的最低置信度
MATCH_IOU = 0.5       # 预测框与 GT 匹配的 IoU 下限
QUALITY = 90          # JPEG 质量
PREVIEW = 8           # 拼一张预览图，0 = 不生成
SEED = 0
DEVICE = "0"
# ============================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA_YAML)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--pad", type=float, default=PAD, help="GT 框外扩倍数（1.3 = 四周各留 15%%）")
    p.add_argument("--jitter", type=float, default=JITTER, help="训练裁片随机平移/缩放比例")
    p.add_argument("--imgsz", type=int, default=CROP_SIZE)
    p.add_argument("--min-px", type=float, default=MIN_PX, help="GT 框短边小于该值不裁")
    p.add_argument("--head-repeat", type=int, default=HEAD_REPEAT, help="head 类过采样倍数（只作用于 train）")
    p.add_argument("--degrade-prob", type=float, default=DEGRADE_PROB,
                   help="训练裁片随机降质的概率（模糊/降采样），0 = 不做")
    p.add_argument("--blur-max", type=float, default=BLUR_MAX, help="高斯模糊半径上限")
    p.add_argument("--from-pred", action="store_true",
                   help="用检测器的预测框裁片（只取与 GT IoU≥--match-iou 的），让训练分布对齐推理")
    p.add_argument("--det", type=Path, default=None, help="--from-pred 时使用的检测权重")
    p.add_argument("--det-imgsz", type=int, default=DET_IMGSZ)
    p.add_argument("--det-conf", type=float, default=DET_CONF, help="预测框的最低置信度")
    p.add_argument("--match-iou", type=float, default=MATCH_IOU, help="预测框与 GT 匹配的 IoU 下限")
    p.add_argument("--quality", type=int, default=QUALITY, help="JPEG 质量")
    p.add_argument("--preview", type=int, default=PREVIEW, help="拼一张预览图，0 = 不生成")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def read_yaml(path: Path) -> dict:
    out, section = {}, None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith((" ", "\t")) and section:
            k, _, v = line.partition(":")
            out.setdefault(section, {})[k.strip()] = v.strip().strip("'\"")
            continue
        k, _, v = line.partition(":")
        section = k.strip() if not v.strip() else None
        out[k.strip()] = v.strip().strip("'\"")
    return out


def class_names(cfg: dict) -> dict[int, str]:
    names = cfg.get("names", {})
    if isinstance(names, str):
        items = [x.strip().strip("'\"") for x in names.strip().strip("[]").split(",") if x.strip()]
        return {i: n for i, n in enumerate(items)}
    return {int(k): v for k, v in names.items()}


def degrade(im: Image.Image, rng: random.Random, prob: float, blur_max: float, size: int) -> Image.Image:
    """随机降质：模糊 / 降采样再放大 / 两者叠加。

    为什么要做：head 裁片中位 35×44、helmet 80×98，缩放后清晰度天然不同，
    模型很容易学到"越糊越像头盔"的捷径。对两类做同样的降质，逼它看内容而不是清晰度。
    """
    if rng.random() > prob:
        return im
    kind = rng.choice(("blur", "down", "both"))
    if kind in ("blur", "both"):
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, blur_max)))
    if kind in ("down", "both"):
        side = max(32, int(size * rng.uniform(0.35, 0.7)))
        im = im.resize((side, side), Image.BILINEAR).resize((size, size), Image.BILINEAR)
    return im


def crop_box(box, pad: float, jitter: float, rng: random.Random, w: int, h: int):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * pad, (y2 - y1) * pad
    if jitter:
        cx += rng.uniform(-jitter, jitter) * bw
        cy += rng.uniform(-jitter, jitter) * bh
        bw *= 1 + rng.uniform(-jitter, jitter) / 2
        bh *= 1 + rng.uniform(-jitter, jitter) / 2
    nx1, ny1 = max(0, int(round(cx - bw / 2))), max(0, int(round(cy - bh / 2)))
    nx2, ny2 = min(w, int(round(cx + bw / 2))), min(h, int(round(cy + bh / 2)))
    if nx2 - nx1 < 4 or ny2 - ny1 < 4:
        return None
    return nx1, ny1, nx2, ny2


def main(argv=None) -> int:
    args = parse_args(argv)
    data = args.data.expanduser().resolve()
    if not data.is_file():
        die(f"data.yaml 不存在：{data}")
    cfg = read_yaml(data)
    names = class_names(cfg)
    root = Path(cfg["path"])

    echo_config("头部裁片配置", [
        ("数据", data),
        ("图片根", root),
        ("输出", args.out.expanduser().resolve()),
        ("裁片", f"{args.imgsz}px    pad {args.pad}    jitter {args.jitter}    min_px {args.min_px}"),
        ("过采样", f"head ×{args.head_repeat}（仅 train）    降质概率 {args.degrade_prob}"),
        ("框来源", f"预测框（imgsz {args.det_imgsz}, conf {args.det_conf}, IoU≥{args.match_iou}）"
                   if args.from_pred else "GT 框"),
        ("随机种子", args.seed),
    ])

    rng = random.Random(args.seed)
    preview_buf: list[tuple[Image.Image, str]] = []
    stat = {split: {n: 0 for n in names.values()} for split in ("train", "val")}
    skipped = {"small": 0, "degenerate": 0, "empty": 0, "unmatched": 0}

    det = None
    if args.from_pred:
        if not args.det or not Path(args.det).expanduser().is_file():
            die("--from-pred 需要 --det 指向检测权重")
        from ultralytics import YOLO
        det = YOLO(str(Path(args.det).expanduser()))

    def iou_one(a, b) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    for split in ("train", "val"):
        img_dir = root / cfg.get(split, f"images/{split}")
        lbl_dir = root / "labels" / split
        if not img_dir.is_dir():
            print(f"[警告] 跳过 {split}：{img_dir} 不存在")
            continue
        out_dirs = {n: args.out / split / n for n in names.values()}
        if not args.dry_run:
            for d in out_dirs.values():
                d.mkdir(parents=True, exist_ok=True)

        imgs = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        bar = progress(imgs, desc=f"裁片 {split}", unit="img")
        for img_path in bar:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.is_file():
                skipped["empty"] += 1
                continue
            lines = [ln.split() for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                skipped["empty"] += 1
                continue

            im = Image.open(img_path).convert("RGB")
            w, h = im.size

            # 组装 (类别, 框)：默认用 GT；--from-pred 时用“与 GT 对得上的预测框”，
            # 让裁片分布和推理时一致（框会抖、会松）。
            items: list[tuple[int, tuple]] = []
            if det is None:
                for p in lines:
                    if len(p) < 5:
                        continue
                    cid = int(float(p[0]))
                    x, y, bw, bh = map(float, p[1:5])
                    items.append((cid, ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)))
            else:
                gts = [(int(float(p[0])), ((float(p[1]) - float(p[3]) / 2) * w, (float(p[2]) - float(p[4]) / 2) * h,
                                          (float(p[1]) + float(p[3]) / 2) * w, (float(p[2]) + float(p[4]) / 2) * h))
                       for p in lines if len(p) >= 5]
                res = det.predict(str(img_path), imgsz=args.det_imgsz, conf=args.det_conf,
                                  iou=0.7, max_det=300, device=args.device, verbose=False)[0]
                for bx in res.boxes.xyxy.cpu().numpy():
                    best_i, best_v = -1, 0.0
                    for gi, (_, gb) in enumerate(gts):
                        v = iou_one(bx, gb)
                        if v > best_v:
                            best_i, best_v = gi, v
                    if best_v >= args.match_iou:
                        items.append((gts[best_i][0], tuple(float(v) for v in bx)))
                    else:
                        skipped["unmatched"] += 1

            for i, (cid, box) in enumerate(items):
                if min(box[2] - box[0], box[3] - box[1]) < args.min_px:
                    skipped["small"] += 1
                    continue

                repeats = args.head_repeat if (split == "train" and names.get(cid) == "head") else 1
                for r in range(repeats):
                    cb = crop_box(box, args.pad, args.jitter if split == "train" else 0, rng, w, h)
                    if cb is None:
                        skipped["degenerate"] += 1
                        continue
                    crop = im.crop(cb).resize((args.imgsz, args.imgsz), Image.BILINEAR)
                    if split == "train" and args.degrade_prob > 0:
                        crop = degrade(crop, rng, args.degrade_prob, args.blur_max, args.imgsz)
                    name = f"{img_path.stem}_{i}" + (f"_r{r}" if r else "") + ".jpg"
                    stat[split][names[cid]] += 1
                    if not args.dry_run:
                        crop.save(out_dirs[names[cid]] / name, quality=args.quality)
                    if args.preview and len(preview_buf) < args.preview:
                        preview_buf.append((crop.copy(), names[cid]))
            bar.set_postfix(crops=sum(stat[split].values()))

    print(f"裁片统计（pad={args.pad}, jitter={args.jitter}, min_px={args.min_px}, head_repeat={args.head_repeat}）")
    for split, d in stat.items():
        print(f"  {split:5s} " + "  ".join(f"{k}={v}" for k, v in d.items()))
    print(f"  跳过：{skipped}")

    if args.dry_run:
        print("\n[dry-run] 没有写任何文件")
        return 0

    if args.preview and preview_buf:
        cols, cell = 4, args.imgsz
        rows = (len(preview_buf) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
        for i, (crop, _) in enumerate(preview_buf):
            sheet.paste(crop, ((i % cols) * cell, (i // cols) * cell))
        preview_path = args.out / "preview.jpg"
        sheet.save(preview_path, quality=90)
        print(f"预览图 -> {preview_path}（4 列，前 {len(preview_buf)} 张）")

    total = sum(sum(d.values()) for d in stat.values())
    print(f"\n共 {total} 张裁片 -> {args.out}")
    print("下一步：yolo classify train model=yolo11n-cls.pt data=<这个目录> imgsz=224 epochs=30")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
