#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把"难例贴片"贴回训练图，构造带负样本的训练集。

    python scripts/build_negative_aug.py --out datasets/shwd_neg --n-images 300

做什么
------
1. 用硬链接把原 SHWD 的 images/labels 复制进新目录（不额外占盘）；
2. 随机挑 --n-images 张训练图，每张贴 1—3 个难例贴片（默认 14—30 px，对齐推理尺寸）；
3. 贴片位置避开已有标注框（不遮挡真目标），标签文件原样复制——贴片本身没有标签，
   模型要学的是"这里不该报"；
4. 生成 `SHWD_neg.yaml`，可以直接拿去训练。

注意：**只增强 train**，val / test 原样不动。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _ui import die, echo_config, progress  # noqa: E402

# ==================== 可调参数（只改这里） ====================
DATA_YAML = (ROOT / "ultralytics-8.3.163" / "ultralytics-8.3.163"
             / "ultralytics" / "cfg" / "datasets" / "SHWD.yaml")  # 原始 SHWD 划分
PATCHES_DIR = ROOT / "datasets" / "negatives" / "hands"   # mine_hard_negatives.py 挖出的贴片
OUT_DIR = ROOT / "datasets" / "shwd_neg"                   # 负样本增强后的训练集
N_IMAGES = 300        # 贴片增强的训练图数量（neg_v1 用的就是 300）
PER_IMAGE = (1, 3)    # 每张图贴 1—3 个贴片
PATCH_SIZE = (14, 30)  # 贴片边长范围（px），对齐推理时的手部框尺寸
SEED = 0
# ============================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA_YAML, help="原 SHWD yaml")
    p.add_argument("--patches", type=Path, default=PATCHES_DIR)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--n-images", type=int, default=N_IMAGES, help="贴片增强的训练图数量")
    p.add_argument("--per-image", type=int, nargs=2, default=list(PER_IMAGE), metavar=("MIN", "MAX"))
    p.add_argument("--size", type=int, nargs=2, default=list(PATCH_SIZE), metavar=("MIN", "MAX"),
                   help="贴片边长范围")
    p.add_argument("--seed", type=int, default=SEED)
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


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_boxes(label: Path, w: int, h: int):
    boxes = []
    if label.is_file():
        for line in label.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            x, y, bw, bh = map(float, p[1:5])
            boxes.append(((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
    return boxes


def feather_mask(s: int) -> np.ndarray:
    """边缘羽化的方形蒙版，避免贴片留下生硬直角（否则模型学的是"方块边"而不是"手"）。"""
    m = np.ones((s, s), dtype=np.float32)
    f = max(2, s // 6)
    ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
    m[:f, :] *= ramp[:, None]
    m[-f:, :] *= ramp[::-1, None]
    m[:, :f] *= ramp[None, :]
    m[:, -f:] *= ramp[::-1][None, :]
    return m


def overlaps(box, boxes, margin=4.0) -> bool:
    x1, y1, x2, y2 = box
    for bx1, by1, bx2, by2 in boxes:
        if x1 < bx2 + margin and x2 > bx1 - margin and y1 < by2 + margin and y2 > by1 - margin:
            return True
    return False


def main(argv=None) -> int:
    args = parse_args(argv)
    data = args.data.expanduser().resolve()
    if not data.is_file():
        die(f"data.yaml 不存在：{data}")
    cfg = read_yaml(data)
    src_root = Path(cfg["path"])
    patches_dir = args.patches.expanduser().resolve()
    patch_files = sorted(patches_dir.glob("*.png"))
    if not patch_files:
        die(f"{patches_dir} 下没有 png 贴片")
    patches = [np.array(Image.open(p).convert("RGB")) for p in patch_files]

    rng = random.Random(args.seed)
    out = args.out.expanduser().resolve()

    echo_config("负样本增强配置", [
        ("数据", data),
        ("贴片", f"{patches_dir}（{len(patch_files)} 张）"),
        ("输出", out),
        ("增强图数", f"{args.n_images}    每图 {args.per_image[0]}—{args.per_image[1]} 个"),
        ("贴片边长", f"{args.size[0]}—{args.size[1]} px    随机种子 {args.seed}"),
    ])

    splits = {"train": cfg.get("train", "images/train"), "val": cfg.get("val", "images/val"),
              "test": cfg.get("test", "images/test")}
    stats = {"links": 0, "aug": 0, "pasted": 0}

    for split in ("train", "val", "test"):
        src_img = src_root / splits[split]
        if not src_img.is_dir():
            print(f"[警告] 缺少 {src_img}，跳过")
            continue
        dst_img = out / "images" / split
        dst_lbl = out / "labels" / split
        imgs = [p for p in sorted(src_img.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        for ip in progress(imgs, desc=f"链接 {split}", unit="img", leave=False):
            link_or_copy(ip, dst_img / ip.name)
            lp = src_root / "labels" / split / f"{ip.stem}.txt"
            if lp.is_file():
                link_or_copy(lp, dst_lbl / lp.name)
            stats["links"] += 1

    train_img = src_root / splits["train"]
    train_files = sorted(p for p in train_img.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    picked = rng.sample(train_files, min(args.n_images, len(train_files)))
    aug_meta: dict[str, list] = {}

    bar = progress(picked, desc="贴片增强", unit="img")
    for ip in bar:
        im = Image.open(ip).convert("RGB")
        arr = np.array(im)
        h, w = arr.shape[:2]
        boxes = load_boxes(src_root / "labels" / "train" / f"{ip.stem}.txt", w, h)
        n = rng.randint(args.per_image[0], args.per_image[1])
        pasted = 0
        boxes_pasted = []
        for _ in range(n * 6):  # 最多尝试，避免无限循环
            if pasted >= n:
                break
            s = rng.randint(args.size[0], args.size[1])
            px, py = rng.randint(0, max(0, w - s)), rng.randint(0, max(0, h - s))
            if overlaps((px, py, px + s, py + s), boxes):
                continue
            patch = patches[rng.randrange(len(patches))]
            patch = np.array(Image.fromarray(patch).resize((s, s), Image.BILINEAR))
            # 随机翻转 / 旋转 / 颜色抖动：避免模型记住"同一个浅色方块"这个捷径
            if rng.random() < 0.5:
                patch = patch[:, ::-1]
            k = rng.randrange(4)
            if k:
                patch = np.rot90(patch, k)
            gain = rng.uniform(0.8, 1.2)
            bias = rng.uniform(-18, 18)
            patch = np.clip(patch.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
            m = feather_mask(s)[:, :, None]
            region = arr[py:py + s, px:px + s].astype(np.float32)
            blended = patch.astype(np.float32) * m + region * (1 - m)
            arr[py:py + s, px:px + s] = blended.astype(np.uint8)
            pasted += 1
            boxes_pasted.append([px, py, s])
        if not pasted:
            continue
        out_img = out / "images" / "train" / f"{ip.stem}__neg.jpg"
        out_lbl = out / "labels" / "train" / f"{ip.stem}__neg.txt"
        if not args.dry_run:
            out_img.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(out_img, quality=92)
            src_lbl = src_root / "labels" / "train" / f"{ip.stem}.txt"
            if src_lbl.is_file():
                shutil.copy2(src_lbl, out_lbl)
            else:
                out_lbl.write_text("", encoding="utf-8")
        stats["aug"] += 1
        stats["pasted"] += pasted
        aug_meta[out_img.name] = boxes_pasted
        bar.set_postfix(aug=stats["aug"], pasted=stats["pasted"])

    yaml_path = out / "SHWD_neg.yaml"
    if not args.dry_run:
        yaml_path.write_text(
            f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
            f"nc: 2\nnames: ['head', 'helmet']\n", encoding="utf-8")
        (out / "aug_meta.json").write_text(json.dumps(aug_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"链接/复制 {stats['links']} 张；增强 {stats['aug']} 张，共贴 {stats['pasted']} 个贴片")
    print(f"数据集：{out}　yaml：{yaml_path}")
    if args.dry_run:
        print("[dry-run] 没有写任何文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
