#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 SHWD / VOC2028 的 Pascal VOC 标注转成本项目的 YOLO 两类格式。

背景
----
SHWD（又名 VOC2028）共 7,581 张图，两类：
    hat     = 戴在头上的安全帽      -> 本项目 class 1 (helmet)
    person  = 没戴帽的头（裸头）    -> 本项目 class 0 (head)
注意它的 `person` 不是"人"，是"头"。

实测发现（2026-11-01，见 docs/数据集调研.md）：
  - 3,219 张图带安全帽（9,047 个 hat 框 + 1,453 个裸头框），场景以工地/作业现场为主；
  - 4,362 张图没有任何 hat 框，却有 110,061 个裸头框，基本来自 SCUT-HEAD 教室人头，
    和工地差异很大，而且抽查发现有人戴着白色安全帽也被标成 person（错标）。
所以默认只转"带安全帽"的那 3,219 张；要用全量加 --include-no-hat。

不复制图片
----------
标签写进 dst/labels/<split>/*.txt，图片列表写进 dst/<split>.txt（绝对路径），
data.yaml 直接指向列表文件，Ultralytics 能直接读，不额外占几百 MB。

用法
----
python convert_shwd.py                       # 默认：只转带 hat 的图，写到 F:\Competition\datasets\shwd
python convert_shwd.py --materialize hardlink  # 图片也放进 images/<split>（同盘硬链接，不占空间）
python convert_shwd.py --include-no-hat      # 全量（含 SCUT-HEAD 部分）
python convert_shwd.py --min-short 200       # 丢掉短边 < 200px 的小图
python convert_shwd.py --drop-hat-area 0.5   # 丢掉"单个 hat 框 > 半个画面"的产品特写
python convert_shwd.py --dry-run             # 只统计，不写文件

退出码：0 成功；1 参数/环境错误。
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SRC_DEFAULT = Path(r"F:\Competition\0make_dataset\VOC2028\VOC2028")
DST_DEFAULT = Path(r"F:\Competition\datasets\shwd")

SPLITS = ("train", "val", "test")
CLS_HEAD, CLS_HELMET = 0, 1
NAME_MAP = {"hat": CLS_HELMET, "person": CLS_HEAD}
CLASS_NAMES = ["head", "helmet"]


def die(msg: str, code: int = 1):
    print(f"[错误] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=SRC_DEFAULT, help="VOC2028 根目录（含 Annotations/JPEGImages/ImageSets）")
    p.add_argument("--dst", type=Path, default=DST_DEFAULT, help="输出目录")
    p.add_argument("--include-no-hat", action="store_true", help="连没有 hat 框的图一起转（默认只转带 hat 的）")
    p.add_argument("--min-short", type=int, default=0, help="短边小于该值的图跳过，0 = 不限制")
    p.add_argument("--drop-hat-area", type=float, default=0.0,
                   help="单个 hat 框面积超过整图该比例就跳过整张图（0 = 不启用，0.5 可去产品特写）")
    p.add_argument("--preview", type=int, default=6, help="输出几张画好框的预览图，0 = 不输出")
    p.add_argument("--materialize", choices=("none", "hardlink", "copy"), default="none",
                   help="把图片也放进 dst/images/<split>/：hardlink 不占空间（同盘推荐），copy 真复制")
    p.add_argument("--dry-run", action="store_true", help="只统计，不写任何文件")
    p.add_argument("--seed", type=int, default=0, help="预览图抽样种子")
    return p.parse_args(argv)


def read_split(split_file: Path) -> list[str]:
    if not split_file.is_file():
        return []
    return [line.strip() for line in split_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()]


def main(argv=None) -> int:
    args = parse_args(argv)
    src: Path = args.src.expanduser().resolve()
    dst: Path = args.dst.expanduser().resolve()

    ann_dir = src / "Annotations"
    img_dir = src / "JPEGImages"
    sets_dir = src / "ImageSets" / "Main"
    if not ann_dir.is_dir() or not img_dir.is_dir():
        die(f"{src} 下找不到 Annotations 或 JPEGImages")

    images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    print(f"源目录 : {src}")
    print(f"图片   : {len(images)} 张，标注 {len(list(ann_dir.glob('*.xml')))} 个 xml")

    # 划分：优先用 ImageSets，缺了就把所有图都当 train
    split_of: dict[str, str] = {}
    for split in SPLITS:
        for stem in read_split(sets_dir / f"{split}.txt"):
            split_of.setdefault(stem, split)
    if not split_of:
        print("[提示] 没读到 ImageSets，全部按 train 处理")

    # 转换
    stats = collections.Counter()
    per_split = collections.Counter()
    skipped = collections.Counter()
    rows: list[tuple[str, str, Path, list[str]]] = []

    for stem, img_path in sorted(images.items()):
        xml_path = ann_dir / f"{stem}.xml"
        if not xml_path.is_file():
            skipped["no_xml"] += 1
            continue
        root = ET.parse(xml_path).getroot()
        W = float(root.findtext("size/width") or 0)
        H = float(root.findtext("size/height") or 0)
        if W <= 0 or H <= 0:
            skipped["bad_size"] += 1
            continue
        if args.min_short and min(W, H) < args.min_short:
            skipped["too_small"] += 1
            continue

        objs = []
        hat_area_max = 0.0
        for o in root.findall("object"):
            name = (o.findtext("name") or "").strip().lower()
            if name not in NAME_MAP:
                skipped[f"class_{name or 'empty'}"] += 1
                continue
            b = o.find("bndbox")
            if b is None:
                continue
            x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            x1, x2 = max(0.0, min(x1, x2)), min(W, max(x1, x2))
            y1, y2 = max(0.0, min(y1, y2)), min(H, max(y1, y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                skipped["tiny_box"] += 1
                continue
            cid = NAME_MAP[name]
            objs.append((cid, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
            if cid == CLS_HELMET:
                hat_area_max = max(hat_area_max, (x2 - x1) * (y2 - y1) / (W * H))

        has_hat = any(c == CLS_HELMET for c, *_ in objs)
        if not has_hat and not args.include_no_hat:
            skipped["no_hat"] += 1
            continue
        if args.drop_hat_area and hat_area_max > args.drop_hat_area:
            skipped["hat_too_big"] += 1
            continue
        if not objs:
            skipped["no_box_left"] += 1
            continue

        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in objs]
        split = split_of.get(stem, "train")
        rows.append((stem, split, img_path, lines))
        for c, *_ in objs:
            stats[CLASS_NAMES[c]] += 1
        per_split[split] += 1
        if has_hat:
            stats["imgs_with_hat"] += 1
        else:
            stats["imgs_no_hat"] += 1

    print(f"\n保留图片 {len(rows)} 张")
    for split in SPLITS:
        print(f"  {split:5s} {per_split[split]:5d} 张")
    print(f"  类别  head {stats['head']}  helmet {stats['helmet']}")
    print(f"  其中带 hat 的图 {stats['imgs_with_hat']} 张，不带 {stats['imgs_no_hat']} 张")
    print(f"  跳过：{dict(skipped)}")

    if args.dry_run:
        print("\n[dry-run] 没有写任何文件")
        return 0

    if not rows:
        die("没有可写的样本，检查过滤参数")

    # 写文件
    for split in SPLITS:
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
    lists: dict[str, list[str]] = {s: [] for s in SPLITS}
    for stem, split, img_path, lines in rows:
        (dst / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        lists[split].append(str(img_path))
    for split in SPLITS:
        (dst / f"{split}.txt").write_text("\n".join(lists[split]) + "\n", encoding="utf-8")

    # 可选：把图片也放进 dst/images/<split>/。hardlink 在同一块盘上不占额外空间。
    if args.materialize != "none":
        linked = copied = existed = 0
        for stem, split, img_path, _ in rows:
            target_dir = dst / "images" / split
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / img_path.name
            if target.exists():
                existed += 1
                continue
            try:
                if args.materialize == "hardlink":
                    os.link(img_path, target)
                    linked += 1
                else:
                    shutil.copy2(img_path, target)
                    copied += 1
            except OSError:  # 跨盘或不支持硬链接时退回复制
                shutil.copy2(img_path, target)
                copied += 1
        print(f"图片：hardlink {linked}，copy {copied}，已存在 {existed} -> {dst / 'images'}")

    names = ", ".join(f"'{n}'" for n in CLASS_NAMES)
    if args.materialize != "none":
        train_ref, val_ref, test_ref = "images/train", "images/val", "images/test"
    else:
        train_ref, val_ref, test_ref = "train.txt", "val.txt", "test.txt"
    (dst / "data.yaml").write_text(
        "# 由 convert_shwd.py 生成：SHWD/VOC2028 -> YOLO，hat=helmet(1), person=head(0)\n"
        f"path: {dst.as_posix()}\n"
        f"train: {train_ref}\n"
        f"val: {val_ref}\n"
        f"test: {test_ref}\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: [{names}]\n",
        encoding="utf-8",
    )

    # 预览图：直接把刚写出的 txt 读回来画，验证转换没错
    if args.preview > 0:
        try:
            from PIL import Image, ImageDraw

            random.Random(args.seed).shuffle(rows)
            out_dir = dst / "preview"
            out_dir.mkdir(parents=True, exist_ok=True)
            colors = {CLS_HEAD: (255, 40, 40), CLS_HELMET: (0, 200, 0)}
            for stem, split, img_path, _ in rows[: args.preview]:
                im = Image.open(img_path).convert("RGB")
                W, H = im.size
                d = ImageDraw.Draw(im)
                for line in (dst / "labels" / split / f"{stem}.txt").read_text(encoding="utf-8").splitlines():
                    c, x, y, w, h = line.split()
                    c = int(c); x, y, w, h = map(float, (x, y, w, h))
                    d.rectangle([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H],
                                outline=colors[c], width=3)
                im.save(out_dir / f"{split}_{stem}.jpg", quality=85)
            print(f"预览图 {min(args.preview, len(rows))} 张 -> {out_dir}")
        except Exception as exc:  # 预览失败不影响数据
            print(f"[警告] 预览图生成失败：{type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"\n输出目录 {dst}")
    if args.materialize != "none":
        print("  data.yaml / images/{train,val,test} / labels/{train,val,test} / preview/")
    else:
        print("  data.yaml / train.txt / val.txt / test.txt / labels/ / preview/")
    print("  训练：把 data 指向这个 data.yaml 即可")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
