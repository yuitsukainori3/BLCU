#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把自有数据和 SHWD 拼成一个训练集，自有 val 保持干净。

为什么需要它
------------
自有 105 张 + SHWD 3,218 张 ≈ 3,323 张。两边标签都已经是 `0 head / 1 helmet`，
目录都是 `images/...` ↔ `labels/...` 的标准结构，所以不用搬图片，只要写两个
图片列表 + 一个 data.yaml，Ultralytics 就能直接训练。

注意
----
- SHWD 的 val/test 也进训练（队里决定全用），所以**没有辅助 val 了**，
  唯一的评测集是自有 45 张（10 个裸头框，±0.1 噪声）；
- `--own-repeat` 用来过采样自有数据：105 张对 3,218 张是 1:30，
  重复 5 次后每轮约有 14% 的自有图，模型不会把它当噪声。

用法
----
python build_merged.py                    # 1:1 拼接
python build_merged.py --own-repeat 5     # 自有数据每轮重复 5 次
python build_merged.py --dry-run          # 只统计
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

OWN_DEFAULT = Path(r"F:\Competition\ultralytics-8.3.163\ultralytics-8.3.163\datasets\small_dataset")
SHWD_DEFAULT = Path(r"F:\Competition\datasets\shwd")
DST_DEFAULT = Path(r"F:\Competition\datasets\merged")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def die(msg: str, code: int = 1):
    print(f"[错误] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def collect(img_dir: Path, label_dir: Path, missing: list[Path]) -> list[Path]:
    out = []
    if not img_dir.is_dir():
        die(f"找不到图片目录 {img_dir}")
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in EXTS:
            continue
        if not (label_dir / f"{p.stem}.txt").is_file():
            missing.append(p)
            continue
        out.append(p)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--own-root", type=Path, default=OWN_DEFAULT)
    ap.add_argument("--shwd-root", type=Path, default=SHWD_DEFAULT)
    ap.add_argument("--dst", type=Path, default=DST_DEFAULT)
    ap.add_argument("--own-repeat", type=int, default=1, help="自有训练图每轮重复几次")
    ap.add_argument("--shwd-repeat", type=int, default=1, help="SHWD 图每轮重复几次")
    ap.add_argument("--folds", type=int, default=0,
                    help=">1 时把自有 150 张（train+val 合并）做 k 折交叉验证，配合 --fold 指定第几折")
    ap.add_argument("--fold", type=int, default=0, help="第几折做 val（0 起）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    own, shwd, dst = (p.expanduser().resolve() for p in (args.own_root, args.shwd_root, args.dst))
    missing: list[Path] = []

    own_train = collect(own / "images" / "train", own / "labels" / "train", missing)
    own_val = collect(own / "images" / "val", own / "labels" / "val", missing)
    shwd_imgs: list[Path] = []
    for split in ("train", "val", "test"):
        img_dir = shwd / "images" / split
        if img_dir.is_dir():
            shwd_imgs += collect(img_dir, shwd / "labels" / split, missing)

    if not own_train or not own_val or not shwd_imgs:
        die(f"数据不全：own_train={len(own_train)} own_val={len(own_val)} shwd={len(shwd_imgs)}")
    if missing:
        print(f"[提示] {len(missing)} 张图没有同名标签，已跳过（例：{missing[0].name}）")

    if args.folds and args.folds > 1:
        if not 0 <= args.fold < args.folds:
            die(f"--fold 必须在 0..{args.folds - 1} 之间")
        pool = list(own_train) + list(own_val)
        random.Random(args.seed).shuffle(pool)
        n = len(pool)
        bounds = [round(i * n / args.folds) for i in range(args.folds + 1)]
        val_set = set(pool[bounds[args.fold]: bounds[args.fold + 1]])
        own_fit = [p for p in pool if p not in val_set]
        own_val_used = sorted(val_set)
        tag = f"fold{args.fold}of{args.folds}"
    else:
        own_fit, own_val_used, tag = list(own_train), list(own_val), ""

    train_lines = [str(p) for p in own_fit] * max(1, args.own_repeat)
    train_lines += [str(p) for p in shwd_imgs] * max(1, args.shwd_repeat)
    val_lines = [str(p) for p in own_val_used]

    total = len(train_lines)
    own_share = 100 * len(own_fit) * max(1, args.own_repeat) / total
    print(f"自有 train {len(own_fit)} 张 × {args.own_repeat} = {len(own_fit) * max(1, args.own_repeat)}"
          + (f"（{tag}，来自 150 张交叉验证池）" if tag else ""))
    print(f"SHWD      {len(shwd_imgs)} 张 × {args.shwd_repeat} = {len(shwd_imgs) * max(1, args.shwd_repeat)}")
    print(f"训练集共 {total} 条，自有占比 {own_share:.1f}%")
    print(f"自有 val  {len(own_val_used)} 张（保持干净，不参与训练）")

    if args.dry_run:
        print("\n[dry-run] 没有写任何文件")
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (dst / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    (dst / "data.yaml").write_text(
        "# 由 build_merged.py 生成：自有 + SHWD，标签均为 head=0 / helmet=1\n"
        f"path: {dst.as_posix()}\n"
        "train: train.txt\n"
        "val: val.txt\n\n"
        "nc: 2\n"
        "names: ['head', 'helmet']\n",
        encoding="utf-8",
    )
    print(f"\n输出 {dst}（train.txt / val.txt / data.yaml）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
