#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 YOLO 数据集里抽一小份子集：默认**只搬图片，不动标签**。

默认行为
--------
源：F:\\Competition\\ultralytics-8.3.163\\ultralytics-8.3.163\\datasets\\
    Hard Hat Workers.v14-raw_headclassonly.yolov11  （train 共 4916 张）
量：100 张
分：70 张 -> small_dataset\\images\\train
    30 张 -> small_dataset\\images\\val
标签：默认不搬。源目录里的 .txt 原样保留，small_dataset 下不会出现旧标签；
      团队按自己的规范（person / helmet / head）重新标注。
      需要连旧标签一起搬时加 --with-labels。

移动是“从源目录剪切”，跑一次源数据就少一批图片，不能靠重跑得到同样的结果。
先用 --dry-run 看一眼再真跑。

用法
----
python make_small_dataset.py                     # 默认 100 张，只搬图片
python make_small_dataset.py --dry-run           # 只打印计划，不动文件
python make_small_dataset.py --num 200           # 抽 200 张
python make_small_dataset.py --names person,helmet,head   # 顺手生成 3 类 data.yaml
python make_small_dataset.py --with-labels       # 连旧标签一起搬（一般不用）
python make_small_dataset.py --source-split all  # train/valid/test 混着抽

退出码：0 成功；1 参数或环境错误；2 目标冲突（不会动任何文件）。
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------- 默认路径 --

SRC_DEFAULT = Path(
    r"F:\Competition\ultralytics-8.3.163\ultralytics-8.3.163\datasets"
    r"\Hard Hat Workers.v14-raw_headclassonly.yolov11"
)
DST_DEFAULT = Path(__file__).resolve().parent / "small_dataset"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test")

MANIFEST_NAME = "move_manifest.csv"
MANIFEST_HEADER = ["time", "source_split", "part", "source_image", "dest_image",
                   "label_moved", "source_label", "dest_label"]


# ---------------------------------------------------------------- 小工具 --


def log(msg: str = "") -> None:
    print(msg, flush=True)


def die(msg: str, code: int = 1):
    print(f"[错误] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def count_files(folder: Path) -> int:
    return sum(1 for path in folder.iterdir() if path.is_file()) if folder.is_dir() else 0


# ---------------------------------------------------------------- 收集数据 --


def collect_items(src_root: Path, splits: tuple[str, ...], with_labels: bool):
    """返回 [(图片路径, 标签路径或 None, split)]，以及缺标签的图片（仅 --with-labels）。"""
    items: list[tuple[Path, Path | None, str]] = []
    missing: list[Path] = []

    for split in splits:
        img_dir = src_root / split / "images"
        if not img_dir.is_dir():
            die(f"源目录里找不到 {img_dir}")
        lbl_dir = src_root / split / "labels"
        if with_labels and not lbl_dir.is_dir():
            die(f"--with-labels 需要标签目录，但找不到 {lbl_dir}")

        for img in sorted(img_dir.iterdir(), key=lambda p: p.name.lower()):
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                continue
            if with_labels:
                label = lbl_dir / f"{img.stem}.txt"
                if not label.is_file():
                    missing.append(img)
                    continue
            else:
                label = None
            items.append((img, label, split))

    return items, missing


def pick(items, num: int, train_ratio: float, seed: int):
    """随机抽 num 个并打上 train/val 归属。"""
    if num < 2:
        die("--num 至少为 2，否则分不出 train 和 val")
    if not 0.0 < train_ratio < 1.0:
        die("--train-ratio 必须在 0 和 1 之间")
    if num > len(items):
        die(f"可用的图片只有 {len(items)} 张，不够抽 {num} 张")

    picked = random.Random(seed).sample(items, num)
    # 常规四舍五入（round 在 .5 处是“银行家舍入”，这里不用）
    n_train = int(num * train_ratio + 0.5)
    n_train = max(1, min(num - 1, n_train))  # 两边都至少留 1 张

    plan = []
    for index, (img, label, split) in enumerate(picked):
        target = "train" if index < n_train else "val"
        plan.append((img, label, split, target))
    return plan


# ------------------------------------------------------------ 目标路径规划 --


def build_dest_plan(plan, dst_root: Path):
    """分配目标文件名，返回 [(src_img, src_lbl, dst_img, dst_lbl, split, part)]。"""
    used: dict[str, Path] = {}
    rows = []
    for src_img, src_lbl, split, part in plan:
        stem = src_img.stem
        # 同名文件：换 split 抽的时候可能撞上，加 split 后缀区分
        if stem in used:
            stem = f"{stem}_{split}"
            counter = 2
            while stem in used:
                stem = f"{src_img.stem}_{split}{counter}"
                counter += 1
        used[stem] = src_img

        dst_img = dst_root / "images" / part / f"{stem}{src_img.suffix.lower()}"
        dst_lbl = dst_root / "labels" / part / f"{stem}.txt" if src_lbl is not None else None
        rows.append((src_img, src_lbl, dst_img, dst_lbl, split, part))
    return rows


def find_conflicts(rows):
    """目标位置已存在的文件（真跑前要先处理掉）。"""
    conflicts = []
    for _, _, dst_img, dst_lbl, _, _ in rows:
        if dst_img.exists():
            conflicts.append(dst_img)
        if dst_lbl is not None and dst_lbl.exists():
            conflicts.append(dst_lbl)
    return conflicts


# ---------------------------------------------------------------- 移动文件 --


def move_one(src: Path, dst: Path, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not force:
            raise FileExistsError(str(dst))
        dst.unlink()
    shutil.move(str(src), str(dst))


def move_pair(src_img, src_lbl, dst_img, dst_lbl, force: bool) -> None:
    """图片和标签一起搬；标签搬失败就把图片搬回去，不留半对。"""
    moved: list[tuple[Path, Path]] = []
    try:
        move_one(src_img, dst_img, force)
        moved.append((src_img, dst_img))
        move_one(src_lbl, dst_lbl, force)
        moved.append((src_lbl, dst_lbl))
    except Exception:
        for back_src, back_dst in reversed(moved):
            try:
                if back_dst.exists():
                    shutil.move(str(back_dst), str(back_src))
            except OSError as exc:  # 回滚失败只能报出来，不吞掉
                print(f"[警告] 回滚失败：{back_dst} -> {back_src} ({exc})",
                      file=sys.stderr, flush=True)
        raise


# -------------------------------------------------------------- data.yaml --


def read_classes(src_root: Path) -> list[str]:
    """从源 data.yaml 读类别名；读不出来就退回本数据集已知的 ['head']。"""
    names = None
    src_yaml = src_root / "data.yaml"
    if src_yaml.is_file():
        text = src_yaml.read_text(encoding="utf-8", errors="replace")
        try:
            import yaml  # 环境里有就用，没有就走下面的正则

            data = yaml.safe_load(text) or {}
            names = data.get("names")
        except Exception:
            names = None
        if names is None:
            match = re.search(r"^names\s*:\s*\[(.*?)\]", text, re.M | re.S)
            if match:
                names = [item.strip().strip("'\"") for item in match.group(1).split(",")]
                names = [item for item in names if item]

    if isinstance(names, dict):
        try:
            names = [names[key] for key in sorted(names, key=int)]
        except (TypeError, ValueError):
            names = [names[key] for key in sorted(names)]
    if not isinstance(names, list) or not names:
        names = ["head"]
        print("[提示] 没能从源 data.yaml 读出类别，按 nc=1 names=['head'] 写入",
              file=sys.stderr, flush=True)
    return [str(name) for name in names]


def parse_names(text: str | None) -> list[str] | None:
    if not text:
        return None
    names = [part.strip() for part in text.split(",") if part.strip()]
    return names or None


def write_data_yaml(dst_root: Path, names: list[str]) -> Path:
    yaml_path = dst_root / "data.yaml"
    quoted = ", ".join(f"'{name}'" for name in names)
    body = (
        "# 由 make_small_dataset.py 生成；类别名以标注规范为准，可自行修改\n"
        f"path: {dst_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        f"nc: {len(names)}\n"
        f"names: [{quoted}]\n"
    )
    yaml_path.write_text(body, encoding="utf-8")
    return yaml_path


# ------------------------------------------------------------------ 清单 --


def open_manifest(manifest_path: Path):
    """打开清单；表头对不上（旧版本脚本留下的）就改名备份，另起新文件。"""
    backup = None
    if manifest_path.exists():
        header = manifest_path.open(encoding="utf-8-sig").readline().strip()
        if header != ",".join(MANIFEST_HEADER):
            backup = manifest_path.with_name(
                f"move_manifest.{datetime.now():%Y%m%d_%H%M%S}.bak.csv")
            manifest_path.rename(backup)
            return manifest_path.open("w", newline="", encoding="utf-8-sig"), False, backup
        return manifest_path.open("a", newline="", encoding="utf-8"), True, None
    return manifest_path.open("w", newline="", encoding="utf-8-sig"), False, None


# ------------------------------------------------------------------ 统计 --


def report(dst_root: Path, with_labels: bool) -> bool:
    """核对目标目录，返回是否正常。"""
    ok = True
    log("")
    log("目标目录核对：")
    for part in ("train", "val"):
        img_dir = dst_root / "images" / part
        n_img = count_files(img_dir)
        log(f"  {part:<5} 图片 {n_img:>4} 张  {img_dir}")
        if with_labels:
            n_lbl = count_files(dst_root / "labels" / part)
            if n_img != n_lbl:
                ok = False
                log(f"        [警告] 标签 {n_lbl} 个，和图片不成对")
    if not with_labels:
        stray = sum(count_files(dst_root / "labels" / part) for part in ("train", "val"))
        if stray:
            ok = False
            log(f"  [警告] labels 目录里还有 {stray} 个旧标签文件，训练前请清理")
    return ok


# ------------------------------------------------------------------ 主流程 --


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="从 YOLO 数据集中移动一批图片（默认不动标签），按比例做成小数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src", type=Path, default=SRC_DEFAULT, help="源 YOLO 数据集根目录")
    parser.add_argument("--dst", type=Path, default=DST_DEFAULT, help="小数据集输出根目录")
    parser.add_argument("--num", type=int, default=50, help="一共抽多少张图片")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="train 占比")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，固定后抽样可复现")
    parser.add_argument("--source-split", choices=("train", "valid", "test", "all"),
                        default="train", help="从源的哪些 split 里抽")
    parser.add_argument("--with-labels", action="store_true",
                        help="连旧标签一起移动（默认只搬图片）")
    parser.add_argument("--names", default=None,
                        help="生成 data.yaml 用的类别名，逗号分隔，如 person,helmet,head")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不移动文件")
    parser.add_argument("--force", action="store_true",
                        help="目标已有同名文件时覆盖（默认直接报错，不动任何文件）")
    parser.add_argument("--no-yaml", action="store_true", help="不生成 data.yaml")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    src_root = args.src.expanduser().resolve()
    dst_root = args.dst.expanduser().resolve()

    if not src_root.is_dir():
        die(f"源目录不存在：{src_root}")
    if src_root == dst_root:
        die("源目录和目标目录不能是同一个")

    splits = SPLITS if args.source_split == "all" else (args.source_split,)
    splits = tuple(s for s in splits if (src_root / s / "images").is_dir())
    if not splits:
        die(f"{src_root} 下没有找到可用的 split（train/valid/test）")

    started = time.time()
    items, missing = collect_items(src_root, splits, args.with_labels)

    log(f"源目录 : {src_root}")
    log(f"抽样范围: {', '.join(splits)}，可用图片 {len(items)} 张")
    log(f"标签   : {'随图片一起搬' if args.with_labels else '不动（源目录 .txt 原样保留）'}")
    if missing:
        log(f"[提示] {len(missing)} 张图片没有同名标签，已跳过（例：{missing[0].name}）")

    rows = build_dest_plan(pick(items, args.num, args.train_ratio, args.seed), dst_root)
    n_train = sum(1 for row in rows if row[5] == "train")
    log(f"抽取   : {len(rows)} 张 -> train {n_train} / val {len(rows) - n_train}"
        f"（种子 {args.seed}）")

    conflicts = find_conflicts(rows)
    if conflicts and not args.force:
        log("")
        log(f"[冲突] 目标位置已有 {len(conflicts)} 个同名文件，没有移动任何文件：")
        for path in conflicts[:10]:
            log(f"  {path}")
        if len(conflicts) > 10:
            log(f"  ... 另外 {len(conflicts) - 10} 个")
        log("[提示] 换 --dst，或确认要覆盖后加 --force")
        return 2

    existing = sum(count_files(dst_root / "images" / part) for part in ("train", "val"))
    if existing:
        log(f"[提示] 目标目录里已经有 {existing} 张图片，本次结果会和它们混在一起")

    if args.dry_run:
        log("")
        log("[dry-run] 计划移动：")
        for src_img, _, _, _, split, part in rows:
            log(f"  {part:<5} <- {split:<5} {src_img.name}")
        log("")
        log("[dry-run] 没有移动任何文件。去掉 --dry-run 才会真的搬。")
        return 0

    total_bytes = 0
    done = 0
    manifest_path = dst_root / MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    handle, manifest_exists, backup = open_manifest(manifest_path)
    if backup:
        log(f"[提示] 旧版清单已备份为 {backup.name}")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        writer = csv.writer(handle)
        if not manifest_exists:
            writer.writerow(MANIFEST_HEADER)

        for index, (src_img, src_lbl, dst_img, dst_lbl, split, part) in enumerate(rows, 1):
            size = src_img.stat().st_size
            if src_lbl is not None:
                size += src_lbl.stat().st_size
                move_pair(src_img, src_lbl, dst_img, dst_lbl, args.force)
            else:
                move_one(src_img, dst_img, args.force)
            total_bytes += size
            done += 1
            writer.writerow([stamp, split, part, src_img, dst_img,
                             src_lbl is not None, src_lbl or "", dst_lbl or ""])
            handle.flush()
            if index % 20 == 0 or index == len(rows):
                log(f"  已移动 {index}/{len(rows)} 张")
    except Exception as exc:
        log("")
        log(f"[错误] 第 {done + 1} 张移动失败：{type(exc).__name__}: {exc}")
        log(f"已成功的 {done} 张记录在 {manifest_path}；先检查目标目录，再决定是否重跑")
        report(dst_root, args.with_labels)
        return 1
    finally:
        handle.close()

    names = parse_names(args.names)
    if names is None and args.with_labels:
        names = read_classes(src_root)
    if names and not args.no_yaml:
        yaml_path = write_data_yaml(dst_root, names)
        log(f"data.yaml: {yaml_path}  (nc={len(names)} names={names})")
    elif not args.no_yaml:
        log("[提示] 没搬标签、也没给 --names：不改写 data.yaml（已有文件保持原样）；"
            "要按新规范生成请加 --names person,helmet,head")

    ok = report(dst_root, args.with_labels)
    elapsed = time.time() - started
    log("")
    log(f"搬走图片 {len(rows)} 张，合计 {human_size(total_bytes)}，用时 {elapsed:.1f}s")
    log(f"清单: {manifest_path}")
    remaining_imgs = count_files(src_root / splits[0] / "images") if len(splits) == 1 else "见各 split"
    log(f"源目录剩余图片（{', '.join(splits)}）: {remaining_imgs}")
    if not ok:
        log("[警告] 目标目录有异常，请检查上面的提示")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
