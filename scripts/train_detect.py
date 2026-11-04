#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测模型训练入口（可复现、有日志、默认不覆盖已有 run）。

**训练必须由人明确指令触发**，本脚本自己不会跑。

    # 先看清楚要跑什么（不占 GPU）
    python scripts/train_detect.py --data datasets/shwd_neg/SHWD_neg.yaml --name neg_v1 --dry-run

    # 确认后再真跑
    python scripts/train_detect.py --data datasets/shwd_neg/SHWD_neg.yaml --name neg_v1

默认配置与 960 基线完全一致：yolo11s、960、50 轮、batch -1、cache ram、workers 0、seed 0，
这样对比才只有一个变量。

日志：同时写屏和 `.tmp/<name>.log`，随时可以 `Get-Content .tmp/<name>.log -Tail 20` 看进度。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "ultralytics-8.3.163" / "ultralytics-8.3.163"


class Tee:
    """把 stdout 同时写到文件。"""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def patch_threadpool():
    """沙箱/受限环境下无法创建命名管道，把标签缓存的多线程改串行。

    只影响首次构建 .cache 的速度，不影响训练结果。
    """
    try:
        import ultralytics.data.utils as du
        import ultralytics.data.dataset as ds
        import ultralytics.data.base as db
    except Exception as exc:      # 导入失败就别拦着
        print(f"[警告] 跳过 ThreadPool 兼容补丁：{exc}", file=sys.stderr)
        return

    class SeqPool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def imap(self, func=None, iterable=None, **k):
            for item in iterable:
                yield func(item)

    for m in (du, ds, db):
        if hasattr(m, "ThreadPool"):
            m.ThreadPool = SeqPool


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="数据集 yaml")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=-1)
    p.add_argument("--cache", default="ram", choices=("ram", "disk", "none"))
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", required=True)
    p.add_argument("--project", type=Path, default=ROOT / "runs" / "detect")
    p.add_argument("--exist-ok", action="store_true", help="允许覆盖同名 run（默认不允许）")
    p.add_argument("--dry-run", action="store_true", help="只打印配置，不训练")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data = args.data.expanduser().resolve()
    if not data.is_file():
        print(f"[错误] data.yaml 不存在：{data}", file=sys.stderr)
        return 1
    run_dir = args.project.expanduser().resolve() / args.name
    cache = False if args.cache == "none" else args.cache

    print("=" * 62)
    print("检测训练配置")
    print(f"  数据      {data}")
    print(f"  模型      {args.model}")
    print(f"  imgsz     {args.imgsz}    epochs {args.epochs}    batch {args.batch}")
    print(f"  cache     {cache}        workers {args.workers}    seed {args.seed}")
    print(f"  输出      {run_dir}")
    print(f"  日志      {ROOT / '.tmp' / (args.name + '.log')}")
    print("=" * 62)

    if run_dir.exists() and not args.exist_ok:
        print(f"[错误] {run_dir} 已存在。换个 --name，或加 --exist-ok 覆盖。", file=sys.stderr)
        return 1
    if args.dry_run:
        print("[dry-run] 配置没问题，没有启动训练。")
        return 0

    sys.path.insert(0, str(VENDORED))
    patch_threadpool()
    from ultralytics import YOLO

    tee = Tee(ROOT / ".tmp" / f"{args.name}.log")
    old = sys.stdout
    sys.stdout = tee
    try:
        print(f"\n[开始] {args.name}")
        YOLO(args.model).train(
            data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            cache=cache, workers=args.workers, seed=args.seed,
            name=args.name, project=str(args.project.expanduser().resolve()),
            exist_ok=args.exist_ok, plots=True, val=True,
        )
        print(f"[完成] 权重：{run_dir / 'weights' / 'best.pt'}")
    finally:
        sys.stdout = old
        tee.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
