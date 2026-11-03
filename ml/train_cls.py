#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练两阶段里的“头部复核分类器”。

输入是已经按类别分好文件夹的裁片：

    datasets/head_cls/{train,val}/{head,helmet}/*.jpg

分类任务不需要 .txt 标签，**文件夹名就是标签**。

用法
----
python ml/train_cls.py                          # 默认 30 轮，写到 runs/classify/cls_224
python ml/train_cls.py --epochs 1 --name smoke  # 冒烟测试
python ml/train_cls.py --epochs 50 --name cls_v2

训练完会自动在 val 上再验一次，把 top1 / top5 打印出来，并在 run 目录写一份 summary.md。
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DEFAULT = ROOT / "datasets" / "head_cls"
MODEL_DEFAULT = ROOT / "ultralytics-8.3.163" / "ultralytics-8.3.163" / "yolo11n-cls.pt"
PROJECT_DEFAULT = ROOT / "runs" / "classify"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA_DEFAULT)
    p.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=0, help="Windows 下建议 0")
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="cls_224")
    p.add_argument("--project", type=Path, default=PROJECT_DEFAULT)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=15, help="验证集不再提升就早停")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.data.is_dir():
        raise SystemExit(f"[错误] 数据目录不存在：{args.data}")
    if not (args.data / "train").is_dir():
        raise SystemExit(f"[错误] 找不到 {args.data / 'train'}，目录结构应为 train/<类别>/*.jpg")
    if not args.model.is_file():
        raise SystemExit(f"[错误] 预训练权重不存在：{args.model}")

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        lr0=args.lr0,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
        cache="ram",     # 裁片只有 78 MB，放内存完全够
        plots=True,
        seed=0,
    )

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"

    # 训练完再单独验一次，拿到干净的 top1 / top5
    top1 = top5 = None
    try:
        m = YOLO(str(best))
        r = m.val(data=str(args.data), imgsz=args.imgsz, batch=args.batch,
                  workers=args.workers, device=args.device, plots=False)
        top1, top5 = float(r.top1), float(r.top5)
    except Exception as exc:
        print(f"[警告] 复验失败：{type(exc).__name__}: {exc}")

    lines = [
        "# 分类器训练记录",
        "",
        f"- 时间：{_dt.datetime.now():%Y-%m-%d %H:%M}",
        f"- 数据：`{args.data}`",
        f"- 模型：`{args.model.name}`",
        f"- imgsz {args.imgsz}　epochs {args.epochs}　batch {args.batch}　lr0 {args.lr0}",
        f"- 权重：`{best}`",
        "",
    ]
    if top1 is not None:
        lines += ["| 指标 | 值 |", "|---|---:|", f"| top1 | {top1:.4f} |", f"| top5 | {top5:.4f} |", ""]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n训练完成：{best}")
    if top1 is not None:
        print(f"val top1 = {top1:.4f}   top5 = {top5:.4f}")
    print(f"记录：{run_dir / 'summary.md'}")
    print("下一步：把 best.pt 路径交给 ml/two_stage.py 做路由与覆盖")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
