#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一条命令出评测报告：逐类 AP、混淆矩阵、误合规率。

数据口径固定在 SHWD 自己的划分上（train / val / test）：
  - 调参、消融只看 val；
  - test 只在最后跑一次，不要拿它选模型。

指标口径（团队统一）：
  误合规率 = 真值 head 被判成 helmet 的框数 / 真值 head 总框数
  —— 对安全帽系统来说，这是最不能接受的错误方向（没戴判成戴了）。

用法
----
python scripts/eval_report.py --weights runs/detect/train4/weights/best.pt --split val
python scripts/eval_report.py --weights ... --split test
python scripts/eval_report.py --weights ... --split val --append docs/评测记录.md

退出码：0 成功；1 参数错误或评测失败。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

DEFAULT_DATA = Path(r"F:\Competition\ultralytics-8.3.163\ultralytics-8.3.163\ultralytics\cfg\datasets\SHWD.yaml")


def die(msg: str, code: int = 1):
    print(f"[错误] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True, help="权重路径，如 runs/detect/train4/weights/best.pt")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="data.yaml")
    p.add_argument("--split", default="val", choices=("val", "test"), help="评测哪个划分")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=0, help="Windows 下建议 0，避免多进程读图卡住")
    p.add_argument("--device", default="0")
    p.add_argument("--append", type=Path, default=None, help="把一行结果追加到这张 markdown 表里")
    p.add_argument("--out", type=Path, default=None, help="把完整报告写到这个文件")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def fmt_row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def main(argv=None) -> int:
    args = parse_args(argv)
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        die(f"权重不存在：{weights}")
    if not Path(args.data).expanduser().is_file():
        die(f"data.yaml 不存在：{args.data}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    r = model.val(
        data=str(args.data), split=args.split, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, device=args.device, plots=True, verbose=False,
        project=r"F:\Competition\runs\eval", name=f"{weights.parent.parent.name}_{args.split}",
        exist_ok=True,
    )

    names = dict(r.names)                      # {0: 'head', 1: 'helmet'}
    head_id = next((k for k, v in names.items() if v == "head"), 0)
    helmet_id = next((k for k, v in names.items() if v == "helmet"), 1)

    lines: list[str] = []
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"### {args.split}  {stamp}")
    lines.append(f"权重 `{weights.name}`（{weights.parent.parent.name}）　imgsz {args.imgsz}")
    lines.append("")
    lines.append(fmt_row(["类别", "P", "R", "AP50", "AP50-95"]))
    lines.append(fmt_row(["---"] * 5))
    for i, n in names.items():
        lines.append(fmt_row([n, f"{float(r.box.p[i]):.4f}", f"{float(r.box.r[i]):.4f}",
                              f"{float(r.box.ap50[i]):.4f}", f"{float(r.box.ap[i]):.4f}"]))
    lines.append(fmt_row(["**all**", "", "", f"**{float(r.box.map50):.4f}**", f"**{float(r.box.map):.4f}**"]))
    lines.append("")

    false_compliance = None
    cm = getattr(r, "confusion_matrix", None)
    if cm is not None and getattr(cm, "matrix", None) is not None:
        m = cm.matrix
        n = len(names)
        label = [names[i] for i in range(n)] + ["background"]
        lines.append("混淆矩阵（行=预测，列=真值；conf 0.25 / IoU 0.45）")
        lines.append("")
        lines.append(fmt_row(["pred \\ true"] + label))
        lines.append(fmt_row(["---"] * (n + 2)))
        for i, row_name in enumerate(label):
            lines.append(fmt_row([row_name] + [int(m[i, j]) for j in range(n + 1)]))
        lines.append("")

        total_head = sum(int(m[i, head_id]) for i in range(n + 1))
        if total_head:
            wrong = int(m[helmet_id, head_id])
            right = int(m[head_id, head_id])
            false_compliance = wrong / total_head
            lines.append(f"- 违规召回：**{right}/{total_head} = {right / total_head:.1%}**")
            lines.append(f"- 误合规率：**{wrong}/{total_head} = {false_compliance:.1%}**（真值 head 被判定为 helmet）")
            lines.append("")
    else:
        lines.append("> 没拿到混淆矩阵（plots 失败），只有 AP。")
        lines.append("")

    report = "\n".join(lines)
    if not args.quiet:
        print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"[已写入] {args.out}")

    if args.append:
        header = ("| 时间 | 权重 | split | imgsz | mAP50 | mAP50-95 | head AP50 | helmet AP50 | 误合规率 |\n"
                  "|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        row = (f"| {stamp} | {weights.parent.parent.name} | {args.split} | {args.imgsz} | "
               f"{float(r.box.map50):.4f} | {float(r.box.map):.4f} | "
               f"{float(r.box.ap50[head_id]):.4f} | {float(r.box.ap50[helmet_id]):.4f} | "
               f"{'' if false_compliance is None else f'{false_compliance:.1%}'} |\n")
        args.append.parent.mkdir(parents=True, exist_ok=True)
        new = not args.append.exists()
        with args.append.open("a", encoding="utf-8") as f:
            if new:
                f.write("# 评测记录\n\n> 由 scripts/eval_report.py --append 自动追加。\n\n" + header)
            f.write(row)
        print(f"[已追加] {args.append}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
