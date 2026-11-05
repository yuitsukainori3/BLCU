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
python scripts/eval_report.py --weights ... --split val --no-slices     # 只要 AP，不分层

默认会额外输出“按尺寸 / 按亮度 / 按拥挤度”三张分层表（只算 head），
它内部调用 error_analysis.py 做逐框匹配，结果落在 runs/eval/_slices/<split>/boxes_<split>.csv。

退出码：0 成功；1 参数错误或评测失败。

默认 imgsz 960、split val（与 docs/评测记录.md 的基线一致）；要跑 640 必须显式传 `--imgsz 640`。
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _ui import die, echo_config, require_torch  # noqa: E402

# ==================== 可调参数（只改这里） ====================
DATA_YAML = (ROOT / "ultralytics-8.3.163" / "ultralytics-8.3.163"
             / "ultralytics" / "cfg" / "datasets" / "SHWD.yaml")  # 原始 SHWD 划分
SPLIT = "val"        # 中间对比一律 val；test 只在最终版本跑一次
IMGSZ = 960          # 基线 960！640 的结果与 docs/评测记录.md 不可比
BATCH = 8            # 8 GB 卡 960 输入下的稳妥值
WORKERS = 0          # Windows 下保持 0，多进程读图容易卡住
DEVICE = "0"
EVAL_DIR = ROOT / "runs" / "eval"   # val 的 plots 与结果落这里
# ============================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True, help="权重路径，如 runs/detect/train4/weights/best.pt")
    p.add_argument("--data", type=Path, default=DATA_YAML, help="data.yaml（默认原始 SHWD 划分）")
    p.add_argument("--split", default=SPLIT, choices=("val", "test"), help="评测哪个划分")
    p.add_argument("--imgsz", type=int, default=IMGSZ, help=f"输入尺寸（默认 {IMGSZ} = 基线）")
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--workers", type=int, default=WORKERS, help="Windows 下建议 0，避免多进程读图卡住")
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--append", type=Path, default=None, help="把一行结果追加到这张 markdown 表里")
    p.add_argument("--out", type=Path, default=None, help="把完整报告写到这个文件")
    p.add_argument("--no-slices", action="store_true", help="跳过分层评测（按尺寸/亮度/拥挤度）")
    p.add_argument("--error-csv", type=Path, default=None, help="复用已有的 boxes_<split>.csv，不重新跑匹配")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _run_error_analysis(weights: Path, data: Path, split: str, imgsz: int, device: str) -> Path:
    """调 error_analysis.py 生成逐框 CSV（复用它的匹配逻辑，避免两套实现漂移）。"""
    out_dir = ROOT / "runs" / "eval" / "_slices" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("error_analysis.py")
    cmd = [sys.executable, str(script), "--weights", str(weights), "--data", str(data),
           "--split", split, "--imgsz", str(imgsz), "--device", str(device),
           "--out", str(out_dir), "--save-crops", "0"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return out_dir / f"boxes_{split}.csv"


def _slice_tables(csv_path: Path) -> str:
    """按目标尺寸 / 图像亮度 / 图像拥挤度切三张表，只看 head。"""
    with csv_path.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["gt_cls"] == "0"]
    if not rows:
        return ""

    def agg(sel):
        n = len(sel)
        if not n:
            return None
        ok = sum(1 for r in sel if r["kind"].startswith("A"))
        wrong = sum(1 for r in sel if r["kind"] == "B_误合规")
        return n, ok / n, wrong / n

    def table(title, buckets, key):
        lines = [f"**{title}**", "", fmt_row(["区间", "框数", "违规召回", "误合规"]), fmt_row(["---"] * 4)]
        for label, lo, hi in buckets:
            sel = [r for r in rows if lo <= key(r) < hi]
            s = agg(sel)
            if s:
                lines.append(fmt_row([label, s[0], f"{s[1]:.1%}", f"{s[2]:.1%}"]))
        return "\n".join(lines + [""])

    out = ["分层评测（只算 head）", ""]
    out.append(table("按目标尺寸（px）", [
        ("<15", 0, 15), ("15—30", 15, 30), ("30—60", 30, 60), (">60", 60, 10**9),
    ], lambda r: float(r["w_px"])))
    out.append(table("按图像亮度", [
        ("暗 <80", -1, 80), ("中 80—110", 80, 110), ("亮 >110", 110, 10**9),
    ], lambda r: float(r["brightness"])))
    out.append(table("按图像拥挤度（同图 GT 数）", [
        ("疏 <8", 0, 8), ("中 8—12", 8, 13), ("密 ≥13", 13, 10**9),
    ], lambda r: float(r["n_gt_image"])))
    return "\n".join(out)


def fmt_row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def main(argv=None) -> int:
    args = parse_args(argv)
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        die(f"权重不存在：{weights}")
    if not Path(args.data).expanduser().is_file():
        die(f"data.yaml 不存在：{args.data}")

    if not args.quiet:
        echo_config("评测配置", [
            ("权重", weights),
            ("数据", Path(args.data).expanduser().resolve()),
            ("划分", f"{args.split}    imgsz {args.imgsz}    batch {args.batch}"),
            ("设备", f"cuda:{args.device}    workers {args.workers}"),
            ("产物", EVAL_DIR / f"{weights.parent.parent.name}_{args.split}"),
        ])

    require_torch("评测")
    from ultralytics import YOLO

    # 先跑分层评测（独立子进程，自带一次推理），再在本进程做 val。
    # 顺序不能反：本进程 val 之后会占住显存，子进程再进来就会 CUDA OOM。
    slices = ""
    if not args.no_slices:
        try:
            csv_path = args.error_csv or _run_error_analysis(weights, Path(args.data), args.split,
                                                             args.imgsz, str(args.device))
            slices = _slice_tables(csv_path)
        except Exception as exc:  # 分层评测失败不影响主报告
            print(f"[警告] 分层评测跳过：{type(exc).__name__}: {exc}", file=sys.stderr)

    model = YOLO(str(weights))
    r = model.val(
        data=str(args.data), split=args.split, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, device=args.device, plots=True, verbose=False,
        project=str(EVAL_DIR), name=f"{weights.parent.parent.name}_{args.split}",
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
    if slices:
        report = report + "\n" + slices

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
        if not args.append.exists():
            args.append.write_text("# 评测记录\n\n> 由 scripts/eval_report.py --append 自动追加。\n\n" + header,
                                   encoding="utf-8")
        # 插到「时间/权重」那张表的最后一行之后；文件末尾堆行会跑到正文外面去
        text = args.append.read_text(encoding="utf-8").splitlines()
        start = next((i for i, ln in enumerate(text) if ln.startswith("| 时间 |")), None)
        if start is None:
            text.append(row.rstrip("\n"))
        else:
            end = start + 1
            while end < len(text) and text[end].startswith("|"):
                end += 1
            text.insert(end, row.rstrip("\n"))
        args.append.write_text("\n".join(text) + "\n", encoding="utf-8")
        print(f"[已追加] {args.append}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
