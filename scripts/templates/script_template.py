#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""【模板】新脚本从这份复制：只改「可调参数」和 run_one()，其余照抄。

用法：
    # 1) 先看配置，不干活
    python scripts\\templates\\script_template.py --dry-run

    # 2) 真跑（示例只统计文件大小，用来演示进度条）
    python scripts\\templates\\script_template.py --input datasets\\shwd_neg\\images\\val

    # 3) 只拿前 20 个试水
    python scripts\\templates\\script_template.py --limit 20

输出：
    控制台：配置回显 + 进度条 + 一行汇总
    日志：  .tmp\\script_template.log（--no-log 关掉）

规范见 docs\\脚本规范.md。复制后记得把 docstring、CONFIG 标题、日志名一起改掉。
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _ui import die, echo_config, progress, Tee  # noqa: E402

# ==================== 可调参数（只改这里） ====================
INPUT_DIR = ROOT / "datasets" / "shwd_neg" / "images" / "val"  # 默认输入目录
PATTERN = "*.jpg"        # 匹配哪些文件；换数据集时改这里
RECURSIVE = False        # True = 连子目录一起扫
LIMIT = 0                # 最多处理几个；0 = 不限制（试水时设 20）
SLEEP_MS = 5.0           # [演示] 每个文件的模拟耗时；真脚本换成真实推理，删掉这行
LOG_FILE = ROOT / ".tmp" / "script_template.log"  # 追加写；不想要就 --no-log
# ============================================================


def parse_args(argv=None):
    """命令行只做「覆盖默认值」，默认值全部来自上面的 CONFIG，不重复写一遍。"""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=INPUT_DIR, help=f"输入目录（默认 {INPUT_DIR}）")
    p.add_argument("--pattern", default=PATTERN, help=f"文件通配符（默认 {PATTERN}）")
    p.add_argument("--recursive", action="store_true", default=RECURSIVE, help="连子目录一起扫")
    p.add_argument("--limit", type=int, default=LIMIT, help="最多处理几个，0 = 不限制")
    p.add_argument("--sleep-ms", type=float, default=SLEEP_MS, help="[演示] 每个文件的模拟耗时")
    p.add_argument("--log", type=Path, default=LOG_FILE, help="日志文件（追加写）")
    p.add_argument("--no-log", action="store_true", help="不写日志")
    p.add_argument("--dry-run", action="store_true", help="只打印配置，不处理文件")
    return p.parse_args(argv)


def run_one(path: Path, sleep_ms: float) -> int:
    """【改这里】单条处理逻辑。模板里只是 sleep + 返回字节数，当占位用。"""
    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)
    return path.stat().st_size


def main(argv=None) -> int:
    args = parse_args(argv)
    src = args.input.expanduser().resolve()
    if not src.is_dir():
        die(f"输入目录不存在：{src}")

    found = src.rglob(args.pattern) if args.recursive else src.glob(args.pattern)
    files = sorted(p for p in found if p.is_file())
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    echo_config("脚本名（改这里）", [
        ("输入", src),
        ("通配符", args.pattern),
        ("待处理", f"{len(files)} 个文件"),
        ("限制", args.limit or "不限"),
        ("日志", "（关闭）" if args.no_log else args.log),
    ])

    if not files:
        die(f"{src} 下没有匹配 {args.pattern} 的文件")
    if args.dry_run:
        print("[dry-run] 配置没问题，没有处理任何文件。")
        return 0

    log = nullcontext() if args.no_log else Tee(args.log)
    total_bytes = 0
    with log:
        print(f"\n[开始] {len(files)} 个文件", flush=True)
        t0 = time.perf_counter()
        bar = progress(files, desc="处理", unit="file")
        with bar:
            for p in bar:
                total_bytes += run_one(p, args.sleep_ms)
                bar.set_postfix(total=f"{total_bytes / 1e6:.1f}MB")
        elapsed = time.perf_counter() - t0
        print(f"[完成] {len(files)} 个文件，{total_bytes / 1e6:.2f} MB，"
              f"用时 {elapsed:.1f} s（{len(files) / max(elapsed, 1e-9):.1f} file/s）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
