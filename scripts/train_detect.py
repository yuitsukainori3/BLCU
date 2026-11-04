#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测模型训练入口（可复现、有日志、默认不覆盖已有 run）。

**训练必须由人明确指令触发**，本脚本自己不会跑。

    # 1) 先看清楚要跑什么（不占 GPU）
    python scripts\\train_detect.py --dry-run

    # 2) 确认后再真跑（用的是「可调参数」区的 DATA_YAML / RUN_NAME）
    python scripts\\train_detect.py

    # 3) 临时换数据或换名字，不改文件
    python scripts\\train_detect.py --data datasets\\shwd_neg\\SHWD_neg.yaml --name neg_v2

用哪份数据、run 叫什么，都在下方「可调参数」区的前两行；命令行只是临时覆盖。
超参与 960 基线完全一致：yolo11s、960、50 轮、batch -1、cache ram、workers 0、seed 0，
这样对比才只有一个变量。

进度条：训练中由 ultralytics 自带（与 `ultralytics-8.3.163\\Train.py` 同款两级 tqdm），
本脚本不再包一层。日志：同时写屏和 `.tmp/<name>.log`；进度条走 stderr，不进日志。

规范见 [docs/脚本规范.md](../docs/脚本规范.md)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "ultralytics-8.3.163" / "ultralytics-8.3.163"
sys.path.insert(0, str(ROOT / "scripts"))
from _ui import Tee, die, echo_config, require_torch  # noqa: E402

# ==================== 可调参数（只改这里） ====================
DATA_YAML = ROOT / "datasets" / "shwd_neg" / "SHWD_neg.yaml"  # 这次训练用哪份数据
RUN_NAME = "neg_v1"       # 这次 run 叫什么；结果写到 runs/detect/<这个名字>
MODEL = "yolo11s.pt"      # 预训练权重；换模型只改这一行
EPOCHS = 50               # 轮数；960 基线就是 50，为省时间砍了就没法比
IMGSZ = 960               # 输入尺寸；基线 960，改了不可比
BATCH = -1                # -1 = 自动；显存不够改 8 或 4
CACHE = "ram"             # ram / disk / none；neg_v1 内存吃紧时可换 disk
WORKERS = 0               # Windows 下保持 0，多进程容易卡死
SEED = 0                  # 固定种子，保证两次可比
PROJECT = ROOT / "runs" / "detect"   # run 输出根目录
LOG_DIR = ROOT / ".tmp"              # 日志目录
# ============================================================


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
    """命令行默认值全部取自上面的「可调参数」，这里只做覆盖。"""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA_YAML, help=f"数据集 yaml（默认 {DATA_YAML.name}）")
    p.add_argument("--model", default=MODEL, help=f"预训练权重（默认 {MODEL}）")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--cache", default=CACHE, choices=("ram", "disk", "none"))
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--name", default=RUN_NAME, help=f"run 名（默认 {RUN_NAME}）")
    p.add_argument("--project", type=Path, default=PROJECT)
    p.add_argument("--exist-ok", action="store_true", help="允许覆盖同名 run（默认不允许）")
    p.add_argument("--dry-run", action="store_true", help="只打印配置，不训练")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data = args.data.expanduser().resolve()
    if not data.is_file():
        die(f"data.yaml 不存在：{data}\n  改顶部 DATA_YAML，或用 --data 指定另一份")
    project = args.project.expanduser().resolve()
    run_dir = project / args.name
    cache = False if args.cache == "none" else args.cache
    log_path = LOG_DIR / f"{args.name}.log"

    echo_config("检测训练配置", [
        ("数据", data),
        ("模型", args.model),
        ("imgsz", f"{args.imgsz}    epochs {args.epochs}    batch {args.batch}"),
        ("cache", f"{cache}        workers {args.workers}    seed {args.seed}"),
        ("输出", run_dir),
        ("日志", log_path),
    ])

    if run_dir.exists() and not args.exist_ok:
        die(f"{run_dir} 已存在。换个 --name，或加 --exist-ok 覆盖。")
    if args.dry_run:
        print("[dry-run] 配置没问题，没有启动训练。")
        return 0

    require_torch("训练", hint="只想看配置不训练时加 --dry-run，那条路径不需要 torch。")
    sys.path.insert(0, str(VENDORED))
    patch_threadpool()
    from ultralytics import YOLO   # 导入很慢，放在 dry-run 之后

    with Tee(log_path):
        print(f"\n[开始] {args.name}（进度条为 ultralytics 自带，中止请 Ctrl+C）", flush=True)
        YOLO(args.model).train(
            data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            cache=cache, workers=args.workers, seed=args.seed,
            name=args.name, project=str(project),
            exist_ok=args.exist_ok, plots=True, val=True,
        )
        best = run_dir / "weights" / "best.pt"
        print(f"[完成] 权重：{best}", flush=True)
        print(f"[完成] 指标：{run_dir / 'results.csv'}（对照 docs/评测记录.md 追加一行）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
