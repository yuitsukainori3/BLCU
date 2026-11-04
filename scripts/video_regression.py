#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回归视频跑分：一条命令输出实验规范要求的三个数。

    python scripts/video_regression.py --weights artifacts/weights/helmet_v0.1.0.pt \
        --video datasets/Pexels_11798561_workers_steel_frame_HD.mp4 --conf 0.18 0.25

输出（控制台 + 可选 --json / --out）：

    | 视频 | conf | 手部 ROI 帧占比 | ROI 框数 | head 总数 | 平均单帧耗时 |
    | R1   | 0.18 | 32.5% (239/736) | 305      | 308       | 21.3 ms      |
    | R1   | 0.25 | 19.2% (141/736) | 184      | ...       | 21.3 ms      |

口径（与 docs/实验规范.md 一致）：
- 推理时用很低的原始 conf（默认 0.05）拿全部候选，再按 --conf 逐个阈值过滤；
- 计入 ROI 的条件：类别 head、框宽 ≤ max_box_width、框中心落在任一 ROI 内；
- 帧占比 = 命中帧数 ÷ 总帧数。

验收基线（R1，helmet_v0.1.0，imgsz 960）：conf 0.18 → 32.5%；conf 0.25 → 19.2%。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _ui import die, progress, require_torch  # noqa: E402

ROI_DEFAULT = ROOT / "configs" / "roi.json"


def load_roi(path: Path):
    if not path.is_file():
        die(f"ROI 配置不存在：{path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return cfg


def scaled_rois(cfg: dict, w: int, h: int):
    base_w, base_h = cfg.get("video_size", [w, h])
    sx, sy = w / base_w, h / base_h
    out = []
    for r in cfg.get("rois", []):
        out.append((r["name"], r["x1"] * sx, r["y1"] * sy, r["x2"] * sx, r["y2"] * sy))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--video", type=Path, nargs="+", required=True)
    p.add_argument("--conf", type=float, nargs="+", default=[0.18, 0.25],
                   help="要报告的生效阈值，可多个")
    p.add_argument("--raw-conf", type=float, default=0.05, help="推理时的原始阈值，必须低于所有 --conf")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default="0")
    p.add_argument("--roi", type=Path, default=ROI_DEFAULT)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--max-box-width", type=float, default=None, help="覆盖 ROI 配置里的 max_box_width")
    p.add_argument("--json", type=Path, default=None, help="把明细写成 JSON")
    p.add_argument("--out", type=Path, default=None, help="把 markdown 表写成文件")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        die(f"权重不存在：{weights}")
    for v in args.video:
        if not v.expanduser().is_file():
            die(f"视频不存在：{v}")
    if min(args.conf) <= args.raw_conf:
        die(f"--conf 必须大于 --raw-conf（{args.raw_conf}）")

    require_torch("回归跑分")
    import cv2
    from ultralytics import YOLO

    cfg = load_roi(args.roi.expanduser())
    max_w = args.max_box_width if args.max_box_width is not None else cfg.get("max_box_width", 30)

    model = YOLO(str(weights))
    rows, detail = [], {}

    for video in args.video:
        video = video.expanduser().resolve()
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            die(f"打不开视频：{video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rois = scaled_rois(cfg, W, H)
        if not rois:
            die(f"{args.roi} 里没有 ROI")

        per_conf = {c: {"frames_hit": 0, "boxes_in_roi": 0, "heads": 0} for c in args.conf}
        n = 0
        elapsed = 0.0
        cap_total = total if not args.max_frames else min(total, args.max_frames) if total else args.max_frames
        if not args.quiet:
            print(f"[视频] {video.name}  {total} 帧  {fps:.0f} fps  {W}x{H}", flush=True)
        bar = progress(total=cap_total, desc=video.stem[:22], unit="frame", disable=args.quiet)
        with bar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if args.max_frames and n >= args.max_frames:
                    break
                n += 1
                t0 = time.perf_counter()
                res = model.predict(frame, imgsz=args.imgsz, conf=args.raw_conf,
                                    device=args.device, verbose=False)[0]
                elapsed += time.perf_counter() - t0
                bar.update(1)
                bar.set_postfix(ms=f"{elapsed / max(1, n) * 1000:.1f}")

                heads = []
                for b, c, cf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int),
                                    res.boxes.conf.cpu().numpy()):
                    if int(c) != 0:
                        continue
                    x1, y1, x2, y2 = (float(v) for v in b)
                    heads.append((float(cf), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1))

                for c in args.conf:
                    st = per_conf[c]
                    hit = False
                    for cf, cx, cy, bw in heads:
                        if cf < c:
                            continue
                        st["heads"] += 1
                        if bw > max_w:
                            continue
                        if any(r[1] <= cx <= r[3] and r[2] <= cy <= r[4] for r in rois):
                            st["boxes_in_roi"] += 1
                            hit = True
                    if hit:
                        st["frames_hit"] += 1
        cap.release()

        ms = elapsed / max(1, n) * 1000
        for c in args.conf:
            st = per_conf[c]
            ratio = st["frames_hit"] / max(1, n)
            rows.append({
                "video": video.name, "conf": c, "ratio": ratio,
                "frames_hit": st["frames_hit"], "frames": n,
                "boxes_in_roi": st["boxes_in_roi"], "heads": st["heads"], "ms": ms,
            })
        detail[video.name] = {"frames": n, "fps": fps, "size": [W, H], "ms": ms, "per_conf": per_conf}

    lines = ["| 视频 | conf | 手部 ROI 帧占比 | ROI 框数 | head 总数 | 平均单帧耗时 |",
             "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['video']} | {r['conf']:.2f} | {r['ratio']:.1%} ({r['frames_hit']}/{r['frames']}) "
                     f"| {r['boxes_in_roi']} | {r['heads']} | {r['ms']:.1f} ms |")
    table = "\n".join(lines)

    if not args.quiet:
        print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8")
        print(f"[已写入] {args.out}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"rows": rows, "detail": detail}, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"[已写入] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
