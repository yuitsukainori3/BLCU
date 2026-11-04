#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从回归视频里挖"难例贴片"——已经被确认误检的区域。

    python scripts/mine_hard_negatives.py --weights artifacts/weights/helmet_v0.1.0.pt ^
        --video datasets/Pexels_11798561_workers_steel_frame_HD.mp4 --out datasets/negatives/hands

做什么
------
1. 逐帧推理，按 ROI 配置取出"框宽小、中心落在 ROI 内、conf 达标"的 head 检测；
2. 按 40px 网格聚簇，每个簇按时间抽帧（默认至少间隔 20 帧），每簇最多取 --per-cluster 张；
3. 外扩 --pad 倍裁出贴片，存成 PNG，并写 meta.json（来源视频、帧号、坐标、置信度）。

这些贴片是"确定不是头"的区域，后续由 build_negative_aug.py 贴回训练图，教模型别报。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROI_DEFAULT = ROOT / "configs" / "roi.json"


def die(msg: str):
    print(f"[错误] {msg}", file=sys.stderr)
    raise SystemExit(1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "datasets" / "negatives" / "hands")
    p.add_argument("--roi", type=Path, default=ROI_DEFAULT)
    p.add_argument("--conf", type=float, default=0.18, help="与展示系统一致的 head 阈值")
    p.add_argument("--raw-conf", type=float, default=0.05)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default="0")
    p.add_argument("--pad", type=float, default=1.5, help="裁片外扩倍数（带一点上下文）")
    p.add_argument("--per-cluster", type=int, default=10)
    p.add_argument("--min-gap", type=int, default=20, help="同一簇内两张贴片的最小帧间隔")
    p.add_argument("--grid", type=int, default=40)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.weights.expanduser().is_file():
        die(f"权重不存在：{args.weights}")
    if not args.video.expanduser().is_file():
        die(f"视频不存在：{args.video}")
    if not args.roi.is_file():
        die(f"ROI 配置不存在：{args.roi}")

    import cv2
    from ultralytics import YOLO

    cfg = json.loads(args.roi.read_text(encoding="utf-8"))
    base_w, base_h = cfg.get("video_size", [1280, 720])
    max_w = cfg.get("max_box_width", 30)

    model = YOLO(str(args.weights.expanduser()))
    cap = cv2.VideoCapture(str(args.video.expanduser()))
    if not cap.isOpened():
        die("打不开视频")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = W / base_w, H / base_h
    rois = [(r["x1"] * sx, r["y1"] * sy, r["x2"] * sx, r["y2"] * sy) for r in cfg.get("rois", [])]

    cands = []  # (frame, conf, x1,y1,x2,y2)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, imgsz=args.imgsz, conf=args.raw_conf, device=args.device, verbose=False)[0]
        for b, c, cf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int),
                            res.boxes.conf.cpu().numpy()):
            if int(c) != 0 or float(cf) < args.conf:
                continue
            x1, y1, x2, y2 = (float(v) for v in b)
            if (x2 - x1) > max_w:
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rois):
                cands.append((idx, float(cf), x1, y1, x2, y2))
        idx += 1
    cap.release()

    # 聚簇：按 40px 网格
    clusters: dict[tuple, list] = {}
    for c in cands:
        key = (int(c[2] // args.grid), int(c[3] // args.grid))
        clusters.setdefault(key, []).append(c)

    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    picked, meta = [], []
    for key, items in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        items.sort(key=lambda x: x[0])
        last = -10**9
        n = 0
        for frame_id, cf, x1, y1, x2, y2 in items:
            if frame_id - last < args.min_gap:
                continue
            last = frame_id
            bw, bh = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hw, hh = bw * args.pad / 2, bh * args.pad / 2
            box = (max(0, int(cx - hw)), max(0, int(cy - hh)), min(W, int(cx + hw)), min(H, int(cy + hh)))
            picked.append((frame_id, cf, box))
            n += 1
            if n >= args.per_cluster:
                break

    if not picked:
        die("没有挖到贴片——检查 ROI 与阈值")

    # 第二遍读帧，裁图
    cap = cv2.VideoCapture(str(args.video.expanduser()))
    want = {p[0]: p for p in picked}
    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            frame_id, cf, (x1, y1, x2, y2) = want[idx]
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                name = f"{args.video.stem}_f{frame_id:04d}.png"
                cv2.imwrite(str(out_dir / name), crop)
                meta.append({"file": name, "video": args.video.name, "frame": frame_id,
                             "conf": round(cf, 3), "box": [x1, y1, x2, y2],
                             "size": [x2 - x1, y2 - y1]})
                saved += 1
        idx += 1
    cap.release()

    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"挖到 {len(cands)} 个候选，聚成 {len(clusters)} 簇，保存 {saved} 张贴片 -> {out_dir}")
    for m in meta[:8]:
        print(f"  {m['file']}  conf {m['conf']}  size {m['size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
