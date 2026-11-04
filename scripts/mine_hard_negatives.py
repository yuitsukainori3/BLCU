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
sys.path.insert(0, str(ROOT / "scripts"))
from _ui import die, echo_config, progress, require_torch  # noqa: E402

# ==================== 可调参数（只改这里） ====================
ROI_JSON = ROOT / "configs" / "roi.json"                  # 回归 ROI（只增不减）
OUT_DIR = ROOT / "datasets" / "negatives" / "hands"       # 贴片输出
CONF = 0.18           # 与展示系统一致的 head 阈值（负样本就在这个工作点上挖）
RAW_CONF = 0.05       # 推理时的原始阈值，必须低于 CONF
IMGSZ = 960
DEVICE = "0"
PAD = 1.5             # 裁片外扩倍数（带一点上下文）
PER_CLUSTER = 10      # 每个簇最多取几张
MIN_GAP = 20          # 同一簇内两张贴片的最小帧间隔
GRID = 40             # 聚簇网格边长（px）
# ============================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--roi", type=Path, default=ROI_JSON)
    p.add_argument("--conf", type=float, default=CONF, help="与展示系统一致的 head 阈值")
    p.add_argument("--raw-conf", type=float, default=RAW_CONF)
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--pad", type=float, default=PAD, help="裁片外扩倍数（带一点上下文）")
    p.add_argument("--per-cluster", type=int, default=PER_CLUSTER)
    p.add_argument("--min-gap", type=int, default=MIN_GAP, help="同一簇内两张贴片的最小帧间隔")
    p.add_argument("--grid", type=int, default=GRID)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.weights.expanduser().is_file():
        die(f"权重不存在：{args.weights}")
    if not args.video.expanduser().is_file():
        die(f"视频不存在：{args.video}")
    if not args.roi.is_file():
        die(f"ROI 配置不存在：{args.roi}")

    require_torch("难例挖掘")
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    sx, sy = W / base_w, H / base_h
    rois = [(r["x1"] * sx, r["y1"] * sy, r["x2"] * sx, r["y2"] * sy) for r in cfg.get("rois", [])]

    echo_config("难例挖掘配置", [
        ("权重", args.weights.expanduser().resolve()),
        ("视频", f"{args.video.expanduser().resolve()}  {W}x{H}  {total_frames or '?'} 帧"),
        ("ROI", f"{args.roi}（{len(rois)} 个）"),
        ("阈值", f"conf {args.conf}（原始 {args.raw_conf}）    imgsz {args.imgsz}"),
        ("聚簇", f"网格 {args.grid}px    每簇 ≤{args.per_cluster} 张    最小间隔 {args.min_gap} 帧"),
        ("输出", args.out.expanduser().resolve()),
    ])

    cands = []  # (frame, conf, x1,y1,x2,y2)
    idx = 0
    bar = progress(total=total_frames, desc="扫描命中", unit="frame")
    with bar:
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
            bar.update(1)
            bar.set_postfix(hits=len(cands))
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
    bar = progress(total=total_frames, desc="裁图", unit="frame")
    with bar:
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
            bar.update(1)
            bar.set_postfix(saved=saved)
    cap.release()

    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"挖到 {len(cands)} 个候选，聚成 {len(clusters)} 簇，保存 {saved} 张贴片 -> {out_dir}")
    for m in meta[:8]:
        print(f"  {m['file']}  conf {m['conf']}  size {m['size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
