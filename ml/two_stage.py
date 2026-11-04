#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""两阶段判定：检测 → 按尺寸路由 → 裁片复核 → 覆盖，并给出端到端对比。

默认同时算两组结果，方便直接看"两阶段到底有没有用"：

    基线     ：只用检测器的类别
    两阶段   ：按尺寸路由 + 分类器覆盖（margin 可调）

指标口径与 eval_report 一致：conf 0.25 工作点、IoU 0.45 匹配。
注意：本脚本算的是"端到端判定指标"，不是 mAP；mAP 仍用 eval_report。

用法
----
python ml/two_stage.py --det artifacts/weights/helmet_v0.1.0.pt ^
    --cls runs/classify/cls_v1/weights/best.pt --split val --imgsz 960 --margin 0.2
python ml/two_stage.py ... --margin 0.1
python ml/two_stage.py ... --max-images 40      # 快速试跑
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DEFAULT = ROOT / "ultralytics-8.3.163" / "ultralytics-8.3.163" / "ultralytics" / "cfg" / "datasets" / "SHWD.yaml"


def die(msg: str):
    print(f"[错误] {msg}", file=sys.stderr)
    raise SystemExit(1)


def read_yaml(path: Path) -> dict:
    out, section = {}, None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith((" ", "\t")) and section:
            k, _, v = line.partition(":")
            out.setdefault(section, {})[k.strip()] = v.strip().strip("'\"")
            continue
        k, _, v = line.partition(":")
        section = k.strip() if not v.strip() else None
        out[k.strip()] = v.strip().strip("'\"")
    return out


def load_gt(label_path: Path, w: int, h: int) -> np.ndarray:
    rows = []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            x, y, bw, bh = map(float, p[1:5])
            rows.append([c, (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    return np.array(rows, dtype=float).reshape(-1, 5)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def route(width: float, small: float, large: float) -> str:
    if width < small:
        return "small"
    return "mid" if width < large else "large"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--det", type=Path, required=True, help="检测权重")
    p.add_argument("--cls", type=Path, required=True, help="分类权重（head/helmet，顺序必须一致）")
    p.add_argument("--data", type=Path, default=DATA_DEFAULT)
    p.add_argument("--split", default="val", choices=("val", "test", "train"))
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.15, help="候选检测阈值（比评测工作点低）")
    p.add_argument("--work-conf", type=float, default=0.25, help="统计用的工作点，与评测口径一致")
    p.add_argument("--iou", type=float, default=0.45, help="匹配 IoU")
    p.add_argument("--margin", type=float, default=0.2, help="分类器把握度阈值 |p-0.5|")
    p.add_argument("--one-way", action="store_true",
                   help="只允许 helmet→head 的改判（不把 head 改成 helmet），用于压低误合规")
    p.add_argument("--small", type=float, default=30.0, help="小于该宽度不送分类器")
    p.add_argument("--large", type=float, default=60.0, help="大于该宽度分类器优先")
    p.add_argument("--pad", type=float, default=1.3, help="裁片外扩倍数，与训练一致")
    p.add_argument("--device", default="0")
    p.add_argument("--max-images", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    for p in (args.det, args.cls):
        if not p.expanduser().is_file():
            die(f"权重不存在：{p}")

    from ultralytics import YOLO

    cfg = read_yaml(args.data.expanduser().resolve())
    root = Path(cfg["path"])
    img_dir = root / cfg[args.split]
    img_files = sorted(x for x in img_dir.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.max_images:
        img_files = img_files[: args.max_images]
    if not img_files:
        die(f"{img_dir} 下没有图片")

    det = YOLO(str(args.det))
    cls = YOLO(str(args.cls))
    if dict(cls.names) != {0: "head", 1: "helmet"}:
        die(f"分类器类别顺序不一致：{cls.names}，应为 {{0: 'head', 1: 'helmet'}}")

    stats = {tag: dict(tp=0, fc=0, miss=0, fp=0, hel_ok=0, hel_bad=0) for tag in ("base", "two")}
    changed = correct = wrong = 0
    by_route = {"small": 0, "mid": 0, "large": 0}
    head_by_size = {tag: {k: [0, 0] for k in ("<15", "15-30", "30-60", ">60")}
                    for tag in ("base", "two")}

    for img_path in img_files:
        res = det.predict(str(img_path), imgsz=args.imgsz, conf=args.conf, iou=0.7,
                          max_det=300, device=args.device, verbose=False)[0]
        frame = res.orig_img
        h, w = frame.shape[:2]
        gts = load_gt(root / "labels" / args.split / f"{img_path.stem}.txt", w, h)

        boxes = res.boxes.xyxy.cpu().numpy()
        det_cls = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()

        # ---- 两阶段：路由 + 复核 ----
        final = det_cls.copy()
        crops, idx = [], []
        for i, b in enumerate(boxes):
            if route(b[2] - b[0], args.small, args.large) != "small":
                x1, y1, x2, y2 = b
                bw, bh = (x2 - x1) * args.pad, (y2 - y1) * args.pad
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                X1, Y1 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
                X2, Y2 = min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2))
                if X2 - X1 >= 8 and Y2 - Y1 >= 8:
                    crops.append(frame[Y1:Y2, X1:X2])
                    idx.append(i)

        if crops:
            probs = cls.predict(crops, imgsz=224, device=args.device, verbose=False)
            for j, i in enumerate(idx):
                p = probs[j].probs.data.cpu().numpy()
                r = route(boxes[i][2] - boxes[i][0], args.small, args.large)
                by_route[r] += 1
                if abs(p[1] - 0.5) > args.margin:
                    new = 1 if p[1] > 0.5 else 0
                    if args.one_way and new == 1:
                        continue          # 单向：不允许把 head 改成 helmet
                    if new != det_cls[i]:
                        if gts.size:
                            gi = int(np.argmax(iou_matrix(boxes[i][None, :], gts[:, 1:5])[0]))
                            ok = iou_matrix(boxes[i][None, :], gts[gi:gi + 1, 1:5])[0, 0] >= args.iou
                            if ok and new == int(gts[gi, 0]):
                                correct += 1
                            elif ok:
                                wrong += 1
                        changed += 1
                        final[i] = new

        # ---- 评估：对每个 GT 找最佳匹配（按尺寸分档统计全部 head GT，含未匹配）----
        for tag, preds_cls in (("base", det_cls), ("two", final)):
            keep = confs >= args.work_conf
            pb, pc = boxes[keep], preds_cls[keep]
            m = iou_matrix(pb, gts[:, 1:5]) if len(pb) and len(gts) else np.zeros((len(pb), len(gts)))
            used_p, used_g = set(), set()
            for gi in range(len(gts)):
                gc = int(gts[gi, 0])
                span = gts[gi, 3] - gts[gi, 1]
                k = "<15" if span < 15 else "15-30" if span < 30 else "30-60" if span < 60 else ">60"
                matched = False
                if len(pb):
                    pi = int(np.argmax(m[:, gi]))
                    if m[pi, gi] >= args.iou and pi not in used_p:
                        used_p.add(pi)
                        used_g.add(gi)
                        matched = True
                        if gc == 0:
                            if pc[pi] == 0:
                                stats[tag]["tp"] += 1
                            else:
                                stats[tag]["fc"] += 1
                        else:
                            if pc[pi] == 1:
                                stats[tag]["hel_ok"] += 1
                            else:
                                stats[tag]["hel_bad"] += 1
                if gc == 0:
                    head_by_size[tag][k][0] += 1
                    if matched and pc[pi] == 0:
                        head_by_size[tag][k][1] += 1
            stats[tag]["fp"] += len(pb) - len(used_p)
            stats[tag]["miss"] += int((gts[:, 0] == 0).sum()) - sum(
                1 for gi in used_g if int(gts[gi, 0]) == 0)

    print(f"# 两阶段对比（{args.split}，{len(img_files)} 张，imgsz {args.imgsz}）")
    print(f"检测 conf {args.conf} → 工作点 {args.work_conf}；路由 small<{args.small:.0f} / large>{args.large:.0f}；margin {args.margin}\n")
    print("| 配置 | 违规召回 | 误合规 | 漏检 | 误报(FP) | helmet 判对 | helmet 判错 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for tag, name in (("base", "基线（检测器）"), ("two", "两阶段")):
        s = stats[tag]
        tp, fc, miss = s["tp"], s["fc"], s["miss"]
        rec = tp / max(1, tp + fc + miss)
        fcr = fc / max(1, tp + fc + miss)
        print(f"| {name} | {tp}/{tp+fc+miss} = {rec:.1%} | {fcr:.1%} | {miss} | {s['fp']} | {s['hel_ok']} | {s['hel_bad']} |")

    print(f"\n改判 {changed} 个；其中改对 {correct}、改错 {wrong}")
    print(f"进入分类器的框：mid {by_route['mid']} / large {by_route['large']}")

    for tag, name in (("base", "基线"), ("two", "两阶段")):
        rows = head_by_size[tag]
        n_head = sum(v[0] for v in rows.values())
        if not n_head:
            continue
        print(f"\n按尺寸（{name}）：")
        print("| 尺寸 | 框数 | 判对 | 召回 |")
        print("|---|---:|---:|---:|")
        for k, (tot, ok) in rows.items():
            if tot:
                print(f"| {k} | {tot} | {ok} | {ok / tot:.1%} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
