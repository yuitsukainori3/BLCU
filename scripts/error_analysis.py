#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E3 错误分解：把“漏检”拆成能动手的几类，并给出每一类的特征画像。

只看“漏了 51 个裸头”没法决策；要先知道这 51 个属于哪一种：

  A 判对            IoU≥0.45 且类别对，conf≥工作点
  B 误合规          IoU≥0.45 但被判成 helmet，conf≥工作点
  C 低置信          IoU≥0.45 类别对，但 conf 低于工作点   → 降阈值就能救
  D 定位偏差        0.1 ≤ IoU < 0.45                    → 框不准 / 密集遮挡
  E 完全没检出      IoU < 0.1                           → 模型没看到

每一类再统计：框的像素尺寸、亮度、同图目标数、是否贴边、与最近 helmet 框的重叠。

用法
----
python scripts/error_analysis.py --weights runs/detect/train4/weights/best.pt --split val --out runs/error_analysis
python scripts/error_analysis.py --weights ... --split test --conf 0.25
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

DATA_DEFAULT = Path(r"F:\Competition\ultralytics-8.3.163\ultralytics-8.3.163\ultralytics\cfg\datasets\SHWD.yaml")


def die(msg: str, code: int = 1):
    print(f"[错误] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, default=DATA_DEFAULT)
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="工作点阈值（与评测口径一致）")
    p.add_argument("--iou", type=float, default=0.45, help="匹配用的 IoU 阈值")
    p.add_argument("--low-conf", type=float, default=0.05, help="低置信探测的下限")
    p.add_argument("--device", default="0")
    p.add_argument("--out", type=Path, default=Path(r"F:\Competition\runs\error_analysis"))
    p.add_argument("--save-crops", type=int, default=0, help="为前 N 个 E 类样本保存裁剪图")
    return p.parse_args(argv)


def read_yaml(path: Path) -> dict:
    out, section = {}, None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith((" ", "\t")) and section:
            k, v = line.split(":", 1) if ":" in line else (None, None)
            if k is None:
                continue
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


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def main(argv=None) -> int:
    args = parse_args(argv)
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        die(f"权重不存在：{weights}")

    cfg = read_yaml(Path(args.data))
    names = cfg.get("names", {})
    if isinstance(names, str):  # 行内写法 names: ['head', 'helmet']
        items = [x.strip().strip("'\"") for x in names.strip().strip("[]").split(",") if x.strip()]
        names = {i: n for i, n in enumerate(items)}
    else:                       # 块写法 0: head / 1: helmet
        names = {int(k): v for k, v in names.items()}
    head_id = next((k for k, v in names.items() if v == "head"), 0)
    helmet_id = next((k for k, v in names.items() if v == "helmet"), 1)

    root = Path(cfg["path"])
    img_dir = root / cfg[args.split]
    img_files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not img_files:
        die(f"{img_dir} 下没有图片")

    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.predict([str(p) for p in img_files], imgsz=args.imgsz, conf=args.low_conf,
                            iou=0.7, max_det=300, device=args.device, verbose=False, stream=False)

    records = []
    for img_path, res in zip(img_files, results):
        h, w = res.orig_shape
        gt = load_gt(root / "labels" / args.split / f"{img_path.stem}.txt", w, h)
        pred = np.hstack([res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()[:, None]])
        pred_cls = res.boxes.cls.cpu().numpy().astype(int)
        gray = np.asarray(Image.open(img_path).convert("L"), dtype=float)
        bright = float(gray.mean())
        contrast = float(gray.std())

        for g in gt:
            gcls, gbox = int(g[0]), g[1:5]
            ious = iou(gbox[None, :], pred[:, :4])[0] if len(pred) else np.zeros(0)
            k = int(np.argmax(ious)) if len(ious) else -1
            best_iou = float(ious[k]) if k >= 0 else 0.0
            best_conf = float(pred[k, 4]) if k >= 0 else 0.0
            best_cls = int(pred_cls[k]) if k >= 0 else -1

            if best_iou >= args.iou and best_conf >= args.conf:
                kind = "A_判对" if best_cls == gcls else ("B_误合规" if gcls == head_id else "B_误判")
            elif best_iou >= args.iou:
                kind = "C_低置信"
            elif best_iou >= 0.1:
                kind = "D_定位偏差"
            else:
                kind = "E_完全没检出"

            # 与最近的 helmet 框的重叠（对 head 有诊断意义）
            helm = gt[(gt[:, 0] == helmet_id)]
            near_iou = float(np.max(iou(gbox[None, :], helm[:, 1:5]))) if len(helm) else 0.0

            bw, bh = gbox[2] - gbox[0], gbox[3] - gbox[1]
            edge = gbox[0] <= 2 or gbox[1] <= 2 or gbox[2] >= w - 2 or gbox[3] >= h - 2
            records.append({
                "image": img_path.name, "gt_cls": gcls, "kind": kind,
                "cx": round(float((gbox[0] + gbox[2]) / 2), 1),
                "cy": round(float((gbox[1] + gbox[3]) / 2), 1),
                "w_px": round(float(bw), 1), "h_px": round(float(bh), 1),
                "area_px": round(float(bw * bh), 1), "aspect": round(float(bw / max(bh, 1e-6)), 2),
                "best_iou": round(best_iou, 3), "best_conf": round(best_conf, 3), "best_cls": best_cls,
                "near_helmet_iou": round(near_iou, 3),
                "brightness": round(bright, 1), "contrast": round(contrast, 1),
                "n_gt_image": len(gt), "touch_edge": int(edge),
            })

    out: Path = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"boxes_{args.split}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        wr.writeheader()
        wr.writerows(records)

    # ---- 汇总 ----
    def stats(kind: str, cls: int | None = None) -> dict:
        sel = [r for r in records if r["kind"] == kind and (cls is None or r["gt_cls"] == cls)]
        if not sel:
            return {"n": 0}
        arr = lambda k: np.array([r[k] for r in sel], dtype=float)
        return {
            "n": len(sel),
            "w": float(np.median(arr("w_px"))), "h": float(np.median(arr("h_px"))),
            "area": float(np.median(arr("area_px"))),
            "iou": float(np.median(arr("best_iou"))), "conf": float(np.median(arr("best_conf"))),
            "near": float(np.median(arr("near_helmet_iou"))),
            "bright": float(np.median(arr("brightness"))),
            "crowd": float(np.median(arr("n_gt_image"))),
            "edge": float(arr("touch_edge").mean()),
        }

    lines = [f"# 错误分解（{args.split}）", "",
             f"权重 `{weights.name}`　imgsz {args.imgsz}　工作点 conf {args.conf} / IoU {args.iou}", ""]
    for cls, cname in ((head_id, "head"), (helmet_id, "helmet")):
        lines += [f"## {cname}", "", "| 类型 | 数量 | 中位宽×高(px) | 中位面积 | 中位 best IoU | 中位 best conf | 与最近 helmet 的 IoU | 贴边比例 |",
                  "|---|---:|---|---:|---:|---:|---:|---:|"]
        total = 0
        for kind in ("A_判对", "B_误合规", "B_误判", "C_低置信", "D_定位偏差", "E_完全没检出"):
            s = stats(kind, cls)
            if not s["n"]:
                continue
            total += s["n"]
            lines.append(f"| {kind} | {s['n']} | {s['w']:.0f}×{s['h']:.0f} | {s['area']:.0f} | "
                         f"{s['iou']:.3f} | {s['conf']:.3f} | {s['near']:.3f} | {s['edge']:.0%} |")
        lines += ["", f"小计 {total} 个 {cname} 框", ""]

    # 尺寸分档（按 COCO 口径：面积 <32² / 32²–96² / >96²）
    lines += ["## 尺寸分档（按 GT 框面积）", "",
              "| 类别 | <32² px | 32²–96² | >96² |", "|---|---:|---:|---:|"]
    for cls, cname in ((head_id, "head"), (helmet_id, "helmet")):
        sel = [r["area_px"] for r in records if r["gt_cls"] == cls]
        s = sum(1 for a in sel if a < 1024); m = sum(1 for a in sel if 1024 <= a <= 9216)
        l = sum(1 for a in sel if a > 9216)
        lines.append(f"| {cname} | {s} | {m} | {l} |")
    lines.append("")

    report = "\n".join(lines)
    (out / f"summary_{args.split}.md").write_text(report + "\n", encoding="utf-8")
    print(report)

    if args.save_crops > 0:
        crop_dir = out / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        picks = [r for r in records if r["kind"].startswith("E")][: args.save_crops]
        for i, r in enumerate(picks):
            img_path = img_dir / r["image"]
            im = Image.open(img_path).convert("RGB")
            w, h = im.size
            cx = float(r["cx"]); cy = float(r["cy"]); bw = r["w_px"]; bh = r["h_px"]
            pad = max(bw, bh) * 1.5
            box = (max(0, cx - bw / 2 - pad), max(0, cy - bh / 2 - pad),
                   min(w, cx + bw / 2 + pad), min(h, cy + bh / 2 + pad))
            im.crop(box).resize((224, 224)).save(crop_dir / f"{i:02d}_c{r['gt_cls']}_{r['image']}", quality=85)
        print(f"[已保存] {len(picks)} 张 E 类裁剪图 -> {crop_dir}")

    print(f"\n[已写出] {out / f'summary_{args.split}.md'} 和 boxes_{args.split}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
