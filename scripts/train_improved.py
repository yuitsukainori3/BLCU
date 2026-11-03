"""Reproducible high-resolution YOLO11 fine-tuning for the SHWD head/helmet task."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDORED_ULTRALYTICS = ROOT / "ultralytics-8.3.163" / "ultralytics-8.3.163"
if VENDORED_ULTRALYTICS.is_dir():
    sys.path.insert(0, str(VENDORED_ULTRALYTICS))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Two-class dataset YAML: head=0, helmet=1")
    parser.add_argument("--model", default="yolo11s.pt", help="Pretrained model or previous checkpoint")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=float, default=-1, help="-1 enables Ultralytics auto batch")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="helmet_yolo11s_960_improved")
    args = parser.parse_args()
    data = args.data.resolve()
    if not data.is_file():
        parser.error(f"数据 YAML 不存在：{data}")
    results = YOLO(args.model).train(
        data=str(data), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        device=args.device, seed=args.seed, deterministic=True, patience=30,
        cos_lr=True, close_mosaic=15, mosaic=1.0, scale=.5,
        hsv_h=.015, hsv_s=.7, hsv_v=.4,
        workers=0, cache="disk", project="runs/train", name=args.name,
        plots=True, save=True, val=True,
    )
    print(f"Training completed: {getattr(results, 'save_dir', 'runs/train/' + args.name)}")


if __name__ == "__main__":
    main()
