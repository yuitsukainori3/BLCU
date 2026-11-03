"""Create a train-only repeated-image list for the minority `head` class."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="Dataset YAML with head=0, helmet=1")
    p.add_argument("--out", type=Path, default=Path("runs/data/head_repeated"))
    p.add_argument("--repeat", type=int, default=2, help="Extra copies of each image containing head")
    a = p.parse_args()
    if a.repeat < 1:
        p.error("--repeat must be at least 1")
    cfg_path = a.data.resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", cfg_path.parent))
    if not root.is_absolute():
        root = (cfg_path.parent / root).resolve()
    train_spec = Path(str(cfg["train"]))
    train_root = train_spec if train_spec.is_absolute() else root / train_spec
    if train_root.is_file() and train_root.suffix.lower() == ".txt":
        images = [Path(line.strip()) for line in train_root.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        images = sorted(x for x in train_root.rglob("*") if x.suffix.lower() in IMAGE_EXTS)
    if not images:
        p.error(f"训练图像未找到：{train_root}")
    lines: list[str] = []
    minority: list[Path] = []
    for image in images:
        parts = list(image.parts)
        try:
            idx = parts.index("images")
            parts[idx] = "labels"
            label = Path(*parts).with_suffix(".txt")
        except ValueError:
            p.error(f"图像路径缺少 images 目录，无法匹配 YOLO 标签：{image}")
        if not label.is_file():
            p.error(f"标签不存在：{label}")
        has_head = any(row.split() and row.split()[0] == "0" for row in label.read_text(encoding="utf-8").splitlines())
        lines.append(str(image.resolve()))
        if has_head:
            minority.append(image)
    if not minority:
        p.error("训练集中没有类别 0（head）的图像，请检查标签类别映射。")
    lines.extend(str(x.resolve()) for _ in range(a.repeat) for x in minority)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    list_path = out / "train_head_repeated.txt"
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg["path"] = str(root)
    cfg["train"] = str(list_path)
    for split in ("val", "test"):
        if cfg.get(split) and not Path(str(cfg[split])).is_absolute():
            cfg[split] = str((root / str(cfg[split])).resolve())
    yaml_path = out / "helmet_head_repeated.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"训练图片：{len(images)}；含 head 图片：{len(minority)}；附加重复：{len(minority) * a.repeat}")
    print(f"新数据配置：{yaml_path}")
    print("验证集与测试集保持原样；重采样收益必须通过独立评估确认。")


if __name__ == "__main__":
    main()
