"""Video inference helpers for the helmet monitoring showcase."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class Box:
    cls: int
    name: str
    conf: float
    xyxy: tuple[int, int, int, int]
    track_id: int | None = None


class EventEngine:
    """Turn frame-level head classifications into stable, de-duplicated events."""

    def __init__(self, window: int = 15, violation_ratio: float = .6,
                 min_observations: int = 5, cooldown_s: float = 20):
        self.votes: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.last_event: dict[str, float] = {}
        self.violation_ratio = violation_ratio
        self.min_observations = min_observations
        self.cooldown_s = cooldown_s
        self.events: list[dict] = []

    def update(self, track: str, state: str, timestamp: float, region: str) -> dict | None:
        q = self.votes[track]
        q.append(state)
        if len(q) < self.min_observations:
            return None
        ratio = sum(x == "violation" for x in q) / len(q)
        if ratio < self.violation_ratio or timestamp - self.last_event.get(track, -1e9) < self.cooldown_s:
            return None
        self.last_event[track] = timestamp
        item = {"track": track, "time_s": round(timestamp, 2), "region": region,
                "state": "疑似未佩戴", "window_violation_ratio": round(ratio, 3)}
        self.events.append(item)
        return item


def assign_head_to_person(head_xyxy: Iterable[int], people: list[Box], head_region_ratio: float = .35) -> Box | None:
    x1, y1, x2, y2 = map(float, head_xyxy)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    candidates = []
    for person in people:
        px1, py1, px2, py2 = person.xyxy
        if px1 <= cx <= px2 and py1 <= cy <= py1 + (py2 - py1) * head_region_ratio:
            candidates.append((abs(cx - (px1 + px2) / 2), person))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def collect_boxes(result, names: dict[int, str], thresholds: dict[str, float]) -> list[Box]:
    out = []
    if result.boxes is None:
        return out
    ids = result.boxes.id
    for i, row in enumerate(result.boxes):
        cls = int(row.cls.item())
        name = names.get(cls, str(cls))
        conf = float(row.conf.item())
        if conf < thresholds.get(name, .25):
            continue
        xyxy = tuple(int(v) for v in row.xyxy[0].tolist())
        track_id = int(ids[i].item()) if ids is not None else None
        out.append(Box(cls, name, conf, xyxy, track_id))
    return out


def smooth_boxes(new_boxes: list[Box], previous_boxes: list[Box], new_weight: float = .7) -> list[Box]:
    """Blend refreshed detections with the optical-flow boxes to reduce frame-to-frame jumps."""
    old_by_track = {b.track_id: b for b in previous_boxes if b.name == "person" and b.track_id is not None}
    used: set[int] = set()
    for box in new_boxes:
        old = old_by_track.get(box.track_id) if box.name == "person" and box.track_id is not None else None
        if old is None:
            x1, y1, x2, y2 = box.xyxy
            best_iou = 0.0
            for idx, candidate in enumerate(previous_boxes):
                if idx in used or candidate.name != box.name:
                    continue
                ox1, oy1, ox2, oy2 = candidate.xyxy
                ix1, iy1 = max(x1, ox1), max(y1, oy1)
                ix2, iy2 = min(x2, ox2), min(y2, oy2)
                intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = max(1, x2 - x1) * max(1, y2 - y1)
                area_b = max(1, ox2 - ox1) * max(1, oy2 - oy1)
                iou = intersection / (area_a + area_b - intersection)
                if iou > best_iou:
                    best_iou, old = iou, candidate
            if best_iou < .1:
                old = None
        if old is not None:
            used.add(previous_boxes.index(old))
            box.xyxy = tuple(int(round(new_weight * new + (1 - new_weight) * old_value))
                             for new, old_value in zip(box.xyxy, old.xyxy))
    return new_boxes


def propagate_boxes(previous_gray: np.ndarray | None, current_gray: np.ndarray,
                    boxes: list[Box]) -> list[Box]:
    """Carry boxes through skipped frames using sparse optical flow."""
    if previous_gray is None or not boxes:
        return boxes
    height, width = current_gray.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(width, x2), min(height, y2)
        if x2c - x1c < 4 or y2c - y1c < 4:
            continue
        roi = previous_gray[y1c:y2c, x1c:x2c]
        points = cv2.goodFeaturesToTrack(roi, maxCorners=24, qualityLevel=.01,
                                         minDistance=3, blockSize=3)
        if points is None or len(points) < 3:
            continue
        points[:, 0, 0] += x1c
        points[:, 0, 1] += y1c
        moved, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None,
                                                     winSize=(21, 21), maxLevel=3)
        if moved is None or status is None:
            continue
        valid = status.reshape(-1).astype(bool)
        if valid.sum() < 3:
            continue
        delta = np.median(moved[valid, 0] - points[valid, 0], axis=0)
        dx, dy = int(round(float(delta[0]))), int(round(float(delta[1])))
        box.xyxy = (max(0, x1 + dx), max(0, y1 + dy),
                    min(width, x2 + dx), min(height, y2 + dy))
    return boxes


def annotate(frame: np.ndarray, boxes: list[Box], events: list[dict]) -> np.ndarray:
    image = frame.copy()
    colors = {"head": (40, 60, 245), "helmet": (60, 200, 80), "person": (235, 180, 35)}
    violation_tracks = {str(e["track"]) for e in events}
    for box in boxes:
        label = f"{box.name} {box.conf:.2f}"
        if box.track_id is not None:
            label += f" ID:{box.track_id}"
        color = colors.get(box.name, (240, 240, 240))
        x1, y1, x2, y2 = box.xyxy
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    return image


def region_summary(events: list[dict], regions: list[str], state_counts: dict[str, dict[str, int]] | None = None,
                   previous_scores: dict[str, float] | None = None) -> list[dict]:
    counts = {r: 0 for r in regions}
    for item in events:
        counts[item["region"]] = counts.get(item["region"], 0) + 1
    state_counts = state_counts or {}
    previous_scores = previous_scores or {}
    summary = []
    for region, n in counts.items():
        statuses = state_counts.get(region, {})
        violations = statuses.get("violation", 0)
        compliant = statuses.get("compliant", 0)
        observations = violations + compliant
        raw_risk = violations / observations if observations else 0.0
        smoothed = .7 * previous_scores.get(region, raw_risk) + .3 * raw_risk if observations else previous_scores.get(region, 0.0)
        level = "重点关注" if smoothed >= .35 else ("关注" if smoothed >= .15 else ("正常" if observations else "暂无数据"))
        summary.append({"区域": region, "头部观察数": observations,
                        "佩戴率": f"{compliant / observations:.1%}" if observations else "暂无数据",
                        "疑似事件": n, "风险权重": round(1 + 2 * smoothed, 2),
                        "平滑风险": round(smoothed, 3), "风险等级": level})
    return summary


def model_path_or_pointer(path: str | Path) -> tuple[bool, str]:
    p = Path(path)
    if not p.is_file():
        return False, f"模型权重不存在：{p}"
    try:
        if p.stat().st_size < 1024:
            return False, "当前权重是 Git LFS 指针文件，请在仓库根目录运行 git lfs pull。"
    except OSError as exc:
        return False, str(exc)
    manifest = p.parents[2] / "models" / "manifest.json"
    try:
        info = json.loads(manifest.read_text(encoding="utf-8"))
        entry = next((v for v in info.get("models", {}).values()
                      if (p.parents[2] / v.get("file", "")).resolve() == p.resolve()), None)
        expected = entry.get("sha256") if entry else None
        if expected:
            with p.open("rb") as model_file:
                digest = hashlib.file_digest(model_file, "sha256").hexdigest()
            if digest.lower() != expected.lower():
                return False, "模型文件 SHA256 与 models/manifest.json 不一致，请重新同步 Git LFS 权重。"
    except (OSError, ValueError, KeyError, StopIteration):
        pass
    return True, ""
