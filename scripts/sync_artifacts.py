#!/usr/bin/env python
"""Sync and verify the shared artifacts bundle (Git LFS)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "manifest.json"
POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"


def is_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(len(POINTER_HEADER)) == POINTER_HEADER
    except OSError:
        return False


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST.exists():
        print(f"missing manifest: {MANIFEST}", file=sys.stderr)
        return 2

    pointers = [p for p in (ROOT / "artifacts").rglob("*") if p.is_file() and is_pointer(p)]
    if pointers:
        print(f"found {len(pointers)} Git LFS pointer file(s); pulling...")
        if shutil.which("git") is None:
            print("git is not available", file=sys.stderr)
            return 2
        if run(["git", "lfs", "version"]) != 0:
            print("Git LFS is not installed. Install it, then run this script again.", file=sys.stderr)
            return 2
        run(["git", "lfs", "install"])
        if run(["git", "lfs", "pull"]) != 0:
            print("git lfs pull failed; check repository access.", file=sys.stderr)
            return 2

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    model = data.get("model", {})
    model_path = ROOT / model.get("path", "")
    expected = str(model.get("sha256", "")).lower()
    if not model_path.exists():
        print(f"missing model: {model_path}", file=sys.stderr)
        return 2
    if expected:
        actual = sha256(model_path).lower()
        if actual != expected:
            print(f"sha256 mismatch: expected {expected}, got {actual}", file=sys.stderr)
            return 3
        print("model sha256: OK")

    print(f"artifacts version: {data.get('version')}")
    print(f"model: {model_path.relative_to(ROOT)}")
    print(f"runs: {data.get('runs', {}).get('path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
