#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脚本公共件：配置回显 + 控制台进度条（与 YOLO 官方训练同款样式）。

为什么有这个文件：[docs/脚本规范.md](../docs/脚本规范.md) 要求所有脚本「参数放最前面、
跑起来看得见进度」。这两件事每个脚本都要做，集中在这里，避免 8 个脚本各写一套。

用法（脚本里）：

    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT / "scripts"))
    from _ui import die, echo_config, progress, Tee

    echo_config("批量裁片", [("输入", src), ("输出", dst)])
    for p in progress(sorted(src.glob("*.jpg")), desc="裁片", unit="file"):
        ...

样式与 ultralytics 8.3.163 的进度条一致
（`ultralytics/utils/__init__.py:44` 的 `TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"`）：

    裁片: 100%|██████████| 282/282 [00:01<00:00, 210.4file/s]

三条兜底，脚本不会因为显示问题挂掉：

1. 没装 tqdm → 自动降级为每 5% 打一行的朴素输出；
2. `NO_PROGRESS=1` → 完全静默（重定向写日志时用）；
3. 进度的书写到 stderr，不污染 stdout 的结果表/JSON。
"""

from __future__ import annotations

import os
import sys
import unicodedata
from pathlib import Path

# 与 ultralytics/utils/__init__.py:44 保持一致，改这里要一起改
BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"
BAR_WIDTH = 10
ENV_OFF = "NO_PROGRESS"

try:  # tqdm 是可选依赖：没有也别让整条流水线崩掉
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover - 环境问题
    _tqdm = None

HAS_TQDM = _tqdm is not None


# --------------------------------------------------------------------------- #
# 文本宽度：中文占 2 格，否则配置回显会歪
# --------------------------------------------------------------------------- #
def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in str(text))


def _pad(text, width: int) -> str:
    text = str(text)
    return text + " " * max(0, width - _display_width(text))


def echo_config(title: str, items, stream=None, width: int = 12) -> None:
    """开跑前打印「本次到底用了什么配置」。

    Args:
        title: 标题，例如 "回归视频跑分"。
        items: (名字, 值) 的可迭代；值可以是路径、数字、字符串。
        stream: 默认 stdout。
        width: 名字列宽（显示宽度，中文算 2）。
    """
    stream = stream or sys.stdout
    print("=" * 64, file=stream)
    print(f" {title}", file=stream)
    print("-" * 64, file=stream)
    for name, value in items:
        print(f"  {_pad(name, width)} {value}", file=stream)
    print("=" * 64, file=stream)
    stream.flush()


def die(message: str) -> None:
    """可预期错误（参数错、文件缺失）：中文提示 + 退出码 1，不打 traceback。"""
    print(f"[错误] {message}", file=sys.stderr)
    raise SystemExit(1)


def require_torch(what: str = "这个脚本", hint: str = "") -> None:
    """需要 GPU 环境（torch）。用错解释器时给一句人话，而不是甩 ModuleNotFoundError。

    踩过的坑：拿一个空 venv 的 python 跑训练，报错指向 ultralytics 内部，看不出是环境问题。
    """
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        message = (f"当前解释器没有 torch：{sys.executable}\n"
                   f"  {what}要用装了 torch + ultralytics 的环境（本项目是 conda 环境 yolo），不是随便一个 venv。")
        if hint:
            message += f"\n  {hint}"
        die(message)


# --------------------------------------------------------------------------- #
# 进度条
# --------------------------------------------------------------------------- #
class _PlainBar:
    """没装 tqdm 时的兜底：每 5% 打一行，接口只覆盖脚本里用到的子集。"""

    def __init__(self, iterable=None, total=None, desc="", unit="it", stream=None, **_):
        self.iterable = iterable
        if total is None:
            total = len(iterable) if hasattr(iterable, "__len__") else 0
        self.total = int(total or 0)
        self.desc = desc or ""
        self.unit = unit
        self.stream = stream or sys.stderr
        self.n = 0
        self._next = 0.0
        self._postfix = ""

    def __iter__(self):
        if self.iterable is None:
            return
        for item in self.iterable:
            yield item
            self.n += 1
            self._report()

    def update(self, k: int = 1) -> None:
        self.n += k
        self._report()

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs) -> "_PlainBar":
        data = dict(ordered_dict or {})
        data.update(kwargs)
        self._postfix = " ".join(f"{k}={v}" for k, v in data.items())
        return self

    def close(self) -> None:
        if self.total and self.n >= self.total:
            self._report()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _report(self) -> None:
        if self.total <= 0:
            return
        pct = self.n / self.total
        if pct + 1e-9 >= self._next:
            line = f"{self.desc}: {pct:5.1%} {self.n}/{self.total} {self.unit}".rstrip()
            if self._postfix:
                line += f" | {self._postfix}"
            print(line, file=self.stream, flush=True)
            while self._next <= pct + 1e-9:
                self._next += 0.05


class _NullBar:
    """完全静默：NO_PROGRESS=1，或脚本自己判断不该刷屏时用。"""

    def __init__(self, iterable=None, total=None, **_):
        self.iterable = iterable
        self.total = total
        self.n = 0

    def __iter__(self):
        if self.iterable is None:
            return
        for item in self.iterable:
            yield item
            self.n += 1

    def update(self, k: int = 1) -> None:
        self.n += k

    def set_postfix(self, *args, **kwargs) -> "_NullBar":
        return self

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def progress(iterable=None, total=None, desc="", unit="it", leave=True, stream=None, **kwargs):
    """统一进度条入口，返回的对象既能 for 循环也能 .update()/.set_postfix()。

    Args:
        iterable: 被遍历的对象；只传 total 时用于手动 update。
        total: 总步数；给了 iterable 可省略。
        desc: 左侧文字，建议用动宾短语（"裁片"、"推理"、"跑 R1"）。
        unit: 单位，``it`` / ``file`` / ``frame``。
        leave: 单条短任务设 False，免得刷一屏。
        stream: 默认 stderr。
        **kwargs: 透传给 tqdm（例如 ``ncols=100``）。

    Environment:
        NO_PROGRESS=1 时返回静默对象，一行都不打。
    """
    stream = stream or sys.stderr
    off = str(os.environ.get(ENV_OFF, "")).strip().lower()
    if off not in ("", "0", "false", "no") or kwargs.pop("disable", False):
        return _NullBar(iterable, total=total)
    if not HAS_TQDM:
        return _PlainBar(iterable, total=total, desc=desc, unit=unit, stream=stream)
    kwargs.setdefault("bar_format", BAR_FORMAT)
    return _tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave, file=stream, **kwargs)


# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
class Tee:
    """把 stdout 同时写文件（追加模式）；用 ``with Tee(path):`` 包住主流程。

    注意：进度条写在 stderr，不会进这个日志——这是有意的，日志里只留结果。
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = None
        self.stdout = None

    def __enter__(self) -> "Tee":
        self.f = open(self.path, "a", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *exc) -> bool:
        sys.stdout = self.stdout
        self.close()
        return False

    def write(self, text: str) -> int:
        self.stdout.write(text)
        self.f.write(text)
        return len(text)

    def flush(self) -> None:
        try:
            self.stdout.flush()
            self.f.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return bool(getattr(self.stdout, "isatty", lambda: False)())

    @property
    def encoding(self):
        return "utf-8"

    def close(self) -> None:
        try:
            if self.f:
                self.f.close()
        except Exception:
            pass


if __name__ == "__main__":
    # 自检：python scripts/_ui.py  应当看到一条 20 步的进度条
    echo_config("_ui 自检", [("tqdm", HAS_TQDM), ("格式", BAR_FORMAT)])
    for i in progress(range(20), desc="自检", unit="step"):
        pass
