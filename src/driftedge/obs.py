"""Observability: structured logging + metrics for everything DriftEdge does.

Mirrors the PinSight obs module. Channels: api, persist, fit, signal, run, error.
Every event has timestamp, channel, kind, level, plus arbitrary fields.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from rich.console import Console

_console = Console(stderr=True)

_LOG_DIR: Path | None = None
_RUN_ID: str | None = None
_LEVEL: str = "INFO"

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}


def configure(log_dir: Path, level: str = "INFO", run_id: str | None = None) -> str:
    global _LOG_DIR, _RUN_ID, _LEVEL
    log_dir.mkdir(parents=True, exist_ok=True)
    _LOG_DIR = log_dir
    _LEVEL = level.upper()
    _RUN_ID = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    event(channel="run", kind="run.start", level="INFO", run_id=_RUN_ID, pid=os.getpid())
    return _RUN_ID


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def event(channel: str, kind: str, level: str = "INFO", **fields: Any) -> None:
    if _LOG_DIR is None:
        _console.print(f"[yellow]obs not configured[/yellow] {kind}", style="dim")
        return

    payload = {
        "ts": _ts(),
        "run_id": _RUN_ID,
        "channel": channel,
        "kind": kind,
        "level": level,
        **fields,
    }

    # Filter file writes by configured level (DEBUG events skipped at INFO+).
    level_num = _LEVELS.get(level, 20)
    configured_num = _LEVELS.get(_LEVEL, 20)

    if level_num >= configured_num:
        path = _LOG_DIR / f"{channel}-{_today()}.jsonl"
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except OSError as exc:
            _console.print(f"[red]obs write failed:[/red] {exc}")

    if level_num >= configured_num:
        colour = {"DEBUG": "dim", "INFO": "cyan", "WARNING": "yellow", "ERROR": "red"}.get(level, "")
        _console.print(
            f"[{colour}]{level:<7}[/{colour}] [bold]{channel}[/bold] {kind}",
            *(f"{k}={v}" for k, v in fields.items()),
        )


@dataclass
class Timed:
    kind: str
    channel: str
    started_ns: int = field(default_factory=time.perf_counter_ns)
    extra: dict[str, Any] = field(default_factory=dict)

    def add(self, **kwargs: Any) -> None:
        self.extra.update(kwargs)


@contextmanager
def timed(channel: str, kind: str, done_level: str = "INFO",
          **start_fields: Any) -> Iterator[Timed]:
    """Time a block. `done_level` controls the success-event log level
    (use 'DEBUG' for noisy per-call instrumentation; INFO for events
    worth keeping in the day-to-day log)."""
    t = Timed(kind=kind, channel=channel)
    event(channel=channel, kind=f"{kind}.start", level="DEBUG", **start_fields)
    try:
        yield t
    except Exception as exc:
        elapsed_ms = (time.perf_counter_ns() - t.started_ns) / 1e6
        event(
            channel="error",
            kind=f"{kind}.error",
            level="ERROR",
            duration_ms=round(elapsed_ms, 2),
            exc_type=type(exc).__name__,
            exc_msg=str(exc),
            traceback=traceback.format_exc(),
            **start_fields,
            **t.extra,
        )
        raise
    else:
        elapsed_ms = (time.perf_counter_ns() - t.started_ns) / 1e6
        event(
            channel=channel,
            kind=f"{kind}.done",
            level=done_level,
            duration_ms=round(elapsed_ms, 2),
            **start_fields,
            **t.extra,
        )


@dataclass
class RunSummary:
    api_calls: int = 0
    api_errors: int = 0
    persist_writes: int = 0
    rows_written: int = 0
    bytes_written: int = 0
    fits: int = 0
    signals: int = 0
    started_at: str = field(default_factory=_ts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SUMMARY = RunSummary()


def bump(field_name: str, by: int = 1) -> None:
    setattr(_SUMMARY, field_name, getattr(_SUMMARY, field_name) + by)


def finish() -> dict[str, Any]:
    payload = _SUMMARY.to_dict()
    payload["finished_at"] = _ts()
    event(channel="run", kind="run.end", level="INFO", **payload)
    return payload


def install_excepthook() -> None:
    def _hook(exc_type, exc, tb):
        event(
            channel="error",
            kind="unhandled",
            level="ERROR",
            exc_type=exc_type.__name__,
            exc_msg=str(exc),
            traceback="".join(traceback.format_exception(exc_type, exc, tb)),
        )
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook
