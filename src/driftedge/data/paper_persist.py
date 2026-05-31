"""Append/upsert paper-trade rows in data/paper_trades.parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import obs


def _path(data_dir: Path) -> Path:
    return data_dir / "paper_trades.parquet"


def load_positions(data_dir: Path) -> list[dict]:
    p = _path(data_dir)
    if not p.exists():
        return []
    try:
        df = pd.read_parquet(p)
    except Exception as exc:
        obs.event(channel="error", kind="paper.load_fail", level="WARNING",
                  err=str(exc))
        return []
    return df.where(pd.notna(df), None).to_dict(orient="records")


def upsert_positions(data_dir: Path, *, opened: list[dict],
                     closed: list[dict]) -> None:
    """Append new opens; replace rows for closed (by trade_id)."""
    p = _path(data_dir)
    existing = load_positions(data_dir)
    by_id = {r["trade_id"]: r for r in existing}

    for o in opened:
        by_id[o["trade_id"]] = o
    for c in closed:
        # Use trade_id from closed (carried over from open).
        by_id[c["trade_id"]] = c

    rows = list(by_id.values())
    df = pd.DataFrame(rows)
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   p, compression="snappy")

    obs.event(channel="persist", kind="paper.upsert", level="INFO",
              opened=len(opened), closed=len(closed),
              total_rows=len(rows), bytes=p.stat().st_size)

    obs.bump("persist_writes")
    obs.bump("rows_written", by=len(opened) + len(closed))
