"""Parquet persistence with audit logs for DriftEdge.

Mirror of PinSight's persistence pattern. Every write emits a `persist.write`
event with path, row count, byte count, and column list.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import obs


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def write_markets_snapshot(df: pd.DataFrame, data_dir: Path,
                           venue: str = "polymarket") -> Path:
    """Append today's market-list snapshot. One file per venue per day."""
    out_dir = data_dir / "markets" / venue
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_today()}.parquet"

    df = df.copy()
    df["_snapshot_ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with obs.timed("persist", "write.markets", venue=venue, path=str(path)) as t:
        if path.exists():
            existing = pq.read_table(path).to_pandas()
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df
        pq.write_table(pa.Table.from_pandas(combined, preserve_index=False),
                       path, compression="snappy")
        t.add(rows=len(df), total_rows=len(combined),
              bytes=path.stat().st_size, cols=list(df.columns))

    obs.bump("persist_writes")
    obs.bump("rows_written", by=len(df))
    obs.bump("bytes_written", by=path.stat().st_size)
    return path


def write_orderbook_snapshot(book: dict, data_dir: Path, *,
                             venue: str, market_id: str,
                             token_id: str) -> Path:
    """Append one orderbook snapshot. One file per (venue, market) per day.

    Stores flattened: each price level becomes a row with side/level/price/size.
    """
    out_dir = data_dir / "books" / venue / market_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_today()}.parquet"

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []

    # Polymarket book shape: { bids: [{price, size}, ...], asks: [{price, size}, ...] }
    for side in ("bids", "asks"):
        for level, lvl in enumerate(book.get(side, []) or []):
            try:
                price = float(lvl.get("price"))
                size = float(lvl.get("size"))
            except (TypeError, ValueError):
                continue
            rows.append({
                "snapshot_ts": ts,
                "market_id": market_id,
                "token_id": token_id,
                "side": side[:-1],  # bid | ask
                "level": level,
                "price": price,
                "size": size,
            })

    if not rows:
        obs.event(channel="persist", kind="book.empty", level="DEBUG",
                  venue=venue, market_id=market_id, token_id=token_id)
        return path

    df = pd.DataFrame(rows)
    with obs.timed("persist", "write.book", venue=venue, market_id=market_id,
                   path=str(path)) as t:
        if path.exists():
            existing = pq.read_table(path).to_pandas()
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df
        pq.write_table(pa.Table.from_pandas(combined, preserve_index=False),
                       path, compression="snappy")
        t.add(rows=len(df), total_rows=len(combined),
              bytes=path.stat().st_size)

    obs.bump("persist_writes")
    obs.bump("rows_written", by=len(df))
    obs.bump("bytes_written", by=path.stat().st_size)
    return path


def write_trades(df: pd.DataFrame, data_dir: Path, *,
                 venue: str, market_id: str) -> Path:
    """Append recent trades for a market. One file per (venue, market) per day."""
    if df is None or df.empty:
        return data_dir / "trades" / venue / market_id / f"{_today()}.parquet"

    out_dir = data_dir / "trades" / venue / market_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_today()}.parquet"

    with obs.timed("persist", "write.trades", venue=venue, market_id=market_id,
                   path=str(path)) as t:
        if path.exists():
            existing = pq.read_table(path).to_pandas()
            # Dedup by trade id if present
            if "id" in df.columns and "id" in existing.columns:
                combined = pd.concat([existing, df], ignore_index=True) \
                    .drop_duplicates(subset=["id"], keep="last")
            else:
                combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df
        pq.write_table(pa.Table.from_pandas(combined, preserve_index=False),
                       path, compression="snappy")
        t.add(rows=len(df), total_rows=len(combined),
              bytes=path.stat().st_size)

    obs.bump("persist_writes")
    obs.bump("rows_written", by=len(df))
    obs.bump("bytes_written", by=path.stat().st_size)
    return path
