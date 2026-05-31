"""Prove the paper-trading engine does not use future data.

The contract: for any past timestamp T, the decision made at T must be
identical regardless of whether future-dated snapshots (T+k) exist in
the data store. We construct a minimal in-memory data store, take a
decision at T, then add T+k snapshots and re-take the decision at T —
both must match.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from driftedge.paper import (
    BookTop,
    EntryRule,
    latest_book_top,
    should_open,
    should_close,
    open_position,
    close_position,
)


def _write_book(books_dir: Path, market_id: str,
                snapshot_ts: str, bids: list[tuple[float, float]],
                asks: list[tuple[float, float]]) -> None:
    market_dir = books_dir / market_id
    market_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for level, (price, size) in enumerate(bids):
        rows.append({"snapshot_ts": snapshot_ts, "market_id": market_id,
                     "token_id": "tok", "side": "bid", "level": level,
                     "price": price, "size": size})
    for level, (price, size) in enumerate(asks):
        rows.append({"snapshot_ts": snapshot_ts, "market_id": market_id,
                     "token_id": "tok", "side": "ask", "level": level,
                     "price": price, "size": size})
    df = pd.DataFrame(rows)
    path = market_dir / "2026-05-30.parquet"
    if path.exists():
        existing = pq.read_table(path).to_pandas()
        df = pd.concat([existing, df], ignore_index=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   path, compression="snappy")


def test_latest_book_top_filters_future_snapshots():
    with tempfile.TemporaryDirectory() as tmp:
        books = Path(tmp)
        _write_book(books, "M1", "2026-05-30T10:00:00",
                    bids=[(0.32, 1000)], asks=[(0.33, 1000)])
        _write_book(books, "M1", "2026-05-30T11:00:00",
                    bids=[(0.55, 800)],  asks=[(0.56, 800)])
        _write_book(books, "M1", "2026-05-30T12:00:00",
                    bids=[(0.61, 600)],  asks=[(0.62, 600)])

        # Decision made AT 10:30 must see only the 10:00 snapshot.
        top = latest_book_top(books, "M1", as_of_ts="2026-05-30T10:30:00")
        assert top is not None
        assert top.snapshot_ts == "2026-05-30T10:00:00"
        assert top.best_ask == 0.33

        # Decision made AT 11:30 must see 11:00, not 12:00.
        top = latest_book_top(books, "M1", as_of_ts="2026-05-30T11:30:00")
        assert top is not None
        assert top.snapshot_ts == "2026-05-30T11:00:00"
        assert top.best_ask == 0.56


def test_should_open_does_not_change_when_future_data_added():
    """Backtest invariant: a past decision must not flip when future
    data is added to the store."""
    with tempfile.TemporaryDirectory() as tmp:
        books = Path(tmp)
        _write_book(books, "M1", "2026-05-30T10:00:00",
                    bids=[(0.32, 1000)], asks=[(0.33, 1000)])

        rule = EntryRule(entry_low=0.30, entry_high=0.40)
        top_before = latest_book_top(books, "M1", as_of_ts="2026-05-30T10:30:00")
        decision_before = should_open(top_before, rule)

        # Add a future snapshot where price has moved out of entry zone.
        _write_book(books, "M1", "2026-05-30T11:00:00",
                    bids=[(0.55, 800)], asks=[(0.56, 800)])

        top_after = latest_book_top(books, "M1", as_of_ts="2026-05-30T10:30:00")
        decision_after = should_open(top_after, rule)

        assert decision_before == decision_after, "decision changed after future data was added!"
        assert top_before.snapshot_ts == top_after.snapshot_ts


def test_close_assertion_blocks_lookahead():
    """If somehow a future snapshot is passed to close_position, the
    assertion must fire."""
    rule = EntryRule()
    book = BookTop(snapshot_ts="2026-05-30T11:00:00",
                   best_bid=0.65, best_ask=0.66,
                   bid_depth=100, ask_depth=100)
    pos = {"trade_id": "x", "entry_price": 0.35, "shares": 285.7}
    try:
        close_position(pos, book, "target", as_of_ts="2026-05-30T10:00:00")
    except AssertionError as exc:
        assert "lookahead" in str(exc)
        return
    raise AssertionError("close_position did not detect the lookahead")
