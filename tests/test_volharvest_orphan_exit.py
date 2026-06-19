"""Regression: VolHarvest must close held positions even when the market
has dropped out of the daemon's tracked-snapshot list.

The original bug: `volharvest.tick` iterated only `for m in markets:`. If
a held position's market was no longer in `markets` (volume rank, force
window, or near-certain filters), no exit branch ever ran for it. The
position was invisible to force-close, early-exit, and MTM.

The fix: at the top of `tick`, build a synthetic-stub list for held
positions whose `(venue, market_id)` isn't in `markets`, and prepend them
to the iterated list. Each stub carries the position's stored
`resolution_ts` so `should_force_close` still has its anchor.

These tests pin both behaviours:
  1. An orphan-market position IS visited and (if eligible) force-closed.
  2. The entry side does NOT re-fire on an orphan stub even when its book
     is in the underdog window — the book may be days stale.
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from driftedge.agents import volharvest


# ── Helpers ──────────────────────────────────────────────────────────────

def _write_book(books_root: Path, venue: str, market_id: str,
                snapshot_ts: str, yes_bid: float, yes_ask: float) -> None:
    """Write a 1-level book snapshot the same shape `latest_book_top` reads."""
    market_dir = books_root / venue / market_id
    market_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"snapshot_ts": snapshot_ts, "market_id": market_id, "token_id": "tok",
         "side": "bid", "level": 0, "price": yes_bid, "size": 1000.0},
        {"snapshot_ts": snapshot_ts, "market_id": market_id, "token_id": "tok",
         "side": "ask", "level": 0, "price": yes_ask, "size": 1000.0},
    ]
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   market_dir / f"{snapshot_ts[:10]}.parquet",
                   compression="snappy")


def _write_held_dog(data_dir: Path, venue: str, market_id: str,
                     entry_price: float, resolution_ts: str | None,
                     entry_ts: str) -> dict:
    """Write a single open volharvest dog position to paper_trades.parquet."""
    pos = {
        "trade_id": str(uuid.uuid4()),
        "trader": "volharvest",
        "venue": venue,
        "market_id": market_id,
        "question": "Test market",
        "category": "other",
        "yes_token_id": "tok",
        "no_token_id": "tok_no",
        "leg": "dog",
        "side": "yes",
        "entry_ts": entry_ts,
        "entry_snapshot_ts": entry_ts,
        "entry_price": entry_price,
        "entry_size_usd": 200.0,
        "shares": 200.0 / entry_price if entry_price > 0 else 0.0,
        "resolution_ts": resolution_ts,
        "target": None,
        "stop": None,
        "linked_trade_id": None,
        "status": "open",
        "exit_ts": None,
        "exit_snapshot_ts": None,
        "exit_price": None,
        "exit_reason": None,
        "pnl_per_share": None,
        "pnl_usd": None,
    }
    df = pd.DataFrame([pos])
    df.to_parquet(data_dir / "paper_trades.parquet",
                  compression="snappy", index=False)
    return pos


def _write_init_state(data_dir: Path) -> None:
    """Minimal paper_state with a volharvest row at $10k clean."""
    df = pd.DataFrame([{
        "trader": "volharvest",
        "bankroll_init": 10000.0,
        "cash_usd": 9800.0,            # one $200 dog already deployed
        "open_exposure": 200.0,
        "closed_pnl": 0.0,
        "peak_equity": 10000.0,
        "current_drawdown_pct": 0.0,
        "updated_ts": "2026-06-04T00:00:00+00:00",
    }])
    df.to_parquet(data_dir / "paper_state.parquet",
                  compression="snappy", index=False)


# ── Tests ────────────────────────────────────────────────────────────────

def test_orphan_position_is_visited_and_force_closed():
    """Held position whose market is NOT in `markets` must still get
    visited and force-closed when within the resolution window."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "books").mkdir(parents=True)

        # ── Set up an orphan dog: held position on a market that won't
        # appear in the `markets` list passed to tick().
        as_of = "2026-06-04T20:00:00+00:00"   # decision moment
        # resolution 1h from now → inside the 6h force-exit window
        resolution = "2026-06-04T21:00:00+00:00"
        _write_book(data_dir / "books", venue="kalshi",
                    market_id="ORPHAN-1",
                    snapshot_ts="2026-06-04T19:00:00+00:00",
                    yes_bid=0.10, yes_ask=0.12)
        pos = _write_held_dog(data_dir, venue="kalshi",
                               market_id="ORPHAN-1",
                               entry_price=0.15,
                               resolution_ts=resolution,
                               entry_ts="2026-06-02T12:00:00+00:00")
        _write_init_state(data_dir)

        # `markets` deliberately does NOT include ORPHAN-1 — that's the
        # whole point of the orphan case.
        result = volharvest.tick(data_dir, markets=[], as_of_ts=as_of,
                                  bankroll=10000.0)

        assert result["orphans_visited"] == 1, (
            f"orphan stub was not prepended: {result}")
        assert result["close_dogonly"] == 1, (
            f"orphan position should have force-closed: {result}")

        post = pd.read_parquet(data_dir / "paper_trades.parquet")
        assert (post["status"] == "open").sum() == 0, (
            "no positions should remain open after force-close")
        closed = post.iloc[0]
        assert str(closed["status"]).startswith("closed_"), closed["status"]
        assert closed["exit_reason"] == "time"
        # Exit price = yes_bid (we sell the dog into the bid)
        assert abs(float(closed["exit_price"]) - 0.10) < 1e-9


def test_orphan_position_far_from_resolution_stays_open():
    """Same orphan visibility, but resolution is >6h away AND yes_bid
    hasn't drifted up far enough for early-exit. Position should remain
    open."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "books").mkdir(parents=True)

        as_of = "2026-06-04T20:00:00+00:00"
        # 30h to resolution → outside force-exit window
        resolution = "2026-06-06T02:00:00+00:00"
        _write_book(data_dir / "books", venue="polymarket",
                    market_id="ORPHAN-2",
                    snapshot_ts="2026-06-04T19:00:00+00:00",
                    yes_bid=0.16, yes_ask=0.18)
        # Entry was 0.15; current bid 0.16; locked = +0.01 < 0.05 → no early
        _write_held_dog(data_dir, venue="polymarket",
                         market_id="ORPHAN-2",
                         entry_price=0.15,
                         resolution_ts=resolution,
                         entry_ts="2026-06-02T12:00:00+00:00")
        _write_init_state(data_dir)

        result = volharvest.tick(data_dir, markets=[], as_of_ts=as_of,
                                  bankroll=10000.0)

        assert result["orphans_visited"] == 1
        assert result["close_dogonly"] == 0
        assert result["close_resolution"] == 0
        assert result["open_dog"] == 0      # must NOT re-enter

        post = pd.read_parquet(data_dir / "paper_trades.parquet")
        assert (post["status"] == "open").sum() == 1, (
            "the position should still be open")


def test_orphan_stub_does_not_trigger_re_entry():
    """Once early-exit closes the dog, the entry branch could theoretically
    re-fire because own_by_market is cleared. For an orphan stub, this
    must NOT happen — re-entering on a stale book commits capital at a
    price the daemon can't refresh."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "books").mkdir(parents=True)

        as_of = "2026-06-04T20:00:00+00:00"
        resolution = "2026-06-06T02:00:00+00:00"   # well outside force window
        # yes_bid >> entry: triggers early-exit AND looks like a fresh
        # underdog (in [0.10, 0.30]) for re-entry purposes.
        _write_book(data_dir / "books", venue="kalshi",
                    market_id="ORPHAN-3",
                    snapshot_ts="2026-06-04T19:00:00+00:00",
                    yes_bid=0.26, yes_ask=0.28)
        _write_held_dog(data_dir, venue="kalshi",
                         market_id="ORPHAN-3",
                         entry_price=0.18,
                         resolution_ts=resolution,
                         entry_ts="2026-06-02T12:00:00+00:00")
        _write_init_state(data_dir)

        result = volharvest.tick(data_dir, markets=[], as_of_ts=as_of,
                                  bankroll=10000.0)

        # Early-exit fires (locked = 0.26 - 0.18 = 0.08 >= 0.05).
        assert result["orphans_visited"] == 1
        assert result["close_dogonly"] == 1
        # CRITICAL: must not re-open on the orphan's stale book.
        assert result["open_dog"] == 0, (
            f"entry branch re-fired on a stale orphan stub: {result}")


def test_non_orphan_market_in_markets_list_works_unchanged():
    """Sanity check: the patch must not break the standard path. When the
    market IS in `markets`, the existing loop handles it and orphans_visited
    is 0."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "books").mkdir(parents=True)

        as_of = "2026-06-04T20:00:00+00:00"
        resolution = "2026-06-04T21:00:00+00:00"   # force window
        _write_book(data_dir / "books", venue="kalshi",
                    market_id="LIVE-1",
                    snapshot_ts="2026-06-04T19:30:00+00:00",
                    yes_bid=0.10, yes_ask=0.12)
        _write_held_dog(data_dir, venue="kalshi",
                         market_id="LIVE-1",
                         entry_price=0.15,
                         resolution_ts=resolution,
                         entry_ts="2026-06-04T15:00:00+00:00")
        _write_init_state(data_dir)

        markets = [{
            "venue": "kalshi",
            "market_id": "LIVE-1",
            "end_date": resolution,
            "category": "other",
        }]
        result = volharvest.tick(data_dir, markets=markets, as_of_ts=as_of,
                                  bankroll=10000.0)

        assert result["orphans_visited"] == 0, (
            "LIVE-1 is in markets — should not be considered orphan")
        assert result["close_dogonly"] == 1
