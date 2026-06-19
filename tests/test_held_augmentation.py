"""Regression: the poll daemon must keep held markets in the tracked
list even when they fall out of the volume/window/price filters.

Original bug: `cmd_poll` computed `tracked_poly` and `tracked_kalshi` via
top-N-by-volume after a tradeable-window + near-certain filter. Any held
position whose market drifted out of these filters lost its book polling.
Without book polling, exit logic on the consumer side (paper.tick,
volharvest.tick, resolution.tick) couldn't compute fresh prices, MTM, or
force-close anything.

Fix: `_augment_tracked_with_held(tracked, venue, data_dir)` finds open
positions whose `(venue, market_id)` isn't in `tracked` and looks each
one up in the most recent venue markets parquet. Found stubs are
appended; missing ones are skipped (the agent-side orphan loop is
responsible for those).

These tests pin the augmentation behaviour:
  1. A held market NOT in tracked IS added.
  2. A held market ALREADY in tracked is NOT double-added.
  3. A held market with NO parquet record at all is skipped.
  4. Closed positions are NOT augmented (only `status=='open'`).
  5. Cross-venue isolation: a held kalshi market doesn't get added to
     the polymarket tracked list.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pandas as pd

from driftedge.cli import (
    _augment_tracked_with_held,
    _held_market_ids_by_venue,
    _market_stub_from_parquets,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _write_markets_parquet(data_dir: Path, venue: str, day: str,
                            rows: list[dict]) -> None:
    p = data_dir / "markets" / venue
    p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(p / f"{day}.parquet",
                                   compression="snappy", index=False)


def _write_positions(data_dir: Path, positions: list[dict]) -> None:
    pd.DataFrame(positions).to_parquet(data_dir / "paper_trades.parquet",
                                        compression="snappy", index=False)


def _pos(venue: str, market_id: str, status: str = "open",
         trader: str = "volharvest") -> dict:
    return {
        "trade_id": str(uuid.uuid4()),
        "trader": trader,
        "venue": venue,
        "market_id": market_id,
        "status": status,
        "entry_price": 0.15,
        "entry_size_usd": 200.0,
        "shares": 1333.33,
        "leg": "dog",
    }


def _market_row(venue: str, market_id: str, end_date: str = "2026-06-10T00:00Z",
                yes_token_id: str = "tok", **extra) -> dict:
    base = {
        "venue": venue,
        "market_id": market_id,
        "condition_id": "cond",
        "slug": "slug",
        "question": "Q?",
        "category": "other",
        "end_date": end_date,
        "yes_token_id": yes_token_id,
        "no_token_id": "tok_no",
        "yes_label": "Yes",
        "no_label": "No",
        "yes_price": 0.5,
        "no_price": 0.5,
        "volume_24h": 1000.0,
        "volume_total": 5000.0,
        "best_bid": 0.49,
        "best_ask": 0.51,
        "active": True,
        "closed": False,
        "liquidity": 1.0,
    }
    base.update(extra)
    return base


# ── Tests ────────────────────────────────────────────────────────────────

def test_held_market_not_in_tracked_is_added():
    """The headline behaviour."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        # 1 held kalshi position; 1 kalshi market in the parquet for that id;
        # tracked list is empty.
        _write_positions(data_dir, [_pos("kalshi", "K-HELD-1")])
        _write_markets_parquet(data_dir, "kalshi", "2026-06-04",
                                 [_market_row("kalshi", "K-HELD-1")])

        augmented, n_added = _augment_tracked_with_held([], "kalshi", data_dir)

        assert n_added == 1
        assert len(augmented) == 1
        assert augmented[0]["market_id"] == "K-HELD-1"
        # The added stub must carry the fields the downstream code reads.
        assert augmented[0]["end_date"] == "2026-06-10T00:00Z"
        assert augmented[0]["yes_token_id"] == "tok"


def test_held_market_already_in_tracked_is_not_double_added():
    """No duplicates."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        _write_positions(data_dir, [_pos("kalshi", "K-DUP-1")])
        _write_markets_parquet(data_dir, "kalshi", "2026-06-04",
                                 [_market_row("kalshi", "K-DUP-1")])

        # K-DUP-1 is already in tracked from the top-N filter.
        existing = [_market_row("kalshi", "K-DUP-1", question="from top-N")]
        augmented, n_added = _augment_tracked_with_held(
            existing, "kalshi", data_dir)

        assert n_added == 0
        assert len(augmented) == 1
        assert augmented[0]["question"] == "from top-N"   # original preserved


def test_held_market_missing_from_all_parquets_is_skipped():
    """If a held market has no metadata anywhere, we can't synthesize a
    stub. Skip and let the agent-side orphan loop handle it."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        _write_positions(data_dir, [_pos("kalshi", "K-MISSING-1")])
        # Only an unrelated market in the parquet.
        _write_markets_parquet(data_dir, "kalshi", "2026-06-04",
                                 [_market_row("kalshi", "K-OTHER-1")])

        augmented, n_added = _augment_tracked_with_held([], "kalshi", data_dir)

        assert n_added == 0
        assert augmented == []


def test_closed_positions_are_not_augmented():
    """Only open positions need their markets polled."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        _write_positions(data_dir, [
            _pos("kalshi", "K-OPEN-1", status="open"),
            _pos("kalshi", "K-CLOSED-1", status="closed_time"),
        ])
        _write_markets_parquet(data_dir, "kalshi", "2026-06-04", [
            _market_row("kalshi", "K-OPEN-1"),
            _market_row("kalshi", "K-CLOSED-1"),
        ])

        augmented, n_added = _augment_tracked_with_held([], "kalshi", data_dir)

        assert n_added == 1
        assert augmented[0]["market_id"] == "K-OPEN-1"


def test_cross_venue_isolation():
    """A held kalshi market must not get added to the polymarket tracked
    list, and vice versa."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        _write_positions(data_dir, [
            _pos("kalshi", "K-1"),
            _pos("polymarket", "P-1"),
        ])
        _write_markets_parquet(data_dir, "kalshi", "2026-06-04",
                                 [_market_row("kalshi", "K-1")])
        _write_markets_parquet(data_dir, "polymarket", "2026-06-04",
                                 [_market_row("polymarket", "P-1")])

        aug_poly, n_poly = _augment_tracked_with_held(
            [], "polymarket", data_dir)
        aug_kal, n_kal = _augment_tracked_with_held(
            [], "kalshi", data_dir)

        assert n_poly == 1 and aug_poly[0]["market_id"] == "P-1"
        assert n_kal == 1 and aug_kal[0]["market_id"] == "K-1"


def test_held_market_ids_by_venue_indexes_correctly():
    """Spot-check the underlying ID helper."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        _write_positions(data_dir, [
            _pos("kalshi", "K-1"),
            _pos("kalshi", "K-2"),
            _pos("polymarket", "P-1"),
            _pos("polymarket", "P-2", status="closed_time"),  # excluded
        ])
        held = _held_market_ids_by_venue(data_dir)

        assert held == {"kalshi": {"K-1", "K-2"}, "polymarket": {"P-1"}}


def test_stub_lookup_prefers_most_recent_parquet():
    """When the same market_id appears in multiple daily parquets, the
    most recent file wins (handles updated end_dates)."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        _write_markets_parquet(data_dir, "kalshi", "2026-06-02",
            [_market_row("kalshi", "K-1", end_date="2026-06-05T00:00Z")])
        _write_markets_parquet(data_dir, "kalshi", "2026-06-04",
            [_market_row("kalshi", "K-1", end_date="2026-06-09T00:00Z")])

        stub = _market_stub_from_parquets("kalshi", "K-1", data_dir)
        assert stub is not None
        assert stub["end_date"] == "2026-06-09T00:00Z", (
            "should pick the newer parquet's end_date")
