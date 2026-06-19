"""Regression tests for the 2026-06-05 critical fixes:

  Bug 1: orphan-exit pattern in paper.tick and resolution.tick
  Bug 2: stale-bid guard on force-close (paper, volharvest, resolution)
  Bug 3: cash floor in sizing._apply_caps
  Bug 6: string-compare lookahead → parsed-datetime in latest_book_top
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from driftedge.paper import (
    BookTop,
    EntryRule,
    latest_book_top,
    parse_iso,
    hours_until,
    should_close,
)
from driftedge import paper, sizing
from driftedge.agents import resolution, volharvest


# ── Helpers ──────────────────────────────────────────────────────────────

def _write_book(books_root: Path, venue: str, market_id: str,
                snapshot_ts: str, yes_bid: float, yes_ask: float) -> None:
    md = books_root / venue / market_id
    md.mkdir(parents=True, exist_ok=True)
    rows = [
        {"snapshot_ts": snapshot_ts, "market_id": market_id, "token_id": "tok",
         "side": "bid", "level": 0, "price": yes_bid, "size": 1000.0},
        {"snapshot_ts": snapshot_ts, "market_id": market_id, "token_id": "tok",
         "side": "ask", "level": 0, "price": yes_ask, "size": 1000.0},
    ]
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False),
                   md / f"{snapshot_ts[:10]}.parquet", compression="snappy")


def _empty_state(data_dir: Path, trader: str = "kelly", bankroll: float = 10000.0,
                   cash: float = 10000.0, exposure: float = 0.0,
                   closed: float = 0.0) -> None:
    df = pd.DataFrame([{
        "trader": trader, "bankroll_init": bankroll, "cash_usd": cash,
        "open_exposure": exposure, "closed_pnl": closed,
        "peak_equity": bankroll, "current_drawdown_pct": 0.0,
        "updated_ts": "2026-06-05T00:00:00+00:00",
    }])
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   data_dir / "paper_state.parquet", compression="snappy")


# ── Bug 3: cash floor ────────────────────────────────────────────────────

def test_cash_floor_clamps_size_to_available_cash():
    """Trader with $50 cash but $5k exposure headroom must not open a
    position larger than $50."""
    state = sizing.TraderState(
        trader="kelly", bankroll_init=10000.0,
        cash_usd=50.0,                  # very low
        open_exposure=0.0,              # no exposure — headroom is full $5k
        closed_pnl=-9950.0,
    )
    size = sizing.equal_weight_size(state, c=0.30, target=0.60, stop=0.20)
    # MIN_POSITION_USD is 5.00; cash=$50 caps size to $50.
    assert 0 < size <= 50.0


def test_cash_floor_returns_zero_when_cash_below_min():
    """Cash below the $5 min-position floor should return 0."""
    state = sizing.TraderState(
        trader="kelly", bankroll_init=10000.0,
        cash_usd=2.0,                   # below the MIN_POSITION_USD = 5.0
        open_exposure=0.0,
        closed_pnl=-9998.0,
    )
    assert sizing.equal_weight_size(state, c=0.30, target=0.60, stop=0.20) == 0.0


def test_cash_floor_does_not_change_normal_case():
    """When cash >> required size, the cash floor is a no-op."""
    state = sizing.TraderState(
        trader="kelly", bankroll_init=10000.0,
        cash_usd=10000.0, open_exposure=0.0, closed_pnl=0.0,
    )
    size = sizing.equal_weight_size(state, c=0.30, target=0.60, stop=0.20)
    # Per-position cap is 2% × 10000 = $200.
    assert size == 200.0


# ── Bug 6: parsed-datetime lookahead, accepts mixed Z / +00:00 ──────────

def test_latest_book_top_accepts_z_suffix_snapshot_with_plus_00_as_of():
    """A snapshot timestamp with 'Z' suffix and an as_of_ts with '+00:00'
    represent the same instant; lex-compare flagged this as lookahead
    incorrectly. Must now resolve correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        books_root = Path(tmp)
        # Snapshot: Z; as_of_ts: +00:00. Same instant.
        _write_book(books_root, "kalshi", "M-Z-1",
                    snapshot_ts="2026-06-05T10:00:00Z",
                    yes_bid=0.45, yes_ask=0.46)
        top = latest_book_top(books_root / "kalshi", "M-Z-1",
                              as_of_ts="2026-06-05T10:00:00+00:00")
        assert top is not None
        assert top.best_ask == 0.46


def test_latest_book_top_still_rejects_truly_future_snapshot_with_mixed_suffix():
    """Even with mixed Z/+00:00, a genuinely future snapshot must be
    filtered out by the as_of_ts boundary."""
    with tempfile.TemporaryDirectory() as tmp:
        books_root = Path(tmp)
        _write_book(books_root, "kalshi", "M-Z-2",
                    snapshot_ts="2026-06-05T11:00:00Z",  # +1 h in future
                    yes_bid=0.45, yes_ask=0.46)
        top = latest_book_top(books_root / "kalshi", "M-Z-2",
                              as_of_ts="2026-06-05T10:00:00+00:00")
        assert top is None    # the only snapshot is future → filtered out


def test_parse_iso_handles_all_three_common_suffix_styles():
    a = parse_iso("2026-06-05T10:00:00Z")
    b = parse_iso("2026-06-05T10:00:00+00:00")
    c = parse_iso("2026-06-05T10:00:00")        # naïve → assumed UTC
    assert a == b == c


# ── Bug 2: stale-bid guard on force-close ────────────────────────────────

def test_should_close_returns_none_when_book_is_pre_resolution_and_past_resolution():
    """Post-resolution force-close with a pre-resolution book → defer.
    Bid=0.20 EQUALS the rule.stop threshold; without the guard `stop`
    would fire on a stale quote, recording the wrong exit price."""
    rule = EntryRule()
    book = BookTop(snapshot_ts="2026-06-05T09:00:00+00:00",   # 1 h pre-resolution
                   best_bid=0.20, best_ask=0.22,
                   bid_depth=100, ask_depth=100)
    position = {"market_id": "M1", "entry_price": 0.32}
    reason = should_close(book, position, rule,
                          as_of_ts="2026-06-05T11:00:00+00:00",   # 1 h post-resolution
                          resolution_ts="2026-06-05T10:00:00+00:00")
    # Guard fires BEFORE stop check → None despite bid <= stop.
    assert reason is None


def test_should_close_fires_time_when_book_is_post_resolution():
    """Post-resolution with a fresh (post-resolution) book in the
    between-stop-and-target zone → proceed to time-based close."""
    rule = EntryRule()
    book = BookTop(snapshot_ts="2026-06-05T10:30:00+00:00",   # 0.5 h post-resolution
                   best_bid=0.45, best_ask=0.47,              # between stop=0.2, target=0.6
                   bid_depth=100, ask_depth=100)
    position = {"market_id": "M1", "entry_price": 0.32}
    reason = should_close(book, position, rule,
                          as_of_ts="2026-06-05T11:00:00+00:00",
                          resolution_ts="2026-06-05T10:00:00+00:00")
    assert reason == "time"


def test_should_close_with_post_resolution_book_fires_target_at_yes_settlement():
    """Post-resolution YES settlement (bid ≈ 0.98) with a fresh book
    should fire target — that's the correct realization of the
    settlement payoff at the take-profit threshold."""
    rule = EntryRule()
    book = BookTop(snapshot_ts="2026-06-05T10:30:00+00:00",
                   best_bid=0.98, best_ask=0.99,
                   bid_depth=100, ask_depth=100)
    position = {"market_id": "M1", "entry_price": 0.32}
    reason = should_close(book, position, rule,
                          as_of_ts="2026-06-05T11:00:00+00:00",
                          resolution_ts="2026-06-05T10:00:00+00:00")
    assert reason == "target"


def test_should_close_fires_time_in_pre_resolution_force_window():
    """Pre-resolution but within the force-exit window → still close
    (the bid reflects the current live quote, not stale)."""
    rule = EntryRule()
    book = BookTop(snapshot_ts="2026-06-05T09:55:00+00:00",
                   best_bid=0.30, best_ask=0.32,
                   bid_depth=100, ask_depth=100)
    position = {"market_id": "M1", "entry_price": 0.32}
    # resolution is 1 h away; force_exit_hours = 6 → triggers.
    reason = should_close(book, position, rule,
                          as_of_ts="2026-06-05T09:00:00+00:00",
                          resolution_ts="2026-06-05T10:00:00+00:00")
    assert reason == "time"


def test_volharvest_should_force_close_blocks_on_stale_post_resolution_book():
    rule = volharvest.VolHarvestRule()
    book = BookTop(snapshot_ts="2026-06-05T09:00:00+00:00",
                   best_bid=0.20, best_ask=0.22,
                   bid_depth=100, ask_depth=100)
    # hours_left = -1, book pre-resolution → False.
    assert volharvest.should_force_close(
        book, rule,
        as_of_ts="2026-06-05T11:00:00+00:00",
        resolution_ts="2026-06-05T10:00:00+00:00") is False


def test_resolution_should_force_close_blocks_on_stale_post_resolution_book():
    rule = resolution.ResolutionRule()
    book = BookTop(snapshot_ts="2026-06-05T09:00:00+00:00",
                   best_bid=0.20, best_ask=0.22,
                   bid_depth=100, ask_depth=100)
    assert resolution.should_force_close(
        rule,
        as_of_ts="2026-06-05T11:00:00+00:00",
        resolution_ts="2026-06-05T10:00:00+00:00",
        book=book) is False


# ── Bug 1: orphan-exit safety net in paper.tick + resolution.tick ───────

def test_paper_tick_visits_orphan_position_and_force_closes():
    """A held kelly position on a market not in `markets` must still get
    its exit logic run via the synthetic-stub prepend."""
    with tempfile.TemporaryDirectory() as tmp:
        dd = Path(tmp)
        (dd / "books").mkdir()
        as_of = "2026-06-05T09:00:00+00:00"
        # 1 h to resolution → within 6 h force-exit window.
        resolution_ts = "2026-06-05T10:00:00+00:00"
        # Book is at as_of (fresh, pre-resolution).
        _write_book(dd / "books", "kalshi", "ORPHAN-1",
                    snapshot_ts="2026-06-05T08:55:00+00:00",
                    yes_bid=0.31, yes_ask=0.33)
        # Held position on ORPHAN-1.
        pos = {
            "trade_id": str(uuid.uuid4()), "trader": "kelly", "venue": "kalshi",
            "market_id": "ORPHAN-1", "question": "Q", "category": "other",
            "yes_token_id": "tok", "entry_ts": "2026-06-04T12:00:00+00:00",
            "entry_snapshot_ts": "2026-06-04T12:00:00+00:00",
            "entry_price": 0.35, "entry_size_usd": 200.0, "shares": 571.42,
            "resolution_ts": resolution_ts,
            "target": 0.60, "stop": 0.20, "status": "open",
            "exit_ts": None, "exit_snapshot_ts": None, "exit_price": None,
            "exit_reason": None, "pnl_per_share": None, "pnl_usd": None,
        }
        pd.DataFrame([pos]).to_parquet(dd / "paper_trades.parquet",
                                          compression="snappy", index=False)
        _empty_state(dd, trader="kelly", cash=9800.0, exposure=200.0)

        # Tick with empty markets list — ORPHAN-1 only reachable via stub.
        result = paper.tick(dd, markets=[], rule=EntryRule(),
                            as_of_ts=as_of, bankroll=10000.0)

        assert result["orphans_visited"] == 1
        # Force-close should fire (1 h to resolution, within 6 h window).
        assert result["closed_by_trader"]["kelly"] == 1


def test_resolution_tick_visits_orphan_position_and_force_closes():
    with tempfile.TemporaryDirectory() as tmp:
        dd = Path(tmp)
        (dd / "books").mkdir()
        as_of = "2026-06-05T09:30:00+00:00"
        # 30 min to resolution → within 1 h force-exit window (resolution
        # agent uses force_exit_h = 1.0).
        resolution_ts = "2026-06-05T10:00:00+00:00"
        _write_book(dd / "books", "kalshi", "RES-ORPHAN-1",
                    snapshot_ts="2026-06-05T09:25:00+00:00",
                    yes_bid=0.40, yes_ask=0.42)
        pos = {
            "trade_id": str(uuid.uuid4()), "trader": "resolution",
            "venue": "kalshi", "market_id": "RES-ORPHAN-1",
            "question": "Q", "category": "other", "yes_token_id": "tok",
            "leg": "yes", "side": "yes",
            "entry_ts": "2026-06-04T12:00:00+00:00",
            "entry_snapshot_ts": "2026-06-04T12:00:00+00:00",
            "entry_price": 0.35, "entry_size_usd": 200.0, "shares": 571.42,
            "resolution_ts": resolution_ts,
            "target": None, "stop": None, "status": "open",
            "exit_ts": None, "exit_snapshot_ts": None, "exit_price": None,
            "exit_reason": None, "pnl_per_share": None, "pnl_usd": None,
        }
        pd.DataFrame([pos]).to_parquet(dd / "paper_trades.parquet",
                                          compression="snappy", index=False)
        _empty_state(dd, trader="resolution", cash=9800.0, exposure=200.0)

        result = resolution.tick(dd, markets=[], as_of_ts=as_of,
                                  bankroll=10000.0)

        assert result["orphans_visited"] == 1
        assert result["close_force"] == 1


def test_hours_until_helper():
    """Hoisted closure replacement (Bug 12). Behaviour sanity check."""
    assert hours_until("2026-06-05T10:00:00+00:00",
                       "2026-06-05T08:00:00+00:00") == 2.0
    assert hours_until("2026-06-05T08:00:00Z",
                       "2026-06-05T10:00:00+00:00") == -2.0
    assert hours_until(None, "2026-06-05T08:00:00+00:00") is None
    assert hours_until("bad", "2026-06-05T08:00:00+00:00") is None
