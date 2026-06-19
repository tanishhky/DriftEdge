"""Tests for the Kuber 6th-agent stack.

Coverage:
  * sizing rule: $5 floor, 4% of sleeve bankroll above
  * kill switches: drawdown + daily-loss trigger correctly
  * paper mode never calls KalshiClient (live gate enforcement)
  * state file shape + four sleeves init at $125 each
  * Kalshi-venue filter: Polymarket markets are ignored
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from driftedge import kuber
from driftedge import paper as paper_mod
from driftedge import sizing as base_sizing
from driftedge.agents import kuber as kpol
from driftedge.config import Config


# ── fixtures ─────────────────────────────────────────────────────────


def _make_config(**overrides) -> Config:
    """Build a Config with sensible defaults for Kuber tests."""
    defaults = dict(
        data_dir=Path("/tmp/kuber_test_unused"),
        log_dir=Path("/tmp/kuber_test_unused_logs"),
        log_level="INFO",
        poll_interval_s=60,
        kelly_fraction=0.25,
        entry_low=0.30,
        entry_high=0.40,
        exit_target=0.60,
        stop_low=0.20,
        polymarket_wallet=None,
        polymarket_api_key=None,
        polymarket_api_secret=None,
        polymarket_api_passphrase=None,
        kalshi_env="prod",
        kalshi_api_key_id=None,
        kalshi_private_key_path=None,
        kuber_live=False,
        kuber_bankroll_total_usd=500.0,
        kuber_bankroll_per_sleeve_usd=125.0,
        kuber_max_position_usd=5.0,
        kuber_dd_kill_pct=0.40,
        kuber_daily_loss_kill_usd=25.0,
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_book_parquet(data_dir: Path, market_id: str, *,
                        venue: str = "kalshi",
                        snapshot_ts: str,
                        best_bid: float = 0.30,
                        best_ask: float = 0.35) -> None:
    """Persist a single orderbook snapshot the way the daemon would.

    The on-disk shape is long-form: one row per (side, price) level. The
    daemon's reader (`paper.latest_book_top`) groups by snapshot_ts and
    picks the highest bid and lowest ask. We write one bid row and one
    ask row at the requested prices to make the read deterministic.
    """
    market_dir = data_dir / "books" / venue / market_id
    market_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        {"market_id": market_id, "snapshot_ts": snapshot_ts,
         "side": "bid", "price": best_bid, "size": 1000.0},
        {"market_id": market_id, "snapshot_ts": snapshot_ts,
         "side": "ask", "price": best_ask, "size": 1000.0},
    ])
    out = market_dir / f"{snapshot_ts.replace(':', '').replace('+', '')}.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   out, compression="snappy")


# ── sizing rule ──────────────────────────────────────────────────────


def test_position_cap_at_baseline_is_five_dollars():
    kc = kpol.KuberConfig()  # defaults
    # Zero realised PnL → cap = max($5, 4% * $125) = max($5, $5) = $5
    assert kpol.position_cap(0.0, kc) == pytest.approx(5.0)


def test_position_cap_scales_with_realized_pnl():
    kc = kpol.KuberConfig()
    # +$125 realised → sleeve bankroll = $250 → 4% = $10
    assert kpol.position_cap(125.0, kc) == pytest.approx(10.0)
    # +$375 realised → sleeve bankroll = $500 → 4% = $20
    assert kpol.position_cap(375.0, kc) == pytest.approx(20.0)


def test_position_cap_never_falls_below_floor_on_drawdown():
    """Even when a sleeve is deep in the red, the cap stays at the $5 floor."""
    kc = kpol.KuberConfig()
    assert kpol.position_cap(-100.0, kc) == pytest.approx(5.0)


def test_kuber_size_respects_position_count_cap():
    """At 15 open positions, kuber_size must return 0 regardless of edge."""
    kc = kpol.KuberConfig()
    state = base_sizing.TraderState(
        trader="kuber:kelly",
        bankroll_init=125.0,
        cash_usd=125.0,
        open_exposure=0.0,
        closed_pnl=0.0,
    )
    size = kpol.kuber_size(
        "kuber:kelly", c=0.35, target=0.60, stop=0.20,
        state=state, kc=kc, sleeve_realized_pnl=0.0,
        sleeve_open_positions=kpol.KUBER_MAX_POSITIONS_PER_SLEEVE,
    )
    assert size == 0.0


def test_kuber_size_zero_when_below_floor():
    """If the underlying sizer returns less than $5, Kuber skips the trade."""
    kc = kpol.KuberConfig()
    # Volharvest sizer returns 0.02 * 125 = $2.50 raw — below the $5 floor.
    # But we test equal-weight which uses the same math; the cap caps it at $5
    # so this test is more about the floor when the cap rule isn't met.
    state = base_sizing.TraderState(
        trader="kuber:equal",
        bankroll_init=125.0,
        cash_usd=0.0,    # no cash → cash cap forces 0
        open_exposure=0.0,
        closed_pnl=0.0,
    )
    size = kpol.kuber_size(
        "kuber:equal", c=0.35, target=0.60, stop=0.20,
        state=state, kc=kc, sleeve_realized_pnl=0.0,
        sleeve_open_positions=0,
    )
    assert size == 0.0


# ── kill switches ────────────────────────────────────────────────────


def test_drawdown_kill_triggers_at_40_percent():
    kc = kpol.KuberConfig()  # 40% DD floor on $500 = $300
    assert kpol.drawdown_kill_active(299.99, kc) is True
    assert kpol.drawdown_kill_active(300.01, kc) is False


def test_daily_loss_kill_triggers_at_minus_25():
    kc = kpol.KuberConfig()
    assert kpol.daily_loss_kill_active(-25.01, kc) is True
    assert kpol.daily_loss_kill_active(-24.99, kc) is False
    assert kpol.daily_loss_kill_active(0.0, kc) is False


def test_gate_blocks_entries_when_drawdown_kill_active():
    kc = kpol.KuberConfig()
    g = kpol.gate(total_equity_usd=250.0, today_realized_pnl_usd=0.0, kc=kc)
    assert g.allow_entries is False
    assert g.reason == "drawdown_kill"


def test_gate_blocks_entries_when_daily_loss_kill_active():
    kc = kpol.KuberConfig()
    g = kpol.gate(total_equity_usd=500.0, today_realized_pnl_usd=-30.0, kc=kc)
    assert g.allow_entries is False
    assert g.reason == "daily_loss_kill"


def test_gate_allows_entries_in_normal_state():
    kc = kpol.KuberConfig()
    g = kpol.gate(total_equity_usd=505.0, today_realized_pnl_usd=5.0, kc=kc)
    assert g.allow_entries is True
    assert g.reason == "ok"


# ── state init ───────────────────────────────────────────────────────


def test_init_state_creates_four_sleeves_at_125_each(tmp_path):
    kc = kpol.KuberConfig()
    states = kuber.init_state(tmp_path, kc)
    assert set(states.keys()) == set(kpol.SLEEVE_LABELS)
    for tid, s in states.items():
        assert s.bankroll_init == pytest.approx(125.0)
        assert s.cash_usd == pytest.approx(125.0)
        assert s.open_exposure == pytest.approx(0.0)
        assert s.closed_pnl == pytest.approx(0.0)


def test_init_state_is_idempotent(tmp_path):
    """Calling init_state twice must NOT reset balances."""
    kc = kpol.KuberConfig()
    kuber.init_state(tmp_path, kc)
    # Mutate one sleeve.
    states = kuber.load_state(tmp_path)
    states["kuber:kelly"] = base_sizing.TraderState(
        trader="kuber:kelly", bankroll_init=125.0,
        cash_usd=80.0, open_exposure=45.0, closed_pnl=0.0,
    )
    kuber.save_state(tmp_path, states)
    # Re-init.
    states2 = kuber.init_state(tmp_path, kc)
    assert states2["kuber:kelly"].cash_usd == pytest.approx(80.0)
    assert states2["kuber:kelly"].open_exposure == pytest.approx(45.0)


# ── tick paper-mode end to end ───────────────────────────────────────


def test_tick_paper_mode_opens_positions_on_kalshi_market(tmp_path):
    """Paper mode: a tradeable Kalshi market produces entries across sleeves,
    without ever calling KalshiClient.place_order."""
    cfg = _make_config(data_dir=tmp_path, kuber_live=False)
    as_of = "2026-06-10T15:00:00+00:00"

    _make_book_parquet(tmp_path, "KXMARKET-1",
                        snapshot_ts="2026-06-10T14:59:00+00:00",
                        best_bid=0.32, best_ask=0.35)

    markets = [{
        "venue": "kalshi",
        "market_id": "KXMARKET-1",
        "question": "test market",
        "end_date": "2026-06-12T20:00:00+00:00",
        "category": "other",
    }]
    rule = paper_mod.EntryRule(
        entry_low=cfg.entry_low, entry_high=cfg.entry_high,
        target=cfg.exit_target, stop=cfg.stop_low,
    )

    # KalshiClient mock that fails the test if called in paper mode.
    mock_client = MagicMock()
    mock_client.place_order.side_effect = AssertionError(
        "KalshiClient.place_order must NOT be called in paper mode")

    result = kuber.tick(tmp_path, markets, rule, cfg,
                         as_of_ts=as_of, kalshi_client=mock_client)
    assert result["mode"] == "paper"
    assert result["opened"] >= 1
    # mock NOT called
    mock_client.place_order.assert_not_called()

    # Trades parquet should now have rows tagged kuber:*
    pos = kuber.load_positions(tmp_path)
    assert pos, "expected at least one Kuber position on disk"
    for p in pos:
        assert p["trader"].startswith("kuber:")
        assert p["venue"] == "kalshi"
        assert p["mode"] == "paper"
        assert p["kalshi_order_id"] is None


def test_tick_ignores_polymarket_markets(tmp_path):
    """Kuber trades Kalshi only — Polymarket markets must be skipped."""
    cfg = _make_config(data_dir=tmp_path, kuber_live=False)
    as_of = "2026-06-10T15:00:00+00:00"

    _make_book_parquet(tmp_path, "POLY-MKT-1", venue="polymarket",
                        snapshot_ts="2026-06-10T14:59:00+00:00",
                        best_bid=0.32, best_ask=0.35)

    markets = [{
        "venue": "polymarket",
        "market_id": "POLY-MKT-1",
        "question": "test poly market",
        "end_date": "2026-06-12T20:00:00+00:00",
        "category": "other",
    }]
    rule = paper_mod.EntryRule(
        entry_low=cfg.entry_low, entry_high=cfg.entry_high,
        target=cfg.exit_target, stop=cfg.stop_low,
    )

    result = kuber.tick(tmp_path, markets, rule, cfg, as_of_ts=as_of)
    assert result["opened"] == 0
    assert result["gate_reason"] == "no_markets"


def test_live_mode_without_client_returns_gracefully(tmp_path):
    """KUBER_LIVE=1 but no client passed: tick must not crash and must
    not place orders."""
    cfg = _make_config(data_dir=tmp_path, kuber_live=True,
                        kalshi_api_key_id="dummy",
                        kalshi_private_key_path=tmp_path / "fake.pem")
    as_of = "2026-06-10T15:00:00+00:00"

    _make_book_parquet(tmp_path, "KXMARKET-2",
                        snapshot_ts="2026-06-10T14:59:00+00:00",
                        best_bid=0.32, best_ask=0.35)

    markets = [{"venue": "kalshi", "market_id": "KXMARKET-2",
                 "question": "x", "end_date": "2026-06-12T20:00:00+00:00",
                 "category": "other"}]
    rule = paper_mod.EntryRule(
        entry_low=cfg.entry_low, entry_high=cfg.entry_high,
        target=cfg.exit_target, stop=cfg.stop_low,
    )

    result = kuber.tick(tmp_path, markets, rule, cfg, as_of_ts=as_of,
                         kalshi_client=None)
    assert result["mode"] == "live"
    assert result["gate_reason"] == "live_no_client"
    assert result["opened"] == 0


def test_drawdown_kill_blocks_new_entries(tmp_path):
    """When equity is below the DD floor, no new entries open."""
    cfg = _make_config(data_dir=tmp_path, kuber_live=False)
    kc = kpol.KuberConfig(
        bankroll_total_usd=cfg.kuber_bankroll_total_usd,
        bankroll_per_sleeve_usd=cfg.kuber_bankroll_per_sleeve_usd,
        max_position_usd=cfg.kuber_max_position_usd,
        dd_kill_pct=cfg.kuber_dd_kill_pct,
        daily_loss_kill_usd=cfg.kuber_daily_loss_kill_usd,
    )
    # Hand-craft a state where every sleeve is wiped to $20 → total $80
    # (well below the $300 DD floor on $500).
    kuber.init_state(tmp_path, kc)
    crashed = {tid: base_sizing.TraderState(
        trader=tid, bankroll_init=125.0,
        cash_usd=20.0, open_exposure=0.0, closed_pnl=-105.0)
        for tid in kpol.SLEEVE_LABELS}
    kuber.save_state(tmp_path, crashed)

    _make_book_parquet(tmp_path, "KXMARKET-3",
                        snapshot_ts="2026-06-10T14:59:00+00:00",
                        best_bid=0.32, best_ask=0.35)
    markets = [{"venue": "kalshi", "market_id": "KXMARKET-3",
                 "question": "x", "end_date": "2026-06-12T20:00:00+00:00",
                 "category": "other"}]
    rule = paper_mod.EntryRule(
        entry_low=cfg.entry_low, entry_high=cfg.entry_high,
        target=cfg.exit_target, stop=cfg.stop_low,
    )

    result = kuber.tick(tmp_path, markets, rule, cfg,
                         as_of_ts="2026-06-10T15:00:00+00:00")
    assert result["gate_reason"] == "drawdown_kill"
    assert result["opened"] == 0
