"""Tests for the 2026-07-04 fixes:

  1. settlement.settle_stuck_positions — zombie positions on resolved
     markets settle from the venue outcome (0/1), state rolls correctly,
     unresolved markets are left alone, fallback ladder works.
  2. calibration.bet_stats — martingale prior gives EV exactly 0 (no
     evidence, no bet), evidence moves the posterior, and future-dated
     trades are excluded (no lookahead).
  3. volharvest adaptive max-horizon gate — cold start, adaptation from
     winners, and the entry-side rejections (missing resolution_ts,
     beyond-horizon resolution).
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from driftedge import calibration, settlement
from driftedge.agents import volharvest
from driftedge.data import paper_persist as pp
from driftedge.data import state_persist as sp
from driftedge.paper import BookTop


AS_OF = "2026-07-04T12:00:00+00:00"


# ── Helpers ──────────────────────────────────────────────────────────────

def _state_file(data_dir: Path, rows: list[dict]) -> None:
    df = pd.DataFrame([{
        "trader": r["trader"], "bankroll_init": 10000.0,
        "cash_usd": r.get("cash", 9800.0),
        "open_exposure": r.get("exposure", 200.0),
        "closed_pnl": r.get("closed", 0.0),
        "peak_equity": 10000.0, "current_drawdown_pct": 0.0,
        "updated_ts": "2026-07-01T00:00:00+00:00",
    } for r in rows])
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   data_dir / "paper_state.parquet", compression="snappy")


def _open_pos(trader: str, venue: str, market_id: str, *,
              entry_price: float = 0.20, size: float = 200.0,
              resolution_ts: str = "2026-07-02T00:00:00+00:00",
              side: str = "yes") -> dict:
    return {
        "trade_id": str(uuid.uuid4()), "trader": trader, "venue": venue,
        "market_id": market_id, "question": "q?", "category": "sports",
        "yes_token_id": "tok", "side": side, "leg": "dog",
        "entry_ts": "2026-07-01T00:00:00+00:00",
        "entry_snapshot_ts": "2026-07-01T00:00:00+00:00",
        "entry_price": entry_price, "entry_size_usd": size,
        "shares": size / entry_price, "resolution_ts": resolution_ts,
        "target": None, "stop": None, "status": "open",
        "exit_ts": None, "exit_snapshot_ts": None, "exit_price": None,
        "exit_reason": None, "pnl_per_share": None, "pnl_usd": None,
    }


class _FakePoly:
    """Gamma stub: {market_id: yes_outcome or None (=unresolved)}."""
    def __init__(self, outcomes: dict):
        self.outcomes = outcomes

    def get_market(self, market_id: str) -> dict:
        o = self.outcomes.get(market_id, "missing")
        if o == "missing":
            raise RuntimeError("404")
        if o is None:
            return {"id": market_id, "closed": False}
        return {"id": market_id, "closed": True,
                "umaResolutionStatus": "resolved",
                "outcomePrices": f'["{o}", "{1 - o}"]'}


class _FakeKalshi:
    def __init__(self, results: dict):
        self.results = results

    def get_market(self, ticker: str) -> dict:
        r = self.results.get(ticker)
        if r is None:
            return {"market": {"ticker": ticker, "status": "active",
                               "result": ""}}
        return {"market": {"ticker": ticker, "status": "finalized",
                           "result": r}}


# ── Settlement ───────────────────────────────────────────────────────────

def test_settlement_closes_zombies_at_venue_outcome():
    with tempfile.TemporaryDirectory() as tmp:
        dd = Path(tmp)
        _state_file(dd, [{"trader": "volharvest", "cash": 9600.0,
                          "exposure": 400.0},
                         {"trader": "equal", "cash": 9800.0,
                          "exposure": 200.0}])
        lost = _open_pos("volharvest", "polymarket", "PM1",
                         entry_price=0.20, size=200.0)
        won = _open_pos("equal", "kalshi", "KX1",
                        entry_price=0.35, size=200.0)
        extra = _open_pos("volharvest", "polymarket", "PM2",
                          entry_price=0.25, size=200.0,
                          resolution_ts="2026-08-01T00:00:00+00:00")  # future
        pp.upsert_positions(dd, opened=[lost, won, extra], closed=[])

        settlement._NEXT_CHECK.clear()
        counts = settlement.settle_stuck_positions(
            dd, poly_client=_FakePoly({"PM1": 0}),
            kalshi_client=_FakeKalshi({"KX1": "yes"}),
            as_of_ts=AS_OF)

        assert counts["settled"] == 2
        rows = {r["market_id"]: r for r in pp.load_positions(dd)}
        # Lost dog: settles at 0 → full premium gone.
        assert rows["PM1"]["status"] == "closed_settlement"
        assert rows["PM1"]["exit_price"] == 0.0
        assert rows["PM1"]["pnl_usd"] == pytest.approx(-200.0)
        # Won YES at 0.35: payoff 1 → pnl = (1-0.35) * (200/0.35)
        assert rows["KX1"]["status"] == "closed_settlement"
        assert rows["KX1"]["pnl_usd"] == pytest.approx(
            (1 - 0.35) * (200 / 0.35))
        # Unresolved market untouched.
        assert rows["PM2"]["status"] == "open"

        # State rolled: volharvest lost 200 (cash unchanged), equal won.
        states = sp.load_state(dd)
        assert states["volharvest"].open_exposure == pytest.approx(200.0)
        assert states["volharvest"].closed_pnl == pytest.approx(-200.0)
        assert states["equal"].open_exposure == pytest.approx(0.0)
        assert states["equal"].closed_pnl == pytest.approx(
            (1 - 0.35) * (200 / 0.35))


def test_settlement_leaves_unresolved_market_within_grace():
    with tempfile.TemporaryDirectory() as tmp:
        dd = Path(tmp)
        _state_file(dd, [{"trader": "equal"}])
        # Resolution 2h before as_of, venue says not resolved yet.
        pos = _open_pos("equal", "polymarket", "PM9",
                        resolution_ts="2026-07-04T10:00:00+00:00")
        pp.upsert_positions(dd, opened=[pos], closed=[])
        settlement._NEXT_CHECK.clear()
        counts = settlement.settle_stuck_positions(
            dd, poly_client=_FakePoly({"PM9": None}),
            kalshi_client=None, as_of_ts=AS_OF, grace_hours=24.0)
        assert counts["settled"] == 0 and counts["pending"] == 1
        assert pp.load_positions(dd)[0]["status"] == "open"


def test_settlement_writeoff_past_grace_with_no_book_and_no_outcome():
    with tempfile.TemporaryDirectory() as tmp:
        dd = Path(tmp)
        _state_file(dd, [{"trader": "equal"}])
        pos = _open_pos("equal", "polymarket", "PMX",
                        resolution_ts="2026-07-01T00:00:00+00:00")  # 3.5d ago
        pp.upsert_positions(dd, opened=[pos], closed=[])
        settlement._NEXT_CHECK.clear()
        counts = settlement.settle_stuck_positions(
            dd, poly_client=_FakePoly({"PMX": None}),
            kalshi_client=None, as_of_ts=AS_OF, grace_hours=24.0)
        assert counts["written_off"] == 1
        row = pp.load_positions(dd)[0]
        assert row["status"] == "closed_settlement_writeoff"
        assert row["pnl_usd"] == pytest.approx(-200.0)


def test_liquidate_trader_closes_everything():
    with tempfile.TemporaryDirectory() as tmp:
        dd = Path(tmp)
        _state_file(dd, [{"trader": "volwt", "cash": 9600.0,
                          "exposure": 400.0}])
        p1 = _open_pos("volwt", "polymarket", "PMA")
        p2 = _open_pos("volwt", "polymarket", "PMB",
                       resolution_ts="2026-12-31T00:00:00+00:00")
        pp.upsert_positions(dd, opened=[p1, p2], closed=[])
        result = settlement.liquidate_trader(
            dd, "volwt", poly_client=_FakePoly({"PMA": 1}),
            as_of_ts=AS_OF)
        assert result["closed"] == 2
        rows = {r["market_id"]: r for r in pp.load_positions(dd)}
        assert rows["PMA"]["status"] == "closed_settlement"   # outcome known
        assert rows["PMB"]["status"] == "closed_retired"      # no book → 0
        states = sp.load_state(dd)
        assert states["volwt"].open_exposure == pytest.approx(0.0)


# ── Calibration ──────────────────────────────────────────────────────────

def test_prior_only_ev_is_exactly_zero():
    """The martingale prior with design payoffs is a fair game — Kelly's
    cold-start refusal to bet falls out of the math, not a special case."""
    for c in (0.31, 0.35, 0.39):
        bs = calibration.prior_only(c=c, target=0.60, stop=0.20)
        assert bs.ev_per_usd == pytest.approx(0.0, abs=1e-12)


def _closed_trade(entry: float, pnl: float, *, category="sports",
                  trader="equal", exit_ts="2026-07-01T00:00:00+00:00") -> dict:
    size = 200.0
    return {"trader": trader, "status": "closed_x", "category": category,
            "entry_price": entry, "entry_size_usd": size, "pnl_usd": pnl,
            "exit_ts": exit_ts}


def test_bet_stats_updates_from_evidence():
    # 30 wins, 10 losses at entry ~0.35 → posterior p above prior.
    trades = ([_closed_trade(0.35, +120.0) for _ in range(30)]
              + [_closed_trade(0.35, -160.0) for _ in range(10)])
    bs = calibration.bet_stats(trades, c=0.35, target=0.60, stop=0.20,
                               category="sports", as_of_ts=AS_OF)
    assert bs.n_evidence == 40
    assert bs.p_hat > bs.p0
    assert bs.ev_per_usd > 0
    # Loss magnitude estimate reflects realized -80% losses, above the
    # designed (c-S)/c = 43%.
    assert bs.b_hat > (0.35 - 0.20) / 0.35


def test_bet_stats_excludes_future_trades_no_lookahead():
    future = [_closed_trade(0.35, +120.0,
                            exit_ts="2026-07-10T00:00:00+00:00")
              for _ in range(30)]
    bs = calibration.bet_stats(future, c=0.35, target=0.60, stop=0.20,
                               category="sports", as_of_ts=AS_OF)
    assert bs.n_evidence == 0
    assert bs.ev_per_usd == pytest.approx(0.0, abs=1e-12)


def test_bet_stats_pools_categories_when_sparse():
    trades = [_closed_trade(0.35, +120.0, category="crypto")
              for _ in range(15)]
    bs = calibration.bet_stats(trades, c=0.35, target=0.60, stop=0.20,
                               category="sports", as_of_ts=AS_OF)
    assert bs.n_evidence == 15
    assert not bs.category_matched


# ── Volharvest horizon gate ──────────────────────────────────────────────

def _vh_winner(hold_hours: float) -> dict:
    entry = "2026-07-01T00:00:00+00:00"
    exit_dt = pd.Timestamp(entry) + pd.Timedelta(hours=hold_hours)
    return {"trader": "volharvest", "exit_reason": "early_exit",
            "pnl_usd": 50.0, "entry_ts": entry,
            "exit_ts": exit_dt.isoformat()}


def test_adaptive_horizon_cold_start_and_adaptation():
    rule = volharvest.VolHarvestRule()
    # Cold start: fewer winners than horizon_min_evidence.
    h = volharvest.adaptive_max_horizon_h([_vh_winner(1.0)], rule,
                                          as_of_ts=AS_OF)
    assert h == rule.horizon_cold_start_h
    # 20 winners, p90 hold ≈ 10h → 10 * multiplier, floored at 48h.
    winners = [_vh_winner(hh) for hh in list(range(1, 10)) * 2 + [10.0, 10.0]]
    h2 = volharvest.adaptive_max_horizon_h(winners, rule, as_of_ts=AS_OF)
    assert rule.horizon_floor_h <= h2 <= rule.horizon_cap_h
    # Explicit override wins.
    rule_fixed = volharvest.VolHarvestRule(max_horizon_h=72.0)
    assert volharvest.adaptive_max_horizon_h(winners, rule_fixed,
                                             as_of_ts=AS_OF) == 72.0


def test_should_open_dog_rejects_missing_and_far_resolution():
    rule = volharvest.VolHarvestRule()
    book = BookTop(snapshot_ts=AS_OF, best_bid=0.18, best_ask=0.20,
                   bid_depth=100, ask_depth=100)
    # Missing resolution_ts → reject (zombie prevention).
    assert not volharvest.should_open_dog(book, rule, as_of_ts=AS_OF,
                                          resolution_ts=None,
                                          max_horizon_h=168.0)
    # Resolves in 30 days, horizon 7 days → reject.
    assert not volharvest.should_open_dog(
        book, rule, as_of_ts=AS_OF,
        resolution_ts="2026-08-03T12:00:00+00:00", max_horizon_h=168.0)
    # Resolves in 2 days → accept.
    assert volharvest.should_open_dog(
        book, rule, as_of_ts=AS_OF,
        resolution_ts="2026-07-06T12:00:00+00:00", max_horizon_h=168.0)
