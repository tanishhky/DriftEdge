"""Volatility-harvest agent (cross-side hedge).

Strategy (binary prediction market, two outcomes):

  Leg 1 (dog):
      Buy YES of the underdog when yes_ask is in [dog_low, dog_high].
      Treat the YES as a long claim on outcome A.

  Leg 2 (hedge):
      Once the dog leg is open, watch for a moment when the favourite's price
      drops. Specifically, the synthetic NO ask
      (`no_ask_synthetic = 1 - yes_bid`) drops enough that

          locked_profit_per_share = 1 - dog_entry - no_ask_synthetic
                                  >= min_locked_profit

      When the trigger fires, buy NO (synthetic) for the SAME number of shares
      as the dog leg. Now the position is balanced and pays $1/share regardless
      of outcome. Locked profit per share is exactly the inequality above.

  Exit:
      Both legs open → hold to resolution; payoff is deterministic.
      Dog-only at resolution → standard binary outcome.
      Both rules respect the same force-exit window as the standard tick.

Lookahead discipline (per ADR 0004):
  Every read of a book is filtered by `snapshot_ts <= as_of_ts`. Decisions
  are stamped with `as_of_ts`, not `datetime.now()`. An assertion guards the
  book read at the boundary.

Sizing:
  Dog leg uses the same per-position cap as the other agents (2 % of bankroll).
  Hedge leg is sized to MATCH the dog leg's SHARE count, not its dollar size,
  so the payoff matrix is symmetric. Dollar size = dog_shares * no_ask_synth.
  Hedge respects the aggregate-exposure cap; if it would exceed it, the
  hedge is skipped this tick and re-tried next tick.

This is NOT arbitrage. It's a "long underdog + free option to hedge"
structure whose EV comes from intra-market path variance (see
`docs/decisions/0005-volharvest-strategy.md` once written).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .. import obs
from ..paper import latest_book_top, BookTop


TRADER_ID = "volharvest"


@dataclass(frozen=True)
class VolHarvestRule:
    """All knobs for the volatility-harvest agent.

    `exit_mode`:
      'early_exit' (default, new) — when the dog leg is in profit by at least
          `min_locked_profit`, SELL the dog at the YES bid. This realises the
          same P&L as opening a hedge leg (binary-market parity:
          yes_bid + no_ask = 1 ⇒ yes_bid - dog_entry = 1 - dog_entry - no_ask)
          but doesn't tie up additional capital. Better in nearly every
          scenario.
      'hedge' (legacy) — when the synthetic NO ask drops far enough,
          BUY the NO leg to lock in payoff at resolution. Originally
          designed; preserved for A/B comparison.
    """
    dog_low: float = 0.10
    dog_high: float = 0.30
    min_locked_profit: float = 0.05      # cents per share locked at exit/hedge
    force_exit_hours_before_resolution: float = 6.0
    per_position_cap_pct: float = 0.02   # of bankroll for the DOG leg
    aggregate_cap_pct: float = 0.50      # of bankroll across all open exposure
    min_position_usd: float = 5.00
    exit_mode: str = "early_exit"        # 'early_exit' | 'hedge'


# ── Decision predicates ──────────────────────────────────────────────────

def should_open_dog(book: BookTop, rule: VolHarvestRule, *,
                    as_of_ts: str,
                    resolution_ts: Optional[str]) -> bool:
    """Open the dog leg when the YES ask is in the underdog window AND
    we're not already in the force-exit window."""
    if not (rule.dog_low <= book.best_ask <= rule.dog_high):
        return False
    if resolution_ts:
        try:
            t_now = datetime.fromisoformat(as_of_ts.replace("Z", "+00:00"))
            t_res = datetime.fromisoformat(resolution_ts.replace("Z", "+00:00"))
            hours_left = (t_res - t_now).total_seconds() / 3600.0
            if hours_left < rule.force_exit_hours_before_resolution:
                return False
        except (ValueError, TypeError):
            pass
    return True


def hedge_trigger(book: BookTop, dog_entry: float,
                  rule: VolHarvestRule) -> Optional[float]:
    """Return synthetic NO ask if the hedge trigger fires, else None.

    Synthetic NO ask = 1 - yes_bid (price you'd pay to buy NO via selling
    YES at the bid). The hedge fires only if the locked profit per share
    after both legs >= rule.min_locked_profit.
    """
    no_ask_synth = 1.0 - book.best_bid
    if no_ask_synth <= 0 or no_ask_synth >= 1:
        return None
    locked = 1.0 - dog_entry - no_ask_synth
    if locked >= rule.min_locked_profit:
        return no_ask_synth
    return None


def should_force_close(book: BookTop, rule: VolHarvestRule, *,
                       as_of_ts: str,
                       resolution_ts: Optional[str]) -> bool:
    """Time-based force exit, identical semantics to paper.should_close."""
    if not resolution_ts:
        return False
    try:
        t_now = datetime.fromisoformat(as_of_ts.replace("Z", "+00:00"))
        t_res = datetime.fromisoformat(resolution_ts.replace("Z", "+00:00"))
        hours_left = (t_res - t_now).total_seconds() / 3600.0
        return hours_left < rule.force_exit_hours_before_resolution
    except (ValueError, TypeError):
        return False


# ── Position constructors (pure) ──────────────────────────────────────────

def _open_dog(market: dict, book: BookTop, as_of_ts: str, *,
              size_usd: float, venue: str) -> dict:
    assert book.snapshot_ts <= as_of_ts, "lookahead in volharvest._open_dog"
    return {
        "trade_id": str(uuid.uuid4()),
        "trader": TRADER_ID,
        "venue": venue,
        "market_id": str(market.get("market_id", "")),
        "question": market.get("question", ""),
        "category": market.get("category") or "other",
        "yes_token_id": market.get("yes_token_id"),
        "no_token_id": market.get("no_token_id"),
        "leg": "dog",
        "side": "yes",
        "entry_ts": as_of_ts,
        "entry_snapshot_ts": book.snapshot_ts,
        "entry_price": book.best_ask,
        "entry_size_usd": size_usd,
        "shares": size_usd / book.best_ask if book.best_ask > 0 else 0.0,
        # target/stop/exit_reason are not used by volharvest; carried for
        # schema compatibility with paper_trades.parquet.
        "target": None,
        "stop": None,
        "status": "open",
        "exit_ts": None,
        "exit_snapshot_ts": None,
        "exit_price": None,
        "exit_reason": None,
        "pnl_per_share": None,
        "pnl_usd": None,
    }


def _open_hedge(market: dict, book: BookTop, as_of_ts: str, *,
                no_price: float, shares: float, venue: str,
                dog_trade_id: str) -> dict:
    assert book.snapshot_ts <= as_of_ts, "lookahead in volharvest._open_hedge"
    size_usd = shares * no_price
    return {
        "trade_id": str(uuid.uuid4()),
        "trader": TRADER_ID,
        "venue": venue,
        "market_id": str(market.get("market_id", "")),
        "question": market.get("question", ""),
        "category": market.get("category") or "other",
        "yes_token_id": market.get("yes_token_id"),
        "no_token_id": market.get("no_token_id"),
        "leg": "hedge",
        "side": "no",
        "linked_trade_id": dog_trade_id,
        "entry_ts": as_of_ts,
        "entry_snapshot_ts": book.snapshot_ts,
        "entry_price": no_price,      # synthetic NO ask (1 - yes_bid)
        "entry_size_usd": size_usd,
        "shares": shares,
        "target": None,
        "stop": None,
        "status": "open",
        "exit_ts": None,
        "exit_snapshot_ts": None,
        "exit_price": None,
        "exit_reason": None,
        "pnl_per_share": None,
        "pnl_usd": None,
    }


def _close_dog_only(pos: dict, book: BookTop, as_of_ts: str,
                    reason: str) -> dict:
    """Close a dog-only leg at the YES bid (sell). Pure."""
    assert book.snapshot_ts <= as_of_ts, "lookahead in volharvest._close_dog_only"
    closed = dict(pos)
    exit_price = book.best_bid
    pnl_per_share = exit_price - pos["entry_price"]
    pnl_usd = pnl_per_share * pos.get("shares", 0.0)
    closed.update({
        "exit_ts": as_of_ts,
        "exit_snapshot_ts": book.snapshot_ts,
        "exit_price": exit_price,
        "exit_reason": reason,
        "status": f"closed_{reason}",
        "pnl_per_share": pnl_per_share,
        "pnl_usd": pnl_usd,
    })
    return closed


def _close_hedge(pos: dict, book: BookTop, as_of_ts: str,
                  reason: str) -> dict:
    """Close a hedge leg at the synthetic NO bid.

    no_bid_synthetic = 1 - yes_ask (sell NO by buying YES at the ask).
    """
    assert book.snapshot_ts <= as_of_ts, "lookahead in volharvest._close_hedge"
    closed = dict(pos)
    no_bid_synth = max(0.0, 1.0 - book.best_ask)
    pnl_per_share = no_bid_synth - pos["entry_price"]
    pnl_usd = pnl_per_share * pos.get("shares", 0.0)
    closed.update({
        "exit_ts": as_of_ts,
        "exit_snapshot_ts": book.snapshot_ts,
        "exit_price": no_bid_synth,
        "exit_reason": reason,
        "status": f"closed_{reason}",
        "pnl_per_share": pnl_per_share,
        "pnl_usd": pnl_usd,
    })
    return closed


# ── Sizing ────────────────────────────────────────────────────────────────

def _dog_size(bankroll_init: float, cash_usd: float, open_exposure: float,
              rule: VolHarvestRule) -> float:
    per_cap = bankroll_init * rule.per_position_cap_pct
    agg_cap = bankroll_init * rule.aggregate_cap_pct
    available = max(0.0, agg_cap - open_exposure)
    size = min(per_cap, available, cash_usd)
    return size if size >= rule.min_position_usd else 0.0


def _hedge_size_ok(shares: float, no_price: float,
                   bankroll_init: float, cash_usd: float,
                   open_exposure: float,
                   rule: VolHarvestRule) -> tuple[bool, float]:
    """Hedge size must equal dog SHARES; check the resulting $ fits within
    the aggregate cap and available cash. Returns (ok, size_usd)."""
    size_usd = shares * no_price
    agg_cap = bankroll_init * rule.aggregate_cap_pct
    available = max(0.0, agg_cap - open_exposure)
    if size_usd > available or size_usd > cash_usd:
        return False, size_usd
    if size_usd < rule.min_position_usd:
        return False, size_usd
    return True, size_usd


# ── Tick driver ───────────────────────────────────────────────────────────

def tick(data_dir: Path, markets: list[dict],
         rule: Optional[VolHarvestRule] = None,
         as_of_ts: Optional[str] = None,
         bankroll: float = 10000.0) -> dict:
    """Run one tick of the volharvest agent.

    Called by the daemon after `paper.tick`. Reads `paper_trades.parquet`,
    writes opens/closes back via `paper_persist.upsert_positions`. Updates
    `paper_state.parquet` and appends to `equity_history.parquet` only for
    its own trader_id; the other agents' rows are left untouched.
    """
    if rule is None:
        rule = VolHarvestRule()
    if as_of_ts is None:
        as_of_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    from ..data import paper_persist as pp
    from ..data import state_persist as sp
    from ..data import equity_persist as ep

    states = sp.init_state(data_dir, bankroll=bankroll)
    if TRADER_ID not in states:
        # Should never happen — state_persist seeds all_trader_labels(). But
        # be defensive.
        obs.event(channel="error", kind="volharvest.no_state",
                  level="WARNING", trader=TRADER_ID)
        return {"as_of_ts": as_of_ts, "skipped": True}

    state = states[TRADER_ID]
    cash_usd = float(state.cash_usd)
    bankroll_init = float(state.bankroll_init)
    open_exposure = float(state.open_exposure)
    closed_pnl = float(state.closed_pnl)

    positions = pp.load_positions(data_dir)
    own_open = [p for p in positions
                if p.get("trader") == TRADER_ID and p.get("status") == "open"]

    # Index by (venue, market_id) → {leg → position}
    own_by_market: dict[tuple[str, str], dict[str, dict]] = {}
    for p in own_open:
        key = (p.get("venue", "polymarket"), p.get("market_id", ""))
        leg = p.get("leg") or "dog"
        own_by_market.setdefault(key, {})[leg] = p

    opened: list[dict] = []
    closed: list[dict] = []
    book_mids: dict[tuple[str, str], float] = {}
    actions: dict[str, int] = {"open_dog": 0, "open_hedge": 0,
                                "close_resolution": 0, "close_dogonly": 0}

    for m in markets:
        mid_id = str(m.get("market_id") or "")
        if not mid_id:
            continue
        venue = m.get("venue") or "polymarket"
        books_dir = data_dir / "books" / venue
        book = latest_book_top(books_dir, mid_id, as_of_ts=as_of_ts)
        if book is None:
            continue
        book_mids[(venue, mid_id)] = book.mid

        held = own_by_market.get((venue, mid_id), {})
        resolution_ts = m.get("end_date")

        # ── EXIT side first ──
        # Force-exit window. If we hold any leg here, close it (and any sibling
        # leg). Hedge close uses synthetic NO bid; dog close uses YES bid.
        if held and should_force_close(book, rule, as_of_ts=as_of_ts,
                                        resolution_ts=resolution_ts):
            dog_pos = held.get("dog")
            hedge_pos = held.get("hedge")
            if dog_pos:
                closed_pos = _close_dog_only(dog_pos, book, as_of_ts,
                                              reason="time")
                closed.append(closed_pos)
                cash_usd += float(dog_pos["entry_size_usd"]) + float(closed_pos["pnl_usd"])
                open_exposure -= float(dog_pos["entry_size_usd"])
                closed_pnl += float(closed_pos["pnl_usd"])
            if hedge_pos:
                closed_pos = _close_hedge(hedge_pos, book, as_of_ts,
                                           reason="time")
                closed.append(closed_pos)
                cash_usd += float(hedge_pos["entry_size_usd"]) + float(closed_pos["pnl_usd"])
                open_exposure -= float(hedge_pos["entry_size_usd"])
                closed_pnl += float(closed_pos["pnl_usd"])
            if dog_pos and hedge_pos:
                actions["close_resolution"] += 1
            elif dog_pos:
                actions["close_dogonly"] += 1
            # Clear so the entry side doesn't reopen this tick.
            own_by_market.pop((venue, mid_id), None)
            held = {}

        # ── PROFIT-TAKE / HEDGE side (only if dog held alone) ──
        if held.get("dog") and not held.get("hedge"):
            dog_pos = held["dog"]
            dog_entry = float(dog_pos["entry_price"])
            no_price = hedge_trigger(book, dog_entry, rule)
            if no_price is not None:
                if rule.exit_mode == "early_exit":
                    # Capital-efficient path: sell the dog at the YES bid.
                    # Realised P&L per share = yes_bid − dog_entry, which
                    # equals 1 − dog_entry − no_ask by binary parity.
                    closed_pos = _close_dog_only(dog_pos, book, as_of_ts,
                                                  reason="early_exit")
                    closed.append(closed_pos)
                    cash_usd += float(dog_pos["entry_size_usd"]) + float(closed_pos["pnl_usd"])
                    open_exposure -= float(dog_pos["entry_size_usd"])
                    closed_pnl += float(closed_pos["pnl_usd"])
                    actions["close_dogonly"] += 1
                    own_by_market.pop((venue, mid_id), None)
                    held = {}
                else:
                    # Legacy hedge path.
                    shares = float(dog_pos["shares"])
                    ok, size_usd = _hedge_size_ok(shares, no_price,
                                                   bankroll_init, cash_usd,
                                                   open_exposure, rule)
                    if ok:
                        hedge_pos = _open_hedge(
                            m, book, as_of_ts, no_price=no_price,
                            shares=shares, venue=venue,
                            dog_trade_id=dog_pos["trade_id"],
                        )
                        opened.append(hedge_pos)
                        cash_usd -= size_usd
                        open_exposure += size_usd
                        actions["open_hedge"] += 1
                        own_by_market[(venue, mid_id)]["hedge"] = hedge_pos

        # ── ENTRY side (only if nothing held on this market) ──
        if not own_by_market.get((venue, mid_id)):
            if should_open_dog(book, rule, as_of_ts=as_of_ts,
                                resolution_ts=resolution_ts):
                size_usd = _dog_size(bankroll_init, cash_usd, open_exposure,
                                      rule)
                if size_usd > 0:
                    dog_pos = _open_dog(m, book, as_of_ts,
                                         size_usd=size_usd, venue=venue)
                    opened.append(dog_pos)
                    cash_usd -= size_usd
                    open_exposure += size_usd
                    actions["open_dog"] += 1
                    own_by_market[(venue, mid_id)] = {"dog": dog_pos}

    # Persist if anything changed.
    if opened or closed:
        pp.upsert_positions(data_dir, opened=opened, closed=closed)

    # Update state.
    from ..sizing import TraderState
    new_state = TraderState(
        trader=TRADER_ID,
        bankroll_init=bankroll_init,
        cash_usd=cash_usd,
        open_exposure=open_exposure,
        closed_pnl=closed_pnl,
    )
    states[TRADER_ID] = new_state
    sp.save_state(data_dir, states)

    # Equity snapshot — append a row for THIS trader only, MTM included.
    own_open_now = [p for slot in own_by_market.values() for p in slot.values()]
    peaks = ep.latest_peaks(data_dir)
    snaps = ep.build_snapshot(
        ts=as_of_ts,
        states={TRADER_ID: new_state},
        open_positions=own_open_now,
        book_mids=book_mids,
        peak_by_trader=peaks,
    )
    if snaps:
        ep.append_snapshot(data_dir, snaps)

    obs.event(channel="fit", kind="volharvest.tick", level="INFO",
              as_of_ts=as_of_ts, markets_seen=len(markets),
              **actions, equity_rows=len(snaps))

    return {"as_of_ts": as_of_ts, **actions}
