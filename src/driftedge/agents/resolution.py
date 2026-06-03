"""Resolution agent — hold-to-binary prediction market trader.

Strategy:
  Enter YES on markets resolving within MAX_HORIZON_H (72 h) when the ask
  price is in [entry_low, entry_high] = [0.25, 0.50].

  Unlike the other agents we do NOT take soft profits or momentum stops.
  We wait for the market to resolve (price → 1.0 on YES or → 0.0 on NO).

  Time-weighted sizing: position size scales mildly upward as resolution
  approaches (1.0× at 72 h → 1.5× at <1 h remaining).

  Dynamic stop: if within NEAR_RESOLUTION_H AND the YES bid has fallen
  LOSS_THRESHOLD below entry, we liquidate — it's better to take a clean
  loss than ride to zero.

  Force exit: 1 h before resolution (vs 6 h for standard traders) so we
  avoid holding a stale position through the resolution event itself.

Lookahead discipline (per ADR 0004):
  All book reads filtered by snapshot_ts <= as_of_ts.
  Decisions stamped with as_of_ts, never datetime.now().
  Assertions guard every book boundary.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from .. import obs
from ..paper import latest_book_top, BookTop


TRADER_ID = "resolution"


@dataclass(frozen=True)
class ResolutionRule:
    entry_low: float = 0.25             # YES ask entry zone lower bound
    entry_high: float = 0.50            # YES ask entry zone upper bound
    max_horizon_h: float = 72.0         # skip markets resolving > 72 h away
    near_resolution_h: float = 6.0      # "close to resolution" for dynamic stop
    loss_threshold: float = 0.15        # liquidate if bid is ≥ this far below entry at near-resolution
    force_exit_h: float = 1.0           # force exit 1 h before resolution
    per_position_cap_pct: float = 0.02  # of bankroll
    aggregate_cap_pct: float = 0.50     # of bankroll across all open exposure
    min_position_usd: float = 5.00


# ── Helpers ───────────────────────────────────────────────────────────────

def _hours_to(resolution_ts: Optional[str], as_of_ts: str) -> Optional[float]:
    if not resolution_ts:
        return None
    try:
        t_now = datetime.fromisoformat(as_of_ts.replace("Z", "+00:00"))
        t_res = datetime.fromisoformat(str(resolution_ts).replace("Z", "+00:00"))
        return (t_res - t_now).total_seconds() / 3600.0
    except (ValueError, TypeError):
        return None


# ── Decision predicates ───────────────────────────────────────────────────

def should_open(book: BookTop, rule: ResolutionRule, *,
                as_of_ts: str, resolution_ts: Optional[str]) -> Tuple[bool, Optional[float]]:
    """Open if ask is in entry zone AND within max_horizon AND not in force-exit window."""
    if not (rule.entry_low <= book.best_ask <= rule.entry_high):
        return False, None
    hours = _hours_to(resolution_ts, as_of_ts)
    if hours is None:
        return False, None          # unknown resolution date — skip
    if hours > rule.max_horizon_h:
        return False, None          # too far out — high opp cost
    if hours < rule.force_exit_h:
        return False, None          # already in force-exit window
    return True, hours


def should_dynamic_stop(book: BookTop, entry_price: float,
                        rule: ResolutionRule, *,
                        as_of_ts: str, resolution_ts: Optional[str]) -> bool:
    """True if we're near resolution AND the bid has fallen enough below entry."""
    hours = _hours_to(resolution_ts, as_of_ts)
    if hours is None or hours > rule.near_resolution_h:
        return False
    return (book.best_bid - entry_price) <= -rule.loss_threshold


def should_force_close(rule: ResolutionRule, *,
                       as_of_ts: str, resolution_ts: Optional[str]) -> bool:
    hours = _hours_to(resolution_ts, as_of_ts)
    if hours is None:
        return False
    return hours < rule.force_exit_h


# ── Position constructors (pure) ──────────────────────────────────────────

def _open_position(market: dict, book: BookTop, as_of_ts: str, *,
                   size_usd: float, venue: str) -> dict:
    assert book.snapshot_ts <= as_of_ts, "lookahead in resolution._open_position"
    return {
        "trade_id": str(uuid.uuid4()),
        "trader": TRADER_ID,
        "venue": venue,
        "market_id": str(market.get("market_id", "")),
        "question": market.get("question", ""),
        "category": market.get("category") or "other",
        "yes_token_id": market.get("yes_token_id"),
        "no_token_id": market.get("no_token_id"),
        "leg": "yes",
        "side": "yes",
        "entry_ts": as_of_ts,
        "entry_snapshot_ts": book.snapshot_ts,
        "entry_price": book.best_ask,
        "entry_size_usd": size_usd,
        "shares": size_usd / book.best_ask if book.best_ask > 0 else 0.0,
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


def _close_position(pos: dict, book: BookTop, as_of_ts: str, reason: str) -> dict:
    assert book.snapshot_ts <= as_of_ts, "lookahead in resolution._close_position"
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


# ── Sizing ────────────────────────────────────────────────────────────────

def _size(bankroll_init: float, cash_usd: float, open_exposure: float,
          rule: ResolutionRule, hours: float) -> float:
    # Scale up mildly as resolution approaches: 1.0× at 72h → 1.5× at ≈0h
    scale = 1.0 + 0.5 * max(0.0, 1.0 - hours / rule.max_horizon_h)
    scale = min(1.5, scale)
    per_cap = bankroll_init * rule.per_position_cap_pct * scale
    agg_cap = bankroll_init * rule.aggregate_cap_pct
    available = max(0.0, agg_cap - open_exposure)
    size = min(per_cap, available, cash_usd)
    return size if size >= rule.min_position_usd else 0.0


# ── Tick driver ───────────────────────────────────────────────────────────

def tick(data_dir: Path, markets: list[dict],
         rule: Optional[ResolutionRule] = None,
         as_of_ts: Optional[str] = None,
         bankroll: float = 10_000.0) -> dict:
    """Run one tick of the resolution agent.

    Called by the daemon after volharvest.tick. Self-managed: reads and
    writes only its own rows in paper_trades / paper_state / equity_history.
    """
    if rule is None:
        rule = ResolutionRule()
    if as_of_ts is None:
        as_of_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    from ..data import paper_persist as pp
    from ..data import state_persist as sp
    from ..data import equity_persist as ep

    states = sp.init_state(data_dir, bankroll=bankroll)
    if TRADER_ID not in states:
        obs.event(channel="error", kind="resolution.no_state",
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

    own_by_market: dict[str, dict] = {
        str(p.get("market_id", "")): p for p in own_open
    }

    opened: list[dict] = []
    closed: list[dict] = []
    book_mids: dict[tuple[str, str], float] = {}
    actions: dict[str, int] = {"open": 0, "close_force": 0,
                                "close_stop": 0, "skipped_horizon": 0}

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

        held = own_by_market.get(mid_id)
        resolution_ts = m.get("end_date")

        # ── EXIT: force close ──
        if held and should_force_close(rule, as_of_ts=as_of_ts,
                                        resolution_ts=resolution_ts):
            closed_pos = _close_position(held, book, as_of_ts, "time")
            closed.append(closed_pos)
            cash_usd += float(held["entry_size_usd"]) + float(closed_pos["pnl_usd"])
            open_exposure -= float(held["entry_size_usd"])
            closed_pnl += float(closed_pos["pnl_usd"])
            actions["close_force"] += 1
            own_by_market.pop(mid_id, None)
            held = None

        # ── EXIT: dynamic stop (near resolution + large loss) ──
        if held and should_dynamic_stop(book, float(held["entry_price"]),
                                         rule, as_of_ts=as_of_ts,
                                         resolution_ts=resolution_ts):
            closed_pos = _close_position(held, book, as_of_ts, "dynamic_stop")
            closed.append(closed_pos)
            cash_usd += float(held["entry_size_usd"]) + float(closed_pos["pnl_usd"])
            open_exposure -= float(held["entry_size_usd"])
            closed_pnl += float(closed_pos["pnl_usd"])
            actions["close_stop"] += 1
            own_by_market.pop(mid_id, None)
            held = None

        # ── ENTRY: open if nothing held on this market ──
        if not own_by_market.get(mid_id):
            ok, hours = should_open(book, rule, as_of_ts=as_of_ts,
                                    resolution_ts=resolution_ts)
            if not ok:
                if hours is None or (hours is not None and hours > rule.max_horizon_h):
                    actions["skipped_horizon"] += 1
            else:
                size_usd = _size(bankroll_init, cash_usd, open_exposure, rule, hours)
                if size_usd > 0:
                    pos = _open_position(m, book, as_of_ts,
                                          size_usd=size_usd, venue=venue)
                    opened.append(pos)
                    cash_usd -= size_usd
                    open_exposure += size_usd
                    actions["open"] += 1
                    own_by_market[mid_id] = pos

    if opened or closed:
        pp.upsert_positions(data_dir, opened=opened, closed=closed)

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

    own_open_now = list(own_by_market.values())

    # Book-mid fallback for off-tracking positions.
    for pos in own_open:
        v = pos.get("venue", "polymarket")
        mid_key = str(pos.get("market_id") or "")
        if not mid_key or (v, mid_key) in book_mids:
            continue
        fallback = latest_book_top(data_dir / "books" / v, mid_key,
                                   as_of_ts=as_of_ts)
        if fallback is not None:
            book_mids[(v, mid_key)] = fallback.mid

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

    obs.event(channel="fit", kind="resolution.tick", level="INFO",
              as_of_ts=as_of_ts, markets_seen=len(markets),
              **actions, equity_rows=len(snaps))

    return {"as_of_ts": as_of_ts, **actions}
