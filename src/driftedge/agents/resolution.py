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
from ..paper import latest_book_top, BookTop, parse_iso


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
        return (parse_iso(str(resolution_ts)) - parse_iso(as_of_ts)).total_seconds() / 3600.0
    except (ValueError, TypeError):
        return None


def _book_pre_resolution(book: BookTop, resolution_ts: Optional[str]) -> bool:
    """True if the book we have was captured strictly BEFORE resolution.
    Indicates the bid is a pre-settlement quote — not a safe exit price
    for force-close in the post-resolution window."""
    if not resolution_ts:
        return False
    try:
        return parse_iso(book.snapshot_ts) < parse_iso(str(resolution_ts))
    except (ValueError, TypeError):
        return False


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
    """True if we're near resolution AND the bid has fallen enough below entry.

    Stale-bid guard (2026-06-05): post-resolution with a pre-resolution
    book → defer (would trigger a stop on a stale, pre-settlement quote).
    """
    hours = _hours_to(resolution_ts, as_of_ts)
    if hours is None or hours > rule.near_resolution_h:
        return False
    if hours <= 0 and _book_pre_resolution(book, resolution_ts):
        return False
    return (book.best_bid - entry_price) <= -rule.loss_threshold


def should_force_close(rule: ResolutionRule, *,
                       as_of_ts: str, resolution_ts: Optional[str],
                       book: Optional[BookTop] = None) -> bool:
    """Time-based force exit. If book is supplied AND we're past resolution
    AND the book is pre-resolution, defer (the bid would be a stale,
    pre-settlement quote and the recorded exit would underprice the actual
    payoff)."""
    hours = _hours_to(resolution_ts, as_of_ts)
    if hours is None:
        return False
    if hours >= rule.force_exit_h:
        return False
    # Stale-bid guard for post-resolution closes.
    if hours <= 0 and book is not None and _book_pre_resolution(book, resolution_ts):
        return False
    return True


# ── Position constructors (pure) ──────────────────────────────────────────

def _open_position(market: dict, book: BookTop, as_of_ts: str, *,
                   size_usd: float, venue: str) -> dict:
    assert parse_iso(book.snapshot_ts) <= parse_iso(as_of_ts), (
        f"lookahead in resolution._open_position: snapshot={book.snapshot_ts} "
        f"as_of_ts={as_of_ts}")
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
        "resolution_ts": market.get("end_date"),
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
    assert parse_iso(book.snapshot_ts) <= parse_iso(as_of_ts), (
        f"lookahead in resolution._close_position: snapshot={book.snapshot_ts} "
        f"as_of_ts={as_of_ts}")
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

    # Key by (venue, market_id) so colliding market_ids across venues are
    # separate entries. (Bug fix 2026-06-05.)
    own_by_market: dict[tuple[str, str], dict] = {
        (p.get("venue") or "polymarket", str(p.get("market_id", ""))): p
        for p in own_open
    }

    # ── Synthetic-stub prepend for orphan positions ──
    # If a held resolution position's market isn't in `markets`, the loop
    # wouldn't visit it and force-close would never fire. Prepend a stub
    # using the position's stored resolution_ts.
    tracked_keys = {
        (m.get("venue") or "polymarket", str(m.get("market_id") or ""))
        for m in markets
    }
    orphan_stubs: list[dict] = []
    for key, pos in own_by_market.items():
        if key in tracked_keys:
            continue
        venue, mid_id = key
        orphan_stubs.append({
            "venue": venue,
            "market_id": mid_id,
            "end_date": pos.get("resolution_ts"),
            "category": pos.get("category") or "other",
            "question": pos.get("question") or "",
            "_orphan": True,
        })

    opened: list[dict] = []
    closed: list[dict] = []
    book_mids: dict[tuple[str, str], float] = {}
    actions: dict[str, int] = {"open": 0, "close_force": 0,
                                "close_stop": 0, "skipped_horizon": 0}

    for m in list(markets) + orphan_stubs:
        mid_id = str(m.get("market_id") or "")
        if not mid_id:
            continue
        venue = m.get("venue") or "polymarket"
        books_dir = data_dir / "books" / venue
        book = latest_book_top(books_dir, mid_id, as_of_ts=as_of_ts)
        if book is None:
            continue
        book_mids[(venue, mid_id)] = book.mid

        held = own_by_market.get((venue, mid_id))
        resolution_ts = m.get("end_date")

        # ── EXIT: force close (with stale-bid guard for post-resolution) ──
        if held and should_force_close(rule, as_of_ts=as_of_ts,
                                        resolution_ts=resolution_ts,
                                        book=book):
            closed_pos = _close_position(held, book, as_of_ts, "time")
            closed.append(closed_pos)
            cash_usd += float(held["entry_size_usd"]) + float(closed_pos["pnl_usd"])
            open_exposure -= float(held["entry_size_usd"])
            closed_pnl += float(closed_pos["pnl_usd"])
            actions["close_force"] += 1
            own_by_market.pop((venue, mid_id), None)
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
            own_by_market.pop((venue, mid_id), None)
            held = None

        # ── ENTRY: open if nothing held on this market ──
        # Never re-enter on an orphan stub (its book is whatever the daemon
        # last fetched, possibly stale by days).
        if m.get("_orphan"):
            continue
        if not own_by_market.get((venue, mid_id)):
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
                    own_by_market[(venue, mid_id)] = pos

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
              orphans_visited=len(orphan_stubs),
              **actions, equity_rows=len(snaps))

    return {"as_of_ts": as_of_ts,
            "orphans_visited": len(orphan_stubs), **actions}
