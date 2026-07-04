"""Resolution settlement for open positions stuck past market resolution.

Why this exists (2026-07-04): the normal exit path (target / stop / time
force-exit) needs a FRESH orderbook to price the exit, and `should_close` /
`should_force_close` deliberately refuse to act on a book captured before
resolution (the stale-bid guard). But once a market resolves, its orderbook
is deleted by the venue (the /book 404s) so a fresh book never arrives.
Any position whose market resolved while the daemon was down (laptop sleep)
therefore stayed open FOREVER: marked at a stale pre-resolution mid,
consuming aggregate-exposure cap, and silently overstating equity. On
2026-07-04 there were 26 such zombies pinning ~$5.2k across traders.

The fix: ask the venue what actually happened.

  * Polymarket Gamma `GET /markets/{id}` -> closed + outcomePrices
    (aligned with the outcomes array; our YES side is index 0, the same
    convention normalize._tokens uses for yes_token_id).
  * Kalshi `GET /markets/{ticker}` -> status in {finalized, settled} +
    result in {yes, no}.

A YES position settles at the resolved outcome price o (0 or 1):
    pnl_per_share = o - entry_price
A NO position (volharvest legacy hedge leg) settles at 1 - o.

Fallback ladder when the venue API cannot give an outcome after
`grace_hours` past resolution: close at the last known book bid
(exit_reason `settlement_stale`), else write the position off at 0
(exit_reason `settlement_writeoff`) - conservative for long positions,
and loudly logged either way.

No-lookahead note: settlement uses REALIZED resolution data strictly after
`resolution_ts <= as_of_ts`. It never influences entries and never touches
positions whose market has not resolved yet, so the as_of discipline of the
decision paths is preserved.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import obs
from .paper import latest_book_top, parse_iso


# Markets the venue reported as not-yet-resolved: don't re-ask every sweep.
# {(venue, market_id): unix_epoch_of_next_allowed_check}
_NEXT_CHECK: dict[tuple[str, str], float] = {}


# ── Venue outcome lookups ─────────────────────────────────────────────────

def yes_settlement_polymarket(client, market_id: str) -> Optional[float]:
    """Resolved YES price (0.0/1.0) via Gamma, or None if not resolved."""
    try:
        m = client.get_market(market_id)
    except Exception as exc:
        obs.event(channel="error", kind="settle.lookup_fail", level="WARNING",
                  venue="polymarket", market_id=market_id, err=str(exc))
        return None
    if not m or not m.get("closed"):
        return None
    raw = m.get("outcomePrices")
    try:
        prices = json.loads(raw) if isinstance(raw, str) else raw
        yes_price = float(prices[0])
    except (TypeError, ValueError, IndexError):
        return None
    # Only trust unambiguous settlements; mid-flight prices like 0.5 mean
    # the UMA resolution is still pending even though trading closed.
    if yes_price not in (0.0, 1.0):
        status = str(m.get("umaResolutionStatus") or "")
        if status != "resolved":
            return None
    return yes_price


def yes_settlement_kalshi(client, ticker: str) -> Optional[float]:
    """Resolved YES price (0.0/1.0) via Kalshi, or None if not settled."""
    try:
        payload = client.get_market(ticker)
    except Exception as exc:
        obs.event(channel="error", kind="settle.lookup_fail", level="WARNING",
                  venue="kalshi", market_id=ticker, err=str(exc))
        return None
    m = payload.get("market") if isinstance(payload, dict) else None
    if not m:
        return None
    if m.get("status") not in ("finalized", "settled"):
        return None
    result = str(m.get("result") or "").lower()
    if result == "yes":
        return 1.0
    if result == "no":
        return 0.0
    return None


def _yes_settlement(venue: str, market_id: str, *,
                    poly_client=None, kalshi_client=None) -> Optional[float]:
    if venue == "polymarket" and poly_client is not None:
        return yes_settlement_polymarket(poly_client, market_id)
    if venue == "kalshi" and kalshi_client is not None:
        return yes_settlement_kalshi(kalshi_client, market_id)
    return None


# ── Close constructors (pure) ─────────────────────────────────────────────

def _close_at(pos: dict, *, exit_price: float, reason: str,
              as_of_ts: str) -> dict:
    """Close a position at an externally determined price. Pure."""
    side = str(pos.get("side") or "yes").lower()
    payoff = exit_price if side != "no" else 1.0 - exit_price
    pnl_per_share = payoff - float(pos.get("entry_price") or 0.0)
    pnl_usd = pnl_per_share * float(pos.get("shares") or 0.0)
    closed = dict(pos)
    closed.update({
        "exit_ts": as_of_ts,
        "exit_snapshot_ts": None,
        "exit_price": payoff,
        "exit_reason": reason,
        "status": f"closed_{reason}",
        "pnl_per_share": pnl_per_share,
        "pnl_usd": pnl_usd,
    })
    return closed


# ── Sweeps ────────────────────────────────────────────────────────────────

def settle_stuck_positions(data_dir: Path, *,
                           poly_client=None, kalshi_client=None,
                           as_of_ts: Optional[str] = None,
                           grace_hours: float = 24.0,
                           recheck_interval_s: float = 1800.0) -> dict:
    """Settle every open position whose market resolved before `as_of_ts`.

    Runs on the market-refresh cadence in the poll loop; also exposed as
    `driftedge settle` for one-off sweeps. One venue-API call per stuck
    MARKET (positions share the lookup), with a per-market backoff for
    markets the venue still reports unresolved.
    """
    if as_of_ts is None:
        as_of_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    as_of_dt = parse_iso(as_of_ts)

    from .data import paper_persist as pp

    positions = pp.load_positions(data_dir)
    stuck: dict[tuple[str, str], list[dict]] = {}
    for p in positions:
        if p.get("status") != "open":
            continue
        res = p.get("resolution_ts")
        if not res:
            continue
        try:
            if parse_iso(str(res)) >= as_of_dt:
                continue
        except (ValueError, TypeError):
            continue
        key = (p.get("venue") or "polymarket", str(p.get("market_id") or ""))
        stuck.setdefault(key, []).append(p)

    if not stuck:
        return {"checked": 0, "settled": 0, "stale_closed": 0, "written_off": 0}

    closed: list[dict] = []
    counts = {"checked": 0, "settled": 0, "stale_closed": 0, "written_off": 0,
              "pending": 0}
    now_epoch = time.time()

    for (venue, mid), plist in stuck.items():
        if _NEXT_CHECK.get((venue, mid), 0.0) > now_epoch:
            counts["pending"] += len(plist)
            continue
        counts["checked"] += 1
        outcome = _yes_settlement(venue, mid,
                                  poly_client=poly_client,
                                  kalshi_client=kalshi_client)
        if outcome is not None:
            for p in plist:
                closed.append(_close_at(p, exit_price=outcome,
                                        reason="settlement",
                                        as_of_ts=as_of_ts))
            counts["settled"] += len(plist)
            _NEXT_CHECK.pop((venue, mid), None)
            continue

        # Venue can't (yet) tell us. Inside the grace window: back off and
        # retry later. Past it: close at the last known book, else at 0.
        res_dt = parse_iso(str(plist[0].get("resolution_ts")))
        hours_past = (as_of_dt - res_dt).total_seconds() / 3600.0
        if hours_past <= grace_hours:
            _NEXT_CHECK[(venue, mid)] = now_epoch + recheck_interval_s
            counts["pending"] += len(plist)
            continue

        book = latest_book_top(data_dir / "books" / venue, mid,
                               as_of_ts=as_of_ts)
        if book is not None and book.best_bid == book.best_bid:  # not NaN
            for p in plist:
                closed.append(_close_at(p, exit_price=float(book.best_bid),
                                        reason="settlement_stale",
                                        as_of_ts=as_of_ts))
            counts["stale_closed"] += len(plist)
        else:
            for p in plist:
                closed.append(_close_at(p, exit_price=0.0,
                                        reason="settlement_writeoff",
                                        as_of_ts=as_of_ts))
            counts["written_off"] += len(plist)
        _NEXT_CHECK.pop((venue, mid), None)

    if closed:
        _apply_closes(data_dir, closed)

    obs.event(channel="fit", kind="settle.sweep", level="INFO",
              as_of_ts=as_of_ts, stuck_markets=len(stuck), **counts)
    return counts


def liquidate_trader(data_dir: Path, trader: str, *,
                     poly_client=None, kalshi_client=None,
                     as_of_ts: Optional[str] = None) -> dict:
    """Close ALL open positions of a trader (used when retiring one).

    Resolution outcome is used when the venue has one; otherwise the last
    known book bid; otherwise 0. History rows are preserved - only status
    and exit fields change.
    """
    if as_of_ts is None:
        as_of_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    from .data import paper_persist as pp

    positions = pp.load_positions(data_dir)
    own = [p for p in positions
           if p.get("trader") == trader and p.get("status") == "open"]
    closed: list[dict] = []
    for p in own:
        venue = p.get("venue") or "polymarket"
        mid = str(p.get("market_id") or "")
        outcome = _yes_settlement(venue, mid, poly_client=poly_client,
                                  kalshi_client=kalshi_client)
        if outcome is not None:
            closed.append(_close_at(p, exit_price=outcome,
                                    reason="settlement", as_of_ts=as_of_ts))
            continue
        book = latest_book_top(data_dir / "books" / venue, mid,
                               as_of_ts=as_of_ts)
        if book is not None and book.best_bid == book.best_bid:
            closed.append(_close_at(p, exit_price=float(book.best_bid),
                                    reason="retired", as_of_ts=as_of_ts))
        else:
            closed.append(_close_at(p, exit_price=0.0,
                                    reason="retired", as_of_ts=as_of_ts))

    if closed:
        _apply_closes(data_dir, closed)
    obs.event(channel="fit", kind="settle.liquidate_trader", level="INFO",
              trader=trader, closed=len(closed), as_of_ts=as_of_ts)
    return {"trader": trader, "closed": len(closed)}


def _apply_closes(data_dir: Path, closed: list[dict]) -> None:
    """Persist closes + roll the affected traders' cash/exposure/pnl."""
    from .data import paper_persist as pp
    from .data import state_persist as sp

    pp.upsert_positions(data_dir, opened=[], closed=closed)
    states = sp.load_state(data_dir)
    touched = False
    for c in closed:
        tid = c.get("trader")
        if tid not in states:
            obs.event(channel="error", kind="settle.unknown_trader",
                      level="WARNING", trader=tid)
            continue
        states[tid] = sp.apply_close(states[tid],
                                     float(c.get("entry_size_usd") or 0.0),
                                     float(c.get("pnl_usd") or 0.0))
        touched = True
        obs.event(channel="fit", kind="settle.close", level="INFO",
                  trader=tid, venue=c.get("venue"),
                  market_id=c.get("market_id"),
                  reason=c.get("exit_reason"),
                  exit_price=c.get("exit_price"),
                  pnl_usd=round(float(c.get("pnl_usd") or 0.0), 2))
    if touched:
        sp.save_state(data_dir, states)
