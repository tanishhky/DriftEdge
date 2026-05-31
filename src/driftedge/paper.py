"""Paper-trading engine for DriftEdge.

Strict no-lookahead discipline:
  * Every function that makes a trade decision takes an explicit `as_of_ts`
    parameter (ISO-8601 string, UTC).
  * Functions reading historical snapshots MUST filter to
    `snapshot_ts <= as_of_ts`.
  * Functions writing decisions MUST stamp them with `as_of_ts`, never
    `datetime.now()`.
  * Assertions enforce these rules at runtime.
  * Tests in `tests/test_paper_no_lookahead.py` prove that adding
    future-dated snapshots to the data store does not change a past
    decision.

Why this matters: in real time, this discipline is trivially satisfied
(we only have the present). But the same code paths get used for
backtesting historical data. Without the `as_of_ts` discipline, a
backtest accidentally reads forward and the reported P&L is fiction.

Entry rule (v0 minimal — no path/flow engine yet):
  open a paper-long Yes position when the market's current best ask
  is in [entry_low, entry_high]. Position size = fixed notional (no
  Kelly until we have p_estimated).

Exit rule:
  close when best bid >= target  (take profit)
  close when best ask <= stop    (stop loss)
  close when time-to-resolution < 6h  (force exit, no event variance)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from . import obs


# ── Domain types ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EntryRule:
    entry_low: float = 0.30
    entry_high: float = 0.40
    target: float = 0.60
    stop: float = 0.20
    notional_usd: float = 100.0
    force_exit_hours_before_resolution: float = 6.0


@dataclass(frozen=True)
class BookTop:
    """Top-of-book pair from a single snapshot."""
    snapshot_ts: str
    best_bid: float       # highest bid price
    best_ask: float       # lowest ask price
    bid_depth: float
    ask_depth: float

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0


# ── Lookahead-protected snapshot reader ──────────────────────────────────

def latest_book_top(books_dir: Path, market_id: str,
                    as_of_ts: str) -> Optional[BookTop]:
    """Return the most recent book snapshot for this market with
    snapshot_ts <= as_of_ts. Returns None if no eligible snapshot.

    The `as_of_ts` filter is THE no-lookahead guarantee for this function.
    """
    market_dir = books_dir / market_id
    if not market_dir.exists():
        return None

    best: Optional[BookTop] = None
    for parquet in market_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(parquet)
        except Exception as exc:
            obs.event(channel="error", kind="paper.read_fail", level="WARNING",
                      path=str(parquet), err=str(exc))
            continue
        if df.empty or "snapshot_ts" not in df.columns:
            continue
        # ── critical: filter out any snapshot newer than as_of_ts ──
        df = df[df["snapshot_ts"] <= as_of_ts]
        if df.empty:
            continue
        latest_ts = df["snapshot_ts"].max()
        snap = df[df["snapshot_ts"] == latest_ts]

        bids = snap[snap["side"] == "bid"].sort_values("price", ascending=False)
        asks = snap[snap["side"] == "ask"].sort_values("price", ascending=True)
        if bids.empty or asks.empty:
            continue

        top = BookTop(
            snapshot_ts=str(latest_ts),
            best_bid=float(bids.iloc[0]["price"]),
            best_ask=float(asks.iloc[0]["price"]),
            bid_depth=float(bids["size"].sum()),
            ask_depth=float(asks["size"].sum()),
        )
        # We may have multiple parquets per market across days; pick newest.
        if best is None or top.snapshot_ts > best.snapshot_ts:
            best = top

    # Assertion: the chosen snapshot cannot be in the future.
    if best is not None:
        assert best.snapshot_ts <= as_of_ts, (
            f"LOOKAHEAD VIOLATION: chose snapshot {best.snapshot_ts} > "
            f"as_of_ts {as_of_ts} for market {market_id}"
        )
    return best


# ── Decision functions (pure, take as_of_ts) ─────────────────────────────

def should_open(book: BookTop, rule: EntryRule, *,
                as_of_ts: Optional[str] = None,
                resolution_ts: Optional[str] = None) -> bool:
    """Decide whether to open a paper-long position right now.

    Caller is responsible for ensuring `book.snapshot_ts <= as_of_ts`.
    Will NOT open if the market is within the force-exit window — opening
    just to immediately time-close is pure churn.
    """
    if not (rule.entry_low <= book.best_ask <= rule.entry_high):
        return False
    if as_of_ts and resolution_ts:
        try:
            t_now = datetime.fromisoformat(as_of_ts.replace("Z", "+00:00"))
            t_res = datetime.fromisoformat(resolution_ts.replace("Z", "+00:00"))
            hours_left = (t_res - t_now).total_seconds() / 3600.0
            if hours_left < rule.force_exit_hours_before_resolution:
                return False
        except (ValueError, TypeError):
            pass
    return True


def should_close(book: BookTop, position: dict,
                 rule: EntryRule, as_of_ts: str,
                 resolution_ts: Optional[str]) -> Optional[str]:
    """Return an exit reason string, or None to hold.

    Reasons:
      'target' — best_bid reached target (we can SELL to take profit)
      'stop'   — best_ask fell to stop (we'd realize at best_bid)
      'time'   — within force_exit_hours of resolution
    """
    # Take-profit check: we'd SELL at the bid.
    if book.best_bid >= rule.target:
        return "target"

    # Stop check: same — we exit at the bid.
    if book.best_bid <= rule.stop:
        return "stop"

    # Time-based force exit.
    if resolution_ts:
        try:
            t_now = datetime.fromisoformat(as_of_ts.replace("Z", "+00:00"))
            t_res = datetime.fromisoformat(resolution_ts.replace("Z", "+00:00"))
            hours_left = (t_res - t_now).total_seconds() / 3600.0
            if hours_left < rule.force_exit_hours_before_resolution:
                return "time"
        except (ValueError, TypeError):
            pass

    return None


# ── Position lifecycle ───────────────────────────────────────────────────

def open_position(market: dict, book: BookTop, rule: EntryRule,
                  as_of_ts: str) -> dict:
    """Construct a new position dict. Pure — no side effects."""
    assert book.snapshot_ts <= as_of_ts, "lookahead in open_position"
    return {
        "trade_id": str(uuid.uuid4()),
        "venue": "polymarket",
        "market_id": str(market.get("market_id", "")),
        "question": market.get("question", ""),
        "yes_token_id": market.get("yes_token_id"),
        "entry_ts": as_of_ts,
        "entry_snapshot_ts": book.snapshot_ts,
        "entry_price": book.best_ask,
        "entry_size_usd": rule.notional_usd,
        "shares": rule.notional_usd / book.best_ask if book.best_ask > 0 else 0.0,
        "target": rule.target,
        "stop": rule.stop,
        "status": "open",
        "exit_ts": None,
        "exit_snapshot_ts": None,
        "exit_price": None,
        "exit_reason": None,
        "pnl_per_share": None,
        "pnl_usd": None,
    }


def close_position(position: dict, book: BookTop, reason: str,
                   as_of_ts: str) -> dict:
    """Realize P&L. Pure — returns a new dict with exit fields populated."""
    assert book.snapshot_ts <= as_of_ts, "lookahead in close_position"
    closed = dict(position)
    exit_price = book.best_bid  # we SELL at the bid (conservative)
    pnl_per_share = exit_price - position["entry_price"]
    closed.update({
        "exit_ts": as_of_ts,
        "exit_snapshot_ts": book.snapshot_ts,
        "exit_price": exit_price,
        "exit_reason": reason,
        "status": f"closed_{reason}",
        "pnl_per_share": pnl_per_share,
        "pnl_usd": pnl_per_share * position.get("shares", 0.0),
    })
    return closed


# ── Tick driver (uses the daemon's data dir) ─────────────────────────────

def tick(data_dir: Path, markets: list[dict], rule: EntryRule,
         as_of_ts: Optional[str] = None) -> dict:
    """Run one paper-trading evaluation cycle.

    For each market in `markets`:
      * load the latest book snapshot with snapshot_ts <= as_of_ts
      * if we have no open position on this market and should_open → open
      * if we have an open position and should_close → close

    Returns a summary dict.
    """
    if as_of_ts is None:
        as_of_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    from .data import paper_persist as pp
    positions = pp.load_positions(data_dir)
    open_by_market = {
        p["market_id"]: p for p in positions if p.get("status") == "open"
    }

    opened, closed = [], []
    books_dir = data_dir / "books" / "polymarket"

    for m in markets:
        mid = str(m.get("market_id") or "")
        if not mid:
            continue
        book = latest_book_top(books_dir, mid, as_of_ts=as_of_ts)
        if book is None:
            continue

        if mid in open_by_market:
            reason = should_close(book, open_by_market[mid], rule, as_of_ts,
                                  resolution_ts=m.get("end_date"))
            if reason:
                closed_pos = close_position(open_by_market[mid], book, reason, as_of_ts)
                closed.append(closed_pos)
        else:
            if should_open(book, rule, as_of_ts=as_of_ts,
                           resolution_ts=m.get("end_date")):
                new_pos = open_position(m, book, rule, as_of_ts)
                opened.append(new_pos)

    if opened or closed:
        pp.upsert_positions(data_dir, opened=opened, closed=closed)

    obs.event(channel="fit", kind="paper.tick", level="INFO",
              as_of_ts=as_of_ts, markets_seen=len(markets),
              opened=len(opened), closed=len(closed),
              total_open=len(open_by_market) + len(opened) - len(closed))

    return {"as_of_ts": as_of_ts, "opened": len(opened), "closed": len(closed)}
