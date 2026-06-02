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
                  as_of_ts: str, *, trader: str, size_usd: float,
                  venue: str = "polymarket") -> dict:
    """Construct a new position dict. Pure — no side effects."""
    assert book.snapshot_ts <= as_of_ts, "lookahead in open_position"
    return {
        "trade_id": str(uuid.uuid4()),
        "trader": trader,
        "venue": venue,
        "market_id": str(market.get("market_id", "")),
        "question": market.get("question", ""),
        "category": market.get("category") or "other",
        "yes_token_id": market.get("yes_token_id"),
        "entry_ts": as_of_ts,
        "entry_snapshot_ts": book.snapshot_ts,
        "entry_price": book.best_ask,
        "entry_size_usd": size_usd,
        "shares": size_usd / book.best_ask if book.best_ask > 0 else 0.0,
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
    pnl_usd = pnl_per_share * position.get("shares", 0.0)
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


# ── Tick driver (uses the daemon's data dir) ─────────────────────────────

def tick(data_dir: Path, markets: list[dict], rule: EntryRule,
         as_of_ts: Optional[str] = None,
         bankroll: float = 10000.0) -> dict:
    """Run one paper-trading evaluation cycle for ALL traders.

    Three independent traders (kelly, equal, volwt) each see the same
    markets via the same entry/exit triggers, but size positions
    differently. The lookahead discipline applies to all of them: every
    snapshot read is filtered to snapshot_ts <= as_of_ts.

    Returns a per-trader summary dict.
    """
    if as_of_ts is None:
        as_of_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    from .data import paper_persist as pp
    from .data import state_persist as sp
    from .data import equity_persist as ep
    from . import sizing
    from . import exit_rules as er

    early_rule = er.EarlyExitRule()

    # Initialise per-trader state if first run; load otherwise.
    states = sp.init_state(data_dir, bankroll=bankroll)

    positions = pp.load_positions(data_dir)
    # Key positions by (trader, venue, market_id) so each trader has its own
    # book per venue (handles cross-venue same-event markets).
    open_by_key = {
        (p.get("trader"), p.get("venue", "polymarket"), p.get("market_id")): p
        for p in positions if p.get("status") == "open"
    }

    # book_mids feeds the MTM snapshot at the end of the tick. We populate
    # it for every market we see a book for, whether or not we open/close.
    book_mids: dict[tuple[str, str], float] = {}

    opened, closed = [], []

    per_trader_opened: dict[str, int] = {t: 0 for t in sizing.trader_labels()}
    per_trader_closed: dict[str, int] = {t: 0 for t in sizing.trader_labels()}

    for m in markets:
        mid = str(m.get("market_id") or "")
        if not mid:
            continue
        venue = m.get("venue") or "polymarket"
        books_dir = data_dir / "books" / venue
        book = latest_book_top(books_dir, mid, as_of_ts=as_of_ts)
        if book is None:
            continue

        book_mids[(venue, mid)] = book.mid

        # ── Realized vol for this market (per hour) — drives the vol-aware
        # early-exit predicate below. Computed once per market to keep the
        # tick cheap. None ⇒ fall back to the rule's default.
        market_vol_ph: Optional[float] = None
        if any(open_by_key.get((tid, venue, mid)) is not None
               for tid in sizing.trader_labels()):
            market_vol_ph = er.realized_vol_per_hour(
                books_dir, mid, as_of_ts=as_of_ts)
        market_vol_ph = market_vol_ph if market_vol_ph is not None else early_rule.default_vol_per_hour

        # Hours-to-resolution helper, shared between the standard close
        # check and the early-exit check.
        def _hours_left() -> Optional[float]:
            res_ts = m.get("end_date")
            if not res_ts:
                return None
            try:
                t_now = datetime.fromisoformat(as_of_ts.replace("Z", "+00:00"))
                t_res = datetime.fromisoformat(res_ts.replace("Z", "+00:00"))
                return max(0.0, (t_res - t_now).total_seconds() / 3600.0)
            except (ValueError, TypeError):
                return None
        hours_left_val = _hours_left()

        # ── EXIT side first (frees up capital before evaluating new opens) ──
        for trader_id in sizing.trader_labels():
            existing = open_by_key.get((trader_id, venue, mid))
            if existing is None:
                continue
            # Standard target / stop / time first.
            reason = should_close(book, existing, rule, as_of_ts,
                                  resolution_ts=m.get("end_date"))
            # Vol-aware early exit only if the standard rule didn't already
            # decide. This preserves backwards compat: if early-exit is
            # disabled, behaviour is identical to before.
            if reason is None:
                reason = er.early_exit_reason(
                    best_bid=book.best_bid,
                    entry_price=float(existing.get("entry_price") or 0.0),
                    target=rule.target,
                    hours_left=hours_left_val,
                    vol_per_hour=market_vol_ph,
                    rule=early_rule,
                )
            if reason:
                closed_pos = close_position(existing, book, reason, as_of_ts)
                closed.append(closed_pos)
                per_trader_closed[trader_id] += 1
                size_usd = float(existing.get("entry_size_usd", 0.0))
                pnl_usd = float(closed_pos.get("pnl_usd", 0.0))
                states[trader_id] = sp.apply_close(states[trader_id],
                                                   size_usd, pnl_usd)
                del open_by_key[(trader_id, venue, mid)]

        # ── ENTRY side ──
        if not should_open(book, rule, as_of_ts=as_of_ts,
                           resolution_ts=m.get("end_date")):
            continue
        for trader_id, sizer_fn in sizing.SIZERS.items():
            if (trader_id, venue, mid) in open_by_key:
                continue
            size_usd = sizer_fn(states[trader_id], c=book.best_ask,
                                target=rule.target, stop=rule.stop)
            if size_usd <= 0:
                continue
            new_pos = open_position(m, book, rule, as_of_ts,
                                    trader=trader_id, size_usd=size_usd,
                                    venue=venue)
            opened.append(new_pos)
            per_trader_opened[trader_id] += 1
            states[trader_id] = sp.apply_open(states[trader_id], size_usd)

    if opened or closed:
        pp.upsert_positions(data_dir, opened=opened, closed=closed)
        sp.save_state(data_dir, states)

    # ── Equity snapshot (every tick, regardless of opens/closes) ──
    # Rebuild the open-positions list AFTER the entry/exit pass so the
    # MTM reflects the position book we actually hold at as_of_ts.
    open_positions_now = list(open_by_key.values()) + opened
    peaks = ep.latest_peaks(data_dir)
    snapshots = ep.build_snapshot(
        ts=as_of_ts,
        states=states,
        open_positions=open_positions_now,
        book_mids=book_mids,
        peak_by_trader=peaks,
    )
    if snapshots:
        ep.append_snapshot(data_dir, snapshots)

    obs.event(channel="fit", kind="paper.tick", level="INFO",
              as_of_ts=as_of_ts, markets_seen=len(markets),
              opened_by_trader=per_trader_opened,
              closed_by_trader=per_trader_closed,
              equity_rows=len(snapshots))

    return {"as_of_ts": as_of_ts,
            "opened_by_trader": per_trader_opened,
            "closed_by_trader": per_trader_closed,
            "equity_snapshot_rows": len(snapshots)}
