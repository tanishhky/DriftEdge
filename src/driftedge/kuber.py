"""Kuber — execution layer (the *how*).

`agents.kuber` defines the *policy* (sleeves, sizing, kill switches).
This module is the *executor*: it runs one tick across all four sleeves,
either in PAPER mode (simulated fills, default) or LIVE mode (real
Kalshi orders, gated on ``config.kuber_live``).

State files (separate from the 5 paper agents):
    data/kuber_trades.parquet     — Kuber's trades, one row per fill
    data/kuber_state.parquet      — per-sleeve cash, exposure, realised PnL
    data/kuber_orders.parquet     — resting live orders (live-mode only)

Persistence contract: file shapes mirror the paper-agent equivalents so
Sentinel readers can be added cheaply (same columns, just different
sources).
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import obs
from . import paper
from . import sizing as base_sizing
from .agents import kuber as kpol
from .config import Config


# ── Persistence (Kuber-specific files) ───────────────────────────────

KUBER_TRADES_FILE = "kuber_trades.parquet"
KUBER_STATE_FILE = "kuber_state.parquet"


def _trades_path(data_dir: Path) -> Path:
    return data_dir / KUBER_TRADES_FILE


def _state_path(data_dir: Path) -> Path:
    return data_dir / KUBER_STATE_FILE


def load_positions(data_dir: Path) -> list[dict]:
    p = _trades_path(data_dir)
    if not p.exists():
        return []
    try:
        df = pd.read_parquet(p)
    except Exception as exc:
        obs.event(channel="error", kind="kuber.load_fail", level="WARNING",
                  err=str(exc))
        return []
    return df.where(pd.notna(df), None).to_dict(orient="records")


def upsert_positions(data_dir: Path, *, opened: list[dict],
                     closed: list[dict]) -> None:
    p = _trades_path(data_dir)
    existing = load_positions(data_dir)
    by_id = {r["trade_id"]: r for r in existing}
    for o in opened:
        by_id[o["trade_id"]] = o
    for c in closed:
        by_id[c["trade_id"]] = c
    rows = list(by_id.values())
    df = pd.DataFrame(rows)
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   p, compression="snappy")
    obs.event(channel="persist", kind="kuber.upsert", level="INFO",
              opened=len(opened), closed=len(closed),
              total_rows=len(rows))


def init_state(data_dir: Path, kc: kpol.KuberConfig
               ) -> dict[str, base_sizing.TraderState]:
    """Idempotent: create the per-sleeve state on first run, else load."""
    p = _state_path(data_dir)
    bankroll = kc.bankroll_per_sleeve_usd

    if not p.exists():
        rows = []
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for tid in kpol.SLEEVE_LABELS:
            rows.append({
                "trader": tid,
                "bankroll_init": bankroll,
                "cash_usd": bankroll,
                "open_exposure": 0.0,
                "closed_pnl": 0.0,
                "peak_equity": bankroll,
                "current_drawdown_pct": 0.0,
                "updated_ts": now,
            })
        df = pd.DataFrame(rows)
        data_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                       p, compression="snappy")
        obs.event(channel="persist", kind="kuber.state.init", level="INFO",
                  sleeves=len(rows), bankroll_per_sleeve=bankroll)

    return load_state(data_dir)


def load_state(data_dir: Path) -> dict[str, base_sizing.TraderState]:
    p = _state_path(data_dir)
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    out: dict[str, base_sizing.TraderState] = {}
    for _, r in df.iterrows():
        out[r["trader"]] = base_sizing.TraderState(
            trader=r["trader"],
            bankroll_init=float(r["bankroll_init"]),
            cash_usd=float(r["cash_usd"]),
            open_exposure=float(r["open_exposure"]),
            closed_pnl=float(r["closed_pnl"]),
        )
    return out


def save_state(data_dir: Path,
                 states: dict[str, base_sizing.TraderState]) -> None:
    p = _state_path(data_dir)
    existing = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    existing_idx = ({r["trader"]: r for _, r in existing.iterrows()}
                     if not existing.empty else {})

    rows = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for tid, s in states.items():
        prev_peak = float(existing_idx.get(tid, {}).get(
            "peak_equity", s.bankroll_init))
        peak = max(prev_peak, s.total_equity)
        dd = ((peak - s.total_equity) / peak * 100.0) if peak > 0 else 0.0
        rows.append({
            "trader": tid,
            "bankroll_init": s.bankroll_init,
            "cash_usd": s.cash_usd,
            "open_exposure": s.open_exposure,
            "closed_pnl": s.closed_pnl,
            "peak_equity": peak,
            "current_drawdown_pct": round(dd, 3),
            "updated_ts": now,
        })
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   p, compression="snappy")


# ── Helpers ──────────────────────────────────────────────────────────

def _today_realized_pnl_total(positions: list[dict],
                                as_of_ts: str) -> float:
    """Sum realised PnL across all sleeves for the UTC day of as_of_ts.
    Drives the daily-loss kill switch.
    """
    today = kpol.utc_date_of(as_of_ts)
    total = 0.0
    for p in positions:
        if str(p.get("status", "")).startswith("closed"):
            exit_ts = p.get("exit_ts")
            if exit_ts and kpol.utc_date_of(str(exit_ts)) == today:
                v = p.get("pnl_usd")
                if v is not None and pd.notna(v):
                    total += float(v)
    return total


def _sleeve_realized_pnl(positions: list[dict], sleeve_label: str) -> float:
    total = 0.0
    for p in positions:
        if p.get("trader") != sleeve_label:
            continue
        if str(p.get("status", "")).startswith("closed"):
            v = p.get("pnl_usd")
            if v is not None and pd.notna(v):
                total += float(v)
    return total


def _sleeve_open_count(positions: list[dict], sleeve_label: str) -> int:
    return sum(1 for p in positions
                if p.get("trader") == sleeve_label
                and p.get("status") == "open")


def _total_equity(states: dict[str, base_sizing.TraderState]) -> float:
    return sum(s.total_equity for s in states.values())


def _kuber_open_position(market: dict, book: paper.BookTop,
                           rule: paper.EntryRule, as_of_ts: str,
                           *, trader: str, size_usd: float,
                           venue: str = "kalshi") -> dict:
    """Same shape as paper.open_position but tagged for Kuber and Kalshi
    only. The trader label includes the ``kuber:`` prefix.
    """
    pos = paper.open_position(market, book, rule, as_of_ts,
                                trader=trader, size_usd=size_usd,
                                venue=venue)
    pos["mode"] = "paper"   # flipped to "live" when Kalshi accepts the order
    pos["kalshi_order_id"] = None
    return pos


# ── Live exec (Kalshi) ───────────────────────────────────────────────

def _place_kalshi_limit(client, *, position: dict,
                         book: paper.BookTop, kc: kpol.KuberConfig
                         ) -> tuple[bool, Optional[str]]:
    """Attempt a real Kalshi BUY YES limit at our edge price.

    Limit price = the book ask we'd have paid in paper mode, but in
    integer cents. post_only=True means we won't cross the spread if the
    price moved against us between snapshot and now.

    Returns (filled, order_id). For the first cut we treat the API's
    immediate response as "accepted but resting"; reconciliation against
    Kalshi's truth source (`get_orders`/`get_fills`) is done on the next
    tick. A future iteration can add a synchronous fill-or-cancel.
    """
    ticker = position["market_id"]
    n_contracts = max(1, int(round(position["entry_size_usd"]
                                     / max(book.best_ask, 0.01))))
    # Limit price in cents — round to integer. Stay at our intended price
    # (book.best_ask) and let post_only handle adverse moves.
    price_cents = max(1, min(99, int(round(book.best_ask * 100))))
    try:
        resp = client.place_order(
            market_ticker=ticker,
            side="yes",
            action="buy",
            order_type="limit",
            count=n_contracts,
            price_cents=price_cents,
            client_order_id=position["trade_id"][:32],
            post_only=True,
        )
    except Exception as exc:
        obs.event(channel="error", kind="kuber.live_order_fail",
                  level="WARNING", trader=position["trader"],
                  market=ticker, err=str(exc))
        return False, None
    order = resp.get("order") if isinstance(resp, dict) else None
    order_id = order.get("order_id") if isinstance(order, dict) else None
    obs.event(channel="persist", kind="kuber.live_order_placed",
              level="INFO", trader=position["trader"], market=ticker,
              order_id=order_id, n_contracts=n_contracts,
              price_cents=price_cents)
    return True, order_id


# ── Tick ─────────────────────────────────────────────────────────────

def _entry_book(books_dir: Path, market_id: str, as_of_ts: str,
                allow_one_sided: bool):
    """Latest book for ENTRY. Falls back to an ask-only snapshot (common on thin
    Kalshi markets) when allow_one_sided is True; a missing bid is NaN so
    bid-based exits simply don't fire until a bid appears."""
    bt = paper.latest_book_top(books_dir, market_id, as_of_ts=as_of_ts)
    if bt is not None or not allow_one_sided:
        return bt
    md = books_dir / market_id
    if not md.exists():
        return None
    as_of_dt = paper.parse_iso(as_of_ts)
    best = None
    best_dt = None
    for pq in md.glob("*.parquet"):
        try:
            df = pd.read_parquet(pq)
        except Exception:
            continue
        if df.empty or "snapshot_ts" not in df.columns:
            continue
        ts = pd.to_datetime(df["snapshot_ts"], utc=True, format="mixed")
        df = df.assign(_t=ts)
        df = df[df["_t"] <= as_of_dt]
        if df.empty:
            continue
        latest = df["_t"].max()
        snap = df[df["_t"] == latest]
        asks = snap[snap["side"] == "ask"].sort_values("price")
        if asks.empty:
            continue
        bids = snap[snap["side"] == "bid"].sort_values("price", ascending=False)
        bb = float(bids.iloc[0]["price"]) if not bids.empty else float("nan")
        top = paper.BookTop(
            snapshot_ts=str(snap["snapshot_ts"].iloc[0]),
            best_bid=bb, best_ask=float(asks.iloc[0]["price"]),
            bid_depth=float(bids["size"].sum()) if not bids.empty else 0.0,
            ask_depth=float(asks["size"].sum()))
        ld = latest.to_pydatetime() if hasattr(latest, "to_pydatetime") else latest
        if best is None or (best_dt is not None and ld > best_dt):
            best = top
            best_dt = ld
    return best


def tick(data_dir: Path, markets: list[dict], rule: paper.EntryRule,
         cfg: Config, *,
         as_of_ts: Optional[str] = None,
         kalshi_client: Any = None) -> dict:
    """Run Kuber once. Idempotent; paper-or-live by config.

    Kuber trades **Kalshi only** — Polymarket is filtered out because we
    have no on-chain wallet wired up.

    Decision contract identical to paper.tick:
      * read books from the same data dir as the paper agents (we share
        the market snapshot stream)
      * apply paper.should_open / paper.should_close
      * route the fill through either simulated or live exec

    Returns a per-sleeve summary dict.
    """
    if as_of_ts is None:
        as_of_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    kc = kpol.KuberConfig(
        bankroll_total_usd=cfg.kuber_bankroll_total_usd,
        bankroll_per_sleeve_usd=cfg.kuber_bankroll_per_sleeve_usd,
        max_position_usd=cfg.kuber_max_position_usd,
        dd_kill_pct=cfg.kuber_dd_kill_pct,
        daily_loss_kill_usd=cfg.kuber_daily_loss_kill_usd,
        live=cfg.kuber_live,
    )

    states = init_state(data_dir, kc)
    positions = load_positions(data_dir)

    # ── Kalshi-only filter ──
    kalshi_markets = [m for m in markets
                       if (m.get("venue") or "").lower() == "kalshi"]
    if not kalshi_markets:
        obs.event(channel="run", kind="kuber.no_kalshi_markets",
                  level="DEBUG", as_of_ts=as_of_ts)
        return {"as_of_ts": as_of_ts, "opened": 0, "closed": 0,
                 "gate_reason": "no_markets", "mode": "live" if kc.live else "paper"}

    # ── Gate (drawdown + daily loss) ──
    today_pnl = _today_realized_pnl_total(positions, as_of_ts)
    total_eq = _total_equity(states)
    gate_decision = kpol.gate(total_eq, today_pnl, kc, as_of_ts)

    # ── Live-mode preflight: require credentials and a client ──
    if kc.live:
        if kalshi_client is None:
            obs.event(channel="error", kind="kuber.live_no_client",
                      level="ERROR", as_of_ts=as_of_ts)
            return {"as_of_ts": as_of_ts, "opened": 0, "closed": 0,
                     "gate_reason": "live_no_client", "mode": "live"}
        if not (cfg.kalshi_api_key_id and cfg.kalshi_private_key_path):
            obs.event(channel="error", kind="kuber.live_no_credentials",
                      level="ERROR", as_of_ts=as_of_ts)
            return {"as_of_ts": as_of_ts, "opened": 0, "closed": 0,
                     "gate_reason": "live_no_credentials", "mode": "live"}

    # Build (trader, market_id) index for existing opens.
    open_by_key: dict[tuple[str, str], dict] = {
        (p.get("trader"), str(p.get("market_id"))): p
        for p in positions if p.get("status") == "open"
    }

    opened: list[dict] = []
    closed: list[dict] = []
    books_usable = 0

    for m in kalshi_markets:
        mid = str(m.get("market_id") or "")
        if not mid:
            continue
        books_dir = data_dir / "books" / "kalshi"
        book = _entry_book(books_dir, mid, as_of_ts, cfg.kuber_allow_one_sided)
        if book is None:
            continue
        books_usable += 1

        for sleeve in kpol.SLEEVE_LABELS:
            existing = open_by_key.get((sleeve, mid))
            state = states[sleeve]

            # ── CLOSE branch (allowed even when gate blocks entries) ──
            if existing is not None:
                reason = paper.should_close(book, existing, rule,
                                              as_of_ts=as_of_ts,
                                              resolution_ts=existing.get("resolution_ts"))
                if reason is not None:
                    closed_pos = paper.close_position(existing, book, reason,
                                                       as_of_ts=as_of_ts)
                    # cost-aware accounting
                    cost = float(existing.get("entry_size_usd", 0.0))
                    pnl = float(closed_pos.get("pnl_usd", 0.0) or 0.0)
                    states[sleeve] = base_sizing.TraderState(
                        trader=sleeve, bankroll_init=state.bankroll_init,
                        cash_usd=state.cash_usd + cost + pnl,
                        open_exposure=state.open_exposure - cost,
                        closed_pnl=state.closed_pnl + pnl,
                    )
                    closed.append(closed_pos)
                    obs.event(channel="fit", kind="kuber.close",
                              level="INFO", trader=sleeve, market=mid,
                              reason=reason, pnl=round(pnl, 2),
                              mode=("live" if kc.live else "paper"))
                continue   # don't re-open in the same tick on the same market

            # ── OPEN branch (only if gate allows) ──
            if not gate_decision.allow_entries:
                continue
            if not paper.should_open(book, rule, as_of_ts=as_of_ts,
                                       resolution_ts=m.get("end_date")):
                continue

            sleeve_pnl = _sleeve_realized_pnl(positions, sleeve)
            sleeve_open = _sleeve_open_count(positions, sleeve)
            size_usd = kpol.kuber_size(
                sleeve_label=sleeve,
                c=book.mid,
                target=rule.target,
                stop=rule.stop,
                state=state,
                kc=kc,
                sleeve_realized_pnl=sleeve_pnl,
                sleeve_open_positions=sleeve_open,
            )
            if size_usd <= 0:
                continue

            new_pos = _kuber_open_position(m, book, rule, as_of_ts,
                                             trader=sleeve, size_usd=size_usd,
                                             venue="kalshi")

            # Live mode: place real Kalshi order. If the call fails, do
            # NOT record a position — protects against ghosting (where we
            # think we have it but Kalshi never accepted).
            if kc.live:
                ok, order_id = _place_kalshi_limit(kalshi_client,
                                                     position=new_pos,
                                                     book=book, kc=kc)
                if not ok:
                    continue
                new_pos["mode"] = "live"
                new_pos["kalshi_order_id"] = order_id

            states[sleeve] = base_sizing.TraderState(
                trader=sleeve, bankroll_init=state.bankroll_init,
                cash_usd=state.cash_usd - size_usd,
                open_exposure=state.open_exposure + size_usd,
                closed_pnl=state.closed_pnl,
            )
            opened.append(new_pos)
            # update index so we don't double-open within this tick
            open_by_key[(sleeve, mid)] = new_pos
            obs.event(channel="fit", kind="kuber.open",
                      level="INFO", trader=sleeve, market=mid,
                      size_usd=round(size_usd, 2),
                      mode=("live" if kc.live else "paper"))

    # Persist.
    if opened or closed:
        upsert_positions(data_dir, opened=opened, closed=closed)
    save_state(data_dir, states)

    obs.event(channel="run", kind="kuber.tick", level="INFO",
              markets_seen=len(kalshi_markets), books_usable=books_usable,
              opened=len(opened), closed=len(closed),
              gate_reason=gate_decision.reason,
              allow_one_sided=cfg.kuber_allow_one_sided,
              mode=("live" if kc.live else "paper"))

    return {
        "as_of_ts": as_of_ts,
        "opened": len(opened),
        "closed": len(closed),
        "gate_reason": gate_decision.reason,
        "mode": "live" if kc.live else "paper",
        "sleeve_summary": {
            tid: {
                "cash_usd": round(s.cash_usd, 2),
                "open_exposure": round(s.open_exposure, 2),
                "closed_pnl": round(s.closed_pnl, 2),
                "total_equity": round(s.total_equity, 2),
            } for tid, s in states.items()
        },
    }
