"""Per-trader equity-history persistence.

A continuous mark-to-market record of every trader's portfolio equity,
written at every tick of the paper engine. This file backs the Sentinel
equity curve and the risk-stats panel.

Schema (one row per trader per tick):

    ts                  ISO 8601 UTC
    trader              kelly | equal | volwt
    cash_usd            free cash
    open_exposure_usd   sum of entry_size_usd across all open positions
    mtm_unrealized_usd  sum over open positions of (current_mid - entry_price) * shares
    closed_pnl_usd      cumulative realized P&L
    total_equity_usd    cash_usd + open_exposure_usd + mtm_unrealized_usd
                        (== bankroll_init + closed_pnl_usd + mtm_unrealized_usd)
    peak_equity_usd     running max of total_equity_usd
    drawdown_pct        (peak - total) / peak * 100

Append-only. One Parquet file (small, rewritten each tick — acceptable until
we exceed ~10MB; at that point partition by date).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import obs


_FILE = "equity_history.parquet"


def _path(data_dir: Path) -> Path:
    return data_dir / _FILE


def load(data_dir: Path) -> pd.DataFrame:
    p = _path(data_dir)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except Exception as exc:
        obs.event(channel="error", kind="equity.load_fail",
                  level="WARNING", err=str(exc))
        return pd.DataFrame()


def append_snapshot(data_dir: Path, snapshots: list[dict]) -> Path:
    """Append per-trader MTM rows. `snapshots` is one row per trader.

    Each row must include the schema fields documented in the module header.
    """
    if not snapshots:
        return _path(data_dir)

    p = _path(data_dir)
    df_new = pd.DataFrame(snapshots)

    if p.exists():
        try:
            existing = pd.read_parquet(p)
            combined = pd.concat([existing, df_new], ignore_index=True)
        except Exception:
            combined = df_new
    else:
        combined = df_new

    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False),
                   p, compression="snappy")
    obs.event(channel="persist", kind="equity.snapshot",
              level="DEBUG", traders=len(df_new),
              total_rows=len(combined), bytes=p.stat().st_size)
    obs.bump("persist_writes")
    return p


def build_snapshot(ts: str,
                   states: dict,
                   open_positions: list[dict],
                   book_mids: dict[tuple[str, str], float],
                   peak_by_trader: dict[str, float]) -> list[dict]:
    """Construct one row per trader given the current world state.

    `states`        — dict[trader_id, TraderState]
    `open_positions` — list of dicts with 'trader', 'venue', 'market_id',
                       'entry_price', 'shares', 'entry_size_usd'
    `book_mids`     — (venue, market_id) -> current mid price; missing → use entry
    `peak_by_trader` — running peaks (updated in place)
    """
    rows: list[dict] = []
    by_trader: dict[str, list[dict]] = {}
    for pos in open_positions:
        by_trader.setdefault(pos.get("trader", ""), []).append(pos)

    for trader_id, st in states.items():
        positions = by_trader.get(trader_id, [])
        mtm = 0.0
        for pos in positions:
            key = (pos.get("venue", "polymarket"), pos.get("market_id", ""))
            mid = book_mids.get(key)
            entry_price = float(pos.get("entry_price") or 0.0)
            shares = float(pos.get("shares") or 0.0)
            if mid is None or entry_price <= 0:
                continue
            # ── Side-aware MTM. YES legs MTM at the YES mid directly. NO
            # legs (synthetic, used by volharvest's hedge) MTM at (1 - yes_mid)
            # because buying NO is the inverse claim on the same underlying.
            side = (pos.get("side") or "yes").lower()
            current_price = (1.0 - float(mid)) if side == "no" else float(mid)
            mtm += (current_price - entry_price) * shares

        cash = float(st.cash_usd)
        exposure = float(st.open_exposure)
        total_equity = cash + exposure + mtm
        prev_peak = float(peak_by_trader.get(trader_id, st.bankroll_init))
        peak = max(prev_peak, total_equity)
        peak_by_trader[trader_id] = peak
        dd_pct = ((peak - total_equity) / peak * 100.0) if peak > 0 else 0.0

        rows.append({
            "ts": ts,
            "trader": trader_id,
            "cash_usd": round(cash, 4),
            "open_exposure_usd": round(exposure, 4),
            "mtm_unrealized_usd": round(mtm, 4),
            "closed_pnl_usd": round(float(st.closed_pnl), 4),
            "total_equity_usd": round(total_equity, 4),
            "peak_equity_usd": round(peak, 4),
            "drawdown_pct": round(dd_pct, 4),
        })
    return rows


def latest_peaks(data_dir: Path) -> dict[str, float]:
    """Recover running peaks from the history file (so restarts don't reset DD)."""
    df = load(data_dir)
    if df.empty or "trader" not in df.columns:
        return {}
    out: dict[str, float] = {}
    for tid, grp in df.groupby("trader"):
        out[str(tid)] = float(grp["peak_equity_usd"].max())
    return out


def trim(data_dir: Path, max_rows: int = 500_000) -> None:
    """Keep the most recent `max_rows` rows. Cheap insurance against unbounded growth."""
    df = load(data_dir)
    if df.empty or len(df) <= max_rows:
        return
    keep = df.sort_values("ts").tail(max_rows)
    p = _path(data_dir)
    pq.write_table(pa.Table.from_pandas(keep, preserve_index=False),
                   p, compression="snappy")
    obs.event(channel="persist", kind="equity.trim", level="INFO",
              kept=len(keep))
