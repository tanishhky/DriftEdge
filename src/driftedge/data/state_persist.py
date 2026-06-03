"""Per-trader portfolio state persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import obs
from ..sizing import TraderState, all_trader_labels


_STATE_FILE = "paper_state.parquet"


def _path(data_dir: Path) -> Path:
    return data_dir / _STATE_FILE


def init_state(data_dir: Path, bankroll: float = 10000.0) -> dict[str, TraderState]:
    """Idempotent: create initial state file if missing, else load existing.

    If the file exists but is missing rows for newly-added traders (e.g. a
    new SELF_MANAGED_TRADERS entry), those rows are appended automatically so
    the state file stays in sync with the codebase without a manual reset.
    """
    p = _path(data_dir)
    if not p.exists():
        rows = []
        for tid in all_trader_labels():
            rows.append({
                "trader": tid,
                "bankroll_init": bankroll,
                "cash_usd": bankroll,
                "open_exposure": 0.0,
                "closed_pnl": 0.0,
                "peak_equity": bankroll,
                "current_drawdown_pct": 0.0,
                "updated_ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
        df = pd.DataFrame(rows)
        data_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                       p, compression="snappy")
        obs.event(channel="persist", kind="paper.state.init", level="INFO",
                  traders=len(rows), bankroll=bankroll)
        return load_state(data_dir)

    # File exists — add any traders absent from the on-disk state.
    existing = load_state(data_dir)
    missing = [tid for tid in all_trader_labels() if tid not in existing]
    if missing:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        df_old = pd.read_parquet(p)
        new_rows = pd.DataFrame([{
            "trader": tid,
            "bankroll_init": bankroll,
            "cash_usd": bankroll,
            "open_exposure": 0.0,
            "closed_pnl": 0.0,
            "peak_equity": bankroll,
            "current_drawdown_pct": 0.0,
            "updated_ts": now,
        } for tid in missing])
        df_merged = pd.concat([df_old, new_rows], ignore_index=True)
        pq.write_table(pa.Table.from_pandas(df_merged, preserve_index=False),
                       p, compression="snappy")
        obs.event(channel="persist", kind="paper.state.backfill", level="INFO",
                  added=missing, bankroll=bankroll)
        return load_state(data_dir)

    return existing


def load_state(data_dir: Path) -> dict[str, TraderState]:
    p = _path(data_dir)
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    out: dict[str, TraderState] = {}
    for _, r in df.iterrows():
        out[r["trader"]] = TraderState(
            trader=r["trader"],
            bankroll_init=float(r["bankroll_init"]),
            cash_usd=float(r["cash_usd"]),
            open_exposure=float(r["open_exposure"]),
            closed_pnl=float(r["closed_pnl"]),
        )
    return out


def save_state(data_dir: Path, states: dict[str, TraderState]) -> None:
    p = _path(data_dir)
    existing = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    existing_idx = {r["trader"]: r for _, r in existing.iterrows()} if not existing.empty else {}

    rows = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for tid, s in states.items():
        prev = existing_idx.get(tid, {})
        peak = max(float(prev.get("peak_equity", s.bankroll_init)), s.total_equity)
        dd_pct = ((peak - s.total_equity) / peak * 100.0) if peak > 0 else 0.0
        rows.append({
            "trader": tid,
            "bankroll_init": s.bankroll_init,
            "cash_usd": s.cash_usd,
            "open_exposure": s.open_exposure,
            "closed_pnl": s.closed_pnl,
            "peak_equity": peak,
            "current_drawdown_pct": round(dd_pct, 3),
            "updated_ts": now,
        })
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   p, compression="snappy")
    obs.bump("persist_writes")


def apply_open(state: TraderState, size_usd: float) -> TraderState:
    return TraderState(
        trader=state.trader,
        bankroll_init=state.bankroll_init,
        cash_usd=state.cash_usd - size_usd,
        open_exposure=state.open_exposure + size_usd,
        closed_pnl=state.closed_pnl,
    )


def apply_close(state: TraderState, size_usd: float, pnl_usd: float) -> TraderState:
    return TraderState(
        trader=state.trader,
        bankroll_init=state.bankroll_init,
        cash_usd=state.cash_usd + size_usd + pnl_usd,
        open_exposure=state.open_exposure - size_usd,
        closed_pnl=state.closed_pnl + pnl_usd,
    )
