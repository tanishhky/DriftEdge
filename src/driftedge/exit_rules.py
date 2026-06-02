"""Vol-aware early-exit rules for the standard paper traders.

The default `paper.should_close` exits only at target / stop / time. That
leaves capital tied up for hours in positions that have already captured
most of their theoretical profit. This module adds a vol-aware early-exit
layer powered by realized volatility of each market's mid price.

The math is in TODO.md and the commit message; the rule logic is below.
Tunable via `EarlyExitRule`; off by default (callers opt in via the rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class EarlyExitRule:
    """Three-tier early exit. First matching trigger fires.

      1. Capital cap        — f >= f_high → exit
      2. Vol-gated          — f >= f_mid AND σ_remaining < γ * headroom → exit
      3. Time pressure      — hours_left < t_pressure AND f >= f_low → exit

    `f` = realized profit fraction = (B − E) / (T − E)
    Set `enabled=False` to disable entirely (matches legacy behaviour).
    """
    enabled: bool = True
    f_high: float = 0.80          # capital-efficiency exit
    f_mid: float = 0.50           # vol-gated exit
    f_low: float = 0.30           # time-pressure exit
    gamma: float = 0.70           # vol headroom factor; lower = exit sooner
    t_pressure_hours: float = 24.0
    default_vol_per_hour: float = 0.03  # fallback when history is missing


# ── Realized vol helper ──────────────────────────────────────────────────

def realized_vol_per_hour(books_dir: Path, market_id: str, *,
                            as_of_ts: str,
                            lookback_hours: float = 4.0) -> Optional[float]:
    """Compute σ_h from recent orderbook snapshots.

    Returns None if there aren't enough observations (<3) to compute a
    stdev. Caller falls back to `rule.default_vol_per_hour`.

    Strict no-lookahead: only reads snapshots with snapshot_ts <= as_of_ts.
    """
    market_path = books_dir / market_id
    if not market_path.exists():
        return None
    try:
        t_now = datetime.fromisoformat(as_of_ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    cutoff = (t_now - pd.Timedelta(hours=lookback_hours)).isoformat()

    # Read book parquets; concat just the snapshot_ts/side/price columns.
    rows: list[pd.DataFrame] = []
    for p in market_path.glob("*.parquet"):
        try:
            df = pd.read_parquet(p, columns=["snapshot_ts", "side", "price"])
        except Exception:
            continue
        if df.empty:
            continue
        df = df[(df["snapshot_ts"] <= as_of_ts)
                & (df["snapshot_ts"] >= cutoff)]
        if df.empty:
            continue
        rows.append(df)
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)

    # Build mid per snapshot: best bid + best ask / 2.
    best = df.sort_values("snapshot_ts").groupby(
        ["snapshot_ts", "side"]
    ).agg(price=("price", lambda s: s.max() if s.name and s.name[1] == "bid" else s.min()))
    # The groupby above gets messy with non-MultiIndex returns; do it
    # with explicit loops instead — fewer surprises.
    mids: dict[str, float] = {}
    for ts, snap in df.groupby("snapshot_ts"):
        bids = snap[snap["side"] == "bid"]["price"]
        asks = snap[snap["side"] == "ask"]["price"]
        if bids.empty or asks.empty:
            continue
        mids[str(ts)] = (float(bids.max()) + float(asks.min())) / 2.0
    if len(mids) < 3:
        return None

    ts_sorted = sorted(mids.keys())
    mid_series = [mids[t] for t in ts_sorted]
    # Stdev of first-differences (simple returns), then annualise to per-hour.
    diffs = [mid_series[i + 1] - mid_series[i] for i in range(len(mid_series) - 1)]
    intervals_h = []
    for i in range(len(ts_sorted) - 1):
        a = datetime.fromisoformat(ts_sorted[i].replace("Z", "+00:00"))
        b = datetime.fromisoformat(ts_sorted[i + 1].replace("Z", "+00:00"))
        intervals_h.append(max(1e-6, (b - a).total_seconds() / 3600.0))
    if not diffs or not intervals_h:
        return None
    # Per-step variance / step time = per-hour variance.
    var_per_step = sum(d * d for d in diffs) / len(diffs)
    avg_interval = sum(intervals_h) / len(intervals_h)
    var_per_hour = var_per_step / max(avg_interval, 1e-6)
    return float(math.sqrt(var_per_hour))


# ── Decision predicate ──────────────────────────────────────────────────

def early_exit_reason(*,
                      best_bid: float,
                      entry_price: float,
                      target: float,
                      hours_left: Optional[float],
                      vol_per_hour: float,
                      rule: EarlyExitRule) -> Optional[str]:
    """Return an exit-reason tag or None.

    Tags:
      'early_target_high' — f ≥ f_high
      'early_target_vol'  — f ≥ f_mid AND σ_remaining < γ × headroom
      'early_target_time' — hours_left < t_pressure AND f ≥ f_low
    """
    if not rule.enabled:
        return None
    if best_bid <= entry_price:
        return None  # not in profit; the standard stop handles losses
    if target <= entry_price:
        return None
    f = (best_bid - entry_price) / (target - entry_price)

    if f >= rule.f_high:
        return "early_target_high"

    if hours_left is not None:
        if hours_left < rule.t_pressure_hours and f >= rule.f_low:
            return "early_target_time"
        sigma_remaining = vol_per_hour * math.sqrt(max(hours_left, 0.0))
        headroom = max(target - best_bid, 0.0)
        if f >= rule.f_mid and sigma_remaining < rule.gamma * headroom:
            return "early_target_vol"

    return None
