"""Kuber — the 6th DriftEdge agent (real-money-capable).

Design
------
Kuber is a *meta-agent* with four sleeves sharing one $500 wallet:

    kuber:kelly        $125  (Kelly sizing on edge)
    kuber:equal        $125  (equal-weight)
    kuber:volwt        $125  (inverse-vol weighted)
    kuber:volharvest   $125  (vol-harvest strategy)

Each sleeve runs the same decision logic as its paper-trading namesake
(see ``sizing.py`` and ``agents.volharvest``) but writes to a separate
state file (``kuber_state.parquet``) and uses Kuber-specific caps so the
positions and bankroll never bleed into the 5 paper agents.

Sizing rule (per position, per sleeve)
--------------------------------------
::

    sleeve_bankroll = $125 + realized_pnl_for_that_sleeve
    max_position    = max($5, 0.04 * sleeve_bankroll)
    floor           = $5  (Kalshi min-trade size + commission floor)

At $0 realised PnL the cap is $5 (4% of $125). At +$125 realised in a
sleeve, the cap is $10. At +$375 it's $20. Linear, monotonic, transparent.

Kill switches (applied before every entry)
------------------------------------------
1. **Drawdown kill**: if total Kuber equity dips below
   ``(1 - kuber_dd_kill_pct) * bankroll_total`` (default 60% of $500 =
   $300), all sleeves freeze new entries.
2. **Daily-loss kill**: if Kuber's net realised PnL over the current UTC
   day is below ``-kuber_daily_loss_kill_usd`` (default −$25), all
   sleeves freeze new entries for the rest of that UTC day.
3. **Sleeve-cash floor**: a sleeve cannot open a position larger than
   its remaining cash (delegated to ``sizing._apply_caps``).
4. **Per-sleeve position count cap**: hard limit of 15 concurrent
   positions per sleeve (avoid hyper-fragmentation).

Paper/live gate
---------------
The decision functions in this module are mode-agnostic. The execution
side (``driftedge.kuber.tick``) reads ``config.kuber_live`` to decide
between simulated fills and real Kalshi orders. This module never
imports KalshiClient — it just emits decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .. import sizing as base_sizing


# ── Kuber-specific constants ─────────────────────────────────────────

# Sleeve labels — colon prefix keeps them grouped in dashboards and
# avoids collision with the paper-agent labels (kelly, equal, ...).
SLEEVE_LABELS: list[str] = [
    "kuber:kelly",
    "kuber:equal",
    "kuber:volwt",
    "kuber:volharvest",
]

# Floor — Kalshi imposes a 1-contract minimum, and below $5 commission
# eats too much of the trade.
KUBER_MIN_POSITION_USD = 5.00

# Per-sleeve position-count cap (avoid over-fragmentation).
KUBER_MAX_POSITIONS_PER_SLEEVE = 15

# Default per-position fraction of sleeve bankroll above the floor.
KUBER_FRACTION_OF_SLEEVE = 0.04


@dataclass(frozen=True)
class KuberConfig:
    """Runtime knobs for Kuber, mirroring `config.Config` fields.

    Kept as a separate dataclass so tests can construct one without
    needing a full DriftEdge env. The CLI bridges Config -> KuberConfig
    at startup.
    """
    bankroll_total_usd: float = 500.0
    bankroll_per_sleeve_usd: float = 125.0
    max_position_usd: float = 5.0
    dd_kill_pct: float = 0.40
    daily_loss_kill_usd: float = 25.0
    live: bool = False


# ── Sizing ───────────────────────────────────────────────────────────

def position_cap(sleeve_realized_pnl: float, kc: KuberConfig) -> float:
    """The max USD a Kuber sleeve will commit on a single position, given
    its realised PnL to date.

    Floor of `max_position_usd` (the $5 starting cap). Scales linearly
    with sleeve bankroll above that floor.
    """
    sleeve_bankroll = kc.bankroll_per_sleeve_usd + float(sleeve_realized_pnl)
    return max(kc.max_position_usd,
               KUBER_FRACTION_OF_SLEEVE * sleeve_bankroll)


def _raw_size(sleeve_kind: str, *, bankroll: float, c: float,
                target: float, stop: float, p_estimated: float) -> float:
    """The sleeve's *uncapped* raw recommendation, in USD.

    These reimplement the formulas from `sizing.py` but DROP the base
    caps (`MAX_SINGLE_EXPOSURE`, `MAX_TOTAL_EXPOSURE`, `MIN_POSITION_USD`)
    because Kuber has its own cap profile that overrides them. The base
    caps are designed for the $10k paper bankroll; at Kuber's $125
    per-sleeve scale they produce $2.50 sizes which fall below the
    base's own $5 floor — a contradiction.
    """
    if sleeve_kind == "kelly":
        if c <= 0 or c >= 1 or stop >= c or target <= c:
            return 0.0
        a = (target - c) / c
        b = (c - stop) / c
        if a <= 0 or b <= 0:
            return 0.0
        p = max(0.0, min(1.0, p_estimated))
        q = 1.0 - p
        f_star = (p * a - q * b) / (a * b)
        if f_star <= 0:
            return 0.0
        return bankroll * base_sizing.KELLY_KAPPA * f_star

    if sleeve_kind == "equal":
        # Equal weight: 4% of bankroll (Kuber's per-position fraction).
        return bankroll * KUBER_FRACTION_OF_SLEEVE

    if sleeve_kind == "volwt":
        if c <= 0 or c >= 1:
            return 0.0
        sigma = math.sqrt(c * (1.0 - c))
        if sigma <= 0:
            return 0.0
        scale = min(1.5, 0.5 / sigma)
        return bankroll * KUBER_FRACTION_OF_SLEEVE * scale

    if sleeve_kind == "volharvest":
        # Volharvest's own decision module passes its raw recommendation
        # in directly; this branch is a fallback when the wrapper sleeve
        # is invoked through the standard tick loop. Equal-weight is the
        # safest default until volharvest-as-a-Kuber-sleeve is fully
        # wired (uses its own should_open/should_close upstream).
        return bankroll * KUBER_FRACTION_OF_SLEEVE

    return 0.0


def kuber_size(sleeve_label: str, *, c: float, target: float, stop: float,
               state: base_sizing.TraderState,
               kc: KuberConfig,
               sleeve_realized_pnl: float,
               sleeve_open_positions: int,
               p_estimated: float = base_sizing.DEFAULT_P_ESTIMATED) -> float:
    """Return the USD to commit on this candidate, or 0 to skip.

    Pipeline:
      1. Compute the raw size for the sleeve's strategy (Kuber-scaled,
         no base caps).
      2. Cap by Kuber's per-position rule (`position_cap`).
      3. Cap by sleeve cash + per-sleeve position-count limit.
      4. Apply the $5 floor (Kuber's min trade size).
      5. Round to whole cents.
    """
    if sleeve_open_positions >= KUBER_MAX_POSITIONS_PER_SLEEVE:
        return 0.0

    sleeve_kind = sleeve_label.split(":", 1)[-1]
    sleeve_bankroll = kc.bankroll_per_sleeve_usd + float(sleeve_realized_pnl)
    raw = _raw_size(sleeve_kind, bankroll=sleeve_bankroll, c=c,
                     target=target, stop=stop, p_estimated=p_estimated)

    # Cap by Kuber's per-position rule (the explicit ceiling).
    raw = min(raw, position_cap(sleeve_realized_pnl, kc))
    # Cap by remaining cash.
    raw = min(raw, max(0.0, state.cash_usd))
    # Floor: below this the trade is not worth the fees.
    if raw < KUBER_MIN_POSITION_USD - 1e-9:
        return 0.0
    return round(raw, 2)


# ── Kill switches ────────────────────────────────────────────────────

def drawdown_kill_active(total_equity_usd: float, kc: KuberConfig) -> bool:
    """True if total Kuber equity has fallen below the drawdown floor."""
    floor = (1.0 - kc.dd_kill_pct) * kc.bankroll_total_usd
    return total_equity_usd < floor


def daily_loss_kill_active(today_realized_pnl_usd: float,
                            kc: KuberConfig) -> bool:
    """True if today's UTC realised PnL has breached the daily-loss floor."""
    return today_realized_pnl_usd <= -abs(kc.daily_loss_kill_usd)


@dataclass(frozen=True)
class GateDecision:
    """Why entries are (or aren't) allowed this tick."""
    allow_entries: bool
    reason: str


def gate(total_equity_usd: float,
         today_realized_pnl_usd: float,
         kc: KuberConfig,
         as_of_ts: Optional[str] = None) -> GateDecision:
    """Aggregate kill-switch decision for a tick.

    Order: drawdown first (terminal), then daily loss (resets next UTC
    day). When closed, exits are still allowed — only NEW entries are
    blocked.
    """
    if drawdown_kill_active(total_equity_usd, kc):
        return GateDecision(False, "drawdown_kill")
    if daily_loss_kill_active(today_realized_pnl_usd, kc):
        return GateDecision(False, "daily_loss_kill")
    return GateDecision(True, "ok")


# ── Helpers ──────────────────────────────────────────────────────────

def utc_date_of(ts_iso: str) -> str:
    """Return the YYYY-MM-DD UTC date of an ISO timestamp.

    Used by the daily-loss kill to bucket realised PnL by UTC day.
    """
    if ts_iso.endswith("Z"):
        ts_iso = ts_iso[:-1] + "+00:00"
    return datetime.fromisoformat(ts_iso).astimezone(timezone.utc).strftime("%Y-%m-%d")


def today_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
