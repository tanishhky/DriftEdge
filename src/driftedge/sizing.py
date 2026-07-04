"""Trader sizers — same entry/exit rules, different position sizing.

The framework:
  - Each trader has a starting bankroll, tracked cash, and tracked open
    exposure (sum of currently-open position sizes).
  - When a candidate trade arrives, each trader's sizer returns a USD
    amount (0 to skip the trade).
  - Per-position cap and aggregate exposure cap apply universally.
  - Sizers are pure functions — they consult bankroll + exposure + the
    candidate's market state (+ a calibration posterior), and return a
    number.

Active traders (2026-07-04 roster):
  1. KELLY: fractional Kelly sized from the empirical-Bayes calibration
     posterior (see `calibration.py`). Refuses any trade whose posterior
     EV <= 0 — with no evidence of edge it does not bet. This replaced
     the hardcoded p_estimated=0.45, which asserted a permanent 5pp edge
     over the martingale prior that realized data never supported.
  2. EQUAL: every trade gets the same fraction of bankroll (max_single).
     The "naive diversifier" baseline — deliberately NOT gated on the
     calibration posterior, so it keeps producing the unconditional
     evidence stream that calibration learns from.

Retired traders (2026-07-04, history preserved in paper_trades.parquet):
  - VOLWT: inverse-Bernoulli-stddev weighting degenerated to equal sizing
    in practice — its entire trade record was byte-identical to EQUAL's
    (the mild <=1.5x scale always saturated the same caps). Running it was
    paying storage and attention for a duplicate.
  - RESOLUTION: bought YES on any [0.25,0.50] near-resolution market with
    no probability estimate at all (price != probability, ADR 0003);
    realized -$4.2k before the 2026-06-18 quarantine. Retired outright.

All sizing limits are env-overridable (no in-code retuning needed).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

from .calibration import BetStats, prior_only


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Shared hard limits (apply to every trader).
MAX_SINGLE_EXPOSURE = _env_float("DRIFTEDGE_MAX_SINGLE_EXPOSURE", 0.02)
MAX_TOTAL_EXPOSURE = _env_float("DRIFTEDGE_MAX_TOTAL_EXPOSURE", 0.50)
MIN_POSITION_USD = _env_float("DRIFTEDGE_MIN_POSITION_USD", 5.00)
KELLY_KAPPA = _env_float("DRIFTEDGE_KELLY_FRACTION", 0.25)


@dataclass(frozen=True)
class TraderState:
    """Snapshot of a trader's portfolio state at a moment in time."""
    trader: str
    bankroll_init: float
    cash_usd: float
    open_exposure: float
    closed_pnl: float

    @property
    def total_equity(self) -> float:
        """Cash + at-cost open exposure + realized P&L.
        Marking-to-market would require current bids on open positions;
        we use entry cost here for simplicity (conservative)."""
        return self.cash_usd + self.open_exposure

    @property
    def available_for_new(self) -> float:
        """How much more we're allowed to commit before hitting the
        aggregate exposure cap, given current state."""
        cap = self.bankroll_init * MAX_TOTAL_EXPOSURE
        return max(0.0, cap - self.open_exposure)


# ── Sizer functions ──────────────────────────────────────────────────────
#
# Each returns the USD size to commit (0 to skip). All accept the shared
# calibration posterior (may be None; only Kelly uses it).

def _apply_caps(size_usd: float, state: TraderState) -> float:
    """Apply per-position cap, aggregate-exposure cap, cash floor, then
    min-size floor.

    The cash floor (added 2026-06-05) prevents opening positions that
    would push cash negative when the trader has accumulated large
    realized losses but exposure-cap headroom remains. Without it,
    `available_for_new = agg_cap − open_exposure` could exceed cash, the
    sizer returns a positive number, and `apply_open` subtracts more than
    cash holds.
    """
    per_position_cap = state.bankroll_init * MAX_SINGLE_EXPOSURE
    size_usd = min(size_usd, per_position_cap)
    size_usd = min(size_usd, state.available_for_new)
    size_usd = min(size_usd, max(0.0, state.cash_usd))
    return size_usd if size_usd >= MIN_POSITION_USD else 0.0


def kelly_size(state: TraderState, *, c: float, target: float, stop: float,
               calib: Optional[BetStats] = None) -> float:
    """Fractional Kelly from the calibration posterior.

    With win prob p, win return a, loss magnitude b (all per $ staked):
        EV  = p*a - q*b          (q = 1 - p)
        f*  = EV / (a*b)
        size = kappa * f* * bankroll,  0 when EV <= 0.

    With calib=None the martingale prior applies, whose EV is exactly 0 —
    no evidence, no bet. This is intentional: the sizer only deploys when
    realized history says the (band, category) bet has positive
    expectancy, and b_hat reflects the REAL gap-through-stop losses, not
    the designed stop distance.
    """
    if c <= 0 or c >= 1 or stop >= c or target <= c:
        return 0.0
    if calib is None:
        calib = prior_only(c=c, target=target, stop=stop)
    a, b = calib.a_hat, calib.b_hat
    if a <= 0 or b <= 0:
        return 0.0
    ev = calib.ev_per_usd
    if ev <= 0:
        return 0.0
    f_star = ev / (a * b)
    raw = state.bankroll_init * KELLY_KAPPA * f_star
    return _apply_caps(raw, state)


def equal_weight_size(state: TraderState, *, c: float, target: float,
                      stop: float, calib: Optional[BetStats] = None) -> float:
    """Fixed per-position fraction = MAX_SINGLE_EXPOSURE. Pure naive
    diversifier and the calibration evidence generator — takes every
    rule-qualified trade unconditionally."""
    raw = state.bankroll_init * MAX_SINGLE_EXPOSURE
    return _apply_caps(raw, state)


def vol_weighted_size(state: TraderState, *, c: float, target: float,
                      stop: float, calib: Optional[BetStats] = None) -> float:
    """RETIRED 2026-07-04 — kept for reference and old tests only.

    Inverse-Bernoulli-stddev weighting: weight = 0.5 / sqrt(c*(1-c)),
    capped at 1.5x. In the [0.30, 0.40] entry band the scale lives in
    [1.02, 1.09] and the per-position cap flattens it entirely, so this
    produced a trade record byte-identical to EQUAL's. Not registered in
    SIZERS.
    """
    if c <= 0 or c >= 1:
        return 0.0
    sigma = math.sqrt(c * (1.0 - c))
    if sigma <= 0:
        return 0.0
    scale = min(1.5, 0.5 / sigma)
    raw = state.bankroll_init * MAX_SINGLE_EXPOSURE * scale
    return _apply_caps(raw, state)


# Registry — active standard traders only.
SIZERS = {
    "kelly":  kelly_size,
    "equal":  equal_weight_size,
}


# Traders whose entry/exit doesn't fit the global EntryRule and are run by
# their own dedicated tick (see `driftedge.agents.*`). They still need state
# (bankroll, cash, equity) — so they appear in `all_trader_labels()` and get
# seeded by state_persist — but NOT in `SIZERS` / `trader_labels()`, which
# drives the standard paper.tick loop.
SELF_MANAGED_TRADERS: list[str] = ["volharvest"]

# Retired traders: no ticks, no new positions. Their state rows and trade
# history stay on disk (and remain calibration evidence where applicable).
# Retire via `driftedge retire-trader <name>` which liquidates open
# positions first.
RETIRED_TRADERS: list[str] = ["volwt", "resolution"]


def trader_labels() -> list[str]:
    """Traders managed by the standard `paper.tick` loop."""
    return list(SIZERS.keys())


def all_trader_labels() -> list[str]:
    """Every ACTIVE trader, for state-init + dashboards."""
    return list(SIZERS.keys()) + list(SELF_MANAGED_TRADERS)
