"""Three trader sizers — same entry/exit rules, different position sizing.

The framework:
  - Each trader has a starting bankroll, tracked cash, and tracked open
    exposure (sum of currently-open position sizes).
  - When a candidate trade arrives, each trader's sizer returns a USD
    amount (0 to skip the trade).
  - Per-position cap and aggregate exposure cap apply universally.
  - Sizers are pure functions — they consult bankroll + exposure + the
    candidate's market state, and return a number.

The three traders:
  1. KELLY: quarter-Kelly with conservative p_estimated default (0.45).
     Sized for edge; swap p_estimated for path-engine output when M2 ships.
  2. EQUAL: every trade gets the same fraction of bankroll (max_single).
     The "naive diversifier" baseline.
  3. VOLWT: inverse-Bernoulli-stddev weighted. Markets with lower
     variance get more capital. Risk-parity-style for binary contracts.

All three share the same hard caps (per-position max and aggregate max)
so blow-up risk is identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Shared hard limits (apply to every trader; configurable later).
MAX_SINGLE_EXPOSURE = 0.02   # 2% of bankroll per position
MAX_TOTAL_EXPOSURE = 0.50    # 50% of bankroll across all open positions
MIN_POSITION_USD = 5.00      # below this, fees would dominate
DEFAULT_P_ESTIMATED = 0.45   # used by Kelly until path engine ships
KELLY_KAPPA = 0.25           # fractional Kelly multiplier


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
# Each returns the USD size to commit (0 to skip).

def _apply_caps(size_usd: float, state: TraderState) -> float:
    """Apply per-position and aggregate caps, then min-size floor."""
    per_position_cap = state.bankroll_init * MAX_SINGLE_EXPOSURE
    size_usd = min(size_usd, per_position_cap)
    size_usd = min(size_usd, state.available_for_new)
    return size_usd if size_usd >= MIN_POSITION_USD else 0.0


def kelly_size(state: TraderState, *, c: float, target: float, stop: float,
               p_estimated: float = DEFAULT_P_ESTIMATED) -> float:
    """Quarter-Kelly with conservative p_estimated default.

    For long Yes at price c, target T, stop S:
        win_return  a = (T - c) / c
        loss_return b = (c - S) / c
        Kelly       f* = (p*a - q*b) / (a*b),  where q = 1 - p
        applied size = max(0, kappa * f*) * bankroll
    Then capped by per-position and aggregate exposure limits.
    """
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
    raw = state.bankroll_init * KELLY_KAPPA * f_star
    return _apply_caps(raw, state)


def equal_weight_size(state: TraderState, *, c: float, target: float,
                      stop: float, p_estimated: float = DEFAULT_P_ESTIMATED) -> float:
    """Fixed per-position fraction = MAX_SINGLE_EXPOSURE. Pure naive diversifier."""
    raw = state.bankroll_init * MAX_SINGLE_EXPOSURE
    return _apply_caps(raw, state)


def vol_weighted_size(state: TraderState, *, c: float, target: float,
                      stop: float, p_estimated: float = DEFAULT_P_ESTIMATED) -> float:
    """Inverse-Bernoulli-stddev weighting.

    For a Bernoulli(c), stddev = sqrt(c*(1-c)).
    Reference stddev at c=0.5 (max uncertainty) = 0.5.
    Weight = 0.5 / stddev(c), capped at 1.5 so we don't over-allocate
    extreme markets where the rule rarely fires.

    Result: markets closer to 0 or 1 get slightly MORE capital than
    markets near 0.5. The scaling is intentionally mild (factor < 1.5×).
    """
    if c <= 0 or c >= 1:
        return 0.0
    sigma = math.sqrt(c * (1.0 - c))
    if sigma <= 0:
        return 0.0
    scale = min(1.5, 0.5 / sigma)
    raw = state.bankroll_init * MAX_SINGLE_EXPOSURE * scale
    return _apply_caps(raw, state)


# Registry — keep adding new sizers here.
SIZERS = {
    "kelly":  kelly_size,
    "equal":  equal_weight_size,
    "volwt":  vol_weighted_size,
}


# Traders whose entry/exit doesn't fit the global EntryRule and are run by
# their own dedicated tick (see `driftedge.agents.*`). They still need state
# (bankroll, cash, equity) — so they appear in `all_trader_labels()` and get
# seeded by state_persist — but NOT in `SIZERS` / `trader_labels()`, which
# drives the standard paper.tick loop.
SELF_MANAGED_TRADERS: list[str] = ["volharvest", "resolution"]


def trader_labels() -> list[str]:
    """Traders managed by the standard `paper.tick` loop."""
    return list(SIZERS.keys())


def all_trader_labels() -> list[str]:
    """Every trader the platform knows about, for state-init + dashboards."""
    return list(SIZERS.keys()) + list(SELF_MANAGED_TRADERS)
