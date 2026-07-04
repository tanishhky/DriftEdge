"""Empirical-Bayes calibration of the target/stop bet from realized history.

Replaces the hardcoded `p_estimated = 0.45` the Kelly sizer shipped with
(2026-07-04). That constant was not an estimate of anything: it asserted a
flat 45% win rate for every market, every category, forever - and the
realized standard-trader hit rate was ~30-37%, so Kelly sized for an edge
that did not exist.

The formulation, no magic probabilities:

Prior - for a driftless (martingale) price, the probability that a long
YES entered at price c hits target T before stop S is the gambler's-ruin
ratio

    p0 = (c - S) / (T - S)

This is the market-implied win probability OF THE BET. It is derived from
c but is not c: the OUTCOME probability and the BET-winning probability
are different quantities (ADR 0003, price != probability). Under this
prior with the design payoffs, Kelly's expected value is exactly zero -
i.e., with no evidence of edge the sizer refuses to bet. That is the
correct cold-start behavior.

Evidence - realized closed trades from the standard entry/exit engine
whose entry price lies within `band` of c, same category when enough of
them exist. Wins/losses update a Beta prior centred on p0 with
pseudo-count `prior_strength`:

    p_hat = (p0 * n0 + wins) / (n0 + n)

Payoffs - the design ratios a0 = (T-c)/c and b0 = (c-S)/c are known to be
optimistic on the loss side: binary markets gap through the stop when the
underlying event goes against the position (designed -40% stops filled at
-80% in the 2026-06-30 sweep). Both payoff sides are therefore estimated
as shrunk means of realized per-trade returns around the design priors.

No lookahead: only trades with exit_ts <= as_of_ts count as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .paper import parse_iso


# Traders whose history is valid evidence for the standard target/stop bet.
# volwt is retired but its realized trades were produced by the exact same
# entry/exit rules, so they remain usable evidence.
STANDARD_EVIDENCE_TRADERS = ("kelly", "equal", "volwt")


@dataclass(frozen=True)
class BetStats:
    """Posterior estimate of one (entry-band, category) bet's parameters."""
    p_hat: float      # posterior P(hit target before stop)
    a_hat: float      # estimated win return per $ staked
    b_hat: float      # estimated loss magnitude per $ staked (positive)
    p0: float         # martingale prior for reference/logging
    n_evidence: int   # realized trades behind the posterior
    category_matched: bool  # False when we had to pool across categories

    @property
    def ev_per_usd(self) -> float:
        """Expected value per $ staked under the posterior."""
        return self.p_hat * self.a_hat - (1.0 - self.p_hat) * self.b_hat


def design_priors(*, c: float, target: float,
                  stop: float) -> tuple[float, float, float]:
    """(p0, a0, b0) implied by the rule geometry alone."""
    p0 = (c - stop) / (target - stop)
    a0 = (target - c) / c
    b0 = (c - stop) / c
    return p0, a0, b0


def prior_only(*, c: float, target: float, stop: float) -> BetStats:
    """BetStats with zero evidence - EV is exactly 0 by construction."""
    p0, a0, b0 = design_priors(c=c, target=target, stop=stop)
    return BetStats(p_hat=p0, a_hat=a0, b_hat=b0, p0=p0,
                    n_evidence=0, category_matched=False)


def bet_stats(closed_trades: list[dict], *, c: float, target: float,
              stop: float, category: str, as_of_ts: str,
              band: float = 0.05,
              prior_strength: float = 20.0,
              payoff_prior_strength: float = 5.0,
              min_category_evidence: int = 10) -> BetStats:
    """Posterior bet parameters for an entry at price c.

    `closed_trades` is the already-loaded paper_trades row list (dicts) -
    callers pass what they have in hand so this never re-reads parquet
    inside the tick loop.
    """
    if not (0.0 < stop < c < target < 1.0):
        return prior_only(c=c, target=target, stop=stop)

    p0, a0, b0 = design_priors(c=c, target=target, stop=stop)
    as_of_dt = parse_iso(as_of_ts)

    def _collect(require_category: bool) -> list[tuple[bool, float]]:
        """[(won, |return per $|), ...] for matching realized trades."""
        out: list[tuple[bool, float]] = []
        for t in closed_trades:
            if t.get("status") == "open":
                continue
            if (t.get("trader") or "") not in STANDARD_EVIDENCE_TRADERS:
                continue
            if require_category and (t.get("category") or "other") != category:
                continue
            entry = t.get("entry_price")
            size = t.get("entry_size_usd")
            pnl = t.get("pnl_usd")
            exit_ts = t.get("exit_ts")
            if entry is None or size is None or pnl is None or not exit_ts:
                continue
            try:
                if parse_iso(str(exit_ts)) > as_of_dt:
                    continue  # not realized yet as of the decision time
                entry_f, size_f, pnl_f = float(entry), float(size), float(pnl)
            except (TypeError, ValueError):
                continue
            if size_f <= 0 or pnl_f != pnl_f:  # NaN guard
                continue
            if abs(entry_f - c) > band:
                continue
            ret = pnl_f / size_f
            out.append((ret > 0, abs(ret)))
        return out

    evidence = _collect(require_category=True)
    category_matched = True
    if len(evidence) < min_category_evidence:
        evidence = _collect(require_category=False)
        category_matched = False

    n = len(evidence)
    wins = [r for won, r in evidence if won]
    losses = [r for won, r in evidence if not won]

    p_hat = (p0 * prior_strength + len(wins)) / (prior_strength + n)
    a_hat = ((a0 * payoff_prior_strength + sum(wins))
             / (payoff_prior_strength + len(wins)))
    b_hat = ((b0 * payoff_prior_strength + sum(losses))
             / (payoff_prior_strength + len(losses)))

    return BetStats(p_hat=p_hat, a_hat=a_hat, b_hat=b_hat, p0=p0,
                    n_evidence=n, category_matched=category_matched)
