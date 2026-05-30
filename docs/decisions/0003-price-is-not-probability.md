# 3. Market price is not an unbiased probability estimate

Date: 2026-05-30

## Status

Accepted.

## Context

A naive reading of prediction markets — encouraged by every popular-press
article — is "the price IS the probability." This is the single most
dangerous conceptual error a sizing engine can make. If we treat `c =
mid_price` as `p_true`, then the Kelly formula `f* = (p - c) / (1 - c)`
returns zero for every market and we never trade. Worse, if we trade on
flow signals without distinguishing `c` from our model's `p_estimated`,
we are systematically miscomputing edge.

Reasons `c` ≠ `p_true`:

1. **Bid-ask spread.** "The price" is undefined; mid is a convention.
2. **Trading fees.** Kalshi ~$0.07 round-trip per contract; Polymarket
   takes 2% of resolution gains on many markets.
3. **Withdrawal / conversion frictions.** USDC ↔ USD on Polymarket;
   bank wires on Kalshi. Amortized per trade for small accounts.
4. **Favorite-longshot bias.** Empirically documented at 1–8% magnitude
   in betting and prediction markets (Snowberg & Wolfers 2010).
5. **Risk aversion.** Traders demand a premium to hold risky payoffs.
   The bias is structural and hard to invert.
6. **Manipulation / informed flow.** arXiv 2503.03312 shows shocks
   persist 60 days. Until shocks decay, price reflects manipulation,
   not truth — this is what the flow engine is *built to detect*.
7. **Liquidity / depth.** Thin books wander from true probability under
   light news flow.
8. **Oracle / settlement ambiguity.** Rare but catastrophic; the
   resolution mechanism itself has nonzero uncertainty.
9. **Time value of money.** For long-dated markets, settled $1 is
   worth less than $1 today.

## Decision

The codebase must always treat `c` (market price) and `p_estimated` (our
modeled probability) as distinct variables. Naming convention:

- `c` or `mid_price` — observed market quantity
- `p_estimated`, `p_true_hat`, or `p_model` — output of an engine

Functions that compute Kelly or edge MUST take `p_estimated` as an
explicit argument; they must NOT have a fallback to `mid_price` when
`p_estimated` is missing. If the engine has no `p_estimated`, the system
passes — no trade.

The sizing engine corrects expected value for fees, slippage, and
discounting before evaluating the edge. The Kelly fraction itself uses
the raw `(p_estimated, c)` pair, with fees and slippage handled via
size caps and minimum-edge thresholds (so they cut weak edges first).

## Consequences

- Every signal log entry records `c`, `p_estimated`, `corrected_edge`,
  and the engine that produced `p_estimated`. We can audit later whether
  any engine's estimate was systematically wrong.
- No "shortcut path" exists where `p_estimated = c`. Attempting it makes
  the system silent — clean failure mode.
- Adding new `p_estimated` sources (e.g., sports models in M7+) is
  straightforward: implement a function returning `p_estimated`, plug
  into the blend.
- The favorite-longshot bias correction is a stretch goal — populating
  it requires labeled historical data we'll accumulate during M1–M5.
