# 4. No lookahead bias — architectural rule

Date: 2026-05-30

## Status

Accepted. Enforced for the entire lifetime of DriftEdge.

## Context

Lookahead bias is the single most common way backtests lie. A backtest
that accidentally reads a price from time T+1 when making a decision at
time T will report fictional P&L. Once that bias enters the codebase, it
is extremely hard to detect — the system "works," the numbers look great,
the strategy "wins" in backtest and loses in live trading.

DriftEdge's paper-trading and (future) backtesting code paths share the
same machinery as the live-decision path. We cannot rely on "it's live
so there's no future data" because:

1. Backtests replay the same functions over historical snapshots.
2. Snapshot files may contain rows with `snapshot_ts > now()` if
   timestamps are stored as the *response received* time but the data
   itself reflects a moment earlier — or vice versa.
3. Future contributors will add caches, joins, or denormalizations that
   may inadvertently surface future data.

The PinSight ChronoFund engine already proves this discipline works
when enforced at four independent layers. DriftEdge adopts the same
posture.

## Decision

1. **Every function that consults historical market data takes an
   explicit `as_of_ts: str` argument.** This argument represents the
   moment the decision is being made. The function MUST filter all
   reads to `snapshot_ts <= as_of_ts`.

2. **Decision functions are pure.** They take the relevant market state
   and the `as_of_ts` and return a decision. No side effects, no
   timestamps from `datetime.now()` inside the function.

3. **Assertions guard at every boundary.** When a `BookTop` flows into
   `open_position` or `close_position`, an `assert
   book.snapshot_ts <= as_of_ts` fires before any P&L math. Violations
   crash loudly rather than silently producing fake numbers.

4. **Tests prove invariance under future-data injection.** For each
   decision function, `tests/test_paper_no_lookahead.py` has a test
   that:
     a) Builds a synthetic data store with snapshots T₁ < T₂.
     b) Makes a decision at T₁ using the store.
     c) Adds snapshots T₃ > T₂ (future data).
     d) Makes the same decision at T₁ again.
     e) Asserts both decisions are identical.

5. **Persistence stamps `as_of_ts` on every row.** Paper-trade entries
   record both `entry_ts` (the decision time / as_of_ts) and
   `entry_snapshot_ts` (the snapshot the decision read). They may
   differ slightly but `entry_snapshot_ts <= entry_ts` always.

## Consequences

- More verbose function signatures — `as_of_ts` everywhere.
- Slight runtime cost from filtering. Negligible.
- Refactoring backtests requires explicitly passing `as_of_ts` through
  every layer; impossible to forget without the type/argument check
  failing loudly.
- New contributors must read this ADR before adding any function that
  touches market data. The convention is mandatory, not aspirational.

## Related

- PinSight's ChronoFund engine (4-layer lookahead prevention via
  `acceptance_datetime`) is the inspiration.
- ADR 0003 (price is not probability) ensures we don't conflate
  `c_market` with `p_estimated`; this ADR ensures the data feeding
  both is from the right time.
