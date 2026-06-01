# DriftEdge TODO

Tracked work for the prediction-markets engine. See
`~/dev/Sentinel/CONTRACT-SPEC.md` for the contract this engine implements and
`docs/decisions/0004-no-lookahead-bias.md` for the lookahead rule that governs
every paper-engine change.

---

## Next up — contract slice + lookahead safety net

The contract is documented; this engine has to implement its side of it before
any further strategy work. Without it, the entry rule stays hardcoded in
`paper.py` and every experiment requires a code commit + restart.

- [ ] **`manifest/manifest.json`** declaring `engine_id=driftedge`,
      version, `departments=[predmarkets]`, `agents=[kelly, equal, volwt]`,
      `risk_limits` (per-position cap 2%, aggregate cap 50%, min trade $5),
      `schemas` (parquet shape for paper_trades / books / equity_history),
      `ui_tabs` (paper / markets / books / news / logs),
      `capabilities` (`kill_switch`, `dynamic_reallocation`).
- [ ] **`manifest_runtime.py`**: load manifest, expose to other modules.
- [ ] **config-driven entry rule**: move the hardcoded
      `EntryRule(entry_low=0.30, entry_high=0.40, target=0.60, stop=0.20,
      notional_usd=100, force_exit_hours=6)` into
      `data/config/entry_rule.json`. The poll daemon reads this each tick.
- [ ] **allocations poller**: every 30s, read `data/allocations.json`. If
      `allocations_version` increased, update active agents + per-agent
      budgets and write an audit row.
- [ ] **`state.json` writer**: every tick, write `data/state.json` with
      `last_tick_ts`, `kill_switch`, `allocations_version_loaded`,
      per-agent equity, current drawdown.
- [ ] **`allocation-audit.jsonl` writer**: append a row on every reallocation,
      enable/disable, pause/resume, kill_switch toggle. Schema per
      `CONTRACT-SPEC.md`.
- [ ] **kill-switch honored in `tick()`**: when `kill_switch=true`,
      `should_open()` returns False unconditionally (still closes, never
      opens). Test for this.

## Lookahead safety net (overdue)

`docs/decisions/0004-no-lookahead-bias.md` says every paper-trade repo must
ship a test that proves invariance under future-data injection. Need to
confirm this test exists and is in CI.

- [ ] **audit the test**: does `tests/test_paper_no_lookahead.py` actually
      inject T+k snapshots and assert the past decision is unchanged?
      If not, write it.
- [ ] **assertion in `latest_book_top`**: already exists. Add one in
      `should_close()` / `should_open()` too — they receive `book` and
      `as_of_ts`, so `assert book.snapshot_ts <= as_of_ts`.
- [ ] **fuzz test**: random sequence of past + future snapshots, replay,
      check that paper_trades.parquet is bit-identical to the past-only run.

## Engine candidate: cross-side hedge (volatility-harvest)

Not arbitrage, but a real positive-EV structure that the current
`[0.30, 0.40] → 0.60` rule throws away. Worth building as a separate agent
once the contract slice is in.

**Idea (binary two-outcome market — sports, election head-to-head):**

1. Buy the underdog YES at price `c_dog` (e.g. NYK at 0.21).
2. Hold. If at any point during the market's life the favourite's YES ask
   drops below `1 − c_dog − ε` (i.e., the underdog's price rises enough),
   buy the favourite YES to hedge.
3. Locked profit = `1 − c_dog − c_fav_at_hedge − fees`.
4. If the hedge never fires, you end up with a directional bet at the
   original underdog price; EV ≈ 0 if the market was efficient at entry.

**Why this works:**
- Prediction markets price the settlement probability; intra-game / intra-
  market path variance is a free option you embed by buying early and
  hedging late.
- Equivalent to "selling realised variance" on the price path.

**Why it can fail:**
- Spread crossed twice (bid-ask × 2).
- Slippage when the hedge-trigger price is brief or thin.
- Capital lockup for hours / days.
- Markets where path variance is low (long-dated political / macro
  resolution) — `q` (hedge-fire probability) is too low to pay for the
  capital + spread cost.

**Pre-reqs before building:**

- [ ] Contract slice landed (so this is a new agent under the
      `predmarkets` department with its own budget, not a code fork of
      `paper.py`).
- [ ] Historical replay infrastructure: simulate the strategy over the
      orderbook archive we already have for finished markets. Compute
      realized `q` and avg locked profit per category.
- [ ] Category filter: only enable on high-vol categories (sports first,
      then short-dated political head-to-heads).

**Tasks once pre-reqs are met:**

- [ ] `agents/hedge_volharvest.py`: new agent class. Two-leg position
      state (leg1_open, leg2_open). Entry rule: buy underdog when
      `c_dog ∈ [entry_lo, entry_hi]`. Trigger rule: hedge when
      `c_fav ≤ trigger`. Force-exit at resolution.
- [ ] Replace the implicit `p_estimated` confusion: this agent does NOT
      need an estimate of true `p` — the EV comes from path variance,
      not from a probability edge. Document this in
      `docs/decisions/0005-volharvest-strategy.md`.
- [ ] Per-leg risk caps: max two open legs per market, max aggregate
      exposure per market.

## Engine refinement (only after the above is in)

The current rule (open at ask in `[0.30, 0.40]`, target 0.60, stop 0.20) is a
losing strategy in production (~25% hit rate, EV ≈ −$15/trade). These tasks
attack that — but they belong in config files, not in code, once the contract
slice is wired.

- [ ] **honest `p_estimated`**: today Kelly uses `p=0.45` constant. Replace
      with a real estimate per market category — at minimum, historical
      mean closing price by `(category, days-to-resolution bucket)`.
      Document the source in `docs/decisions/0003-price-is-not-probability.md`.
- [ ] **per-category entry rule**: politics / sports / crypto behave
      differently. Allow `entry_rule.json` to nest rules under a category key.
- [ ] **diagnose the 25% hit-rate**: per-category hit rate is already in the
      paper summary. Use it to identify which categories should be disabled
      via allocations (not via code removal).
- [ ] **Kalshi closed-trade visibility**: closed positions are mostly
      Polymarket. Confirm Kalshi exits are being detected — could be a force-
      exit window bug, a resolution-ts parsing issue, or no markets passed
      the entry filter on Kalshi.

## Smaller fixes

- [ ] **archive `equity_history.parquet`**: today it grows unbounded. Trim
      job at 500k rows already exists in `equity_persist.trim()`; wire it
      into the poll daemon (run once per hour).
- [ ] **classifier review queue**: count any remaining low-confidence markets
      and decide them with `driftedge set-manual <market-id> <category>` so
      future ticks classify without ambiguity.
- [ ] **README**: update endpoint expectations now that Sentinel reads
      `equity_history.parquet` directly.

## Done (recent)

- [x] Continuous MTM equity snapshots per tick (`equity_persist.py`)
- [x] Cross-venue keying (`(trader, venue, market_id)`)
- [x] Rule-based persistent classifier + review queue
- [x] News subsystem with VADER sentiment (RSS + GDELT + Reddit)
- [x] Polymarket + Kalshi adapters in the polling daemon
