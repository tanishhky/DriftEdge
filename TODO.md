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
