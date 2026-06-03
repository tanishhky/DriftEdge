# Architecture & Design

Living document. Decisions land as ADRs in `docs/decisions/`.

---

## 1. Three engines, one signal

DriftEdge mirrors PinSight's structure: three engines that run on the same data feed, fused by a thin signal layer.

```
        ┌──────────────────┐
data ──▶│  Path Engine     │──▶ price-path features (drift, vol, momentum)
        ├──────────────────┤
        │  Flow Engine     │──▶ anomaly score (informed-flow proxy)
        ├──────────────────┤
        │  Sizing Engine   │──▶ Kelly fraction given p_estimated
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  Signal Layer    │──▶ entry / exit / stop + position size
        └──────────────────┘
```

Compared to PinSight: there's no RND engine because the "implied distribution" of a prediction-market contract is a single Bernoulli with parameter equal to the price. The "path" replaces the "distribution" — we care about how that single number drifts and accelerates through time.

---

## 2. Path Engine

**Goal:** For each market, maintain a feature vector describing the recent probability path.

**Inputs:**
- Time-series of mid prices `p_t` from orderbook snapshots
- Time-series of trade prices and sizes
- Resolution date `T_resolve`

**Features (per market, computed per snapshot):**

| Feature | Definition |
|---|---|
| `price_now` | Current mid (= implied probability) |
| `price_24h_change` | `p_t − p_{t−24h}` |
| `price_7d_change` | `p_t − p_{t−7d}` |
| `realized_vol_24h` | std of `Δlog(p/(1−p))` (logit-vol, not raw) over 24h window |
| `momentum_z` | z-score of 24h change against the trailing 14-day distribution of 24h changes |
| `path_distance_to_entry` | `entry_low − price_now` (negative if already in entry zone) |
| `path_distance_to_target` | `exit_target − price_now` |
| `time_to_resolution` | `T_resolve − t_now`, in hours |
| `time_decay_pressure` | `time_to_resolution / 24h_change` — how fast the path is moving relative to time remaining |

**Why logit-vol not raw-vol:** prices are bounded in `[0, 1]`. Raw vol underestimates moves near 0 or 1 (because they can't go further) and overestimates moves near 0.5. Working on `logit(p) = log(p/(1−p))` puts the price on the real line where standard time-series tools behave correctly.

---

## 3. Flow Engine

**Goal:** Score each market on a 0–1 anomaly scale. High score = likely informed flow agreeing with the path.

**Features per market:**

1. **Trade volume z-score** — z-score of 1h volume vs trailing 14-day distribution of 1h volumes.
2. **Order book imbalance** — `(bid_depth_total − ask_depth_total) / (bid_depth_total + ask_depth_total)` at multiple depth levels (top-5, top-20).
3. **Large-trade indicator** — fraction of 1h volume coming from trades larger than the 95th percentile of trailing trade sizes.
4. **Aggressor flow** — for each trade, classify as buy-initiated or sell-initiated based on which side of the mid it crossed; track 1h net buy volume.
5. **Cross-market correlation** — for markets in the same topic family (e.g., multiple election state-level markets), is flow coordinated? High correlation suggests broad news event.
6. **Path-flow agreement** — are flow features aligned with the recent price drift? If price up + buy flow + book bid-heavy: confirming. Mixed: noise.

**Composite anomaly score:**

`score = max-of-top-2( sigmoid_k(z_feature_k) )` — same max-of-top-2 pattern from PinSight's flow engine. Requires at least two features to agree before firing.

---

## 4. Sizing Engine

**Goal:** Given path/flow output and an estimated `p`, compute an actionable position size.

### CRITICAL: price ≠ probability

The market price `c` is NOT a clean estimate of the true probability `p`. They diverge for at least nine documented reasons:

| Divergence source | Typical magnitude | Treatment |
|---|---|---|
| Bid-ask spread | 1–5¢ liquid, much wider thin | Use mid; reject if spread > 4¢ |
| Trading fees (round-trip) | Kalshi ~$0.07/contract; Polymarket 2% on resolution gains | Subtract from expected payoff |
| Withdrawal / on-chain costs (Polymarket) | Variable | Amortize per trade |
| Favorite-longshot bias | 1–8% (Snowberg-Wolfers 2010) | Empirical correction curve when populated |
| Risk aversion | Hard to quantify | Live with it (creates persistent edge for risk-neutral systematic trader) |
| Manipulation / informed flow | shocks persist 60d (arXiv 2503.03312) | This is what the flow engine *exploits* |
| Liquidity / depth | Thin books drift from true p | Volume-weighted confidence in our estimate |
| Oracle / settlement ambiguity | Rare but catastrophic | Skip markets with known oracle disputes |
| Time-value of money (long-dated) | r·T | Apply discount factor for markets > 30d |

**Implication:** the engine must NEVER use `c` as `p`. They are distinct inputs. If we ever wrote `p = mid_price`, the edge is mechanically zero and Kelly returns no trade. The whole reason path + flow + (eventual) external-data engines exist is to produce a `p_estimated` that *disagrees* with `c` in a defensible way.

### Step 1: estimate `p_estimated`

Options for producing `p_estimated`:
- **Path-based:** project recent drift forward, with Brownian-bridge correction toward 0.5 if resolution is far
- **Flow-based shrinkage:** if flow strongly agrees with a side, shrink `p_estimated` toward that side; if flow is balanced, *do not trade* — we have no information beyond the market
- **External-data-based:** sports models, poll aggregations, ELO ratings — out of scope for v0

For v0: blend `path_estimate` and `flow_estimate` with weights determined by confidence. **If both reduce to `c`, the system passes — no trade.**

### Step 2: compute corrected expected value

```
fee_cost          = expected_round_trip_fees(c, side, venue)
slippage_cost     = expected_slippage(orderbook_depth, intended_size)
discount_factor   = exp(-r · time_to_resolution_yrs)    # negligible for short markets

expected_payoff_yes = discount_factor · 1.0            # if we win $1
expected_payoff_no  = 0.0                              # if we lose, contract worth 0

corrected_edge = p_estimated · expected_payoff_yes
               + (1 - p_estimated) · expected_payoff_no
               - c
               - fee_cost
               - slippage_cost
```

Only proceed if `corrected_edge > 0`.

### Step 3: compute Kelly fraction

```
f* = (p_estimated - c) / (1 - c)         # standard prediction-market Kelly
                                          # (assumes unit payoff and uses RAW c, not fee-adjusted —
                                          #  fees are handled in the size cap below)
```

If `f*` is negative, don't trade.

### Step 4: apply fractional dampening and caps

```
f = κ · f*                               # κ = 0.25 default (quarter-Kelly)
f = min(f, max_single_market_exposure)   # default 5% of bankroll
```

Skip if `f · bankroll < min_trade_usd` (default $5) or if `corrected_edge < min_edge_threshold` (default 1¢).

**Output:** `position_size_usd`, `entry_price`, `target_exit_price`, `stop_price`, `p_estimated`, `corrected_edge`.

---

## 5. Signal Layer

**Inputs:**
- Path features for the market
- Flow anomaly score
- Sizing output

**Initial entry rule (handwritten, replace with calibrated model later):**

```
if (entry_low ≤ price_now ≤ entry_high)
   AND (path.momentum_z > +0.5)            # price has been rising
   AND (flow.anomaly_score > 0.7)          # informed flow agrees
   AND (sizing.f > 0):                     # Kelly says trade
    side = 'long Yes'
    size = sizing.f
    target = exit_target
    stop = stop_low
    conviction = flow.anomaly_score * sizing.f
else:
    side = 'pass'
```

**Exit rule:**

```
if price_now ≥ exit_target:                # take profit
    exit_signal
elif price_now ≤ stop_low:                 # stop loss
    exit_signal
elif flow.anomaly_score < 0.3:             # flow reversed
    exit_signal (de-risk)
elif time_to_resolution < 6 hours:         # avoid event variance
    exit_signal (force-exit before resolution)
```

Every signal is logged to `logs/signals.jsonl` regardless of whether it fires. This is the dataset for replacing the rule with a calibrated model later.

---

## 5b. Paper-trading layer (multi-trader, shipped)

A lookahead-safe paper-trading engine runs on every poll iteration. Lives in `src/driftedge/paper.py`, persists to `data/paper_trades.parquet` and `data/paper_equity_history.parquet`.

**Five active traders** (each seeded at $10k bankroll):

| Trader | Managed by | Sizing strategy |
|---|---|---|
| `kelly` | `paper.tick` | Quarter-Kelly (κ=0.25, p_estimated=0.45) |
| `equal` | `paper.tick` | Fixed 2% bankroll per trade |
| `volwt` | `paper.tick` | Inverse-Bernoulli-stddev weight, capped 1.5× |
| `volharvest` | `agents/volharvest.py` | Underdog YES + synthetic-NO hedge |
| `resolution` | `agents/resolution.py` | Hold-to-binary; entry [0.25, 0.50]; ≤72h horizon |

Standard traders (`kelly`, `equal`, `volwt`) run inside `paper.tick` on every poll iteration. Self-managed traders (`volharvest`, `resolution`) have their own tick function wired after `paper.tick` in `cli.py`. All five are seeded via `state_persist.init_state()` which auto-backfills missing traders on every call.

**Resolution agent** — hold-to-binary strategy:
- Entry: best_ask ∈ [0.25, 0.50], ≤72h to resolution
- Sizing: scale = 1.0 + 0.5 × (1 − hours/72), capped at 1.5× — time-weighted toward expiry
- Dynamic stop: exit when `best_bid − entry ≤ −0.15` and `hours_remaining ≤ 6`
- Force-exit: 1h before resolution, no exceptions

**Decision rule (standard traders)** — for each tracked market on each tick:
- **Open** a paper-long Yes position when `entry_low ≤ best_ask ≤ entry_high` AND the market is not within `force_exit_hours_before_resolution`.
- **Close** when `best_bid ≥ target` (take profit) OR `best_bid ≤ stop` (stop loss) OR `time_to_resolution < force_exit_hours_before_resolution` (force exit before event variance).

**Near-certain filter** — before tracking, both Polymarket and Kalshi markets where `best_ask ≤ 0.05` or `best_ask ≥ 0.95` are dropped (no entry opportunity, no meaningful MTM signal).

**Equity history schema** (`data/paper_equity_history.parquet`):
```
trader, ts, total_equity_usd, cash_usd, open_exposure_usd,
closed_pnl_usd, drawdown_pct, mtm_unrealized_usd
```

**Strict no-lookahead** — see ADR 0004:
- All snapshot reads filter `snapshot_ts <= as_of_ts`.
- Decision/lifecycle functions take `as_of_ts` explicitly.
- Assertions in `open_position` and `close_position` raise if `book.snapshot_ts > as_of_ts`.
- `tests/test_paper_no_lookahead.py` proves that injecting future-dated data into the store does not change past decisions.

---

## 6. Logging discipline

Same as PinSight: every fetch, fit, score, signal goes to JSONL with timestamp, channel, duration, status. The `obs` module is copied/adapted from PinSight directly.

Channels:
- `api` — every Polymarket request
- `persist` — every Parquet write
- `fit` — every path/flow score computation
- `signal` — every entry/exit decision
- `run` — process lifecycle
- `error` — anything that fails

---

## 7. Storage

Parquet with typed schemas, partitioned where useful.

**Schemas (planned):**

- `data/markets/<YYYY-MM-DD>.parquet` — daily snapshot of active markets (id, slug, end_date, category)
- `data/books/<market_id>/<YYYY-MM-DD>.parquet` — orderbook snapshots (one row per snapshot time, with bid/ask levels as struct columns)
- `data/trades/<market_id>/<YYYY-MM-DD>.parquet` — individual trades (price, size, side, timestamp)
- `data/paths/<market_id>.parquet` — long-running path-feature time-series, append-only
- `data/signals.parquet` — every entry/exit signal ever fired, append-only

---

## 8. Testing strategy

- **Unit tests** for math primitives: Kelly formula edge cases (negative edge, `c = 0` or `1`), logit-vol on bounded prices, large-trade percentile.
- **Replay tests** for the flow engine: take a known historical event (e.g., a court ruling resolved a market in days), feed in the trade history, verify the detector fires on the right side.
- **Synthetic-trace tests** for the path engine: generate a known logit-Brownian path, verify the engine recovers the drift and vol.
- **Live integration smoke**: hit the Polymarket public API in CI weekly. If their schema changes, we want to know.

---

## 9. What we're explicitly NOT doing in v0

- No order placement. Signal is logged, not routed. Manual execution only.
- No on-chain interaction. All read endpoints are off-chain HTTP.
- No multi-platform until M6. Polymarket only through M5.
- No domain-specific `p` estimators (sports/polls/etc.). The v0 sizer uses a conservative blended estimate.
- No machine-learning model for signal generation until we have ≥500 labeled signals from the rule-based fusion layer.
