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

**Step 1: estimate `p`.** This is the hardest step. Options:
- **Path-based estimate:** project recent drift forward to resolution (works for stable trends)
- **Flow-based shrinkage:** if flow strongly agrees, shrink `p` toward the side flow indicates; if mixed, anchor `p` to current price (no edge)
- **External-data-based:** for sports/political markets, plug in domain models (polls, ELO ratings). Out of scope for v0.

For v0: a simple weighted average of (current price) and (flow-implied probability), tuned to be conservative.

**Step 2: compute Kelly fraction.**
```
f* = (p − c) / (1 − c)        # for buying Yes at price c
```
If `f*` is negative, don't trade.

**Step 3: apply fractional dampening.**
```
f = κ · f*    where κ = 0.25  (quarter-Kelly default; configurable)
```

**Step 4: apply caps.**
- Max single-market exposure: `f ≤ 0.05` (5% of bankroll per market regardless of Kelly)
- Min trade size: skip if `f · bankroll < $5`
- Slippage adjustment: reduce by expected slippage given orderbook depth

**Output:** `position_size_usd`, `entry_price`, `target_exit_price`, `stop_price`.

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
