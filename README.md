# DriftEdge

**A free, open research platform for prediction markets: probability-path inference, flow-anomaly detection, Kelly-sized path trades.**

DriftEdge studies prediction markets the way [PinSight](https://github.com/tanishhky/PinSight) studies 0DTE options. The thesis: a prediction market contract is a binary option whose price equals its implied probability. We can therefore reuse most of the options-market machinery (flow detection, microstructure analysis, Kelly sizing) and add what's unique to prediction markets — **the probability path through time**.

The trade we are trying to find:

> Buy at implied probability **0.36**, exit at **0.60**, *before* the event resolves.

That's a 24-cent move on a 36-cent ticket (+67% return) without taking event variance. Compared to holding to resolution, exiting early reduces variance dramatically and improves Sharpe — at the cost of giving up the rest of the upside.

---

## Status

**M1-M5 shipped (2026-05-30 to 2026-05-31):** Polymarket ingestion, normalized Parquet persistence, continuous polling daemon, paper-trading engine with strict no-lookahead enforcement (ADR 0004), and a **3-trader sizing horse race** (Kelly + Equal-Wt + Vol-Wt) each running with $10,000 paper bankrolls. Kalshi adapter built but not yet wired into polling (M6).

**Known v0 limitation:** entry/target/stop thresholds (`[0.30, 0.40]` entry, `0.60` target, `0.20` stop) are currently **hardcoded**, not computed from market statistics. The proper version (per-market confidence intervals from observed price history) is on the roadmap. See ADR 0003 for the price-vs-probability discipline that keeps this swap clean.

**Operationally:** the project lives at `~/dev/DriftEdge/`. A `KeepAlive` launchd job polls Polymarket continuously (markets every 5 min, top-20 orderbooks every 30 s) and runs paper decisions on every tick.

## What's wired up today

| Component | Status |
|---|---|
| `driftedge ping` / `fetch-markets` / `fetch-orderbook` / `paper-tick` / `poll` | shipped |
| Polymarket adapter (Gamma + CLOB read endpoints, no auth required) | shipped |
| Kalshi adapter (REST v2 read methods + RSA-PSS signing scaffold) | shipped, **not yet integrated into polling** |
| Normalization layer (uniform DataFrame schema across venues) | shipped |
| Parquet persistence (markets snapshots, orderbook snapshots, trades) | shipped |
| Continuous polling daemon (launchd KeepAlive) | shipped |
| Paper-trading engine, lookahead-safe by construction | shipped |
| 3-trader sizing engine (Kelly + Equal + Vol-Wt) | shipped |
| Per-trader portfolio state (bankroll, cash, exposure, drawdown, peak equity) | shipped |
| Near-resolution-market filter in polling loop | shipped |
| Skip-on-resolution-window in `should_open` | shipped |
| Path engine (logit drift + vol features) | planned (M2) |
| Flow engine (OB imbalance, volume z-scores) | planned (M3) |
| Computed entry/target/stop from market history | planned |
| Cross-venue arbitrage (Polymarket ↔ Kalshi) | planned (M6) |

## The 3-trader paper-trading horse race

Three independent sizers each operate on the same entry/exit triggers but with different position-sizing logic. Each gets $10,000 starting paper bankroll. The winner is whichever finishes the highest equity with reasonable drawdown.

| Trader | Sizing rule |
|---|---|
| **Kelly** | Quarter-Kelly with conservative `p_estimated = 0.45`. Smaller positions on smaller edges; built so swapping `p_estimated` for path-engine output is a one-line change. |
| **Equal-Wt** | Fixed 2 % of bankroll per position (max single exposure cap). The "naive diversifier" baseline. |
| **Vol-Wt** | Inverse-Bernoulli-stddev weighted. Slight tilt toward extreme prices vs. mid-band. Risk-parity for binary contracts. |

Shared caps applied to all three: 2 % per position, 50 % aggregate exposure, $5 min trade.

## No-lookahead enforcement (ADR 0004)

Every function that reads historical data takes an explicit `as_of_ts` parameter and filters reads to `snapshot_ts <= as_of_ts`. Position math asserts `book.snapshot_ts <= as_of_ts`. `tests/test_paper_no_lookahead.py` proves invariance under future-data injection (5/5 tests passing).

## Install / install autorun

```
cd ~/dev/DriftEdge
python3 -m venv .venv
.venv/bin/pip install -e .
./scripts/launchd/install.sh    # starts continuous poll daemon
```

## Why this works in theory

1. **Prices are probabilities.** Unlike options, where the implied distribution has to be extracted via Breeden–Litzenberger, in prediction markets the implied probability *is* the price. Inference collapses; what remains is detecting whether the path is favorable.
2. **Time-series of `p_t` is informative.** Recent literature (e.g., Bayesian inverse formulations, 2026) shows that price-volume histories let us identify latent trader types and predict future drift.
3. **News flow has visible footprints.** Polls, earnings, sports stats, court rulings — every information event shows up in the orderbook before it shows up in the marginal price. The flow engine catches this.
4. **Exit early ⇒ no event risk.** A 0.36 → 0.60 trade has finite, measurable variance because both endpoints are observable in the market. A 0.36 → resolution trade adds an event coin flip on top.

## Why this works in the market

- **Path-dependence is documented** (Path Dependence in AMM-Based Markets, arXiv 2503.00201). Even for CLOB markets, microstructure leaves footprints in the orderbook.
- **Price shocks persist for weeks** (How Manipulable Are Prediction Markets?, arXiv 2503.03312) — meaning informed flow doesn't get arbitraged away instantly.
- **Favorite-longshot bias** (sports betting + politics) shows the market is not uniformly calibrated; structural mispricings exist.

## Architecture (planned)

```
                ┌──────────────────────────────────────────┐
                │            Data Ingestion Layer          │
                │   Polymarket CLOB · Gamma · Data API     │
                │   (read endpoints: no API key needed)    │
                └────────────────────┬─────────────────────┘
                                     │
                ┌────────────────────▼─────────────────────┐
                │       Storage (Parquet, typed schemas)   │
                │   trades / books / markets-metadata      │
                └─────┬──────────────┬──────────────┬──────┘
                      │              │              │
        ┌─────────────▼────┐ ┌───────▼────────┐ ┌───▼───────────┐
        │  Path Engine     │ │  Flow Engine   │ │ Sizing Engine │
        │  · price drift   │ │  · vol spikes  │ │ · Kelly       │
        │  · momentum      │ │  · OB imbalance│ │ · fractional  │
        │  · realized vol  │ │  · large trades│ │ · slippage    │
        │  · path features │ │  · x-market    │ │ · fees        │
        └─────────────┬────┘ └────────┬───────┘ └───────┬───────┘
                      │               │                 │
                ┌─────▼───────────────▼─────────────────▼─────┐
                │           Signal & Decision Layer           │
                │  · entry: low-p + flow agreement            │
                │  · exit: target hit OR flow reverses        │
                │  · stop: path-low + flow neutral            │
                └──────────────────────┬──────────────────────┘
                                       │
                ┌──────────────────────▼──────────────────────┐
                │   Output: logs, alerts, CLI (web later)     │
                └─────────────────────────────────────────────┘
```

## Research foundation

See `docs/research/papers.md` for the annotated bibliography. Key threads:

| Theme | Anchor |
|---|---|
| Prediction-market efficiency | Wolfers & Zitzewitz (2004) — NBER |
| Manipulation persistence | Anonymous (2025), arXiv 2503.03312 — shocks visible 60 days |
| Path-dependence in microstructure | Anonymous (2025), arXiv 2503.00201 |
| Bayesian price-volume inverse | Anonymous (2026), arXiv 2601.18815 |
| Kelly criterion for binary outcomes | Application to Prediction Markets (arXiv 2412.14144) |
| Favorite-longshot bias | Levitt (2004); Whelan (2024) |
| Order book information content | Anonymous, arXiv 1609.03471 |
| LMSR mechanism (legacy) | Hanson (2003) |
| CLOB microstructure | SoK: Decentralized Prediction Markets (2025), arXiv 2510.15612 |

## Data sources

Two co-equal free sources from M1, both with auth-free read endpoints:

### Polymarket
- **CLOB API** (`https://clob.polymarket.com`) — orderbook, prices, spreads, trades
- **Gamma API** (`https://gamma-api.polymarket.com`) — market metadata, events, descriptions
- **Data API** — aggregated analytics
- Auth required *only* for placing orders. Reads are fully open.
- Official `py-clob-client` Python wrapper.

### Kalshi
- **REST v2** (`https://api.elections.kalshi.com/trade-api/v2`) — markets, events, orderbook, series, prices
- **Demo sandbox** (`https://demo-api.kalshi.co`) for testing
- Auth required *only* for trading. Read endpoints are public — no key needed.
- CFTC-regulated US exchange, USD-denominated.
- Rate limits: 30 req/sec public, 10 req/sec authenticated.

Both work the same way from M1: pull market lists, snapshot orderbooks, archive trade tape. The adapter pattern keeps engine code identical regardless of source — cross-market arbitrage (M6) becomes a natural product of having both.

See `docs/research/data_sources.md` for the full adapter plan.

## The path-trade in math

For a Yes contract bought at price `c` with our estimated probability `p`:

- **Kelly fraction:** `f* = (p - c) / (1 - c)` (different from sports-Kelly because of unit-payoff structure)
- **Expected log-growth:** `g(f*) = p·log(1 + f*·(1-c)/c) + (1-p)·log(1 - f*)`
- **Fractional Kelly (default):** `f = 0.25 · f*` — captures ~75% of long-term growth with ~25% of variance.

A path-favorable trade is one where:
- `p_estimated - c > threshold` (positive edge)
- Recent path momentum agrees (price rising, not falling into our entry)
- Flow features confirm (volume up, OB imbalance pro-Yes)
- A 0.36 → 0.60 target zone has positive expected value even before considering eventual resolution

## Repo layout

```
DriftEdge/
├── README.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── docs/
│   ├── research/
│   │   ├── papers.md
│   │   ├── architecture.md
│   │   └── data_sources.md
│   └── decisions/
├── src/
│   └── driftedge/
│       ├── __init__.py
│       ├── config.py
│       ├── obs.py
│       ├── data/        # Polymarket adapter, persistence, normalizers
│       ├── path/        # probability-path features
│       ├── flow/        # volume/OB anomaly detection
│       └── sizing/      # Kelly + fractional + slippage
├── tests/
├── logs/
└── data/
```

## Roadmap

- **M1 — Polymarket + Kalshi ingestion (week 1).** Pull active markets and orderbooks from both venues via their public read APIs; persist to Parquet; log every fetch.
- **M2 — Path engine (week 2).** Price-time-series features per market: momentum, logit-vol, distance-to-entry-zone.
- **M3 — Flow engine (week 3).** Volume z-scores, OB imbalance, large-trade detection. Adapt Chesney–Crameri–Mancini (2015) volume/OI logic to prediction-market trade flow.
- **M4 — Sizing engine (week 4).** Kelly formula for prediction markets, fractional Kelly, slippage and fee modeling.
- **M5 — Signal layer (week 5).** Combine path + flow + sizing into entry/exit/stop signals; log every decision; backtest on accumulated history.
- **M6 — Cross-venue arbitrage.** Detect when Polymarket and Kalshi price the same event differently — exploit when spread exceeds combined slippage + withdrawal cost.
- **M7 — Web UI (later).** Optional dashboard.

## Risk / Legal note

This is **research infrastructure**, not a trading bot. Read-only market data analysis is universally legal. Whether to act on signals — and on which platform — depends on user-side terms of service and applicable regulation. Polymarket has had varying US accessibility; verify your status before considering execution.

## License

TBD (likely MIT).
