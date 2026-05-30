# Data Sources — Evaluation

Goal: assemble enough free data to build and validate DriftEdge without paying for a feed. Honest about gaps.

---

## Polymarket — primary source

Polymarket exposes three free REST APIs plus a WebSocket feed. None require an API key for **read-only** access to market data.

### CLOB API — orderbook, trades, prices
Base URL: `https://clob.polymarket.com`

Relevant read endpoints (no auth):
- `GET /markets` — list of markets with conditions, tokens, rewards
- `GET /book?token_id=<>` — current orderbook for a token (Yes or No side of a market)
- `GET /price?token_id=<>&side=BUY|SELL` — best bid/ask
- `GET /trades?market=<>` — recent trades
- `GET /prices-history?market=<>&interval=<>` — price history

### Gamma API — market metadata, events
Base URL: `https://gamma-api.polymarket.com`

- `GET /markets` — richer market metadata with descriptions, categories, end dates
- `GET /events` — top-level events that may contain multiple related markets

### Data API — analytics
Aggregated stats; less critical for v0 but useful later.

### WebSocket
Real-time orderbook deltas, trades, prices. Use the `py-clob-client` library or raw WebSocket for our own client.

### Authentication
- **None** for market data above.
- HMAC-SHA256 signing required only for order placement. Out of scope for v0.

### Python client
`pip install py-clob-client` — official, well-maintained, wraps everything above.

### Rate limits
Not publicly documented as a hard number. Empirically, polling every 1-5 seconds per endpoint works without hitting limits. We'll target one orderbook snapshot per minute per active market plus an event-driven WebSocket subscription for hot markets.

---

## Kalshi — co-equal M1 source

CFTC-regulated US prediction market. USD-denominated. **Public read endpoints require no auth**, so usable from M1 alongside Polymarket.

### Base URLs
- **Production REST**: `https://api.elections.kalshi.com/trade-api/v2`
- **Demo sandbox**: `https://demo-api.kalshi.co/trade-api/v2`

### Read endpoints (no auth required)
- `GET /markets` — list all markets with filters (status, event, series, ticker)
- `GET /markets/{ticker}` — single market detail
- `GET /markets/{ticker}/orderbook` — full L2 book (yes bids and no bids; Kalshi convention is to return both sides as bids, not bid/ask)
- `GET /markets/{ticker}/trades` — recent trades
- `GET /events` — list events (each contains 1+ markets)
- `GET /events/{event_ticker}` — single event with all its markets
- `GET /series` — series of recurring events
- `GET /series/{series_ticker}` — single series

### Authenticated endpoints (RSA-PSS signed; out of scope for M1)
- Account info, portfolio, balance — requires auth
- Order placement / cancellation — requires auth
- Trade history (your own) — requires auth

### Rate limits
- **Public (read)**: ~30 req/sec
- **Authenticated**: ~10 req/sec

### Authentication (when we eventually trade)
- RSA-PSS signed requests (more complex than Polymarket's HMAC-SHA256)
- Generate key pair at https://kalshi.com/account/profile (Developer → API Keys)
- Headers required: `KALSHI-ACCESS-KEY` (the Key ID), `KALSHI-ACCESS-SIGNATURE` (base64 of RSA-PSS sign of `timestamp + METHOD + path`), `KALSHI-ACCESS-TIMESTAMP` (ms epoch)
- Private key never re-downloadable; download once and save securely
- Python: `cryptography` library handles signing

### Market structure note
Kalshi markets are **binary by construction** with explicit Yes/No tokens. The orderbook returns "yes bids" and "no bids" rather than bid/ask. Conversion to a unified `(bid, ask)` per side requires the identity `ask_yes = 100 - best_no_bid` (prices are in cents 0–100, not 0.0–1.0).

### Universe
- Politics (elections, policy questions)
- Climate / weather
- Economics (CPI, Fed decisions, GDP)
- Sports
- Crypto and finance (subject to CFTC categorization)
- NO general entertainment markets (more restricted than Polymarket)

### Strategic value vs Polymarket
- **Pros**: Regulated (less manipulation risk), USD-denominated (no on-chain friction), cleaner data, demo sandbox for testing
- **Cons**: Smaller market universe, lower volume in most markets, US-only platform
- **Cross-venue arbitrage**: When the same event is listed on both Polymarket and Kalshi (e.g., US election markets), spreads sometimes exceed combined fees/slippage. The M6 arb engine will exploit this.

---

## Manifold Markets — testing only

Play-money markets. Free API. Useful for testing infrastructure without real-money market noise/manipulation. Probably not useful for edge research because the markets are thinly traded with hobbyist participants.

---

## On-chain data (later, optional)

Polymarket settles on Polygon (the blockchain). Trade/order events are emittable on-chain and indexable via The Graph subgraphs. Free, but:
- Adds blockchain dependency to the stack
- Useful only if we want to reconstruct order-book state historically beyond Polymarket's API history retention

Defer until we have a specific need (e.g., backtesting beyond Polymarket's price-history window).

---

## External signal sources (much later)

For domain-specific `p` estimation:
- **Sports**: ESPN public APIs, basketball-reference, baseball-reference (all free).
- **Politics**: 538 historical data, federal/state election commission APIs.
- **Crypto**: yfinance, exchange APIs.
- **Weather**: NOAA public API.

These would inform the sizing engine when we want to actually estimate `p` beyond market-implied. Out of scope for v0 — the v0 sizer is purely market-data-driven.

---

## What this means for the build

- **M1 ingestion** = Polymarket (CLOB + Gamma) **and** Kalshi (REST v2) in parallel. No auth, no wallet.
- **Archive starts at M1.** Every poll persisted. By M5 we should have weeks of orderbook and trade history on both venues.
- **No fabrication.** If a request fails, log it as missing. No interpolation in the data layer.
- **Adapter pattern.** Both sources implement the same interface (`list_markets`, `fetch_book`, `fetch_trades`, `fetch_price_history`). The engines downstream don't know which venue produced the data.
- **Cross-venue identity mapping.** A small registry maps Polymarket market IDs ↔ Kalshi tickers where they cover the same event. This is required for M6 arb but built incrementally.
