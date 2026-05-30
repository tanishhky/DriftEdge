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

## Kalshi — secondary source (M6)

CFTC-regulated US prediction market. USD-denominated.

- API requires account signup (free).
- REST + WebSocket.
- More limited market universe than Polymarket (no crypto, restrictions on certain categories).
- Useful for cross-platform arbitrage and as a sanity-check on Polymarket pricing.

Out of scope until M6.

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

- **M1 ingestion** = Polymarket CLOB + Gamma only. No auth, no wallet.
- **Archive starts at M1.** Every poll persisted. By M5 we should have weeks of orderbook and trade history.
- **No fabrication.** If a request fails, log it as missing. No interpolation in the data layer.
- **Adapter pattern.** All sources implement the same interface so Kalshi/Manifold can drop in later.
