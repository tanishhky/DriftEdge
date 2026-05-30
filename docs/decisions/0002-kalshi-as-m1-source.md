# 2. Kalshi as a co-equal M1 source

Date: 2026-05-30

## Status

Accepted.

## Context

Original roadmap deferred Kalshi to M6 ("after Polymarket M1–M5 ships"). The
assumption was that Kalshi requires account signup before any data access. On
verification, this is false: Kalshi's public read endpoints (markets, orderbook,
events, series, trades) require no authentication. Rate limits are generous
(30 req/sec public).

Polymarket and Kalshi are not redundant: they have different market universes
(Polymarket has crypto/entertainment; Kalshi has CFTC-regulated political and
economic markets) and overlap on a small set of major events. Differences in
their orderbooks on the overlap are the input to the M6 arbitrage engine.

## Decision

Build the Kalshi adapter at M1 alongside the Polymarket adapter, behind the same
adapter interface. M6 ("cross-venue arbitrage") becomes a natural product of
M1's existence rather than a separate build.

The adapter is a stub today (only the REST client + signing scaffold). M1 work
proper happens once we start writing fetch orchestration.

## Consequences

- M1 has slightly more surface area: two adapters instead of one.
- Engines (path, flow, sizing) remain venue-agnostic since they consume
  normalized DataFrames.
- A small registry `data/identities.parquet` maps Polymarket market IDs to
  Kalshi tickers for events present on both venues. Built incrementally.
- `.env.example` keeps Kalshi credential fields, but they're documented as
  trading-only — read endpoints work without them.
- Architecture diagram in README shows Kalshi on equal footing with Polymarket.
