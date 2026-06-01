"""Normalize Polymarket/Kalshi payloads into uniform DataFrames.

The engines downstream consume normalized data. The shape:

markets DataFrame columns:
    venue, market_id, slug, question, category, end_date,
    yes_token_id, no_token_id, volume_24h, volume_total,
    liquidity, best_bid_yes, best_ask_yes, last_price

If data_dir is passed to normalize_*_markets, each new market is
classified once via classifier.classify_and_cache() and the decision
persisted to market_categories.parquet. Subsequent fetches read the
cached category — the classifier is never rerun on a known market.

trades DataFrame columns:
    id, market_id, ts, price, size, side, taker_side
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .. import obs


def _f(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return f if not pd.isna(f) else None
    except (TypeError, ValueError):
        return None


def _tokens(market: dict) -> tuple[Optional[str], Optional[str]]:
    """Extract (yes_token_id, no_token_id) from a Polymarket market record.

    Polymarket stores the two token IDs in a JSON string under `clobTokenIds`
    in the Gamma response. Tokens[0] is Yes, [1] is No (by Polymarket convention).
    """
    raw = market.get("clobTokenIds")
    if not raw:
        return None, None
    try:
        ids = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None, None
    if not isinstance(ids, list) or len(ids) < 2:
        return None, None
    return str(ids[0]), str(ids[1])


def _outcomes(market: dict) -> tuple[Optional[str], Optional[str]]:
    """Extract (yes_label, no_label) — usually 'Yes'/'No' but not always."""
    raw = market.get("outcomes")
    if not raw:
        return None, None
    try:
        labels = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None, None
    if not isinstance(labels, list) or len(labels) < 2:
        return None, None
    return str(labels[0]), str(labels[1])


def _outcome_prices(market: dict) -> tuple[Optional[float], Optional[float]]:
    raw = market.get("outcomePrices")
    if not raw:
        return None, None
    try:
        prices = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None, None
    if not isinstance(prices, list) or len(prices) < 2:
        return None, None
    return _f(prices[0]), _f(prices[1])


def normalize_polymarket_markets(payload: list[dict],
                                 data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Flatten Polymarket Gamma /markets response.

    If data_dir is provided, categorization goes through the persistent
    classify-once cache (classifier.classify_and_cache).
    """
    if data_dir is not None:
        from ..classifier import classify_and_cache
    rows: list[dict] = []
    for m in payload:
        yes_token, no_token = _tokens(m)
        yes_label, no_label = _outcomes(m)
        yes_price, no_price = _outcome_prices(m)
        question = m.get("question")
        market_id = str(m.get("id") or m.get("conditionId") or "")
        api_cat = m.get("category")
        if api_cat:
            category = api_cat
        elif data_dir is not None and market_id:
            cls = classify_and_cache(data_dir, venue="polymarket",
                                     market_id=market_id, question=question)
            category = cls.category
        else:
            from ..categorize import categorize_question
            category = categorize_question(question)
        rows.append({
            "venue": "polymarket",
            "market_id": str(m.get("id") or m.get("conditionId") or ""),
            "condition_id": m.get("conditionId"),
            "slug": m.get("slug"),
            "question": question,
            "category": category,
            "end_date": m.get("endDate"),
            "yes_token_id": yes_token,
            "no_token_id": no_token,
            "yes_label": yes_label,
            "no_label": no_label,
            "yes_price": yes_price,
            "no_price": no_price,
            "volume_24h": _f(m.get("volume24hr")),
            "volume_total": _f(m.get("volume")),
            "liquidity": _f(m.get("liquidity")),
            "spread": _f(m.get("spread")),
            "best_bid": _f(m.get("bestBid")),
            "best_ask": _f(m.get("bestAsk")),
            "last_price": _f(m.get("lastTradePrice")),
            "active": m.get("active"),
            "closed": m.get("closed"),
        })
    df = pd.DataFrame(rows)
    obs.event(channel="persist", kind="polymarket.markets.normalized",
              level="INFO", returned=len(df),
              with_tokens=int(df["yes_token_id"].notna().sum()) if not df.empty else 0)
    return df


def normalize_polymarket_trades(payload: list[dict],
                                market_id: str) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    rows: list[dict] = []
    for t in payload:
        rows.append({
            "id": t.get("id") or t.get("transactionHash"),
            "market_id": market_id,
            "ts": t.get("timestamp") or t.get("createdAt"),
            "price": _f(t.get("price")),
            "size": _f(t.get("size")),
            "side": t.get("side"),
            "taker_side": t.get("takerSide") or t.get("taker_side"),
            "maker_address": t.get("makerAddress"),
        })
    return pd.DataFrame(rows)


def normalize_kalshi_markets(payload: dict,
                              data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Flatten Kalshi /markets response. Prices already in USD (0.0-1.0)."""
    if data_dir is not None:
        from ..classifier import classify_and_cache
    else:
        from ..categorize import categorize_kalshi_ticker
    rows: list[dict] = []
    markets = payload.get("markets", []) if isinstance(payload, dict) else []
    for m in markets:
        title = m.get("title") or m.get("yes_sub_title")
        ticker = m.get("ticker")
        event_ticker = m.get("event_ticker")
        yes_bid = _f(m.get("yes_bid_dollars"))
        yes_ask = _f(m.get("yes_ask_dollars"))
        if yes_bid is not None and yes_ask is not None:
            yes_price = (yes_bid + yes_ask) / 2
        else:
            yes_price = _f(m.get("last_price_dollars"))
        no_price = (1.0 - yes_price) if yes_price is not None else None
        spread = (yes_ask - yes_bid) if (yes_ask is not None and yes_bid is not None) else None

        if data_dir is not None and ticker:
            cls = classify_and_cache(data_dir, venue="kalshi",
                                     market_id=ticker, question=title,
                                     event_ticker=event_ticker)
            category = cls.category
        else:
            category = categorize_kalshi_ticker(ticker, event_ticker, title)

        rows.append({
            "venue": "kalshi",
            "market_id": ticker,
            "condition_id": event_ticker,
            "slug": ticker,
            "question": title,
            "category": category,
            "end_date": m.get("close_time") or m.get("expiration_time"),
            "yes_token_id": m.get("ticker"),
            "no_token_id": None,
            "yes_label": "Yes",
            "no_label": "No",
            "yes_price": yes_price,
            "no_price": no_price,
            "volume_24h": _f(m.get("volume_24h_fp")),
            "volume_total": _f(m.get("volume_fp")),
            "liquidity": _f(m.get("liquidity_dollars")),
            "spread": spread,
            "best_bid": yes_bid,
            "best_ask": yes_ask,
            "last_price": _f(m.get("last_price_dollars")),
            "active": m.get("status") == "active",
            "closed": m.get("status") not in ("active", "open"),
        })
    df = pd.DataFrame(rows)
    obs.event(channel="persist", kind="kalshi.markets.normalized",
              level="INFO", returned=len(df),
              with_quotes=int(df["best_ask"].notna().sum()) if not df.empty else 0)
    return df


def normalize_kalshi_orderbook(payload: dict) -> dict:
    """Convert Kalshi orderbook into Polymarket-style {bids, asks} on YES.

    Kalshi returns yes_dollars (YES bids) and no_dollars (NO bids).
    YES ask is inverted from NO bid: ask_yes_price = 1 - no_bid_price.
    """
    ob = (payload or {}).get("orderbook_fp", {}) or {}
    yes_side = ob.get("yes_dollars") or []
    no_side = ob.get("no_dollars") or []

    bids: list[dict] = []
    for lvl in yes_side:
        try:
            bids.append({"price": float(lvl[0]), "size": float(lvl[1])})
        except (ValueError, TypeError, IndexError):
            continue
    bids.sort(key=lambda x: x["price"], reverse=True)

    asks: list[dict] = []
    for lvl in no_side:
        try:
            asks.append({"price": 1.0 - float(lvl[0]), "size": float(lvl[1])})
        except (ValueError, TypeError, IndexError):
            continue
    asks.sort(key=lambda x: x["price"])

    return {"bids": bids, "asks": asks}
