"""Normalize Polymarket/Kalshi payloads into uniform DataFrames.

The engines downstream consume normalized data. The shape:

markets DataFrame columns:
    venue, market_id, slug, question, category, end_date,
    yes_token_id, no_token_id, volume_24h, volume_total,
    liquidity, best_bid_yes, best_ask_yes, last_price

trades DataFrame columns:
    id, market_id, ts, price, size, side, taker_side
"""

from __future__ import annotations

import json
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


def normalize_polymarket_markets(payload: list[dict]) -> pd.DataFrame:
    """Flatten Polymarket Gamma /markets response."""
    rows: list[dict] = []
    for m in payload:
        yes_token, no_token = _tokens(m)
        yes_label, no_label = _outcomes(m)
        yes_price, no_price = _outcome_prices(m)
        rows.append({
            "venue": "polymarket",
            "market_id": str(m.get("id") or m.get("conditionId") or ""),
            "condition_id": m.get("conditionId"),
            "slug": m.get("slug"),
            "question": m.get("question"),
            "category": m.get("category"),
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
