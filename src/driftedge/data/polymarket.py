"""Polymarket adapter — REST clients for Gamma (metadata) and CLOB (orderbook).

All read endpoints used here are public (no auth). Authentication is required
only for placing orders, which is out of scope for v0.

Endpoint references:
  Gamma:  https://gamma-api.polymarket.com  (markets, events with rich metadata)
  CLOB:   https://clob.polymarket.com       (orderbook, prices, trades)
"""

from __future__ import annotations

from typing import Any, Optional

import requests

from .. import obs


_GAMMA_BASE = "https://gamma-api.polymarket.com"
_CLOB_BASE = "https://clob.polymarket.com"


class PolymarketError(RuntimeError):
    pass


class PolymarketClient:
    """Combined Gamma + CLOB read client.

    Stateless apart from the underlying requests.Session. Safe to instantiate
    once per process and reuse across polling iterations.
    """

    def __init__(self, *, timeout_s: float = 15.0) -> None:
        self._timeout = timeout_s
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ---------- low-level ----------

    def _get(self, base: str, path: str,
             params: Optional[dict[str, Any]] = None) -> Any:
        url = f"{base}{path}"
        with obs.timed("api", "polymarket.get", endpoint=path,
                       params=params, base=base) as t:
            try:
                resp = self._session.get(url, params=params, timeout=self._timeout)
            except requests.RequestException as exc:
                obs.bump("api_errors")
                raise PolymarketError(f"network error on {path}: {exc}") from exc

            t.add(status=resp.status_code, bytes=len(resp.content))
            obs.bump("api_calls")

            if resp.status_code >= 400:
                obs.bump("api_errors")
                raise PolymarketError(
                    f"HTTP {resp.status_code} on {path}: {resp.text[:300]}")
            try:
                return resp.json()
            except ValueError as exc:
                obs.bump("api_errors")
                raise PolymarketError(f"non-JSON response on {path}") from exc

    # ---------- Gamma API: market metadata ----------

    def list_markets(self, *, active: bool = True, closed: bool = False,
                     archived: bool = False, limit: int = 100,
                     offset: int = 0, order: str = "volume24hr",
                     ascending: bool = False) -> list[dict[str, Any]]:
        """List markets via Gamma API.

        Filters default to currently-tradeable: active=True, closed=False.
        Sort default is by 24h volume descending — useful for picking which
        markets to track.
        """
        params: dict[str, Any] = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "archived": str(archived).lower(),
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower(),
        }
        data = self._get(_GAMMA_BASE, "/markets", params=params)
        results = data if isinstance(data, list) else data.get("data", []) or []
        obs.event(channel="api", kind="polymarket.markets.summary", level="INFO",
                  returned=len(results), order=order)
        return results

    def get_market(self, market_id: str) -> dict[str, Any]:
        return self._get(_GAMMA_BASE, f"/markets/{market_id}")

    def list_events(self, *, active: bool = True, closed: bool = False,
                    limit: int = 100, offset: int = 0,
                    order: str = "volume24hr") -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": "false",
        }
        data = self._get(_GAMMA_BASE, "/events", params=params)
        return data if isinstance(data, list) else data.get("data", []) or []

    # ---------- CLOB API: orderbook + prices ----------

    def get_orderbook(self, token_id: str) -> dict[str, Any]:
        """L2 orderbook for one token (one side of a binary market)."""
        return self._get(_CLOB_BASE, "/book", params={"token_id": token_id})

    def get_price(self, token_id: str, side: str = "BUY") -> dict[str, Any]:
        """Best bid (side=SELL) or ask (side=BUY) for a token."""
        return self._get(_CLOB_BASE, "/price",
                         params={"token_id": token_id, "side": side.upper()})

    def get_midpoint(self, token_id: str) -> dict[str, Any]:
        return self._get(_CLOB_BASE, "/midpoint",
                         params={"token_id": token_id})

    def get_trades(self, market: str, *, next_cursor: Optional[str] = None) -> dict[str, Any]:
        """Recent trades for a market (a Polymarket 'condition' id)."""
        params: dict[str, Any] = {"market": market}
        if next_cursor:
            params["next_cursor"] = next_cursor
        return self._get(_CLOB_BASE, "/trades", params=params)

    def get_prices_history(self, market: str, *, interval: str = "1h",
                           startTs: Optional[int] = None,
                           endTs: Optional[int] = None) -> dict[str, Any]:
        """Historical price series for a market.

        interval: '1m' | '1h' | '6h' | '1d' | '1w' | 'max'
        """
        params: dict[str, Any] = {"market": market, "interval": interval}
        if startTs is not None: params["startTs"] = startTs
        if endTs is not None: params["endTs"] = endTs
        return self._get(_CLOB_BASE, "/prices-history", params=params)
