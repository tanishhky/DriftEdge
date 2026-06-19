"""Kalshi adapter — REST v2 client with optional RSA-PSS signing.

Read endpoints (markets, orderbook, trades, events, series) require no auth.
RSA-PSS signing kicks in only for authenticated endpoints (portfolio, trading).

API reference: docs.kalshi.com/api-reference
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Optional

import requests

from .. import obs


_PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"


class KalshiError(RuntimeError):
    pass


class KalshiClient:
    """REST v2 client for Kalshi.

    For read-only use: instantiate with no key args.
    For trading: pass api_key_id and private_key_path; signing happens
    automatically per request.
    """

    def __init__(self, *, env: str = "prod",
                 api_key_id: Optional[str] = None,
                 private_key_path: Optional[Path] = None,
                 timeout_s: float = 20.0) -> None:
        self._base = _PROD_BASE if env == "prod" else _DEMO_BASE
        self._timeout = timeout_s
        self._key_id = api_key_id
        self._private_key = None
        if private_key_path is not None:
            self._private_key = self._load_private_key(private_key_path)
        self._session = requests.Session()

    @staticmethod
    def _load_private_key(path: Path):
        try:
            from cryptography.hazmat.primitives import serialization
        except ImportError as exc:
            raise KalshiError(
                "cryptography package required for Kalshi auth: pip install cryptography"
            ) from exc
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)

    def _sign(self, method: str, request_path: str) -> dict[str, str]:
        """Build the three Kalshi auth headers via RSA-PSS over
        timestamp + method + path. Returns empty dict if not authenticated.
        """
        if self._private_key is None or self._key_id is None:
            return {}

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        ts_ms = str(int(time.time() * 1000))
        msg = (ts_ms + method.upper() + request_path).encode()
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        }

    def _get(self, path: str, params: Optional[dict[str, Any]] = None,
             auth: bool = False) -> dict[str, Any]:
        url = f"{self._base}{path}"
        headers = self._sign("GET", path) if auth else {}

        with obs.timed("api", "kalshi.get", endpoint=path, params=params,
                       authed=auth) as t:
            try:
                resp = self._session.get(url, params=params, headers=headers,
                                         timeout=self._timeout)
            except requests.RequestException as exc:
                obs.bump("api_errors")
                raise KalshiError(f"network error on {path}: {exc}") from exc

            t.add(status=resp.status_code, bytes=len(resp.content))
            obs.bump("api_calls")

            if resp.status_code >= 400:
                obs.bump("api_errors")
                raise KalshiError(f"HTTP {resp.status_code} on {path}: {resp.text[:300]}")
            try:
                return resp.json()
            except ValueError as exc:
                obs.bump("api_errors")
                raise KalshiError(f"non-JSON response on {path}") from exc

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """RSA-signed POST. Always authed — order placement needs it."""
        if self._private_key is None or self._key_id is None:
            raise KalshiError(
                "POST requires Kalshi credentials: set KALSHI_API_KEY_ID "
                "and KALSHI_PRIVATE_KEY_PATH.")
        url = f"{self._base}{path}"
        headers = self._sign("POST", path)
        headers["Content-Type"] = "application/json"

        with obs.timed("api", "kalshi.post", endpoint=path) as t:
            try:
                resp = self._session.post(url, data=json.dumps(body),
                                            headers=headers,
                                            timeout=self._timeout)
            except requests.RequestException as exc:
                obs.bump("api_errors")
                raise KalshiError(f"network error on POST {path}: {exc}") from exc

            t.add(status=resp.status_code, bytes=len(resp.content))
            obs.bump("api_calls")

            if resp.status_code >= 400:
                obs.bump("api_errors")
                raise KalshiError(
                    f"HTTP {resp.status_code} on POST {path}: {resp.text[:400]}")
            try:
                return resp.json()
            except ValueError as exc:
                obs.bump("api_errors")
                raise KalshiError(f"non-JSON response on POST {path}") from exc

    def _delete(self, path: str) -> dict[str, Any]:
        """RSA-signed DELETE — used for cancelling resting orders."""
        if self._private_key is None or self._key_id is None:
            raise KalshiError("DELETE requires Kalshi credentials.")
        url = f"{self._base}{path}"
        headers = self._sign("DELETE", path)

        with obs.timed("api", "kalshi.delete", endpoint=path) as t:
            try:
                resp = self._session.delete(url, headers=headers,
                                              timeout=self._timeout)
            except requests.RequestException as exc:
                obs.bump("api_errors")
                raise KalshiError(f"network error on DELETE {path}: {exc}") from exc

            t.add(status=resp.status_code, bytes=len(resp.content))
            obs.bump("api_calls")

            if resp.status_code >= 400:
                obs.bump("api_errors")
                raise KalshiError(
                    f"HTTP {resp.status_code} on DELETE {path}: {resp.text[:400]}")
            try:
                return resp.json() if resp.content else {}
            except ValueError:
                return {}

    # ---------- public read endpoints (no auth required) ----------

    def list_markets(self, *, status: Optional[str] = None,
                     event_ticker: Optional[str] = None,
                     series_ticker: Optional[str] = None,
                     limit: int = 200,
                     cursor: Optional[str] = None) -> dict[str, Any]:
        """List markets with optional filters. status in {open, closed, settled}."""
        params: dict[str, Any] = {"limit": limit}
        if status: params["status"] = status
        if event_ticker: params["event_ticker"] = event_ticker
        if series_ticker: params["series_ticker"] = series_ticker
        if cursor: params["cursor"] = cursor
        return self._get("/markets", params=params)

    def get_market(self, ticker: str) -> dict[str, Any]:
        return self._get(f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 100) -> dict[str, Any]:
        return self._get(f"/markets/{ticker}/orderbook", params={"depth": depth})

    def get_trades(self, ticker: str, *, limit: int = 100,
                   cursor: Optional[str] = None) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if cursor: params["cursor"] = cursor
        return self._get(f"/markets/{ticker}/trades", params=params)

    def list_events(self, *, status: Optional[str] = None,
                    series_ticker: Optional[str] = None,
                    limit: int = 200,
                    cursor: Optional[str] = None) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if status: params["status"] = status
        if series_ticker: params["series_ticker"] = series_ticker
        if cursor: params["cursor"] = cursor
        return self._get("/events", params=params)

    def get_event(self, event_ticker: str) -> dict[str, Any]:
        return self._get(f"/events/{event_ticker}")

    def list_series(self, *, limit: int = 200,
                    cursor: Optional[str] = None) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if cursor: params["cursor"] = cursor
        return self._get("/series", params=params)

    # ---------- authed portfolio + trading endpoints ----------

    def get_balance(self) -> dict[str, Any]:
        """Return cents-denominated cash balance and total exposure."""
        return self._get("/portfolio/balance", auth=True)

    def get_positions(self, *, ticker: Optional[str] = None,
                      limit: int = 200,
                      cursor: Optional[str] = None) -> dict[str, Any]:
        """List currently-held positions across markets."""
        params: dict[str, Any] = {"limit": limit}
        if ticker: params["ticker"] = ticker
        if cursor: params["cursor"] = cursor
        return self._get("/portfolio/positions", params=params, auth=True)

    def get_orders(self, *, ticker: Optional[str] = None,
                   status: Optional[str] = None,
                   limit: int = 200,
                   cursor: Optional[str] = None) -> dict[str, Any]:
        """List orders. status ∈ {resting, canceled, executed}."""
        params: dict[str, Any] = {"limit": limit}
        if ticker: params["ticker"] = ticker
        if status: params["status"] = status
        if cursor: params["cursor"] = cursor
        return self._get("/portfolio/orders", params=params, auth=True)

    def get_fills(self, *, ticker: Optional[str] = None,
                  limit: int = 200,
                  cursor: Optional[str] = None) -> dict[str, Any]:
        """List recent fills — used for reconciliation."""
        params: dict[str, Any] = {"limit": limit}
        if ticker: params["ticker"] = ticker
        if cursor: params["cursor"] = cursor
        return self._get("/portfolio/fills", params=params, auth=True)

    def place_order(self, *, market_ticker: str, side: str,
                    action: str = "buy", order_type: str = "limit",
                    count: int, price_cents: int,
                    client_order_id: Optional[str] = None,
                    expiration_ts: Optional[int] = None,
                    post_only: bool = True) -> dict[str, Any]:
        """Place a limit order on Kalshi.

        side : "yes" or "no" (Kalshi binary outcomes)
        action : "buy" or "sell"
        count : number of contracts
        price_cents : limit price in cents (1-99)
        post_only : reject if the order would cross (passive only)

        Returns the API's order dict including `order_id`.
        """
        assert side in ("yes", "no"), f"bad side: {side}"
        assert action in ("buy", "sell"), f"bad action: {action}"
        assert 1 <= int(price_cents) <= 99, (
            f"price_cents must be 1-99, got {price_cents}")
        assert int(count) >= 1, f"count must be >=1, got {count}"

        body: dict[str, Any] = {
            "ticker": market_ticker,
            "side": side,
            "action": action,
            "type": order_type,
            "count": int(count),
        }
        # Kalshi uses yes_price / no_price as the limit price (cents).
        if side == "yes":
            body["yes_price"] = int(price_cents)
        else:
            body["no_price"] = int(price_cents)
        if client_order_id:
            body["client_order_id"] = client_order_id
        if expiration_ts:
            body["expiration_ts"] = int(expiration_ts)
        if post_only:
            body["post_only"] = True
        return self._post("/portfolio/orders", body=body)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel a resting order by its API-assigned id."""
        return self._delete(f"/portfolio/orders/{order_id}")
