"""Runtime configuration loaded from environment.

Reads `.env` if present (via python-dotenv) and exposes a typed Config object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    data_dir: Path
    log_dir: Path
    log_level: str

    poll_interval_s: int
    kelly_fraction: float
    entry_low: float
    entry_high: float
    exit_target: float
    stop_low: float

    polymarket_wallet: Optional[str]
    polymarket_api_key: Optional[str]
    polymarket_api_secret: Optional[str]
    polymarket_api_passphrase: Optional[str]

    kalshi_env: str
    kalshi_api_key_id: Optional[str]
    kalshi_private_key_path: Optional[Path]

    # Kuber — the 6th agent (real-money capable).
    # Defaults: paper mode with $500 split 4 ways across kelly/equal/volwt/
    # volharvest sleeves. Flip kuber_live=True (env KUBER_LIVE=1) to route
    # entries through KalshiClient.place_order instead of simulated fills.
    kuber_live: bool
    kuber_bankroll_total_usd: float
    kuber_bankroll_per_sleeve_usd: float
    kuber_max_position_usd: float
    kuber_dd_kill_pct: float
    kuber_daily_loss_kill_usd: float
    kuber_allow_one_sided: bool = False


def load() -> Config:
    return Config(
        data_dir=Path(os.getenv("DRIFTEDGE_DATA_DIR", "./data")).resolve(),
        log_dir=Path(os.getenv("DRIFTEDGE_LOG_DIR", "./logs")).resolve(),
        log_level=os.getenv("DRIFTEDGE_LOG_LEVEL", "INFO"),

        poll_interval_s=int(os.getenv("DRIFTEDGE_POLL_INTERVAL_S", "60")),
        kelly_fraction=float(os.getenv("DRIFTEDGE_KELLY_FRACTION", "0.25")),
        entry_low=float(os.getenv("DRIFTEDGE_ENTRY_LOW", "0.30")),
        entry_high=float(os.getenv("DRIFTEDGE_ENTRY_HIGH", "0.40")),
        exit_target=float(os.getenv("DRIFTEDGE_EXIT_TARGET", "0.60")),
        stop_low=float(os.getenv("DRIFTEDGE_STOP_LOW", "0.20")),

        polymarket_wallet=os.getenv("POLYMARKET_WALLET_ADDRESS") or None,
        polymarket_api_key=os.getenv("POLYMARKET_API_KEY") or None,
        polymarket_api_secret=os.getenv("POLYMARKET_API_SECRET") or None,
        polymarket_api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE") or None,

        kalshi_env=os.getenv("KALSHI_ENV", "prod"),
        kalshi_api_key_id=os.getenv("KALSHI_API_KEY_ID") or None,
        kalshi_private_key_path=(
            Path(os.getenv("KALSHI_PRIVATE_KEY_PATH")).resolve()
            if os.getenv("KALSHI_PRIVATE_KEY_PATH") else None
        ),

        kuber_live=os.getenv("KUBER_LIVE", "0").lower() in ("1", "true", "yes"),
        kuber_bankroll_total_usd=float(os.getenv("KUBER_BANKROLL", "500")),
        kuber_bankroll_per_sleeve_usd=float(
            os.getenv("KUBER_BANKROLL_PER_SLEEVE", "125")),
        kuber_max_position_usd=float(os.getenv("KUBER_MAX_POSITION_USD", "5")),
        kuber_dd_kill_pct=float(os.getenv("KUBER_DD_KILL_PCT", "0.40")),
        kuber_daily_loss_kill_usd=float(
            os.getenv("KUBER_DAILY_LOSS_KILL_USD", "25")),
        kuber_allow_one_sided=os.getenv(
            "KUBER_ALLOW_ONE_SIDED", "0").lower() in ("1", "true", "yes"),
    )
