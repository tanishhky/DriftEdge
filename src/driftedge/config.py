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

    kalshi_email: Optional[str]
    kalshi_password: Optional[str]
    kalshi_env: str


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

        kalshi_email=os.getenv("KALSHI_EMAIL") or None,
        kalshi_password=os.getenv("KALSHI_PASSWORD") or None,
        kalshi_env=os.getenv("KALSHI_ENV", "demo"),
    )
