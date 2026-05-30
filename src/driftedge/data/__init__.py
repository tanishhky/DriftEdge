"""Data ingestion adapters.

Each provider lives in its own module and exposes a uniform interface:

    list_markets(...) -> pd.DataFrame
    fetch_book(market_id) -> dict
    fetch_trades(market_id, ...) -> pd.DataFrame
    fetch_price_history(market_id, ...) -> pd.DataFrame

The orchestration layer decides which provider to call based on config and
availability.
"""
