"""DriftEdge CLI — entry point for fetches, polling, inspection.

    python -m driftedge.cli ping
    python -m driftedge.cli fetch-markets [--limit N]
    python -m driftedge.cli fetch-orderbook <token_id>
    python -m driftedge.cli poll [--top-n N] [--book-interval-s 30] [--market-refresh-s 300]
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import date, datetime, timezone
from typing import Any

from . import config as cfg
from . import obs
from . import paper
from .data import normalize, persistence, paper_persist
from .data.polymarket import PolymarketClient, PolymarketError
from .data.kalshi import KalshiClient, KalshiError


# ── Held-market augmentation ────────────────────────────────────────────

def _held_market_ids_by_venue(data_dir) -> dict[str, set[str]]:
    """Return {venue: {market_id, ...}} of every open position across all
    agents. Used to keep held markets in the poll loop even when they'd
    otherwise be filtered out by volume/window/price gates."""
    positions = paper_persist.load_positions(data_dir)
    out: dict[str, set[str]] = {}
    for p in positions:
        if p.get("status") != "open":
            continue
        venue = p.get("venue") or "polymarket"
        mid = str(p.get("market_id") or "")
        if mid:
            out.setdefault(venue, set()).add(mid)
    return out


def _market_stub_from_parquets(venue: str, market_id: str,
                                 data_dir) -> dict[str, Any] | None:
    """Look up `market_id` in the most recent venue parquet that contains
    it. Returns the row as a dict (suitable for use as a market stub in the
    poll loop), or None if not found anywhere."""
    import glob
    import pandas as pd
    files = sorted(
        glob.glob(str(data_dir / "markets" / venue / "*.parquet")),
        reverse=True,
    )
    for fp in files:
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if "market_id" not in df.columns:
            continue
        match = df[df["market_id"].astype(str) == market_id]
        if match.empty:
            continue
        # Take the most recent snapshot within the file.
        if "snapshot_ts" in match.columns:
            match = match.sort_values("snapshot_ts").tail(1)
        return match.iloc[0].to_dict()
    return None


def _augment_tracked_with_held(tracked: list[dict[str, Any]],
                                venue: str, data_dir
                                ) -> tuple[list[dict[str, Any]], int]:
    """Append stubs for any held position's market that isn't already in
    `tracked`. Held markets are looked up in the most recent venue parquet;
    missing ones are skipped (the volharvest orphan-exit loop covers them
    on the consumer side).

    Returns (augmented_list, n_added).
    """
    tracked_ids = {str(m.get("market_id") or "") for m in tracked}
    held_by_venue = _held_market_ids_by_venue(data_dir)
    held_ids = held_by_venue.get(venue, set())
    to_add = held_ids - tracked_ids
    augmented = list(tracked)
    n_added = 0
    for mid in to_add:
        stub = _market_stub_from_parquets(venue, mid, data_dir)
        if stub is not None:
            augmented.append(stub)
            n_added += 1
    return augmented, n_added


# ---------- subcommands ----------

def cmd_ping(_: argparse.Namespace, c: cfg.Config) -> int:
    """Verify Polymarket connectivity (one cheap call)."""
    client = PolymarketClient()
    try:
        markets = client.list_markets(limit=1)
        obs.event(channel="run", kind="ping.ok", level="INFO",
                  sample_market_id=markets[0].get("id") if markets else None)
        return 0
    except PolymarketError as exc:
        obs.event(channel="run", kind="ping.fail", level="ERROR", err=str(exc))
        return 1


def cmd_fetch_markets(args: argparse.Namespace, c: cfg.Config) -> int:
    client = PolymarketClient()
    obs.event(channel="run", kind="fetch_markets.begin", level="INFO",
              limit=args.limit)
    try:
        payload = client.list_markets(limit=args.limit)
    except PolymarketError as exc:
        obs.event(channel="run", kind="fetch_markets.fail",
                  level="ERROR", err=str(exc))
        return 1

    df = normalize.normalize_polymarket_markets(payload)
    path = persistence.write_markets_snapshot(df, c.data_dir, venue="polymarket")
    obs.event(channel="run", kind="fetch_markets.done", level="INFO",
              rows=len(df), path=str(path))
    return 0


def cmd_fetch_orderbook(args: argparse.Namespace, c: cfg.Config) -> int:
    client = PolymarketClient()
    try:
        book = client.get_orderbook(args.token_id)
    except PolymarketError as exc:
        obs.event(channel="run", kind="fetch_book.fail",
                  level="ERROR", err=str(exc))
        return 1
    # We don't know the market_id from a bare token; use token as folder.
    persistence.write_orderbook_snapshot(book, c.data_dir, venue="polymarket",
                                         market_id=args.token_id,
                                         token_id=args.token_id)
    return 0


def cmd_poll(args: argparse.Namespace, c: cfg.Config) -> int:
    """Long-running polling daemon. Refresh markets periodically; snapshot
    orderbooks for the top-N by 24h volume on a tighter loop.
    """
    obs.event(channel="run", kind="poll.start", level="INFO",
              top_n=args.top_n, book_interval_s=args.book_interval_s,
              market_refresh_s=args.market_refresh_s)

    client = PolymarketClient()
    kalshi = KalshiClient(env="prod")
    stop = {"flag": False}

    def _shutdown(signum, frame):
        obs.event(channel="run", kind="poll.signal", level="INFO",
                  signal=signum)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    tracked_poly: list[dict[str, Any]] = []
    tracked_kalshi: list[dict[str, Any]] = []
    # Polymarket token ids whose orderbook is gone (market resolved/closed ->
    # HTTP 404 "No orderbook exists"). Re-requesting these every cycle is the
    # source of the /book 404 storm (~10k/day) that drags poll cadence, so we
    # skip them once seen. Rebuilt on restart (in-memory only). (2026-06-18)
    dead_tokens: set[str] = set()
    last_market_refresh = 0.0
    last_news_sweep = 0.0
    NEWS_INTERVAL_S = 900   # 15 minutes
    iteration = 0

    def _has_tradeable_window(end_date_str: Any, now_utc: datetime,
                               horizon_h: float) -> bool:
        if not end_date_str:
            return True
        try:
            end = datetime.fromisoformat(str(end_date_str).replace("Z", "+00:00"))
            return (end - now_utc).total_seconds() / 3600.0 >= horizon_h
        except (ValueError, TypeError):
            return True

    while not stop["flag"]:
        now = time.time()
        try:
            if now - last_market_refresh >= args.market_refresh_s:
                # Mark the refresh as attempted BEFORE running it. If any
                # step below raises (e.g. a bad sort column, an API blip),
                # the outer handler must not leave last_market_refresh stale
                # — otherwise every iteration re-enters this block, re-raises,
                # and skips all downstream ticks, bricking the daemon into a
                # do-nothing spin. (Bug: liquidity-KeyError spin, 2026-06-18.)
                last_market_refresh = now
                now_utc = datetime.now(timezone.utc)
                horizon_h = 6.0

                # ── POLYMARKET refresh ──
                try:
                    payload = client.list_markets(limit=args.top_n * 6)
                    df = normalize.normalize_polymarket_markets(
                        payload, data_dir=c.data_dir)
                    persistence.write_markets_snapshot(df, c.data_dir,
                                                       venue="polymarket")
                    df = df.dropna(subset=["yes_token_id"])
                    df = df[(df["active"] == True) & (df["closed"] == False)]
                    # Remove near-certain markets: ask ≤ 0.05 (market already
                    # near-NO) or ask ≥ 0.95 (market already near-YES). These
                    # occupy top-volume slots but have no entry opportunity and
                    # no meaningful MTM signal.
                    ask_col = df["best_ask"].fillna(0.5)
                    before_price = len(df)
                    df = df[(ask_col > 0.05) & (ask_col < 0.95)]
                    before = len(df)
                    df = df[df["end_date"].apply(
                        lambda s: _has_tradeable_window(s, now_utc, horizon_h))]
                    tracked_poly = (df.sort_values("volume_24h", ascending=False)
                                      .head(args.top_n).to_dict("records"))
                    # ── Augment with held-position markets that the top-N
                    # / volume / window / price filters would otherwise drop.
                    tracked_poly, n_held_added_poly = _augment_tracked_with_held(
                        tracked_poly, "polymarket", c.data_dir)
                    obs.event(channel="run", kind="poll.tracked_refresh",
                              level="INFO", venue="polymarket",
                              tracked=len(tracked_poly),
                              held_added=n_held_added_poly,
                              filtered_near_certain=before_price - before,
                              filtered_near_resolution=before - len(df))
                except PolymarketError as exc:
                    obs.event(channel="error", kind="poll.markets_fail",
                              level="WARNING", venue="polymarket", err=str(exc))

                # ── KALSHI refresh ──
                try:
                    kpayload = kalshi.list_markets(status="open",
                                                    limit=min(args.top_n * 4, 200))
                    kdf = normalize.normalize_kalshi_markets(
                        kpayload, data_dir=c.data_dir)
                    persistence.write_markets_snapshot(kdf, c.data_dir,
                                                       venue="kalshi")
                    kdf = kdf[kdf["active"] == True]
                    # Need a tradeable spread (best_ask present)
                    kdf = kdf.dropna(subset=["best_ask"])
                    kask = kdf["best_ask"].fillna(0.5)
                    before_kprice = len(kdf)
                    kdf = kdf[(kask > 0.05) & (kask < 0.95)]
                    before_k = len(kdf)
                    kdf = kdf[kdf["end_date"].apply(
                        lambda s: _has_tradeable_window(s, now_utc, horizon_h))]
                    # Kalshi volumes often 0 for new markets; sort by volume but
                    # fall back to liquidity.
                    sort_col = "volume_24h" if (
                        not kdf.empty and kdf["volume_24h"].fillna(0).sum() > 0
                    ) else "liquidity"
                    # normalize_kalshi_markets does not emit a 'liquidity'
                    # column (only the persisted snapshot schema does), so the
                    # fallback could reference a column that isn't on the
                    # in-memory frame -> KeyError. Degrade gracefully.
                    if sort_col not in kdf.columns:
                        sort_col = ("volume_total" if "volume_total" in kdf.columns
                                    else "market_id")
                    tracked_kalshi = (kdf.sort_values(sort_col, ascending=False,
                                                      na_position="last")
                                         .head(args.top_n).to_dict("records"))
                    # ── Same augmentation as polymarket.
                    tracked_kalshi, n_held_added_kalshi = _augment_tracked_with_held(
                        tracked_kalshi, "kalshi", c.data_dir)
                    obs.event(channel="run", kind="poll.tracked_refresh",
                              level="INFO", venue="kalshi",
                              tracked=len(tracked_kalshi),
                              held_added=n_held_added_kalshi,
                              filtered_near_certain=before_kprice - before_k,
                              filtered_near_resolution=before_k - len(kdf),
                              sort_col=sort_col)
                except KalshiError as exc:
                    obs.event(channel="error", kind="poll.markets_fail",
                              level="WARNING", venue="kalshi", err=str(exc))

                # ── equity_history housekeeping ──
                # `append_snapshot` rewrites the whole parquet on every
                # tick. Without bounded trim, write latency grows with file
                # size after ~70 days of running. Cheap call when under the
                # threshold. (Bug 7 fix, 2026-06-05.)
                try:
                    from .data import equity_persist as ep
                    ep.trim(c.data_dir)
                except Exception as exc:
                    obs.event(channel="error", kind="equity.trim_fail",
                              level="WARNING", err=str(exc))

                # (last_market_refresh is set at the top of this block so a
                # mid-refresh exception can't brick the loop into re-refreshing
                # every iteration.)

            # ── ORDERBOOK snapshots: Polymarket ──
            for m in tracked_poly:
                if stop["flag"]:
                    break
                token_id = m.get("yes_token_id")
                market_id = m.get("market_id")
                if not token_id or not market_id:
                    continue
                if str(token_id) in dead_tokens:
                    continue  # resolved market, orderbook gone — don't re-poll
                try:
                    book = client.get_orderbook(token_id)
                    persistence.write_orderbook_snapshot(
                        book, c.data_dir, venue="polymarket",
                        market_id=market_id, token_id=token_id)
                except PolymarketError as exc:
                    # A 404 "No orderbook exists" means the market resolved and
                    # its book is gone for good. Mark the token dead so we stop
                    # hammering it every cycle (the 404-storm cadence drag).
                    msg = str(exc)
                    if "404" in msg or "No orderbook" in msg:
                        dead_tokens.add(str(token_id))
                    obs.event(channel="error", kind="poll.book_fail",
                              level="WARNING", venue="polymarket",
                              market_id=market_id, err=msg)
                time.sleep(0.3)

            # ── ORDERBOOK snapshots: Kalshi ──
            for m in tracked_kalshi:
                if stop["flag"]:
                    break
                ticker = m.get("market_id")
                if not ticker:
                    continue
                try:
                    raw = kalshi.get_orderbook(ticker, depth=20)
                    normalized = normalize.normalize_kalshi_orderbook(raw)
                    persistence.write_orderbook_snapshot(
                        normalized, c.data_dir, venue="kalshi",
                        market_id=ticker, token_id=ticker)
                except KalshiError as exc:
                    obs.event(channel="error", kind="poll.book_fail",
                              level="WARNING", venue="kalshi",
                              market_id=ticker, err=str(exc))
                time.sleep(0.15)  # Kalshi 30 req/sec public; can be faster

            # Paper-trading tick — strictly uses as_of_ts=now so it sees
            # only the snapshots we just wrote, never anything labelled
            # in the future. Now sees BOTH venues.
            try:
                rule = paper.EntryRule(
                    entry_low=c.entry_low,
                    entry_high=c.entry_high,
                    target=c.exit_target,
                    stop=c.stop_low,
                )
                all_markets = tracked_poly + tracked_kalshi
                paper.tick(c.data_dir, all_markets, rule)
            except Exception as exc:
                obs.event(channel="error", kind="paper.tick_fail",
                          level="WARNING", err=str(exc),
                          exc_type=type(exc).__name__)

            # Volharvest tick — runs after the standard paper tick so it
            # sees the same world state. Its own entry/exit logic (binary
            # underdog + opportunistic synthetic-NO hedge); independent
            # state under trader_id='volharvest'.
            try:
                from .agents import volharvest
                volharvest.tick(c.data_dir, all_markets)
            except Exception as exc:
                obs.event(channel="error", kind="volharvest.tick_fail",
                          level="WARNING", err=str(exc),
                          exc_type=type(exc).__name__)

            # Resolution tick — hold-to-binary agent. Enters [0.25, 0.50]
            # markets resolving ≤72h away; holds until resolution or
            # dynamic stop; force-exits 1h before resolution.
            # Resolution agent is QUARANTINED by default — it has no edge
            # (buys YES on any [0.25,0.50] near-resolution market, price !=
            # probability) and bled -$4.2k. Re-enable only with a real
            # p_estimate via DRIFTEDGE_RESOLUTION_ENABLED=1. (2026-06-18)
            if c.resolution_enabled:
                try:
                    from .agents import resolution as resolution_agent
                    resolution_agent.tick(c.data_dir, all_markets)
                except Exception as exc:
                    obs.event(channel="error", kind="resolution.tick_fail",
                              level="WARNING", err=str(exc),
                              exc_type=type(exc).__name__)

            # Kuber tick (6th agent) — Kalshi-only, real-money-capable.
            # Runs LAST so it observes the same markets but maintains its
            # own state. Defaults to paper mode unless KUBER_LIVE=1.
            try:
                from . import kuber
                kalshi_client_for_kuber = None
                if c.kuber_live and c.kalshi_api_key_id and c.kalshi_private_key_path:
                    kalshi_client_for_kuber = KalshiClient(
                        env=c.kalshi_env,
                        api_key_id=c.kalshi_api_key_id,
                        private_key_path=c.kalshi_private_key_path,
                    )
                kuber.tick(c.data_dir, all_markets, rule, c,
                            kalshi_client=kalshi_client_for_kuber)
            except Exception as exc:
                obs.event(channel="error", kind="kuber.tick_fail",
                          level="WARNING", err=str(exc),
                          exc_type=type(exc).__name__)

            # News sweep every NEWS_INTERVAL_S seconds.
            if now - last_news_sweep >= NEWS_INTERVAL_S:
                try:
                    from .data import news as news_mod
                    news_mod.fetch_all(c.data_dir)
                    last_news_sweep = now
                except Exception as exc:
                    obs.event(channel="error", kind="news.sweep_fail",
                              level="WARNING", err=str(exc))

            iteration += 1
            obs.event(channel="run", kind="poll.iteration_done",
                      level="DEBUG", iteration=iteration,
                      sleep_s=args.book_interval_s)
            # Sleep in small chunks so signals interrupt promptly.
            slept = 0.0
            while slept < args.book_interval_s and not stop["flag"]:
                time.sleep(1.0)
                slept += 1.0

        except Exception as exc:
            obs.event(channel="error", kind="poll.loop_error", level="ERROR",
                      err=str(exc), exc_type=type(exc).__name__)
            time.sleep(5.0)

    obs.event(channel="run", kind="poll.stop", level="INFO",
              iterations=iteration)
    return 0


# ---------- parser ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="driftedge", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ping", help="Verify Polymarket connectivity")\
       .set_defaults(func=cmd_ping)

    fm = sub.add_parser("fetch-markets",
                        help="Pull active markets snapshot")
    fm.add_argument("--limit", type=int, default=100)
    fm.set_defaults(func=cmd_fetch_markets)

    fb = sub.add_parser("fetch-orderbook",
                        help="Pull one orderbook by token_id")
    fb.add_argument("token_id")
    fb.set_defaults(func=cmd_fetch_orderbook)

    pl = sub.add_parser("poll",
                        help="Long-running daemon: markets + orderbooks + paper tick")
    pl.add_argument("--top-n", type=int, default=20,
                    help="Number of top-by-volume markets to track (default 20)")
    pl.add_argument("--book-interval-s", type=int, default=30,
                    help="Seconds between orderbook snapshots (default 30)")
    pl.add_argument("--market-refresh-s", type=int, default=300,
                    help="Seconds between market-list refreshes (default 300)")
    pl.set_defaults(func=cmd_poll)

    pt = sub.add_parser("paper-tick",
                        help="Run one paper-trading evaluation cycle (no-lookahead).")
    pt.set_defaults(func=cmd_paper_tick)

    fn = sub.add_parser("fetch-news",
                        help="One sweep of news adapters (RSS + GDELT + Reddit) with VADER sentiment.")
    fn.set_defaults(func=cmd_fetch_news)

    return p


def cmd_fetch_news(_: argparse.Namespace, c: cfg.Config) -> int:
    from .data import news as news_mod
    result = news_mod.fetch_all(c.data_dir)
    obs.event(channel="run", kind="news.cli_done", level="INFO", **result)
    return 0


def cmd_paper_tick(_: argparse.Namespace, c: cfg.Config) -> int:
    """One-off paper tick — useful for testing or manual triggers."""
    client = PolymarketClient()
    try:
        payload = client.list_markets(limit=60)
    except PolymarketError as exc:
        obs.event(channel="run", kind="paper.tick_fetch_fail",
                  level="ERROR", err=str(exc))
        return 1
    df = normalize.normalize_polymarket_markets(payload)
    df = df.dropna(subset=["yes_token_id"])
    df = df[(df["active"] == True) & (df["closed"] == False)]
    tracked = df.head(20).to_dict("records")
    rule = paper.EntryRule(
        entry_low=c.entry_low, entry_high=c.entry_high,
        target=c.exit_target, stop=c.stop_low,
    )
    result = paper.tick(c.data_dir, tracked, rule)
    obs.event(channel="run", kind="paper.tick.cli_done",
              level="INFO", **result)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    c = cfg.load()
    obs.configure(c.log_dir, level=c.log_level)
    obs.install_excepthook()
    try:
        code = args.func(args, c)
    finally:
        obs.finish()
    return code


if __name__ == "__main__":
    sys.exit(main())
