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
from .data import normalize, persistence
from .data.polymarket import PolymarketClient, PolymarketError


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
    stop = {"flag": False}

    def _shutdown(signum, frame):
        obs.event(channel="run", kind="poll.signal", level="INFO",
                  signal=signum)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    tracked: list[dict[str, Any]] = []
    last_market_refresh = 0.0
    iteration = 0

    while not stop["flag"]:
        now = time.time()
        try:
            if now - last_market_refresh >= args.market_refresh_s:
                payload = client.list_markets(limit=args.top_n * 4)
                df = normalize.normalize_polymarket_markets(payload)
                persistence.write_markets_snapshot(df, c.data_dir,
                                                   venue="polymarket")
                df = df.dropna(subset=["yes_token_id"])
                df = df[(df["active"] == True) & (df["closed"] == False)]

                # Skip markets too close to resolution — we can't open a
                # new position there anyway (force-exit window), so polling
                # their books is waste.
                now_utc = datetime.now(timezone.utc)
                horizon_h = c.force_exit_hours_before_resolution if hasattr(
                    c, "force_exit_hours_before_resolution") else 6.0

                def _has_tradeable_window(end_date_str: Any) -> bool:
                    if not end_date_str:
                        return True
                    try:
                        end = datetime.fromisoformat(
                            str(end_date_str).replace("Z", "+00:00"))
                        return (end - now_utc).total_seconds() / 3600.0 >= horizon_h
                    except (ValueError, TypeError):
                        return True

                before_filter = len(df)
                df = df[df["end_date"].apply(_has_tradeable_window)]
                after_filter = len(df)

                tracked = (df.sort_values("volume_24h", ascending=False)
                             .head(args.top_n)
                             .to_dict("records"))
                last_market_refresh = now
                obs.event(channel="run", kind="poll.tracked_refresh",
                          level="INFO", tracked=len(tracked),
                          filtered_near_resolution=before_filter - after_filter)

            for m in tracked:
                if stop["flag"]:
                    break
                token_id = m.get("yes_token_id")
                market_id = m.get("market_id")
                if not token_id or not market_id:
                    continue
                try:
                    book = client.get_orderbook(token_id)
                    persistence.write_orderbook_snapshot(
                        book, c.data_dir, venue="polymarket",
                        market_id=market_id, token_id=token_id)
                except PolymarketError as exc:
                    obs.event(channel="error", kind="poll.book_fail",
                              level="WARNING", market_id=market_id,
                              err=str(exc))
                # be polite — don't hammer
                time.sleep(0.3)

            # Paper-trading tick — strictly uses as_of_ts=now so it sees
            # only the snapshots we just wrote, never anything labelled
            # in the future.
            try:
                rule = paper.EntryRule(
                    entry_low=c.entry_low,
                    entry_high=c.entry_high,
                    target=c.exit_target,
                    stop=c.stop_low,
                )
                paper.tick(c.data_dir, tracked, rule)
            except Exception as exc:
                obs.event(channel="error", kind="paper.tick_fail",
                          level="WARNING", err=str(exc),
                          exc_type=type(exc).__name__)

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

    return p


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
