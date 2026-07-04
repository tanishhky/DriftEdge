"""Regression tests for the 2026-07-04 news-sweep hang.

feedparser.parse(url) fetches the URL itself through urllib with NO
timeout; a stalled RSS host froze the daemon's main thread in a bare
socket read for 13 hours. Two guarantees are pinned here:

  1. fetch_rss never hands feedparser a URL: the bytes are fetched by
     requests WITH a finite timeout, and feedparser only parses payloads.
  2. The poll daemon installs a process-wide default socket timeout, so
     any future library call that opens a socket without an explicit
     timeout inherits a bound instead of blocking forever.
"""

from __future__ import annotations

import socket

from driftedge.data import news


_RSS_BODY = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>t</title>
<item><title>Headline one</title><link>http://x/1</link>
<pubDate>Fri, 04 Jul 2026 12:00:00 GMT</pubDate></item>
</channel></rss>"""


class _FakeResponse:
    content = _RSS_BODY

    def raise_for_status(self) -> None:
        return None


def test_fetch_rss_uses_requests_with_finite_timeout(monkeypatch):
    seen: dict = {}

    def fake_get(url, **kwargs):
        seen["url"] = url
        seen["timeout"] = kwargs.get("timeout")
        return _FakeResponse()

    real_parse = news.feedparser.parse

    def guarded_parse(arg):
        seen["parse_arg_type"] = type(arg).__name__
        assert not (isinstance(arg, str) and arg.startswith("http")), (
            "feedparser was handed a URL again - it will fetch it with no "
            "timeout and can hang the daemon forever")
        return real_parse(arg)

    monkeypatch.setattr(news.requests, "get", fake_get)
    monkeypatch.setattr(news.feedparser, "parse", guarded_parse)

    items = news.fetch_rss("test_feed", "https://example.com/rss.xml")

    assert seen["timeout"] is not None and seen["timeout"] > 0
    assert seen["parse_arg_type"] == "bytes"
    assert len(items) == 1 and items[0]["headline"] == "Headline one"


def test_fetch_rss_survives_network_failure(monkeypatch):
    def fake_get(url, **kwargs):
        raise news.requests.exceptions.ConnectTimeout("stalled host")

    monkeypatch.setattr(news.requests, "get", fake_get)
    assert news.fetch_rss("dead_feed", "https://dead.example/rss") == []


def test_config_carries_socket_timeout():
    from driftedge import config as cfg
    c = cfg.load()
    assert c.socket_timeout_s > 0


def test_poll_daemon_default_socket_timeout_semantics():
    """setdefaulttimeout governs sockets created without an explicit
    timeout - the exact hole the feedparser fetch fell through."""
    old = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(30.0)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        assert s.gettimeout() == 30.0
        s.close()
    finally:
        socket.setdefaulttimeout(old)
