"""News ingestion + VADER sentiment.

Multi-source news fetching with deterministic rule-based sentiment scoring
(no API keys, no model downloads). Persists to data/news/<date>.parquet.

Sources (all free, no key required):
  GDELT 2.0 DOC API   — global news event firehose, sentiment + geography
  Reddit JSON         — public subreddit listings (politics, crypto, etc.)
  RSS feeds           — Reuters / AP / CoinDesk / etc.

Each item is normalized to:
    {source, headline, url, published_ts, raw_text, category,
     sentiment_score, sentiment_label, fetched_at}
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import feedparser
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .. import obs
from ..categorize import categorize_question


_SENTIMENT = SentimentIntensityAnalyzer()


# ── Sentiment classification ─────────────────────────────────────────────

def score_sentiment(text: str) -> tuple[float, str]:
    """Return (compound_score, label) for a single piece of text.

    VADER compound score ∈ [-1, 1]. Conventional thresholds:
        > 0.05   -> 'positive'
        < -0.05  -> 'negative'
        else     -> 'neutral'
    """
    if not text:
        return 0.0, "neutral"
    s = _SENTIMENT.polarity_scores(text)["compound"]
    if s >= 0.05:
        label = "positive"
    elif s <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return float(s), label


# ── Adapters ──────────────────────────────────────────────────────────────

# Curated RSS feeds covering our market categories. All free.
_RSS_FEEDS: list[tuple[str, str]] = [
    ("reuters_world",      "https://feeds.reuters.com/Reuters/worldNews"),
    ("reuters_politics",   "https://feeds.reuters.com/Reuters/PoliticsNews"),
    ("reuters_business",   "https://feeds.reuters.com/reuters/businessNews"),
    ("ap_top",             "https://rsshub.app/apnews/topics/apf-topnews"),
    ("bbc_world",          "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("coindesk",           "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("cnbc_economy",       "https://www.cnbc.com/id/20910258/device/rss/rss.html"),
    ("espn_top",           "https://www.espn.com/espn/rss/news"),
    ("aljazeera",          "https://www.aljazeera.com/xml/rss/all.xml"),
]


def fetch_rss(source_name: str, url: str, limit: int = 30) -> list[dict]:
    """Fetch one RSS feed and normalize to our schema."""
    items: list[dict] = []
    with obs.timed("api", "news.rss", done_level="DEBUG",
                   source=source_name, url=url) as t:
        try:
            feed = feedparser.parse(url)
        except Exception as exc:
            obs.event(channel="error", kind="news.rss_fail",
                      level="WARNING", source=source_name, err=str(exc))
            return []
        t.add(entries=len(feed.entries))

    for entry in feed.entries[:limit]:
        headline = (entry.get("title") or "").strip()
        if not headline:
            continue
        url_ = entry.get("link") or ""
        published = entry.get("published") or entry.get("updated") or ""
        try:
            published_ts = datetime(*entry.published_parsed[:6],
                                     tzinfo=timezone.utc).isoformat()
        except Exception:
            published_ts = published

        summary = entry.get("summary", "") or ""
        sentiment, label = score_sentiment(headline + " " + summary[:400])
        items.append({
            "source": source_name,
            "headline": headline,
            "url": url_,
            "published_ts": published_ts,
            "raw_summary": summary[:600],
            "category": categorize_question(headline + " " + summary[:200]),
            "sentiment_score": round(sentiment, 4),
            "sentiment_label": label,
        })
    return items


def fetch_gdelt(query: str = "(prediction market OR election OR crypto)",
                limit: int = 50) -> list[dict]:
    """GDELT 2.0 DOC API. Free, no key. Returns events worldwide."""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": limit,
        "format": "JSON",
        "sort": "DateDesc",
    }
    items: list[dict] = []
    with obs.timed("api", "news.gdelt", done_level="DEBUG",
                   query=query) as t:
        try:
            r = requests.get(url, params=params, timeout=20)
            t.add(status=r.status_code, bytes=len(r.content))
            if r.status_code != 200:
                return []
            data = r.json()
        except Exception as exc:
            obs.event(channel="error", kind="news.gdelt_fail",
                      level="WARNING", err=str(exc))
            return []

    for art in data.get("articles", []) or []:
        headline = (art.get("title") or "").strip()
        if not headline:
            continue
        sentiment, label = score_sentiment(headline)
        items.append({
            "source": f"gdelt:{art.get('domain', 'unknown')}",
            "headline": headline,
            "url": art.get("url", ""),
            "published_ts": art.get("seendate", ""),
            "raw_summary": art.get("title", ""),
            "category": categorize_question(headline),
            "sentiment_score": round(sentiment, 4),
            "sentiment_label": label,
        })
    return items


def fetch_reddit(subreddit: str, limit: int = 25) -> list[dict]:
    """Reddit public JSON. No key, but UA header recommended."""
    url = f"https://www.reddit.com/r/{subreddit}/hot.json"
    headers = {"User-Agent": "DriftEdge-NewsBot/0.1"}
    items: list[dict] = []
    with obs.timed("api", "news.reddit", done_level="DEBUG",
                   subreddit=subreddit) as t:
        try:
            r = requests.get(url, params={"limit": limit},
                             headers=headers, timeout=15)
            t.add(status=r.status_code, bytes=len(r.content))
            if r.status_code != 200:
                return []
            data = r.json()
        except Exception as exc:
            obs.event(channel="error", kind="news.reddit_fail",
                      level="WARNING", subreddit=subreddit, err=str(exc))
            return []

    for child in data.get("data", {}).get("children", [])[:limit]:
        d = child.get("data", {})
        headline = (d.get("title") or "").strip()
        if not headline:
            continue
        ts_unix = d.get("created_utc")
        published_ts = (datetime.fromtimestamp(ts_unix, tz=timezone.utc)
                        .isoformat() if ts_unix else "")
        sentiment, label = score_sentiment(headline)
        items.append({
            "source": f"reddit:{subreddit}",
            "headline": headline,
            "url": "https://reddit.com" + d.get("permalink", ""),
            "published_ts": published_ts,
            "raw_summary": d.get("selftext", "")[:400],
            "category": categorize_question(headline),
            "sentiment_score": round(sentiment, 4),
            "sentiment_label": label,
        })
    return items


_REDDIT_SUBS = ["politics", "worldnews", "cryptocurrency", "wallstreetbets",
                "economics", "geopolitics", "soccer", "nba"]


# ── Orchestration + persistence ──────────────────────────────────────────

def _path(data_dir: Path) -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return data_dir / "news" / f"{today}.parquet"


def _hash_id(source: str, url: str, headline: str) -> str:
    return hashlib.sha1(
        f"{source}|{url}|{headline}".encode("utf-8")).hexdigest()[:16]


def persist(items: list[dict], data_dir: Path) -> Path:
    """Append items to today's news parquet. Dedupes by (source, url, headline)."""
    if not items:
        return _path(data_dir)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    df_new = pd.DataFrame(items)
    df_new["id"] = df_new.apply(
        lambda r: _hash_id(r["source"], r["url"], r["headline"]), axis=1)
    df_new["fetched_at"] = fetched_at

    p = _path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:
            existing = pd.read_parquet(p)
            combined = pd.concat([existing, df_new], ignore_index=True)
            combined = combined.drop_duplicates(subset=["id"], keep="first")
        except Exception:
            combined = df_new
    else:
        combined = df_new

    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False),
                   p, compression="snappy")
    obs.event(channel="persist", kind="news.write", level="INFO",
              path=str(p), new_items=len(df_new),
              total_today=len(combined),
              bytes=p.stat().st_size)
    return p


def fetch_all(data_dir: Path) -> dict[str, Any]:
    """One full sweep: RSS + GDELT + Reddit. Returns counts per source."""
    counts: dict[str, int] = {}
    obs.event(channel="run", kind="news.sweep_start", level="INFO")

    all_items: list[dict] = []
    for source_name, url in _RSS_FEEDS:
        try:
            it = fetch_rss(source_name, url)
            counts[source_name] = len(it)
            all_items.extend(it)
        except Exception as exc:
            obs.event(channel="error", kind="news.rss_sweep_fail",
                      level="WARNING", source=source_name, err=str(exc))
        time.sleep(0.5)

    for sub in _REDDIT_SUBS:
        try:
            it = fetch_reddit(sub)
            counts[f"reddit:{sub}"] = len(it)
            all_items.extend(it)
        except Exception as exc:
            obs.event(channel="error", kind="news.reddit_sweep_fail",
                      level="WARNING", subreddit=sub, err=str(exc))
        time.sleep(0.6)  # Reddit rate-limits hard if too quick

    try:
        it = fetch_gdelt()
        counts["gdelt"] = len(it)
        all_items.extend(it)
    except Exception as exc:
        obs.event(channel="error", kind="news.gdelt_sweep_fail",
                  level="WARNING", err=str(exc))

    persist(all_items, data_dir)

    obs.event(channel="run", kind="news.sweep_done", level="INFO",
              total_items=len(all_items),
              by_source=counts)
    return {"total": len(all_items), "by_source": counts}
