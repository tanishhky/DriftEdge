"""Classify-once-and-persist market categorization.

User requirement: 100% accuracy because each category routes to a different
engine. A probabilistic LLM cannot guarantee 100%; the honest 100% answer
is a rule-based classifier with a manual review queue for anything the
rules cannot pin down.

Workflow:
  1. New market arrives -> look up in market_categories.parquet by
     (venue, market_id). If present, use stored category.
  2. If absent, run classify(): returns (category, confidence).
     - 'high' confidence -> persist as decided=True, auto-use
     - 'medium' or 'low' confidence -> persist as decided=False (needs review)
  3. The Sentinel review UI exposes the queue. User picks a category;
     row is updated to decided=True with reviewer='manual'.
  4. After human review, no market re-classifies. Ever.

Categories: sports, politics, geopolitics, crypto, macro, entertainment,
weather, other. The 'other' category requires manual review by default.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import obs


# Classifier version: bump when rules change so we know which markets
# were classified by which ruleset. Old decisions are NOT re-run.
CLASSIFIER_VERSION = "rules-v1"


# ── HIGH-CONFIDENCE rules (unambiguous keyword/prefix matches) ──
# Order matters; first hit wins. Specific tournaments before generic verbs.

_HIGH_CONF_RULES: list[tuple[str, re.Pattern]] = [
    # Crypto — unambiguous coin/exchange names
    ("crypto", re.compile(
        r"\b(Bitcoin|BTC|Ethereum|ETH|Solana|SOL|Dogecoin|DOGE|XRP|"
        r"Cardano|ADA|MicroStrategy|Coinbase|Binance|stablecoin|"
        r"halving|altcoin|hashrate)\b", re.IGNORECASE)),

    # Geopolitics — war/conflict/diplomacy named entities
    ("geopolitics", re.compile(
        r"\b(ceasefire|invasion|sanctions|hostage(?:s)?|peace deal|"
        r"Iran|Israel|Russia|Ukraine|Gaza|Hamas|Hezbollah|Houthi|"
        r"Strait of Hormuz|Suez Canal|Taiwan invasion|North Korea|"
        r"NATO|UN Security Council|airstrike|missile strike|"
        r"nuclear test|nuclear weapon)\b", re.IGNORECASE)),

    # Politics — elections / officials
    ("politics", re.compile(
        r"\b(election|elected|presidential|primary|primaries|"
        r"nominee|candidate|Trump|Biden|Harris|DeSantis|Newsom|"
        r"impeachment|Supreme Court|Justice|"
        r"prime minister|chancellor|"
        r"Senate seat|House seat|gubernatorial)\b", re.IGNORECASE)),

    # Macro — Fed/inflation/specific instruments
    ("macro", re.compile(
        r"\b(FOMC|Federal Reserve|interest rate (cut|hike)|CPI|"
        r"PPI|GDP growth|recession|jobless claims|jobs report|NFP|"
        r"S&P 500 close|SPX close|Dow Jones close|Nasdaq close|"
        r"WTI Crude|Brent Crude|10-year yield|treasury yield|"
        r"natural gas futures)\b", re.IGNORECASE)),

    # Sports — leagues and tournaments
    ("sports", re.compile(
        r"\b(NBA Finals|NFL|MLB|NHL|MLS|FIFA World Cup|UEFA Champions League|"
        r"Premier League|Roland Garros|Wimbledon|US Open Tennis|"
        r"ATP Finals|WTA Finals|Super Bowl|World Series|"
        r"Stanley Cup|NBA Playoffs)\b", re.IGNORECASE)),

    # Sports — team names (proper nouns)
    ("sports", re.compile(
        r"\b(?:Will the |The )?(?:New York |Los Angeles |Chicago |Boston )?"
        r"(Knicks|Lakers|Celtics|Warriors|Bulls|Heat|Spurs|Thunder|"
        r"Yankees|Red Sox|Dodgers|Mets|Cubs|Astros|"
        r"Cowboys|Patriots|Chiefs|Eagles|49ers|"
        r"Real Madrid|Barcelona|Arsenal|Liverpool|Manchester United|"
        r"Manchester City|Paris Saint-Germain|PSG|Bayern Munich|"
        r"AC Milan|Inter Milan|Juventus)\b", re.IGNORECASE)),

    # Weather — specific phenomena
    ("weather", re.compile(
        r"\b(hurricane|tornado|earthquake|magnitude \d|temperature|"
        r"reach \d+ degrees|hottest day|coldest day|"
        r"snowfall|rainfall total|heat wave|wildfire|category [1-5])\b",
        re.IGNORECASE)),

    # Entertainment — specific awards / events
    ("entertainment", re.compile(
        r"\b(Grammy|Oscar|Emmy|Tony Award|Golden Globe|"
        r"Cannes Film Festival|box office|"
        r"Marvel|DC Comics|Star Wars|Disney\+|"
        r"Bachelor finale|Survivor finale)\b", re.IGNORECASE)),
]

# ── MEDIUM-CONFIDENCE rules (keyword present but ambiguous context) ──
# These match common terms that often but not always indicate the category.

_MEDIUM_CONF_RULES: list[tuple[str, re.Pattern]] = [
    ("sports", re.compile(
        r"\b(match|game|tournament|playoffs?|championship|"
        r"vs\.?|defeat|beat|wins on)\b", re.IGNORECASE)),
    ("politics", re.compile(
        r"\b(president|senator|congress|policy|bill|veto|"
        r"appointed|resign)\b", re.IGNORECASE)),
    ("macro", re.compile(
        r"\b(Fed|inflation|GDP|unemployment|"
        r"USD|EUR|JPY|gold|silver|oil price|stocks)\b", re.IGNORECASE)),
    ("entertainment", re.compile(
        r"\b(movie|film|album|tour|streaming|"
        r"Netflix|Spotify|Amazon Prime)\b", re.IGNORECASE)),
]


# ── KALSHI ticker prefix → high-confidence category ──
_KALSHI_HIGH_PREFIXES: dict[str, str] = {
    "KXMLB": "sports", "KXNBA": "sports", "KXNFL": "sports",
    "KXNHL": "sports", "KXMLS": "sports", "KXNCAA": "sports",
    "KXATP": "sports", "KXWTA": "sports", "KXSOCCER": "sports",
    "KXEPL": "sports", "KXMVESPORTS": "sports",
    "KXFED": "macro", "KXCPI": "macro", "KXGDP": "macro",
    "KXNFP": "macro", "KXJOBLESS": "macro",
    "KXSP500": "macro", "KXNASDAQ": "macro",
    "KXOIL": "macro", "KXGOLD": "macro",
    "KXBTC": "crypto", "KXETH": "crypto", "KXSOL": "crypto",
    "KXCRYPTO": "crypto",
    "KXELECTION": "politics", "KXPRES": "politics",
    "KXSENATE": "politics", "KXTRUMP": "politics",
    "KXCONGRESS": "politics", "KXSCOTUS": "politics",
    "KXISRAEL": "geopolitics", "KXUKRAINE": "geopolitics",
    "KXIRAN": "geopolitics", "KXCEASEFIRE": "geopolitics",
    "KXTEMP": "weather", "KXHURRICANE": "weather", "KXSNOW": "weather",
    "KXOSCAR": "entertainment", "KXGRAMMY": "entertainment",
    "KXEMMY": "entertainment",
    "KXMVECROSSCATEGORY": None,   # explicit "needs review"
}


@dataclass(frozen=True)
class Classification:
    category: str
    confidence: str    # 'high' | 'medium' | 'low'
    matched_rule: Optional[str]


def classify(question: str | None, *, venue: str = "polymarket",
             market_id: str | None = None,
             event_ticker: str | None = None) -> Classification:
    """Apply rules. Returns (category, confidence, matched_rule)."""
    # Kalshi: try ticker prefix first (high confidence)
    if venue == "kalshi":
        for src in (market_id, event_ticker):
            if not src:
                continue
            for prefix, cat in _KALSHI_HIGH_PREFIXES.items():
                if src.upper().startswith(prefix):
                    if cat is None:
                        return Classification("other", "low", f"kalshi:{prefix}")
                    return Classification(cat, "high", f"kalshi:{prefix}")

    if not question:
        return Classification("other", "low", None)

    for cat, pattern in _HIGH_CONF_RULES:
        m = pattern.search(question)
        if m:
            return Classification(cat, "high", f"high:{cat}:{m.group(0)[:40]}")

    for cat, pattern in _MEDIUM_CONF_RULES:
        m = pattern.search(question)
        if m:
            return Classification(cat, "medium", f"med:{cat}:{m.group(0)[:40]}")

    return Classification("other", "low", None)


# ── Persistence ────────────────────────────────────────────────────────────

_CACHE_FILE = "market_categories.parquet"


def _cache_path(data_dir: Path) -> Path:
    return data_dir / _CACHE_FILE


_EMPTY_CACHE_COLS = [
    "venue", "market_id", "category", "confidence", "matched_rule",
    "decided", "reviewer", "classifier_version",
    "first_seen", "decided_at", "question",
]


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee every expected column exists, so callers can safely do
    cache["market_id"] etc. A cache read that came back missing a column (empty
    or partial-schema parquet) used to raise KeyError('market_id') straight into
    the poll loop; adding the absent columns as empty prevents that."""
    for col in _EMPTY_CACHE_COLS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    return df


def _load_cache(data_dir: Path) -> pd.DataFrame:
    p = _cache_path(data_dir)
    if not p.exists():
        return pd.DataFrame(columns=_EMPTY_CACHE_COLS)
    try:
        return _ensure_schema(pd.read_parquet(p))
    except Exception as exc:  # noqa: BLE001 - corrupt/truncated cache must self-heal
        # A corrupt cache (e.g. a write interrupted by a restart) used to throw
        # here and brick the entire market-refresh loop, starving the daemon of
        # tradeable markets. Quarantine the bad file and rebuild from empty:
        # the cache is a derived lookup, so losing it costs only re-classification.
        try:
            bad = p.with_suffix(".corrupt")
            p.replace(bad)
            obs.event(channel="error", kind="classifier.cache_corrupt",
                      level="WARNING", path=str(p), quarantined=str(bad), err=str(exc))
        except Exception:  # noqa: BLE001
            obs.event(channel="error", kind="classifier.cache_corrupt_unlink_fail",
                      level="WARNING", path=str(p), err=str(exc))
        return pd.DataFrame(columns=_EMPTY_CACHE_COLS)


def _save_cache(data_dir: Path, df: pd.DataFrame) -> None:
    p = _cache_path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    # Atomic write: serialize to a temp file in the same directory, then
    # os.replace() it into place. A crash/restart mid-write can corrupt only
    # the temp file, never the live cache. (Same fix as the 2026-06-22 Kuber
    # brick, ported to DriftEdge where the bug was still latent.)
    tmp = p.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   tmp, compression="snappy")
    os.replace(tmp, p)


def classify_and_cache(data_dir: Path, *,
                       venue: str, market_id: str,
                       question: str | None = None,
                       event_ticker: str | None = None) -> Classification:
    """Look up in cache; if absent, classify and persist.

    Once a market has a row in the cache, this function NEVER reclassifies
    it. The category returned is whatever was decided at first sight (or
    later by manual review).
    """
    cache = _load_cache(data_dir)
    mask = (cache["venue"] == venue) & (cache["market_id"] == market_id)
    if mask.any():
        row = cache[mask].iloc[0]
        return Classification(
            category=str(row["category"]),
            confidence=str(row["confidence"]),
            matched_rule=row.get("matched_rule"),
        )

    cls = classify(question, venue=venue, market_id=market_id,
                   event_ticker=event_ticker)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_row = {
        "venue": venue,
        "market_id": market_id,
        "category": cls.category,
        "confidence": cls.confidence,
        "matched_rule": cls.matched_rule,
        "decided": cls.confidence == "high",
        "reviewer": "auto" if cls.confidence == "high" else None,
        "classifier_version": CLASSIFIER_VERSION,
        "first_seen": now,
        "decided_at": now if cls.confidence == "high" else None,
        "question": question,
    }
    cache = pd.concat([cache, pd.DataFrame([new_row])], ignore_index=True)
    _save_cache(data_dir, cache)
    obs.event(channel="fit", kind="classifier.new", level="DEBUG",
              venue=venue, market_id=market_id,
              category=cls.category, confidence=cls.confidence)
    return cls


def needs_review(data_dir: Path) -> pd.DataFrame:
    """Return rows where decided=False (waiting on human)."""
    cache = _load_cache(data_dir)
    if cache.empty:
        return cache
    return cache[~cache["decided"]].copy().sort_values("first_seen",
                                                       ascending=False)


def set_manual(data_dir: Path, *, venue: str, market_id: str,
               category: str, reviewer: str = "user") -> bool:
    """Apply manual decision. Returns True if row was found and updated."""
    cache = _load_cache(data_dir)
    mask = (cache["venue"] == venue) & (cache["market_id"] == market_id)
    if not mask.any():
        return False
    cache.loc[mask, "category"] = category
    cache.loc[mask, "confidence"] = "manual"
    cache.loc[mask, "decided"] = True
    cache.loc[mask, "reviewer"] = reviewer
    cache.loc[mask, "decided_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    _save_cache(data_dir, cache)
    obs.event(channel="fit", kind="classifier.manual", level="INFO",
              venue=venue, market_id=market_id, category=category,
              reviewer=reviewer)
    return True


def stats(data_dir: Path) -> dict[str, Any]:
    cache = _load_cache(data_dir)
    if cache.empty:
        return {"total": 0}
    return {
        "total": len(cache),
        "decided": int(cache["decided"].sum()),
        "needs_review": int((~cache["decided"]).sum()),
        "by_confidence": cache["confidence"].value_counts().to_dict(),
        "by_category": cache["category"].value_counts().to_dict(),
    }
