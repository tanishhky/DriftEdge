"""Categorize prediction-market questions by topic.

Polymarket's API returns category=null for most markets in practice, so we
classify from the question text using keyword priorities. This is intentionally
simple — heuristics tuned to the markets we actually see. Categories are used
for per-segment P&L analysis and (eventually) for category-specific entry
zones once the path engine ships.

Categories returned:
    sports        — leagues, players, tournaments, match-level
    politics      — elections, policy, government
    geopolitics   — war, ceasefires, sanctions, international tensions
    crypto        — bitcoin / eth / on-chain markets
    macro         — Fed, CPI, rates, FX, commodities, equities
    entertainment — movies, music, awards, celebrity events
    weather       — temperature, hurricane, climate
    other         — fallback
"""

from __future__ import annotations

import re

# Order matters — first match wins. Sport leagues before generic verbs.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("sports", re.compile(
        r"\b(NBA|NFL|MLB|NHL|MLS|UEFA|FIFA|World Cup|Champions League|"
        r"Premier League|Roland Garros|Wimbledon|US Open|ATP|WTA|"
        r"Spurs|Thunder|Knicks|Lakers|Warriors|Celtics|Heat|"
        r"Yankees|Red Sox|Dodgers|Mets|Cubs|"
        r"Cowboys|Patriots|Chiefs|Eagles|"
        r"Real Madrid|Barcelona|Arsenal|Liverpool|Manchester|PSG|"
        r"Bayern|Juventus|AC Milan|"
        r"vs\.|win the .* (Cup|Finals|Championship)|"
        r"win on \d{4}-\d{2}-\d{2}|"
        r"match|game|tournament|playoffs|playoff)\b",
        re.IGNORECASE)),
    ("crypto", re.compile(
        r"\b(Bitcoin|BTC|Ethereum|ETH|Solana|SOL|Dogecoin|DOGE|"
        r"crypto|stablecoin|MicroStrategy|Coinbase|Binance|on-chain|"
        r"blockchain|halving|altcoin)\b",
        re.IGNORECASE)),
    ("geopolitics", re.compile(
        r"\b(ceasefire|war|invasion|nuclear|sanctions|hostage|peace deal|"
        r"Iran|Israel|Russia|Ukraine|Gaza|Hamas|Hezbollah|Houthi|"
        r"Hormuz|Suez|Taiwan|North Korea|"
        r"NATO|UN Security|blockade|airstrike|missile)\b",
        re.IGNORECASE)),
    ("politics", re.compile(
        r"\b(election|elected|president|presidential|senate|congress|"
        r"primary|primaries|nominee|candidate|"
        r"Trump|Biden|Harris|DeSantis|Newsom|"
        r"Congress|Senate|House|Supreme Court|Justice|impeachment|"
        r"resign|appointed|policy|bill|veto|"
        r"prime minister|chancellor)\b",
        re.IGNORECASE)),
    ("macro", re.compile(
        r"\b(Fed|FOMC|interest rate|rate (cut|hike)|CPI|inflation|"
        r"GDP|recession|unemployment|jobless|jobs report|"
        r"S&P 500|SPX|Dow|Nasdaq|"
        r"WTI|Brent|crude|oil|gold|silver|natural gas|"
        r"USD|EUR|JPY|GBP|treasury|10-year yield)\b",
        re.IGNORECASE)),
    ("entertainment", re.compile(
        r"\b(Grammy|Oscar|Emmy|Tony|Golden Globe|"
        r"Taylor Swift|Beyoncé|Beyonce|Drake|Kanye|"
        r"Box office|movie|film|album|tour|"
        r"Netflix Top 10|streaming|"
        r"reality|Bachelor|Survivor|Top Chef|"
        r"Marvel|DC|Avengers|Star Wars)\b",
        re.IGNORECASE)),
    ("weather", re.compile(
        r"\b(hurricane|tornado|earthquake|temperature|degrees|"
        r"snow|rainfall|heat wave|wildfire|climate)\b",
        re.IGNORECASE)),
]


def categorize_question(question: str | None) -> str:
    """Return a single category label for a market question. 'other' on no match."""
    if not question:
        return "other"
    for label, pattern in _PATTERNS:
        if pattern.search(question):
            return label
    return "other"


# Kalshi ticker prefixes → category. More reliable than question text
# for parlay markets where the title is just a list of leg outcomes.
_KALSHI_PREFIXES: dict[str, str] = {
    "KXMLB": "sports",        # baseball
    "KXNBA": "sports",
    "KXNFL": "sports",
    "KXNHL": "sports",
    "KXMLS": "sports",
    "KXNCAA": "sports",
    "KXATP": "sports",        # tennis
    "KXWTA": "sports",
    "KXSOCCER": "sports",
    "KXEPL": "sports",
    "KXMVESPORTS": "sports",
    "KXFED": "macro",
    "KXCPI": "macro",
    "KXGDP": "macro",
    "KXNFP": "macro",
    "KXJOBLESS": "macro",
    "KXSP500": "macro",
    "KXOIL": "macro",
    "KXGOLD": "macro",
    "KXBTC": "crypto",
    "KXETH": "crypto",
    "KXSOL": "crypto",
    "KXCRYPTO": "crypto",
    "KXELECTION": "politics",
    "KXPRES": "politics",
    "KXSENATE": "politics",
    "KXTRUMP": "politics",
    "KXCONGRESS": "politics",
    "KXSCOTUS": "politics",
    "KXISRAEL": "geopolitics",
    "KXUKRAINE": "geopolitics",
    "KXIRAN": "geopolitics",
    "KXCEASEFIRE": "geopolitics",
    "KXTEMP": "weather",
    "KXHURRICANE": "weather",
    "KXSNOW": "weather",
    "KXOSCAR": "entertainment",
    "KXGRAMMY": "entertainment",
    "KXEMMY": "entertainment",
    "KXMVECROSSCATEGORY": "other",   # explicit mixed
}


def categorize_kalshi_ticker(ticker: str | None,
                              event_ticker: str | None = None,
                              question: str | None = None) -> str:
    """Categorize a Kalshi market by ticker prefix first, then event prefix,
    then question text as fallback."""
    for src in (ticker, event_ticker):
        if not src:
            continue
        for prefix, label in _KALSHI_PREFIXES.items():
            if src.upper().startswith(prefix):
                return label
    return categorize_question(question)


def categorize_many(questions: list[str | None]) -> list[str]:
    return [categorize_question(q) for q in questions]
