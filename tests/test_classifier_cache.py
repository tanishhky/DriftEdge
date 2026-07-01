"""Classifier cache: corruption self-heal, atomic write, schema robustness.

Two production bugs are covered:
  1. The 2026-06-22 Kuber brick (ported here where it was still latent): a
     non-atomic write interrupted by a restart truncated the cache, and the
     corrupt-but-exists read then threw on every market refresh.
  2. The recurring KeyError('market_id') in the DriftEdge poll loop (49x on
     2026-06-30): classify_and_cache did cache["market_id"] assuming the
     column existed, so a cache read that came back without it crashed the
     iteration. _load_cache now guarantees the schema.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from driftedge import classifier as C


def test_corrupt_cache_self_heals(tmp_path: Path):
    p = tmp_path / "market_categories.parquet"
    p.write_bytes(b"not a parquet file")
    df = C._load_cache(tmp_path)
    assert len(df) == 0 and not p.exists()
    assert (tmp_path / "market_categories.corrupt").exists()


def test_save_cache_is_atomic(tmp_path: Path):
    C._save_cache(tmp_path, pd.DataFrame([{c: None for c in C._EMPTY_CACHE_COLS}]))
    assert not (tmp_path / "market_categories.parquet.tmp").exists()
    assert len(C._load_cache(tmp_path)) == 1


def test_partial_schema_cache_does_not_keyerror(tmp_path: Path):
    """A cache parquet missing the market_id column must NOT raise KeyError in
    classify_and_cache (the 2026-06-30 poll-loop bug)."""
    partial = pd.DataFrame({"venue": ["polymarket"], "category": ["sports"]})
    pq.write_table(pa.Table.from_pandas(partial, preserve_index=False),
                   tmp_path / "market_categories.parquet")
    # loads with the full schema restored
    cache = C._load_cache(tmp_path)
    assert "market_id" in cache.columns
    # and the hot-path caller runs cleanly
    res = C.classify_and_cache(tmp_path, venue="polymarket", market_id="123",
                               question="Will X win?")
    assert res.category is not None
