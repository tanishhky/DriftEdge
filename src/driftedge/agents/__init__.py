"""Self-managed agents whose entry/exit doesn't fit the global EntryRule.

Each module here exports a `tick(data_dir, markets, as_of_ts=None)` function
that the daemon calls after the standard `paper.tick`. Agents read the same
book/markets data, write to the same `paper_trades.parquet`, and update the
same `paper_state.parquet` — they just operate under their own rules.
"""
