# 1. Record architecture decisions

Date: 2026-05-30

## Status

Accepted.

## Context

DriftEdge will make several non-obvious choices: logit-vol instead of raw vol, quarter-Kelly default, max-of-top-2 signal fusion, JSONL logging, Polymarket-first with CLOB API. These will be questioned later. We want a paper trail.

## Decision

Every meaningful design choice gets an ADR in `docs/decisions/NNNN-<slug>.md`.

Format: Context, Decision, Consequences. Date and status header.

## Consequences

- Slight overhead per decision.
- Future contributors can read *why* something was built the way it was without re-deriving.
- Encourages thinking before coding.
