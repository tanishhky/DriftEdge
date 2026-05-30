"""Sizing engine — Kelly criterion for prediction markets with fractional dampening.

For Yes at price c with estimated probability p:
    f* = (p - c) / (1 - c)
    f  = kelly_fraction * f*   (default 0.25 = quarter-Kelly)

See docs/research/architecture.md §4.
"""
