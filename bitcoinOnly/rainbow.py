"""
rainbow.py
- Computes BTC rainbow thresholds (blue/green/yellow) for a given datetime.

This uses the classic "original rainbow chart" regression form:
    price = 10^(A * ln(weeks_since_2009_01_09) + B)
Bitbo documents an example of this regression form. :contentReference[oaicite:3]{index=3}

We then apply multipliers to define band boundaries that you can tune:
    blue  (very undervalued)  -> lower threshold
    green (undervalued)       -> middle threshold
    yellow(overvalued-ish)    -> upper threshold
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

# --- Base regression constants (tune if you want a different rainbow model) ---
A = 3.109106
B = -8.164198
GENESIS = dt.datetime(2009, 1, 9, tzinfo=dt.timezone.utc)

# --- Band multipliers (tune these!) ---
# Interpretation:
# - If price < green_level: target 20% allocation in BTC
# - If price < blue_level : target 100% allocation in BTC
# - If price > yellow_level: sell all BTC
#
# blue_level should be LOWER than green_level.
BLUE_MULTIPLIER = 0.55
GREEN_MULTIPLIER = 0.85
YELLOW_MULTIPLIER = 1.60


@dataclass(frozen=True)
class RainbowLevels:
    baseline: float
    blue_level: float
    green_level: float
    yellow_level: float


def _weeks_since_genesis(t: dt.datetime) -> float:
    if t.tzinfo is None:
        # assume UTC if naive
        t = t.replace(tzinfo=dt.timezone.utc)
    delta = t - GENESIS
    days = delta.total_seconds() / 86400.0
    weeks = days / 7.0
    # Avoid ln(0) for extremely early dates:
    return max(weeks, 1e-6)


def baseline_price(t: dt.datetime) -> float:
    weeks = _weeks_since_genesis(t)
    # price = 10^(A*ln(weeks) + B)
    return 10 ** (A * math.log(weeks) + B)


def get_levels(
    t: dt.datetime | None = None,
    *,
    blue_multiplier: float = BLUE_MULTIPLIER,
    green_multiplier: float = GREEN_MULTIPLIER,
    yellow_multiplier: float = YELLOW_MULTIPLIER,
) -> RainbowLevels:
    if t is None:
        t = dt.datetime.now(dt.timezone.utc)

    base = baseline_price(t)
    return RainbowLevels(
        baseline=base,
        blue_level=base * blue_multiplier,
        green_level=base * green_multiplier,
        yellow_level=base * yellow_multiplier,
    )
