"""
Signal types for strategy output.
"""

from __future__ import annotations

from enum import IntEnum


class Signal(IntEnum):
    """Trading signal types."""
    STRONG_SHORT = -2
    SHORT = -1
    FLAT = 0
    LONG = 1
    STRONG_LONG = 2


# Convenience mappings
SIGNAL_MAP = {
    "strong_short": Signal.STRONG_SHORT,
    "short": Signal.SHORT,
    "flat": Signal.FLAT,
    "long": Signal.LONG,
    "strong_long": Signal.STRONG_LONG,
}

LONG_SIGNALS = {Signal.LONG, Signal.STRONG_LONG}
SHORT_SIGNALS = {Signal.SHORT, Signal.STRONG_SHORT}
ACTIVE_SIGNALS = LONG_SIGNALS | SHORT_SIGNALS
