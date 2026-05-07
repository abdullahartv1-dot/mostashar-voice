"""Tracks GPU-hour cost per stage and per session."""
from typing import List, Tuple, Dict


def calc_cost(seconds: float, rate_per_hour: float) -> float:
    """Returns USD cost rounded to 4 decimal places."""
    return round(seconds / 3600 * rate_per_hour, 4)


def format_breakdown(
    stages: List[Tuple[str, float]],
    rate_per_hour: float,
) -> Dict[str, float]:
    """Convert list of (stage_name, seconds) into a cost breakdown dict.

    Returns:
        {"stt": 0.0045, "diar": 0.0023, ..., "total_usd": 0.0068}
    """
    out = {name: calc_cost(secs, rate_per_hour) for name, secs in stages}
    total_secs = sum(s for _, s in stages)
    out["total_usd"] = calc_cost(total_secs, rate_per_hour)
    return out
