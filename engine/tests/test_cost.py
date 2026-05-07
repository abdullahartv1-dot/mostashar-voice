"""Tests for cost_tracker."""
from engine.cost_tracker import calc_cost, format_breakdown


def test_calc_cost_one_hour_at_27_cents():
    assert calc_cost(seconds=3600, rate_per_hour=0.27) == 0.27


def test_calc_cost_60_seconds():
    # 60s = 1/60 hour = 0.27/60 = 0.0045
    assert calc_cost(seconds=60, rate_per_hour=0.27) == round(0.0045, 4)


def test_calc_cost_zero_seconds_is_zero():
    assert calc_cost(seconds=0, rate_per_hour=0.27) == 0.0


def test_format_breakdown_returns_dict_with_keys():
    breakdown = format_breakdown([
        ("stt", 60),
        ("diar", 30),
    ], rate_per_hour=0.27)
    assert "stt" in breakdown
    assert "diar" in breakdown
    assert "total_usd" in breakdown
    assert breakdown["total_usd"] == round((60 + 30) / 3600 * 0.27, 4)
