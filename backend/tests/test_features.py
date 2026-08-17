"""Unit tests for backend.app.features.

Coverage checklist
------------------
F01  Constant-series ols_slope returns exactly 0.0.
F02  Known positive slope (hand-computed).
F03  Known negative slope (hand-computed).
F04  Single-element series returns 0.0.
F05  Two-element series slope matches hand-computed value.
F06  extract_features returns exactly 12 keys.
F07  Correct 12-key names.
F08  soc_latest equals the final sample value.
F09  soc_mean equals the arithmetic mean.
F10  soc_min equals the minimum.
F11  voltage_latest equals the final sample value.
F12  voltage_min equals the minimum.
F13  solar_current_mean equals the arithmetic mean.
F14  payload_draw_mean equals the arithmetic mean.
F15  payload_draw_max equals the maximum.
F16  high_draw_fraction — zero when no sample exceeds 100 W.
F17  high_draw_fraction — correct fraction when some samples exceed 100 W.
F18  high_draw_fraction — 1.0 when all samples exceed 100 W.
F19  soc_slope sign is positive for a rising series.
F20  soc_slope sign is negative for a declining series.
F21  voltage_slope is zero for a constant series.
F22  solar_current_slope sign matches trend direction.
F23  Slope units are per hour (not per minute).
F24  Rejected: wrong sample count (too few).
F25  Rejected: wrong sample count (too many).
F26  Rejected: missing battery_soc_percent field.
F27  Rejected: missing bus_voltage_v field.
F28  Rejected: missing solar_array_current_a field.
F29  Rejected: missing payload_power_draw_w field.
F30  Rejected: missing timestamp field.
F31  Rejected: boolean value for a numeric field.
F32  Rejected: string value for a numeric field.
F33  Rejected: NaN value.
F34  Rejected: +infinity value.
F35  Rejected: −infinity value.
F36  Rejected: invalid timestamp format.
F37  Rejected: non-ascending (duplicate) timestamps.
F38  Rejected: non-ascending (reversed) timestamps.
F39  Rejected: wrong cadence (6-minute step instead of 5-minute).
F40  Input list is not mutated.
F41  Input dicts are not mutated.
F42  Excluded fields (scenario_id, split, communications_status, etc.)
     present in samples do not affect feature values.
"""

from __future__ import annotations

import copy
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from backend.app.features import extract_features, ols_slope

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N = 72
_STEP_MINUTES = 5.0
_EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_STEP_S = 300  # 5 minutes in seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(index: int) -> str:
    """ISO-8601 UTC timestamp for sample index (5-min cadence from _EPOCH)."""
    return (_EPOCH + timedelta(seconds=index * _STEP_S)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _make_sample(
    index: int,
    soc: float = 60.0,
    voltage: float = 28.0,
    solar: float = 5.0,
    draw: float = 50.0,
    **extra: Any,
) -> dict[str, Any]:
    """Build a minimal valid telemetry sample."""
    return {
        "timestamp":              _ts(index),
        "battery_soc_percent":    soc,
        "bus_voltage_v":          voltage,
        "solar_array_current_a":  solar,
        "payload_power_draw_w":   draw,
        **extra,
    }


def _make_samples(
    soc: float = 60.0,
    voltage: float = 28.0,
    solar: float = 5.0,
    draw: float = 50.0,
) -> list[dict[str, Any]]:
    """Return 72 identical samples with constant field values."""
    return [_make_sample(i, soc=soc, voltage=voltage, solar=solar, draw=draw)
            for i in range(_N)]


def _make_samples_varying(
    soc_values: list[float],
    voltage_values: list[float] | None = None,
    solar_values: list[float] | None = None,
    draw_values: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Build 72 samples where each field can vary independently."""
    assert len(soc_values) == _N
    v = voltage_values or [28.0] * _N
    s = solar_values   or [5.0]  * _N
    d = draw_values    or [50.0] * _N
    return [_make_sample(i, soc=soc_values[i], voltage=v[i], solar=s[i], draw=d[i])
            for i in range(_N)]


# ---------------------------------------------------------------------------
# F01–F05  ols_slope unit tests
# ---------------------------------------------------------------------------

def test_F01_constant_series_slope_is_zero() -> None:
    values = [42.0] * 10
    assert ols_slope(values) == 0.0


def test_F02_known_positive_slope() -> None:
    # y_i = i (per sample);  t_i = i * 5/60 hours
    # slope = dy/dt = 1 / (5/60) = 12.0 per hour
    n = 10
    values = [float(i) for i in range(n)]
    expected = 1.0 / (_STEP_MINUTES / 60.0)   # 12.0 /h
    result = ols_slope(values)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_F03_known_negative_slope() -> None:
    # y_i = -i;  slope = -1 / (5/60) = -12.0 per hour
    n = 10
    values = [float(-i) for i in range(n)]
    expected = -1.0 / (_STEP_MINUTES / 60.0)  # -12.0 /h
    result = ols_slope(values)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_F04_single_element_returns_zero() -> None:
    assert ols_slope([99.9]) == 0.0


def test_F05_two_element_slope_hand_computed() -> None:
    # y = [0.0, 1.0]; Δt = 5/60 h; slope = 1/(5/60) = 12.0
    result = ols_slope([0.0, 1.0])
    assert math.isclose(result, 12.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# F06–F07  Output structure
# ---------------------------------------------------------------------------

_EXPECTED_KEYS = frozenset({
    "soc_latest",
    "soc_mean",
    "soc_min",
    "soc_slope",
    "voltage_latest",
    "voltage_min",
    "voltage_slope",
    "solar_current_mean",
    "solar_current_slope",
    "payload_draw_mean",
    "payload_draw_max",
    "high_draw_fraction",
})


def test_F06_exactly_12_keys() -> None:
    features = extract_features(_make_samples())
    assert len(features) == 12


def test_F07_correct_12_key_names() -> None:
    features = extract_features(_make_samples())
    assert set(features.keys()) == _EXPECTED_KEYS


# ---------------------------------------------------------------------------
# F08–F15  Correct latest / mean / min / max calculations
# ---------------------------------------------------------------------------

def test_F08_soc_latest_is_final_sample_value() -> None:
    soc = [float(i) for i in range(30, 30 + _N)]  # 30..101
    # clamp to valid range
    soc = [min(s, 95.0) for s in soc]
    features = extract_features(_make_samples_varying(soc_values=soc))
    assert features["soc_latest"] == soc[-1]


def test_F09_soc_mean_is_arithmetic_mean() -> None:
    soc = [60.0 + i * 0.1 for i in range(_N)]
    features = extract_features(_make_samples_varying(soc_values=soc))
    expected = sum(soc) / _N
    assert math.isclose(features["soc_mean"], expected, rel_tol=1e-12)


def test_F10_soc_min_is_minimum() -> None:
    soc = [80.0] * _N
    soc[10] = 30.0  # inject a dip
    features = extract_features(_make_samples_varying(soc_values=soc))
    assert features["soc_min"] == 30.0


def test_F11_voltage_latest_is_final_sample_value() -> None:
    volt = [27.0 + i * 0.01 for i in range(_N)]
    features = extract_features(_make_samples_varying(
        soc_values=[60.0] * _N, voltage_values=volt
    ))
    assert features["voltage_latest"] == volt[-1]


def test_F12_voltage_min_is_minimum() -> None:
    volt = [28.0] * _N
    volt[35] = 26.5  # inject a dip
    features = extract_features(_make_samples_varying(
        soc_values=[60.0] * _N, voltage_values=volt
    ))
    assert features["voltage_min"] == 26.5


def test_F13_solar_current_mean_is_arithmetic_mean() -> None:
    solar = [3.0 + i * 0.05 for i in range(_N)]
    features = extract_features(_make_samples_varying(
        soc_values=[60.0] * _N, solar_values=solar
    ))
    expected = sum(solar) / _N
    assert math.isclose(features["solar_current_mean"], expected, rel_tol=1e-12)


def test_F14_payload_draw_mean_is_arithmetic_mean() -> None:
    draw = [40.0 + i * 0.5 for i in range(_N)]
    features = extract_features(_make_samples_varying(
        soc_values=[60.0] * _N, draw_values=draw
    ))
    expected = sum(draw) / _N
    assert math.isclose(features["payload_draw_mean"], expected, rel_tol=1e-12)


def test_F15_payload_draw_max_is_maximum() -> None:
    draw = [50.0] * _N
    draw[20] = 150.0  # spike
    features = extract_features(_make_samples_varying(
        soc_values=[60.0] * _N, draw_values=draw
    ))
    assert features["payload_draw_max"] == 150.0


# ---------------------------------------------------------------------------
# F16–F18  high_draw_fraction
# ---------------------------------------------------------------------------

def test_F16_high_draw_fraction_zero_when_none_exceed_threshold() -> None:
    features = extract_features(_make_samples(draw=100.0))
    # 100.0 is not > 100.0
    assert features["high_draw_fraction"] == 0.0


def test_F17_high_draw_fraction_correct_partial() -> None:
    draw = [50.0] * _N
    # Set the first 18 samples above the threshold (18/72 = 0.25)
    for i in range(18):
        draw[i] = 101.0
    features = extract_features(_make_samples_varying(
        soc_values=[60.0] * _N, draw_values=draw
    ))
    assert math.isclose(features["high_draw_fraction"], 18 / _N, rel_tol=1e-12)


def test_F18_high_draw_fraction_one_when_all_exceed_threshold() -> None:
    features = extract_features(_make_samples(draw=200.0))
    assert features["high_draw_fraction"] == 1.0


# ---------------------------------------------------------------------------
# F19–F22  Slope sign tests
# ---------------------------------------------------------------------------

def test_F19_soc_slope_positive_for_rising_series() -> None:
    soc = [50.0 + i * 0.1 for i in range(_N)]
    features = extract_features(_make_samples_varying(soc_values=soc))
    assert features["soc_slope"] > 0.0


def test_F20_soc_slope_negative_for_declining_series() -> None:
    soc = [90.0 - i * 0.1 for i in range(_N)]
    features = extract_features(_make_samples_varying(soc_values=soc))
    assert features["soc_slope"] < 0.0


def test_F21_voltage_slope_zero_for_constant_series() -> None:
    features = extract_features(_make_samples(voltage=28.0))
    assert features["voltage_slope"] == 0.0


def test_F22_solar_current_slope_sign_matches_trend() -> None:
    solar_up   = [3.0 + i * 0.01 for i in range(_N)]
    solar_down = [5.0 - i * 0.01 for i in range(_N)]
    feat_up   = extract_features(_make_samples_varying(
        soc_values=[60.0] * _N, solar_values=solar_up
    ))
    feat_down = extract_features(_make_samples_varying(
        soc_values=[60.0] * _N, solar_values=solar_down
    ))
    assert feat_up["solar_current_slope"]   > 0.0
    assert feat_down["solar_current_slope"] < 0.0


# ---------------------------------------------------------------------------
# F23  Slope units are per hour, not per minute
# ---------------------------------------------------------------------------

def test_F23_slope_units_are_per_hour() -> None:
    # y increases by 1.0 per 5-minute step → slope should be 12.0 /h, not 0.2 /min
    soc = [50.0 + i * 1.0 for i in range(_N)]
    features = extract_features(_make_samples_varying(soc_values=soc))
    # 1 unit per step × 12 steps per hour = 12 units per hour
    expected_slope = 12.0
    assert math.isclose(features["soc_slope"], expected_slope, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# F24–F25  Wrong sample count
# ---------------------------------------------------------------------------

def test_F24_rejected_too_few_samples() -> None:
    samples = [_make_sample(i) for i in range(71)]
    with pytest.raises(ValueError, match="exactly 72 samples"):
        extract_features(samples)


def test_F25_rejected_too_many_samples() -> None:
    samples = [_make_sample(i) for i in range(73)]
    with pytest.raises(ValueError, match="exactly 72 samples"):
        extract_features(samples)


# ---------------------------------------------------------------------------
# F26–F30  Missing fields
# ---------------------------------------------------------------------------

def test_F26_rejected_missing_battery_soc_percent() -> None:
    samples = _make_samples()
    del samples[5]["battery_soc_percent"]
    with pytest.raises(ValueError, match="battery_soc_percent"):
        extract_features(samples)


def test_F27_rejected_missing_bus_voltage_v() -> None:
    samples = _make_samples()
    del samples[0]["bus_voltage_v"]
    with pytest.raises(ValueError, match="bus_voltage_v"):
        extract_features(samples)


def test_F28_rejected_missing_solar_array_current_a() -> None:
    samples = _make_samples()
    del samples[10]["solar_array_current_a"]
    with pytest.raises(ValueError, match="solar_array_current_a"):
        extract_features(samples)


def test_F29_rejected_missing_payload_power_draw_w() -> None:
    samples = _make_samples()
    del samples[71]["payload_power_draw_w"]
    with pytest.raises(ValueError, match="payload_power_draw_w"):
        extract_features(samples)


def test_F30_rejected_missing_timestamp() -> None:
    samples = _make_samples()
    del samples[0]["timestamp"]
    with pytest.raises(ValueError, match="timestamp"):
        extract_features(samples)


# ---------------------------------------------------------------------------
# F31–F33  Bad numeric types
# ---------------------------------------------------------------------------

def test_F31_rejected_boolean_value_for_numeric_field() -> None:
    samples = _make_samples()
    samples[0]["battery_soc_percent"] = True
    with pytest.raises(ValueError, match="boolean"):
        extract_features(samples)


def test_F32_rejected_string_value_for_numeric_field() -> None:
    samples = _make_samples()
    samples[3]["bus_voltage_v"] = "27.5"
    with pytest.raises(ValueError, match="str"):
        extract_features(samples)


def test_F33_rejected_nan_value() -> None:
    samples = _make_samples()
    samples[0]["battery_soc_percent"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        extract_features(samples)


def test_F34_rejected_positive_infinity() -> None:
    samples = _make_samples()
    samples[0]["battery_soc_percent"] = float("inf")
    with pytest.raises(ValueError, match="infinity"):
        extract_features(samples)


def test_F35_rejected_negative_infinity() -> None:
    samples = _make_samples()
    samples[0]["bus_voltage_v"] = float("-inf")
    with pytest.raises(ValueError, match="infinity"):
        extract_features(samples)


# ---------------------------------------------------------------------------
# F36  Invalid timestamp format
# ---------------------------------------------------------------------------

def test_F36_rejected_invalid_timestamp_format() -> None:
    samples = _make_samples()
    samples[2]["timestamp"] = "2026-01-01 00:10:00"   # space instead of 'T'
    with pytest.raises(ValueError, match="invalid format"):
        extract_features(samples)


# ---------------------------------------------------------------------------
# F37–F38  Non-ascending timestamps
# ---------------------------------------------------------------------------

def test_F37_rejected_duplicate_timestamps() -> None:
    samples = _make_samples()
    # Make sample[5] have the same timestamp as sample[4]
    samples[5]["timestamp"] = samples[4]["timestamp"]
    with pytest.raises(ValueError, match="strictly ascending"):
        extract_features(samples)


def test_F38_rejected_reversed_timestamps() -> None:
    samples = _make_samples()
    # Give sample[11] a timestamp that is BEFORE sample[10] so that the
    # delta from sample[10] → sample[11] is negative (−300 s), which
    # triggers the "strictly ascending" check.
    samples[11]["timestamp"] = (
        _EPOCH + timedelta(seconds=9 * _STEP_S)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    with pytest.raises(ValueError, match="strictly ascending"):
        extract_features(samples)


# ---------------------------------------------------------------------------
# F39  Wrong cadence
# ---------------------------------------------------------------------------

def test_F39_rejected_wrong_cadence_6_minute_step() -> None:
    # Build samples with a 6-minute step (360 s) instead of 5 minutes (300 s)
    epoch = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    wrong_step_s = 360
    samples = [
        {
            "timestamp": (epoch + timedelta(seconds=i * wrong_step_s)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "battery_soc_percent":   60.0,
            "bus_voltage_v":         28.0,
            "solar_array_current_a": 5.0,
            "payload_power_draw_w":  50.0,
        }
        for i in range(_N)
    ]
    with pytest.raises(ValueError, match="cadence"):
        extract_features(samples)


# ---------------------------------------------------------------------------
# F40–F41  Input immutability
# ---------------------------------------------------------------------------

def test_F40_input_list_is_not_mutated() -> None:
    samples = _make_samples()
    original_len = len(samples)
    original_first_id = id(samples[0])
    extract_features(samples)
    assert len(samples) == original_len
    assert id(samples[0]) == original_first_id


def test_F41_input_dicts_are_not_mutated() -> None:
    samples = _make_samples()
    originals = [copy.deepcopy(s) for s in samples]
    extract_features(samples)
    for i, (s, orig) in enumerate(zip(samples, originals)):
        assert s == orig, f"sample[{i}] was mutated"


# ---------------------------------------------------------------------------
# F42  Excluded fields do not affect features
# ---------------------------------------------------------------------------

def test_F42_excluded_fields_do_not_affect_features() -> None:
    """Adding forbidden/extra fields to samples must not change feature values."""
    samples_clean = _make_samples()
    features_clean = extract_features(samples_clean)

    # Add all the excluded fields that must not be read
    samples_extra = _make_samples()
    for i, s in enumerate(samples_extra):
        s["scenario_id"]           = "SYNTH-SHOULD-BE-IGNORED"
        s["split"]                 = "train"
        s["power_constraint_breach_within_24h"] = 1
        s["breach_detail"]         = {"occurs": True, "breach_type": "SOC_ONLY"}
        s["communications_status"] = "GROUND_CONTACT"
        s["command_activity"]      = "PAYLOAD_IMAGING_BURST"
        s["image_utility_score"]   = 0.99

    features_extra = extract_features(samples_extra)
    assert features_clean == features_extra
