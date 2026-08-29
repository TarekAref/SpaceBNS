"""Tests for backend.app.policy.apply_power_thresholds.

Coverage:
- Each of the four thresholds fires independently.
- Values exactly at each threshold (boundary — must NOT fire).
- Values strictly inside each threshold (must fire).
- All four fire together.
- Nominal sample produces no findings.
- Canonical finding order is preserved.
- Input dict is not mutated by the function.
"""

from __future__ import annotations

import copy

import pytest

from backend.app.policy import apply_power_thresholds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nominal() -> dict:
    """A sample that does not breach any threshold."""
    return {
        "bus_voltage_v": 28.0,
        "battery_soc_percent": 60.0,
        "payload_power_draw_w": 50.0,
        "image_utility_score": 0.80,
    }


def _finding_codes(sample: dict) -> list[str]:
    return [f["code"] for f in apply_power_thresholds(sample)]


# ---------------------------------------------------------------------------
# Nominal — no findings
# ---------------------------------------------------------------------------

def test_nominal_sample_produces_no_findings() -> None:
    assert apply_power_thresholds(_nominal()) == []


# ---------------------------------------------------------------------------
# BUS_VOLTAGE_LOW  (threshold: < 26.0 V)
# ---------------------------------------------------------------------------

def test_bus_voltage_below_threshold_fires() -> None:
    sample = {**_nominal(), "bus_voltage_v": 25.9}
    codes = _finding_codes(sample)
    assert "BUS_VOLTAGE_LOW" in codes


def test_bus_voltage_at_threshold_does_not_fire() -> None:
    sample = {**_nominal(), "bus_voltage_v": 26.0}
    codes = _finding_codes(sample)
    assert "BUS_VOLTAGE_LOW" not in codes


def test_bus_voltage_above_threshold_does_not_fire() -> None:
    sample = {**_nominal(), "bus_voltage_v": 26.1}
    codes = _finding_codes(sample)
    assert "BUS_VOLTAGE_LOW" not in codes


def test_bus_voltage_finding_fields() -> None:
    sample = {**_nominal(), "bus_voltage_v": 25.0}
    findings = apply_power_thresholds(sample)
    match = next(f for f in findings if f["code"] == "BUS_VOLTAGE_LOW")
    assert match["evidence"] == "bus_voltage_v below public demo threshold"


# ---------------------------------------------------------------------------
# BATTERY_SOC_LOW  (threshold: < 25.0 %)
# ---------------------------------------------------------------------------

def test_battery_soc_below_threshold_fires() -> None:
    sample = {**_nominal(), "battery_soc_percent": 24.9}
    codes = _finding_codes(sample)
    assert "BATTERY_SOC_LOW" in codes


def test_battery_soc_at_threshold_does_not_fire() -> None:
    sample = {**_nominal(), "battery_soc_percent": 25.0}
    codes = _finding_codes(sample)
    assert "BATTERY_SOC_LOW" not in codes


def test_battery_soc_above_threshold_does_not_fire() -> None:
    sample = {**_nominal(), "battery_soc_percent": 25.1}
    codes = _finding_codes(sample)
    assert "BATTERY_SOC_LOW" not in codes


def test_battery_soc_finding_fields() -> None:
    sample = {**_nominal(), "battery_soc_percent": 10.0}
    findings = apply_power_thresholds(sample)
    match = next(f for f in findings if f["code"] == "BATTERY_SOC_LOW")
    assert match["evidence"] == "battery_soc_percent below public demo threshold"


# ---------------------------------------------------------------------------
# PAYLOAD_LOAD_HIGH  (threshold: > 100.0 W)
# ---------------------------------------------------------------------------

def test_payload_draw_above_threshold_fires() -> None:
    sample = {**_nominal(), "payload_power_draw_w": 100.1}
    codes = _finding_codes(sample)
    assert "PAYLOAD_LOAD_HIGH" in codes


def test_payload_draw_at_threshold_does_not_fire() -> None:
    sample = {**_nominal(), "payload_power_draw_w": 100.0}
    codes = _finding_codes(sample)
    assert "PAYLOAD_LOAD_HIGH" not in codes


def test_payload_draw_below_threshold_does_not_fire() -> None:
    sample = {**_nominal(), "payload_power_draw_w": 99.9}
    codes = _finding_codes(sample)
    assert "PAYLOAD_LOAD_HIGH" not in codes


def test_payload_draw_finding_fields() -> None:
    sample = {**_nominal(), "payload_power_draw_w": 110.0}
    findings = apply_power_thresholds(sample)
    match = next(f for f in findings if f["code"] == "PAYLOAD_LOAD_HIGH")
    assert match["evidence"] == "payload_power_draw_w above public demo threshold"


# ---------------------------------------------------------------------------
# IMAGE_UTILITY_LOW  (threshold: < 0.30)
# ---------------------------------------------------------------------------

def test_image_utility_below_threshold_fires() -> None:
    sample = {**_nominal(), "image_utility_score": 0.29}
    codes = _finding_codes(sample)
    assert "IMAGE_UTILITY_LOW" in codes


def test_image_utility_at_threshold_does_not_fire() -> None:
    sample = {**_nominal(), "image_utility_score": 0.30}
    codes = _finding_codes(sample)
    assert "IMAGE_UTILITY_LOW" not in codes


def test_image_utility_above_threshold_does_not_fire() -> None:
    sample = {**_nominal(), "image_utility_score": 0.31}
    codes = _finding_codes(sample)
    assert "IMAGE_UTILITY_LOW" not in codes


def test_image_utility_finding_fields() -> None:
    sample = {**_nominal(), "image_utility_score": 0.10}
    findings = apply_power_thresholds(sample)
    match = next(f for f in findings if f["code"] == "IMAGE_UTILITY_LOW")
    assert match["evidence"] == "edge-reported image utility below public demo threshold"


# ---------------------------------------------------------------------------
# All four fire together
# ---------------------------------------------------------------------------

def test_all_four_findings_fire() -> None:
    sample = {
        "bus_voltage_v": 25.0,
        "battery_soc_percent": 20.0,
        "payload_power_draw_w": 110.0,
        "image_utility_score": 0.10,
    }
    codes = _finding_codes(sample)
    assert codes == [
        "BUS_VOLTAGE_LOW",
        "BATTERY_SOC_LOW",
        "PAYLOAD_LOAD_HIGH",
        "IMAGE_UTILITY_LOW",
    ]


# ---------------------------------------------------------------------------
# Canonical finding order
# ---------------------------------------------------------------------------

def test_finding_order_is_canonical() -> None:
    """Order must be BUS_VOLTAGE_LOW, BATTERY_SOC_LOW, PAYLOAD_LOAD_HIGH,
    IMAGE_UTILITY_LOW regardless of which subset fires."""
    sample = {
        "bus_voltage_v": 25.0,
        "battery_soc_percent": 20.0,
        "payload_power_draw_w": 110.0,
        "image_utility_score": 0.10,
    }
    codes = _finding_codes(sample)
    assert codes.index("BUS_VOLTAGE_LOW") < codes.index("BATTERY_SOC_LOW")
    assert codes.index("BATTERY_SOC_LOW") < codes.index("PAYLOAD_LOAD_HIGH")
    assert codes.index("PAYLOAD_LOAD_HIGH") < codes.index("IMAGE_UTILITY_LOW")


def test_partial_finding_order_voltage_before_soc() -> None:
    sample = {**_nominal(), "bus_voltage_v": 25.0, "battery_soc_percent": 20.0}
    codes = _finding_codes(sample)
    assert codes.index("BUS_VOLTAGE_LOW") < codes.index("BATTERY_SOC_LOW")


# ---------------------------------------------------------------------------
# Input immutability
# ---------------------------------------------------------------------------

def test_input_dict_is_not_mutated() -> None:
    sample = {
        "bus_voltage_v": 25.0,
        "battery_soc_percent": 20.0,
        "payload_power_draw_w": 110.0,
        "image_utility_score": 0.10,
    }
    original = copy.deepcopy(sample)
    apply_power_thresholds(sample)
    assert sample == original


def test_input_dict_extra_keys_are_ignored_and_preserved() -> None:
    sample = {**_nominal(), "extra_key": "should_survive"}
    apply_power_thresholds(sample)
    assert sample["extra_key"] == "should_survive"
