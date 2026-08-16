"""Tests for backend.scripts.generate_scenarios.generate_prototype_scenarios().

Coverage checklist
------------------
T01  Exactly nine scenarios are returned.
T02  Complete 3×3 type × timing-band coverage.
T03  Deterministic: two calls return byte-identical JSON.
T04  Exact raw sample schema (eight fields, correct keys).
T05  History contains exactly 72 samples.
T06  Future contains exactly 288 samples.
T07  Five-minute cadence throughout history and future.
T08  First future sample is exactly 5 minutes after last history sample.
T09  Zero history SOC breaches (all samples ≥ 25 %).
T10  Zero history voltage breaches (all samples ≥ 26 V).
T11  Correct first-breach type matches scenario category.
T12  Correct timing band for each scenario.
T13  History-to-future trajectory is continuous (no state reset or snap-back).
T14  No per-step SOC change exceeds 2 percentage points.
T15  No per-step bus-voltage change exceeds 0.30 V.
T16  SOC is within [0, 100] % throughout all samples.
T17  Bus voltage is positive throughout all samples.
T18  Safety metadata is present with correct values.
T19  No duplicate label field (only power_constraint_breach_within_24h).
T20  All nine scenarios carry label = 1 (positive breach examples).
T21  breach_detail contains occurs, breach_type, timing_band, hour_offset.
T22  Scenario IDs are unique.
T23  SOC and voltage are numeric (float) in every sample.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from backend.scripts.generate_scenarios import generate_prototype_scenarios

# ---------------------------------------------------------------------------
# Module-level fixture — generate once, shared across tests
# ---------------------------------------------------------------------------

_SCENARIOS: list[dict[str, Any]] = generate_prototype_scenarios()

REQUIRED_SAMPLE_KEYS: frozenset[str] = frozenset(
    {
        "timestamp",
        "solar_array_current_a",
        "payload_power_draw_w",
        "bus_voltage_v",
        "battery_soc_percent",
        "command_activity",
        "communications_status",
        "image_utility_score",
    }
)

EXPECTED_METADATA: dict[str, str] = {
    "data_source":       "SYNTHETIC",
    "prototype_status":  "NOT_FLIGHT_QUALIFIED",
    "command_authority": "NONE",
    "policy_decision":   "PERMITTED_FOR_SIMULATION_ONLY",
}

# Mapping from scenario-ID prefix to expected breach_type
BREACH_TYPE_BY_PREFIX: dict[str, str] = {
    "PROTO-SOC":  "SOC_ONLY",
    "PROTO-VOLT": "VOLTAGE_ONLY",
    "PROTO-BOTH": "BOTH",
}

# Valid timing bands: (lower_hour_inclusive, upper_hour_inclusive)
TIMING_BAND_RANGES: dict[str, tuple[float, float]] = {
    "early":  (2.0,  8.0),
    "middle": (8.0, 16.0),
    "late":  (16.0, 23.0),
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_utc(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _all_samples(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    return scenario["history"] + scenario["future"]


def _prefix(scenario_id: str) -> str:
    """Return the type prefix, e.g. 'PROTO-SOC' from 'PROTO-SOC-001-EARLY'."""
    parts = scenario_id.split("-")
    return "-".join(parts[:2])


def _timing_from_id(scenario_id: str) -> str:
    """Return the timing suffix in lower case, e.g. 'early'."""
    return scenario_id.split("-")[-1].lower()


# ---------------------------------------------------------------------------
# T01 — Exactly nine scenarios
# ---------------------------------------------------------------------------

def test_exactly_nine_scenarios():
    assert len(_SCENARIOS) == 9


# ---------------------------------------------------------------------------
# T02 — Complete 3×3 coverage
# ---------------------------------------------------------------------------

def test_complete_3x3_coverage():
    expected_types = {"SOC_ONLY", "VOLTAGE_ONLY", "BOTH"}
    expected_bands = {"early", "middle", "late"}

    coverage: set[tuple[str, str]] = set()
    for s in _SCENARIOS:
        sid   = s["scenario_id"]
        bd    = s["breach_detail"]
        btype = bd["breach_type"]
        band  = bd["timing_band"]

        if "SOC" in sid and "BOTH" not in sid:
            cat = "SOC_ONLY"
        elif "VOLT" in sid:
            cat = "VOLTAGE_ONLY"
        else:
            cat = "BOTH"

        coverage.add((cat, band))

    for t in expected_types:
        for b in expected_bands:
            assert (t, b) in coverage, f"Missing coverage for ({t}, {b})"


# ---------------------------------------------------------------------------
# T03 — Determinism
# ---------------------------------------------------------------------------

def test_deterministic_across_calls():
    run_a = generate_prototype_scenarios()
    run_b = generate_prototype_scenarios()
    assert json.dumps(run_a, sort_keys=True) == json.dumps(run_b, sort_keys=True), (
        "generate_prototype_scenarios() returned different results on two calls"
    )


# ---------------------------------------------------------------------------
# T04 — Exact raw sample schema
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_sample_schema(scenario: dict[str, Any]):
    for idx, sample in enumerate(_all_samples(scenario)):
        assert set(sample.keys()) == REQUIRED_SAMPLE_KEYS, (
            f"scenario {scenario['scenario_id']} sample {idx}: "
            f"got keys {set(sample.keys())}"
        )


# ---------------------------------------------------------------------------
# T05 — History count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_history_sample_count(scenario: dict[str, Any]):
    assert len(scenario["history"]) == 72, (
        f"{scenario['scenario_id']}: history len = {len(scenario['history'])}"
    )


# ---------------------------------------------------------------------------
# T06 — Future count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_future_sample_count(scenario: dict[str, Any]):
    assert len(scenario["future"]) == 288, (
        f"{scenario['scenario_id']}: future len = {len(scenario['future'])}"
    )


# ---------------------------------------------------------------------------
# T07 — Five-minute cadence throughout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_five_minute_cadence(scenario: dict[str, Any]):
    samples = _all_samples(scenario)
    for i in range(len(samples) - 1):
        t0 = _parse_utc(samples[i]["timestamp"])
        t1 = _parse_utc(samples[i + 1]["timestamp"])
        gap = (t1 - t0).total_seconds()
        assert gap == 300, (
            f"{scenario['scenario_id']} step {i}→{i+1}: gap={gap}s (expected 300)"
        )


# ---------------------------------------------------------------------------
# T08 — First future sample is 5 minutes after last history sample
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_history_future_timestamp_join(scenario: dict[str, Any]):
    t_hist = _parse_utc(scenario["history"][-1]["timestamp"])
    t_fut  = _parse_utc(scenario["future"][0]["timestamp"])
    gap = (t_fut - t_hist).total_seconds()
    assert gap == 300, (
        f"{scenario['scenario_id']}: history→future gap={gap}s (expected 300)"
    )


# ---------------------------------------------------------------------------
# T09 — Zero history SOC breaches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_no_history_soc_breach(scenario: dict[str, Any]):
    for i, sample in enumerate(scenario["history"]):
        soc = sample["battery_soc_percent"]
        assert soc >= 25.0, (
            f"{scenario['scenario_id']} history[{i}]: SOC={soc:.4f} < 25 %"
        )


# ---------------------------------------------------------------------------
# T10 — Zero history voltage breaches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_no_history_voltage_breach(scenario: dict[str, Any]):
    for i, sample in enumerate(scenario["history"]):
        v = sample["bus_voltage_v"]
        assert v >= 26.0, (
            f"{scenario['scenario_id']} history[{i}]: V={v:.4f} V < 26 V"
        )


# ---------------------------------------------------------------------------
# T11 — Correct first-breach type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_breach_type_matches_category(scenario: dict[str, Any]):
    sid    = scenario["scenario_id"]
    prefix = _prefix(sid)
    bd     = scenario["breach_detail"]
    assert bd["occurs"] is True, f"{sid}: breach_detail.occurs is False"
    expected_type = BREACH_TYPE_BY_PREFIX[prefix]
    assert bd["breach_type"] == expected_type, (
        f"{sid}: breach_type={bd['breach_type']!r}, expected {expected_type!r}"
    )


# ---------------------------------------------------------------------------
# T12 — Correct timing band
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_timing_band_correct(scenario: dict[str, Any]):
    sid    = scenario["scenario_id"]
    timing = _timing_from_id(sid)
    bd     = scenario["breach_detail"]

    assert bd["timing_band"] == timing, (
        f"{sid}: timing_band={bd['timing_band']!r}, expected {timing!r}"
    )
    lo, hi = TIMING_BAND_RANGES[timing]
    hour   = bd["hour_offset"]
    assert lo <= hour <= hi, (
        f"{sid}: hour_offset={hour:.4f} outside [{lo}, {hi}]"
    )


# ---------------------------------------------------------------------------
# T13 — Continuous history-to-future join (no snap-back)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_continuous_trajectory_join(scenario: dict[str, Any]):
    """The values at the join point must be consistent with one update step."""
    h_last = scenario["history"][-1]
    f_first = scenario["future"][0]

    # SOC and voltage change across the join must obey the same step limits
    # as the rest of the trajectory.
    soc_jump = abs(f_first["battery_soc_percent"] - h_last["battery_soc_percent"])
    v_jump   = abs(f_first["bus_voltage_v"]        - h_last["bus_voltage_v"])

    assert soc_jump <= 2.0, (
        f"{scenario['scenario_id']}: SOC jump at join={soc_jump:.6f} % > 2 %"
    )
    assert v_jump <= 0.30, (
        f"{scenario['scenario_id']}: V jump at join={v_jump:.6f} V > 0.30 V"
    )


# ---------------------------------------------------------------------------
# T14 — No per-step SOC change > 2 %
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_soc_step_limit(scenario: dict[str, Any]):
    samples = _all_samples(scenario)
    for i in range(len(samples) - 1):
        delta = abs(samples[i + 1]["battery_soc_percent"] - samples[i]["battery_soc_percent"])
        assert delta <= 2.0, (
            f"{scenario['scenario_id']} step {i}→{i+1}: ΔSOC={delta:.6f} % > 2 %"
        )


# ---------------------------------------------------------------------------
# T15 — No per-step voltage change > 0.30 V
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_voltage_step_limit(scenario: dict[str, Any]):
    samples = _all_samples(scenario)
    for i in range(len(samples) - 1):
        delta = abs(samples[i + 1]["bus_voltage_v"] - samples[i]["bus_voltage_v"])
        assert delta <= 0.30, (
            f"{scenario['scenario_id']} step {i}→{i+1}: ΔV={delta:.6f} V > 0.30 V"
        )


# ---------------------------------------------------------------------------
# T16 — SOC within [0, 100] %
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_soc_numeric_bounds(scenario: dict[str, Any]):
    for i, sample in enumerate(_all_samples(scenario)):
        soc = sample["battery_soc_percent"]
        assert isinstance(soc, (int, float)), f"SOC is not numeric at index {i}"
        assert 0.0 <= soc <= 100.0, (
            f"{scenario['scenario_id']} sample {i}: SOC={soc:.4f} outside [0, 100]"
        )


# ---------------------------------------------------------------------------
# T17 — Bus voltage is positive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_voltage_positive(scenario: dict[str, Any]):
    for i, sample in enumerate(_all_samples(scenario)):
        v = sample["bus_voltage_v"]
        assert isinstance(v, (int, float)), f"Voltage is not numeric at index {i}"
        assert v > 0.0, (
            f"{scenario['scenario_id']} sample {i}: V={v:.4f} ≤ 0"
        )


# ---------------------------------------------------------------------------
# T18 — Safety metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_safety_metadata(scenario: dict[str, Any]):
    meta = scenario.get("metadata", {})
    for key, expected_value in EXPECTED_METADATA.items():
        assert key in meta, f"{scenario['scenario_id']}: metadata missing key {key!r}"
        assert meta[key] == expected_value, (
            f"{scenario['scenario_id']}: metadata[{key!r}]={meta[key]!r}, "
            f"expected {expected_value!r}"
        )


# ---------------------------------------------------------------------------
# T19 — No duplicate label field
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_no_duplicate_label_field(scenario: dict[str, Any]):
    assert "label" not in scenario, (
        f"{scenario['scenario_id']}: spurious top-level 'label' field found"
    )
    assert "power_constraint_breach_within_24h" in scenario, (
        f"{scenario['scenario_id']}: missing 'power_constraint_breach_within_24h'"
    )


# ---------------------------------------------------------------------------
# T20 — All scenarios carry label = 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_all_scenarios_are_positive(scenario: dict[str, Any]):
    label = scenario["power_constraint_breach_within_24h"]
    assert label == 1, f"{scenario['scenario_id']}: label={label}, expected 1"


# ---------------------------------------------------------------------------
# T21 — breach_detail structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_breach_detail_structure(scenario: dict[str, Any]):
    bd = scenario.get("breach_detail", {})
    for field in ("occurs", "breach_type", "timing_band", "hour_offset"):
        assert field in bd, (
            f"{scenario['scenario_id']}: breach_detail missing field {field!r}"
        )
    assert bd["occurs"] is True
    assert bd["breach_type"] in ("SOC_ONLY", "VOLTAGE_ONLY", "BOTH")
    assert bd["timing_band"] in ("early", "middle", "late")
    assert isinstance(bd["hour_offset"], (int, float))
    assert bd["hour_offset"] > 0.0


# ---------------------------------------------------------------------------
# T22 — Unique scenario IDs
# ---------------------------------------------------------------------------

def test_unique_scenario_ids():
    ids = [s["scenario_id"] for s in _SCENARIOS]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"


# ---------------------------------------------------------------------------
# T23 — SOC and voltage are numeric in every sample
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s["scenario_id"] for s in _SCENARIOS])
def test_numeric_fields(scenario: dict[str, Any]):
    for i, sample in enumerate(_all_samples(scenario)):
        assert isinstance(sample["battery_soc_percent"], (int, float)), (
            f"{scenario['scenario_id']} sample {i}: battery_soc_percent not numeric"
        )
        assert isinstance(sample["bus_voltage_v"], (int, float)), (
            f"{scenario['scenario_id']} sample {i}: bus_voltage_v not numeric"
        )
        assert isinstance(sample["solar_array_current_a"], (int, float)), (
            f"{scenario['scenario_id']} sample {i}: solar_array_current_a not numeric"
        )
        assert isinstance(sample["payload_power_draw_w"], (int, float)), (
            f"{scenario['scenario_id']} sample {i}: payload_power_draw_w not numeric"
        )
