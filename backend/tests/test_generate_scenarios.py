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


# =============================================================================
# Training-corpus tests  (generate_training_corpus)
# =============================================================================
# Coverage checklist
# ------------------
# C01  Exact total count: 300 scenarios.
# C02  Label counts: 150 positive, 150 negative.
# C03  Exact split counts: train=180, validation=60, test=60.
# C04  Per-split class balance: each split has equal positive and negative counts.
# C05  Unique scenario IDs across all 300 scenarios.
# C06  Zero scenario-ID overlap between any two splits.
# C07  Same-seed determinism: two calls with seed=42 return identical JSON.
# C08  Different-seed variation: seed=42 and seed=99 produce different corpora.
# C09  Unique telemetry histories: all 300 history windows are distinct.
# C10  Sample counts: every scenario has 72 history and 288 future samples.
# C11  Sample schema: each sample has exactly the eight required keys.
# C12  Cadence and continuity: 5-minute steps throughout; join gap = 300 s.
# C13  History eligibility: no history sample has SOC < 25 % or V < 26 V.
# C14  Future-label correctness:
#        - positive label only if a future breach actually occurs
#        - negative label only if no future breach occurs
# C15  Positive type/timing coverage: all nine (type × band) cells represented.
# C16  All three difficult-negative families present with expected counts.
# C17  Physical bounds: SOC in [0, 100]; voltage > 0.
# C18  Safety metadata: four mandatory fields with correct values.
# C19  Ignored output-file behavior: data/scenarios/ is gitignored.
# =============================================================================

import json as _json
import os as _os
import subprocess as _subprocess

from backend.scripts.generate_scenarios import (
    generate_training_corpus,
    _NEG_FAMILIES,
    _CORPUS_TOTAL,
    _CORPUS_POSITIVE,
    _CORPUS_NEGATIVE,
    _SPLIT_TRAIN_POS,
    _SPLIT_VAL_POS,
    _SPLIT_TEST_POS,
    _SPLIT_TRAIN_NEG,
    _SPLIT_VAL_NEG,
    _SPLIT_TEST_NEG,
)

# ---------------------------------------------------------------------------
# Module-level fixture — generate corpus once for the parametrised tests
# ---------------------------------------------------------------------------

_CORPUS: list[dict[str, Any]] = generate_training_corpus(seed=42)

_CORPUS_POS = [s for s in _CORPUS if s["power_constraint_breach_within_24h"] == 1]
_CORPUS_NEG = [s for s in _CORPUS if s["power_constraint_breach_within_24h"] == 0]


def _corpus_split(corpus: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    return [s for s in corpus if s.get("split") == split_name]


# ---------------------------------------------------------------------------
# C01 — Exact total count
# ---------------------------------------------------------------------------

def test_corpus_total_count():
    """C01: generate_training_corpus() returns exactly 300 scenarios."""
    assert len(_CORPUS) == _CORPUS_TOTAL, (
        f"Expected {_CORPUS_TOTAL} scenarios, got {len(_CORPUS)}"
    )


# ---------------------------------------------------------------------------
# C02 — Label counts
# ---------------------------------------------------------------------------

def test_corpus_label_counts():
    """C02: 150 positive, 150 negative."""
    n_pos = sum(1 for s in _CORPUS if s["power_constraint_breach_within_24h"] == 1)
    n_neg = sum(1 for s in _CORPUS if s["power_constraint_breach_within_24h"] == 0)
    assert n_pos == _CORPUS_POSITIVE, f"Expected {_CORPUS_POSITIVE} positives, got {n_pos}"
    assert n_neg == _CORPUS_NEGATIVE, f"Expected {_CORPUS_NEGATIVE} negatives, got {n_neg}"


# ---------------------------------------------------------------------------
# C03 — Exact split counts
# ---------------------------------------------------------------------------

def test_corpus_split_counts():
    """C03: train=180, validation=60, test=60."""
    train_scen = _corpus_split(_CORPUS, "train")
    val_scen   = _corpus_split(_CORPUS, "validation")
    test_scen  = _corpus_split(_CORPUS, "test")
    assert len(train_scen) == 180, f"train={len(train_scen)}, expected 180"
    assert len(val_scen)   == 60,  f"validation={len(val_scen)}, expected 60"
    assert len(test_scen)  == 60,  f"test={len(test_scen)}, expected 60"


# ---------------------------------------------------------------------------
# C04 — Per-split class balance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split_name,expected_pos,expected_neg", [
    ("train",      90, 90),
    ("validation", 30, 30),
    ("test",       30, 30),
])
def test_corpus_split_balance(split_name: str, expected_pos: int, expected_neg: int):
    """C04: each split has equal positive and negative counts."""
    scen    = _corpus_split(_CORPUS, split_name)
    n_pos   = sum(1 for s in scen if s["power_constraint_breach_within_24h"] == 1)
    n_neg   = len(scen) - n_pos
    assert n_pos == expected_pos, f"{split_name}: pos={n_pos}, expected {expected_pos}"
    assert n_neg == expected_neg, f"{split_name}: neg={n_neg}, expected {expected_neg}"


# ---------------------------------------------------------------------------
# C05 — Unique scenario IDs
# ---------------------------------------------------------------------------

def test_corpus_unique_ids():
    """C05: all 300 scenario IDs are unique."""
    ids = [s["scenario_id"] for s in _CORPUS]
    assert len(ids) == len(set(ids)), (
        f"Duplicate IDs found: {[x for x in ids if ids.count(x) > 1]}"
    )


# ---------------------------------------------------------------------------
# C06 — Zero split leakage
# ---------------------------------------------------------------------------

def test_corpus_zero_split_leakage():
    """C06: no scenario_id appears in more than one split."""
    split_id_sets: dict[str, set[str]] = {}
    for sp in ("train", "validation", "test"):
        split_id_sets[sp] = {s["scenario_id"] for s in _corpus_split(_CORPUS, sp)}

    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]
    for a, b in pairs:
        overlap = split_id_sets[a] & split_id_sets[b]
        assert not overlap, (
            f"Scenario IDs appear in both '{a}' and '{b}': {overlap}"
        )


# ---------------------------------------------------------------------------
# C07 — Same-seed determinism
# ---------------------------------------------------------------------------

def test_corpus_same_seed_determinism():
    """C07: two calls with seed=42 return byte-identical JSON."""
    run_a = generate_training_corpus(seed=42)
    run_b = generate_training_corpus(seed=42)
    assert _json.dumps(run_a, sort_keys=True) == _json.dumps(run_b, sort_keys=True), (
        "generate_training_corpus(seed=42) returned different results on two calls"
    )


# ---------------------------------------------------------------------------
# C08 — Different-seed variation
# ---------------------------------------------------------------------------

def test_corpus_different_seed_variation():
    """C08: seed=42 and seed=99 produce different corpora."""
    run_a = generate_training_corpus(seed=42)
    run_b = generate_training_corpus(seed=99)
    assert _json.dumps(run_a, sort_keys=True) != _json.dumps(run_b, sort_keys=True), (
        "generate_training_corpus() returned identical results for seed=42 and seed=99"
    )


# ---------------------------------------------------------------------------
# C09 — Unique telemetry histories
# ---------------------------------------------------------------------------

def test_corpus_unique_histories():
    """C09: all 300 history windows are distinct (not duplicates with different IDs)."""
    # Use first 5 history SOC values as a fingerprint.  If any two histories
    # share all five values, they are considered duplicates.
    fingerprints: list[tuple[float, ...]] = []
    for s in _CORPUS:
        fp = tuple(round(samp["battery_soc_percent"], 6) for samp in s["history"][:5])
        fingerprints.append(fp)
    assert len(fingerprints) == len(set(fingerprints)), (
        f"Duplicate history fingerprints found ({len(fingerprints) - len(set(fingerprints))} collisions)"
    )


# ---------------------------------------------------------------------------
# C10 — Sample counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _CORPUS, ids=[s["scenario_id"] for s in _CORPUS])
def test_corpus_history_count(scenario: dict[str, Any]):
    """C10a: each corpus scenario has exactly 72 history samples."""
    assert len(scenario["history"]) == 72, (
        f"{scenario['scenario_id']}: history len={len(scenario['history'])}"
    )


@pytest.mark.parametrize("scenario", _CORPUS, ids=[s["scenario_id"] for s in _CORPUS])
def test_corpus_future_count(scenario: dict[str, Any]):
    """C10b: each corpus scenario has exactly 288 future samples."""
    assert len(scenario["future"]) == 288, (
        f"{scenario['scenario_id']}: future len={len(scenario['future'])}"
    )


# ---------------------------------------------------------------------------
# C11 — Sample schema
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _CORPUS, ids=[s["scenario_id"] for s in _CORPUS])
def test_corpus_sample_schema(scenario: dict[str, Any]):
    """C11: each sample carries exactly the eight required telemetry keys."""
    for idx, sample in enumerate(_all_samples(scenario)):
        assert set(sample.keys()) == REQUIRED_SAMPLE_KEYS, (
            f"{scenario['scenario_id']} sample {idx}: "
            f"got keys {set(sample.keys())}"
        )


# ---------------------------------------------------------------------------
# C12 — Cadence and continuity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _CORPUS, ids=[s["scenario_id"] for s in _CORPUS])
def test_corpus_cadence(scenario: dict[str, Any]):
    """C12a: 5-minute (300 s) cadence throughout all samples."""
    samples = _all_samples(scenario)
    for i in range(len(samples) - 1):
        t0 = _parse_utc(samples[i]["timestamp"])
        t1 = _parse_utc(samples[i + 1]["timestamp"])
        gap = (t1 - t0).total_seconds()
        assert gap == 300, (
            f"{scenario['scenario_id']} step {i}→{i+1}: gap={gap}s (expected 300)"
        )


@pytest.mark.parametrize("scenario", _CORPUS, ids=[s["scenario_id"] for s in _CORPUS])
def test_corpus_history_future_join(scenario: dict[str, Any]):
    """C12b: first future sample is exactly 5 minutes after last history sample."""
    t_hist = _parse_utc(scenario["history"][-1]["timestamp"])
    t_fut  = _parse_utc(scenario["future"][0]["timestamp"])
    gap = (t_fut - t_hist).total_seconds()
    assert gap == 300, (
        f"{scenario['scenario_id']}: history→future gap={gap}s (expected 300)"
    )


# ---------------------------------------------------------------------------
# C13 — History eligibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _CORPUS, ids=[s["scenario_id"] for s in _CORPUS])
def test_corpus_no_history_soc_breach(scenario: dict[str, Any]):
    """C13a: no history sample has SOC < 25 %."""
    for i, sample in enumerate(scenario["history"]):
        soc = sample["battery_soc_percent"]
        assert soc >= 25.0, (
            f"{scenario['scenario_id']} history[{i}]: SOC={soc:.4f} < 25 %"
        )


@pytest.mark.parametrize("scenario", _CORPUS, ids=[s["scenario_id"] for s in _CORPUS])
def test_corpus_no_history_voltage_breach(scenario: dict[str, Any]):
    """C13b: no history sample has bus voltage < 26 V."""
    for i, sample in enumerate(scenario["history"]):
        v = sample["bus_voltage_v"]
        assert v >= 26.0, (
            f"{scenario['scenario_id']} history[{i}]: V={v:.4f} V < 26 V"
        )


# ---------------------------------------------------------------------------
# C14 — Future-label correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _CORPUS_POS,
                          ids=[s["scenario_id"] for s in _CORPUS_POS])
def test_corpus_positive_label_has_future_breach(scenario: dict[str, Any]):
    """C14a: positive-labelled scenario actually has a breach in its future window."""
    label = scenario["power_constraint_breach_within_24h"]
    assert label == 1
    bd = scenario["breach_detail"]
    assert bd["occurs"] is True, (
        f"{scenario['scenario_id']}: label=1 but breach_detail.occurs is False"
    )
    # Verify the future window contains the breach
    found_breach = False
    for samp in scenario["future"]:
        if samp["battery_soc_percent"] < 25.0 or samp["bus_voltage_v"] < 26.0:
            found_breach = True
            break
    assert found_breach, (
        f"{scenario['scenario_id']}: label=1 but no actual breach found in future samples"
    )


@pytest.mark.parametrize("scenario", _CORPUS_NEG,
                          ids=[s["scenario_id"] for s in _CORPUS_NEG])
def test_corpus_negative_label_has_no_future_breach(scenario: dict[str, Any]):
    """C14b: negative-labelled scenario has no breach in its future window."""
    label = scenario["power_constraint_breach_within_24h"]
    assert label == 0
    bd = scenario["breach_detail"]
    assert bd["occurs"] is False, (
        f"{scenario['scenario_id']}: label=0 but breach_detail.occurs is True"
    )
    for i, samp in enumerate(scenario["future"]):
        soc = samp["battery_soc_percent"]
        v   = samp["bus_voltage_v"]
        assert soc >= 25.0, (
            f"{scenario['scenario_id']} future[{i}]: SOC={soc:.4f} < 25 % (breach in negative scenario)"
        )
        assert v >= 26.0, (
            f"{scenario['scenario_id']} future[{i}]: V={v:.4f} < 26 V (breach in negative scenario)"
        )


# ---------------------------------------------------------------------------
# C15 — Positive type/timing coverage
# ---------------------------------------------------------------------------

def test_corpus_positive_type_timing_coverage():
    """C15: all nine (breach_type × timing_band) cells are represented."""
    expected_types = {"SOC_ONLY", "VOLTAGE_ONLY", "BOTH"}
    expected_bands = {"early", "middle", "late"}
    coverage: set[tuple[str, str]] = set()
    for s in _CORPUS_POS:
        bd = s["breach_detail"]
        coverage.add((bd["breach_type"], bd["timing_band"]))
    for t in expected_types:
        for b in expected_bands:
            assert (t, b) in coverage, (
                f"Positive coverage missing ({t}, {b})"
            )


# ---------------------------------------------------------------------------
# C16 — All three difficult-negative families present
# ---------------------------------------------------------------------------

def test_corpus_negative_family_counts():
    """C16: each of the three difficult-negative families has 50 scenarios."""
    for family in _NEG_FAMILIES:
        count = sum(1 for s in _CORPUS_NEG if family in s["scenario_id"])
        assert count == 50, (
            f"Family {family}: expected 50 scenarios, got {count}"
        )


# ---------------------------------------------------------------------------
# C17 — Physical bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _CORPUS, ids=[s["scenario_id"] for s in _CORPUS])
def test_corpus_soc_physical_bounds(scenario: dict[str, Any]):
    """C17a: SOC is in [0, 100] % throughout all samples."""
    for i, samp in enumerate(_all_samples(scenario)):
        soc = samp["battery_soc_percent"]
        assert 0.0 <= soc <= 100.0, (
            f"{scenario['scenario_id']} sample {i}: SOC={soc:.4f} outside [0, 100]"
        )


@pytest.mark.parametrize("scenario", _CORPUS, ids=[s["scenario_id"] for s in _CORPUS])
def test_corpus_voltage_positive(scenario: dict[str, Any]):
    """C17b: bus voltage is positive throughout all samples."""
    for i, samp in enumerate(_all_samples(scenario)):
        v = samp["bus_voltage_v"]
        assert v > 0.0, (
            f"{scenario['scenario_id']} sample {i}: V={v:.4f} ≤ 0"
        )


# ---------------------------------------------------------------------------
# C18 — Safety metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", _CORPUS, ids=[s["scenario_id"] for s in _CORPUS])
def test_corpus_safety_metadata(scenario: dict[str, Any]):
    """C18: all four safety metadata fields are present with correct values."""
    meta = scenario.get("metadata", {})
    for key, expected_value in EXPECTED_METADATA.items():
        assert key in meta, (
            f"{scenario['scenario_id']}: metadata missing key {key!r}"
        )
        assert meta[key] == expected_value, (
            f"{scenario['scenario_id']}: metadata[{key!r}]={meta[key]!r}, "
            f"expected {expected_value!r}"
        )


# ---------------------------------------------------------------------------
# C19 — Ignored output-file behavior
# ---------------------------------------------------------------------------

def test_gitignore_covers_scenarios_directory():
    """C19: data/scenarios/ is listed in .gitignore so the corpus is never staged."""
    repo_root = _os.path.join(_os.path.dirname(__file__), "..", "..")
    gitignore_path = _os.path.join(repo_root, ".gitignore")
    with open(gitignore_path, encoding="utf-8") as fh:
        content = fh.read()
    # Accept either the bare directory pattern or a wildcard variant
    assert "data/scenarios/" in content or "data/scenarios" in content, (
        ".gitignore does not contain a pattern for data/scenarios/"
    )
    assert "data/models/" in content or "data/models" in content, (
        ".gitignore does not contain a pattern for data/models/"
    )
