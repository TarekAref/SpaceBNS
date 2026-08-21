"""Contract tests for the SpaceBNS power-risk prediction endpoints.

Coverage checklist
------------------
P01  Normal POST prediction returns 200 with ok status.
P02  Demonstration GET prediction returns 200 with ok status.
P03  Exact four safety fields present in normal POST response.
P04  Exact four safety fields present in GET response.
P05  breach_probability is finite and in [0, 1].
P06  Exact 12-feature order in all_contributions matches FEATURE_ORDER.
P07  Top-three contributions ranked by descending absolute magnitude.
P08  Model output is not hardcoded — two different windows produce different probabilities.
P09  Feature extraction uses only history samples, not future/scenario/label fields.
P10  Shared L3 policy: POST and GET endpoint share apply_power_thresholds output for same sample.
P11  Degraded mode: fewer than 72 samples returns 200 with status=degraded, ai_prediction=null.
P12  Missing model returns 503 with error=MODEL_NOT_LOADED and safety envelope.
P13  Missing history returns 503 with error=HISTORY_UNAVAILABLE and safety envelope.
P14  Empty samples array returns 422 with safety envelope.
P15  Invalid/non-numeric sample field returns 422 with safety envelope.
P16  Generic exception handler returns 500 with safety envelope.
P17  POST CORS: POST method is in allow_methods.
P18  No command authority: command_authority=NONE on all response paths.
P19  Input samples are not mutated by the prediction service.
P20  Normal response contains predicted_class.
P21  Normal response contains probability_note (synthetic-distribution warning).
P22  advisory is present and advisory.authority_note mentions no automated action.
P23  L4 advisory has no command creation or execution — recommendation is advisory string only.
P24  audit.action_mode is simulation-only.
P25  Safety envelope on validation error (RequestValidationError path).
"""

from __future__ import annotations

import copy
import json
import math
import unittest.mock as mock
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_ORDER = [
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
]

SAFETY_FIELDS = {
    "data_source": "SYNTHETIC",
    "prototype_status": "NOT_FLIGHT_QUALIFIED",
    "command_authority": "NONE",
    "policy_decision": "PERMITTED_FOR_SIMULATION_ONLY",
}


def _make_samples(
    n: int = 72,
    *,
    soc_start: float = 55.0,
    soc_slope_per_sample: float = -0.05,
    voltage: float = 28.0,
    solar: float = 7.5,
    payload: float = 35.0,
    start: datetime | None = None,
) -> list[dict[str, Any]]:
    """Generate *n* valid telemetry samples at 5-minute cadence."""
    if start is None:
        start = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    samples = []
    for i in range(n):
        ts = start + timedelta(minutes=5 * i)
        soc = max(30.0, soc_start + soc_slope_per_sample * i)
        samples.append(
            {
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "battery_soc_percent": round(soc, 3),
                "bus_voltage_v": voltage,
                "solar_array_current_a": solar,
                "payload_power_draw_w": payload,
                "command_activity": "NOMINAL_ATTITUDE_HOLD",
                "communications_status": "NO_CONTACT_WINDOW",
                "image_utility_score": 0.75,
            }
        )
    return samples


def _make_samples_high_risk(n: int = 72) -> list[dict[str, Any]]:
    """Generate samples with high breach risk (low SOC, declining fast)."""
    return _make_samples(
        n,
        soc_start=38.0,
        soc_slope_per_sample=-0.08,
        voltage=27.1,
        solar=3.0,
        payload=90.0,
    )


# ---------------------------------------------------------------------------
# Client fixture — loads from the real app instance with model loaded
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client() -> TestClient:
    """TestClient using the real app (model must be present on disk)."""
    from backend.app.main import app
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# P01 — Normal POST prediction returns 200 with ok status
# ---------------------------------------------------------------------------

def test_p01_normal_post_returns_200(client: TestClient) -> None:
    samples = _make_samples(72)
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": samples}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# P02 — Demonstration GET prediction returns 200 with ok status
# ---------------------------------------------------------------------------

def test_p02_get_demo_returns_200(client: TestClient) -> None:
    resp = client.get("/api/v1/mock/power-risk-prediction")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# P03 — Exact four safety fields present in normal POST response
# ---------------------------------------------------------------------------

def test_p03_safety_fields_in_post_response(client: TestClient) -> None:
    samples = _make_samples(72)
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": samples}
    )
    body = resp.json()
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value, f"Safety field {field!r} mismatch"


# ---------------------------------------------------------------------------
# P04 — Exact four safety fields present in GET response
# ---------------------------------------------------------------------------

def test_p04_safety_fields_in_get_response(client: TestClient) -> None:
    resp = client.get("/api/v1/mock/power-risk-prediction")
    body = resp.json()
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value, f"Safety field {field!r} mismatch"


# ---------------------------------------------------------------------------
# P05 — breach_probability is finite and in [0, 1]
# ---------------------------------------------------------------------------

def test_p05_finite_probability(client: TestClient) -> None:
    samples = _make_samples(72)
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": samples}
    )
    body = resp.json()
    prob = body["ai_prediction"]["breach_probability"]
    assert isinstance(prob, float)
    assert math.isfinite(prob)
    assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# P06 — Exact 12-feature order in all_contributions matches FEATURE_ORDER
# ---------------------------------------------------------------------------

def test_p06_exact_12_feature_order(client: TestClient) -> None:
    samples = _make_samples(72)
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": samples}
    )
    body = resp.json()
    contributions = body["ai_prediction"]["all_contributions"]
    assert len(contributions) == 12
    actual_order = [c["feature"] for c in contributions]
    assert actual_order == FEATURE_ORDER, (
        f"Feature order mismatch.\nExpected: {FEATURE_ORDER}\nGot: {actual_order}"
    )


# ---------------------------------------------------------------------------
# P07 — Top-three contributions ranked by descending absolute magnitude
# ---------------------------------------------------------------------------

def test_p07_top_three_contribution_ordering(client: TestClient) -> None:
    samples = _make_samples(72)
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": samples}
    )
    body = resp.json()
    top3 = body["ai_prediction"]["top_contributions"]
    assert len(top3) == 3
    mags = [abs(c["contribution"]) for c in top3]
    # Verify descending order
    assert mags == sorted(mags, reverse=True), (
        f"Top contributions not sorted by magnitude: {mags}"
    )
    # Verify they are actually the top 3 from all_contributions
    all_mags = sorted(
        [abs(c["contribution"]) for c in body["ai_prediction"]["all_contributions"]],
        reverse=True,
    )
    assert mags[0] == all_mags[0]
    assert mags[1] == all_mags[1]
    assert mags[2] == all_mags[2]


# ---------------------------------------------------------------------------
# P08 — Model output is not hardcoded: two different windows → different probabilities
# ---------------------------------------------------------------------------

def test_p08_model_output_not_hardcoded(client: TestClient) -> None:
    low_risk = _make_samples(72, soc_start=80.0, solar=10.0, payload=20.0)
    high_risk = _make_samples_high_risk(72)

    resp_low = client.post(
        "/api/v1/power-risk/predict", json={"samples": low_risk}
    )
    resp_high = client.post(
        "/api/v1/power-risk/predict", json={"samples": high_risk}
    )
    assert resp_low.status_code == 200
    assert resp_high.status_code == 200

    p_low = resp_low.json()["ai_prediction"]["breach_probability"]
    p_high = resp_high.json()["ai_prediction"]["breach_probability"]

    # A different input must produce a different probability
    assert p_low != p_high, (
        f"Model returned identical probability {p_low} for two distinct windows"
    )
    # High-risk scenario should yield higher probability
    assert p_high > p_low, (
        f"Expected high-risk probability ({p_high}) > low-risk ({p_low})"
    )


# ---------------------------------------------------------------------------
# P09 — History-only extraction: forbidden fields are never used as features
# ---------------------------------------------------------------------------

def test_p09_history_only_extraction(client: TestClient) -> None:
    """Confirm that adding future/scenario/label fields to samples does not
    change the prediction (they must be ignored)."""
    samples_clean = _make_samples(72)
    samples_with_extras = copy.deepcopy(samples_clean)
    for s in samples_with_extras:
        s["scenario_id"] = "SHOULD-NOT-AFFECT"
        s["split"] = "test"
        s["power_constraint_breach_within_24h"] = 0
        s["breach_detail"] = {"occurs": False}

    resp_clean = client.post(
        "/api/v1/power-risk/predict", json={"samples": samples_clean}
    )
    resp_extras = client.post(
        "/api/v1/power-risk/predict", json={"samples": samples_with_extras}
    )
    assert resp_clean.status_code == 200
    assert resp_extras.status_code == 200

    p_clean = resp_clean.json()["ai_prediction"]["breach_probability"]
    p_extras = resp_extras.json()["ai_prediction"]["breach_probability"]
    assert p_clean == p_extras, (
        "Forbidden extra fields changed the prediction; they must not be ML inputs"
    )


# ---------------------------------------------------------------------------
# P10 — Shared L3 policy: POST and GET share apply_power_thresholds for same sample
# ---------------------------------------------------------------------------

def test_p10_shared_l3_policy(client: TestClient) -> None:
    """POST with mock-history-equivalent samples must produce the same L3 findings
    as the GET endpoint (both call apply_power_thresholds on the latest sample)."""
    from backend.app.policy import apply_power_thresholds

    # Use a sample with no L3 breach to compare
    samples = _make_samples(72, soc_start=55.0, voltage=28.0, payload=35.0)
    expected_findings = apply_power_thresholds(samples[-1])

    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": samples}
    )
    assert resp.status_code == 200
    actual_findings = resp.json()["safety_threshold_findings"]
    assert actual_findings == expected_findings


# ---------------------------------------------------------------------------
# P11 — Degraded mode: fewer than 72 samples
# ---------------------------------------------------------------------------

def test_p11_degraded_short_window(client: TestClient) -> None:
    samples = _make_samples(5)
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": samples}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["ai_prediction"] is None
    assert body["breach_probability"] is None
    # Safety envelope still present
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value
    # L3 and L4 still computed
    assert "safety_threshold_findings" in body
    assert "advisory" in body
    assert body["advisory"]["risk_summary"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# P12 — Missing model returns 503 MODEL_NOT_LOADED + safety envelope
# ---------------------------------------------------------------------------

def test_p12_missing_model_503(tmp_path: Path) -> None:
    import backend.app.prediction_service as svc
    import backend.app.main as main_module
    from backend.app.main import app

    original_pipeline = svc._pipeline
    original_error = svc._pipeline_load_error
    try:
        svc._pipeline = None
        svc._pipeline_load_error = "MODEL_NOT_LOADED"

        c = TestClient(app, raise_server_exceptions=False)
        samples = _make_samples(72)
        resp = c.post(
            "/api/v1/power-risk/predict", json={"samples": samples}
        )
        assert resp.status_code == 503
        body = resp.json()
        # Safety envelope
        for field, value in SAFETY_FIELDS.items():
            assert body.get(field) == value, f"Missing safety field {field!r} on 503"
        assert "MODEL_NOT_LOADED" in json.dumps(body)
    finally:
        svc._pipeline = original_pipeline
        svc._pipeline_load_error = original_error


# ---------------------------------------------------------------------------
# P13 — Missing history returns 503 HISTORY_UNAVAILABLE + safety envelope
# ---------------------------------------------------------------------------

def test_p13_missing_history_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.app.main as main_module
    from backend.app.main import app

    # Point history path to a file that does not exist
    monkeypatch.setattr(main_module, "MOCK_HISTORY_PATH", tmp_path / "nonexistent.json")

    c = TestClient(app, raise_server_exceptions=False)
    resp = c.get("/api/v1/mock/power-risk-prediction")
    assert resp.status_code == 503
    body = resp.json()
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value, f"Missing safety field {field!r} on 503"
    assert "HISTORY_UNAVAILABLE" in json.dumps(body)


# ---------------------------------------------------------------------------
# P14 — Empty samples array returns 422 with safety envelope
# ---------------------------------------------------------------------------

def test_p14_empty_samples_422(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": []}
    )
    assert resp.status_code == 422
    body = resp.json()
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value, f"Missing safety field {field!r} on 422"


# ---------------------------------------------------------------------------
# P15 — Invalid/missing required field returns 422 with safety envelope
# ---------------------------------------------------------------------------

def test_p15_invalid_sample_schema_422(client: TestClient) -> None:
    # Missing required field battery_soc_percent
    bad_sample = {
        "timestamp": "2026-08-01T00:00:00Z",
        "bus_voltage_v": 28.0,
        "solar_array_current_a": 7.5,
        "payload_power_draw_w": 35.0,
        # battery_soc_percent deliberately omitted
    }
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": [bad_sample]}
    )
    assert resp.status_code == 422
    body = resp.json()
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value, f"Missing safety field {field!r} on 422"


# ---------------------------------------------------------------------------
# P16 — Generic exception handler returns 500 with safety envelope
# ---------------------------------------------------------------------------

def test_p16_generic_exception_safety_handler() -> None:
    """Force an unhandled exception through the global handler and verify
    the safety envelope is present in the 500 response."""
    from backend.app.main import app
    from backend.app.safety import SAFETY_ENVELOPE

    import backend.app.prediction_service as svc

    original_run = svc.run_prediction

    def _exploding_prediction(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Injected test failure — not a real error")

    svc.run_prediction = _exploding_prediction  # type: ignore[assignment]
    try:
        c = TestClient(app, raise_server_exceptions=False)
        samples = _make_samples(72)
        resp = c.post(
            "/api/v1/power-risk/predict", json={"samples": samples}
        )
        # The endpoint catches Exception and raises HTTPException(500)
        assert resp.status_code == 500
        body = resp.json()
        for field, value in SAFETY_FIELDS.items():
            assert body.get(field) == value, (
                f"Safety field {field!r} missing on 500 response: {body}"
            )
    finally:
        svc.run_prediction = original_run  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# P17 — POST CORS: POST method is in allow_methods
# ---------------------------------------------------------------------------

def test_p17_cors_post_allowed(client: TestClient) -> None:
    """Verify the CORS preflight response allows POST."""
    resp = client.options(
        "/api/v1/power-risk/predict",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    # 200 or 204 are both valid preflight responses
    assert resp.status_code in (200, 204)
    allow = resp.headers.get("access-control-allow-methods", "")
    assert "POST" in allow, f"POST not in Allow-Methods: {allow!r}"


# ---------------------------------------------------------------------------
# P18 — No command authority on all response paths
# ---------------------------------------------------------------------------

def test_p18_no_command_authority_normal(client: TestClient) -> None:
    samples = _make_samples(72)
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": samples}
    )
    assert resp.json()["command_authority"] == "NONE"


def test_p18_no_command_authority_degraded(client: TestClient) -> None:
    samples = _make_samples(5)
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": samples}
    )
    assert resp.json()["command_authority"] == "NONE"


def test_p18_no_command_authority_get(client: TestClient) -> None:
    resp = client.get("/api/v1/mock/power-risk-prediction")
    assert resp.json()["command_authority"] == "NONE"


# ---------------------------------------------------------------------------
# P19 — Input samples are not mutated by the prediction service
# ---------------------------------------------------------------------------

def test_p19_input_samples_not_mutated(client: TestClient) -> None:
    samples = _make_samples(72)
    original_copy = copy.deepcopy(samples)

    client.post("/api/v1/power-risk/predict", json={"samples": samples})

    # The Python dicts were not mutated (Pydantic copies them)
    assert samples == original_copy


# ---------------------------------------------------------------------------
# P20 — Normal response contains predicted_class
# ---------------------------------------------------------------------------

def test_p20_predicted_class_present(client: TestClient) -> None:
    samples = _make_samples(72)
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": samples}
    )
    body = resp.json()
    assert "predicted_class" in body["ai_prediction"]
    assert body["ai_prediction"]["predicted_class"] in (0, 1)


# ---------------------------------------------------------------------------
# P21 — Normal response contains probability_note (synthetic-distribution warning)
# ---------------------------------------------------------------------------

def test_p21_probability_note_present(client: TestClient) -> None:
    samples = _make_samples(72)
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": samples}
    )
    note = resp.json()["ai_prediction"]["probability_note"]
    assert "synthetic" in note.lower(), (
        f"probability_note does not mention synthetic distribution: {note!r}"
    )


# ---------------------------------------------------------------------------
# P22 — Advisory present with authority_note about no automated action
# ---------------------------------------------------------------------------

def test_p22_advisory_authority_note(client: TestClient) -> None:
    samples = _make_samples(72)
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": samples}
    )
    advisory = resp.json()["advisory"]
    assert "authority_note" in advisory
    note = advisory["authority_note"].lower()
    assert "automated" in note or "no automated" in note or "advisory" in note


# ---------------------------------------------------------------------------
# P23 — L4 advisory is a text recommendation only (no command creation/execution)
# ---------------------------------------------------------------------------

def test_p23_advisory_is_text_only_no_command(client: TestClient) -> None:
    samples = _make_samples(72)
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": samples}
    )
    advisory = resp.json()["advisory"]
    recommendation = advisory["recommendation"]
    # Must be a plain string — no callable, no side-effect possible
    assert isinstance(recommendation, str)
    # Must not contain command-execution language
    forbidden_terms = ["execute", "send_command", "uplink", "arm", "fire", "actuate"]
    for term in forbidden_terms:
        assert term not in recommendation.lower(), (
            f"Advisory recommendation contains forbidden term {term!r}"
        )


# ---------------------------------------------------------------------------
# P24 — audit.action_mode is simulation-only
# ---------------------------------------------------------------------------

def test_p24_audit_action_mode(client: TestClient) -> None:
    samples = _make_samples(72)
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": samples}
    )
    audit = resp.json()["audit"]
    assert audit["action_mode"] == "simulation-only"


# ---------------------------------------------------------------------------
# P25 — Safety envelope on RequestValidationError (missing request body)
# ---------------------------------------------------------------------------

def test_p25_validation_error_safety_envelope(client: TestClient) -> None:
    # Send completely invalid JSON body (missing samples key entirely)
    resp = client.post(
        "/api/v1/power-risk/predict",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    body = resp.json()
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value, (
            f"Safety field {field!r} missing on validation error: {body}"
        )


# ===========================================================================
# NEW CORRECTION TESTS
# ===========================================================================

# ---------------------------------------------------------------------------
# Helper: minimal valid L2 assumptions
# ---------------------------------------------------------------------------

def _make_l2_assumptions(
    start: datetime | None = None,
    capacity_wh: float = 500.0,
    base_load_w: float = 50.0,
    efficiency: float = 0.9,
    sunlight_windows: int = 1,
    payload_draw_w: float = 30.0,
) -> dict[str, Any]:
    """Build a complete, valid set of L2 projection assumptions."""
    if start is None:
        start = datetime(2026, 8, 1, 6, 0, 0, tzinfo=timezone.utc)

    sun_start = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    sun_end = (start + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")

    pay_start = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    pay_end = (start + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "battery_capacity_wh": capacity_wh,
        "base_spacecraft_load_w": base_load_w,
        "power_conversion_efficiency": efficiency,
        "sunlight_schedule": [{"start": sun_start, "end": sun_end}],
        "payload_schedule": [{"start": pay_start, "end": pay_end, "draw_w": payload_draw_w}],
    }


# ---------------------------------------------------------------------------
# P26 — Exception containing fake secret/path is fully redacted
# ---------------------------------------------------------------------------

def test_p26_exception_is_fully_redacted() -> None:
    """str(exc) with a fake secret and filesystem path must NOT appear in response."""
    from backend.app.main import app
    import backend.app.prediction_service as svc

    original_run = svc.run_prediction

    def _leaky_prediction(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "Error at /home/user/secrets/api_key=SUPERSECRETKEY123"
        )

    svc.run_prediction = _leaky_prediction  # type: ignore[assignment]
    try:
        c = TestClient(app, raise_server_exceptions=False)
        samples = _make_samples(72)
        resp = c.post("/api/v1/power-risk/predict", json={"samples": samples})
        assert resp.status_code == 500
        body_text = resp.text
        assert "SUPERSECRETKEY123" not in body_text
        assert "/home/user/secrets" not in body_text
        assert "leaky" not in body_text.lower()
    finally:
        svc.run_prediction = original_run  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# P27 — Invalid short-window timestamp returns 422, not degraded
# ---------------------------------------------------------------------------

def test_p27_invalid_short_window_timestamp_is_422(client: TestClient) -> None:
    """A short window with an invalid timestamp must return 422, not degraded."""
    bad_sample = {
        "timestamp": "NOT-A-TIMESTAMP",
        "battery_soc_percent": 55.0,
        "bus_voltage_v": 28.0,
        "solar_array_current_a": 7.5,
        "payload_power_draw_w": 35.0,
    }
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": [bad_sample]}
    )
    assert resp.status_code == 422
    body = resp.json()
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value, f"Safety field {field!r} missing on 422"


# ---------------------------------------------------------------------------
# P28 — Wrong cadence returns 422
# ---------------------------------------------------------------------------

def test_p28_wrong_cadence_returns_422(client: TestClient) -> None:
    """Samples at 10-minute cadence (not 5-minute) must return 422."""
    start = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    bad_samples = []
    for i in range(5):
        ts = start + timedelta(minutes=10 * i)  # 10-min cadence
        bad_samples.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "battery_soc_percent": 55.0,
            "bus_voltage_v": 28.0,
            "solar_array_current_a": 7.5,
            "payload_power_draw_w": 35.0,
        })
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": bad_samples}
    )
    assert resp.status_code == 422
    body = resp.json()
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value


def test_p28_duplicate_timestamps_returns_422(client: TestClient) -> None:
    """Duplicate timestamps must return 422."""
    start = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    bad_samples = [
        {
            "timestamp": ts_str,
            "battery_soc_percent": 55.0,
            "bus_voltage_v": 28.0,
            "solar_array_current_a": 7.5,
            "payload_power_draw_w": 35.0,
        },
        {
            "timestamp": ts_str,  # duplicate
            "battery_soc_percent": 54.0,
            "bus_voltage_v": 28.0,
            "solar_array_current_a": 7.5,
            "payload_power_draw_w": 35.0,
        },
    ]
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": bad_samples}
    )
    assert resp.status_code == 422
    body = resp.json()
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value


# ---------------------------------------------------------------------------
# P29 — Bool, numeric-string, NaN, infinity telemetry rejected
# ---------------------------------------------------------------------------

def test_p29_bool_value_rejected(client: TestClient) -> None:
    """Boolean value for a numeric field must be rejected with 422."""
    start = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    bad_sample = {
        "timestamp": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "battery_soc_percent": True,  # bool — must be rejected
        "bus_voltage_v": 28.0,
        "solar_array_current_a": 7.5,
        "payload_power_draw_w": 35.0,
    }
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": [bad_sample]}
    )
    assert resp.status_code == 422
    body = resp.json()
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value


def test_p29_nan_value_rejected(client: TestClient) -> None:
    """NaN is not valid JSON but if passed via Python dict it must be rejected."""
    import backend.app.prediction_service as svc
    from backend.app.safety import SAFETY_ENVELOPE

    start = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    bad_samples = [{
        "timestamp": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "battery_soc_percent": float("nan"),
        "bus_voltage_v": 28.0,
        "solar_array_current_a": 7.5,
        "payload_power_draw_w": 35.0,
    }]
    with pytest.raises(ValueError, match="NON_FINITE_VALUE|INVALID_TELEMETRY|EMPTY_WINDOW"):
        svc._validate_samples(bad_samples)


def test_p29_infinity_value_rejected(client: TestClient) -> None:
    """Infinity is not a valid telemetry value and must be rejected."""
    import backend.app.prediction_service as svc

    start = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    bad_samples = [{
        "timestamp": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "battery_soc_percent": float("inf"),
        "bus_voltage_v": 28.0,
        "solar_array_current_a": 7.5,
        "payload_power_draw_w": 35.0,
    }]
    with pytest.raises(ValueError, match="NON_FINITE_VALUE"):
        svc._validate_samples(bad_samples)


# ---------------------------------------------------------------------------
# P30 — Valid short window returns degraded without a loaded model
# ---------------------------------------------------------------------------

def test_p30_valid_short_window_degraded_no_model() -> None:
    """A short but valid window returns 200 degraded even if model is unloaded."""
    import backend.app.prediction_service as svc
    from backend.app.main import app

    original_pipeline = svc._pipeline
    original_error = svc._pipeline_load_error
    try:
        svc._pipeline = None
        svc._pipeline_load_error = "MODEL_NOT_LOADED"

        c = TestClient(app, raise_server_exceptions=False)
        samples = _make_samples(5)
        resp = c.post("/api/v1/power-risk/predict", json={"samples": samples})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["ai_prediction"] is None
        for field, value in SAFETY_FIELDS.items():
            assert body.get(field) == value
    finally:
        svc._pipeline = original_pipeline
        svc._pipeline_load_error = original_error


# ---------------------------------------------------------------------------
# P31 — Empty request returns 422 without a loaded model
# ---------------------------------------------------------------------------

def test_p31_empty_request_422_no_model() -> None:
    """Empty samples list returns 422 regardless of whether the model is loaded."""
    import backend.app.prediction_service as svc
    from backend.app.main import app

    original_pipeline = svc._pipeline
    original_error = svc._pipeline_load_error
    try:
        svc._pipeline = None
        svc._pipeline_load_error = "MODEL_NOT_LOADED"

        c = TestClient(app, raise_server_exceptions=False)
        resp = c.post("/api/v1/power-risk/predict", json={"samples": []})
        assert resp.status_code == 422
        body = resp.json()
        for field, value in SAFETY_FIELDS.items():
            assert body.get(field) == value
    finally:
        svc._pipeline = original_pipeline
        svc._pipeline_load_error = original_error


# ---------------------------------------------------------------------------
# P32 — Valid complete L2 assumptions produce 24 entries and not_ai_output true
# ---------------------------------------------------------------------------

def test_p32_l2_complete_assumptions_24_entries(client: TestClient) -> None:
    """A complete, valid L2 assumptions block must produce 24 hourly entries."""
    samples = _make_samples(72)
    assumptions = _make_l2_assumptions(
        start=datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    )
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": assumptions},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    projection = body["deterministic_projection"]
    assert projection is not None, "Expected deterministic_projection but got None"
    assert projection["not_ai_output"] is True
    entries = projection["hourly_projection"]
    assert len(entries) == 24, f"Expected 24 entries, got {len(entries)}"
    for i, entry in enumerate(entries):
        assert entry["hour_offset"] == i + 1
        assert "projected_soc_percent" in entry
        assert "projected_breach" in entry


# ---------------------------------------------------------------------------
# P33 — Partial L2 assumptions produce null projection
# ---------------------------------------------------------------------------

def test_p33_partial_l2_assumptions_null_projection(client: TestClient) -> None:
    """Missing one required L2 assumption must return null projection (not 422)."""
    samples = _make_samples(72)
    partial = {
        "battery_capacity_wh": 500.0,
        "base_spacecraft_load_w": 50.0,
        # power_conversion_efficiency, sunlight_schedule, payload_schedule absent
    }
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": partial},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["deterministic_projection"] is None
    assert body.get("projection_omitted_reason") is not None


# ---------------------------------------------------------------------------
# P34 — Invalid L2 assumptions return 422
# ---------------------------------------------------------------------------

def test_p34_invalid_capacity_returns_422(client: TestClient) -> None:
    """battery_capacity_wh <= 0 must return 422."""
    samples = _make_samples(72)
    assumptions = _make_l2_assumptions(capacity_wh=0.0)
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": assumptions},
    )
    assert resp.status_code == 422
    body = resp.json()
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value


def test_p34_invalid_efficiency_returns_422(client: TestClient) -> None:
    """power_conversion_efficiency > 1 must return 422."""
    samples = _make_samples(72)
    assumptions = _make_l2_assumptions(efficiency=1.5)
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": assumptions},
    )
    assert resp.status_code == 422
    body = resp.json()
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value


def test_p34_zero_efficiency_returns_422(client: TestClient) -> None:
    """power_conversion_efficiency == 0 must return 422."""
    samples = _make_samples(72)
    assumptions = _make_l2_assumptions(efficiency=0.0)
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": assumptions},
    )
    assert resp.status_code == 422


def test_p34_negative_base_load_returns_422(client: TestClient) -> None:
    """base_spacecraft_load_w < 0 must return 422."""
    samples = _make_samples(72)
    assumptions = _make_l2_assumptions(base_load_w=-10.0)
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": assumptions},
    )
    assert resp.status_code == 422


def test_p34_negative_payload_draw_returns_422(client: TestClient) -> None:
    """Negative payload draw_w must return 422."""
    samples = _make_samples(72)
    assumptions = _make_l2_assumptions(payload_draw_w=-5.0)
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": assumptions},
    )
    assert resp.status_code == 422


def test_p34_reversed_sunlight_interval_returns_422(client: TestClient) -> None:
    """sunlight_schedule with start >= end must return 422."""
    samples = _make_samples(72)
    assumptions = _make_l2_assumptions()
    assumptions["sunlight_schedule"] = [
        {"start": "2026-08-01T12:00:00Z", "end": "2026-08-01T06:00:00Z"}
    ]
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": assumptions},
    )
    assert resp.status_code == 422


def test_p34_overlapping_payload_intervals_returns_422(client: TestClient) -> None:
    """Overlapping payload_schedule intervals must return 422."""
    samples = _make_samples(72)
    assumptions = _make_l2_assumptions()
    assumptions["payload_schedule"] = [
        {"start": "2026-08-01T06:00:00Z", "end": "2026-08-01T10:00:00Z", "draw_w": 30.0},
        {"start": "2026-08-01T08:00:00Z", "end": "2026-08-01T12:00:00Z", "draw_w": 40.0},
    ]
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": assumptions},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# P35 — Contribution arithmetic: standardized_value × coefficient
# ---------------------------------------------------------------------------

def test_p35_contribution_arithmetic(client: TestClient) -> None:
    """Verify that contribution == standardized_value × coefficient for each feature.

    Tolerance is 1e-5 to account for floating-point rounding introduced by
    round(..., 6) serialization at the JSON boundary.
    """
    samples = _make_samples(72)
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": samples}
    )
    assert resp.status_code == 200
    contributions = resp.json()["ai_prediction"]["all_contributions"]
    for c in contributions:
        expected = round(c["standardized_value"] * c["coefficient"], 6)
        # Allow up to 2 ULP of round(..., 6) — 1e-5 is sufficient
        assert abs(c["contribution"] - expected) < 1e-5, (
            f"Arithmetic mismatch for {c['feature']!r}: "
            f"{c['standardized_value']} × {c['coefficient']} = {expected}, "
            f"got {c['contribution']}"
        )


# ---------------------------------------------------------------------------
# P36 — Directly test run_prediction input immutability with deepcopy
# ---------------------------------------------------------------------------

def test_p36_run_prediction_input_immutability() -> None:
    """Directly call run_prediction and confirm the original list is untouched."""
    import backend.app.prediction_service as svc

    original_samples = _make_samples(72)
    snapshot = copy.deepcopy(original_samples)

    # run_prediction should not raise (model must be loaded for this test)
    if svc.get_pipeline() is None:
        pytest.skip("Model not loaded — skipping run_prediction direct test")

    svc.run_prediction(original_samples, scenario_id="IMMUTABILITY-TEST")

    assert original_samples == snapshot, (
        "run_prediction mutated the caller's sample list"
    )


# ---------------------------------------------------------------------------
# P37 — Malformed/tampered pipeline rejection
# ---------------------------------------------------------------------------

def test_p37_malformed_pipeline_rejected(tmp_path: Path) -> None:
    """A pipeline missing the required named steps must not be loaded."""
    import joblib
    import backend.app.prediction_service as svc
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Build a pipeline with only a scaler, no clf step
    bad_pipeline = Pipeline([("scaler", StandardScaler())])
    # fit on dummy data so mean_ and scale_ exist (not strictly required for this test)
    import numpy as np
    bad_pipeline.fit(np.zeros((2, 12)), [0, 1])

    bad_path = str(tmp_path / "bad_pipeline.joblib")
    joblib.dump(bad_pipeline, bad_path)

    original_pipeline = svc._pipeline
    original_error = svc._pipeline_load_error
    try:
        svc.load_pipeline(bad_path)
        assert svc.get_pipeline() is None, "Malformed pipeline should not be accepted"
        assert svc.get_pipeline_load_error() == "MODEL_NOT_LOADED"
    finally:
        svc._pipeline = original_pipeline
        svc._pipeline_load_error = original_error


def test_p37_tampered_pipeline_wrong_classes(tmp_path: Path) -> None:
    """A pipeline with non-binary classifier classes must be rejected."""
    import joblib
    import numpy as np
    import backend.app.prediction_service as svc
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    # Train with 3 classes
    X = np.random.default_rng(0).random((30, 12))
    y = np.array([0, 1, 2] * 10)
    bad_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    bad_pipeline.fit(X, y)

    bad_path = str(tmp_path / "bad_classes.joblib")
    joblib.dump(bad_pipeline, bad_path)

    original_pipeline = svc._pipeline
    original_error = svc._pipeline_load_error
    try:
        svc.load_pipeline(bad_path)
        assert svc.get_pipeline() is None, "Pipeline with wrong classes should be rejected"
        assert svc.get_pipeline_load_error() == "MODEL_NOT_LOADED"
    finally:
        svc._pipeline = original_pipeline
        svc._pipeline_load_error = original_error


# ---------------------------------------------------------------------------
# P38 — Every error response retains command_authority: "NONE"
# ---------------------------------------------------------------------------

def test_p38_command_authority_on_422(client: TestClient) -> None:
    """422 response must carry command_authority: NONE."""
    resp = client.post(
        "/api/v1/power-risk/predict", json={"samples": []}
    )
    assert resp.status_code == 422
    assert resp.json().get("command_authority") == "NONE"


def test_p38_command_authority_on_503(tmp_path: Path) -> None:
    """503 response must carry command_authority: NONE."""
    import backend.app.prediction_service as svc
    from backend.app.main import app

    original_pipeline = svc._pipeline
    original_error = svc._pipeline_load_error
    try:
        svc._pipeline = None
        svc._pipeline_load_error = "MODEL_NOT_LOADED"

        c = TestClient(app, raise_server_exceptions=False)
        resp = c.post(
            "/api/v1/power-risk/predict", json={"samples": _make_samples(72)}
        )
        assert resp.status_code == 503
        assert resp.json().get("command_authority") == "NONE"
    finally:
        svc._pipeline = original_pipeline
        svc._pipeline_load_error = original_error


def test_p38_command_authority_on_500() -> None:
    """500 response must carry command_authority: NONE."""
    from backend.app.main import app
    import backend.app.prediction_service as svc

    original_run = svc.run_prediction

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("forced failure")

    svc.run_prediction = _boom  # type: ignore[assignment]
    try:
        c = TestClient(app, raise_server_exceptions=False)
        resp = c.post(
            "/api/v1/power-risk/predict", json={"samples": _make_samples(72)}
        )
        assert resp.status_code == 500
        assert resp.json().get("command_authority") == "NONE"
    finally:
        svc.run_prediction = original_run  # type: ignore[assignment]


# ===========================================================================
# CORRECTION REGRESSION TESTS  (Fix 1–5)
# ===========================================================================

# ---------------------------------------------------------------------------
# Fix 1 — Degraded-mode L4 follows contract priority rules
# ---------------------------------------------------------------------------

def test_fix1_degraded_with_no_l3_findings_is_unknown(client: TestClient) -> None:
    """Degraded mode with no L3 findings must produce UNKNOWN advisory."""
    # soc/voltage/payload all well within thresholds — no L3 firing
    samples = _make_samples(5, soc_start=60.0, voltage=28.5, payload=40.0)
    resp = client.post("/api/v1/power-risk/predict", json={"samples": samples})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["advisory"]["risk_summary"] == "UNKNOWN"
    assert body["advisory"]["recommendation"] == "SUPPLY_SUFFICIENT_HISTORY"


def test_fix1_degraded_with_1_l3_finding_is_elevated(client: TestClient) -> None:
    """Degraded mode with 1 L3 finding must produce ELEVATED, not UNKNOWN."""
    # payload_power_draw_w > 100 triggers PAYLOAD_LOAD_HIGH (1 finding)
    samples = _make_samples(5, soc_start=60.0, voltage=28.5, payload=120.0)
    resp = client.post("/api/v1/power-risk/predict", json={"samples": samples})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["ai_prediction"] is None
    # 1 finding must escalate to ELEVATED regardless of missing AI
    assert body["advisory"]["risk_summary"] == "ELEVATED", (
        f"Expected ELEVATED with 1 L3 finding in degraded mode, got "
        f"{body['advisory']['risk_summary']}"
    )
    assert body["advisory"]["recommendation"] == "INCREASE_MONITORING_FREQUENCY"


def _make_samples_direct(
    n: int,
    *,
    soc: float = 55.0,
    voltage: float = 28.0,
    solar: float = 7.5,
    payload: float = 35.0,
) -> list[dict[str, Any]]:
    """Build n valid 300s-cadence samples with all 8 required fields (exact unclamped values)."""
    start = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    result = []
    for i in range(n):
        ts = start + timedelta(seconds=300 * i)
        result.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "battery_soc_percent": soc,
            "bus_voltage_v": voltage,
            "solar_array_current_a": solar,
            "payload_power_draw_w": payload,
            "command_activity": "NOMINAL_ATTITUDE_HOLD",
            "communications_status": "NO_CONTACT_WINDOW",
            "image_utility_score": 0.75,
        })
    return result


def test_fix1_degraded_with_3_l3_findings_is_high(client: TestClient) -> None:
    """Degraded mode with >=3 L3 findings must produce HIGH, not UNKNOWN/ELEVATED."""
    # Trigger 3 findings: BUS_VOLTAGE_LOW (< 26), BATTERY_SOC_LOW (< 25), PAYLOAD_LOAD_HIGH (> 100)
    # Use direct builder to avoid the soc=max(30,...) clamp in _make_samples
    samples = _make_samples_direct(
        5,
        soc=22.0,      # < 25 → BATTERY_SOC_LOW
        voltage=25.5,  # < 26 → BUS_VOLTAGE_LOW
        payload=120.0, # > 100 → PAYLOAD_LOAD_HIGH
    )
    resp = client.post("/api/v1/power-risk/predict", json={"samples": samples})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["ai_prediction"] is None
    assert len(body["safety_threshold_findings"]) >= 3, (
        f"Expected >=3 L3 findings, got {body['safety_threshold_findings']}"
    )
    assert body["advisory"]["risk_summary"] == "HIGH", (
        f"Expected HIGH with >=3 L3 findings in degraded mode, got "
        f"{body['advisory']['risk_summary']}"
    )
    assert body["advisory"]["recommendation"] == "DEFER_LOW_PRIORITY_FUTURE_IMAGING"


def test_fix1_degraded_with_active_breach_is_elevated(client: TestClient) -> None:
    """Degraded mode with 1 active breach (BATTERY_SOC_LOW) produces ELEVATED."""
    # Only BATTERY_SOC_LOW fires (1 finding) — use direct builder to avoid SOC clamp
    samples = _make_samples_direct(5, soc=22.0, voltage=28.5, payload=40.0)
    resp = client.post("/api/v1/power-risk/predict", json={"samples": samples})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["ai_prediction"] is None
    # 1 L3 finding → ELEVATED
    assert body["advisory"]["risk_summary"] == "ELEVATED"


# ---------------------------------------------------------------------------
# Fix 2 — apply_power_thresholds() exceptions must not be suppressed
# ---------------------------------------------------------------------------

def test_fix2_policy_exception_not_suppressed_in_degraded_mode() -> None:
    """An exception from apply_power_thresholds() in degraded mode must propagate
    to the endpoint handler as a 500, not be swallowed into an empty findings list."""
    from backend.app.main import app
    import backend.app.prediction_service as svc

    # prediction_service imports apply_power_thresholds directly into its namespace;
    # we must patch it there, not in the policy module.
    original_fn = svc.apply_power_thresholds

    def _exploding_policy(sample: Any) -> list:
        raise RuntimeError("Injected policy failure")

    svc.apply_power_thresholds = _exploding_policy  # type: ignore[assignment]
    try:
        c = TestClient(app, raise_server_exceptions=False)
        # 5-sample window → degraded path
        samples = _make_samples(5)
        resp = c.post("/api/v1/power-risk/predict", json={"samples": samples})
        # Must be 500 with safety envelope — not 200 degraded with empty findings
        assert resp.status_code == 500, (
            f"Expected 500 when policy raises, got {resp.status_code}: {resp.json()}"
        )
        body = resp.json()
        for field, value in SAFETY_FIELDS.items():
            assert body.get(field) == value, (
                f"Safety field {field!r} missing on policy-failure 500"
            )
    finally:
        svc.apply_power_thresholds = original_fn  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fix 3 — Exactly 300 s between samples (no tolerance)
# ---------------------------------------------------------------------------

def _make_samples_with_cadence(
    n: int,
    cadence_seconds: int,
) -> list[dict[str, Any]]:
    """Build n samples at the given cadence in seconds (all 8 required fields)."""
    start = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    samples = []
    for i in range(n):
        ts = start + timedelta(seconds=cadence_seconds * i)
        samples.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "battery_soc_percent": 55.0,
            "bus_voltage_v": 28.0,
            "solar_array_current_a": 7.5,
            "payload_power_draw_w": 35.0,
            "command_activity": "NOMINAL_ATTITUDE_HOLD",
            "communications_status": "NO_CONTACT_WINDOW",
            "image_utility_score": 0.75,
        })
    return samples


def test_fix3_299_second_cadence_rejected(client: TestClient) -> None:
    """299-second cadence must return 422."""
    samples = _make_samples_with_cadence(5, cadence_seconds=299)
    resp = client.post("/api/v1/power-risk/predict", json={"samples": samples})
    assert resp.status_code == 422, (
        f"Expected 422 for 299s cadence, got {resp.status_code}"
    )
    body = resp.json()
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value


def test_fix3_301_second_cadence_rejected(client: TestClient) -> None:
    """301-second cadence must return 422."""
    samples = _make_samples_with_cadence(5, cadence_seconds=301)
    resp = client.post("/api/v1/power-risk/predict", json={"samples": samples})
    assert resp.status_code == 422, (
        f"Expected 422 for 301s cadence, got {resp.status_code}"
    )
    body = resp.json()
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value


def test_fix3_300_second_cadence_accepted(client: TestClient) -> None:
    """Exactly 300-second cadence must be accepted (not rejected)."""
    samples = _make_samples_with_cadence(5, cadence_seconds=300)
    resp = client.post("/api/v1/power-risk/predict", json={"samples": samples})
    # 5 samples → degraded; must be 200, not 422
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


def test_fix3_validate_samples_299s_raises() -> None:
    """Unit-level: _validate_samples raises for 299s cadence."""
    import backend.app.prediction_service as svc
    samples = _make_samples_with_cadence(3, cadence_seconds=299)
    with pytest.raises(ValueError, match="WRONG_CADENCE"):
        svc._validate_samples(samples)


def test_fix3_validate_samples_301s_raises() -> None:
    """Unit-level: _validate_samples raises for 301s cadence."""
    import backend.app.prediction_service as svc
    samples = _make_samples_with_cadence(3, cadence_seconds=301)
    with pytest.raises(ValueError, match="WRONG_CADENCE"):
        svc._validate_samples(samples)


# ---------------------------------------------------------------------------
# Fix 4 — Fully validated L2 schedules
# ---------------------------------------------------------------------------

def test_fix4_sunlight_non_dict_entry_returns_422(client: TestClient) -> None:
    """A non-dict sunlight_schedule entry must return 422, not 500."""
    samples = _make_samples(72)
    assumptions = _make_l2_assumptions()
    assumptions["sunlight_schedule"] = ["not-a-dict"]
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": assumptions},
    )
    assert resp.status_code == 422, (
        f"Expected 422 for non-dict sunlight entry, got {resp.status_code}"
    )
    body = resp.json()
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value


def test_fix4_sunlight_missing_start_key_returns_422(client: TestClient) -> None:
    """A sunlight_schedule entry missing 'start' must return 422."""
    samples = _make_samples(72)
    assumptions = _make_l2_assumptions()
    assumptions["sunlight_schedule"] = [{"end": "2026-08-01T18:00:00Z"}]
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": assumptions},
    )
    assert resp.status_code == 422


def test_fix4_sunlight_missing_end_key_returns_422(client: TestClient) -> None:
    """A sunlight_schedule entry missing 'end' must return 422."""
    samples = _make_samples(72)
    assumptions = _make_l2_assumptions()
    assumptions["sunlight_schedule"] = [{"start": "2026-08-01T06:00:00Z"}]
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": assumptions},
    )
    assert resp.status_code == 422


def test_fix4_payload_non_dict_entry_returns_422(client: TestClient) -> None:
    """A non-dict payload_schedule entry must return 422, not 500."""
    samples = _make_samples(72)
    assumptions = _make_l2_assumptions()
    assumptions["payload_schedule"] = [42]
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": assumptions},
    )
    assert resp.status_code == 422, (
        f"Expected 422 for non-dict payload entry, got {resp.status_code}"
    )
    body = resp.json()
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value


def test_fix4_payload_missing_draw_w_returns_422(client: TestClient) -> None:
    """A payload_schedule entry missing 'draw_w' must return 422."""
    samples = _make_samples(72)
    assumptions = _make_l2_assumptions()
    assumptions["payload_schedule"] = [
        {"start": "2026-08-01T06:00:00Z", "end": "2026-08-01T08:00:00Z"}
        # draw_w absent
    ]
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": assumptions},
    )
    assert resp.status_code == 422


def test_fix4_numeric_string_capacity_returns_422(client: TestClient) -> None:
    """A numeric string for battery_capacity_wh must return 422."""
    samples = _make_samples(72)
    assumptions = _make_l2_assumptions()
    assumptions["battery_capacity_wh"] = "500.0"   # string — must be rejected
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": assumptions},
    )
    assert resp.status_code == 422, (
        f"Expected 422 for numeric string capacity, got {resp.status_code}"
    )


def test_fix4_bool_efficiency_returns_422(client: TestClient) -> None:
    """A boolean for power_conversion_efficiency must return 422."""
    samples = _make_samples(72)
    assumptions = _make_l2_assumptions()
    assumptions["power_conversion_efficiency"] = True
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": assumptions},
    )
    assert resp.status_code == 422


def test_fix4_partial_assumptions_still_200(client: TestClient) -> None:
    """Genuinely partial assumptions (missing fields) must return 200, projection null."""
    samples = _make_samples(72)
    # Only one field — all others absent
    partial = {"battery_capacity_wh": 500.0}
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": partial},
    )
    assert resp.status_code == 200, (
        f"Expected 200 for partial assumptions, got {resp.status_code}"
    )
    body = resp.json()
    assert body["deterministic_projection"] is None
    assert body.get("projection_omitted_reason") is not None


# ---------------------------------------------------------------------------
# Fix 5 — communications_status / command_activity / image_utility_score
#          accepted in schema but never enter ML features
# ---------------------------------------------------------------------------

def test_fix5_all_8_raw_fields_accepted(client: TestClient) -> None:
    """All 8 raw telemetry fields (including the 3 non-ML fields) are accepted."""
    samples = _make_samples(72)   # already includes command_activity, comm_status, score
    assert all("command_activity" in s for s in samples)
    assert all("communications_status" in s for s in samples)
    assert all("image_utility_score" in s for s in samples)
    resp = client.post("/api/v1/power-risk/predict", json={"samples": samples})
    assert resp.status_code == 200, resp.text


# The eight required raw telemetry field names (contract Section 5).
_ALL_8_REQUIRED_FIELDS = [
    "timestamp",
    "battery_soc_percent",
    "bus_voltage_v",
    "solar_array_current_a",
    "payload_power_draw_w",
    "command_activity",
    "communications_status",
    "image_utility_score",
]


@pytest.mark.parametrize("missing_field", _ALL_8_REQUIRED_FIELDS)
def test_fix5_missing_required_field_returns_422(
    client: TestClient, missing_field: str
) -> None:
    """Omitting any one of the 8 required raw fields from all samples must return
    422 with error=INVALID_SAMPLE_SCHEMA, sample_index=0, field=<missing_field>,
    plus all four mandatory safety-envelope fields."""
    # Build one complete sample then remove the target field
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    sample = {
        "timestamp": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "battery_soc_percent": 55.0,
        "bus_voltage_v": 28.0,
        "solar_array_current_a": 7.5,
        "payload_power_draw_w": 35.0,
        "command_activity": "NOMINAL_ATTITUDE_HOLD",
        "communications_status": "NO_CONTACT_WINDOW",
        "image_utility_score": 0.75,
    }
    del sample[missing_field]

    resp = client.post("/api/v1/power-risk/predict", json={"samples": [sample]})
    assert resp.status_code == 422, (
        f"Expected 422 when {missing_field!r} is absent, got {resp.status_code}"
    )
    body = resp.json()
    # Safety envelope must be present on every error response
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value, (
            f"Safety field {field!r} missing when {missing_field!r} is absent"
        )
    # Structured missing-field diagnostic
    assert body.get("error") == "INVALID_SAMPLE_SCHEMA", (
        f"Expected error=INVALID_SAMPLE_SCHEMA when {missing_field!r} is absent, "
        f"got {body.get('error')!r}"
    )
    assert body.get("sample_index") == 0, (
        f"Expected sample_index=0 when {missing_field!r} is absent, "
        f"got {body.get('sample_index')!r}"
    )
    assert body.get("field") == missing_field, (
        f"Expected field={missing_field!r}, got {body.get('field')!r}"
    )


def test_fix5_non_ml_fields_do_not_affect_probability(client: TestClient) -> None:
    """Different values for communications_status, command_activity, image_utility_score
    must not change the breach_probability (they must not enter the ML pipeline).
    Both sample sets carry all 8 required fields; only the 3 non-ML fields differ."""
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    base_samples = []
    for i in range(72):
        ts = start + timedelta(seconds=300 * i)
        base_samples.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "battery_soc_percent": 55.0,
            "bus_voltage_v": 28.0,
            "solar_array_current_a": 7.5,
            "payload_power_draw_w": 35.0,
            "command_activity": "NOMINAL_ATTITUDE_HOLD",
            "communications_status": "NO_CONTACT_WINDOW",
            "image_utility_score": 0.50,
        })

    extra_samples = copy.deepcopy(base_samples)
    for s in extra_samples:
        s["command_activity"] = "IMAGING_BURST"
        s["communications_status"] = "IN_CONTACT_WINDOW"
        s["image_utility_score"] = 0.99

    resp_base = client.post("/api/v1/power-risk/predict", json={"samples": base_samples})
    resp_extra = client.post("/api/v1/power-risk/predict", json={"samples": extra_samples})
    assert resp_base.status_code == 200
    assert resp_extra.status_code == 200

    p_base = resp_base.json()["ai_prediction"]["breach_probability"]
    p_extra = resp_extra.json()["ai_prediction"]["breach_probability"]
    assert p_base == p_extra, (
        f"Non-ML fields changed breach_probability: {p_base} vs {p_extra}. "
        "communications_status/command_activity/image_utility_score must never enter ML features."
    )


def test_fix5_ml_features_only_contain_approved_12(client: TestClient) -> None:
    """all_contributions must contain exactly the 12 approved feature names."""
    samples = _make_samples(72)
    resp = client.post("/api/v1/power-risk/predict", json={"samples": samples})
    assert resp.status_code == 200
    features_used = [
        c["feature"] for c in resp.json()["ai_prediction"]["all_contributions"]
    ]
    forbidden = {"communications_status", "command_activity", "image_utility_score"}
    for f in features_used:
        assert f not in forbidden, (
            f"Forbidden non-ML field {f!r} appeared in ML contributions"
        )
    assert set(features_used) == set(FEATURE_ORDER), (
        f"ML features mismatch.\nExpected: {sorted(FEATURE_ORDER)}\nGot: {sorted(features_used)}"
    )


# ===========================================================================
# FINAL AUDIT CORRECTION REGRESSION TESTS  (Fixes 1–6)
# ===========================================================================

# ---------------------------------------------------------------------------
# Fix 1 — probability_note reflects 240-scenario final fit
# ---------------------------------------------------------------------------

def test_fixA1_probability_note_mentions_240_scenarios(client: TestClient) -> None:
    """probability_note must state the final pipeline was fitted on 240 scenarios."""
    samples = _make_samples(72)
    resp = client.post("/api/v1/power-risk/predict", json={"samples": samples})
    assert resp.status_code == 200
    note = resp.json()["ai_prediction"]["probability_note"]
    assert "240" in note, (
        f"probability_note must mention 240-scenario final fit; got: {note!r}"
    )
    # Original synthetic-only warnings must be preserved
    assert "synthetic" in note.lower(), (
        f"probability_note must warn about synthetic distribution: {note!r}"
    )
    assert "not a validated real-spacecraft" in note.lower() or "not a validated" in note.lower(), (
        f"probability_note must carry real-spacecraft warning: {note!r}"
    )


def test_fixA1_probability_note_does_not_say_only_180(client: TestClient) -> None:
    """probability_note must not claim the classifier was trained only on 180 scenarios
    without mentioning the 240-scenario final fit."""
    samples = _make_samples(72)
    resp = client.post("/api/v1/power-risk/predict", json={"samples": samples})
    note = resp.json()["ai_prediction"]["probability_note"]
    # If the note says "180" it must also say "240" (the final fit count)
    if "180" in note:
        assert "240" in note, (
            f"probability_note mentions 180 but not 240 (final fit): {note!r}"
        )


# ---------------------------------------------------------------------------
# Fix 2 — L2 projection runs independently in degraded mode
# ---------------------------------------------------------------------------

def test_fixA2_degraded_with_complete_assumptions_has_projection(
    client: TestClient,
) -> None:
    """A degraded-mode response with complete valid assumptions must include
    a non-null deterministic_projection with 24 hourly entries."""
    samples = _make_samples(5)  # degraded: < 72 samples
    assumptions = _make_l2_assumptions(
        start=datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    )
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": assumptions},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded", f"Expected status=degraded, got {body['status']!r}"
    assert body["ai_prediction"] is None, "ai_prediction must be null in degraded mode"
    proj = body.get("deterministic_projection")
    assert proj is not None, (
        "deterministic_projection must be non-null when complete assumptions are supplied "
        "even in degraded mode"
    )
    assert proj["not_ai_output"] is True
    assert len(proj["hourly_projection"]) == 24, (
        f"Expected 24 hourly entries in degraded L2 projection, got "
        f"{len(proj['hourly_projection'])}"
    )


def test_fixA2_degraded_with_partial_assumptions_has_null_projection(
    client: TestClient,
) -> None:
    """A degraded-mode response with partial assumptions must have null projection."""
    samples = _make_samples(5)
    partial = {"battery_capacity_wh": 500.0}  # only one of five required fields
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": partial},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["deterministic_projection"] is None
    assert body.get("projection_omitted_reason") is not None


def test_fixA2_degraded_with_invalid_assumptions_returns_422(
    client: TestClient,
) -> None:
    """Complete but invalid assumptions in degraded mode must return 422."""
    samples = _make_samples(5)
    bad = _make_l2_assumptions()
    bad["battery_capacity_wh"] = -1.0  # invalid: must be > 0
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": samples, "projection_assumptions": bad},
    )
    assert resp.status_code == 422
    body = resp.json()
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value


def test_fixA2_degraded_l2_no_model_required() -> None:
    """Degraded-mode L2 must work even when the model pipeline is not loaded."""
    import backend.app.prediction_service as svc
    from backend.app.main import app

    original_pipeline = svc._pipeline
    original_error = svc._pipeline_load_error
    try:
        svc._pipeline = None
        svc._pipeline_load_error = "MODEL_NOT_LOADED"

        c = TestClient(app, raise_server_exceptions=False)
        samples = _make_samples(5)
        assumptions = _make_l2_assumptions(
            start=datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
        )
        resp = c.post(
            "/api/v1/power-risk/predict",
            json={"samples": samples, "projection_assumptions": assumptions},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["deterministic_projection"] is not None, (
            "L2 projection must succeed in degraded mode even without a loaded model"
        )
        assert len(body["deterministic_projection"]["hourly_projection"]) == 24
    finally:
        svc._pipeline = original_pipeline
        svc._pipeline_load_error = original_error


# ---------------------------------------------------------------------------
# Fix 3 — POST {} returns 422 EMPTY_WINDOW with safety envelope
# ---------------------------------------------------------------------------

def test_fixA3_post_empty_object_returns_422_empty_window(client: TestClient) -> None:
    """POST {} (missing samples key) must return 422 with error=EMPTY_WINDOW
    and all four safety-envelope fields."""
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={},
    )
    assert resp.status_code == 422, (
        f"Expected 422 for POST {{}}, got {resp.status_code}"
    )
    body = resp.json()
    assert body.get("error") == "EMPTY_WINDOW", (
        f"Expected error=EMPTY_WINDOW for POST {{}}, got {body.get('error')!r}"
    )
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value, (
            f"Safety field {field!r} missing on POST {{}} 422 response"
        )


def test_fixA3_post_empty_samples_array_also_422_empty_window(
    client: TestClient,
) -> None:
    """POST {samples: []} must also return 422 with error=EMPTY_WINDOW."""
    resp = client.post(
        "/api/v1/power-risk/predict",
        json={"samples": []},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body.get("error") == "EMPTY_WINDOW", (
        f"Expected error=EMPTY_WINDOW for empty array, got {body.get('error')!r}"
    )
    for field, value in SAFETY_FIELDS.items():
        assert body.get(field) == value


# ---------------------------------------------------------------------------
# Fix 4 — _validate_samples enforces all 8 required raw fields
# ---------------------------------------------------------------------------

def test_fixA4_missing_command_activity_raises_in_validate_samples() -> None:
    """_validate_samples must raise when command_activity is missing."""
    import backend.app.prediction_service as svc
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    sample = {
        "timestamp": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "battery_soc_percent": 55.0,
        "bus_voltage_v": 28.0,
        "solar_array_current_a": 7.5,
        "payload_power_draw_w": 35.0,
        # command_activity absent
        "communications_status": "NO_CONTACT_WINDOW",
        "image_utility_score": 0.75,
    }
    with pytest.raises(ValueError, match="MISSING_FIELD:command_activity"):
        svc._validate_samples([sample])


def test_fixA4_missing_communications_status_raises_in_validate_samples() -> None:
    """_validate_samples must raise when communications_status is missing."""
    import backend.app.prediction_service as svc
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    sample = {
        "timestamp": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "battery_soc_percent": 55.0,
        "bus_voltage_v": 28.0,
        "solar_array_current_a": 7.5,
        "payload_power_draw_w": 35.0,
        "command_activity": "NOMINAL_ATTITUDE_HOLD",
        # communications_status absent
        "image_utility_score": 0.75,
    }
    with pytest.raises(ValueError, match="MISSING_FIELD:communications_status"):
        svc._validate_samples([sample])


def test_fixA4_missing_image_utility_score_raises_in_validate_samples() -> None:
    """_validate_samples must raise when image_utility_score is missing."""
    import backend.app.prediction_service as svc
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    sample = {
        "timestamp": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "battery_soc_percent": 55.0,
        "bus_voltage_v": 28.0,
        "solar_array_current_a": 7.5,
        "payload_power_draw_w": 35.0,
        "command_activity": "NOMINAL_ATTITUDE_HOLD",
        "communications_status": "NO_CONTACT_WINDOW",
        # image_utility_score absent
    }
    with pytest.raises(ValueError, match="MISSING_FIELD:image_utility_score"):
        svc._validate_samples([sample])


def test_fixA4_non_string_command_activity_raises() -> None:
    """_validate_samples must reject non-string command_activity."""
    import backend.app.prediction_service as svc
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    sample = {
        "timestamp": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "battery_soc_percent": 55.0,
        "bus_voltage_v": 28.0,
        "solar_array_current_a": 7.5,
        "payload_power_draw_w": 35.0,
        "command_activity": 42,  # not a string
        "communications_status": "NO_CONTACT_WINDOW",
        "image_utility_score": 0.75,
    }
    with pytest.raises(ValueError, match="INVALID_TYPE"):
        svc._validate_samples([sample])


def test_fixA4_bool_image_utility_score_raises() -> None:
    """_validate_samples must reject bool image_utility_score."""
    import backend.app.prediction_service as svc
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    sample = {
        "timestamp": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "battery_soc_percent": 55.0,
        "bus_voltage_v": 28.0,
        "solar_array_current_a": 7.5,
        "payload_power_draw_w": 35.0,
        "command_activity": "NOMINAL_ATTITUDE_HOLD",
        "communications_status": "NO_CONTACT_WINDOW",
        "image_utility_score": True,  # bool
    }
    with pytest.raises(ValueError, match="INVALID_TYPE"):
        svc._validate_samples([sample])


def test_fixA4_non_ml_fields_absent_from_feature_order() -> None:
    """FEATURE_ORDER must not contain command_activity, communications_status,
    or image_utility_score."""
    from backend.scripts.train_power_risk_model import FEATURE_ORDER as FO
    forbidden = {"command_activity", "communications_status", "image_utility_score"}
    assert not forbidden.intersection(set(FO)), (
        f"Non-ML fields found in FEATURE_ORDER: {forbidden.intersection(set(FO))}"
    )


# ---------------------------------------------------------------------------
# Fix 5 — Contract doc assertions (checked via the running service)
# ---------------------------------------------------------------------------

def test_fixA5_scenario_id_is_synth_demo_public(client: TestClient) -> None:
    """GET demo endpoint must return scenario_id=SYNTH-DEMO-PUBLIC-001."""
    resp = client.get("/api/v1/mock/power-risk-prediction")
    assert resp.status_code == 200
    assert resp.json()["scenario_id"] == "SYNTH-DEMO-PUBLIC-001", (
        f"Expected scenario_id=SYNTH-DEMO-PUBLIC-001, got "
        f"{resp.json().get('scenario_id')!r}"
    )


def test_fixA5_high_probability_triggers_high_advisory(client: TestClient) -> None:
    """breach_probability >= 0.70 must produce HIGH advisory (not ELEVATED).
    Verifies that 0.79 exceeds the 0.70 threshold as stated in the contract."""
    import backend.app.prediction_service as svc

    original_run = svc.run_prediction

    def _mock_high_prob(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_run(*args, **kwargs)
        # Force the probability to a value clearly above 0.70
        result["ai_prediction"]["breach_probability"] = 0.79
        result["advisory"] = svc._derive_advisory(0.79, [], result["ai_prediction"])
        return result

    svc.run_prediction = _mock_high_prob  # type: ignore[assignment]
    try:
        c = TestClient(__import__("backend.app.main", fromlist=["app"]).app,
                       raise_server_exceptions=False)
        resp = c.post("/api/v1/power-risk/predict", json={"samples": _make_samples(72)})
        assert resp.status_code == 200
        advisory = resp.json()["advisory"]
        assert advisory["risk_summary"] == "HIGH", (
            f"0.79 must trigger HIGH (≥ 0.70 threshold); got {advisory['risk_summary']!r}"
        )
    finally:
        svc.run_prediction = original_run  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fix 6 — Stale ±1-second docstring is removed; _query_timestamp is clean
# ---------------------------------------------------------------------------

def test_fixA6_validate_samples_docstring_no_1s_tolerance() -> None:
    """_validate_samples docstring must not claim ±1 s tolerance."""
    import backend.app.prediction_service as svc
    import inspect
    src = inspect.getdoc(svc._validate_samples) or ""
    assert "± 1 s" not in src and "+/- 1" not in src and "1 s tolerance" not in src, (
        "Stale ±1-second tolerance claim found in _validate_samples docstring"
    )


def test_fixA6_query_timestamp_returns_iso_string() -> None:
    """_query_timestamp must return a valid ISO-8601 UTC string with no dead code."""
    import backend.app.prediction_service as svc
    ts = svc._query_timestamp()
    assert isinstance(ts, str)
    # Must parse as ISO-8601 UTC
    from datetime import datetime, timezone
    parsed = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert parsed is not None
