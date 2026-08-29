"""Core prediction service for SpaceBNS power-risk assessment.

This module is the single callable used by both:
  - POST /api/v1/power-risk/predict
  - GET  /api/v1/mock/power-risk-prediction

Design constraints (contract Sections 3–9)
------------------------------------------
- Only ``history`` samples from the request (or mock file) are used as ML inputs.
- Feature extraction calls ``extract_features()`` — no direct field access.
- The feature vector is built in ``FEATURE_ORDER`` from ``train_power_risk_model``.
- Contributions = standardized_value × coefficient for each feature.
- ``apply_power_thresholds()`` is called on the *latest* sample for L3.
- The input sample list is never mutated.
- No command authority; all recommendations are advisory only.
- Stack traces and filesystem paths are never exposed.
"""

from __future__ import annotations

import copy
import math
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Imports — deferred heavy dependencies (joblib, sklearn) are imported here
# only once, not at module import time, to allow the module to be imported
# safely in tests that patch the pipeline.
# ---------------------------------------------------------------------------

from backend.app.features import extract_features
from backend.app.policy import apply_power_thresholds
from backend.app.safety import SAFETY_ENVELOPE
from backend.scripts.train_power_risk_model import (
    FEATURE_ORDER,
    NEAR_CONSTANT_SCALE_FLOOR,
)

# ---------------------------------------------------------------------------
# Model pipeline (loaded once at import time; None if file absent)
# ---------------------------------------------------------------------------

_pipeline: Any = None
_pipeline_load_error: str | None = None

MODEL_VERSION = "0.1.1"
_PROBABILITY_NOTE = (
    "Probability estimated from a logistic regression classifier. The final "
    "pipeline was fitted on 240 synthetic scenarios (train + validation "
    "combined) after regularisation strength C was selected using the "
    "180-scenario training split and a 60-scenario validation split. This is "
    "an estimate learned from the synthetic scenario distribution and is not a "
    "validated real-spacecraft failure probability. No fixed demonstration "
    "value may be hardcoded or presented as operational truth. Features with "
    "no measurable variation in the synthetic training split are retained in "
    "the schema but neutralized so numerical noise cannot dominate inference."
)

# Expected cadence between consecutive samples (seconds)
_EXPECTED_CADENCE_S = 300  # 5 minutes


def load_pipeline(path: str) -> None:
    """Load the serialised sklearn pipeline from *path* into the module cache.

    Called once at application startup.  Sets ``_pipeline`` on success or
    records the error string on failure so the endpoint can return 503.

    After loading, validates:
    - pipeline contains ``scaler`` and ``clf`` named steps
    - exactly 12 input features
    - classifier classes are exactly [0, 1]
    - scaler/coefficient dimensions match FEATURE_ORDER
    """
    global _pipeline, _pipeline_load_error
    try:
        import joblib  # noqa: PLC0415
        candidate = joblib.load(path)

        # Validate pipeline structure
        if not (hasattr(candidate, "named_steps")
                and "scaler" in candidate.named_steps
                and "clf" in candidate.named_steps):
            _pipeline = None
            _pipeline_load_error = "MODEL_NOT_LOADED"
            return

        scaler = candidate.named_steps["scaler"]
        clf = candidate.named_steps["clf"]

        # Exactly 12 input features
        n_features = (
            scaler.mean_.shape[0]
            if hasattr(scaler, "mean_") and hasattr(scaler.mean_, "shape")
            else None
        )
        if n_features != len(FEATURE_ORDER):
            _pipeline = None
            _pipeline_load_error = "MODEL_NOT_LOADED"
            return

        # Classifier classes must be exactly [0, 1]
        classes = list(clf.classes_) if hasattr(clf, "classes_") else []
        if classes != [0, 1]:
            _pipeline = None
            _pipeline_load_error = "MODEL_NOT_LOADED"
            return

        # Scaler and coefficient dimensions must match FEATURE_ORDER
        coef = clf.coef_
        coef_list = coef.tolist() if hasattr(coef, "tolist") else list(coef)
        # Binary LR: shape (1, n) or (n,)
        n_coef = len(coef_list[0]) if isinstance(coef_list[0], list) else len(coef_list)
        if n_coef != len(FEATURE_ORDER):
            _pipeline = None
            _pipeline_load_error = "MODEL_NOT_LOADED"
            return
        scale_values = [float(value) for value in scaler.scale_]
        if len(scale_values) != len(FEATURE_ORDER):
            _pipeline = None
            _pipeline_load_error = "MODEL_NOT_LOADED"
            return
        if any(
            not math.isfinite(scale) or scale < NEAR_CONSTANT_SCALE_FLOOR
            for scale in scale_values
        ):
            _pipeline = None
            _pipeline_load_error = "MODEL_NOT_LOADED"
            return

        coefficient_values = (
            coef_list[0] if isinstance(coef_list[0], list) else coef_list
        )
        if any(not math.isfinite(float(value)) for value in coefficient_values):
            _pipeline = None
            _pipeline_load_error = "MODEL_NOT_LOADED"
            return

        _pipeline = candidate
        _pipeline_load_error = None
    except Exception:  # noqa: BLE001
        _pipeline = None
        _pipeline_load_error = "MODEL_NOT_LOADED"


def get_pipeline() -> Any:
    """Return the loaded pipeline, or None if unavailable."""
    return _pipeline


def get_pipeline_load_error() -> str | None:
    """Return the load error string, or None if the pipeline loaded successfully."""
    return _pipeline_load_error


# ---------------------------------------------------------------------------
# Telemetry validation
# ---------------------------------------------------------------------------

_REQUIRED_NUMERIC_FIELDS = [
    "battery_soc_percent",
    "bus_voltage_v",
    "solar_array_current_a",
    "payload_power_draw_w",
]


def _parse_iso_strict(ts: Any) -> datetime:
    """Parse an ISO-8601 UTC timestamp string.  Raises ValueError on failure."""
    if not isinstance(ts, str):
        raise ValueError("timestamp must be a string")
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


# Required string fields (not ML features — validated but never enter FEATURE_ORDER)
_REQUIRED_STRING_FIELDS = [
    "command_activity",
    "communications_status",
]

# image_utility_score: required finite float, not an ML feature
_IMAGE_UTILITY_FIELD = "image_utility_score"


def _validate_samples(samples: list[dict[str, Any]]) -> None:
    """Validate a list of telemetry samples.

    Raises
    ------
    ValueError
        On any of the following:
        - missing required fields (all 8 raw telemetry fields)
        - bool values for numeric fields
        - numeric strings instead of actual numbers
        - NaN or infinity in numeric fields
        - non-string values for string fields
        - invalid or non-string timestamp
        - duplicate timestamps
        - non-ascending timestamps
        - wrong cadence (not exactly 5 minutes)
    """
    if not samples:
        raise ValueError("EMPTY_WINDOW")

    parsed_timestamps: list[float] = []

    for idx, sample in enumerate(samples):
        # --- timestamp ---
        ts_raw = sample.get("timestamp")
        try:
            ts_dt = _parse_iso_strict(ts_raw)
        except (ValueError, TypeError):
            raise ValueError(f"INVALID_TIMESTAMP at sample index {idx}")

        ts_epoch = ts_dt.timestamp()

        # --- numeric power fields (used by ML feature extraction) ---
        for field in _REQUIRED_NUMERIC_FIELDS:
            if field not in sample:
                raise ValueError(f"MISSING_FIELD:{field} at sample index {idx}")
            val = sample[field]
            # bool is a subclass of int in Python — reject it
            if isinstance(val, bool):
                raise ValueError(f"INVALID_TYPE:bool for {field} at sample index {idx}")
            # numeric strings are not accepted
            if isinstance(val, str):
                raise ValueError(
                    f"INVALID_TYPE:numeric_string for {field} at sample index {idx}"
                )
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"INVALID_TYPE for {field} at sample index {idx}"
                )
            fval = float(val)
            if not math.isfinite(fval):
                raise ValueError(
                    f"NON_FINITE_VALUE for {field} at sample index {idx}"
                )

        # --- required string fields (NOT ML features) ---
        for field in _REQUIRED_STRING_FIELDS:
            if field not in sample:
                raise ValueError(f"MISSING_FIELD:{field} at sample index {idx}")
            val = sample[field]
            if not isinstance(val, str):
                raise ValueError(
                    f"INVALID_TYPE:not_a_string for {field} at sample index {idx}"
                )

        # --- image_utility_score: required finite float (NOT an ML feature) ---
        if _IMAGE_UTILITY_FIELD not in sample:
            raise ValueError(
                f"MISSING_FIELD:{_IMAGE_UTILITY_FIELD} at sample index {idx}"
            )
        ius_val = sample[_IMAGE_UTILITY_FIELD]
        if isinstance(ius_val, bool):
            raise ValueError(
                f"INVALID_TYPE:bool for {_IMAGE_UTILITY_FIELD} at sample index {idx}"
            )
        if isinstance(ius_val, str):
            raise ValueError(
                f"INVALID_TYPE:not_a_number for {_IMAGE_UTILITY_FIELD} at sample index {idx}"
            )
        if not isinstance(ius_val, (int, float)):
            raise ValueError(
                f"INVALID_TYPE for {_IMAGE_UTILITY_FIELD} at sample index {idx}"
            )
        if not math.isfinite(float(ius_val)):
            raise ValueError(
                f"NON_FINITE_VALUE for {_IMAGE_UTILITY_FIELD} at sample index {idx}"
            )

        parsed_timestamps.append(ts_epoch)

    # --- temporal ordering ---
    for i in range(1, len(parsed_timestamps)):
        delta = parsed_timestamps[i] - parsed_timestamps[i - 1]
        if delta <= 0:
            raise ValueError(
                f"DUPLICATE_OR_NONASCENDING_TIMESTAMP at sample index {i}"
            )
        # Cadence check: must be exactly 300 s (no tolerance)
        if delta != _EXPECTED_CADENCE_S:
            raise ValueError(
                f"WRONG_CADENCE at sample index {i}: "
                f"expected {_EXPECTED_CADENCE_S}s, got {delta:.1f}s"
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _query_timestamp() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_contributions(
    raw_feature_vector: list[float],
    pipeline: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute per-feature contributions from the fitted scaler and LR coefficients.

    contribution_i = standardized_value_i × coefficient_i

    Returns
    -------
    all_contributions : list of 12 dicts in FEATURE_ORDER
    top_contributions : top-3 by absolute magnitude (descending)
    """
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]

    # sklearn stores mean_ and scale_ as numpy arrays; convert to plain floats
    means = [float(m) for m in scaler.mean_]
    scales = [float(s) for s in scaler.scale_]
    # coefficients for class 1 (index 0 = class 0, index 1 = class 1 or
    # 1-D for binary; handle both shapes)
    coef_array = clf.coef_
    if hasattr(coef_array, "tolist"):
        coef_list_raw = coef_array.tolist()
    else:
        coef_list_raw = list(coef_array)

    # For binary LR, coef_ is shape (1, n_features) or (n_features,)
    if isinstance(coef_list_raw[0], list):
        # shape (1, 12) — take the first (and only) row
        coefficients = coef_list_raw[0]
    else:
        coefficients = coef_list_raw

    all_contributions: list[dict[str, Any]] = []
    for i, feature_name in enumerate(FEATURE_ORDER):
        raw_val = float(raw_feature_vector[i])
        coef = float(coefficients[i])
        if (
            not math.isfinite(raw_val)
            or not math.isfinite(means[i])
            or not math.isfinite(scales[i])
            or scales[i] < NEAR_CONSTANT_SCALE_FLOOR
            or not math.isfinite(coef)
        ):
            raise ValueError("Invalid model contribution parameters")

        std_val = (raw_val - means[i]) / scales[i]
        contribution = std_val * coef
        if not math.isfinite(std_val) or not math.isfinite(contribution):
            raise ValueError("Invalid model contribution result")
        all_contributions.append(
            {
                "feature": feature_name,
                "standardized_value": round(std_val, 6),
                "coefficient": round(coef, 6),
                "contribution": round(contribution, 6),
            }
        )

    top_contributions = sorted(
        all_contributions,
        key=lambda d: abs(d["contribution"]),
        reverse=True,
    )[:3]

    return all_contributions, top_contributions


def _derive_advisory(
    breach_probability: float | None,
    findings: list[dict[str, str]],
    ai_prediction: dict[str, Any] | None,
) -> dict[str, Any]:
    """Derive L4 advisory from L1 and L3 (contract Section 8.5).

    Conditions evaluated in order; first match applies.

    When ai_prediction is null (degraded mode), L3 findings still drive
    HIGH/ELEVATED.  UNKNOWN is returned only when AI is unavailable AND no
    L3 condition requires a stronger result.
    """
    authority_note = (
        "Advisory output only. No automated action has been or will be taken."
    )

    # Check for active breach in L3
    active_breach_codes = {"BATTERY_SOC_LOW", "BUS_VOLTAGE_LOW"}
    has_active_breach = any(
        f["code"] in active_breach_codes for f in findings
    )

    n_findings = len(findings)

    # Build basis string — depends on whether AI is available
    if breach_probability is None:
        if has_active_breach:
            basis = (
                "Active breach detected by L3 threshold rules. "
                "L1 prediction not applicable; L3 takes precedence."
            )
        else:
            basis = (
                f"AI prediction unavailable; insufficient sample count. "
                f"{n_findings} safety threshold finding"
                f"{'s' if n_findings != 1 else ''} active."
            )
    else:
        if has_active_breach:
            basis = (
                "Active breach detected by L3 threshold rules. "
                "L1 prediction not applicable; L3 takes precedence."
            )
        else:
            basis = (
                f"AI breach probability {round(breach_probability, 2)} "
                f"{'exceeds 0.70 threshold' if breach_probability >= 0.70 else 'exceeds 0.40 threshold' if breach_probability >= 0.40 else 'below advisory thresholds'}; "
                f"{n_findings} safety threshold finding{'s' if n_findings != 1 else ''} active."
            )

    # --- Priority rule 1: HIGH ---
    # Triggered by AI probability OR by L3 finding count — regardless of AI availability
    high_by_prob = breach_probability is not None and breach_probability >= 0.70
    if high_by_prob or n_findings >= 3:
        return {
            "risk_summary": "HIGH",
            "recommendation": "DEFER_LOW_PRIORITY_FUTURE_IMAGING",
            "basis": basis,
            "human_action_required": True,
            "authority_note": authority_note,
        }

    # --- Priority rule 2: ELEVATED ---
    elevated_by_prob = breach_probability is not None and breach_probability >= 0.40
    if elevated_by_prob or n_findings >= 1:
        return {
            "risk_summary": "ELEVATED",
            "recommendation": "INCREASE_MONITORING_FREQUENCY",
            "basis": basis,
            "human_action_required": True,
            "authority_note": authority_note,
        }

    # --- Priority rule 3: UNKNOWN (AI unavailable, no L3 condition) ---
    if breach_probability is None:
        return {
            "risk_summary": "UNKNOWN",
            "recommendation": "SUPPLY_SUFFICIENT_HISTORY",
            "basis": basis,
            "human_action_required": True,
            "authority_note": authority_note,
        }

    # --- Priority rule 4: NOMINAL ---
    return {
        "risk_summary": "NOMINAL",
        "recommendation": "CONTINUE_MONITORING",
        "basis": basis,
        "human_action_required": False,
        "authority_note": authority_note,
    }


def _compute_l2_projection(
    latest_sample: dict[str, Any],
    assumptions: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Attempt to compute the deterministic energy projection (L2).

    Returns
    -------
    (projection_dict, None)
        on success.
    (None, reason_string)
        when any required assumption is missing (partial assumptions).

    Raises
    ------
    ValueError
        When all required fields are present but contain invalid values.

    Contract Section 9: implements only what is explicitly specified.
    Marked as not AI output.

    Physical assumptions preserved:
    1. Physics-based energy balance over a 72-sample (6-hour) window.
    2. Hourly midpoint sampling — the schedule is sampled at the midpoint of
       each projected hour; this is a simplifying prototype assumption.
    3. Constant latest solar current — the solar array current from the latest
       telemetry sample is held constant throughout the projection; this is a
       simplifying prototype assumption.
    4. Nominal bus voltage (28 V) used for power conversion.
    5. SOC clamped to [0, 100] %.
    """
    required_fields = [
        "battery_capacity_wh",
        "base_spacecraft_load_w",
        "power_conversion_efficiency",
        "sunlight_schedule",
        "payload_schedule",
    ]
    missing = [f for f in required_fields if f not in assumptions or assumptions[f] is None]
    if missing:
        return None, "Required physical assumptions not supplied"

    # --- validate numeric assumptions ---
    def _require_finite(name: str, val: Any) -> float:
        if isinstance(val, bool):
            raise ValueError(f"INVALID_PROJECTION_ASSUMPTIONS: {name} must be numeric")
        # Reject numeric strings — float("1.5") would silently accept them
        if isinstance(val, str):
            raise ValueError(
                f"INVALID_PROJECTION_ASSUMPTIONS: {name} must be numeric, not a string"
            )
        try:
            fval = float(val)
        except (TypeError, ValueError):
            raise ValueError(f"INVALID_PROJECTION_ASSUMPTIONS: {name} must be numeric")
        if not math.isfinite(fval):
            raise ValueError(f"INVALID_PROJECTION_ASSUMPTIONS: {name} must be finite")
        return fval

    capacity_wh = _require_finite("battery_capacity_wh", assumptions["battery_capacity_wh"])
    base_load_w = _require_finite("base_spacecraft_load_w", assumptions["base_spacecraft_load_w"])
    efficiency = _require_finite("power_conversion_efficiency", assumptions["power_conversion_efficiency"])

    if capacity_wh <= 0:
        raise ValueError("INVALID_PROJECTION_ASSUMPTIONS: battery_capacity_wh must be > 0")
    if base_load_w < 0:
        raise ValueError("INVALID_PROJECTION_ASSUMPTIONS: base_spacecraft_load_w must be >= 0")
    if not (0 < efficiency <= 1):
        raise ValueError(
            "INVALID_PROJECTION_ASSUMPTIONS: power_conversion_efficiency must be in (0, 1]"
        )

    sunlight_schedule = assumptions["sunlight_schedule"]
    payload_schedule = assumptions["payload_schedule"]

    if not isinstance(sunlight_schedule, list):
        raise ValueError("INVALID_PROJECTION_ASSUMPTIONS: sunlight_schedule must be a list")
    if not isinstance(payload_schedule, list):
        raise ValueError("INVALID_PROJECTION_ASSUMPTIONS: payload_schedule must be a list")

    # Parse ISO timestamp helper
    def _parse_iso(ts: Any) -> datetime:
        if not isinstance(ts, str):
            raise ValueError("INVALID_PROJECTION_ASSUMPTIONS: timestamps must be ISO strings")
        try:
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            raise ValueError(
                f"INVALID_PROJECTION_ASSUMPTIONS: invalid ISO timestamp: {ts!r}"
            )

    # Parse and validate sunlight intervals
    sunlight_intervals: list[tuple[float, float]] = []
    for i, window in enumerate(sunlight_schedule):
        if not isinstance(window, dict):
            raise ValueError(
                f"INVALID_PROJECTION_ASSUMPTIONS: sunlight_schedule[{i}] must be a dict"
            )
        if "start" not in window:
            raise ValueError(
                f"INVALID_PROJECTION_ASSUMPTIONS: sunlight_schedule[{i}] missing 'start'"
            )
        if "end" not in window:
            raise ValueError(
                f"INVALID_PROJECTION_ASSUMPTIONS: sunlight_schedule[{i}] missing 'end'"
            )
        t_start = _parse_iso(window["start"])
        t_end = _parse_iso(window["end"])
        if t_start >= t_end:
            raise ValueError(
                f"INVALID_PROJECTION_ASSUMPTIONS: sunlight_schedule[{i}] start >= end"
            )
        sunlight_intervals.append((t_start.timestamp(), t_end.timestamp()))

    # Check for overlapping sunlight intervals
    for i in range(len(sunlight_intervals)):
        for j in range(i + 1, len(sunlight_intervals)):
            s1, e1 = sunlight_intervals[i]
            s2, e2 = sunlight_intervals[j]
            if s1 < e2 and s2 < e1:
                raise ValueError(
                    "INVALID_PROJECTION_ASSUMPTIONS: sunlight_schedule contains overlapping intervals"
                )

    # Parse and validate payload intervals
    payload_intervals: list[tuple[float, float, float]] = []
    for i, segment in enumerate(payload_schedule):
        if not isinstance(segment, dict):
            raise ValueError(
                f"INVALID_PROJECTION_ASSUMPTIONS: payload_schedule[{i}] must be a dict"
            )
        for key in ("start", "end", "draw_w"):
            if key not in segment:
                raise ValueError(
                    f"INVALID_PROJECTION_ASSUMPTIONS: payload_schedule[{i}] missing '{key}'"
                )
        t_start = _parse_iso(segment["start"])
        t_end = _parse_iso(segment["end"])
        if t_start >= t_end:
            raise ValueError(
                f"INVALID_PROJECTION_ASSUMPTIONS: payload_schedule[{i}] start >= end"
            )
        draw_raw = segment["draw_w"]
        draw_w = _require_finite(f"payload_schedule[{i}].draw_w", draw_raw)
        if draw_w < 0:
            raise ValueError(
                f"INVALID_PROJECTION_ASSUMPTIONS: payload_schedule[{i}].draw_w must be >= 0"
            )
        payload_intervals.append((t_start.timestamp(), t_end.timestamp(), draw_w))

    # Check for overlapping payload intervals
    for i in range(len(payload_intervals)):
        for j in range(i + 1, len(payload_intervals)):
            s1, e1, _ = payload_intervals[i]
            s2, e2, _ = payload_intervals[j]
            if s1 < e2 and s2 < e1:
                raise ValueError(
                    "INVALID_PROJECTION_ASSUMPTIONS: payload_schedule contains overlapping intervals"
                )

    # Starting SOC from the latest history sample
    starting_soc: float = float(latest_sample["battery_soc_percent"])
    # Solar current from the latest sample for generation estimate
    # (held constant — simplifying prototype assumption)
    solar_current_a: float = float(latest_sample["solar_array_current_a"])
    # Nominal bus voltage for power conversion (~28 V typical)
    _BUS_VOLTAGE_NOMINAL = 28.0

    # Latest timestamp as projection origin
    latest_ts_str = latest_sample["timestamp"]
    t0 = _parse_iso(latest_ts_str).timestamp()

    SOC_BREACH = 25.0

    hourly_entries: list[dict[str, Any]] = []
    soc = starting_soc

    for h in range(1, 25):
        # Midpoint of the hour for schedule lookup (simplifying prototype assumption)
        t_mid = t0 + (h - 0.5) * 3600.0
        t_end_h = t0 + h * 3600.0
        forecast_ts = datetime.fromtimestamp(t_end_h, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        # Determine whether the midpoint falls in a sunlight window
        in_sunlight = any(s <= t_mid <= e for (s, e) in sunlight_intervals)

        # Generation: solar array current × efficiency × bus voltage, in W
        generation_w = (
            solar_current_a * _BUS_VOLTAGE_NOMINAL * efficiency
            if in_sunlight
            else 0.0
        )

        # Payload draw at the midpoint (first matching segment wins)
        payload_draw_w = 0.0
        for (ps, pe, pw) in payload_intervals:
            if ps <= t_mid <= pe:
                payload_draw_w = pw
                break

        total_load_w = base_load_w + payload_draw_w
        net_power_w = generation_w - total_load_w

        # Energy delta over 1 hour (Wh)
        delta_wh = net_power_w * 1.0  # 1 hour

        # SOC change
        soc += (delta_wh / capacity_wh) * 100.0
        soc = max(0.0, min(100.0, soc))

        projected_breach = soc < SOC_BREACH

        hourly_entries.append(
            {
                "hour_offset": h,
                "forecast_timestamp": forecast_ts,
                "projected_soc_percent": round(soc, 4),
                "projected_breach": projected_breach,
            }
        )

    assumption_note = (
        f"Physics-based energy balance using: battery_capacity_wh={capacity_wh}, "
        f"base_spacecraft_load_w={base_load_w}, "
        f"power_conversion_efficiency={efficiency}, "
        f"sunlight_schedule={len(sunlight_intervals)} window(s), "
        f"payload_schedule={len(payload_intervals)} segment(s). "
        "Hourly midpoint sampling and constant latest solar current are "
        "simplifying prototype assumptions. "
        "This is NOT an AI output. Results depend entirely on the supplied assumptions."
    )

    return {
        "method": "physics-based-energy-balance-synthetic",
        "not_ai_output": True,
        "assumption_note": assumption_note,
        "window_complete": len(hourly_entries) == 24,
        "hourly_projection": hourly_entries,
    }, None


# ---------------------------------------------------------------------------
# Core prediction service
# ---------------------------------------------------------------------------

def run_prediction(
    samples: list[dict[str, Any]],
    *,
    scenario_id: str = "PUBLIC-DEMO-HISTORY-001",
    projection_assumptions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the four-layer power-risk prediction over *samples*.

    Parameters
    ----------
    samples:
        Telemetry history dicts.  Must contain the four numeric fields plus
        timestamp for feature extraction.  Input dicts are never mutated.
    scenario_id:
        Passed through to the response for identification.
    projection_assumptions:
        Optional dict with the five L2 physical assumptions.  Omitted when
        None or when any required field is absent.

    Returns
    -------
    dict
        Full four-layer response including the safety envelope.

    Raises
    ------
    ValueError
        If samples are empty, contain invalid fields, timestamps, cadence,
        bool values, numeric strings, NaN/inf values, or invalid projection
        assumptions.
    """
    # Work on an independent copy so the caller's list is never mutated
    samples = copy.deepcopy(samples)

    pipeline = get_pipeline()
    base = dict(SAFETY_ENVELOPE)
    base["scenario_id"] = scenario_id
    base["query_timestamp"] = _query_timestamp()
    base["model_claim"] = (
        "logistic-regression-synthetic-not-trained-on-real-spacecraft"
    )
    base["model_version"] = MODEL_VERSION

    # ------------------------------------------------------------------
    # Validate all supplied samples before any mode decision
    # ------------------------------------------------------------------
    _validate_samples(samples)

    n_samples = len(samples)

    # ------------------------------------------------------------------
    # Degraded mode: fewer than 72 valid samples
    # AI inference is not performed; no model required.
    # ------------------------------------------------------------------
    if n_samples < 72:
        # L3 on latest sample — do NOT suppress exceptions; let unexpected
        # policy failures propagate to the endpoint's 500 handler so the
        # safety envelope is still guaranteed (contract Section 10/12).
        findings: list[dict[str, str]] = apply_power_thresholds(samples[-1])

        # L2 projection runs independently of AI inference; a valid short
        # history can still produce a deterministic projection if complete
        # assumptions are supplied.  No model is required for this path.
        deterministic_projection: dict[str, Any] | None = None
        projection_omitted_reason: str | None = None
        if projection_assumptions:
            deterministic_projection, projection_omitted_reason = _compute_l2_projection(
                samples[-1], projection_assumptions
            )
        else:
            projection_omitted_reason = "Required physical assumptions not supplied"

        window_hours = round(n_samples * 5.0 / 60.0, 4)
        base.update(
            {
                "status": "degraded",
                "degraded_reason": (
                    "Fewer than 72 samples available; AI inference requires "
                    "exactly 72. ai_prediction is null."
                ),
                "ai_prediction": None,
                "breach_probability": None,
                "deterministic_projection": deterministic_projection,
                "projection_omitted_reason": projection_omitted_reason,
                "safety_threshold_findings": findings,
                "advisory": _derive_advisory(None, findings, None),
                "audit": {
                    "features_used": 0,
                    "window_complete": False,
                    "samples_used": n_samples,
                    "window_hours": window_hours,
                    "action_mode": "simulation-only",
                },
            }
        )
        return base

    # ------------------------------------------------------------------
    # Normal mode: use the latest 72 samples
    # ------------------------------------------------------------------
    window = samples[-72:]  # never mutates input (already deepcopied)

    # L3 threshold findings from latest sample
    findings = apply_power_thresholds(window[-1])

    # Feature extraction
    feat_dict = extract_features(window)
    raw_vector = [feat_dict[k] for k in FEATURE_ORDER]

    # ML inference
    feature_matrix = [raw_vector]
    proba_matrix = pipeline.predict_proba(feature_matrix)
    breach_probability = float(proba_matrix[0][1])

    if not math.isfinite(breach_probability) or not (0.0 <= breach_probability <= 1.0):
        raise ValueError(
            "Model returned invalid probability"
        )

    predicted_class = int(1 if breach_probability >= 0.5 else 0)

    all_contributions, top_contributions = _build_contributions(raw_vector, pipeline)

    ai_prediction = {
        "label": "power_constraint_breach_within_24h",
        "predicted_class": predicted_class,
        "breach_probability": round(breach_probability, 6),
        "probability_note": _PROBABILITY_NOTE,
        "top_contributions": top_contributions,
        "all_contributions": all_contributions,
    }

    # L2 deterministic projection
    deterministic_projection = None
    projection_omitted_reason: str | None = None
    if projection_assumptions:
        deterministic_projection, projection_omitted_reason = _compute_l2_projection(
            window[-1], projection_assumptions
        )
    else:
        projection_omitted_reason = "Required physical assumptions not supplied"

    # L4 advisory
    advisory = _derive_advisory(breach_probability, findings, ai_prediction)

    base.update(
        {
            "status": "ok",
            "ai_prediction": ai_prediction,
            "deterministic_projection": deterministic_projection,
            "projection_omitted_reason": projection_omitted_reason,
            "safety_threshold_findings": findings,
            "advisory": advisory,
            "audit": {
                "features_used": 12,
                "window_complete": True,
                "samples_used": 72,
                "window_hours": 6.0,
                "action_mode": "simulation-only",
            },
        }
    )
    return base
