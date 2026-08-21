"""Public-safe FastAPI scaffold for the SpaceBNS challenge prototype."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from backend.app.policy import apply_power_thresholds
from backend.app.safety import SAFETY_ENVELOPE, register_exception_handlers
import backend.app.prediction_service as _svc

APP_VERSION = "0.1.0"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MOCK_TELEMETRY_PATH = REPOSITORY_ROOT / "data" / "mock" / "telemetry.json"
MOCK_HISTORY_PATH = REPOSITORY_ROOT / "data" / "mock" / "history.json"
MODEL_PATH = str(
    REPOSITORY_ROOT / "data" / "models" / "power_risk_classifier.joblib"
)

app = FastAPI(
    title="SpaceBNS Public MVP API",
    version=APP_VERSION,
    description=(
        "Synthetic telemetry and deterministic baseline services for the "
        "IBM August Challenge prototype. Not for flight use."
    ),
)

# ---------------------------------------------------------------------------
# Register global exception handlers (safety envelope on all error paths)
# ---------------------------------------------------------------------------
register_exception_handlers(app)

# ---------------------------------------------------------------------------
# CORS — GET preserved; POST added for the prediction endpoint
# ---------------------------------------------------------------------------
allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model load at startup
# ---------------------------------------------------------------------------
_svc.load_pipeline(MODEL_PATH)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TelemetrySample(BaseModel):
    """One raw telemetry sample (contract Section 5 — all 8 raw telemetry fields).

    All eight fields are required in every sample.  The four power-related
    numeric fields are used by ML feature extraction.  The three additional
    display/status fields (``command_activity``, ``communications_status``,
    ``image_utility_score``) are passed through to the policy layer but are
    NEVER fed into the 12-feature ML vector.
    Extra fields beyond these eight are also accepted (e.g. scenario metadata).
    """

    model_config = {"extra": "allow"}

    # --- Required power fields (used by ML feature extraction) ---
    timestamp: str
    battery_soc_percent: float
    bus_voltage_v: float
    solar_array_current_a: float
    payload_power_draw_w: float

    # --- Required raw telemetry fields (NOT ML features) ---
    command_activity: str
    communications_status: str
    image_utility_score: float

    @field_validator(
        "battery_soc_percent",
        "bus_voltage_v",
        "solar_array_current_a",
        "payload_power_draw_w",
        mode="before",
    )
    @classmethod
    def _reject_non_numeric(cls, v: Any) -> Any:
        """Reject boolean and string values for numeric telemetry fields.

        JSON ``true``/``false`` and numeric strings (e.g. ``"28.5"``) are not
        valid numeric telemetry and must produce a 422 rather than being
        silently coerced.
        """
        if isinstance(v, bool):
            raise ValueError("boolean values are not valid for numeric telemetry fields")
        if isinstance(v, str):
            raise ValueError("string values are not valid for numeric telemetry fields")
        return v

    @field_validator("command_activity", "communications_status", mode="before")
    @classmethod
    def _reject_non_string_status(cls, v: Any) -> Any:
        """Reject non-string values for string telemetry fields.

        JSON numbers, booleans, and null are not valid string telemetry and
        must produce a 422 rather than being silently coerced.
        """
        if not isinstance(v, str):
            raise ValueError("only string values are valid for this telemetry field")
        return v

    @field_validator("image_utility_score", mode="before")
    @classmethod
    def _reject_non_numeric_score(cls, v: Any) -> Any:
        """Reject bool, strings, NaN and infinity for image_utility_score."""
        import math as _math
        if isinstance(v, bool):
            raise ValueError("boolean values are not valid for image_utility_score")
        if isinstance(v, str):
            raise ValueError("string values are not valid for image_utility_score")
        if not isinstance(v, (int, float)):
            raise ValueError("image_utility_score must be numeric")
        if not _math.isfinite(float(v)):
            raise ValueError("image_utility_score must be finite")
        return v


class PredictRequest(BaseModel):
    """POST /api/v1/power-risk/predict request body."""

    samples: list[TelemetrySample]
    projection_assumptions: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_mock_telemetry() -> list[dict[str, Any]]:
    """Load explicitly synthetic telemetry committed for public demonstration."""

    with MOCK_TELEMETRY_PATH.open(encoding="utf-8") as telemetry_file:
        payload = json.load(telemetry_file)

    if not isinstance(payload, list) or not payload:
        raise ValueError("Mock telemetry must be a non-empty JSON array")

    return payload


def _load_mock_history() -> list[dict[str, Any]]:
    """Load the 72-sample public mock history.  Raises HTTPException 503 on failure."""
    try:
        with MOCK_HISTORY_PATH.open(encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:  # noqa: BLE001
        err = dict(SAFETY_ENVELOPE)
        err["error"] = "HISTORY_UNAVAILABLE"
        raise HTTPException(status_code=503, detail="HISTORY_UNAVAILABLE")

    samples = doc.get("samples") if isinstance(doc, dict) else doc
    if not isinstance(samples, list) or not samples:
        raise HTTPException(status_code=503, detail="HISTORY_UNAVAILABLE")
    return samples


def _require_pipeline() -> None:
    """Raise 503 if the model pipeline is not loaded."""
    if _svc.get_pipeline() is None:
        raise HTTPException(status_code=503, detail="MODEL_NOT_LOADED")


# ---------------------------------------------------------------------------
# Existing endpoints (unchanged)
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Return a deployment probe with the prototype's safety mode."""

    return {
        "status": "ok",
        "service": "spacebns-api",
        "version": APP_VERSION,
        "data_mode": "synthetic-only",
        "action_mode": "simulation-only",
    }


@app.get("/api/v1/mock/telemetry")
def mock_telemetry() -> dict[str, Any]:
    """Return synthetic housekeeping telemetry for the public dashboard."""

    samples = load_mock_telemetry()
    return {
        "source": "synthetic-public-demonstration",
        "count": len(samples),
        "samples": samples,
    }


@app.get("/api/v1/mock/assessment")
def mock_assessment() -> dict[str, Any]:
    """Run a transparent deterministic baseline over the latest mock sample.

    This is deliberately not presented as trained AI or flight FDIR. It gives
    the frontend a stable contract while the challenge AI modules are built.
    """

    latest = load_mock_telemetry()[-1]
    findings = apply_power_thresholds(latest)

    risk_level = "high" if len(findings) >= 3 else "advisory"
    recommendation = (
        "DEFER_LOW_PRIORITY_FUTURE_IMAGING"
        if risk_level == "high"
        else "CONTINUE_MONITORING"
    )

    return {
        "scenario_id": "PUBLIC-DEMO-POWER-IMAGE-001",
        "timestamp": latest["timestamp"],
        "risk_level": risk_level,
        "findings": findings,
        "diagnostic_hypothesis": (
            "Recent payload imaging activity is temporally associated with the "
            "elevated load; additional evidence is required for root-cause confirmation."
        ),
        "recommendation": recommendation,
        "policy_decision": "PERMITTED_FOR_SIMULATION_ONLY",
        "command_authority": "NONE",
        "model_claim": "deterministic-public-baseline-not-trained-ai",
    }


# ---------------------------------------------------------------------------
# New prediction endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/power-risk/predict")
def power_risk_predict(request: PredictRequest) -> dict[str, Any]:
    """AI-assisted power-risk prediction from caller-supplied telemetry history.

    Returns the four-layer response (L1 AI + L2 projection + L3 thresholds +
    L4 advisory).  All outputs are advisory only; command_authority is NONE.
    """
    # Validate non-empty before any other check so that empty-window returns
    # 422 even when the model is absent.
    if not request.samples:
        raise HTTPException(status_code=422, detail="EMPTY_WINDOW")

    # Convert Pydantic models to plain dicts; do not mutate originals
    raw_samples: list[dict[str, Any]] = [
        s.model_dump() for s in request.samples
    ]

    # Validate telemetry before checking model availability so that:
    # - invalid inputs always return 422 (not 503)
    # - valid short windows return degraded (not 503) without a loaded model
    try:
        _svc._validate_samples(raw_samples)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="INVALID_TELEMETRY") from exc

    # Only require a loaded model when we actually need AI inference (≥ 72 samples)
    if len(raw_samples) >= 72:
        _require_pipeline()

    try:
        result = _svc.run_prediction(
            raw_samples,
            scenario_id="CALLER-SUPPLIED",
            projection_assumptions=request.projection_assumptions,
        )
    except ValueError as exc:  # noqa: BLE001
        # Validation failures from run_prediction (invalid telemetry or
        # projection assumptions) — return 422 with a fixed public code.
        err_msg = str(exc)
        if "INVALID_PROJECTION_ASSUMPTIONS" in err_msg:
            raise HTTPException(
                status_code=422, detail="INVALID_PROJECTION_ASSUMPTIONS"
            ) from exc
        raise HTTPException(status_code=422, detail="INVALID_TELEMETRY") from exc
    except Exception:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail="FEATURE_EXTRACTION_FAILED"
        )

    return result


@app.get("/api/v1/mock/power-risk-prediction")
def mock_power_risk_prediction() -> dict[str, Any]:
    """Demonstration endpoint: load mock history and run the prediction service.

    Calls the same prediction service as the POST endpoint.  Response
    structure is identical.  All outputs are advisory only; command_authority
    is NONE.
    """
    _require_pipeline()
    samples = _load_mock_history()

    try:
        result = _svc.run_prediction(
            samples,
            scenario_id="SYNTH-DEMO-PUBLIC-001",
        )
    except Exception:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail="FEATURE_EXTRACTION_FAILED"
        )

    return result
