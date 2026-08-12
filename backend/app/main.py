"""Public-safe FastAPI scaffold for the SpaceBNS challenge prototype."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

APP_VERSION = "0.1.0"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MOCK_TELEMETRY_PATH = REPOSITORY_ROOT / "data" / "mock" / "telemetry.json"

app = FastAPI(
    title="SpaceBNS Public MVP API",
    version=APP_VERSION,
    description=(
        "Synthetic telemetry and deterministic baseline services for the "
        "IBM August Challenge prototype. Not for flight use."
    ),
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def load_mock_telemetry() -> list[dict[str, Any]]:
    """Load explicitly synthetic telemetry committed for public demonstration."""

    with MOCK_TELEMETRY_PATH.open(encoding="utf-8") as telemetry_file:
        payload = json.load(telemetry_file)

    if not isinstance(payload, list) or not payload:
        raise ValueError("Mock telemetry must be a non-empty JSON array")

    return payload


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
    findings: list[dict[str, str]] = []

    if latest["bus_voltage_v"] < 26.0:
        findings.append(
            {
                "code": "BUS_VOLTAGE_LOW",
                "evidence": "bus_voltage_v below public demo threshold",
            }
        )
    if latest["battery_soc_percent"] < 25.0:
        findings.append(
            {
                "code": "BATTERY_SOC_LOW",
                "evidence": "battery_soc_percent below public demo threshold",
            }
        )
    if latest["payload_power_draw_w"] > 100.0:
        findings.append(
            {
                "code": "PAYLOAD_LOAD_HIGH",
                "evidence": "payload_power_draw_w above public demo threshold",
            }
        )
    if latest["image_utility_score"] < 0.30:
        findings.append(
            {
                "code": "IMAGE_UTILITY_LOW",
                "evidence": "edge-reported image utility below public demo threshold",
            }
        )

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

