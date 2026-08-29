"""Pure feature-extraction functions for SpaceBNS power-risk prediction.

Exported functions
------------------
ols_slope(values, step_minutes) -> float
    Ordinary least squares slope of a uniformly sampled series, in units per hour.

extract_features(samples) -> dict[str, float]
    Extract the 12 ML features from exactly 72 chronologically ordered telemetry
    samples at five-minute cadence.  Raises ValueError for any invalid input.

Design constraints (from docs/power-risk-contract.md section 6)
---------------------------------------------------------------
- Pure functions: no I/O, no side effects, no global state mutation.
- Standard library only: no ML, statistics, or numerical libraries.
- Input dicts are never mutated.
- Slopes are OLS against elapsed time in hours, so slope units are per hour.
- Only four telemetry fields are used:
      battery_soc_percent, bus_voltage_v,
      solar_array_current_a, payload_power_draw_w
- Forbidden inputs are never accessed:
      scenario_id, split, labels, breach_detail,
      communications_status, command_activity, image_utility_score
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_SAMPLE_COUNT: int = 72
STEP_MINUTES: float = 5.0
STEP_SECONDS: int = 300          # 5 * 60
HIGH_DRAW_THRESHOLD_W: float = 100.0

_REQUIRED_FIELDS: tuple[str, ...] = (
    "timestamp",
    "battery_soc_percent",
    "bus_voltage_v",
    "solar_array_current_a",
    "payload_power_draw_w",
)

_NUMERIC_FIELDS: tuple[str, ...] = (
    "battery_soc_percent",
    "bus_voltage_v",
    "solar_array_current_a",
    "payload_power_draw_w",
)

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_timestamp(ts: Any, index: int) -> datetime:
    """Parse an ISO-8601 UTC timestamp string; raise ValueError on failure."""
    if not isinstance(ts, str):
        raise ValueError(
            f"sample[{index}]['timestamp']: expected a string, "
            f"got {type(ts).__name__}"
        )
    try:
        return datetime.strptime(ts, _TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(
            f"sample[{index}]['timestamp']: invalid format {ts!r}; "
            f"expected YYYY-MM-DDTHH:MM:SSZ"
        )


def _validate_samples(samples: list[dict[str, Any]]) -> None:
    """Raise ValueError with a deterministic message for any invalid input."""

    # --- sample count ---
    if len(samples) != REQUIRED_SAMPLE_COUNT:
        raise ValueError(
            f"extract_features requires exactly {REQUIRED_SAMPLE_COUNT} samples; "
            f"got {len(samples)}"
        )

    prev_ts: datetime | None = None

    for i, sample in enumerate(samples):
        # --- required fields present ---
        for field in _REQUIRED_FIELDS:
            if field not in sample:
                raise ValueError(
                    f"sample[{i}]: missing required field '{field}'"
                )

        # --- timestamp validity and ordering ---
        ts = _parse_timestamp(sample["timestamp"], i)

        if prev_ts is not None:
            delta_s = int((ts - prev_ts).total_seconds())
            if delta_s <= 0:
                raise ValueError(
                    f"sample[{i}]['timestamp']: timestamps must be strictly "
                    f"ascending; sample[{i}] ({sample['timestamp']}) is not "
                    f"after sample[{i - 1}]"
                )
            if delta_s != STEP_SECONDS:
                raise ValueError(
                    f"sample[{i}]['timestamp']: expected {STEP_SECONDS}s cadence "
                    f"({STEP_MINUTES}-minute spacing); "
                    f"got {delta_s}s between sample[{i - 1}] and sample[{i}]"
                )

        prev_ts = ts

        # --- numeric fields: reject booleans, non-numeric types, NaN, infinity ---
        for field in _NUMERIC_FIELDS:
            val = sample[field]
            if isinstance(val, bool):
                raise ValueError(
                    f"sample[{i}]['{field}']: boolean is not a valid numeric value"
                )
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"sample[{i}]['{field}']: expected a number, "
                    f"got {type(val).__name__} ({val!r})"
                )
            if math.isnan(val):
                raise ValueError(
                    f"sample[{i}]['{field}']: NaN is not allowed"
                )
            if math.isinf(val):
                raise ValueError(
                    f"sample[{i}]['{field}']: infinity is not allowed"
                )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ols_slope(values: list[float], step_minutes: float = 5.0) -> float:
    """Compute the OLS slope of a uniformly sampled series.

    Time is measured in hours starting from zero, so the returned slope has
    units of (value-unit) per hour.

    Parameters
    ----------
    values:
        Sequence of observed values, one per equally spaced time step.
    step_minutes:
        Time between consecutive samples in minutes.  Defaults to 5.0.

    Returns
    -------
    float
        OLS slope in units of (value-unit) per hour.  Returns 0.0 for a
        single-element series (no slope defined).

    Notes
    -----
    Standard two-pass formula:
        Sxx = Σ (t_i − t̄)²
        Sxy = Σ (t_i − t̄)(y_i − ȳ)
        slope = Sxy / Sxx
    where t_i = i × (step_minutes / 60) hours.
    """
    n = len(values)
    if n <= 1:
        return 0.0

    step_h = step_minutes / 60.0
    # times in hours: 0, step_h, 2*step_h, ...
    t_mean = (n - 1) * step_h / 2.0
    y_mean = sum(values) / n

    sxx = 0.0
    sxy = 0.0
    for i, y in enumerate(values):
        t = i * step_h
        dt = t - t_mean
        sxx += dt * dt
        sxy += dt * (y - y_mean)

    if sxx == 0.0:
        return 0.0
    return sxy / sxx


def extract_features(samples: list[dict[str, Any]]) -> dict[str, float]:
    """Extract the 12 ML features from a 72-sample telemetry window.

    Parameters
    ----------
    samples:
        Exactly 72 dicts, each containing at minimum:
            timestamp            (ISO-8601 UTC, YYYY-MM-DDTHH:MM:SSZ)
            battery_soc_percent  (float)
            bus_voltage_v        (float)
            solar_array_current_a (float)
            payload_power_draw_w  (float)
        Samples must be in chronological order at exactly 5-minute cadence.
        The input list and its dicts are never mutated.

    Returns
    -------
    dict[str, float]
        Exactly 12 keys:
            soc_latest, soc_mean, soc_min, soc_slope,
            voltage_latest, voltage_min, voltage_slope,
            solar_current_mean, solar_current_slope,
            payload_draw_mean, payload_draw_max, high_draw_fraction

    Raises
    ------
    ValueError
        For any validation failure: wrong sample count, missing fields,
        boolean or non-numeric values, NaN or infinity, invalid timestamps,
        non-ascending timestamps, or incorrect cadence.
    """
    _validate_samples(samples)

    # Extract raw series (read-only; no mutation of input dicts)
    soc    = [float(s["battery_soc_percent"])   for s in samples]
    volt   = [float(s["bus_voltage_v"])          for s in samples]
    solar  = [float(s["solar_array_current_a"])  for s in samples]
    draw   = [float(s["payload_power_draw_w"])   for s in samples]

    n = len(soc)  # == REQUIRED_SAMPLE_COUNT == 72

    # --- SOC features ---
    soc_latest = soc[-1]
    soc_mean   = sum(soc) / n
    soc_min    = min(soc)
    soc_slope  = ols_slope(soc, STEP_MINUTES)

    # --- Voltage features ---
    voltage_latest = volt[-1]
    voltage_min    = min(volt)
    voltage_slope  = ols_slope(volt, STEP_MINUTES)

    # --- Solar current features ---
    solar_current_mean  = sum(solar) / n
    solar_current_slope = ols_slope(solar, STEP_MINUTES)

    # --- Payload draw features ---
    payload_draw_mean = sum(draw) / n
    payload_draw_max  = max(draw)
    high_draw_fraction = sum(1 for d in draw if d > HIGH_DRAW_THRESHOLD_W) / n

    return {
        "soc_latest":           soc_latest,
        "soc_mean":             soc_mean,
        "soc_min":              soc_min,
        "soc_slope":            soc_slope,
        "voltage_latest":       voltage_latest,
        "voltage_min":          voltage_min,
        "voltage_slope":        voltage_slope,
        "solar_current_mean":   solar_current_mean,
        "solar_current_slope":  solar_current_slope,
        "payload_draw_mean":    payload_draw_mean,
        "payload_draw_max":     payload_draw_max,
        "high_draw_fraction":   high_draw_fraction,
    }
