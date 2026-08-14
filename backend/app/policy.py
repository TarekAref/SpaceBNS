"""Shared deterministic power-threshold policy for SpaceBNS.

This module is the single authoritative source for the four instantaneous
power-safety threshold checks.  It is called by both the assessment endpoint
and future prediction endpoints so that thresholds, finding codes, and evidence
strings cannot diverge between callers.

The function is deliberately pure:
- no file I/O;
- no network access;
- no global state mutation;
- no AI or model calls;
- the input dict is never modified.
"""

from __future__ import annotations

from typing import Any


def apply_power_thresholds(sample: dict[str, Any]) -> list[dict[str, str]]:
    """Return a list of threshold findings for one telemetry sample.

    Findings are appended in the fixed canonical order:
      1. BUS_VOLTAGE_LOW
      2. BATTERY_SOC_LOW
      3. PAYLOAD_LOAD_HIGH
      4. IMAGE_UTILITY_LOW

    Args:
        sample: A telemetry sample dict containing at minimum the four keys
            ``bus_voltage_v``, ``battery_soc_percent``,
            ``payload_power_draw_w``, and ``image_utility_score``.
            The dict is not modified.

    Returns:
        A list of finding dicts, each with ``"code"`` and ``"evidence"`` keys.
        The list is empty when no threshold is breached.
    """
    findings: list[dict[str, str]] = []

    if sample["bus_voltage_v"] < 26.0:
        findings.append(
            {
                "code": "BUS_VOLTAGE_LOW",
                "evidence": "bus_voltage_v below public demo threshold",
            }
        )
    if sample["battery_soc_percent"] < 25.0:
        findings.append(
            {
                "code": "BATTERY_SOC_LOW",
                "evidence": "battery_soc_percent below public demo threshold",
            }
        )
    if sample["payload_power_draw_w"] > 100.0:
        findings.append(
            {
                "code": "PAYLOAD_LOAD_HIGH",
                "evidence": "payload_power_draw_w above public demo threshold",
            }
        )
    if sample["image_utility_score"] < 0.30:
        findings.append(
            {
                "code": "IMAGE_UTILITY_LOW",
                "evidence": "edge-reported image utility below public demo threshold",
            }
        )

    return findings
