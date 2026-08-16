"""Deterministic synthetic power-scenario generator — prototype feasibility proof.

Exported function
-----------------
generate_prototype_scenarios() -> list[dict]
    Returns exactly nine deterministic positive scenarios (no randomness):

        SOC_ONLY  × {early, middle, late}
        VOLTAGE_ONLY × {early, middle, late}
        BOTH         × {early, middle, late}

    All nine scenarios share the same physical model and differ only in their
    calibrated initial conditions and degradation parameters.

Physics models
--------------
Battery energy balance (SOC update)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
At each 5-minute step the net energy exchange is:

    P_in   = solar_array_current_a × V_bus × η          [W]
    P_out  = BASE_LOAD_W + payload_power_draw_w          [W]
    ΔE     = (P_in − P_out) × Δt_h                      [Wh],  Δt_h = 5/60 h
    ΔSOC   = ΔE / BATTERY_CAPACITY_WH × 100             [%]

SOC evolves exclusively through this update rule.  No individual sample is
overwritten to force a breach.

Bus-voltage model (regulator + load-droop + independent degradation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    I_load  = (BASE_LOAD_W + payload_power_draw_w) / V_NOM   [A] (first-order)
    V_droop = R_DROOP × I_load                               [V]
    V_deg   = step_index × V_DEG_RATE_PER_STEP               [V] (monotonic)
    V       = V_NOM − V_droop − V_deg                        [V]

V_DEG_RATE_PER_STEP represents progressive resistance growth (e.g. contact
oxidation) and is the independent term that allows VOLTAGE_ONLY scenarios.
SOC and bus voltage are coupled only through P_in = solar_a × V × η; the
decoupling is achieved by choosing solar_a so that net power remains
non-negative for VOLTAGE_ONLY cases.

Parameter calibration (no random rejection loops)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SOC-breach scenarios
  Drain per step d(s) depends on V(s) (which decreases under voltage
  degradation).  The total SOC consumed over T steps is computed by summing
  d(s) for s = 0 … T−1, where each d(s) is computed directly from the physics
  equations.  The starting SOC is then:

      SOC_initial = 25.0 + Σ_{s=0}^{T−1} d(s)

  This is a closed-form cumulative sum — not a search or rejection loop.

Voltage-breach scenarios (VOLTAGE_ONLY)
  The degradation rate is derived analytically from the desired breach step T:

      V_DEG_RATE = (V_initial_droop − 26.0) / T

  Solar current is set to 3.00 A (> P_out / (26.0 × η)) so P_in > P_out at
  every step, guaranteeing SOC only increases throughout the trajectory.

BOTH scenarios
  Voltage is degraded to breach at the same absolute step T as SOC.
  Solar current uses the SOC-breach value.  SOC_initial is computed via the
  cumulative-sum formula (same as SOC-breach) using the voltage trajectory that
  is active.

Safety metadata
---------------
Every scenario carries a top-level metadata block:

    data_source:       SYNTHETIC
    prototype_status:  NOT_FLIGHT_QUALIFIED
    command_authority: NONE
    policy_decision:   PERMITTED_FOR_SIMULATION_ONLY

Schema compliance
-----------------
The contract in docs/power-risk-contract.md names these eight raw telemetry
fields per sample:
    timestamp, solar_array_current_a, payload_power_draw_w, bus_voltage_v,
    battery_soc_percent, command_activity, communications_status,
    image_utility_score

command_activity and communications_status use the same values as
data/mock/telemetry.json.  image_utility_score and communications_status are
raw telemetry fields that are not power-risk ML features.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Physical constants — documented for reproducibility
# ---------------------------------------------------------------------------

BATTERY_CAPACITY_WH: float = 100.0      # Wh — total usable capacity
BASE_LOAD_W: float = 20.0               # W  — constant spacecraft base load
NOMINAL_BUS_VOLTAGE: float = 28.0       # V  — regulated bus nominal
SOLAR_EFFICIENCY: float = 0.90          # η  — solar-to-bus conversion efficiency
R_DROOP: float = 0.05                   # Ω  — load-droop resistance (always present)
DT_H: float = 5.0 / 60.0               # h  — one 5-minute step in hours
DT_S: int = 300                         # s  — one 5-minute step in seconds

# Steady payload draw used across all scenarios
PAYLOAD_DRAW_W: float = 50.0            # W

# Total bus load (base + payload)
P_OUT: float = BASE_LOAD_W + PAYLOAD_DRAW_W  # 70.0 W

# Load current for droop calculation (first-order, using nominal voltage)
I_LOAD: float = P_OUT / NOMINAL_BUS_VOLTAGE  # 2.5 A

# Bus voltage at step 0 with no degradation:  V_NOM − R_DROOP × I_LOAD
V_INITIAL_DROOP: float = NOMINAL_BUS_VOLTAGE - R_DROOP * I_LOAD  # 27.875 V

# ---------------------------------------------------------------------------
# Solar current settings
# ---------------------------------------------------------------------------
#
# SOC_ONLY scenarios: solar current chosen so P_in is slightly below P_out,
# producing a steady net drain of ≈ 0.119 %/step at nominal voltage.
#
#   P_in  = 2.735 × 27.875 × 0.9 ≈ 68.57 W
#   P_out = 70.0 W  →  net ≈ −1.43 W

SOLAR_CURRENT_SOC_BREACH: float = 2.735   # A — steady net drain for SOC_ONLY

# VOLTAGE_ONLY scenarios: solar current > P_out / (V_min × η) so P_in > P_out
# at every voltage point, guaranteeing SOC only increases.
#
#   Required: solar_a × 26.0 × 0.9 ≥ 70.0  →  solar_a ≥ 2.992 A
#   Use 3.00 A for a safe margin.

SOLAR_CURRENT_VOLTAGE_ONLY: float = 3.00  # A — ensures non-negative SOC change

# BOTH scenarios: solar current chosen so the cumulative SOC drain over the
# longest target window (T = 312 steps, late-BOTH) keeps soc_initial ≤ 95 %.
#
#   Required total_drain(late) ≤ 70.001 % with soc_at_breach = 24.999
#   Solved numerically: solar_a = 2.80 A satisfies the constraint for all
#   three timing bands:
#     early  (T=144): soc0 ≈ 50.3 %
#     middle (T=216): soc0 ≈ 63.0 %
#     late   (T=312): soc0 ≈ 80.0 %

SOLAR_CURRENT_BOTH: float = 2.80          # A — balanced drain for BOTH scenarios

# Starting SOC for VOLTAGE_ONLY scenarios
SOC_VOLTAGE_ONLY_START: float = 70.0       # % — comfortable margin above 25 %

# ---------------------------------------------------------------------------
# Timing targets (future-window step indices, 0-based)
# ---------------------------------------------------------------------------
#
# The convention below places the intended breach squarely inside the band:
#
#   band     valid hour range   target hour   target future-step
#   early    [2 h, 8 h]        6 h           72
#   middle   (8 h, 16 h]       12 h          144
#   late     (16 h, 23 h]      20 h          240
#
# Absolute step = HISTORY_STEPS (72) + future_step.

TIMING_TARGETS: dict[str, int] = {
    "early":  72,
    "middle": 144,
    "late":   240,
}

TIMING_BANDS: dict[str, tuple[float, float]] = {
    "early":  (2.0,  8.0),    # hour range [lo, hi]
    "middle": (8.0, 16.0),
    "late":  (16.0, 23.0),
}

HISTORY_STEPS: int = 72
FUTURE_STEPS: int = 288

# ---------------------------------------------------------------------------
# Safety metadata block
# ---------------------------------------------------------------------------

SAFETY_METADATA: dict[str, str] = {
    "data_source":       "SYNTHETIC",
    "prototype_status":  "NOT_FLIGHT_QUALIFIED",
    "command_authority": "NONE",
    "policy_decision":   "PERMITTED_FOR_SIMULATION_ONLY",
}

# ---------------------------------------------------------------------------
# Valid command_activity and communications_status values
# (taken verbatim from data/mock/telemetry.json)
# ---------------------------------------------------------------------------

_CMD_CYCLE: list[str] = [
    "NOMINAL_ATTITUDE_HOLD",
    "PAYLOAD_WARMUP",
    "PAYLOAD_IMAGING_BURST",
    "PAYLOAD_IMAGING_BURST",
    "NOMINAL_ATTITUDE_HOLD",
    "NOMINAL_ATTITUDE_HOLD",
]

_COMMS_CYCLE: list[str] = [
    "GROUND_CONTACT",
    "GROUND_CONTACT",
    "PASS_ENDING",
    "NO_CONTACT_WINDOW",
    "NO_CONTACT_WINDOW",
    "GROUND_CONTACT",
]

# Reference epoch — fixed so all timestamps are deterministic
EPOCH: datetime = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Internal physics helpers
# ---------------------------------------------------------------------------

def _voltage_at_step(step_index: int, payload_w: float, v_deg_rate: float) -> float:
    """
    Bus-voltage model.

        I_load  = (BASE_LOAD_W + payload_w) / V_NOM
        V_droop = R_DROOP × I_load
        V_deg   = step_index × v_deg_rate
        V       = V_NOM − V_droop − V_deg
    """
    i_load  = (BASE_LOAD_W + payload_w) / NOMINAL_BUS_VOLTAGE
    v_droop = R_DROOP * i_load
    v_deg   = step_index * v_deg_rate
    return NOMINAL_BUS_VOLTAGE - v_droop - v_deg


def _soc_next(soc: float, solar_a: float, payload_w: float, v: float) -> float:
    """
    Battery energy-balance state update.

        P_in   = solar_a × V × η
        P_out  = BASE_LOAD_W + payload_w
        ΔE     = (P_in − P_out) × Δt_h
        SOC    += ΔE / BATTERY_CAPACITY_WH × 100
    """
    p_in    = solar_a * v * SOLAR_EFFICIENCY
    p_out   = BASE_LOAD_W + payload_w
    delta_e = (p_in - p_out) * DT_H
    return soc + delta_e / BATTERY_CAPACITY_WH * 100.0


def _calibrate_v_deg_rate(timing: str, payload_w: float) -> float:
    """
    Closed-form voltage degradation rate so V first drops below 26.0 V at
    absolute step T (VOLTAGE_ONLY scenarios).

        T = HISTORY_STEPS + TIMING_TARGETS[timing]
        V_DEG_RATE = (V_initial_droop − 26.0) / T

    This places V exactly at 26.0 at step T; due to floating-point precision
    the stored value is always 26.0 (not strictly < 26.0).  The first strict
    breach therefore occurs at step T+1, which is 5 minutes later.
    For VOLTAGE_ONLY, SOC is far above 25 % so the breach type is always
    BUS_VOLTAGE_LOW regardless of which step the first strict crossing occurs.
    """
    i_load     = (BASE_LOAD_W + payload_w) / NOMINAL_BUS_VOLTAGE
    v_step0    = NOMINAL_BUS_VOLTAGE - R_DROOP * i_load
    abs_target = HISTORY_STEPS + TIMING_TARGETS[timing]
    return (v_step0 - 26.0) / abs_target


def _calibrate_v_deg_rate_both(timing: str, payload_w: float) -> float:
    """
    Voltage degradation rate for BOTH scenarios.

    Guarantees that:
      - V(T−1) ≥ 26.0 V  (no voltage breach before the target step)
      - V(T)   < 26.0 V  (first voltage breach exactly at the target step)

    by using the midpoint between the two boundary values:

        v_deg_lower = (V_initial_droop − 26.0) / T          [first breach at T+1]
        v_deg_upper = (V_initial_droop − 26.0) / (T − 1)    [first breach at T]
        V_DEG_RATE_BOTH = (v_deg_lower + v_deg_upper) / 2

    The midpoint is strictly inside (v_deg_lower, v_deg_upper], ensuring
    V(T-1) ≥ 26.0 and V(T) < 26.0 without relying on floating-point exact equality.
    """
    i_load  = (BASE_LOAD_W + payload_w) / NOMINAL_BUS_VOLTAGE
    v_step0 = NOMINAL_BUS_VOLTAGE - R_DROOP * i_load
    T       = HISTORY_STEPS + TIMING_TARGETS[timing]
    v_deg_lower = (v_step0 - 26.0) / T
    v_deg_upper = (v_step0 - 26.0) / (T - 1)
    return (v_deg_lower + v_deg_upper) / 2.0


def _calibrate_soc_initial(
    timing: str,
    solar_a: float,
    payload_w: float,
    v_deg_rate: float,
    soc_at_breach: float = 25.0,
) -> float:
    """
    Compute the starting SOC so that SOC reaches *soc_at_breach* at absolute step T.

    Drain per step d(s) = −ΔSOC(s) is a function of V(s), which changes as the
    voltage degrades.  The total SOC consumed over T steps is:

        total_drain = Σ_{s=0}^{T−1} d(s)

    where d(s) is computed directly from the physics equations, step by step.
    This is a deterministic closed-form summation — not a search or rejection loop.

        SOC_initial = soc_at_breach + total_drain

    For SOC_ONLY, soc_at_breach = 25.0 (default) so the trajectory arrives
    exactly at the threshold.  For BOTH scenarios, soc_at_breach = 24.999 so
    the SOC is guaranteed to be strictly below the threshold at step T regardless
    of floating-point precision in the summation.
    """
    abs_target = HISTORY_STEPS + TIMING_TARGETS[timing]
    total_drain = 0.0
    dummy_soc   = 0.0  # SOC offset does not affect the drain per step
    for s in range(abs_target):
        v = _voltage_at_step(s, payload_w, v_deg_rate)
        next_soc = _soc_next(dummy_soc, solar_a, payload_w, v)
        step_drain = dummy_soc - next_soc   # positive when draining
        total_drain += step_drain
    return soc_at_breach + total_drain


# ---------------------------------------------------------------------------
# Trajectory and sample builders
# ---------------------------------------------------------------------------

def _timestamp(step: int) -> str:
    """ISO-8601 UTC timestamp for step index (5-min cadence from EPOCH)."""
    return (EPOCH + timedelta(seconds=step * DT_S)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cmd(step: int) -> str:
    return _CMD_CYCLE[step % len(_CMD_CYCLE)]


def _comms(step: int) -> str:
    return _COMMS_CYCLE[step % len(_COMMS_CYCLE)]


def _image_score(step: int) -> float:
    """Deterministic image utility score in [0.40, 0.95], cycling sinusoidally."""
    return round(0.675 + 0.275 * math.sin(step * math.pi / 18.0), 3)


def _make_sample(
    step: int,
    solar_a: float,
    payload_w: float,
    v: float,
    soc: float,
) -> dict[str, Any]:
    """Build one raw telemetry sample with all eight required fields."""
    return {
        "timestamp":             _timestamp(step),
        "solar_array_current_a": round(solar_a, 4),
        "payload_power_draw_w":  round(payload_w, 4),
        "bus_voltage_v":         round(v, 6),
        "battery_soc_percent":   round(soc, 6),
        "command_activity":      _cmd(step),
        "communications_status": _comms(step),
        "image_utility_score":   _image_score(step),
    }


def _generate_trajectory(
    soc_initial: float,
    solar_a: float,
    payload_w: float,
    v_deg_rate: float,
    total_steps: int,
) -> list[tuple[float, float]]:
    """
    Generate one continuous SOC/voltage trajectory of *total_steps* steps.

    Returns a list of (soc, voltage) tuples indexed from step 0.

    SOC is capped at 100.0 % (full battery — physically, excess solar power is
    shed once the battery is fully charged).  This cap is not used to manufacture
    eligibility; it is applied uniformly to every step and every scenario type.

    No state resets, no per-sample overwrites, no eligibility-targeting clamps.
    """
    trajectory: list[tuple[float, float]] = []
    soc = min(soc_initial, 100.0)   # initial state cannot exceed full battery
    for i in range(total_steps):
        v = _voltage_at_step(i, payload_w, v_deg_rate)
        trajectory.append((soc, v))
        # Clamp to [0, 100]: a depleted battery cannot discharge below 0 %,
        # and a full battery sheds excess generation.
        soc = max(0.0, min(_soc_next(soc, solar_a, payload_w, v), 100.0))
    return trajectory


def _find_first_breach(
    trajectory: list[tuple[float, float]],
    start: int,
) -> tuple[int | None, str | None]:
    """
    Scan trajectory[start:] for the first step where SOC < 25 or V < 26.

    Returns (absolute_step, breach_type) or (None, None).
    breach_type is "SOC_ONLY", "VOLTAGE_ONLY", or "BOTH".
    """
    for i in range(start, len(trajectory)):
        soc, v = trajectory[i]
        soc_b = soc < 25.0
        v_b   = v   < 26.0
        if soc_b and v_b:
            return i, "BOTH"
        if soc_b:
            return i, "SOC_ONLY"
        if v_b:
            return i, "VOLTAGE_ONLY"
    return None, None


def _breach_detail(trajectory: list[tuple[float, float]]) -> dict[str, Any]:
    """
    Determine breach_detail from the generated trajectory (not from the template).
    """
    breach_step, breach_type = _find_first_breach(trajectory, start=HISTORY_STEPS)
    if breach_step is None:
        return {
            "occurs":      False,
            "breach_type": None,
            "timing_band": None,
            "hour_offset": None,
        }
    future_step = breach_step - HISTORY_STEPS       # 0-based within future window
    hour_offset = future_step / 12.0                 # 12 steps per hour
    band: str | None = None
    for band_name, (lo, hi) in TIMING_BANDS.items():
        if lo <= hour_offset <= hi:
            band = band_name
            break
    return {
        "occurs":      True,
        "breach_type": breach_type,
        "timing_band": band,
        "hour_offset": round(hour_offset, 4),
    }


def _build_scenario(
    scenario_id: str,
    trajectory: list[tuple[float, float]],
    solar_a: float,
    payload_w: float,
) -> dict[str, Any]:
    """Assemble the final scenario dict from a pre-generated trajectory."""
    total = HISTORY_STEPS + FUTURE_STEPS
    assert len(trajectory) == total

    samples: list[dict[str, Any]] = []
    for i in range(total):
        soc, v = trajectory[i]
        samples.append(_make_sample(i, solar_a, payload_w, v, soc))

    history = samples[:HISTORY_STEPS]
    future  = samples[HISTORY_STEPS:]

    detail = _breach_detail(trajectory)
    label  = 1 if detail["occurs"] else 0

    return {
        "metadata":                             dict(SAFETY_METADATA),
        "scenario_id":                          scenario_id,
        "power_constraint_breach_within_24h":   label,
        "breach_detail":                        detail,
        "history":                              history,
        "future":                               future,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_prototype_scenarios() -> list[dict[str, Any]]:
    """
    Return exactly nine deterministic positive-breach scenarios.

    Coverage
    --------
    Breach type   × Timing band
    SOC_ONLY      × early, middle, late
    VOLTAGE_ONLY  × early, middle, late
    BOTH          × early, middle, late

    Physical guarantees per scenario
    ---------------------------------
    - 72 history samples (6 h at 5-min cadence) and 288 future samples (24 h).
    - History forms a single continuous trajectory with the future window.
    - The first future sample is exactly 5 minutes after the final history sample.
    - No active SOC or voltage breach during the history window.
    - Maximum one-step SOC change ≤ 2 percentage points.
    - Maximum one-step voltage change ≤ 0.30 V.
    - SOC evolves only through the energy-balance equation.
    - No individual samples are overwritten; no trajectory snap-backs.
    """
    scenarios: list[dict[str, Any]] = []
    total_steps = HISTORY_STEPS + FUTURE_STEPS

    # -----------------------------------------------------------------------
    # SOC_ONLY: battery drains below 25 %; bus voltage stays above 26 V.
    # v_deg_rate = 0 → voltage follows pure load droop, never degrading.
    # -----------------------------------------------------------------------
    for idx, timing in enumerate(("early", "middle", "late"), start=1):
        sid     = f"PROTO-SOC-{idx:03d}-{timing.upper()}"
        solar_a = SOLAR_CURRENT_SOC_BREACH
        v_deg   = 0.0  # no voltage degradation → VOLTAGE_ONLY cannot trigger
        soc0    = _calibrate_soc_initial(timing, solar_a, PAYLOAD_DRAW_W, v_deg)
        traj    = _generate_trajectory(soc0, solar_a, PAYLOAD_DRAW_W, v_deg, total_steps)
        scenarios.append(_build_scenario(sid, traj, solar_a, PAYLOAD_DRAW_W))

    # -----------------------------------------------------------------------
    # VOLTAGE_ONLY: bus voltage degrades below 26 V; SOC stays above 25 %.
    # Solar current > P_out / (26.0 × η) guarantees P_in > P_out at all steps,
    # so SOC can only increase throughout.
    # -----------------------------------------------------------------------
    for idx, timing in enumerate(("early", "middle", "late"), start=1):
        sid     = f"PROTO-VOLT-{idx:03d}-{timing.upper()}"
        solar_a = SOLAR_CURRENT_VOLTAGE_ONLY
        v_deg   = _calibrate_v_deg_rate(timing, PAYLOAD_DRAW_W)
        traj    = _generate_trajectory(
            SOC_VOLTAGE_ONLY_START, solar_a, PAYLOAD_DRAW_W, v_deg, total_steps
        )
        scenarios.append(_build_scenario(sid, traj, solar_a, PAYLOAD_DRAW_W))

    # -----------------------------------------------------------------------
    # BOTH: SOC and voltage breach simultaneously at the target step T.
    #
    # _calibrate_v_deg_rate_both() ensures V(T-1) ≥ 26.0 and V(T) < 26.0.
    # _calibrate_soc_initial() with this v_deg ensures SOC(T) < 25.0.
    # Therefore both thresholds are first crossed at the same step T → BOTH.
    # -----------------------------------------------------------------------
    for idx, timing in enumerate(("early", "middle", "late"), start=1):
        sid     = f"PROTO-BOTH-{idx:03d}-{timing.upper()}"
        solar_a = SOLAR_CURRENT_BOTH
        v_deg   = _calibrate_v_deg_rate_both(timing, PAYLOAD_DRAW_W)
        # soc_at_breach=24.999 guarantees SOC is strictly below 25.0 at step T
        # even if floating-point arithmetic produces exactly 25.0 with 25.0 target.
        soc0    = _calibrate_soc_initial(timing, solar_a, PAYLOAD_DRAW_W, v_deg,
                                         soc_at_breach=24.999)
        traj    = _generate_trajectory(soc0, solar_a, PAYLOAD_DRAW_W, v_deg, total_steps)
        scenarios.append(_build_scenario(sid, traj, solar_a, PAYLOAD_DRAW_W))

    return scenarios


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import os

    out_dir  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scenarios")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "power_scenarios.json")

    result = generate_prototype_scenarios()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"Wrote {len(result)} scenarios → {os.path.abspath(out_path)}")
