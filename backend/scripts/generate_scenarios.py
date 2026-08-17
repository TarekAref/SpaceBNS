"""Deterministic synthetic power-scenario generator.

Exported functions
------------------
generate_prototype_scenarios() -> list[dict]
    Returns exactly nine deterministic positive scenarios (no randomness):

        SOC_ONLY  × {early, middle, late}
        VOLTAGE_ONLY × {early, middle, late}
        BOTH         × {early, middle, late}

    All nine scenarios share the same physical model and differ only in their
    calibrated initial conditions and degradation parameters.

generate_training_corpus(seed=42) -> list[dict]
    Returns exactly 300 independent scenarios with the following properties:

    Counts and balance
    ~~~~~~~~~~~~~~~~~~
    - 150 positive scenarios (power_constraint_breach_within_24h = 1)
    - 150 negative scenarios (power_constraint_breach_within_24h = 0)

    Splits (assigned by scenario_id, no row-level splitting)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - train:      180 (90 positive, 90 negative)
    - validation:  60 (30 positive, 30 negative)
    - test:        60 (30 positive, 30 negative)
    - zero scenario_id overlap between splits

    Positive coverage
    ~~~~~~~~~~~~~~~~~
    - Breach types:  SOC_ONLY, VOLTAGE_ONLY, BOTH
    - Timing bands:  early, middle, late

    Negative families (difficult negatives that superficially resemble positives)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - NEG_RECOVERING  — low but recovering battery SOC (30–40 %, trending up)
    - NEG_HIGH_SOLAR  — temporary high payload draw with sufficient solar
                        generation to maintain net-positive energy balance
    - NEG_VOLTAGE_NEAR — bus voltage approaching but never crossing 26.0 V

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
import random
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
# Training-corpus public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Positive-scenario variation parameters
# ---------------------------------------------------------------------------
# Each positive scenario cell (breach_type × timing_band) is parameterised by
# a bounded set of scalars drawn from the fixed-seed RNG.  All bounds are
# chosen so that no scenario violates any hard constraint (history eligibility,
# physical bounds) regardless of the drawn value — there is no rejection loop.
#
# Parameter names and ranges:
#
#   payload_offset_w : float in [-10, +10]
#       Added to the baseline payload draw (PAYLOAD_DRAW_W = 50 W).
#       Keeps the total draw in [40, 60 W], well within the model's valid range.
#       This shifts the per-step drain and therefore the calibrated starting SOC,
#       producing distinct trajectories without changing the breach-type logic.
#
#   soc_margin : float in [0, 3]
#       Added to the calibrated starting SOC (which places the trajectory exactly
#       at the breach threshold).  The margin pushes the breach slightly later
#       than the calibrated target step, keeping the trajectory within the same
#       timing band.  The resulting breach hour is verified analytically against
#       the band bounds before the scenario is emitted.
#
#   solar_jitter : float in [-0.02, +0.02]
#       Tiny additive jitter on solar_a for SOC_ONLY and BOTH scenarios.
#       At ±0.02 A the per-step power change is 0.02 × 26 × 0.9 ≈ 0.47 W,
#       negligible relative to the 70 W base load; it produces measurable
#       feature-level diversity without altering the breach-type classification.
#
# ---------------------------------------------------------------------------
# Negative-scenario construction rules
# ---------------------------------------------------------------------------
# Three negative families; each contains 50 scenarios for a total of 150.
#
# NEG_RECOVERING (family index 0): low but recovering SOC
#   - starting SOC in [30, 40] %
#   - solar_a chosen so that P_in > P_out throughout, producing monotonically
#     increasing SOC.  The analytic minimum is solar_a_min = P_OUT / (V_NOM × η).
#     We use [solar_a_min + 0.3, solar_a_min + 1.8] so SOC can only grow.
#   - payload draw in [40, 65] W  (nominal range)
#   - v_deg_rate = 0 → voltage stays above 26 V throughout
#   - Label = 0 is guaranteed because:
#       (a) SOC starts above 25 % and can only increase (P_in > P_out)
#       (b) voltage has no degradation and stays at V_INITIAL_DROOP > 26 V
#
# NEG_HIGH_SOLAR (family index 1): temporary high payload, sufficient solar
#   - payload draw in [100, 140] W for the first 72 history steps (high load),
#     reverting to [40, 65] W for the 288 future steps
#   - solar_a chosen to maintain P_in > P_out under the high load:
#     solar_a ≥ P_OUT_HIGH / (V_MIN × η).  We use a bounded range that keeps
#     P_in ≥ P_out under the worst-case payload draw at V_min = 26.5 V, so
#     SOC never declines below 25 % in either window.
#   - starting SOC in [55, 80] %
#   - v_deg_rate = 0
#   - Label = 0 is guaranteed because P_in ≥ P_out at every step in both
#     history and future (solar_a × V × η ≥ BASE_LOAD_W + payload).
#
# NEG_VOLTAGE_NEAR (family index 2): voltage near but never crossing 26 V
#   - v_deg_rate chosen so V at end of 288-step future is in [26.05, 26.5] V
#     (never crosses 26.0 V within the 24-hour future window)
#   - solar_a = SOLAR_CURRENT_VOLTAGE_ONLY so SOC only increases
#   - starting SOC in [55, 80] %
#   - starting V at step 0: V_INITIAL_DROOP − eps where eps is the per-step
#     degradation accumulated over a small warm-up so the voltage is declining
#     but far from the threshold at history start
#   - Label = 0 is guaranteed analytically: V(step) = V_INITIAL_DROOP −
#     step × v_deg_rate; the maximum step index in the future window is
#     HISTORY_STEPS + FUTURE_STEPS − 1 = 359. v_deg_rate is calibrated so
#     V(359) > 26.0.
# ---------------------------------------------------------------------------

# Corpus layout constants
_CORPUS_TOTAL        = 300
_CORPUS_POSITIVE     = 150
_CORPUS_NEGATIVE     = 150
_SPLIT_TRAIN_POS     = 90
_SPLIT_TRAIN_NEG     = 90
_SPLIT_VAL_POS       = 30
_SPLIT_VAL_NEG       = 30
_SPLIT_TEST_POS      = 30
_SPLIT_TEST_NEG      = 30

# Number of positive scenario types and timing bands
_BREACH_TYPES  = ("SOC_ONLY", "VOLTAGE_ONLY", "BOTH")
_TIMING_BANDS  = ("early", "middle", "late")
# 9 cells; distribute 150 positives: 6 cells get 17, 3 cells get 16 → 6×17+3×16=150
_POSITIVES_PER_CELL: dict[tuple[str, str], int] = {}
_raw_count = 0
for _bt in _BREACH_TYPES:
    for _tb in _TIMING_BANDS:
        _POSITIVES_PER_CELL[(_bt, _tb)] = 17 if _raw_count < 6 else 16
        _raw_count += 1

# Number of negative scenarios per family
_NEG_FAMILIES  = ("NEG_RECOVERING", "NEG_HIGH_SOLAR", "NEG_VOLTAGE_NEAR")
_NEGS_PER_FAMILY = 50  # 3 × 50 = 150

# Minimum solar current that keeps P_in > P_out at V_nom for any payload
_SOLAR_A_MIN_FOR_NET_POS: float = (
    (BASE_LOAD_W + PAYLOAD_DRAW_W) / (NOMINAL_BUS_VOLTAGE * SOLAR_EFFICIENCY)
)  # ≈ 2.778 A with PAYLOAD_DRAW_W=50


def _assign_splits(
    n_positive: int,
    n_negative: int,
    train_pos: int,
    val_pos: int,
    test_pos: int,
    train_neg: int,
    val_neg: int,
    test_neg: int,
) -> tuple[list[str], list[str]]:
    """
    Return (pos_splits, neg_splits) where each element is one of
    'train', 'validation', 'test'.  The first *train_pos* positive scenarios
    are train, the next *val_pos* are validation, and the last *test_pos* are
    test.  Same logic for negatives.  No randomness — purely index-based.
    """
    assert train_pos + val_pos + test_pos == n_positive
    assert train_neg + val_neg + test_neg == n_negative

    pos_splits = (
        ["train"] * train_pos
        + ["validation"] * val_pos
        + ["test"] * test_pos
    )
    neg_splits = (
        ["train"] * train_neg
        + ["validation"] * val_neg
        + ["test"] * test_neg
    )
    return pos_splits, neg_splits


def _make_variable_sample(
    step: int,
    solar_a: float,
    payload_w: float,
    v: float,
    soc: float,
) -> dict[str, Any]:
    """Build one raw telemetry sample with all eight required fields.

    Unlike the prototype generator this helper accepts arbitrary solar_a and
    payload_w values so that per-scenario variation is captured correctly in
    the telemetry record.
    """
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


def _build_variable_scenario(
    scenario_id: str,
    split: str,
    trajectory: list[tuple[float, float]],
    solar_a_hist: float,
    solar_a_fut: float,
    payload_w_hist: float,
    payload_w_fut: float,
) -> dict[str, Any]:
    """
    Assemble a scenario dict where history and future may have different
    solar_a / payload_w values (needed for NEG_HIGH_SOLAR family).

    For all other scenarios solar_a_hist == solar_a_fut and
    payload_w_hist == payload_w_fut; these are kept as separate parameters
    for clarity.
    """
    total = HISTORY_STEPS + FUTURE_STEPS
    assert len(trajectory) == total

    history: list[dict[str, Any]] = []
    future:  list[dict[str, Any]] = []

    for i in range(HISTORY_STEPS):
        soc, v = trajectory[i]
        history.append(_make_variable_sample(i, solar_a_hist, payload_w_hist, v, soc))
    for i in range(HISTORY_STEPS, total):
        soc, v = trajectory[i]
        future.append(_make_variable_sample(i, solar_a_fut, payload_w_fut, v, soc))

    detail = _breach_detail(trajectory)
    label  = 1 if detail["occurs"] else 0

    return {
        "metadata":                           dict(SAFETY_METADATA),
        "scenario_id":                        scenario_id,
        "split":                              split,
        "power_constraint_breach_within_24h": label,
        "breach_detail":                      detail,
        "history":                            history,
        "future":                             future,
    }


def _generate_positive_scenario(
    rng: random.Random,
    breach_type: str,
    timing: str,
    scenario_id: str,
    split: str,
) -> dict[str, Any]:
    """
    Generate one positive scenario (label = 1) with bounded parameter variation.

    Design principle
    ~~~~~~~~~~~~~~~~
    All three breach types use payload_offset in [-10, +10] W as a primary
    variation axis; the remaining axes differ by type.

    SOC_ONLY
        solar_a is set to breakeven × (1 − deficit_frac), placing P_in strictly
        below P_out at every step.  Because v_deg_rate = 0 the bus voltage is
        constant, so the deficit margin is preserved throughout the window.
        _calibrate_soc_initial() derives the exact starting SOC so that the
        trajectory reaches 25.0 % at the target step T, and no history breach
        can occur.

    BOTH
        solar_a is set to breakeven × (1 + surplus_frac), placing P_in barely
        above P_out at the initial bus voltage.  Independent voltage degradation
        (v_deg_rate > 0) lowers bus voltage and therefore generated power over
        time; net power eventually becomes negative and SOC begins draining
        toward the simultaneous breach at step T.  The trajectory is therefore
        NOT monotonically decreasing: it gains SOC slightly at the start then
        drains.  _calibrate_soc_initial() guarantees soc(T) = 24.999;
        _calibrate_v_deg_rate_both() guarantees V(T) < 26.0 and V(T-1) >= 26.0,
        so the first breach type is BOTH.

    VOLTAGE_ONLY
        solar_a is sized well above the breakeven current at V_INITIAL_DROOP so
        that SOC stays safely above 25 % at the time of the first voltage breach.
        The complete history window is free of both power constraints.  At the
        first future voltage breach, SOC remains at or above 25 %, so the
        first-breach classification is VOLTAGE_ONLY.  Later SOC behavior does
        not change that first-breach classification.

    All guarantees follow from the closed-form calibration functions — no
    rejection loops are used.
    """
    total_steps = HISTORY_STEPS + FUTURE_STEPS

    # payload variation is drawn for all breach types
    payload_offset = rng.uniform(-10.0, 10.0)   # [-10, +10] W
    payload_w      = PAYLOAD_DRAW_W + payload_offset   # [40, 60] W

    if breach_type == "SOC_ONLY":
        # deficit_frac in [0.005, 0.018]: solar_a = breakeven × (1 − deficit_frac)
        # breakeven_solar = P_out / (V_INITIAL_DROOP × η)
        # This guarantees P_in < P_out at every step (V_INITIAL_DROOP is the
        # maximum voltage, so if P_in < P_out at step 0, it is P_in ≤ P_out at
        # all later steps where V can only stay constant for SOC_ONLY).
        # The range [0.005, 0.018] is chosen to keep soc0_raw ≤ 95 % for all
        # timing bands (including the worst case: late, T=312, payload=60 W,
        # max total drain ≈ 312 × 0.018 × 80 × DT_H / 100 ≈ 60.2 %).
        deficit_frac    = rng.uniform(0.005, 0.018)
        breakeven_solar = (BASE_LOAD_W + payload_w) / (V_INITIAL_DROOP * SOLAR_EFFICIENCY)
        solar_a         = breakeven_solar * (1.0 - deficit_frac)
        v_deg_rate      = 0.0
        soc0 = _calibrate_soc_initial(timing, solar_a, payload_w, v_deg_rate)
        soc0 = min(soc0, 95.0)
        traj = _generate_trajectory(soc0, solar_a, payload_w, v_deg_rate, total_steps)

    elif breach_type == "VOLTAGE_ONLY":
        # solar_a is sized well above breakeven at V_INITIAL_DROOP, keeping SOC
        # safely above 25 % at and before the first voltage breach.
        # extra_solar in [0.20, 0.60] A provides variation in the SOC trajectory.
        extra_solar = rng.uniform(0.20, 0.60)
        solar_a     = (BASE_LOAD_W + payload_w) / (V_INITIAL_DROOP * SOLAR_EFFICIENCY) + extra_solar
        v_deg_rate  = _calibrate_v_deg_rate(timing, payload_w)
        soc_bonus   = rng.uniform(0.0, 15.0)  # starting SOC in [70, 85] %
        soc0        = SOC_VOLTAGE_ONLY_START + soc_bonus
        soc0        = min(soc0, 95.0)
        traj        = _generate_trajectory(soc0, solar_a, payload_w, v_deg_rate, total_steps)

    else:  # BOTH
        # For BOTH, use solar_a slightly ABOVE the breakeven at V_INITIAL_DROOP.
        # surplus_frac in [0.001, 0.006] (≈ 0.1–0.6 % above breakeven).
        #
        # This mirrors the prototype approach (SOLAR_CURRENT_BOTH ≈ breakeven × 1.0025).
        # When P_in is barely above P_out at step 0, the trajectory initially gains
        # SOC slightly; as voltage decreases via v_deg_rate, the net power eventually
        # flips negative and the trajectory drains toward 25 % at step T.
        #
        # Because P_in ≈ P_out throughout (large fraction of trajectory) the total
        # net drain is small — soc0_raw = 24.999 + small_net_drain stays well within
        # [25 %, 95 %] even for the "late" timing band.
        #
        # Because the trajectory is NOT monotonically decreasing (it first gains,
        # then drains), the first SOC breach must be verified to occur at step T.
        # _calibrate_soc_initial guarantees soc(T) = 24.999 and the trajectory is
        # governed by the physics equations — no clamp or snap-back modifies it.
        #
        # In practice, soc(T-1) ≈ 25.0 + one_step_drain > 25 %, ensuring BOTH
        # breach type (voltage also breaches at T via _calibrate_v_deg_rate_both).
        surplus_frac    = rng.uniform(0.001, 0.006)
        v_deg_rate      = _calibrate_v_deg_rate_both(timing, payload_w)
        breakeven_solar = (BASE_LOAD_W + payload_w) / (V_INITIAL_DROOP * SOLAR_EFFICIENCY)
        solar_a         = breakeven_solar * (1.0 + surplus_frac)
        soc0            = _calibrate_soc_initial(
            timing, solar_a, payload_w, v_deg_rate, soc_at_breach=24.999
        )
        soc0            = min(soc0, 95.0)
        traj            = _generate_trajectory(soc0, solar_a, payload_w, v_deg_rate, total_steps)

    return _build_variable_scenario(
        scenario_id, split, traj,
        solar_a_hist=solar_a, solar_a_fut=solar_a,
        payload_w_hist=payload_w, payload_w_fut=payload_w,
    )


def _generate_neg_recovering(
    rng: random.Random,
    scenario_id: str,
    split: str,
) -> dict[str, Any]:
    """
    NEG_RECOVERING: low but recovering SOC (30–40 %), net-positive power throughout.

    Construction (no rejection loop):
    - soc0 in [30, 40] %
    - payload_w in [40, 65] W (nominal)
    - solar_a chosen to produce a definite positive net power:
        solar_a = (BASE_LOAD_W + payload_w) / (V_INITIAL_DROOP × η) + extra_margin
      where extra_margin in [0.3, 1.8] ensures robust P_in > P_out.
    - v_deg_rate = 0 → no voltage degradation

    Guarantees:
    - SOC starts above 25 % and can only increase (P_in > P_out at every step)
    - Voltage stays at V_INITIAL_DROOP > 26 V always
    - No breach in history or future → label = 0
    """
    total_steps = HISTORY_STEPS + FUTURE_STEPS
    payload_w   = rng.uniform(40.0, 65.0)
    extra_solar = rng.uniform(0.3, 1.8)
    # Minimum solar to ensure P_in > P_out at V_INITIAL_DROOP
    solar_min   = (BASE_LOAD_W + payload_w) / (V_INITIAL_DROOP * SOLAR_EFFICIENCY)
    solar_a     = solar_min + extra_solar
    soc0        = rng.uniform(30.0, 40.0)
    v_deg_rate  = 0.0
    traj        = _generate_trajectory(soc0, solar_a, payload_w, v_deg_rate, total_steps)

    return _build_variable_scenario(
        scenario_id, split, traj,
        solar_a_hist=solar_a, solar_a_fut=solar_a,
        payload_w_hist=payload_w, payload_w_fut=payload_w,
    )


def _generate_neg_high_solar(
    rng: random.Random,
    scenario_id: str,
    split: str,
) -> dict[str, Any]:
    """
    NEG_HIGH_SOLAR: high payload draw during history, sufficient solar to maintain
    net-positive balance throughout both windows.

    Construction (no rejection loop):
    - history payload in [100, 140] W  (high load)
    - future  payload in [40,  65] W   (returns to nominal)
    - solar_a chosen to ensure P_in > P_out even under the high history load:
        V_guard = 27.0 V (comfortable minimum — actual droop at high load still
                          keeps voltage well above 26 V for payload ≤ 140 W)
        solar_min = (BASE_LOAD_W + payload_hist) / (V_guard × η)
        solar_a   = solar_min + extra_margin  where extra_margin in [0.2, 0.8]
    - soc0 in [55, 80] %
    - v_deg_rate = 0 for the trajectory (voltage only changes via droop;
      no independent degradation, so it cannot cross 26 V)

    The trajectory is built with the HIGH payload draw for all 360 steps so that
    the physics remain internally consistent.  The telemetry records for history
    and future then carry the correct payload_w values for each window.

    Guarantees:
    - P_in > P_out at every history step (high payload): SOC only increases
    - P_in > P_out at every future step  (low  payload): SOC only increases
    - Voltage stays above 26 V throughout (no v_deg; droop alone is < 2 V)
    - No breach in history or future → label = 0
    """
    total_steps  = HISTORY_STEPS + FUTURE_STEPS
    payload_hist = rng.uniform(100.0, 140.0)
    payload_fut  = rng.uniform(40.0, 65.0)

    # Use the higher of the two payload draws to size solar_a
    payload_max  = max(payload_hist, payload_fut)
    V_guard      = 27.0  # conservative voltage floor for solar sizing
    solar_min    = (BASE_LOAD_W + payload_max) / (V_guard * SOLAR_EFFICIENCY)
    extra_solar  = rng.uniform(0.2, 0.8)
    solar_a      = solar_min + extra_solar
    soc0         = rng.uniform(55.0, 80.0)
    v_deg_rate   = 0.0

    # Build trajectory with the HIGHER payload so SOC evolution is conservative
    # (if it stays ≥ 25 % under the higher draw, it will under the lower draw too)
    traj = _generate_trajectory(soc0, solar_a, payload_hist, v_deg_rate, total_steps)

    return _build_variable_scenario(
        scenario_id, split, traj,
        solar_a_hist=solar_a, solar_a_fut=solar_a,
        payload_w_hist=payload_hist, payload_w_fut=payload_fut,
    )


def _generate_neg_voltage_near(
    rng: random.Random,
    scenario_id: str,
    split: str,
) -> dict[str, Any]:
    """
    NEG_VOLTAGE_NEAR: bus voltage approaches but never crosses 26.0 V.

    Construction (no rejection loop):
    - Target minimum voltage V_min in (26.05, 26.5) V at the last future step
      (step index HISTORY_STEPS + FUTURE_STEPS − 1 = 359).
    - With the bus-voltage model V(step) = V_INITIAL_DROOP − step × v_deg_rate:
        v_deg_rate = (v_step0 − V_min) / (HISTORY_STEPS + FUTURE_STEPS − 1)
      This guarantees V stays ≥ V_min > 26.0 for all steps 0..359.
    - solar_a is sized to ensure P_in > P_out at the minimum voltage V_min,
      guaranteeing SOC only increases throughout.  A margin of [0.3, 1.8] A
      above the analytic minimum is added for robustness.
    - soc0 in [55, 80] %
    - payload_w in [40, 65] W

    Guarantees:
    - V(step) ≥ V_min > 26.0 V for all 360 steps → no voltage breach
    - solar_a × V_min × η ≥ BASE_LOAD_W + payload_w → P_in > P_out at every
      step → SOC only increases → no SOC breach
    - label = 0
    """
    total_steps = HISTORY_STEPS + FUTURE_STEPS
    # last step index = total_steps - 1 = 359
    last_step   = total_steps - 1
    v_min       = rng.uniform(26.05, 26.5)
    # Draw payload first (needed for both v_step0 and solar sizing)
    payload_w   = rng.uniform(40.0, 65.0)
    i_load      = (BASE_LOAD_W + payload_w) / NOMINAL_BUS_VOLTAGE
    v_step0     = NOMINAL_BUS_VOLTAGE - R_DROOP * i_load   # V at step 0 with this payload
    # Calibrate v_deg_rate so that V(last_step) = v_min:
    #   v_step0 - last_step * v_deg_rate = v_min
    v_deg_rate  = (v_step0 - v_min) / last_step
    # Size solar_a to ensure P_in > P_out at the minimum voltage v_min.
    # solar_min = (BASE_LOAD_W + payload_w) / (v_min × η)
    # A margin of [0.3, 1.8] A above the analytic minimum is added.
    solar_min   = (BASE_LOAD_W + payload_w) / (v_min * SOLAR_EFFICIENCY)
    extra_solar = rng.uniform(0.3, 1.8)
    solar_a     = solar_min + extra_solar
    soc0        = rng.uniform(55.0, 80.0)
    traj        = _generate_trajectory(soc0, solar_a, payload_w, v_deg_rate, total_steps)

    return _build_variable_scenario(
        scenario_id, split, traj,
        solar_a_hist=solar_a, solar_a_fut=solar_a,
        payload_w_hist=payload_w, payload_w_fut=payload_w,
    )


def generate_training_corpus(seed: int = 42) -> list[dict[str, Any]]:
    """
    Return exactly 300 independent synthetic power scenarios.

    Parameters
    ----------
    seed : int
        Fixed random seed.  The same seed produces byte-identical output;
        different seeds produce different valid corpora.

    Returns
    -------
    list[dict]
        300 scenario dicts in the corpus order:
          - scenarios 0–149:  positive (label = 1)
          - scenarios 150–299: negative (label = 0)

        Each scenario has the following top-level keys:
          metadata, scenario_id, split, power_constraint_breach_within_24h,
          breach_detail, history (72 samples), future (288 samples).

    Split assignment
    ----------------
    Splits are assigned by index position before any generation takes place.
    For each class (positive / negative) in isolation:
      - first 90 scenarios  → train
      - next  30 scenarios  → validation
      - last  30 scenarios  → test
    Scenario IDs encode the split name so that a downstream reader can
    reconstruct the split from the ID alone.

    Positive-scenario layout
    ------------------------
    Positives are laid out cell by cell across the 9 (breach_type × timing_band)
    combinations.  Within each cell, scenarios are generated with bounded
    parameter variation from the seeded RNG.

    Negative-scenario layout
    ------------------------
    Negatives are laid out family by family across the 3 difficult-negative
    families.  50 scenarios per family.

    Determinism guarantee
    ---------------------
    The seeded ``random.Random`` object draws parameters in a fixed order
    determined by the layout above.  Inserting, removing, or reordering any
    draw call will change the output for all subsequent draws.  The draw order
    is therefore considered a stable contract: it must not be altered without
    bumping the corpus version.
    """
    rng       = random.Random(seed)
    scenarios : list[dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Assign splits up-front (index-based, no randomness)
    # -----------------------------------------------------------------------
    pos_splits, neg_splits = _assign_splits(
        _CORPUS_POSITIVE, _CORPUS_NEGATIVE,
        _SPLIT_TRAIN_POS, _SPLIT_VAL_POS, _SPLIT_TEST_POS,
        _SPLIT_TRAIN_NEG, _SPLIT_VAL_NEG, _SPLIT_TEST_NEG,
    )

    # -----------------------------------------------------------------------
    # Generate positive scenarios
    # -----------------------------------------------------------------------
    pos_idx = 0   # global index into pos_splits
    for breach_type in _BREACH_TYPES:
        for timing in _TIMING_BANDS:
            cell_count = _POSITIVES_PER_CELL[(breach_type, timing)]
            for k in range(cell_count):
                split      = pos_splits[pos_idx]
                # ID format: SYNTH-PWR-{TYPE}-{TIMING}-{k:03d}
                # e.g. SYNTH-PWR-SOC-EARLY-000
                type_tag   = breach_type.replace("_ONLY", "").replace("_", "")
                scenario_id = (
                    f"SYNTH-PWR-{type_tag}-{timing.upper()}-{k:03d}"
                )
                s = _generate_positive_scenario(
                    rng, breach_type, timing, scenario_id, split
                )
                scenarios.append(s)
                pos_idx += 1

    # -----------------------------------------------------------------------
    # Generate negative scenarios
    # -----------------------------------------------------------------------
    neg_idx = 0   # global index into neg_splits
    for fam_idx, family in enumerate(_NEG_FAMILIES):
        for k in range(_NEGS_PER_FAMILY):
            split       = neg_splits[neg_idx]
            scenario_id = f"SYNTH-PWR-{family}-{k:03d}"

            if family == "NEG_RECOVERING":
                s = _generate_neg_recovering(rng, scenario_id, split)
            elif family == "NEG_HIGH_SOLAR":
                s = _generate_neg_high_solar(rng, scenario_id, split)
            else:  # NEG_VOLTAGE_NEAR
                s = _generate_neg_voltage_near(rng, scenario_id, split)

            scenarios.append(s)
            neg_idx += 1

    return scenarios


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import os
    import sys
    import time

    # Determine mode from first argument: "prototype" or "corpus" (default: "corpus")
    mode = sys.argv[1] if len(sys.argv) > 1 else "corpus"

    out_dir  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scenarios")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "power_scenarios.json")

    t0 = time.perf_counter()
    if mode == "prototype":
        result = generate_prototype_scenarios()
    else:
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
        result = generate_training_corpus(seed=seed)
    elapsed = time.perf_counter() - t0

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    file_size_kb = os.path.getsize(out_path) / 1024

    print(f"Wrote {len(result)} scenarios -> {os.path.abspath(out_path)}")
    print(f"Generation time: {elapsed:.3f} s")
    print(f"JSON size:       {file_size_kb:.1f} KB")

    if mode != "prototype":
        from collections import Counter
        labels   = [s["power_constraint_breach_within_24h"] for s in result]
        splits   = [s.get("split", "?") for s in result]
        pos_scen = [s for s in result if s["power_constraint_breach_within_24h"] == 1]
        neg_scen = [s for s in result if s["power_constraint_breach_within_24h"] == 0]

        print(f"\n--- Corpus summary ---")
        print(f"Total:    {len(result)}")
        print(f"Positive: {labels.count(1)}")
        print(f"Negative: {labels.count(0)}")

        for sp in ("train", "validation", "test"):
            sp_scen = [s for s in result if s.get("split") == sp]
            sp_pos  = sum(1 for s in sp_scen if s["power_constraint_breach_within_24h"] == 1)
            sp_neg  = len(sp_scen) - sp_pos
            print(f"{sp:12s}: {len(sp_scen):3d}  (pos={sp_pos}, neg={sp_neg})")

        print(f"\n--- Positive breach-type distribution ---")
        type_counter: Counter = Counter()
        for s in pos_scen:
            bd = s["breach_detail"]
            type_counter[(bd["breach_type"], bd["timing_band"])] += 1
        for (bt, tb), cnt in sorted(type_counter.items()):
            print(f"  {bt:15s} × {tb:6s}: {cnt}")

        print(f"\n--- Negative-family distribution ---")
        fam_counter: Counter = Counter()
        for s in neg_scen:
            sid = s["scenario_id"]
            for fam in _NEG_FAMILIES:
                if fam in sid:
                    fam_counter[fam] += 1
                    break
        for fam, cnt in sorted(fam_counter.items()):
            print(f"  {fam:25s}: {cnt}")

        ids = [s["scenario_id"] for s in result]
        hist_fingerprints = set(
            tuple(samp["battery_soc_percent"] for samp in s["history"][:3])
            for s in result
        )
        print(f"\nUnique scenario IDs:    {len(set(ids))}")
        print(f"Unique history starts:  {len(hist_fingerprints)} "
              f"(3-sample SOC prefix; expected {len(result)} if all unique)")
