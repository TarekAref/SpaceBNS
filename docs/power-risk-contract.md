# SpaceBNS Power-Risk Prediction — Implementation Contract

**Status:** implementation complete — backend prediction endpoints implemented
and locally tested; not yet deployed or wired to the frontend. Corpus generated
with `generate_training_corpus(seed=42)`, final model fitted on 240 synthetic
scenarios and serialised; all contract validation gates passed on the first and
only held-out test evaluation.

**Prototype status:** NOT_FLIGHT_QUALIFIED

**Advisory only:** this system has no command authority

**Data:** synthetic only — probability estimates apply to the synthetic scenario
distribution and must not be interpreted as operational spacecraft failure
probabilities

---

## 1. Scope and Exclusions

### In scope

- A core reusable prediction service function callable by multiple endpoints.
- A core POST endpoint (`POST /api/v1/power-risk/predict`) that accepts
  telemetry history in its JSON request body.
- A demonstration GET endpoint (`GET /api/v1/mock/power-risk-prediction`) that
  loads `data/mock/history.json` and calls the same prediction service.
- A `StandardScaler + LogisticRegression` classifier pipeline trained on a
  synthetic scenario corpus of 300 independent scenarios.
- An optional deterministic energy projection (L2) included only when all five
  required physical assumptions are supplied by the caller.
- A shared policy function used by both the new prediction endpoint and the
  existing `GET /api/v1/mock/assessment` endpoint to evaluate L3 safety
  threshold findings, preventing threshold duplication and drift.
- A global FastAPI exception handler that ensures the four mandatory safety
  envelope fields are present on all error responses, including unhandled
  exceptions.

### Explicitly excluded

- Granite, RAG, LLM integration, or natural-language explanation.
- Image processing, image utility scoring, or edge-device inference.
- Collision avoidance, attitude control, or any spacecraft command authority.
- Autonomous or automated action of any kind.
- NetworkX or causal graph reasoning.
- Net-power calculations (require battery capacity, total base load, and
  power-conversion efficiency that are not present in the telemetry schema).
- Communications-status or image-utility-score features.
- Linearly extrapolated SOC as a classifier input feature.
- Training or evaluation on real spacecraft telemetry.
- Flight qualification, certification, or compliance claims.
- Modification of any existing endpoint, test, or dataset file.

---

## 2. Existing Behavior (Preserved)

The following endpoints existed in `backend/app/main.py` before this milestone
and remain behaviorally unchanged.

| Endpoint | Notes |
|---|---|
| `GET /health` | Unchanged |
| `GET /api/v1/mock/telemetry` | Serves `data/mock/telemetry.json` (5 samples); unchanged |
| `GET /api/v1/mock/assessment` | Response behavior unchanged. Its internal threshold logic was refactored into `backend/app/policy.py`; the endpoint was updated to call `apply_power_thresholds()` instead of duplicating the four rules inline. |

`data/mock/telemetry.json` contains 5 samples and is the accelerated dashboard
demonstration input. It is not used for AI training and is not the history
source for the prediction endpoint.

The three tests in `backend/tests/test_api.py` are unchanged. All new tests are
in separate files.

---

## 3. Four-Layer Separation

Every response from either prediction endpoint must carry exactly four logically
distinct, labelled layers. These layers must never be merged or presented as a
single unified output.

| Layer | Name | Description |
|---|---|---|
| L1 | AI classifier output | Probability estimated from the synthetic scenario distribution; `null` in degraded mode |
| L2 | Deterministic energy projection | Physics-based hourly curve; present when all five physical assumptions are supplied, including in degraded mode when L1 is unavailable; labelled as not an AI output |
| L3 | Safety threshold findings | Instantaneous deterministic threshold checks via the shared policy function; run independently of L1 |
| L4 | Advisory recommendation | Human-readable advisory derived from L1 and L3; no command authority |

---

## 4. Observation Window

**Prototype design choice:** 72 samples at 5-minute cadence (6 hours).

This window was chosen for the synthetic scenario corpus because it captures
several repeated activity cycles — payload bursts, quiescent periods, and
generation variation — within the constructed scenarios. It is not a claim about
LEO orbital periods and does not apply universally to other mission profiles or
telemetry cadences.

**Minimum for AI inference:** 72 samples. If fewer than 72 valid samples are
available:

- Return `degraded` status in the response.
- Set `ai_prediction` to `null` and `breach_probability` to `null`.
- Do not fabricate, extrapolate, or substitute an AI result.
- L3 threshold findings and the L4 advisory are still computed from the
  available samples.
- L2 deterministic energy projection is still returned when all projection
  assumptions are valid and complete; the unavailability of L1 does not
  suppress L2.

---

## 5. Synthetic Scenario Corpus

### Prototype development stage

`backend/scripts/generate_scenarios.py` exports two public functions:

```
generate_prototype_scenarios() -> list[dict]
generate_training_corpus(seed: int = 42) -> list[dict]
```

`generate_prototype_scenarios()` returns **exactly 9 deterministic
positive-breach scenarios** covering the 3 × 3 breach-type / timing-band
matrix. These are scientific test anchors only and are not used for training.

`generate_training_corpus(seed=42)` returns the balanced 300-scenario corpus
(150 positive, 150 negative) with the pre-assigned `split` field.  This is the
function called by both `train_power_risk_model.py` and
`evaluate_power_risk_model.py`.

### Generation approach

The corpus is generated reproducibly at build/train time by a fixed-seed Python
script. It is not a committed data file. Generated files are gitignored and must
never enter Git history.

| Item | Value |
|---|---|
| Generator script | `backend/scripts/generate_scenarios.py` |
| Public corpus function | `generate_training_corpus(seed=42)` |
| Random seed | `42` (fixed constant; same seed always produces the same 300 scenarios in the same order) |
| Output file | `data/scenarios/power_scenarios.json` (gitignored) |
| Directory name | `data/scenarios/` — avoids the `.bobignore` conflict with `data/training/` |
| Regeneration command | `python backend/scripts/generate_scenarios.py` |
| Determinism guarantee | Same seed always produces the same 300 scenarios in the same order |

### Scenario counts and split

| Split | Scenarios | Label = 1 (breach) | Label = 0 (no breach) |
|---|---|---|---|
| Training | 180 | 90 | 90 |
| Validation (hyperparameter selection) | 60 | 30 | 30 |
| Test (untouched until final evaluation) | 60 | 30 | 30 |
| **Total** | **300** | **150** | **150** |

Split is performed strictly by `scenario_id`. No individual 5-minute sample from
a training scenario may appear in the validation or test splits, and vice versa.
Random row-level splits are prohibited. The generator assigns each `scenario_id`
to exactly one split before generating its samples.

### Scenario object schema

Each raw telemetry sample contains **8 fields**.  Only the approved
power-related inputs from those 8 fields are transformed into the 12 ML
features described in Section 6.  `communications_status` and
`image_utility_score` are raw telemetry and display fields; they are present in
every sample but are **not** ML prediction features.

```json
{
  "metadata": {
    "data_source":       "SYNTHETIC",
    "prototype_status":  "NOT_FLIGHT_QUALIFIED",
    "command_authority": "NONE",
    "policy_decision":   "PERMITTED_FOR_SIMULATION_ONLY"
  },
  "scenario_id":  "SYNTH-PWR-TRAIN-001",
  "split":        "train",
  "power_constraint_breach_within_24h": 1,
  "breach_detail": {
    "occurs":       true,
    "breach_type":  "SOC_ONLY",
    "timing_band":  "late",
    "hour_offset":  18.5
  },
  "history": [
    {
      "timestamp":             "2026-01-01T00:00:00Z",
      "solar_array_current_a": 7.2,
      "payload_power_draw_w":  45.0,
      "bus_voltage_v":         27.4,
      "battery_soc_percent":   58.0,
      "command_activity":      "NOMINAL_ATTITUDE_HOLD",
      "communications_status": "NO_CONTACT_WINDOW",
      "image_utility_score":   0.72
    }
  ],
  "future": []
}
```

`history` contains exactly 72 samples; `future` contains exactly 288 samples.
`communications_status` and `image_utility_score` are present in every sample
as raw telemetry fields.  They are not included in the 12-feature ML input
vector and must not be added to the feature list.

The `breach_type` vocabulary is: `SOC_ONLY`, `VOLTAGE_ONLY`, `BOTH`.

### Label eligibility

A scenario is labelled as a future-breach example (label = 1) **only** when no
power constraint is already active at the prediction start — that is, when the
final sample of the 72-sample history window does not itself trigger
`BATTERY_SOC_LOW` or `BUS_VOLTAGE_LOW`. An already-active breach belongs to the
deterministic L3 detection layer, not to the AI prediction layer.

Any generated scenario whose final history sample triggers either L3 threshold
is excluded from the label-1 class regardless of what its 24-hour future window
contains.

**Label = 1:** `battery_soc_percent` falls below 25.0 % or `bus_voltage_v`
falls below 26.0 V within the 24-hour synthetic future window, and no
constraint is active at the prediction start.

**Label = 0:** neither condition occurs within 24 hours, and no constraint is
active at the prediction start.

These thresholds are identical to those used in the existing L3 rules in
`mock_assessment()` lines 96 and 89 respectively.

### Corpus variation axes

The generator must cover at minimum:

- Starting `battery_soc_percent`: low-but-not-breaching (28–45 %), medium
  (45–70 %), high (70–95 %).
- Payload duty cycle: sustained high draw (>100 W for >50 % of window),
  intermittent bursts, nominal low draw (<60 W throughout).
- Solar generation rate: declining (eclipse onset), stable, recovering.
- `command_activity` sequence: imaging-burst-heavy, attitude-hold-only, mixed.
- Breach timing for label-1 scenarios: early (hour 2–8), mid (hour 8–16), late
  (hour 16–23).
- **Difficult negative cases (label = 0)** — the negative class must include
  scenarios that superficially resemble positives:
  - Low but recovering `battery_soc_percent` (30–40 %, trending upward);
  - Temporary high payload load with sufficient solar generation to maintain
    net-positive energy balance;
  - `bus_voltage_v` approaching but not crossing 26.0 V within 24 hours.

### Public mock history file

The demonstration GET endpoint requires a 72-sample history. A separate static
file serves this purpose.

| Item | Value |
|---|---|
| Path | `data/mock/history.json` |
| Sample count | Exactly 72 samples (6 hours at 5-minute cadence) |
| Purpose | Demonstration input for `GET /api/v1/mock/power-risk-prediction` only |
| Content constraint | No active L3 breach at the final sample |
| Committed to repo | Yes — it is a hand-authored static file, not a generated artefact |
| Relationship to existing file | Entirely separate from `data/mock/telemetry.json`; that file is not modified |

---

## 6. Exact Feature List (12 features)

All features are derived from the observation window. The classifier receives
one feature vector per query. Feature extraction is implemented as a pure
function with no I/O side effects.

Each raw telemetry sample carries 8 fields (see Section 5 schema).  Only the
power-related fields — `battery_soc_percent`, `bus_voltage_v`,
`solar_array_current_a`, and `payload_power_draw_w` — are transformed into ML
features.  `communications_status` and `image_utility_score` remain raw
telemetry and display fields; they are **not** ML prediction features and must
not be added to the list below.  The approved feature list is frozen at 12
items.

Explicitly excluded from ML features: `communications_status`,
`image_utility_score`, linearly extrapolated 24-hour SOC, and any net-power
calculation.

| # | Feature name | Derivation | Physical rationale |
|---|---|---|---|
| 1 | `soc_latest` | Value of `battery_soc_percent` at the final sample | Current energy state; most proximate to breach threshold |
| 2 | `soc_mean` | Mean of `battery_soc_percent` over all N samples | Average level over the window |
| 3 | `soc_min` | Minimum of `battery_soc_percent` over all N samples | Worst energy state seen in the window |
| 4 | `soc_slope` | OLS slope of `battery_soc_percent` versus elapsed hours | Rate and direction of energy change |
| 5 | `voltage_latest` | Value of `bus_voltage_v` at the final sample | Current bus voltage; second breach-threshold variable |
| 6 | `voltage_min` | Minimum of `bus_voltage_v` over all N samples | Worst-case voltage in the window |
| 7 | `voltage_slope` | OLS slope of `bus_voltage_v` versus elapsed hours | Rate of voltage degradation |
| 8 | `solar_current_mean` | Mean of `solar_array_current_a` over all N samples | Average generation level |
| 9 | `solar_current_slope` | OLS slope of `solar_array_current_a` versus elapsed hours | Whether generation is increasing or decreasing |
| 10 | `payload_draw_mean` | Mean of `payload_power_draw_w` over all N samples | Average load level |
| 11 | `payload_draw_max` | Maximum of `payload_power_draw_w` over all N samples | Peak load event severity |
| 12 | `high_draw_fraction` | Fraction of N samples where `payload_power_draw_w > 100.0 W` | Duty-cycle intensity of high-load activity |

OLS slope is computed using the standard two-pass formula.  The time axis is
expressed in hours: `t_i = i × (step_minutes / 60)` hours, where `step_minutes`
is 5 for a 5-minute cadence window.  Slope units are therefore change per hour.
No external statistics library is required.

---

## 7. Model Pipeline

**Pipeline:** `StandardScaler → LogisticRegression` (scikit-learn).

### Rationale

- **Transparency:** after scaling, each prediction decomposes exactly into a
  sum of (standardized feature value × learned coefficient) + intercept. This
  decomposition is the explanation per prediction; no separate explainability
  library is required.
- **Probability output:** logistic regression outputs a value in [0, 1].  It is
  labelled explicitly as an estimate from the synthetic scenario distribution,
  not as a calibrated or validated real-spacecraft failure probability.
- **Small corpus suitability:** L2-regularised logistic regression is
  well-behaved on 240 training+validation scenarios. More complex models offer
  marginal benefit and are harder to explain per prediction at this scale.
- **ML dependencies:** scikit-learn, joblib, NumPy, and SciPy are required.
  NumPy and SciPy versions are pinned in `backend/requirements.txt` for
  reproducibility.
- **Serialisation:** a scikit-learn Pipeline object serialises to a single
  `.joblib` file. No model server is required.
- **Inference latency:** No deployment latency claim is made until the packaged
  MVP is benchmarked.

### Per-prediction explanation

For each prediction, the response includes a `feature_contributions` array.
Each entry is:

```
contribution_i = standardized_value_i × coefficient_i
```

where `standardized_value_i = (raw_value_i − mean_i) / std_i` from the fitted
scaler and `coefficient_i` is the corresponding logistic regression weight. The
sum of all 12 contributions plus the intercept equals the log-odds of the
prediction. The top 3 contributors by absolute magnitude are surfaced in
`top_contributions`; all 12 are in `all_contributions`.

### Hyperparameter selection

Regularisation strength `C` is selected by evaluating ROC-AUC on the
validation split (60 scenarios). Candidate values: `[0.01, 0.1, 1.0, 10.0]`.
The value achieving the highest validation ROC-AUC is used for the final model
trained on the combined training and validation splits (240 scenarios total)
before test-set evaluation.

### Serialisation

```
data/models/power_risk_classifier.joblib   (gitignored)
```

Loaded once at application startup into a module-level variable. If the file is
absent the endpoint returns HTTP 503.

---

## 8. API Schemas

### 8.1 Shared policy function

The four instantaneous threshold checks that were duplicated between
`mock_assessment()` and the new prediction endpoint are implemented once in:

```
backend/app/policy.py  →  apply_power_thresholds(sample: dict) -> list[dict]
```

Both `GET /api/v1/mock/assessment` and the prediction endpoints call this
function for their L3 findings. The assessment endpoint's response behavior was
unchanged; its inline threshold block was replaced by a call to
`apply_power_thresholds()`, producing identical output.

`main.py` was also modified to add the two new prediction routes, model loading
at startup, global exception handlers from `safety.py`, and `"POST"` in the
CORS `allow_methods` list.

### 8.2 Core POST endpoint

```
POST /api/v1/power-risk/predict
Content-Type: application/json
```

**Request body:**

```json
{
  "samples": [
    {
      "timestamp":              "2026-08-12T14:00:00Z",
      "solar_array_current_a":  8.4,
      "payload_power_draw_w":   28.0,
      "bus_voltage_v":          28.3,
      "battery_soc_percent":    63.0,
      "command_activity":       "NOMINAL_ATTITUDE_HOLD",
      "communications_status":  "NO_CONTACT_WINDOW",
      "image_utility_score":    0.72
    }
  ],
  "projection_assumptions": {
    "battery_capacity_wh":         number,
    "base_spacecraft_load_w":      number,
    "power_conversion_efficiency": number,
    "sunlight_schedule":  [ { "start": "...", "end": "..." } ],
    "payload_schedule":   [ { "start": "...", "end": "...", "draw_w": number } ]
  }
}
```

`projection_assumptions` is optional. All five sub-fields are required together
if L2 is desired; if any one is absent, L2 is omitted and
`projection_omitted_reason` is set.

At least 72 valid samples are required for AI inference. Below 72, the response
is returned with `status: "degraded"`, `ai_prediction: null`, and
`breach_probability: null`. L3 and L4 still run. `deterministic_projection` is
`null` when assumptions are absent or partial; when complete valid assumptions
are supplied, L2 runs independently and a 24-entry projection is returned even
in degraded mode. `projection_omitted_reason` is always present and explains the
omission when applicable.

### 8.3 Demonstration GET endpoint

```
GET /api/v1/mock/power-risk-prediction
```

No request body. Loads `data/mock/history.json` (72 samples) and calls the
same prediction service function used by the POST endpoint. All response fields
are identical in structure to the POST response.

### 8.4 Shared response structure

Both endpoints return the same structure.

**All numeric values in the example below are illustrative documentation
values chosen so that each `contribution` equals `standardized_value ×
coefficient`.  The runtime implementation must populate every numeric field
from the trained model and the live feature vector; these values must never be
hardcoded.  Contributions describe learned model associations, not proven
physical causation.**

```json
{
  "data_source":       "SYNTHETIC",
  "prototype_status":  "NOT_FLIGHT_QUALIFIED",
  "command_authority": "NONE",
  "policy_decision":   "PERMITTED_FOR_SIMULATION_ONLY",

  "scenario_id":       "SYNTH-DEMO-PUBLIC-001",
  "query_timestamp":   "2026-08-12T20:20:00Z",
  "model_claim":       "logistic-regression-synthetic-not-trained-on-real-spacecraft",
  "model_version":     "0.1.0",

  "status":            "ok",

  "ai_prediction": {
    "label":              "power_constraint_breach_within_24h",
    "predicted_class":    1,
    "breach_probability": 0.79,
    "probability_note":   "Probability estimated from a logistic regression classifier. The final pipeline was fitted on 240 synthetic scenarios (train + validation combined) after regularisation strength C was selected using the 180-scenario training split and a 60-scenario validation split. This is an estimate learned from the synthetic scenario distribution and is not a validated real-spacecraft failure probability. No fixed demonstration value may be hardcoded or presented as operational truth.",
    "top_contributions": [
      { "feature": "soc_slope",          "standardized_value": -2.10, "coefficient": -0.84, "contribution":  1.76 },
      { "feature": "soc_latest",         "standardized_value": -1.40, "coefficient": -0.91, "contribution":  1.27 },
      { "feature": "high_draw_fraction", "standardized_value":  1.80, "coefficient":  0.62, "contribution":  1.12 }
    ],
    "all_contributions": [
      { "feature": "soc_latest",          "standardized_value": -1.40, "coefficient": -0.91, "contribution":  1.27 },
      { "feature": "soc_mean",            "standardized_value": -0.80, "coefficient": -0.54, "contribution":  0.43 },
      { "feature": "soc_min",             "standardized_value": -1.10, "coefficient": -0.63, "contribution":  0.69 },
      { "feature": "soc_slope",           "standardized_value": -2.10, "coefficient": -0.84, "contribution":  1.76 },
      { "feature": "voltage_latest",      "standardized_value": -0.30, "coefficient": -0.42, "contribution":  0.13 },
      { "feature": "voltage_min",         "standardized_value": -0.50, "coefficient": -0.38, "contribution":  0.19 },
      { "feature": "voltage_slope",       "standardized_value": -0.60, "coefficient": -0.31, "contribution":  0.19 },
      { "feature": "solar_current_mean",  "standardized_value":  0.20, "coefficient": -0.27, "contribution": -0.05 },
      { "feature": "solar_current_slope", "standardized_value":  0.10, "coefficient": -0.18, "contribution": -0.02 },
      { "feature": "payload_draw_mean",   "standardized_value":  0.90, "coefficient":  0.45, "contribution":  0.41 },
      { "feature": "payload_draw_max",    "standardized_value":  1.20, "coefficient":  0.52, "contribution":  0.62 },
      { "feature": "high_draw_fraction",  "standardized_value":  1.80, "coefficient":  0.62, "contribution":  1.12 }
    ]
  },

  "deterministic_projection": null,
  "projection_omitted_reason": "Required physical assumptions not supplied",

  "safety_threshold_findings": [
    { "code": "PAYLOAD_LOAD_HIGH", "evidence": "payload_power_draw_w above public demo threshold" }
  ],

  "advisory": {
    "risk_summary":          "HIGH",
    "recommendation":        "DEFER_LOW_PRIORITY_FUTURE_IMAGING",
    "basis":                 "AI breach probability 0.79 exceeds 0.70 threshold; 1 safety threshold finding active.",
    "human_action_required": true,
    "authority_note":        "Advisory output only. No automated action has been or will be taken."
  },

  "audit": {
    "features_used":   12,
    "window_complete": true,
    "samples_used":    72,
    "window_hours":    6.0,
    "action_mode":     "simulation-only"
  }
}
```

**Degraded mode response (fewer than 72 samples):**

```json
{
  "data_source":       "SYNTHETIC",
  "prototype_status":  "NOT_FLIGHT_QUALIFIED",
  "command_authority": "NONE",
  "policy_decision":   "PERMITTED_FOR_SIMULATION_ONLY",
  "status":            "degraded",
  "degraded_reason":   "Fewer than 72 samples available; AI inference requires exactly 72. ai_prediction is null.",
  "ai_prediction":     null,
  "breach_probability": null,
  "safety_threshold_findings": [],
  "advisory": {
    "risk_summary":          "UNKNOWN",
    "recommendation":        "SUPPLY_SUFFICIENT_HISTORY",
    "basis":                 "AI prediction unavailable; insufficient sample count.",
    "human_action_required": true,
    "authority_note":        "Advisory output only. No automated action has been or will be taken."
  },
  "audit": {
    "features_used":   0,
    "window_complete": false,
    "samples_used":    5,
    "window_hours":    0.33,
    "action_mode":     "simulation-only"
  }
}
```

### 8.5 Advisory derivation rule (L4)

Conditions are evaluated in order; the first match applies.

| Condition | `risk_summary` | `recommendation` |
|---|---|---|
| `breach_probability >= 0.70` OR `len(safety_threshold_findings) >= 3` | `HIGH` | `DEFER_LOW_PRIORITY_FUTURE_IMAGING` |
| `breach_probability >= 0.40` OR `len(safety_threshold_findings) >= 1` | `ELEVATED` | `INCREASE_MONITORING_FREQUENCY` |
| `ai_prediction is null` | `UNKNOWN` | `SUPPLY_SUFFICIENT_HISTORY` |
| Neither of the above | `NOMINAL` | `CONTINUE_MONITORING` |

When the L3 findings include an already-active breach (`BATTERY_SOC_LOW` or
`BUS_VOLTAGE_LOW`), the `basis` string notes: "Active breach detected by L3
threshold rules. L1 prediction not applicable; L3 takes precedence."

---

## 9. Deterministic Energy Projection (L2)

L2 is omitted (`null`) and `projection_omitted_reason` is set unless all five
of the following assumptions are supplied in the request body. Without them the
projection cannot be computed with defined physical meaning.

| Field | Unit | Description |
|---|---|---|
| `battery_capacity_wh` | Wh | Total usable battery capacity |
| `base_spacecraft_load_w` | W | Constant base load excluding payload |
| `power_conversion_efficiency` | Dimensionless (0–1) | Solar array to bus conversion efficiency |
| `sunlight_schedule` | Array of `{start, end}` UTC pairs | Future sunlight and eclipse windows over the 24-hour horizon |
| `payload_schedule` | Array of `{start, end, draw_w}` UTC triples | Future payload activity and draw over the 24-hour horizon |

When L2 is present, the response must include within the
`deterministic_projection` object:

- `"method": "physics-based-energy-balance-synthetic"`
- `"not_ai_output": true`
- `"assumption_note"`: a string listing the five supplied assumptions and
  stating explicitly that this is not an AI output
- `"window_complete": true | false`
- An hourly array of 24 entries, each containing `hour_offset`, `forecast_timestamp`, `projected_soc_percent` (clamped [0, 100]), and `projected_breach` (boolean)

---

## 10. Safety Envelope

The following four fields are mandatory on every response, including all error
responses.

| Field | Required value |
|---|---|
| `data_source` | `"SYNTHETIC"` |
| `prototype_status` | `"NOT_FLIGHT_QUALIFIED"` |
| `command_authority` | `"NONE"` |
| `policy_decision` | `"PERMITTED_FOR_SIMULATION_ONLY"` |

**Guarantee scope:** these four fields are guaranteed on all handled application
errors — all `HTTPException` raises and explicitly caught exceptions within the
endpoint function. Unhandled exceptions that escape to FastAPI's default handler
will not automatically include them unless the global exception handler in
`backend/app/safety.py` is registered on the application. Registering that
handler is a required implementation step; until it is in place, the guarantee
does not extend to uncaught exceptions.

---

## 11. Validation Gates

All metrics are computed on the untouched test split (60 scenarios). The test
split is not examined during development; it is evaluated once by
`evaluate_power_risk_model.py` before any release decision.

| Metric | Definition | Minimum MVP threshold |
|---|---|---|
| ROC-AUC | Area under the receiver-operating-characteristic curve | >= 0.80 |
| Recall on label = 1 | TP / (TP + FN) | >= 0.75 |
| Precision on label = 1 | TP / (TP + FP) | >= 0.65 |
| F1 on label = 1 | Harmonic mean of precision and recall | >= 0.70 |
| Brier score | Mean squared error of predicted probabilities against true labels | <= 0.20 |
| No data leakage | Zero shared `scenario_id` values across any two splits | Hard — 0 violations |
| Breach-eligibility compliance | Zero label-1 examples with an active breach at the end of their history window | Hard — 0 violations |

The evaluation script checks leakage and breach-eligibility compliance before
computing any metric. If either hard requirement is violated, the script exits
with a non-zero status code and prints a diagnostic message; no metrics are
reported.

---

## 12. Failure Behavior

| Condition | HTTP status | Notes |
|---|---|---|
| Model file absent at startup | 503 | `"error": "MODEL_NOT_LOADED"` — four mandatory fields present |
| History file absent or unreadable (`data/mock/history.json`) | 503 | `"error": "HISTORY_UNAVAILABLE"` — four mandatory fields present |
| Request body `samples` array is empty or missing | 422 | `"error": "EMPTY_WINDOW"` — four mandatory fields present |
| A required field is missing from one or more samples | 422 | `"error": "INVALID_SAMPLE_SCHEMA"` with sample index and field name |
| Fewer than 72 valid samples | 200 — degraded | `status: "degraded"`, `ai_prediction: null`, `breach_probability: null`; L3 and L4 still run |
| L2 assumptions partially supplied | 200 — L2 omitted | `deterministic_projection: null`, `projection_omitted_reason` set; L1, L3, L4 unaffected |
| Feature extraction raises an unexpected exception | 500 | `"error": "FEATURE_EXTRACTION_FAILED"` — four mandatory fields present (handled by explicit try/except in the endpoint) |
| Unhandled exception before global handler is registered | 500 | FastAPI default body — four mandatory fields not guaranteed until `safety.py` handler is registered |

---

## 13. Implemented File Plan

All files listed below have been created.

### Created files

| Path | Status | Purpose |
|---|---|---|
| `backend/scripts/__init__.py` | Created | Makes `backend/scripts` a Python package. |
| `backend/scripts/generate_scenarios.py` | Created | Exports `generate_prototype_scenarios()` (9 deterministic anchor scenarios) and `generate_training_corpus(seed=42)` (300-scenario balanced corpus used for training and evaluation). |
| `backend/scripts/train_power_risk_model.py` | Created | Generates corpus, selects C on validation split, fits final pipeline on 240 train+val scenarios, serialises to `data/models/power_risk_classifier.joblib`. |
| `backend/scripts/evaluate_power_risk_model.py` | Created | One-shot held-out test evaluation; hard leakage and breach-eligibility checks before metric reporting. |
| `backend/app/policy.py` | Created | Exports `apply_power_thresholds(sample: dict) -> list[dict]`. |
| `backend/app/features.py` | Created | Exports `extract_features()` (12 features) and `ols_slope()`. |
| `backend/app/safety.py` | Created | Global FastAPI exception handlers; mandatory safety envelope. |
| `backend/app/prediction_service.py` | Created | Core four-layer prediction service called by both endpoints. |
| `data/mock/history.json` | Created | Static 72-sample public mock history. No active L3 breach at final sample. |
| `backend/tests/test_generate_scenarios.py` | Created | Tests for both `generate_prototype_scenarios()` and `generate_training_corpus()`. |
| `backend/tests/test_power_risk_prediction.py` | Created | Contract, safety-envelope, L1/L3, degraded-mode, and failure-mode endpoint tests. |
| `backend/tests/test_features.py` | Created | Unit tests for `extract_features()` and `ols_slope()`. |
| `backend/tests/test_policy.py` | Created | Unit tests for `apply_power_thresholds()`. |
| `backend/tests/test_model_pipeline.py` | Created | Focused tests (M01–M23) for training and evaluation pipeline isolation. |

### Modified files

| Path | Change |
|---|---|
| `backend/app/main.py` | Added prediction endpoints, shared policy, model load, global exception handlers, POST CORS. |
| `backend/requirements.txt` | Added `scikit-learn==1.6.1` and `joblib==1.4.2`. |
| `.gitignore` | Added `data/scenarios/` and `data/models/`. |
| `docs/bob-usage.md` | Recorded corpus generation, training, evaluation, and endpoint implementation. |

### Generated artefacts (git-ignored)

| Path | Description |
|---|---|
| `data/models/power_risk_classifier.joblib` | Fitted pipeline; not committed. |
| `data/scenarios/power_scenarios.json` | Generated corpus; not committed. |

### Model availability lifecycle

Generated scenarios and model artifacts are absent from a clean Git clone.
To start the prediction endpoint locally:

1. Run the training command, which generates the corpus in memory and writes
   the model:
   ```
   python -m backend.scripts.train_power_risk_model
   ```
2. Start the API server; it loads the model at startup.

Running the scenario generator CLI (`python backend/scripts/generate_scenarios.py`)
is optional — it writes an inspectable corpus JSON file but is not required
before training.

If `data/models/power_risk_classifier.joblib` is absent, the prediction
endpoints return HTTP 503 with `"error": "MODEL_NOT_LOADED"` and all four
mandatory safety-envelope fields.

Docker integration must later generate scenarios and train the reproducible
public demonstration model during image construction or a controlled startup
step. Generated datasets and model artifacts remain excluded from Git history
under all integration strategies.

### Files not modified during this milestone

- `data/mock/telemetry.json` — 5-sample dashboard demo file; unchanged.
- `backend/tests/test_api.py` — three existing contract tests unchanged.
- All frontend files are unchanged during this backend AI milestone. Dashboard
  integration — wiring the prediction endpoint to the operator dashboard — remains
  required before the final MVP freeze.

---

## 14. Scientific Limitations

The following limitations apply to this prototype and must be understood before
interpreting any output.

1. **Synthetic distribution only.** The classifier is trained exclusively on
   procedurally generated scenarios. Its probability outputs reflect the
   synthetic scenario distribution and must not be interpreted as operational
   spacecraft failure probabilities or as calibrated estimates against any real
   spacecraft's behaviour.  `breach_probability` is an estimate learned from
   the synthetic scenario distribution; it is not a validated real-spacecraft
   failure probability.  No fixed demonstration value (such as 82 %) may be
   hardcoded in source code, documentation, or UI and presented as operational
   truth.

2. **Small corpus.** 240 training+validation scenarios (final fit) is a small
   dataset. The classifier may not generalise to power dynamics outside the
   variation axes explicitly covered by the generator.

3. **Linear feature assumptions.** OLS slope features assume linear trends
   within the 6-hour window. Non-linear dynamics — rapid eclipse transitions,
   step-load changes — are partially captured by the max and fraction features
   but may not be fully represented.

4. **No battery model.** The feature set does not include a physics-consistent
   net-power balance. Battery capacity, internal resistance, and depth-of-
   discharge effects are absent. The classifier learns empirical correlations
   from the synthetic data, not physical laws.

5. **No temporal structure.** The classifier operates on a single summary
   feature vector per window, not on the time series directly. Temporal
   ordering within the window is captured only through slope and fraction
   features.

6. **6-hour window is a prototype choice.** The window length was chosen to
   match the synthetic scenario structure. It has not been validated against
   real operational data.

7. **L2 projection is assumption-driven.** When present, the deterministic
   energy projection depends entirely on the five physical assumptions supplied
   by the caller. If those assumptions are inaccurate, the projected curve will
   be inaccurate. It is not an AI output and does not improve with additional
   training data.

8. **Advisory only.** No output from this system constitutes a spacecraft
   command or an authorisation for autonomous action. All outputs require
   human review before any operational decision is made.

9. **Not flight-qualified.** This prototype has not undergone verification,
   validation, certification, or qualification against any flight standard.
