# SpaceBNS Power-Risk Prediction — Implementation Contract

**Status:** design contract — no implementation yet

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

## 2. Existing Behavior (Unchanged)

The following items exist in `backend/app/main.py` and are not modified by this
contract.

| Endpoint | Source location | Notes |
|---|---|---|
| `GET /health` | `main.py:53` | Unchanged |
| `GET /api/v1/mock/telemetry` | `main.py:66` | Serves `data/mock/telemetry.json` (5 samples); unchanged |
| `GET /api/v1/mock/assessment` | `main.py:78` | Four threshold rules at lines 89–116; unchanged except that its threshold logic will be extracted into a shared policy function |

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
| L2 | Deterministic energy projection | Physics-based hourly curve; present only when all five physical assumptions are supplied; labelled as not an AI output |
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

---

## 5. Synthetic Scenario Corpus

### Generation approach

The corpus is generated reproducibly at build/train time by a fixed-seed Python
script. It is not a committed data file. Generated files are gitignored and must
never enter Git history.

| Item | Value |
|---|---|
| Generator script | `backend/scripts/generate_scenarios.py` |
| Random seed | Fixed constant in the script; documented in a comment |
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

```json
{
  "scenario_id":  "SYNTH-PWR-TRAIN-001",
  "split":        "train",
  "label":        1,
  "breach_detail": {
    "occurs":       true,
    "breach_type":  "SOC_BELOW_25",
    "hour_offset":  14
  },
  "history": [
    {
      "timestamp":             "2026-01-01T00:00:00Z",
      "solar_array_current_a": 7.2,
      "payload_power_draw_w":  45.0,
      "bus_voltage_v":         27.4,
      "battery_soc_percent":   58.0,
      "command_activity":      "NOMINAL_ATTITUDE_HOLD"
    }
  ]
}
```

`history` contains exactly 72 samples. Fields `image_utility_score` and
`communications_status` are excluded; they are not used as features and add no
value to the history array.

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
- Breach timing for label-1 scenarios: early (hour 1–8), mid (hour 9–16), late
  (hour 17–24).

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

Explicitly excluded: `image_utility_score`, `communications_status`, linearly
extrapolated 24-hour SOC, and any net-power calculation.

| # | Feature name | Derivation | Physical rationale |
|---|---|---|---|
| 1 | `soc_latest` | Value of `battery_soc_percent` at the final sample | Current energy state; most proximate to breach threshold |
| 2 | `soc_mean` | Mean of `battery_soc_percent` over all N samples | Average level over the window |
| 3 | `soc_min` | Minimum of `battery_soc_percent` over all N samples | Worst energy state seen in the window |
| 4 | `soc_slope` | OLS slope of `battery_soc_percent` versus elapsed minutes | Rate and direction of energy change |
| 5 | `voltage_latest` | Value of `bus_voltage_v` at the final sample | Current bus voltage; second breach-threshold variable |
| 6 | `voltage_min` | Minimum of `bus_voltage_v` over all N samples | Worst-case voltage in the window |
| 7 | `voltage_slope` | OLS slope of `bus_voltage_v` versus elapsed minutes | Rate of voltage degradation |
| 8 | `solar_current_mean` | Mean of `solar_array_current_a` over all N samples | Average generation level |
| 9 | `solar_current_slope` | OLS slope of `solar_array_current_a` versus elapsed minutes | Whether generation is increasing or decreasing |
| 10 | `payload_draw_mean` | Mean of `payload_power_draw_w` over all N samples | Average load level |
| 11 | `payload_draw_max` | Maximum of `payload_power_draw_w` over all N samples | Peak load event severity |
| 12 | `high_draw_fraction` | Fraction of N samples where `payload_power_draw_w > 100.0 W` | Duty-cycle intensity of high-load activity |

OLS slope is computed using the standard two-pass formula over elapsed minutes.
No external statistics library is required.

---

## 7. Model Pipeline

**Pipeline:** `StandardScaler → LogisticRegression` (scikit-learn).

### Rationale

- **Transparency:** after scaling, each prediction decomposes exactly into a
  sum of (standardized feature value × learned coefficient) + intercept. This
  decomposition is the explanation per prediction; no separate explainability
  library is required.
- **Calibrated probability:** logistic regression outputs are directly
  interpretable as probabilities. They are labelled explicitly as estimated from
  the synthetic scenario distribution, not as operational failure probabilities.
- **Small corpus suitability:** L2-regularised logistic regression is
  well-behaved on 180 training scenarios. More complex models offer marginal
  benefit and are harder to explain per prediction at this scale.
- **No additional dependencies:** scikit-learn is the only ML dependency.
- **Serialisation:** a scikit-learn Pipeline object serialises to a single
  `.joblib` file, typically under 100 KB for 12 features. No model server is
  required.
- **Inference latency:** under 5 ms per prediction.

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
trained on the combined training and validation splits before test-set
evaluation.

### Serialisation

```
data/models/power_risk_classifier.joblib   (gitignored)
```

Loaded once at application startup into a module-level variable. If the file is
absent the endpoint returns HTTP 503.

---

## 8. API Schemas

### 8.1 Shared policy function

The four instantaneous threshold checks currently duplicated between
`mock_assessment()` and the new prediction endpoint are extracted into one
function:

```
backend/app/policy.py  →  apply_power_thresholds(sample: dict) -> list[dict]
```

Both `GET /api/v1/mock/assessment` and the new prediction endpoint call this
function for their L3 findings. The thresholds in `main.py` lines 89–116 are
not removed from the existing endpoint; they are replaced by a call to this
shared function, producing identical output. This is the only permitted
modification to `main.py`.

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
      "timestamp":             "2026-08-12T14:00:00Z",
      "solar_array_current_a": 8.4,
      "payload_power_draw_w":  28.0,
      "bus_voltage_v":         28.3,
      "battery_soc_percent":   63.0,
      "command_activity":      "NOMINAL_ATTITUDE_HOLD"
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
`breach_probability: null`. L3 and L4 still run.

**CORS note:** the existing CORS middleware allows only `GET` methods. Adding
the POST endpoint requires adding `"POST"` to `allow_methods`. This is the only
middleware change required.

### 8.3 Demonstration GET endpoint

```
GET /api/v1/mock/power-risk-prediction
```

No request body. Loads `data/mock/history.json` (72 samples) and calls the
same prediction service function used by the POST endpoint. All response fields
are identical in structure to the POST response.

### 8.4 Shared response structure

Both endpoints return the same structure.

```json
{
  "data_source":       "SYNTHETIC",
  "prototype_status":  "NOT_FLIGHT_QUALIFIED",
  "command_authority": "NONE",
  "policy_decision":   "PERMITTED_FOR_SIMULATION_ONLY",

  "scenario_id":       "PUBLIC-DEMO-HISTORY-001",
  "query_timestamp":   "2026-08-12T20:20:00Z",
  "model_claim":       "logistic-regression-synthetic-not-trained-on-real-spacecraft",
  "model_version":     "0.1.0",

  "status":            "ok",

  "ai_prediction": {
    "label":              "power_constraint_breach_within_24h",
    "predicted_class":    1,
    "breach_probability": 0.79,
    "probability_note":   "Probability estimated from a logistic regression classifier trained on 180 synthetic scenarios. This is not an operational spacecraft failure probability.",
    "top_contributions": [
      { "feature": "soc_slope",          "standardized_value": -2.1, "coefficient":  0.84, "contribution": -1.76 },
      { "feature": "soc_latest",         "standardized_value": -1.4, "coefficient":  0.91, "contribution": -1.27 },
      { "feature": "high_draw_fraction", "standardized_value":  1.8, "coefficient":  0.62, "contribution":  1.12 }
    ],
    "all_contributions": []
  },

  "deterministic_projection": null,
  "projection_omitted_reason": "Required physical assumptions not supplied",

  "safety_threshold_findings": [
    { "code": "PAYLOAD_LOAD_HIGH", "evidence": "payload_power_draw_w above public demo threshold" }
  ],

  "advisory": {
    "risk_summary":          "ELEVATED",
    "recommendation":        "INCREASE_MONITORING_FREQUENCY",
    "basis":                 "AI breach probability 0.79 exceeds 0.40 threshold; 1 safety threshold finding active.",
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

## 13. Proposed File Plan

No files have been created or modified by this contract.

### New files to create

| Path | Purpose |
|---|---|
| `data/mock/history.json` | Static 72-sample public mock history for the demonstration GET endpoint. Hand-authored; no active L3 breach at sample 72. Committed to the repository. |
| `backend/app/policy.py` | Exports `apply_power_thresholds(sample: dict) -> list[dict]`. Contains the four threshold rules extracted from `mock_assessment()`. Called by both the existing assessment endpoint and the new prediction endpoint. |
| `backend/app/features.py` | Pure module. Exports `extract_features(samples: list[dict]) -> dict[str, float]` (12 features) and `ols_slope(values: list[float], times: list[float]) -> float`. No I/O; no side effects. |
| `backend/app/safety.py` | Exports the global FastAPI exception handler and the mandatory safety envelope dict. Registers the handler on the app to ensure the four mandatory fields appear on all unhandled error responses. |
| `backend/app/prediction_service.py` | Core prediction service function called by both the POST and GET prediction endpoints. Accepts a list of sample dicts and optional projection assumptions. Returns the full four-layer response dict. |
| `backend/scripts/generate_scenarios.py` | Fixed-seed generator. Produces `data/scenarios/power_scenarios.json` (300 scenarios). Output is gitignored. |
| `backend/scripts/train_power_risk_model.py` | Loads the scenario corpus; extracts 12 features; runs hyperparameter selection on the validation split; trains the final pipeline on train + val; serialises to `data/models/power_risk_classifier.joblib`; prints metrics to stdout. |
| `backend/scripts/evaluate_power_risk_model.py` | Loads the test split; runs inference; checks leakage and breach-eligibility before computing metrics; outputs a JSON report to stdout. Does not modify any file. |
| `backend/tests/test_power_risk_prediction.py` | Contract, safety-envelope, L1/L3 boundary, degraded-mode, and failure-mode tests for both prediction endpoints. Uses `TestClient`. |
| `backend/tests/test_features.py` | Unit tests for `extract_features()` and `ols_slope()` using known synthetic inputs with hand-computed expected values. |
| `backend/tests/test_policy.py` | Unit tests for `apply_power_thresholds()` confirming that its output matches the existing `mock_assessment()` findings for all four threshold conditions. |
| `backend/tests/test_generate_scenarios.py` | Checks determinism, scenario counts, split assignment, and breach-eligibility compliance of the generator. |

### Files to modify

| Path | Change |
|---|---|
| `backend/app/main.py` | (1) Replace the inline threshold block in `mock_assessment()` with a call to `apply_power_thresholds()` from `backend/app/policy.py`. (2) Add `POST /api/v1/power-risk/predict` and `GET /api/v1/mock/power-risk-prediction` routes. (3) Add model load at startup with graceful 503 if absent. (4) Import and register the global exception handler from `safety.py`. (5) Add `"POST"` to `allow_methods` in the CORS middleware. All other existing logic is untouched. |
| `backend/requirements.txt` | Add `scikit-learn` and `joblib` to satisfy the model pipeline and serialisation dependencies introduced by this milestone. |
| `.gitignore` | Add `data/scenarios/` and `data/models/` to prevent generated artefacts from entering Git history. |
| `docs/architecture.md` | Add the four-layer separation (L1–L4) to the component table and the decision contract. Add a row for the power-risk prediction component and the shared policy function. |
| `docs/bob-usage.md` | Record corpus generation, model training, and endpoint implementation as new evidence entries when that work is completed. |

### Model availability lifecycle

Generated scenarios and model artifacts are absent from a clean Git clone.
Before starting the prediction endpoint locally, a developer must run the
scenario generator and then the training command to produce
`data/models/power_risk_classifier.joblib`. If the model file is unavailable,
the endpoint returns its documented HTTP 503 response with
`"error": "MODEL_NOT_LOADED"` and all four mandatory safety envelope fields.

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
   spacecraft's behaviour.

2. **Small corpus.** 180 training scenarios is a small dataset. The classifier
   may not generalise to power dynamics outside the variation axes explicitly
   covered by the generator.

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
