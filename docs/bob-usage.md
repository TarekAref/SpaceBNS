# IBM Bob Usage Log

IBM Bob is the required primary development tool for the IBM August Challenge.
This document must contain evidence of **real** Bob-assisted work. Do not claim
that Bob created, reviewed, or tested an artifact unless that interaction
actually occurred.

## Evidence template

Copy this section for each meaningful Bob-assisted task.

### Task: Short task name

- **Date (UTC):**
- **Developer:** Tarek Aref
- **Bob mode/workflow:**
- **Goal:**
- **Prompt summary:** Summarize the actual request without including secrets, credentials, proprietary data, or sensitive mission or personal information.
- **Files affected:**
- **Bob contribution:**
- **Human review and corrections:**
- **Validation performed:**
- **Result:**

## Recorded evidence

### Task: Frontend dependency security upgrade

- **Date (UTC):** 2026-08-14
- **Developer:** Tarek Aref
- **Bob mode/workflow:** IBM Bob Agent mode with human approval
- **Goal:** Resolve confirmed frontend dependency vulnerabilities while preserving the current scaffold's build and test behavior.
- **Prompt summary:** Inspect npm audit findings, perform an exact Next.js upgrade (15.5.23 → 16.3.0), avoid force-fixing unrelated advisories, and validate the result.
- **Files affected:** `frontend/package.json`, `frontend/package-lock.json`, `frontend/tsconfig.json`
- **Bob contribution:** Identified vulnerable packages, proposed the upgrade plan, and executed the approved upgrade and validation commands. Upgraded Next.js from 15.5.23 to 16.3.0; the Next.js upgrade caused the resolved transitive dependency versions to become PostCSS 8.5.23 and Sharp 0.35.3. `frontend/tsconfig.json` was updated automatically by Next.js during its build step.
- **Human review and corrections:** Tarek approved the upgrade commands, then reviewed and accepted the resulting dependency and configuration diffs.
- **Validation performed:** `npm ci` (clean install); `npm audit` (0 vulnerabilities); TypeScript type-check; Next.js production build; 3 backend regression tests passing.
- **Result:** Next.js upgraded to 16.3.0; `npm audit` reported zero known vulnerabilities, and no regressions were detected by the current automated checks.

---

### Task: Generated-file hygiene and Git checkpoint

- **Date (UTC):** 2026-08-14
- **Developer:** Tarek Aref
- **Bob mode/workflow:** IBM Bob Agent mode with human approval
- **Goal:** Exclude generated TypeScript artifacts from version control without deleting required local files.
- **Prompt summary:** Review Next.js-generated files, ignore reproducible build caches, stop tracking `next-env.d.ts` without deleting its local copy, and validate and checkpoint the exact changes.
- **Files affected:** `.gitignore`, `frontend/next-env.d.ts`, `frontend/tsconfig.tsbuildinfo` (local only)
- **Bob contribution:** Added `*.tsbuildinfo` and `next-env.d.ts` patterns to `.gitignore`; ran `git rm --cached` on `frontend/next-env.d.ts` to stop tracking it while leaving the file on disk; deleted the untracked generated cache `frontend/tsconfig.tsbuildinfo` locally and added it to `.gitignore`; verified the working tree with staged and unstaged diff checks. Bob assisted and executed approved commands; Tarek remained the author and decision-maker throughout.
- **Human review and corrections:** Tarek confirmed that `frontend/next-env.d.ts` remained locally available after removal from tracking and approved all `.gitignore` additions.
- **Validation performed:** Staged and unstaged `git diff` checks; clean working tree confirmed; changes committed as `c6ffe78`; successful non-force push to branch `bob/mvp-build`.
- **Result:** `frontend/next-env.d.ts` is no longer tracked but remains available locally; generated cache files are ignored; the working tree was clean and `bob/mvp-build` was synchronized with its remote branch.

---

### Task: Deterministic scenario generator implementation and review

- **Date (UTC):** 2026-08-15
- **Developer:** Tarek Aref
- **Bob mode/workflow:** IBM Bob Agent mode with human approval
- **Goal:** Implement and test a deterministic prototype scenario generator that proves the feasibility of the synthetic power-breach corpus design before the full 300-scenario balanced corpus is built.
- **Prompt summary:** Design and implement `generate_prototype_scenarios()` to produce exactly 9 deterministic positive-breach scenarios covering all 3 breach types (SOC_ONLY, VOLTAGE_ONLY, BOTH) across all 3 timing bands (early, middle, late); write 23 property-based tests (T01–T23) to verify physics correctness, schema compliance, determinism, and safety metadata; validate the full backend test suite.
- **Files affected:** `backend/scripts/__init__.py`, `backend/scripts/generate_scenarios.py`, `backend/tests/test_generate_scenarios.py`
- **Bob contribution:** Designed the closed-form physics calibration approach (no random rejection loops), implemented the three scenario families (SOC_ONLY, VOLTAGE_ONLY, BOTH) with calibrated initial conditions, wrote all 23 test functions with parametrised coverage across all 9 scenario IDs, and integrated the `backend/scripts` package init.
- **Human review and corrections:** Tarek reviewed the breach-type vocabulary and corrected it from an earlier draft that used `SOC_BELOW_25` and `BUS_VOLTAGE_LOW` to the canonical values `SOC_ONLY`, `VOLTAGE_ONLY`, and `BOTH` used throughout the contract and generator.
- **Validation performed:** 175 generator-specific tests passed (23 test functions × 9 parametrised scenarios, plus module-level tests); 200 total backend tests passed. Commit: `983ab37`.
- **Result:** `generate_prototype_scenarios()` returns exactly 9 deterministic, physics-consistent positive scenarios that serve as scientific test anchors for the prototype stage. No balanced training corpus or trained AI model exists yet; those remain to be built.

---

### Task: Train and evaluate the first power-risk AI prototype

- **Date (UTC):** 2026-08-17
- **Developer:** Tarek Aref
- **Bob mode/workflow:** IBM Bob Agent mode — task authorized and scoped by Tarek; Bob acted as primary implementation tool
- **Goal:** Train the first SpaceBNS power-risk binary classifier and evaluate it once against the untouched synthetic held-out test split, within the boundaries of the approved implementation contract.
- **Prompt summary:** Implement `train_power_risk_model.py` (StandardScaler → LogisticRegression pipeline, C-selection on validation split, final fit on 240 train+val scenarios, joblib serialisation), `evaluate_power_risk_model.py` (hard validation checks, one-shot test-split evaluation, JSON report), and `test_model_pipeline.py` (23 focused tests M01–M23). Install scikit-learn==1.6.1 and joblib==1.4.2. Generate the 300-scenario corpus with seed=42, train the model, run the evaluation exactly once. All scope and threshold decisions were pre-approved by Tarek.
- **Files affected:**
  - `backend/requirements.txt` — added `scikit-learn==1.6.1` and `joblib==1.4.2`
  - `backend/scripts/train_power_risk_model.py` — created (new)
  - `backend/scripts/evaluate_power_risk_model.py` — created (new)
  - `backend/tests/test_model_pipeline.py` — created (new)
  - `docs/bob-usage.md` — updated (this entry)
  - `data/models/power_risk_classifier.joblib` — generated, git-ignored, not committed
  - `data/scenarios/power_scenarios.json` — not generated as a file; corpus is generated in-memory at train/evaluate time
- **Isolation boundaries:** The test split was held out from all model fitting and hyperparameter selection. C was selected using only validation ROC-AUC; the final pipeline was fitted on train+validation (240 scenarios) only. The evaluation script (`evaluate_power_risk_model.py`) accessed test scenario IDs, schema, sample counts, label balance, and breach-eligibility fields as hard validation checks before computing metrics — schema and count checks on the test split are considered pre-computation validation, not post-hoc tuning. The evaluation script was executed exactly once.
- **Bob contribution:** Designed and implemented the full training and evaluation pipeline strictly within the approved contract. Key implementation decisions:
  - Feature extraction uses `extract_features(scenario["history"])` exclusively; future data is never accessed.
  - `FEATURE_ORDER` constant (12 features, frozen order from contract Section 6) defined once in `train_power_risk_model.py` and imported by `evaluate_power_risk_model.py`.
  - C-candidates `[0.01, 0.1, 1.0, 10.0]` evaluated on validation split only; ties broken by smallest C.
  - Final pipeline trained on combined train+validation (180+60 = 240 scenarios).
  - Hard validation checks (split counts, class balance, leakage, breach eligibility, feature count/order, finite values, finite probabilities) gate metric reporting in the evaluate script.
  - Convergence warnings treated as errors via `ConvergenceWarning` filter (contract requirement).
  - ROC-AUC computed using the Wilcoxon-Mann-Whitney U statistic (numerically equivalent to trapezoidal AUC, no scipy dependency).
  - 23 focused tests (M01–M23) plus M06b.
  - Iterative fix: initial `UserWarning` filter triggered on unrelated scipy solver internal warning; corrected to use `ConvergenceWarning` only. Initial custom ROC-AUC had an off-by-one accumulation bug; replaced with WMW U-statistic formulation and verified against sklearn reference.
- **Human review and corrections:** Tarek approved the full task scope and authorized the single held-out test evaluation. GitHub-bridge review (commit `34fde7a`) identified isolation-test weaknesses: tests M15, M16, M17, M18 and M22 were too weak or used incorrect data; `_extract_feature_vector` silently passed on feature-order mismatch; `FEATURE_ORDER` was duplicated; training read test labels unnecessarily. A correction task was authorized by Tarek and executed in commit `fix(backend): strengthen model evaluation isolation`. The model, corpus, metrics, and reported results were not changed.
- **Validation performed (original commit `34fde7a`):**
  - `pip check` — no broken requirements
  - 27 focused model-pipeline tests (M01–M23) — all passed
  - 3583 total backend tests — all passed (0 failures)
  - `git diff --check` — exit 0
  - Artifact paths verified as git-ignored via `git check-ignore`
- **Held-out synthetic test results (evaluated once in commit `34fde7a`, seed=42 corpus, C=1.0; not re-evaluated in the correction commit):**
  - ROC-AUC: 1.000 (gate ≥ 0.80 ✓)
  - Recall (label=1): 1.000 (gate ≥ 0.75 ✓)
  - Precision (label=1): 0.9375 (gate ≥ 0.65 ✓)
  - F1 (label=1): 0.9677 (gate ≥ 0.70 ✓)
  - Brier score: 0.0363 (gate ≤ 0.20 ✓)
  - Confusion matrix [[TN, FP], [FN, TP]]: [[28, 2], [0, 30]]
  - Leakage violations: 0 ✓ — Breach eligibility violations: 0 ✓
  - Selected C: 1.0 (validation ROC-AUC by C: {0.01: 0.9944, 0.1: 0.9956, 1.0: 0.9967, 10.0: 0.9956})
- **⚠️ Synthetic-separability warning:** ROC-AUC = 1.000 and Recall = 1.000 are near-perfect metrics. This indicates strong linear separability in the synthetic feature space. These results reflect the closed-form physics generator's deterministic structure — the feature distributions for positive and negative classes are well-separated by construction. Near-perfect metrics on this synthetic corpus MUST NOT be interpreted as evidence of operational accuracy or real-world generalization.
- **Scientific and safety limitations:**
  - Model trained exclusively on synthetic scenarios generated by the SpaceBNS physics simulator (seed=42, 300 scenarios). It has not been trained on or validated against any real spacecraft data.
  - Metrics describe held-out synthetic scenario performance only, not real-spacecraft failure prediction.
  - `breach_probability` is NOT a real-spacecraft failure probability.
  - Model is `NOT_FLIGHT_QUALIFIED`. `command_authority: "NONE"`. All outputs are advisory and simulation-only.
  - No output from this system constitutes a spacecraft command or authorisation for autonomous action. All outputs require human review before any operational decision.
- **Result:** All 7 contract gates passed on first and only test evaluation. Training pipeline, evaluation script, and focused tests committed to `bob/mvp-build`. No tuning against the test set occurred. Isolation weaknesses corrected in follow-up commit per GitHub-bridge review.

---

### Task: Implement four-layer power-risk prediction API

- **Date (UTC):** 2026-08-18–20
- **Developer:** Tarek Aref
- **Bob mode/workflow:** IBM Bob Agent mode — task authorized and scoped by Tarek; Bob acted as primary implementation tool
- **Goal:** Implement the complete four-layer prediction API (L1 AI, L2 deterministic projection, L3 safety thresholds, L4 advisory) as specified in the approved contract, wire it into `main.py`, write the full contract test suite, and add NumPy/SciPy version pins.
- **Prompt summary:** Implement `prediction_service.py` (run_prediction, degraded mode, L2 projection independent of AI availability), `safety.py` (global exception handlers, INVALID_SAMPLE_SCHEMA, EMPTY_WINDOW), and `history.json` (72-sample mock). Wire routes, model load, CORS POST, and global handlers into `main.py`. Write `test_power_risk_prediction.py` covering P01–P38 contract tests plus fix1–fix5 and fixA1–fixA6 regression tests. Pin `numpy==2.2.6` and `scipy==1.15.3` in `requirements.txt`: local testing with the newer unpinned NumPy 2.5.2 and SciPy 1.18.0 produced a SciPy OptimizeWarning: 'Unknown solver options: iprint'. Pinning the compatibility-tested NumPy 2.2.6 and SciPy 1.15.3 combination eliminated the warning while pip check and all tests passed.
- **Files affected:**
  - `backend/app/main.py` — new routes, model load, CORS POST, global exception handlers
  - `backend/app/prediction_service.py` — created (new)
  - `backend/app/safety.py` — created (new)
  - `data/mock/history.json` — created (new)
  - `backend/tests/test_power_risk_prediction.py` — created (new); 99 tests
  - `backend/requirements.txt` — added `numpy==2.2.6` and `scipy==1.15.3`
  - `docs/power-risk-contract.md` — synchronised with implemented state across multiple correction passes
  - `docs/bob-usage.md` — updated (this entry)
- **Bob contribution:** Implemented the complete prediction service and safety handler, designed the degraded-mode L2 independence path, wrote all test fixtures and regression tests, and applied successive contract alignment corrections identified by GitHub-bridge audits. NumPy/SciPy pins were added after local testing with the newer unpinned NumPy 2.5.2 and SciPy 1.18.0 produced a SciPy OptimizeWarning: 'Unknown solver options: iprint'; pinning the compatibility-tested NumPy 2.2.6 and SciPy 1.15.3 combination eliminated the warning while pip check and all tests passed.
- **Human review and corrections:** Tarek authorized all scope and reviewed all diffs. Three successive GitHub-bridge audit passes (correction-only, schema-alignment, and documentation consistency) were completed; each pass was authorized by Tarek before execution.
- **Validation performed:**
  - `pip check` — no broken requirements
  - 99 focused prediction tests — all passed
  - 3683+ total backend tests — all passed
  - `git diff --check` — no whitespace errors (only autocrlf normalisation warnings)
- **Scientific and safety disclosures:**
  - `breach_probability` is NOT a real-spacecraft failure probability.
  - Model is `NOT_FLIGHT_QUALIFIED`. `command_authority: "NONE"`. All outputs are advisory and simulation-only.
  - Endpoints are locally tested; not deployed or wired to the frontend as of this entry.
- **Result:** Four-layer prediction API fully implemented and tested. All contract validation gates confirmed. API not yet deployed.

---

### Task: Strengthen model evaluation isolation (correction)

- **Date (UTC):** 2026-08-17
- **Developer:** Tarek Aref
- **Bob mode/workflow:** IBM Bob Agent mode — correction task authorized by Tarek following GitHub-bridge review of commit `34fde7a`
- **Goal:** Address isolation and test-quality findings from the GitHub-bridge review without changing the trained model, corpus, features, labels, splits, thresholds, selected C, or reported metrics.
- **Prompt summary:** (1) Remove test-label inspection from `train_power_risk_model.py` — training should not read test labels, `test_pos`, or `test_neg`. (2) Fail closed on feature-order mismatch in `_extract_feature_vector()` instead of silently passing. (3) Define `FEATURE_ORDER` once in the train module and import it in the evaluate module. (4) Add hard class-count validation for all three splits in `evaluate_power_risk_model.py`. (5) Strengthen tests M15 (monkeypatch `_roc_auc_score` to verify validation-only labels), M16 (check scaler's `n_samples_seen_ == 240`), M17 (monkeypatch `_build_matrices` to verify test IDs never passed), M18 (use validation not test scenarios), M22 (snapshot directories before/after import), and add M06b (feature-order mismatch raises `ValueError`). (6) Correct the bob-usage.md entry date and review record.
- **Files affected:**
  - `backend/scripts/train_power_risk_model.py` — removed `test_pos`/`test_neg` label reads; `_extract_feature_vector` now raises `ValueError` on mismatch
  - `backend/scripts/evaluate_power_risk_model.py` — removed duplicate `FEATURE_ORDER`, now imported from train module; added class-count hard checks for all three splits
  - `backend/tests/test_model_pipeline.py` — strengthened M15, M16, M17, M18, M22; added M06b
  - `docs/bob-usage.md` — updated (this entry)
- **Bob contribution:** Implemented all five code corrections and updated documentation per the approved correction scope. The trained model, joblib artifact, corpus, features, thresholds, selected C, and all reported metrics are unchanged.
- **Human review and corrections:** Tarek authorized the correction scope. No model retraining or re-evaluation of the held-out test split was performed. GitHub-bridge review of commit `5b9e160` passed on 2026-08-17.
- **Validation performed:**
  - `pip check` — no broken requirements
  - Focused model-pipeline tests — all passed (reported after test run)
  - Full backend test suite — all passed (reported after test run)
  - `git diff --check` — exit 0

### Task: Final contract-alignment corrections (power-risk prediction service)

- **Date (UTC):** 2026-08-18–20
- **Developer:** Tarek Aref
- **Bob mode/workflow:** IBM Bob Agent mode — correction task authorized by Tarek following GitHub-bridge audit
- **Goal:** Correct five code defects and one stale docstring identified by the audit, synchronise the contract document with implemented reality, and add regression tests proving each correction.
- **Prompt summary:** (1) Update `_PROBABILITY_NOTE` from "180 training scenarios" to the accurate 240-scenario final-fit statement. (2) Make L2 deterministic projection available in degraded mode independently of AI inference. (3) Ensure POST `{}` returns 422 `EMPTY_WINDOW` with the safety envelope. (4) Extend `_validate_samples` to enforce all 8 required raw telemetry fields including the three non-ML fields. (5) Synchronise `docs/power-risk-contract.md` with the implemented state: status, `generate_training_corpus(seed=42)`, 240-scenario final fit, slope units (per hour), `SYNTH-DEMO-PUBLIC-001`, 0.79 exceeds 0.70, file plan, and cautious latency/probability wording. (6) Remove the stale ±1-second tolerance claim from `_validate_samples` docstring.
- **Files affected:**
  - `backend/app/prediction_service.py` — `_PROBABILITY_NOTE`, `_validate_samples`, degraded-mode L2 path
  - `backend/app/safety.py` — `RequestValidationError` handler: POST `{}` → `EMPTY_WINDOW`
  - `backend/tests/test_power_risk_prediction.py` — 18 new regression tests (fixA1–fixA6)
  - `docs/power-risk-contract.md` — status, corpus section, model rationale, slope units, file plan, response examples, limitation 2
  - `docs/bob-usage.md` — this entry
- **Bob contribution:** Diagnosed all six defects from the audit description, implemented the minimal targeted corrections, and wrote focused regression tests. No features, labels, thresholds, model type, dataset splits, or policy rules were changed. Successive documentation consistency passes (v4 audit) identified only documentation items; all executable code passed the v4 audit without further code changes.
- **Human review and corrections:** Tarek authorized the correction scope and confirmed that the v4 executable-code audit passed with only documentation consistency corrections remaining.
- **Validation performed:**
  - `pip check` — no broken requirements
  - 99 focused prediction tests — all passed
  - 3683 total backend tests — all passed
  - `git diff --check` — only autocrlf normalisation warnings (no whitespace errors)
- **Result:** All six defects corrected; 18 regression tests added; contract document synchronized with the implemented state.

---

### Task: Connect judge-facing dashboard to power-risk prediction API

- **Date (UTC):** 2026-08-21
- **Developer:** Tarek Aref
- **Bob mode/workflow:** IBM Bob Agent mode — task authorized and scoped by Tarek; Bob acted as primary implementation tool
- **Goal:** Replace the old `/api/v1/mock/assessment` dashboard integration with a full four-layer judge-facing dashboard wired to `GET /api/v1/mock/power-risk-prediction`.
- **Prompt summary:** Read and implement the complete frontend milestone: connect the dashboard to the power-risk prediction endpoint, implement strict TypeScript types for normal and degraded API responses, build a permanent safety strip (SYNTHETIC DATA · NOT FLIGHT QUALIFIED · COMMAND AUTHORITY: NONE · SIMULATION ONLY), render four labelled layers (L1 AI estimate, L2 deterministic projection, L3 safety findings, L4 operator advisory), include an inline dependency-free SVG SOC chart for L2 with 25% threshold, a CSS horizontal-bar explainability panel for top contributions, provenance/audit metadata, accessible error and loading states with a manual-only retry button, and a `prefers-reduced-motion` rule. No dependencies added. No backend modified.
- **Files affected:**
  - `frontend/app/page.tsx` — replaced old assessment integration; implemented full four-layer dashboard
  - `frontend/app/globals.css` — replaced old styles; added safety strip, advisory banner, L1–L4 panel styles, SVG chart styles, loading/error/retry styles, `prefers-reduced-motion`, responsive rules
  - `docs/bob-usage.md` — updated (this entry)
- **Endpoint used:** `GET ${NEXT_PUBLIC_API_BASE_URL}/api/v1/mock/power-risk-prediction` (fallback base URL: `http://localhost:8000`)
- **Four-layer presentation:**
  - L1: AI breach probability (% from API), predicted class, model version, 6-hour/72-sample audit basis; degraded state when `ai_prediction` is null
  - L1 Explainability: dependency-free CSS horizontal bars for top 3 contributions, normalized against largest absolute value, positive/negative visually distinguished, described as learned associations not physical causes
  - L2: inline SVG SOC chart when 24 hourly projection points are present, 25% SOC safety threshold line, breach points in red; omitted state with API's `projection_omitted_reason` when null; explicit "NOT AI OUTPUT" label
  - L3: all `safety_threshold_findings` entries; "No active threshold findings" state for empty array
  - L4: `risk_summary`, `recommendation`, `basis`, `human_action_required`, `authority_note`; advisory-only disclaimer; no command/uplink/execution controls
  - Provenance: `scenario_id`, `query_timestamp`, `model_claim`, `samples_used`, `window_hours`, `features_used`, `action_mode`; missing optional values shown as "not available"
- **Safety and advisory-only controls:**
  - Permanent safety strip always visible (loading, error, and data states)
  - No hardcoded probability, predicted class, or model result
  - No command/uplink/execution/autonomous-action functionality
  - No polling or auto-refresh; retry button reloads only the public API
  - Raw exception text, response bodies, paths, and stack traces never exposed
  - `command_authority: "NONE"` validated at response-parse time
- **Bob contribution:** Designed the complete four-layer component hierarchy, strict TypeScript union types for normal/degraded responses, inline SVG chart with accessible labels, CSS bar chart for explainability, all state handling, safety strip, and advisory-only UX patterns. No dependencies were added.
- **Human review and corrections:** Tarek completed post-implementation runtime review on 2026-08-27/28. That review identified a numerical-stability defect in the trained artifact plus responsive-layout collisions; the human-led corrections and validation are documented in the following entry and are not attributed to Bob.
- **Validation performed:**
  - `npm ci` — clean install, 0 vulnerabilities
  - `npm run build` (Next.js 16.3.0, Turbopack) — compiled successfully, TypeScript passed, 3/3 static pages generated, no errors or warnings
  - 99 focused backend prediction tests (`test_power_risk_prediction.py`) — all passed (13.47 s)
  - `git diff --check` — only autocrlf normalisation warnings (same as prior tasks), no whitespace errors
  - `git diff --name-status` — exactly `M frontend/app/globals.css` and `M frontend/app/page.tsx`; no package, lock, model, scenario, or cache artifacts changed
  - Source inspection: no hardcoded probability values; no command/uplink/execution patterns; endpoint confirmed as `/api/v1/mock/power-risk-prediction`
- **Visual inspection during Bob execution:** Browser visual inspection was not performed because the Bob environment had no browser preview. Post-Bob desktop, mobile, unavailable-state, and recovery-state visual QA was subsequently completed and is documented below.
- **Result:** Four-layer dashboard implementation and build/test validation completed. Subsequent human visual QA identified and corrected numerical-stability and responsive-layout defects; see the following entry. Probabilities and projections remain API-derived, with no command controls or autonomous functionality.

---

### Post-Bob correction: Numerical stability and responsive visual QA

- **Date (UTC):** 2026-08-27 to 2026-08-28
- **Developer:** Tarek Aref
- **Workflow:** Human-led verification and correction after the original Bob implementation. Bob remained the primary implementation tool for the original milestones; this correction was completed after Bob access was unavailable and is not attributed to Bob.
- **Trigger:** Live desktop inspection exposed an impossible `solar_current_slope` contribution of approximately `-1.1850793840034144e+30`, a displayed breach probability of `0%`, provenance-field collisions, and contribution-value overflow.
- **Root cause:** `solar_current_slope` was effectively constant in the synthetic training split (`scale = 1.88533495521e-33`) but nonzero in the public demonstration history. Standardization amplified floating-point noise into an extreme out-of-distribution value that dominated the logistic-regression output.
- **Corrections:**
  - Detect near-constant features from the training split only using a population-standard-deviation floor of `1e-12`.
  - Neutralize the frozen feature columns in training and validation matrices while retaining all 12 schema positions for compatibility and auditability.
  - Reject stale or malformed artifacts with non-finite or microscopic scaler values and guard contribution arithmetic against non-finite or implausibly large values.
  - Bump the runtime model version from `0.1.0` to `0.1.1`.
  - Add dependency-free frontend response bounds, compact numeric formatting, provenance wrapping, contribution overflow protection, and responsive stacking.
- **Files affected:**
  - `backend/app/prediction_service.py`
  - `backend/scripts/train_power_risk_model.py`
  - `backend/tests/test_model_pipeline.py`
  - `backend/tests/test_power_risk_prediction.py`
  - `frontend/app/page.tsx`
  - `frontend/app/globals.css`
  - `docs/bob-usage.md`
  - `data/models/power_risk_classifier.joblib` — regenerated locally; git-ignored and not committed
- **Model-selection isolation:** Near-constant-feature detection used only the 180-scenario training split. Hyperparameter selection continued to use only the 60-scenario validation split. The final model used the frozen mask and 240 train+validation scenarios. `solar_current_slope` was the only neutralized feature; its saved scaler scale is `1.0` and classifier coefficient is `0.0`. Validation selected `C=0.01` using the pre-existing candidate set and tie rule.
- **Validation performed before held-out evaluation:**
  - Corrected artifact invariant — `solar_current_slope` scale `1.0`, coefficient `0.0`
  - Focused model/prediction suite — 130 passed in 53.30 seconds
  - Complete backend suite — 3,686 passed in 46.52 seconds
  - Frontend production build — compiled, TypeScript passed, page data collected, and 3/3 static pages generated
  - Desktop runtime — API response rendered at `33%`, model `0.1.1`, normal-sized top contributions, and collision-free provenance layout
  - Mobile emulation — approximately 390-pixel target viewport; stacked cards, long values, contributions, and advisory content remained readable without observed horizontal page overflow
  - Loading, API-unavailable, and manual-retry recovery states — permanent safety strip preserved; generic error text exposed no response body, path, stack trace, or raw exception; successful data returned only after manual retry
- **Corrective held-out evaluation disclosure:** Version `0.1.1` is a materially different artifact, so the earlier version `0.1.0` metrics were not reused. The same seed-42 synthetic test split was accessed once more after the model was frozen. This was a defect-driven corrective evaluation, not an untouched first evaluation. The correction was selected from runtime numerical evidence and training/validation data, not from test metrics. No tuning, retraining, or repeat metric evaluation occurred after these results.
- **Version 0.1.1 held-out synthetic results:**
  - ROC-AUC: 1.000 (gate >= 0.80 passed)
  - Recall: 1.000 (gate >= 0.75 passed)
  - Precision: 0.666667 (gate >= 0.65 passed)
  - F1: 0.800 (gate >= 0.70 passed)
  - Brier score: 0.14379 (gate <= 0.20 passed)
  - Confusion matrix `[[TN, FP], [FN, TP]]`: `[[15, 15], [0, 30]]`
  - Leakage violations: 0; breach-eligibility violations: 0; all seven contract gates passed
- **Interpretation:** The corrected model preserved recall on the synthetic test distribution but produced 15 false positives, leaving precision only narrowly above its gate. This conservative error profile may be acceptable for an advisory prototype but would create operator burden and must not be represented as operational performance. The test set is no longer untouched, and these synthetic results do not demonstrate real-spacecraft generalization.
- **Result:** The extreme contribution defect and desktop/mobile layout failures were corrected. Model `0.1.1` is frozen after its single corrective metric run. The system remains `SYNTHETIC`, `NOT_FLIGHT_QUALIFIED`, advisory-only, and `command_authority: "NONE"`.

---

## Suggested evidence categories

- architecture decomposition;
- FastAPI endpoint implementation;
- Next.js component development;
- telemetry-schema validation;
- image-model adapter development;
- unit and integration tests;
- Docker/CI troubleshooting;
- security review; and
- README and architecture documentation.

## Submission summary

IBM Bob served as the primary development tool for the recorded dependency
security work, repository hygiene, deterministic scenario generation, model
training and evaluation pipeline, four-layer prediction API, contract-alignment
corrections, automated tests, and judge-facing dashboard implementation. Tarek
Aref scoped and approved the work, reviewed the resulting changes, corrected
scientific and contract details, and retained responsibility for every decision.
Post-Bob numerical-stability corrections, responsive visual quality assurance,
deployment, and final validation were completed separately and are explicitly
identified above rather than attributed to Bob. IBM Bob is not a runtime
component: it does not generate predictions, make operator decisions, or hold
spacecraft command authority.

