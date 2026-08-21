"""Train the SpaceBNS power-risk classifier.

Pipeline
--------
    StandardScaler → LogisticRegression (scikit-learn)

Process
-------
1. Generate the 300-scenario corpus with seed=42.
2. Split into train (180), validation (60), and test (60) using the pre-assigned
   scenario["split"] field.  Only the test scenario count and scenario IDs are
   checked for split-separation; test labels, features, probabilities, and
   metrics are not inspected during training.
3. Extract 12 features from each scenario's history window.
4. Select regularisation strength C from [0.01, 0.1, 1.0, 10.0] by evaluating
   ROC-AUC on the validation split only.
5. Train the final pipeline on train + validation (240 scenarios) using the best C.
6. Serialise the pipeline to data/models/power_risk_classifier.joblib.
7. Print a JSON summary to stdout.

Scientific limitations
----------------------
- Trained exclusively on synthetic scenarios (seed=42, 300 scenarios).
- Metrics describe performance on the held-out synthetic scenario distribution.
- Metrics are NOT validated real-spacecraft performance.
- breach_probability is NOT a real-spacecraft failure probability.
- Model is NOT_FLIGHT_QUALIFIED.
- command_authority: NONE — output is advisory and simulation-only.
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from typing import Any

# ---------------------------------------------------------------------------
# Frozen feature order (Section 6 of docs/power-risk-contract.md)
# ---------------------------------------------------------------------------

FEATURE_ORDER: list[str] = [
    "soc_latest",
    "soc_mean",
    "soc_min",
    "soc_slope",
    "voltage_latest",
    "voltage_min",
    "voltage_slope",
    "solar_current_mean",
    "solar_current_slope",
    "payload_draw_mean",
    "payload_draw_max",
    "high_draw_fraction",
]

# ---------------------------------------------------------------------------
# Hyperparameter candidates (contract Section 7)
# ---------------------------------------------------------------------------

C_CANDIDATES: list[float] = [0.01, 0.1, 1.0, 10.0]

# ---------------------------------------------------------------------------
# Breach-eligibility thresholds (contract Section 5, Label eligibility)
# ---------------------------------------------------------------------------

SOC_BREACH_THRESHOLD: float = 25.0   # %
VOLTAGE_BREACH_THRESHOLD: float = 26.0  # V

# ---------------------------------------------------------------------------
# Model output path
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MODEL_PATH: str = os.path.normpath(
    os.path.join(_REPO_ROOT, "data", "models", "power_risk_classifier.joblib")
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_breach_eligibility(scenario: dict[str, Any]) -> bool:
    """Return True if the scenario is breach-eligible (no active breach at end of history).

    A positive example is eligible only when the final history sample does NOT
    already trigger BATTERY_SOC_LOW (< 25 %) or BUS_VOLTAGE_LOW (< 26 V).
    """
    last = scenario["history"][-1]
    soc_active = last["battery_soc_percent"] < SOC_BREACH_THRESHOLD
    voltage_active = last["bus_voltage_v"] < VOLTAGE_BREACH_THRESHOLD
    return not (soc_active or voltage_active)


def _extract_feature_vector(scenario: dict[str, Any]) -> list[float]:
    """Return a length-12 feature vector in FEATURE_ORDER from scenario["history"].

    Raises
    ------
    ValueError
        If the keys returned by extract_features do not exactly match
        FEATURE_ORDER (wrong set or wrong order).
    """
    from backend.app.features import extract_features  # noqa: PLC0415

    feat_dict = extract_features(scenario["history"])

    keys_actual = list(feat_dict.keys())
    if keys_actual != FEATURE_ORDER:
        raise ValueError(
            f"Feature order mismatch in scenario {scenario.get('scenario_id', '?')}.\n"
            f"  Expected: {FEATURE_ORDER}\n"
            f"  Got:      {keys_actual}"
        )
    return [feat_dict[k] for k in FEATURE_ORDER]


def _build_matrices(
    scenarios: list[dict[str, Any]],
) -> tuple[list[list[float]], list[int]]:
    """Build X (feature matrix) and y (label vector) for a list of scenarios."""
    X: list[list[float]] = []
    y: list[int] = []
    for s in scenarios:
        X.append(_extract_feature_vector(s))
        y.append(int(s["power_constraint_breach_within_24h"]))
    return X, y


def _roc_auc_score(y_true: list[int], y_prob: list[float]) -> float:
    """Compute ROC-AUC using the Wilcoxon-Mann-Whitney U statistic.

    AUC = (number of (pos, neg) pairs where pos_prob > neg_prob
           + 0.5 * number of tied pairs) / (n_pos * n_neg)

    This is numerically equivalent to the trapezoidal-rule area under the
    ROC curve and avoids the off-by-one accumulation complexity.
    """
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    pos_probs = [p for p, t in zip(y_prob, y_true) if t == 1]
    neg_probs = [p for p, t in zip(y_prob, y_true) if t == 0]

    concordant = 0
    tied = 0
    for pp in pos_probs:
        for np_ in neg_probs:
            if pp > np_:
                concordant += 1
            elif pp == np_:
                tied += 1

    return (concordant + 0.5 * tied) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Public training function (importable by tests)
# ---------------------------------------------------------------------------

def train_power_risk_model(
    save_path: str = MODEL_PATH,
) -> dict[str, Any]:
    """Generate corpus, select C, train final pipeline, serialise and return summary.

    Parameters
    ----------
    save_path:
        Destination .joblib path.  Use a temporary path in tests.

    Returns
    -------
    dict with keys:
        selected_c, val_roc_auc_by_c, train_count, val_count, test_count,
        train_pos, train_neg, val_pos, val_neg,
        leakage_count, breach_eligibility_violations,
        model_path

    Note: test labels, features, probabilities, and metrics are deliberately excluded.
    They belong to the evaluation stage (evaluate_power_risk_model.py).
    Only the test split's count and scenario-ID separation are checked during
    training; its labels, features, probabilities, and metrics are not inspected.
    """
    # -----------------------------------------------------------------------
    # 1. Generate corpus
    # -----------------------------------------------------------------------
    from backend.scripts.generate_scenarios import generate_training_corpus  # noqa: PLC0415

    corpus = generate_training_corpus(seed=42)

    # -----------------------------------------------------------------------
    # 2. Split
    # -----------------------------------------------------------------------
    train_scenarios = [s for s in corpus if s["split"] == "train"]
    val_scenarios   = [s for s in corpus if s["split"] == "validation"]
    test_scenarios  = [s for s in corpus if s["split"] == "test"]

    # Verify exact counts
    assert len(train_scenarios) == 180, f"train={len(train_scenarios)}"
    assert len(val_scenarios)   == 60,  f"val={len(val_scenarios)}"
    assert len(test_scenarios)  == 60,  f"test={len(test_scenarios)}"

    # Verify no scenario_id overlap between splits
    train_ids = {s["scenario_id"] for s in train_scenarios}
    val_ids   = {s["scenario_id"] for s in val_scenarios}
    test_ids  = {s["scenario_id"] for s in test_scenarios}
    leakage = len(train_ids & val_ids) + len(train_ids & test_ids) + len(val_ids & test_ids)
    assert leakage == 0, f"Leakage violations: {leakage}"

    # Verify train/val label balance (test labels are not inspected here)
    train_pos = sum(1 for s in train_scenarios if s["power_constraint_breach_within_24h"] == 1)
    train_neg = len(train_scenarios) - train_pos
    val_pos   = sum(1 for s in val_scenarios   if s["power_constraint_breach_within_24h"] == 1)
    val_neg   = len(val_scenarios) - val_pos
    assert train_pos == 90 and train_neg == 90, f"train balance: {train_pos}/{train_neg}"
    assert val_pos == 30   and val_neg == 30,   f"val balance: {val_pos}/{val_neg}"

    # Verify breach eligibility for positive examples in train+val
    breach_violations = 0
    for s in train_scenarios + val_scenarios:
        if s["power_constraint_breach_within_24h"] == 1:
            if not _check_breach_eligibility(s):
                breach_violations += 1
    assert breach_violations == 0, f"Breach eligibility violations: {breach_violations}"

    # -----------------------------------------------------------------------
    # 3. Build feature matrices (train and validation only; test is untouched)
    # -----------------------------------------------------------------------
    X_train, y_train = _build_matrices(train_scenarios)
    X_val,   y_val   = _build_matrices(val_scenarios)

    # -----------------------------------------------------------------------
    # 4. Hyperparameter selection on validation split
    # -----------------------------------------------------------------------
    from sklearn.exceptions import ConvergenceWarning   # noqa: PLC0415
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.pipeline import Pipeline               # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler    # noqa: PLC0415

    val_roc_auc_by_c: dict[float, float] = {}
    best_c: float = C_CANDIDATES[0]
    best_auc: float = -1.0

    for c in C_CANDIDATES:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(C=c, random_state=42, max_iter=1000)),
        ])
        with warnings.catch_warnings():
            # Treat convergence warnings as failures (contract requirement).
            # Catch only ConvergenceWarning; ignore unrelated scipy/solver warnings.
            warnings.filterwarnings("error", category=ConvergenceWarning)
            pipe.fit(X_train, y_train)

        probs = [p[1] for p in pipe.predict_proba(X_val)]
        auc = _roc_auc_score(y_val, probs)
        val_roc_auc_by_c[c] = auc

        # Highest AUC wins; ties go to smallest C (deterministic regularisation)
        if auc > best_auc or (math.isclose(auc, best_auc, rel_tol=1e-12) and c < best_c):
            best_auc = auc
            best_c = c

    # -----------------------------------------------------------------------
    # 5. Train final pipeline on train + validation (240 scenarios)
    # -----------------------------------------------------------------------
    X_trainval = X_train + X_val
    y_trainval = y_train + y_val
    assert len(X_trainval) == 240, f"train+val={len(X_trainval)}"

    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=best_c, random_state=42, max_iter=1000)),
    ])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        final_pipeline.fit(X_trainval, y_trainval)

    # -----------------------------------------------------------------------
    # 6. Serialise
    # -----------------------------------------------------------------------
    import joblib  # noqa: PLC0415

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(final_pipeline, save_path)

    return {
        "selected_c":               best_c,
        "val_roc_auc_by_c":         val_roc_auc_by_c,
        "train_count":              len(train_scenarios),
        "val_count":                len(val_scenarios),
        "test_count":               len(test_scenarios),
        "train_pos":                train_pos,
        "train_neg":                train_neg,
        "val_pos":                  val_pos,
        "val_neg":                  val_neg,
        "leakage_count":            leakage,
        "breach_eligibility_violations": breach_violations,
        "model_path":               save_path,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = train_power_risk_model()
    print(json.dumps(result, indent=2))
