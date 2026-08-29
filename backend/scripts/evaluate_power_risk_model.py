"""Evaluate the trained SpaceBNS power-risk classifier on the held-out test split.

This script:
1. Generates the 300-scenario corpus with seed=42.
2. Extracts the 60-scenario test split.
3. Runs hard validation checks before computing any metric.
4. Loads the serialised pipeline and runs inference.
5. Computes and prints a JSON report to stdout.

Hard validation checks (exit non-zero and print diagnostic if any fail)
-----------------------------------------------------------------------
- Exact split counts (train=180, val=60, test=60).
- Zero scenario-ID overlap between splits.
- Zero breach-eligibility violations in positive test examples.
- Exactly 12 features in FEATURE_ORDER.
- All feature values are finite.
- All predicted probabilities are finite and in [0.0, 1.0].

Scientific limitations
----------------------
- Trained exclusively on synthetic scenarios (seed=42, 300 scenarios).
- Metrics describe performance on the held-out SYNTHETIC scenario distribution.
- Metrics are NOT validated real-spacecraft performance.
- breach_probability is NOT a real-spacecraft failure probability.
- Model is NOT_FLIGHT_QUALIFIED.
- command_authority: NONE — output is advisory and simulation-only.

This script DOES NOT write or modify any file.  It prints its JSON report
to stdout and exits with code 0 on success, non-zero on hard validation failure.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Frozen feature order — imported from the single authoritative definition
# in train_power_risk_model.  Importing that module is side-effect-free:
# it defines constants and functions only; the __main__ block is guarded.
# ---------------------------------------------------------------------------

from backend.scripts.train_power_risk_model import FEATURE_ORDER  # noqa: E402

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

SOC_BREACH_THRESHOLD: float = 25.0
VOLTAGE_BREACH_THRESHOLD: float = 26.0
CLASSIFICATION_THRESHOLD: float = 0.50

# ---------------------------------------------------------------------------
# Model path (default)
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MODEL_PATH: str = os.path.normpath(
    os.path.join(_REPO_ROOT, "data", "models", "power_risk_classifier.joblib")
)

# ---------------------------------------------------------------------------
# Contract gate thresholds (for reporting pass/fail)
# ---------------------------------------------------------------------------

GATE_ROC_AUC_MIN: float = 0.80
GATE_RECALL_MIN: float = 0.75
GATE_PRECISION_MIN: float = 0.65
GATE_F1_MIN: float = 0.70
GATE_BRIER_MAX: float = 0.20


# ---------------------------------------------------------------------------
# Metric helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def _roc_auc_score(y_true: list[int], y_prob: list[float]) -> float:
    """Compute ROC-AUC using the Wilcoxon-Mann-Whitney U statistic.

    AUC = (concordant_pairs + 0.5 * tied_pairs) / (n_pos * n_neg)

    This is numerically equivalent to the trapezoidal area under the ROC curve.
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


def _brier_score(y_true: list[int], y_prob: list[float]) -> float:
    """Mean squared error of predicted probabilities vs true binary labels."""
    n = len(y_true)
    return sum((p - t) ** 2 for p, t in zip(y_prob, y_true)) / n


def _confusion_matrix(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    """Return [[TN, FP], [FN, TP]]."""
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    return [[tn, fp], [fn, tp]]


def _precision_recall_f1(
    y_true: list[int], y_pred: list[int]
) -> tuple[float, float, float]:
    """Precision, recall, F1 for label=1."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Public evaluation function (importable by tests)
# ---------------------------------------------------------------------------

def evaluate_power_risk_model(
    model_path: str = MODEL_PATH,
) -> dict[str, Any]:
    """Load model, validate test split, run inference, compute metrics.

    Parameters
    ----------
    model_path:
        Path to the .joblib pipeline file.

    Returns
    -------
    dict with all reported fields.

    Raises
    ------
    SystemExit(1)
        If any hard validation check fails.  A diagnostic message is printed
        to stderr before exit.
    """
    from backend.scripts.generate_scenarios import generate_training_corpus  # noqa: PLC0415
    from backend.app.features import extract_features                        # noqa: PLC0415
    import joblib                                                             # noqa: PLC0415

    # -----------------------------------------------------------------------
    # 1. Generate corpus and split
    # -----------------------------------------------------------------------
    corpus = generate_training_corpus(seed=42)

    train_scenarios = [s for s in corpus if s["split"] == "train"]
    val_scenarios   = [s for s in corpus if s["split"] == "validation"]
    test_scenarios  = [s for s in corpus if s["split"] == "test"]

    # --- Hard check: exact split counts ---
    errors: list[str] = []
    if len(train_scenarios) != 180:
        errors.append(f"HARD FAIL: train count={len(train_scenarios)}, expected 180")
    if len(val_scenarios) != 60:
        errors.append(f"HARD FAIL: val count={len(val_scenarios)}, expected 60")
    if len(test_scenarios) != 60:
        errors.append(f"HARD FAIL: test count={len(test_scenarios)}, expected 60")
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    # --- Hard check: zero scenario-ID overlap ---
    train_ids = {s["scenario_id"] for s in train_scenarios}
    val_ids   = {s["scenario_id"] for s in val_scenarios}
    test_ids  = {s["scenario_id"] for s in test_scenarios}
    leakage = (
        len(train_ids & val_ids)
        + len(train_ids & test_ids)
        + len(val_ids   & test_ids)
    )
    if leakage != 0:
        print(f"HARD FAIL: scenario-ID leakage violations={leakage}", file=sys.stderr)
        sys.exit(1)

    # --- Hard check: exact class counts per split ---
    train_pos = sum(1 for s in train_scenarios if s["power_constraint_breach_within_24h"] == 1)
    train_neg = len(train_scenarios) - train_pos
    val_pos   = sum(1 for s in val_scenarios   if s["power_constraint_breach_within_24h"] == 1)
    val_neg   = len(val_scenarios) - val_pos
    test_pos  = sum(1 for s in test_scenarios  if s["power_constraint_breach_within_24h"] == 1)
    test_neg  = len(test_scenarios) - test_pos

    class_errors: list[str] = []
    if train_pos != 90 or train_neg != 90:
        class_errors.append(
            f"HARD FAIL: train class counts pos={train_pos}/neg={train_neg}, expected 90/90"
        )
    if val_pos != 30 or val_neg != 30:
        class_errors.append(
            f"HARD FAIL: val class counts pos={val_pos}/neg={val_neg}, expected 30/30"
        )
    if test_pos != 30 or test_neg != 30:
        class_errors.append(
            f"HARD FAIL: test class counts pos={test_pos}/neg={test_neg}, expected 30/30"
        )
    if class_errors:
        for e in class_errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    # --- Hard check: breach eligibility in positive test examples ---
    breach_violations = 0
    for s in test_scenarios:
        if s["power_constraint_breach_within_24h"] == 1:
            last = s["history"][-1]
            if (
                last["battery_soc_percent"] < SOC_BREACH_THRESHOLD
                or last["bus_voltage_v"] < VOLTAGE_BREACH_THRESHOLD
            ):
                breach_violations += 1
    if breach_violations != 0:
        print(
            f"HARD FAIL: breach-eligibility violations={breach_violations}",
            file=sys.stderr,
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 2. Build feature matrix for test split
    # -----------------------------------------------------------------------
    X_test: list[list[float]] = []
    y_test: list[int] = []

    for s in test_scenarios:
        feat_dict = extract_features(s["history"])

        # --- Hard check: exact 12 features in FEATURE_ORDER ---
        keys_in_order = list(feat_dict.keys())
        if keys_in_order != FEATURE_ORDER:
            print(
                f"HARD FAIL: feature keys mismatch.\n"
                f"  Got:      {keys_in_order}\n"
                f"  Expected: {FEATURE_ORDER}",
                file=sys.stderr,
            )
            sys.exit(1)

        vec = [feat_dict[k] for k in FEATURE_ORDER]

        # --- Hard check: finite feature values ---
        for k, v in zip(FEATURE_ORDER, vec):
            if not math.isfinite(v):
                print(
                    f"HARD FAIL: non-finite feature {k}={v} "
                    f"in scenario {s['scenario_id']}",
                    file=sys.stderr,
                )
                sys.exit(1)

        X_test.append(vec)
        y_test.append(int(s["power_constraint_breach_within_24h"]))

    # -----------------------------------------------------------------------
    # 3. Load model and run inference
    # -----------------------------------------------------------------------
    pipeline = joblib.load(model_path)

    proba_matrix = pipeline.predict_proba(X_test)
    y_prob: list[float] = [float(row[1]) for row in proba_matrix]
    y_pred: list[int]   = [
        1 if p >= CLASSIFICATION_THRESHOLD else 0 for p in y_prob
    ]

    # --- Hard check: finite probabilities in [0, 1] ---
    for i, p in enumerate(y_prob):
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            print(
                f"HARD FAIL: invalid probability={p} at test index {i}",
                file=sys.stderr,
            )
            sys.exit(1)

    # -----------------------------------------------------------------------
    # 4. Compute metrics
    # -----------------------------------------------------------------------
    roc_auc = _roc_auc_score(y_test, y_prob)
    brier   = _brier_score(y_test, y_prob)
    precision, recall, f1 = _precision_recall_f1(y_test, y_pred)
    cm = _confusion_matrix(y_test, y_pred)

    # (train_pos/neg, val_pos/neg, test_pos/neg already computed and validated above)

    # -----------------------------------------------------------------------
    # 5. Contract gate pass/fail
    # -----------------------------------------------------------------------
    gates = {
        "roc_auc_ge_0.80":            roc_auc   >= GATE_ROC_AUC_MIN,
        "recall_ge_0.75":             recall    >= GATE_RECALL_MIN,
        "precision_ge_0.65":          precision >= GATE_PRECISION_MIN,
        "f1_ge_0.70":                 f1        >= GATE_F1_MIN,
        "brier_le_0.20":              brier     <= GATE_BRIER_MAX,
        "leakage_violations_eq_0":    leakage == 0,
        "breach_eligibility_eq_0":    breach_violations == 0,
    }
    all_gates_pass = all(gates.values())

    report: dict[str, Any] = {
        "scientific_disclosure": {
            "training_data":           "SYNTHETIC_ONLY — seed=42, 300 scenarios",
            "metrics_represent":       "held-out synthetic scenario distribution",
            "not_validated_against":   "real spacecraft",
            "breach_probability_note": (
                "NOT a real-spacecraft failure probability; "
                "estimated from synthetic scenario distribution only"
            ),
            "prototype_status":        "NOT_FLIGHT_QUALIFIED",
            "command_authority":       "NONE",
            "advisory_status":         "advisory and simulation-only",
        },
        "split_counts": {
            "train":      len(train_scenarios),
            "validation": len(val_scenarios),
            "test":       len(test_scenarios),
        },
        "label_counts": {
            "train_pos":  train_pos,
            "train_neg":  train_neg,
            "val_pos":    val_pos,
            "val_neg":    val_neg,
            "test_pos":   test_pos,
            "test_neg":   test_neg,
        },
        "leakage_count":               leakage,
        "breach_eligibility_violations": breach_violations,
        "metrics": {
            "roc_auc":   round(roc_auc, 6),
            "recall":    round(recall, 6),
            "precision": round(precision, 6),
            "f1":        round(f1, 6),
            "brier":     round(brier, 6),
        },
        "confusion_matrix": cm,
        "contract_gates":   gates,
        "all_gates_pass":   all_gates_pass,
    }

    if not all_gates_pass:
        failed = [k for k, v in gates.items() if not v]
        report["gate_failures"] = failed

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    report = evaluate_power_risk_model()
    print(json.dumps(report, indent=2))

    if not report["all_gates_pass"]:
        print("\nCONTRACT GATE FAILURES:", report.get("gate_failures"), file=sys.stderr)
        sys.exit(2)
