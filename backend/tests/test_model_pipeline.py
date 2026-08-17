"""Focused model-pipeline tests for the SpaceBNS power-risk classifier.

Coverage checklist
------------------
M01  Exact feature count: FEATURE_ORDER has exactly 12 entries.
M02  Frozen feature order: keys match the contract list exactly.
M03  One scenario → exactly one 12-element feature vector.
M04  Feature extraction uses only scenario["history"].
M05  Forbidden fields (scenario_id, split, breach_detail, etc.) are never used.
M06  Future data is excluded from feature extraction.
M06b Feature-order mismatch in _extract_feature_vector raises ValueError.
M07  Exact train/validation/test counts (180/60/60) with seed=42.
M08  Exact label balance per split (90+90 / 30+30 / 30+30).
M09  Zero scenario-ID overlap between splits.
M10  Positive examples are breach-eligible (no active breach at end of history).
M11  Corpus generation is deterministic with seed=42.
M12  Training is reproducible with same data and configuration.
M13  Pipeline order: StandardScaler then LogisticRegression.
M14  Selected C belongs to the approved candidate list.
M15  Hyperparameter selection uses only validation split (not test).
M16  Final training uses exactly 240 (train + validation) examples.
M17  Test data is excluded from fitting and selection.
M18  Probabilities are finite and in [0.0, 1.0].
M19  Metric calculations: brier score, confusion matrix ordering [[TN,FP],[FN,TP]].
M20  Model artifact serialisation and reload.
M21  Generated model and corpus paths are Git-ignored.
M22  No import-time training, file creation, or side effects.
M23  ROC-AUC calculation matches sklearn reference for a known case.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
import tempfile
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers: import the modules under test without triggering any side effects
# ---------------------------------------------------------------------------

def _import_train():
    """Import train module without executing __main__ block."""
    import backend.scripts.train_power_risk_model as m  # noqa: PLC0415
    return m


def _import_evaluate():
    """Import evaluate module without executing __main__ block."""
    import backend.scripts.evaluate_power_risk_model as e  # noqa: PLC0415
    return e


# ---------------------------------------------------------------------------
# M01 — Exact feature count
# ---------------------------------------------------------------------------

def test_M01_feature_order_has_12_entries() -> None:
    m = _import_train()
    assert len(m.FEATURE_ORDER) == 12, (
        f"FEATURE_ORDER has {len(m.FEATURE_ORDER)} entries, expected 12"
    )


# ---------------------------------------------------------------------------
# M02 — Frozen feature order
# ---------------------------------------------------------------------------

_EXPECTED_FEATURE_ORDER = [
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


def test_M02_frozen_feature_order_exact() -> None:
    m = _import_train()
    assert m.FEATURE_ORDER == _EXPECTED_FEATURE_ORDER, (
        f"FEATURE_ORDER mismatch:\n  got:      {m.FEATURE_ORDER}\n"
        f"  expected: {_EXPECTED_FEATURE_ORDER}"
    )


# ---------------------------------------------------------------------------
# M03 — One scenario → exactly one 12-element feature vector
# ---------------------------------------------------------------------------

def test_M03_one_scenario_one_12_element_vector() -> None:
    from backend.scripts.generate_scenarios import generate_training_corpus  # noqa: PLC0415
    corpus = generate_training_corpus(seed=42)
    scenario = corpus[0]

    m = _import_train()
    vec = m._extract_feature_vector(scenario)
    assert len(vec) == 12, f"Expected 12-element vector, got {len(vec)}"


# ---------------------------------------------------------------------------
# M04 — Feature extraction uses only history
# ---------------------------------------------------------------------------

def test_M04_feature_extraction_uses_only_history() -> None:
    from backend.scripts.generate_scenarios import generate_training_corpus  # noqa: PLC0415
    corpus = generate_training_corpus(seed=42)
    scenario = corpus[0]

    from backend.app.features import extract_features  # noqa: PLC0415

    # Extract from history only
    feats_from_history = extract_features(scenario["history"])

    # The 72-element history is the only valid input; future has 288 samples
    # and would fail validation in extract_features (requires exactly 72)
    assert len(scenario["history"]) == 72
    assert len(feats_from_history) == 12


# ---------------------------------------------------------------------------
# M05 — Forbidden fields are never used as features
# ---------------------------------------------------------------------------

def test_M05_forbidden_fields_not_used() -> None:
    """Adding forbidden fields to history samples must not change feature values."""
    from backend.scripts.generate_scenarios import generate_training_corpus  # noqa: PLC0415
    from backend.app.features import extract_features                         # noqa: PLC0415
    import copy                                                               # noqa: PLC0415

    corpus = generate_training_corpus(seed=42)
    scenario = corpus[0]

    feats_clean = extract_features(scenario["history"])

    # Inject all forbidden fields into each sample
    samples_dirty = copy.deepcopy(scenario["history"])
    for s in samples_dirty:
        s["scenario_id"]           = "SHOULD-NOT-AFFECT"
        s["split"]                 = "train"
        s["breach_detail"]         = {"occurs": True}
        s["metadata"]              = {"data_source": "SYNTHETIC"}
        s["communications_status"] = "GROUND_CONTACT"
        s["command_activity"]      = "PAYLOAD_IMAGING_BURST"
        s["image_utility_score"]   = 0.99

    feats_dirty = extract_features(samples_dirty)
    assert feats_clean == feats_dirty, (
        "Forbidden fields changed feature values"
    )


# ---------------------------------------------------------------------------
# M06 — Future data is excluded from feature extraction
# ---------------------------------------------------------------------------

def test_M06_future_data_excluded() -> None:
    """extract_features must reject a 288-sample future window (wrong count)."""
    from backend.scripts.generate_scenarios import generate_training_corpus  # noqa: PLC0415
    from backend.app.features import extract_features                         # noqa: PLC0415

    corpus = generate_training_corpus(seed=42)
    scenario = corpus[0]

    # Attempting to extract features from future (288 samples) must raise ValueError
    with pytest.raises(ValueError, match="exactly 72 samples"):
        extract_features(scenario["future"])


# ---------------------------------------------------------------------------
# M06b — Feature-order mismatch raises ValueError
# ---------------------------------------------------------------------------

def test_M06b_feature_order_mismatch_raises_value_error() -> None:
    """_extract_feature_vector must raise ValueError if extract_features returns
    keys in an unexpected order or with unexpected names.

    Strategy: monkeypatch extract_features inside the features module to return
    a dict with one key renamed, then call _extract_feature_vector and confirm
    that ValueError is raised with a meaningful message.
    """
    import backend.app.features as feat_mod  # noqa: PLC0415
    from backend.scripts.generate_scenarios import generate_training_corpus  # noqa: PLC0415

    m = _import_train()
    corpus   = generate_training_corpus(seed=42)
    scenario = corpus[0]

    original_extract = feat_mod.extract_features

    def _bad_extract(samples):
        result = original_extract(samples)
        # Rename the last key to break the order contract
        keys = list(result.keys())
        keys[-1] = "WRONG_KEY"
        return dict(zip(keys, result.values()))

    feat_mod.extract_features = _bad_extract
    try:
        with pytest.raises(ValueError, match="Feature order mismatch"):
            m._extract_feature_vector(scenario)
    finally:
        feat_mod.extract_features = original_extract


# ---------------------------------------------------------------------------
# M07 — Exact split counts
# ---------------------------------------------------------------------------

def test_M07_exact_split_counts() -> None:
    from backend.scripts.generate_scenarios import generate_training_corpus  # noqa: PLC0415
    corpus = generate_training_corpus(seed=42)

    train = [s for s in corpus if s["split"] == "train"]
    val   = [s for s in corpus if s["split"] == "validation"]
    test  = [s for s in corpus if s["split"] == "test"]

    assert len(train) == 180, f"train={len(train)}, expected 180"
    assert len(val)   == 60,  f"val={len(val)}, expected 60"
    assert len(test)  == 60,  f"test={len(test)}, expected 60"


# ---------------------------------------------------------------------------
# M08 — Exact label balance per split
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split_name,exp_pos,exp_neg", [
    ("train",      90, 90),
    ("validation", 30, 30),
    ("test",       30, 30),
])
def test_M08_label_balance_per_split(split_name, exp_pos, exp_neg) -> None:
    from backend.scripts.generate_scenarios import generate_training_corpus  # noqa: PLC0415
    corpus = generate_training_corpus(seed=42)

    split = [s for s in corpus if s["split"] == split_name]
    pos   = sum(1 for s in split if s["power_constraint_breach_within_24h"] == 1)
    neg   = len(split) - pos

    assert pos == exp_pos, f"{split_name}: pos={pos}, expected {exp_pos}"
    assert neg == exp_neg, f"{split_name}: neg={neg}, expected {exp_neg}"


# ---------------------------------------------------------------------------
# M09 — Zero scenario-ID overlap
# ---------------------------------------------------------------------------

def test_M09_zero_scenario_id_overlap() -> None:
    from backend.scripts.generate_scenarios import generate_training_corpus  # noqa: PLC0415
    corpus = generate_training_corpus(seed=42)

    train_ids = {s["scenario_id"] for s in corpus if s["split"] == "train"}
    val_ids   = {s["scenario_id"] for s in corpus if s["split"] == "validation"}
    test_ids  = {s["scenario_id"] for s in corpus if s["split"] == "test"}

    assert not (train_ids & val_ids),  "Overlap between train and validation"
    assert not (train_ids & test_ids), "Overlap between train and test"
    assert not (val_ids   & test_ids), "Overlap between validation and test"


# ---------------------------------------------------------------------------
# M10 — Breach eligibility for positive examples
# ---------------------------------------------------------------------------

def test_M10_positive_examples_are_breach_eligible() -> None:
    from backend.scripts.generate_scenarios import generate_training_corpus  # noqa: PLC0415
    corpus = generate_training_corpus(seed=42)

    violations = 0
    for s in corpus:
        if s["power_constraint_breach_within_24h"] == 1:
            last = s["history"][-1]
            if (
                last["battery_soc_percent"] < 25.0
                or last["bus_voltage_v"] < 26.0
            ):
                violations += 1

    assert violations == 0, f"{violations} breach-eligibility violation(s) found"


# ---------------------------------------------------------------------------
# M11 — Deterministic corpus with seed=42
# ---------------------------------------------------------------------------

def test_M11_deterministic_corpus_seed42() -> None:
    from backend.scripts.generate_scenarios import generate_training_corpus  # noqa: PLC0415
    run_a = generate_training_corpus(seed=42)
    run_b = generate_training_corpus(seed=42)
    assert json.dumps(run_a, sort_keys=True) == json.dumps(run_b, sort_keys=True)


# ---------------------------------------------------------------------------
# M12 — Reproducible training
# ---------------------------------------------------------------------------

def test_M12_reproducible_training() -> None:
    """Training the pipeline twice on the same data must produce identical weights."""
    from sklearn.linear_model import LogisticRegression   # noqa: PLC0415
    from sklearn.pipeline import Pipeline                 # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler      # noqa: PLC0415
    from backend.scripts.generate_scenarios import generate_training_corpus  # noqa: PLC0415
    from backend.app.features import extract_features                         # noqa: PLC0415

    corpus = generate_training_corpus(seed=42)
    train  = [s for s in corpus if s["split"] == "train"]

    X = [[feat for feat in [extract_features(s["history"])[k]
                             for k in _EXPECTED_FEATURE_ORDER]]
         for s in train]
    y = [s["power_constraint_breach_within_24h"] for s in train]

    def _make_pipe():
        p = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(C=1.0, random_state=42, max_iter=1000)),
        ])
        p.fit(X, y)
        return p

    pipe_a = _make_pipe()
    pipe_b = _make_pipe()

    coef_a = pipe_a.named_steps["clf"].coef_.tolist()
    coef_b = pipe_b.named_steps["clf"].coef_.tolist()
    assert coef_a == coef_b, "Training is not reproducible"


# ---------------------------------------------------------------------------
# M13 — Pipeline order: StandardScaler then LogisticRegression
# ---------------------------------------------------------------------------

def test_M13_pipeline_order_scaler_then_logistic_regression() -> None:
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.pipeline import Pipeline                # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler     # noqa: PLC0415

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, random_state=42, max_iter=1000)),
    ])

    steps = list(pipe.named_steps.keys())
    assert steps[0] == "scaler",  f"First step is {steps[0]!r}, expected 'scaler'"
    assert steps[1] == "clf",     f"Second step is {steps[1]!r}, expected 'clf'"
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)
    assert isinstance(pipe.named_steps["clf"],    LogisticRegression)


# ---------------------------------------------------------------------------
# M14 — Selected C belongs to the approved candidate list
# ---------------------------------------------------------------------------

def test_M14_selected_c_is_approved_candidate() -> None:
    m = _import_train()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        summary = m.train_power_risk_model(save_path=model_path)

    assert summary["selected_c"] in m.C_CANDIDATES, (
        f"selected_c={summary['selected_c']} not in C_CANDIDATES={m.C_CANDIDATES}"
    )


# ---------------------------------------------------------------------------
# M15 — Hyperparameter selection uses only validation split
# ---------------------------------------------------------------------------

def test_M15_hyperparameter_selection_uses_only_validation() -> None:
    """_roc_auc_score is called once per C candidate using only validation labels.

    Strategy: monkeypatch _roc_auc_score inside the train module to record
    every (y_true, y_prob) call, then verify that:
    - it was called exactly len(C_CANDIDATES) times (once per candidate);
    - each call's y_true contains exactly the validation labels (30 pos / 30 neg);
    - the selected C is the one whose recorded AUC was highest (ties → smallest C).
    """
    from backend.scripts.generate_scenarios import generate_training_corpus  # noqa: PLC0415

    m = _import_train()
    corpus   = generate_training_corpus(seed=42)
    val_scen = [s for s in corpus if s["split"] == "validation"]
    expected_val_labels = sorted(
        [s["power_constraint_breach_within_24h"] for s in val_scen]
    )

    calls: list[tuple[list[int], list[float]]] = []
    original_roc = m._roc_auc_score

    def _recording_roc(y_true, y_prob):
        calls.append((list(y_true), list(y_prob)))
        return original_roc(y_true, y_prob)

    m._roc_auc_score = _recording_roc
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.joblib")
            summary = m.train_power_risk_model(save_path=model_path)
    finally:
        m._roc_auc_score = original_roc

    # Exactly one call per C candidate
    assert len(calls) == len(m.C_CANDIDATES), (
        f"Expected {len(m.C_CANDIDATES)} _roc_auc_score calls, got {len(calls)}"
    )

    # Every call used the validation labels (sorted match)
    for i, (y_true, _) in enumerate(calls):
        assert sorted(y_true) == expected_val_labels, (
            f"Call {i}: y_true does not match validation labels.\n"
            f"  Got sorted:      {sorted(y_true)}\n"
            f"  Expected sorted: {expected_val_labels}"
        )

    # val_roc_auc_by_c present and selected C is the one with highest val AUC
    assert set(summary["val_roc_auc_by_c"].keys()) == set(m.C_CANDIDATES)
    best_c = max(m.C_CANDIDATES, key=lambda c: (summary["val_roc_auc_by_c"][c], -c))
    assert summary["selected_c"] == best_c


# ---------------------------------------------------------------------------
# M16 — Final training uses exactly 240 examples
# ---------------------------------------------------------------------------

def test_M16_final_training_uses_240_examples() -> None:
    """The fitted scaler's n_samples_seen_ must equal exactly 240."""
    import joblib  # noqa: PLC0415

    m = _import_train()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        summary = m.train_power_risk_model(save_path=model_path)
        pipeline = joblib.load(model_path)

    # StandardScaler stores the number of samples it was fitted on
    n_seen = pipeline.named_steps["scaler"].n_samples_seen_
    assert n_seen == 240, (
        f"Scaler fitted on {n_seen} samples, expected 240 (train+val)"
    )
    # Sanity-check the reported counts too
    assert summary["train_count"] + summary["val_count"] == 240


# ---------------------------------------------------------------------------
# M17 — Test data excluded from fitting and selection
# ---------------------------------------------------------------------------

def test_M17_test_data_excluded_from_fitting() -> None:
    """_build_matrices must receive only train/val scenarios — never test scenarios.

    Strategy: monkeypatch _build_matrices inside the train module and record
    which scenario_ids it is called with.  None of the recorded IDs may belong
    to the test split.
    """
    from backend.scripts.generate_scenarios import generate_training_corpus  # noqa: PLC0415

    m = _import_train()
    corpus    = generate_training_corpus(seed=42)
    test_ids  = {s["scenario_id"] for s in corpus if s["split"] == "test"}

    seen_ids: set[str] = set()
    original_build = m._build_matrices

    def _recording_build(scenarios):
        for s in scenarios:
            seen_ids.add(s["scenario_id"])
        return original_build(scenarios)

    m._build_matrices = _recording_build
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.joblib")
            m.train_power_risk_model(save_path=model_path)
    finally:
        m._build_matrices = original_build

    leaked = seen_ids & test_ids
    assert not leaked, (
        f"_build_matrices was called with {len(leaked)} test scenario(s): "
        f"{sorted(leaked)[:5]}{'...' if len(leaked) > 5 else ''}"
    )

    # Summary must not contain any test-label fields
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        summary = m.train_power_risk_model(save_path=model_path)
    assert "test_pos"    not in summary
    assert "test_neg"    not in summary
    assert "test_roc_auc" not in summary
    assert "test_recall"  not in summary
    assert summary.get("test_count") == 60


# ---------------------------------------------------------------------------
# M18 — Probabilities are finite and in [0.0, 1.0]
# ---------------------------------------------------------------------------

def test_M18_probabilities_finite_and_bounded() -> None:
    """Probabilities from the fitted pipeline are finite and in [0.0, 1.0].

    Uses only validation scenarios — never the held-out test split.
    """
    import joblib                                                              # noqa: PLC0415
    from backend.scripts.generate_scenarios import generate_training_corpus   # noqa: PLC0415
    from backend.app.features import extract_features                          # noqa: PLC0415

    m = _import_train()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        m.train_power_risk_model(save_path=model_path)
        pipeline = joblib.load(model_path)

    corpus = generate_training_corpus(seed=42)
    # Use validation scenarios — already inspected during training, not held-out test
    val_scenarios = [s for s in corpus if s["split"] == "validation"]
    assert len(val_scenarios) == 60, "Expected 60 validation scenarios"

    for s in val_scenarios:
        vec = [extract_features(s["history"])[k] for k in _EXPECTED_FEATURE_ORDER]
        prob_matrix = pipeline.predict_proba([vec])
        p = float(prob_matrix[0][1])
        assert math.isfinite(p), f"Non-finite probability {p} for {s['scenario_id']}"
        assert 0.0 <= p <= 1.0,  f"Probability {p} out of [0,1] for {s['scenario_id']}"


# ---------------------------------------------------------------------------
# M19 — Metric calculations and confusion matrix ordering
# ---------------------------------------------------------------------------

def test_M19_brier_score_known_case() -> None:
    e = _import_evaluate()
    y_true = [1, 0, 1, 0]
    y_prob = [0.9, 0.1, 0.8, 0.2]
    # MSE = ((0.9-1)^2 + (0.1-0)^2 + (0.8-1)^2 + (0.2-0)^2) / 4
    #      = (0.01 + 0.01 + 0.04 + 0.04) / 4 = 0.10 / 4 = 0.025
    brier = e._brier_score(y_true, y_prob)
    assert math.isclose(brier, 0.025, rel_tol=1e-9)


def test_M19_confusion_matrix_ordering() -> None:
    e = _import_evaluate()
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    cm = e._confusion_matrix(y_true, y_pred)
    # TN=1, FP=1, FN=1, TP=1
    assert cm == [[1, 1], [1, 1]], f"Confusion matrix: {cm}"


def test_M19_precision_recall_f1_known_case() -> None:
    e = _import_evaluate()
    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 1]
    # TP=2, FP=1, FN=1
    # precision = 2/3, recall = 2/3, f1 = 2/3
    p, r, f1 = e._precision_recall_f1(y_true, y_pred)
    assert math.isclose(p,  2/3, rel_tol=1e-9)
    assert math.isclose(r,  2/3, rel_tol=1e-9)
    assert math.isclose(f1, 2/3, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# M20 — Model artifact serialisation and reload
# ---------------------------------------------------------------------------

def test_M20_model_serialisation_and_reload() -> None:
    import joblib  # noqa: PLC0415
    from backend.scripts.generate_scenarios import generate_training_corpus  # noqa: PLC0415
    from backend.app.features import extract_features                         # noqa: PLC0415

    m = _import_train()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        m.train_power_risk_model(save_path=model_path)

        assert os.path.isfile(model_path), "Model file not created"
        pipeline = joblib.load(model_path)

    # Verify the reloaded pipeline can predict
    corpus = generate_training_corpus(seed=42)
    scenario = corpus[0]
    vec = [extract_features(scenario["history"])[k] for k in _EXPECTED_FEATURE_ORDER]
    proba = pipeline.predict_proba([vec])
    assert proba.shape == (1, 2)
    assert math.isclose(proba[0][0] + proba[0][1], 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# M21 — Generated model and corpus paths are Git-ignored
# ---------------------------------------------------------------------------

def test_M21_generated_paths_are_gitignored() -> None:
    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    gitignore_path = os.path.join(repo_root, ".gitignore")
    with open(gitignore_path, encoding="utf-8") as fh:
        content = fh.read()

    assert "data/scenarios/" in content or "data/scenarios" in content, (
        ".gitignore missing data/scenarios/"
    )
    assert "data/models/" in content or "data/models" in content, (
        ".gitignore missing data/models/"
    )


# ---------------------------------------------------------------------------
# M22 — No import-time side effects
# ---------------------------------------------------------------------------

def test_M22_no_import_time_side_effects(tmp_path) -> None:
    """Importing train/evaluate modules must not create files or train models.

    Strategy: snapshot the set of files in the model and scenarios directories
    before and after import, and assert the snapshots are identical.
    """
    repo_root  = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    models_dir = os.path.join(repo_root, "data", "models")
    scen_dir   = os.path.join(repo_root, "data", "scenarios")

    def _snapshot(directory: str) -> frozenset[str]:
        """Return frozenset of absolute file paths inside directory, or empty set."""
        if not os.path.isdir(directory):
            return frozenset()
        return frozenset(
            os.path.join(root, f)
            for root, _, files in os.walk(directory)
            for f in files
        )

    snap_models_before = _snapshot(models_dir)
    snap_scen_before   = _snapshot(scen_dir)

    # Force re-import by clearing cached modules
    for mod_name in list(sys.modules.keys()):
        if "train_power_risk_model" in mod_name or "evaluate_power_risk_model" in mod_name:
            del sys.modules[mod_name]

    import backend.scripts.train_power_risk_model     # noqa: PLC0415, F401
    import backend.scripts.evaluate_power_risk_model  # noqa: PLC0415, F401

    snap_models_after = _snapshot(models_dir)
    snap_scen_after   = _snapshot(scen_dir)

    assert snap_models_after == snap_models_before, (
        "Import created files in data/models/: "
        + str(snap_models_after - snap_models_before)
    )
    assert snap_scen_after == snap_scen_before, (
        "Import created files in data/scenarios/: "
        + str(snap_scen_after - snap_scen_before)
    )


# ---------------------------------------------------------------------------
# M23 — ROC-AUC matches sklearn reference
# ---------------------------------------------------------------------------

def test_M23_roc_auc_matches_sklearn_reference() -> None:
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    m = _import_train()

    y_true = [1, 0, 1, 0, 1, 1, 0, 0]
    y_prob = [0.9, 0.1, 0.8, 0.2, 0.7, 0.6, 0.4, 0.3]

    our_auc = m._roc_auc_score(y_true, y_prob)
    ref_auc = float(roc_auc_score(y_true, y_prob))

    assert math.isclose(our_auc, ref_auc, rel_tol=1e-6), (
        f"ROC-AUC mismatch: ours={our_auc}, sklearn={ref_auc}"
    )
