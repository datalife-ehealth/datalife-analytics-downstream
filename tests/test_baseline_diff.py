import json

import pytest

from datalife_analytics.baseline.diff import compare_reports
from datalife_analytics.baseline.diff_cli import main


def make_report(accuracy: float = 0.80) -> dict:
    return {
        "lineage": {
            "contract": "synthetic-contract",
            "contract_version": "0.1.0",
            "generator_seed": 42,
            "split_seed": 42,
            "train_fraction": 0.7,
            "n_subjects": 100,
        },
        "split": {
            "n_train": 70,
            "n_test": 30,
            "train_subjects": 50,
            "test_subjects": 20,
        },
        "models": {
            "ridge_logistic": {
                "test": {
                    "n": 30,
                    "accuracy": accuracy,
                    "brier": 0.15,
                    "roc_auc": 0.85,
                }
            }
        },
    }


def test_identical_reports_pass():
    baseline = make_report()
    current = make_report()

    tolerances = {"accuracy": 0.02}

    result = compare_reports(baseline, current, tolerances)
    assert result["status"] == "PASS"
    assert result["checks"]["accuracy"] == "PASS"


def test_metric_exceeding_tolerance_fails():
    baseline = make_report(accuracy=0.80)
    current = make_report(accuracy=0.90)

    tolerances = {"accuracy": 0.02}

    result = compare_reports(baseline, current, tolerances)

    assert result["status"] == "FAIL"
    assert result["checks"]["accuracy"] == "FAIL"


def test_missing_metric_fails():
    baseline = make_report()
    current = make_report()

    del current["models"]["ridge_logistic"]["test"]["accuracy"]

    tolerances = {"accuracy": 0.02}

    result = compare_reports(baseline, current, tolerances)

    assert result["status"] == "FAIL"
    assert result["checks"]["accuracy"] == "MISSING"


def test_contract_version_mismatch_fails():
    baseline = make_report()
    current = make_report()

    current["lineage"]["contract_version"] = "0.2.0"

    tolerances = {"accuracy": 0.02}

    result = compare_reports(baseline, current, tolerances)

    assert result["status"] == "FAIL"
    assert result["checks"]["lineage.contract_version"] == "FAIL"


def test_kfold_reports_cli_pass(tmp_path):
    baseline = make_report()
    current = make_report()

    for report in (baseline, current):
        report["split"] = {
            "mode": "subject_stratified_kfold",
            "n_splits": 5,
            "n_examples": 100,
        }

        report["lineage"]["n_splits"] = 5
        report["lineage"]["train_fraction"] = None

        test = report["models"]["ridge_logistic"]["test"]
        test["n"] = 100
        test["roc_auc"] = None
        test["roc_auc_foldwise"] = {
            "mean": 0.85,
            "std": 0.03,
            "folds": [0.82, 0.84, 0.85, 0.87, 0.87],
            "note": None,
        }

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    tolerances_path = tmp_path / "tolerances.json"

    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")
    tolerances_path.write_text(
        json.dumps({"roc_auc": 0.02}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--tolerances",
            str(tolerances_path),
            "--json",
        ]
    )

    assert exit_code == 0


def test_mixed_split_modes_cli_rejected(tmp_path):
    baseline = make_report()
    current = make_report()

    current["split"] = {
        "mode": "subject_stratified_kfold",
        "n_splits": 5,
        "n_examples": 100,
    }

    current["lineage"]["n_splits"] = 5
    current["lineage"]["train_fraction"] = None

    test = current["models"]["ridge_logistic"]["test"]
    test["n"] = 100
    test["roc_auc"] = None
    test["roc_auc_foldwise"] = {"mean": 0.85}

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    tolerances_path = tmp_path / "tolerances.json"

    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")
    tolerances_path.write_text(
        json.dumps({"roc_auc": 0.02}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--tolerances",
            str(tolerances_path),
        ]
    )

    assert exit_code == 2


def test_sample_count_drift_fails():
    baseline = make_report()
    current = make_report()

    current["models"]["ridge_logistic"]["test"]["n"] = 99

    result = compare_reports(baseline, current, {"accuracy": 0.02})

    assert result["status"] == "FAIL"
    assert result["checks"]["test.n"] == "FAIL"


def test_split_seed_drift_fails():
    baseline = make_report()
    current = make_report()

    current["lineage"]["split_seed"] = 99

    result = compare_reports(baseline, current, {"accuracy": 0.02})

    assert result["status"] == "FAIL"
    assert result["checks"]["lineage.split_seed"] == "FAIL"


def test_kfold_split_count_drift_fails():
    baseline = make_report()
    current = make_report()

    for report in (baseline, current):
        report["split"] = {
            "mode": "subject_stratified_kfold",
            "n_splits": 5,
            "n_examples": 100,
        }
        report["lineage"]["n_splits"] = 5
        report["lineage"]["train_fraction"] = None

    current["split"]["n_splits"] = 10

    result = compare_reports(baseline, current, {"accuracy": 0.02})

    assert result["status"] == "FAIL"
    assert result["checks"]["split.n_splits"] == "FAIL"


def test_nan_in_unconfigured_field_cli_rejected(tmp_path):
    baseline = make_report()
    current = make_report()

    # Add NaN somewhere that isn't being compared as a metric.
    current["extra"] = {"nested": [1, float("nan")]}

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    tolerances_path = tmp_path / "tolerances.json"

    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")
    tolerances_path.write_text(
        json.dumps({"accuracy": 0.02}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--tolerances",
            str(tolerances_path),
        ]
    )

    assert exit_code == 2


def test_infinity_in_unconfigured_field_cli_rejected(tmp_path):
    baseline = make_report()
    current = make_report()

    current["extra"] = {"nested": [1, float("inf"), float("-inf")]}

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    tolerances_path = tmp_path / "tolerances.json"

    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")
    tolerances_path.write_text(
        json.dumps({"accuracy": 0.02}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--tolerances",
            str(tolerances_path),
        ]
    )

    assert exit_code == 2


def test_oversized_integer_cli_rejected(tmp_path):
    baseline = make_report()
    current = make_report()

    current["models"]["ridge_logistic"]["test"]["accuracy"] = 10**1000

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    tolerances_path = tmp_path / "tolerances.json"

    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")
    tolerances_path.write_text(
        json.dumps({"accuracy": 0.02}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--tolerances",
            str(tolerances_path),
        ]
    )

    assert exit_code == 2


def test_oversized_integer_direct_comparison_rejected():
    baseline = make_report()
    current = make_report()

    current["models"]["ridge_logistic"]["test"]["accuracy"] = 10**1000

    with pytest.raises(ValueError):
        compare_reports(baseline, current, {"accuracy": 0.02})


def test_boolean_in_unconfigured_field_cli_rejected(tmp_path):
    baseline = make_report()
    current = make_report()

    current["extra"] = {"nested": [1, {"flag": True}]}

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    tolerances_path = tmp_path / "tolerances.json"

    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")
    tolerances_path.write_text(
        json.dumps({"accuracy": 0.02}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--tolerances",
            str(tolerances_path),
        ]
    )

    assert exit_code == 2


def test_missing_required_metric_cli_returns_two(tmp_path):
    baseline = make_report()
    current = make_report()

    del current["models"]["ridge_logistic"]["test"]["accuracy"]

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    tolerances_path = tmp_path / "tolerances.json"

    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")
    tolerances_path.write_text(
        json.dumps({"accuracy": 0.02}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--tolerances",
            str(tolerances_path),
        ]
    )

    assert exit_code == 2


def test_metric_drift_cli_returns_one(tmp_path):
    baseline = make_report()
    current = make_report()

    current["models"]["ridge_logistic"]["test"]["accuracy"] = 0.50

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    tolerances_path = tmp_path / "tolerances.json"

    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")
    tolerances_path.write_text(
        json.dumps({"accuracy": 0.02}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--tolerances",
            str(tolerances_path),
        ]
    )

    assert exit_code == 1


@pytest.mark.parametrize(
    ("section", "field", "bad_value"),
    [
        ("lineage", "contract", ""),
        ("lineage", "contract_version", 123),
        ("lineage", "generator_seed", "42"),
        ("lineage", "split_seed", "42"),
        ("lineage", "n_subjects", "100"),
        ("lineage", "train_fraction", "0.7"),
        ("split", "n_train", "70"),
        ("split", "n_test", "30"),
        ("split", "train_subjects", "50"),
        ("split", "test_subjects", "20"),
    ],
)
def test_wrong_typed_structural_field_rejected(section, field, bad_value):
    baseline = make_report()
    current = make_report()
    current[section][field] = bad_value

    with pytest.raises(ValueError):
        compare_reports(baseline, current, {"accuracy": 0.02})


def test_identical_wrong_typed_count_cli_returns_two(tmp_path):
    baseline = make_report()
    current = make_report()

    baseline["models"]["ridge_logistic"]["test"]["n"] = "30"
    current["models"]["ridge_logistic"]["test"]["n"] = "30"

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    tolerances_path = tmp_path / "tolerances.json"

    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")
    tolerances_path.write_text(
        json.dumps({"accuracy": 0.02}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--tolerances",
            str(tolerances_path),
        ]
    )

    assert exit_code == 2


@pytest.mark.parametrize("field", ["n_splits", "n_examples"])
def test_kfold_wrong_typed_split_field_rejected(field):
    baseline = make_report()
    current = make_report()

    for report in (baseline, current):
        report["split"] = {
            "mode": "subject_stratified_kfold",
            "n_splits": 5,
            "n_examples": 100,
        }
        report["lineage"]["n_splits"] = 5
        report["lineage"]["train_fraction"] = None

    current["split"][field] = "5"

    with pytest.raises(ValueError):
        compare_reports(baseline, current, {"accuracy": 0.02})


@pytest.mark.parametrize(
    ("section", "field", "bad_value"),
    [
        ("test", "n", 0),
        ("test", "n", -1),
        ("test", "n", 10**300),
        ("split", "n_train", 0),
        ("split", "n_train", -1),
        ("split", "n_train", 10**300),
        ("lineage", "split_seed", -1),
        ("lineage", "split_seed", 2**63),
    ],
)
def test_out_of_range_structural_field_cli_returns_two(tmp_path, section, field, bad_value):
    baseline = make_report()
    current = make_report()

    for report in (baseline, current):
        if section == "test":
            report["models"]["ridge_logistic"]["test"][field] = bad_value
        else:
            report[section][field] = bad_value

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    tolerances_path = tmp_path / "tolerances.json"

    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")
    tolerances_path.write_text(
        json.dumps({"accuracy": 0.02}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--tolerances",
            str(tolerances_path),
        ]
    )

    assert exit_code == 2


@pytest.mark.parametrize("bad_value", [0, 1, 101, 10**300])
def test_kfold_split_count_out_of_range_rejected(bad_value):
    baseline = make_report()
    current = make_report()

    for report in (baseline, current):
        report["split"] = {
            "mode": "subject_stratified_kfold",
            "n_splits": bad_value,
            "n_examples": 100,
        }
        report["lineage"]["n_splits"] = bad_value
        report["lineage"]["train_fraction"] = None

    with pytest.raises(ValueError):
        compare_reports(baseline, current, {"accuracy": 0.02})
