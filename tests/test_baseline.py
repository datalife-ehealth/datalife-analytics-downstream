"""End-to-end baseline study against the v0.1.0 generator and committed fixture."""

from __future__ import annotations

from pathlib import Path

from datalife_analytics.baseline.cli import main
from datalife_analytics.baseline.pipeline import run_study
from datalife_analytics.io import read_dataset


def test_run_study_includes_naive_comparator_and_limitations(small_tables, small_config):
    result = run_study(
        small_tables,
        split_seed=7,
        generator_seed=small_config.seed,
        n_subjects=small_config.n_subjects,
    )
    models = result.payload["models"]
    assert "naive_prevalence" in models
    assert "ridge_logistic" in models
    assert models["naive_prevalence"]["test"]["n"] > 0
    assert models["ridge_logistic"]["test"]["n"] > 0
    assert result.payload["limitations"]
    assert result.payload["lineage"]["contract_version"] == "0.1.0"
    assert result.payload["lineage"]["split_seed"] == 7
    assert result.payload["cohort_flow"]["n_subjects"] == small_config.n_subjects


def test_run_study_is_deterministic(small_tables):
    first = run_study(small_tables, split_seed=4, generator_seed=7, n_subjects=60)
    second = run_study(small_tables, split_seed=4, generator_seed=7, n_subjects=60)
    assert (
        first.payload["models"]["ridge_logistic"]["test"]["accuracy"]
        == second.payload["models"]["ridge_logistic"]["test"]["accuracy"]
    )
    assert first.payload["split"] == second.payload["split"]


def test_impute_medians_are_finite(small_tables):
    result = run_study(small_tables, split_seed=7)
    impute = result.payload["impute"]
    assert isinstance(impute["median_bmi"], float)
    assert isinstance(impute["median_sbp"], float)


def test_cli_one_command_writes_json_and_markdown(tmp_path: Path):
    out = tmp_path / "reports"
    code = main(["--seed", "20260101", "--subjects", "40", "--split-seed", "7", "--out", str(out)])
    assert code == 0
    assert (out / "metrics.json").is_file()
    report = (out / "report.md").read_text("utf-8")
    assert "Naive comparator, test" in report
    assert "Limitations" in report
    assert "0.1.0" in report


def test_cli_from_committed_fixture(tmp_path: Path):
    out = tmp_path / "from-fixture"
    code = main(
        [
            "--input",
            str(Path("tests/fixtures")),
            "--prefix",
            "synthetic_v0_1_0_",
            "--split-seed",
            "1",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    text = (out / "metrics.json").read_text("utf-8")
    assert "tests/fixtures" in text
    assert "0.1.0" in text


def test_cli_rejects_bad_input(tmp_path: Path, capsys):
    code = main(["--input", str(tmp_path / "missing"), "--out", str(tmp_path / "out")])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_read_dataset_feeds_the_pipeline():
    tables = read_dataset("tests/fixtures", prefix="synthetic_v0_1_0_")
    result = run_study(tables, split_seed=2, input_path="tests/fixtures")
    assert result.payload["cohort_flow"]["n_subjects"] == 25
    assert result.payload["lineage"]["contract_version"] == "0.1.0"
