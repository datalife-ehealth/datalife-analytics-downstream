from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from datalife_analytics.contract import load_contract, validate
from datalife_analytics.generator import GeneratorConfig, generate
from datalife_analytics.io import canonical_csv, read_dataset, write_dataset


def _hashes(tables):
    return {k: canonical_csv(v) for k, v in tables.items()}


def test_same_seed_is_byte_identical(small_config, small_tables):
    assert _hashes(generate(small_config)) == _hashes(small_tables)


def test_different_seed_differs(small_config, small_tables):
    other = generate(replace(small_config, seed=small_config.seed + 1))
    assert _hashes(other) != _hashes(small_tables)


def test_subjects_are_stable_when_cohort_grows(small_config, small_tables):
    bigger = generate(replace(small_config, n_subjects=small_config.n_subjects + 40))
    small_keys = set(small_tables["subjects"]["subject_key"])
    big = bigger["subjects"].set_index("subject_key")
    assert small_keys <= set(big.index)
    pd.testing.assert_frame_equal(
        small_tables["subjects"].set_index("subject_key"),
        big.loc[sorted(small_keys)],
    )


def test_output_satisfies_contract(small_tables):
    validate(small_tables)


def test_columns_follow_contract(small_tables):
    contract = load_contract()
    for name, df in small_tables.items():
        assert list(df.columns) == contract.columns(name)


def test_no_identifier_like_columns(small_tables):
    patterns = load_contract().raw["forbidden_field_patterns"]
    for df in small_tables.values():
        for col in df.columns:
            assert not any(p in col.lower() for p in patterns), col


def test_missingness_is_documented(small_tables):
    obs = small_tables["observations"]
    null_value = obs["value"].isna()
    assert null_value.any(), "fixture should exercise not_resulted missingness"
    assert (obs.loc[null_value, "missing_reason"] == "not_resulted").all()
    assert obs.loc[~null_value, "missing_reason"].isna().all()


def test_zero_missingness_rate(small_config):
    tables = generate(replace(small_config, not_resulted_rate=0.0))
    assert not tables["observations"]["value"].isna().any()


def test_timestamps_inside_window(small_config, small_tables):
    enc = pd.to_datetime(small_tables["encounters"]["encounter_at"], utc=True)
    assert enc.min() >= pd.Timestamp(small_config.start, tz="UTC")
    assert enc.max() < pd.Timestamp(small_config.end, tz="UTC")


def test_visit_hours_are_plausible(small_tables):
    enc = small_tables["encounters"]
    at = pd.to_datetime(enc["encounter_at"], utc=True)
    clinic = enc["encounter_type"] != "urgent"
    assert (at[clinic].dt.weekday < 5).all()
    assert at[clinic].dt.hour.between(8, 16).all()
    assert at[~clinic].dt.hour.between(7, 22).all()


def test_one_encounter_per_subject_per_day(small_tables):
    enc = small_tables["encounters"]
    day = pd.to_datetime(enc["encounter_at"], utc=True).dt.date
    assert not pd.DataFrame({"s": enc["subject_key"], "d": day}).duplicated().any()


def test_glucose_consistent_with_hba1c(small_config):
    obs = generate(replace(small_config, n_subjects=400))["observations"]
    means = obs.pivot_table(index="subject_key", columns="marker_code", values="value")
    both = means[["GLU", "HBA1C"]].dropna()
    high_glu = (both["GLU"] >= 7.0).mean()
    high_a1c = (both["HBA1C"] >= 6.5).mean()
    assert abs(high_glu - high_a1c) < 0.10
    assert both.corr().iloc[0, 1] > 0.6


def test_rows_are_sorted(small_tables):
    contract = load_contract()
    for name, df in small_tables.items():
        keys = contract.tables[name]["sort_by"]
        assert df.equals(df.sort_values(keys, kind="mergesort"))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"seed": -1}, "seed"),
        ({"seed": 1, "n_subjects": 0}, "n_subjects"),
        ({"seed": 1, "n_subjects": 10**9}, "n_subjects"),
        ({"seed": 1, "start": date(2025, 1, 1), "end": date(2025, 1, 10)}, "end must be"),
        ({"seed": 1, "not_resulted_rate": 0.9}, "not_resulted_rate"),
    ],
)
def test_invalid_config_fails_clearly(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GeneratorConfig(**kwargs)


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_round_trip(tmp_path, small_config, small_tables, fmt):
    manifest = write_dataset(small_tables, small_config, tmp_path, fmt=fmt)
    assert manifest.exists()
    back = read_dataset(tmp_path)
    validate(back)
    assert _hashes(back) == _hashes(small_tables)


def test_manifest_is_deterministic(tmp_path, small_config, small_tables):
    a = write_dataset(small_tables, small_config, tmp_path / "a", fmt="csv")
    b = write_dataset(generate(small_config), small_config, tmp_path / "b", fmt="csv")
    assert a.read_bytes() == b.read_bytes()
