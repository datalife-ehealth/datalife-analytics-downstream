import pandas as pd
import pytest

from datalife_analytics.contract import ContractError, load_contract, validate


def _errors(tables) -> str:
    with pytest.raises(ContractError) as exc:
        validate(tables)
    return "\n".join(exc.value.errors)


def test_unknown_version():
    with pytest.raises(ValueError, match="Unknown contract version"):
        load_contract("9.9.9")


def test_forbidden_column_is_rejected(tables_copy):
    tables_copy["subjects"]["patient_name"] = "synthetic"
    assert "forbidden identifier-like column" in _errors(tables_copy)


def test_missing_column(tables_copy):
    tables_copy["observations"] = tables_copy["observations"].drop(columns=["unit"])
    assert "missing column(s): unit" in _errors(tables_copy)


def test_missing_table(tables_copy):
    del tables_copy["encounters"]
    assert "missing table(s): encounters" in _errors(tables_copy)


def test_bad_category(tables_copy):
    tables_copy["subjects"].loc[0, "age_band"] = "41"
    assert "subjects.age_band: 1 value(s) outside the allowed set" in _errors(tables_copy)


def test_bad_subject_key_pattern(tables_copy):
    tables_copy["subjects"].loc[0, "subject_key"] = "12345678900"
    assert "subjects.subject_key: 1 value(s) do not match" in _errors(tables_copy)


def test_duplicate_primary_key(tables_copy):
    obs = tables_copy["observations"]
    obs.loc[1, "observation_id"] = obs.loc[0, "observation_id"]
    assert "duplicate primary key" in _errors(tables_copy)


def test_unsorted_rows(tables_copy):
    tables_copy["encounters"] = tables_copy["encounters"].iloc[::-1]
    assert "not sorted" in _errors(tables_copy)


def test_orphan_foreign_key(tables_copy):
    tables_copy["subjects"] = tables_copy["subjects"].iloc[1:]
    assert "no match in subjects.subject_key" in _errors(tables_copy)


def test_encounter_before_enrolment(tables_copy):
    enc = tables_copy["encounters"]
    subj = tables_copy["subjects"]
    key = enc.loc[0, "subject_key"]
    subj.loc[subj["subject_key"] == key, "enrolled_on"] = "2099-01-01"
    assert "before subject enrolment" in _errors(tables_copy)


def test_unit_mismatch(tables_copy):
    obs = tables_copy["observations"]
    row = obs.index[obs["marker_code"] == "SBP"][0]
    obs.loc[row, "unit"] = "mmol/L"
    assert "inconsistent with marker_code" in _errors(tables_copy)


def test_value_out_of_range(tables_copy):
    obs = tables_copy["observations"]
    row = obs.index[obs["value"].notna()][0]
    obs.loc[row, "value"] = 9999.0
    assert "outside the marker's plausible range" in _errors(tables_copy)


def test_missing_reason_inconsistent(tables_copy):
    obs = tables_copy["observations"]
    row = obs.index[obs["value"].notna()][0]
    obs.loc[row, "value"] = float("nan")
    assert "missing_reason does not match" in _errors(tables_copy)


def test_unpaired_blood_pressure(tables_copy):
    obs = tables_copy["observations"]
    row = obs.index[obs["marker_code"] == "DBP"][0]
    tables_copy["observations"] = obs.drop(index=row).reset_index(drop=True)
    assert "SBP or DBP but not both" in _errors(tables_copy)


def test_diastolic_above_systolic(tables_copy):
    obs = tables_copy["observations"]
    enc = obs.loc[obs["marker_code"] == "SBP", "encounter_id"].iloc[0]
    obs.loc[(obs["encounter_id"] == enc) & (obs["marker_code"] == "DBP"), "value"] = 239.0
    assert "DBP is not below SBP" in _errors(tables_copy)


def test_errors_do_not_echo_values(tables_copy):
    secret = "syn-deadbeefdeadbeef-LEAK"
    tables_copy["subjects"].loc[0, "subject_key"] = secret
    assert secret not in _errors(tables_copy)


def test_nulls_in_required_field(tables_copy):
    tables_copy["encounters"].loc[0, "encounter_type"] = pd.NA
    assert "null value(s) in a non-nullable field" in _errors(tables_copy)


@pytest.mark.parametrize(
    "value, problem",
    [
        ("2024-05-06T15:28:00+00:00", "value(s) not in canonical timestamp format"),
        ("2024-05-06T15:28:17Z", "value(s) not in canonical timestamp format"),
        ("2024-05-06T15:28:00.000Z", "value(s) not in canonical timestamp format"),
        ("2024-05-06 15:28:00Z", "value(s) not in canonical timestamp format"),
        ("2024-05-06T15:28Z", "value(s) not in canonical timestamp format"),
        ("2024-02-30T15:28:00Z", "impossible calendar timestamp value(s)"),
        ("2024-05-06T25:28:00Z", "impossible calendar timestamp value(s)"),
    ],
)
def test_non_canonical_timestamp_is_rejected(tables_copy, value, problem):
    tables_copy["encounters"].loc[0, "encounter_at"] = value
    assert f"encounters.encounter_at: 1 {problem}" in _errors(tables_copy)


@pytest.mark.parametrize(
    "value, problem",
    [
        ("2024-03-29T00:00:00Z", "value(s) not in canonical date format"),
        ("2024-3-29", "value(s) not in canonical date format"),
        ("2024-02-30", "impossible calendar date value(s)"),
    ],
)
def test_non_canonical_date_is_rejected(tables_copy, value, problem):
    tables_copy["subjects"].loc[0, "enrolled_on"] = value
    assert f"subjects.enrolled_on: 1 {problem}" in _errors(tables_copy)


def test_parsed_timestamp_object_is_rejected(tables_copy):
    enc = tables_copy["encounters"]
    enc["encounter_at"] = pd.to_datetime(enc["encounter_at"], utc=True).astype(object)
    assert "not in canonical timestamp format" in _errors(tables_copy)


@pytest.mark.parametrize("marker", ["SBP", "DBP", "BMI"])
def test_not_resulted_rejected_for_non_lab_marker(tables_copy, marker):
    obs = tables_copy["observations"]
    row = obs.index[obs["marker_code"] == marker][0]
    obs.loc[row, ["value", "missing_reason"]] = [float("nan"), "not_resulted"]
    assert "null value(s) for non-laboratory markers" in _errors(tables_copy)


def test_not_resulted_allowed_for_lab_marker(tables_copy):
    obs = tables_copy["observations"]
    row = obs.index[obs["marker_code"] == "GLU"][0]
    obs.loc[row, ["value", "missing_reason"]] = [float("nan"), "not_resulted"]
    validate(tables_copy)
