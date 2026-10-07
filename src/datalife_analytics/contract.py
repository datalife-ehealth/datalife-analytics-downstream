"""Load the versioned data contract and validate tables against it.

Error messages name tables, columns, and row counts only. They never echo cell values,
so validation output is safe to paste into a public issue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

import pandas as pd

DEFAULT_VERSION = "0.1.0"


class ContractError(ValueError):
    """Raised when tables do not satisfy the data contract."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Contract validation failed:\n- " + "\n- ".join(errors))


@dataclass(frozen=True)
class Contract:
    raw: dict[str, Any]

    @property
    def version(self) -> str:
        return self.raw["version"]

    @property
    def tables(self) -> dict[str, Any]:
        return self.raw["tables"]

    @property
    def markers(self) -> dict[str, Any]:
        return self.raw["markers"]

    def columns(self, table: str) -> list[str]:
        return [f["name"] for f in self.tables[table]["fields"]]


@cache
def load_contract(version: str = DEFAULT_VERSION) -> Contract:
    package = f"datalife_analytics.contracts.v{version.replace('.', '_')}"
    try:
        text = resources.files(package).joinpath("data_dictionary.json").read_text("utf-8")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Unknown contract version: {version}") from exc
    return Contract(json.loads(text))


def forbidden_columns(columns: list[str], contract: Contract) -> list[str]:
    patterns = contract.raw["forbidden_field_patterns"]
    return [c for c in columns if any(p in c.lower() for p in patterns)]


def _check_field(df: pd.DataFrame, table: str, field: dict[str, Any], errors: list[str]) -> None:
    name = field["name"]
    col = df[name]
    nulls = int(col.isna().sum())
    if not field["nullable"] and nulls:
        errors.append(f"{table}.{name}: {nulls} null value(s) in a non-nullable field")
    present = col.dropna()
    ftype = field["type"]

    if ftype in ("string", "category"):
        as_text = present.astype(str)
        if "pattern" in field:
            bad = int((~as_text.str.fullmatch(field["pattern"])).sum())
            if bad:
                errors.append(f"{table}.{name}: {bad} value(s) do not match the required pattern")
        if "allowed" in field:
            bad = int((~as_text.isin(field["allowed"])).sum())
            if bad:
                errors.append(f"{table}.{name}: {bad} value(s) outside the allowed set")
    elif ftype == "float":
        numeric = pd.to_numeric(present, errors="coerce")
        bad = int(numeric.isna().sum())
        if bad:
            errors.append(f"{table}.{name}: {bad} non-numeric value(s)")
        if "min" in field:
            below = int((numeric < field["min"]).sum())
            if below:
                errors.append(f"{table}.{name}: {below} value(s) below minimum {field['min']}")
    elif ftype in ("date", "timestamp"):
        parsed = pd.to_datetime(present, errors="coerce", utc=True)
        bad = int(parsed.isna().sum())
        if bad:
            errors.append(f"{table}.{name}: {bad} unparseable {ftype} value(s)")


def _sorted_ok(df: pd.DataFrame, keys: list[str]) -> bool:
    if len(df) < 2:
        return True
    expected = df.sort_values(keys, kind="mergesort").index
    return bool((expected == df.index).all())


def validate(tables: dict[str, pd.DataFrame], contract: Contract | None = None) -> None:
    """Validate a mapping of table name to DataFrame. Raises ContractError on failure."""
    contract = contract or load_contract()
    errors: list[str] = []

    expected_tables = set(contract.tables)
    missing = sorted(expected_tables - set(tables))
    extra = sorted(set(tables) - expected_tables)
    if missing:
        errors.append(f"missing table(s): {', '.join(missing)}")
    if extra:
        errors.append(f"unexpected table(s): {', '.join(extra)}")

    for table, spec in contract.tables.items():
        if table not in tables:
            continue
        df = tables[table].reset_index(drop=True)
        expected_cols = contract.columns(table)
        cols = list(df.columns)

        flagged = forbidden_columns(cols, contract)
        if flagged:
            errors.append(f"{table}: forbidden identifier-like column(s): {', '.join(flagged)}")
        absent = [c for c in expected_cols if c not in cols]
        if absent:
            errors.append(f"{table}: missing column(s): {', '.join(absent)}")
        unknown = [c for c in cols if c not in expected_cols]
        if unknown:
            errors.append(f"{table}: unexpected column(s): {', '.join(unknown)}")
        if absent or unknown:
            continue
        if cols != expected_cols:
            errors.append(f"{table}: columns are not in contract order")

        for field in spec["fields"]:
            _check_field(df, table, field, errors)

        pk = spec["primary_key"]
        dupes = int(df.duplicated(pk).sum())
        if dupes:
            errors.append(f"{table}: {dupes} duplicate primary key(s) on {', '.join(pk)}")
        if not _sorted_ok(df, spec["sort_by"]):
            errors.append(f"{table}: rows are not sorted by {', '.join(spec['sort_by'])}")

    if not errors:
        _check_relations(tables, contract, errors)
    if errors:
        raise ContractError(errors)


def _check_relations(
    tables: dict[str, pd.DataFrame], contract: Contract, errors: list[str]
) -> None:
    for table, spec in contract.tables.items():
        for col, target in spec.get("foreign_keys", {}).items():
            ref_table, ref_col = target.split(".")
            orphans = int((~tables[table][col].isin(tables[ref_table][ref_col])).sum())
            if orphans:
                errors.append(f"{table}.{col}: {orphans} value(s) with no match in {target}")
    if errors:
        return

    subjects = tables["subjects"].set_index("subject_key")
    enc = tables["encounters"]
    obs = tables["observations"]

    enrolled = pd.to_datetime(subjects["enrolled_on"], utc=True)
    enc_at = pd.to_datetime(enc["encounter_at"], utc=True)
    early = int((enc_at < enc["subject_key"].map(enrolled)).sum())
    if early:
        errors.append(f"encounters.encounter_at: {early} encounter(s) before subject enrolment")

    enc_time = enc.set_index("encounter_id")["encounter_at"]
    mismatch = int((obs["observed_at"] != obs["encounter_id"].map(enc_time)).sum())
    if mismatch:
        errors.append(f"observations.observed_at: {mismatch} row(s) differ from encounter time")

    enc_subject = enc.set_index("encounter_id")["subject_key"]
    wrong_subject = int((obs["subject_key"] != obs["encounter_id"].map(enc_subject)).sum())
    if wrong_subject:
        errors.append(f"observations: {wrong_subject} row(s) whose encounter has another subject")

    units = {code: m["unit"] for code, m in contract.markers.items()}
    bad_unit = int((obs["unit"] != obs["marker_code"].map(units)).sum())
    if bad_unit:
        errors.append(f"observations.unit: {bad_unit} unit(s) inconsistent with marker_code")

    lo = obs["marker_code"].map({c: m["range"][0] for c, m in contract.markers.items()})
    hi = obs["marker_code"].map({c: m["range"][1] for c, m in contract.markers.items()})
    vals = pd.to_numeric(obs["value"], errors="coerce")
    out = int(((vals < lo) | (vals > hi)).sum())
    if out:
        errors.append(f"observations.value: {out} value(s) outside the marker's plausible range")

    value_null = obs["value"].isna()
    reason_null = obs["missing_reason"].isna()
    inconsistent = int((value_null == reason_null).sum())
    if inconsistent:
        errors.append(
            f"observations: {inconsistent} row(s) where missing_reason does not match a null value"
        )

    dup_marker = int(obs.duplicated(["encounter_id", "marker_code"]).sum())
    if dup_marker:
        errors.append(f"observations: {dup_marker} repeated marker(s) within one encounter")
        return

    bp = obs[obs["marker_code"].isin(["SBP", "DBP"])].pivot(
        index="encounter_id", columns="marker_code", values="value"
    )
    bp = bp.reindex(columns=["SBP", "DBP"])
    half_pairs = int(bp.isna().sum(axis=1).eq(1).sum())
    if half_pairs:
        errors.append(f"observations: {half_pairs} encounter(s) with SBP or DBP but not both")
    inverted = int((bp["DBP"] >= bp["SBP"]).sum())
    if inverted:
        errors.append(f"observations: {inverted} encounter(s) where DBP is not below SBP")
