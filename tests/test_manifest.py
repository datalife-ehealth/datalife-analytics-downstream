"""validate must catch files that drifted from manifest.json after they were written."""

import json

import pytest
from conftest import FIXTURE_DIR, FIXTURE_PREFIX

from datalife_analytics.cli import main
from datalife_analytics.io import read_dataset, verify_manifest, write_dataset


def test_committed_fixture_matches_manifest():
    tables = read_dataset(FIXTURE_DIR, prefix=FIXTURE_PREFIX)
    assert verify_manifest(FIXTURE_DIR, tables, prefix=FIXTURE_PREFIX) == []


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_cli_validate_passes_untouched_dataset(tmp_path, small_config, small_tables, fmt):
    write_dataset(small_tables, small_config, tmp_path, fmt=fmt)
    assert main(["validate", str(tmp_path)]) == 0


def test_byte_edit_breaks_checksum(tmp_path, small_config, small_tables, capsys):
    write_dataset(small_tables, small_config, tmp_path, fmt="csv")
    path = tmp_path / "encounters.csv"
    # Line endings only: the parsed table is unchanged, so only the checksum can catch it.
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert main(["validate", str(tmp_path)]) == 1
    assert "encounters: SHA-256 does not match manifest" in capsys.readouterr().err


def test_dropped_row_breaks_row_count(tmp_path, small_config, small_tables, capsys):
    write_dataset(small_tables, small_config, tmp_path, fmt="csv")
    path = tmp_path / "observations.csv"
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]))
    assert main(["validate", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "observations: " in err and "manifest records" in err


def test_parquet_edit_breaks_checksum(tmp_path, small_config, small_tables, capsys):
    tampered = {k: v.copy() for k, v in small_tables.items()}
    tampered["subjects"].loc[0, "region"] = (
        "R02" if tampered["subjects"].loc[0, "region"] != "R02" else "R03"
    )
    write_dataset(small_tables, small_config, tmp_path / "orig", fmt="parquet")
    write_dataset(tampered, small_config, tmp_path / "new", fmt="parquet")
    (tmp_path / "new" / "manifest.json").write_bytes(
        (tmp_path / "orig" / "manifest.json").read_bytes()
    )
    assert main(["validate", str(tmp_path / "new")]) == 1
    assert "subjects: SHA-256 does not match manifest" in capsys.readouterr().err


def test_edited_manifest_rows_are_reported(tmp_path, small_config, small_tables, capsys):
    manifest = write_dataset(small_tables, small_config, tmp_path, fmt="csv")
    data = json.loads(manifest.read_text())
    data["tables"]["subjects"]["rows"] += 1
    manifest.write_text(json.dumps(data))
    assert main(["validate", str(tmp_path)]) == 1
    assert "subjects: 60 row(s) but manifest records 61" in capsys.readouterr().err


def test_missing_file_fails_cleanly(tmp_path, small_config, small_tables, capsys):
    write_dataset(small_tables, small_config, tmp_path, fmt="csv")
    (tmp_path / "subjects.csv").unlink()
    assert main(["validate", str(tmp_path)]) == 1
    assert "cannot read dataset" in capsys.readouterr().err
