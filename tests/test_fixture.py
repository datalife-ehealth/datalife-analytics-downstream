"""The committed fixture must be exactly what the generator produces from FIXTURE_CONFIG."""

import json

from conftest import FIXTURE_CONFIG, FIXTURE_DIR, FIXTURE_PREFIX

from datalife_analytics.contract import validate
from datalife_analytics.generator import generate
from datalife_analytics.io import canonical_csv, read_dataset, sha256


def test_fixture_validates():
    validate(read_dataset(FIXTURE_DIR, prefix=FIXTURE_PREFIX))


def test_fixture_regenerates_byte_for_byte():
    tables = generate(FIXTURE_CONFIG)
    for name, df in tables.items():
        committed = (FIXTURE_DIR / f"{FIXTURE_PREFIX}{name}.csv").read_bytes()
        assert canonical_csv(df) == committed, (
            f"{name} fixture is stale; run: python scripts/regenerate_fixture.py"
        )


def test_manifest_hashes_match_files():
    manifest = json.loads((FIXTURE_DIR / f"{FIXTURE_PREFIX}manifest.json").read_text())
    assert manifest["config"] == FIXTURE_CONFIG.as_dict()
    for meta in manifest["tables"].values():
        data = (FIXTURE_DIR / meta["path"]).read_bytes()
        assert sha256(data) == meta["canonical_csv_sha256"]
