"""Regenerate the small committed test fixture under tests/fixtures/.

Run only when the contract or generator intentionally changes, and explain the
change in the pull request. Usage: python scripts/regenerate_fixture.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from conftest import FIXTURE_CONFIG, FIXTURE_DIR, FIXTURE_PREFIX  # noqa: E402

from datalife_analytics.contract import validate  # noqa: E402
from datalife_analytics.generator import generate  # noqa: E402
from datalife_analytics.io import write_dataset  # noqa: E402


def main() -> None:
    tables = generate(FIXTURE_CONFIG)
    validate(tables)
    manifest_path = write_dataset(
        tables, FIXTURE_CONFIG, FIXTURE_DIR, fmt="csv", prefix=FIXTURE_PREFIX
    )
    # Environment versions vary by machine; keep the committed manifest machine-neutral.
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest.pop("environment", None)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"fixture written to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
