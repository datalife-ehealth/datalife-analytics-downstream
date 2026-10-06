from datetime import date
from pathlib import Path

import pytest

from datalife_analytics.generator import GeneratorConfig, generate

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_PREFIX = "synthetic_v0_1_0_"
# The committed fixture is regenerated from exactly this configuration.
FIXTURE_CONFIG = GeneratorConfig(
    seed=20261006, n_subjects=25, start=date(2024, 1, 1), end=date(2025, 12, 31)
)


@pytest.fixture(scope="session")
def small_config() -> GeneratorConfig:
    return GeneratorConfig(seed=7, n_subjects=60)


@pytest.fixture(scope="session")
def small_tables(small_config):
    return generate(small_config)


@pytest.fixture()
def tables_copy(small_tables):
    return {k: v.copy() for k, v in small_tables.items()}
