"""Canonical serialisation, manifests, and reading generated datasets back."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import __version__
from .contract import Contract, load_contract
from .generator import GENERATOR_NAME, GENERATOR_VERSION, GeneratorConfig

PROVENANCE = (
    "Synthetic records produced by the DataLife longitudinal fixture generator. "
    "Not derived from real patients. Not suitable for clinical inference."
)
FORMATS = ("csv", "parquet")


def canonical_csv(df: pd.DataFrame) -> bytes:
    """Byte-stable CSV: LF endings, fixed float format, empty string for null."""
    return df.to_csv(index=False, lineterminator="\n", float_format="%.1f", na_rep="").encode(
        "utf-8"
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _arrow_schema(table: str, contract: Contract) -> pa.Schema:
    mapping = {
        "string": pa.string(),
        "category": pa.string(),
        "float": pa.float64(),
        "date": pa.date32(),
        "timestamp": pa.timestamp("s", tz="UTC"),
    }
    return pa.schema(
        [
            pa.field(f["name"], mapping[f["type"]], nullable=f["nullable"])
            for f in contract.tables[table]["fields"]
        ],
        metadata={"provenance": PROVENANCE, "contract_version": contract.version},
    )


def _to_arrow(df: pd.DataFrame, table: str, contract: Contract) -> pa.Table:
    schema = _arrow_schema(table, contract)
    arrays = []
    for f in contract.tables[table]["fields"]:
        col = df[f["name"]]
        if f["type"] == "date":
            arrays.append(pa.array(pd.to_datetime(col).dt.date, type=pa.date32()))
        elif f["type"] == "timestamp":
            arrays.append(pa.array(pd.to_datetime(col, utc=True), type=pa.timestamp("s", tz="UTC")))
        elif f["type"] == "float":
            arrays.append(pa.array(col.to_numpy(dtype="float64"), type=pa.float64()))
        else:
            arrays.append(pa.array(col.astype(object).where(col.notna(), None), type=pa.string()))
    return pa.Table.from_arrays(arrays, schema=schema)


def write_dataset(
    tables: dict[str, pd.DataFrame],
    cfg: GeneratorConfig,
    out_dir: str | Path,
    fmt: str = "csv",
    prefix: str = "",
) -> Path:
    """Write tables plus manifest.json. Returns the manifest path."""
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {', '.join(FORMATS)}")
    contract = load_contract(cfg.contract_version)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = {}
    for name, df in tables.items():
        csv_bytes = canonical_csv(df)
        path = out / f"{prefix}{name}.{fmt}"
        if fmt == "csv":
            path.write_bytes(csv_bytes)
        else:
            pq.write_table(_to_arrow(df, name, contract), path, compression="zstd")
        files[name] = {
            "path": path.name,
            "rows": int(len(df)),
            "canonical_csv_sha256": sha256(csv_bytes),
        }

    manifest = {
        "provenance": PROVENANCE,
        "contract": contract.raw["contract"],
        "contract_version": contract.version,
        "generator": GENERATOR_NAME,
        "generator_version": GENERATOR_VERSION,
        "package_version": __version__,
        "config": cfg.as_dict(),
        "format": fmt,
        "tables": files,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
    }
    manifest_path = out / f"{prefix}manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    return manifest_path


def read_dataset(directory: str | Path, prefix: str = "") -> dict[str, pd.DataFrame]:
    """Read a generated dataset (CSV or Parquet) back into canonical-text DataFrames."""
    directory = Path(directory)
    manifest = json.loads((directory / f"{prefix}manifest.json").read_text("utf-8"))
    contract = load_contract(manifest["contract_version"])
    tables = {}
    for name, meta in manifest["tables"].items():
        path = directory / meta["path"]
        if manifest["format"] == "csv":
            df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        else:
            df = pq.read_table(path).to_pandas()
        tables[name] = _canonicalise(df, name, contract)
    return tables


def _canonicalise(df: pd.DataFrame, table: str, contract: Contract) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for f in contract.tables[table]["fields"]:
        col = df[f["name"]]
        if f["type"] == "float":
            out[f["name"]] = pd.to_numeric(col).astype("float64")
        elif f["type"] == "date":
            out[f["name"]] = pd.to_datetime(col).dt.strftime("%Y-%m-%d")
        elif f["type"] == "timestamp":
            out[f["name"]] = pd.to_datetime(col, utc=True).dt.strftime("%Y-%m-%dT%H:%M:00Z")
        else:
            out[f["name"]] = col.astype(object).where(col.notna(), None)
    return out
