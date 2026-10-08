# data/

Local, generated datasets live here and are ignored by Git. Nothing in this folder
except this README should ever be committed.

Generate a dataset:

```bash
datalife-fixtures generate --seed 42 --subjects 1000 --format parquet --out data/generated
datalife-fixtures validate data/generated
```

Every generated folder contains a `manifest.json` recording the contract version,
generator version, seed, configuration, row counts, content hashes, and library
versions. Keep the manifest with the data so results can be traced back to the
exact inputs.

All records are synthetic. They are not derived from real patients and are not
suitable for clinical inference.
