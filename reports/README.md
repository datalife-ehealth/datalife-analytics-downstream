# Reports

Generated study output (`metrics.json`, `report.md`) belongs in `generated/`, which
is git-ignored. Reproduce a run instead of committing artifacts:

```bash
datalife-baseline --seed 20260101 --subjects 80 --split-seed 7 \
    --out reports/generated
```

See [docs/baseline-study.md](../docs/baseline-study.md) for the question, split,
and limitations.
