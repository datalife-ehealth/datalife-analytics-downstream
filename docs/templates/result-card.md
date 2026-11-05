# Result / Model Card

A short, reviewable record of one completed analysis or model, written **after** the
study so its results, uncertainty, and limits can be checked before it is presented
or reused.

A completed card is a review aid, not a certification. Completing it does not
establish regulatory, clinical, privacy, or fairness compliance; it only records
the results and limits a reviewer needs.

Copy this file next to the result it describes, rename it to the result, and fill
every required field. Replace every `<...>` placeholder.

## Required facts

Keep each fact distinct and specific. This section is the reviewable minimum.

- Environment: interpreter, package versions, hardware, and fixed seeds.
- Data version: the dataset card and schema version this result was produced from.
- Performance with uncertainty: headline metrics with confidence or bootstrap intervals.
- Calibration: predicted-versus-observed behavior where relevant.
- Subgroup results: performance by subgroup and whether any fairness claim is made.
- Failure modes: where and how the model is known to underperform or mislead.
- Limitations: scope, sample size, and what the result does not show.
- Reproduction command: the exact command that reproduces these numbers.

## Optional guidance

Add narrative only where it helps a reviewer; omit sections that do not apply.

- Links to the study card that designed this result.
- Negative or null results that bound the claim.
- Artifacts and where they are stored outside Git.

## Synthetic mini-example

Everything below is invented; no real record, identifier, or private document
is included.

- Environment: Python 3.11, pinned `pandas` and `scikit-learn` versions, `--seed 42`.
- Data version: dataset card `weekly-checkin-synth` at `analytics-snapshot-v0.1.0`.
- Performance with uncertainty: held-out AUROC 0.74 [0.68, 0.80] (1000-bootstrap CI).
- Calibration: Brier 0.21; predicted risk tracks observed risk in weekly bins.
- Subgroup results: stable across synthetic cohort length; no fairness claim is made.
- Failure modes: underperforms for subjects with fewer than 4 check-ins.
- Limitations: synthetic data only; no external or clinical validation.
- Reproduction command: `python -m studies.run --card weekly-dropoff --seed 42`.
