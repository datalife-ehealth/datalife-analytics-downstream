# Study Card

A short, reviewable design record for one analysis or model experiment, completed
**before** the experiment is implemented, so its scope, data, and leakage control
can be checked up front.

A completed card is a review aid, not a certification. Completing it does not
establish regulatory, clinical, privacy, or fairness compliance; it only records
the design and acceptance bar a reviewer needs.

Copy this file next to the analysis it describes, rename it to the study, and fill
every required field. Replace every `<...>` placeholder.

## Required facts

Keep each fact distinct and specific. This section is the reviewable minimum.

- Research question: the single question this study answers, plus explicit non-use cases.
- Cohort: the population and how it was sampled or selected.
- Outcome: the measured outcome and its definition.
- Split strategy: how data is divided across time, subjects, or folds, and why.
- Baselines: the simple baseline(s) the main model must be compared against.
- Metrics: discrimination, calibration, uncertainty, and clinically relevant error metrics.
- Leakage risks: subject-level and temporal leakage paths this design controls for.
- Subgroup plan: which subgroups are evaluated and whether fairness is claimed.
- Acceptance criteria: the concrete results or properties required before claiming success.
- Reproduction link: the exact command or commit that runs this study.

## Optional guidance

Add narrative only where it helps a reviewer; omit sections that do not apply.

- Compute and environment pins.
- Sensitivity analyses planned.
- Related dataset cards and result cards.

## Synthetic mini-example

Everything below is invented; no real record, identifier, or private document
is included.

- Research question: does a transparent baseline flag a synthetic weekly drop-off? Not for care.
- Cohort: 1000 synthetic subjects with weekly check-ins over 52 weeks.
- Outcome: binary drop-off in the next week.
- Split strategy: time-appropriate train (weeks 1-39), validation (40-46), test (47-52).
- Baselines: no-history always-negative and a last-observation logit baseline.
- Metrics: AUROC, Brier score, predicted-vs-observed calibration, 95% CI on AUROC.
- Leakage risks: subject and temporal leakage avoided by the time-based split; no future fields.
- Subgroup plan: evaluate by synthetic cohort length; no fairness claim is made.
- Acceptance criteria: beats the always-negative baseline on held-out AUROC with a reported CI.
- Reproduction link: `pytest -q tests/studies` at commit `<abc1234>`.
