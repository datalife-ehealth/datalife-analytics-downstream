# Dataset Card

A short, reviewable description of one dataset (or one version of a dataset) so a
reader can judge provenance, privacy, and intended use **before** any code or
generated artifact that consumes it is merged.

A completed card is a review aid, not a certification. Completing it does not
establish regulatory, clinical, privacy, or fairness compliance; it only records
the facts and limits a reviewer needs.

Copy this file next to the data contract it describes, rename it to the dataset,
and fill every required field. Replace every `<...>` placeholder.

## Required facts

Keep each fact distinct and specific. This section is the reviewable minimum.

- Origin and creator: where the data came from and who produced or licensed it.
- Synthetic generation method: if synthetic, name the generator, seed, and version.
- License or consent basis: the license, consent, or governance basis that allows use.
- Schema version: the versioned data-contract identifier this card describes.
- Population semantics: who or what the rows/records represent and how they were selected.
- Missingness: which fields can be absent and how absence is encoded.
- Privacy considerations: re-identification risk, aggregation or suppression applied.
- Intended uses: what a downstream analysis may do with this dataset.
- Prohibited uses: what this dataset must not be used for.
- Reproduction link: the exact command or commit that produces this version.

## Optional guidance

Add narrative only where it helps a reviewer; omit sections that do not apply.

- Volume and time span.
- Known biases or coverage gaps.
- Links to related study cards or result cards.

## Synthetic mini-example

Everything below is invented; no real record, identifier, or private document
is included.

- Origin and creator: invented by the DataLife scaffold team for template review.
- Synthetic generation method: `synthgen --seed 42 --version 1.3.0`, 1000 records.
- License or consent basis: CC0 public-domain synthetic fixture.
- Schema version: `analytics-snapshot-snapshot-v0.1.0`.
- Population semantics: 1000 synthetic longitudinal subjects, weekly check-ins.
- Missingness: `last_checkin_date` absent for ~5% of records, encoded as null.
- Privacy considerations: fully synthetic; no live identifiers; aggregate-only release.
- Intended uses: method and pipeline evaluation on a public sandbox.
- Prohibited uses: not for diagnosis, care, or public-health operations.
- Reproduction link: `python -m synthgen generate --seed 42` at commit `<abc1234>`.
