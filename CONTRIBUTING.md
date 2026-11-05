# Contributing to DataLife Analytics Downstream

The organization-wide
[contribution rules](https://github.com/datalife-ehealth/.github/blob/main/CONTRIBUTING.md)
apply here. Analytics contributions must also be reproducible and honest about data,
uncertainty, and intended use.

## Before implementation

Open a focused issue or RFC describing the research question, data contract, legal or
consent basis, expected output, evaluation design, and non-use cases. Use synthetic or
explicitly licensed de-identified data only. Never commit a private dataset merely
because it has had obvious identifiers removed.

## Reproducibility checklist

- Compare with a meaningful simple baseline.
- Prevent subject and temporal leakage.
- Pin dependencies and random seeds where relevant.
- Report uncertainty, calibration, subgroup results, and negative results.
- Move reusable notebook logic into tested modules.
- Clear notebook outputs and local paths before committing.
- Add a model card or study report for material results.

## Card templates

Reviewable documentation templates live in [`docs/templates/`](docs/templates/README.md).
Use the matching card so a reader can judge provenance and limits before the work is
merged:

- **Dataset card** — when a new dataset, export, or a new version of one is
  introduced or consumed.
- **Study card** — when a new analysis or model experiment is designed, before it
  is implemented.
- **Result / model card** — when a study or model is finished and its results are
  presented or reused.

Copy a template next to what it describes, fill every required field, and keep
required facts distinct from optional narrative. Templates are a review aid, not a
certification of regulatory, clinical, privacy, or fairness compliance. Examples
must remain synthetic.

## Pull requests

Keep one experiment or infrastructure concern per pull request. Link its issue or
RFC and provide exact reproduction commands. Generated artifacts and large files do
not belong in Git. Approval from the repository steward is required.
