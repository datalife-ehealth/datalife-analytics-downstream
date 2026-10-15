# DataLife review cards

Lightweight templates that make datasets, studies, and model results reviewable
before code or generated artifacts are merged. Each card records the facts and
limits a reviewer needs; completing one is a review aid, not a certification of
regulatory, clinical, privacy, or fairness compliance.

Copy a template next to what it describes, fill every required field, and keep
each fact distinct from the optional narrative.

| Card | Use when | File |
|---|---|---|
| Dataset card | A new dataset, export, or a new version of one is introduced or consumed. | [dataset-card.md](dataset-card.md) |
| Study card | A new analysis or model experiment is designed, before it is implemented. | [study-card.md](study-card.md) |
| Result / model card | A study or model is finished and its results are presented or reused. | [result-card.md](result-card.md) |

## Keep required facts distinct from optional guidance

Every card separates a short, reviewable "Required facts" list from optional
narrative. Fill the required list completely and specifically; add optional
narrative only where it helps a reviewer. A card may grow new domain sections
without treating this set as exhaustive.

## Boundaries

- No real records, copied private documents, or claims unsupported by evidence.
- No implication that a card certifies compliance of any kind.
- The examples in each template are synthetic mini-examples; keep it that way.
