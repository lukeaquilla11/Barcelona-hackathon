# First-half diagnostic

An earlier executable notebook evaluated a candidate against
`data/first_half_gt.csv`. That notebook has been removed from the presentation
tree because the file is hand-labelled and the diagnostic used an older
candidate. This note preserves the conclusion without presenting it as a
reproducible benchmark.

This file is an internal, hand-labelled reference. It is useful for finding recurring failure modes, but it is not authoritative ground truth and should not be used to tune every transition boundary.

The archived low-motion candidate scored:

| Metric | Value |
|---|---:|
| Macro F1 | 0.7802 |
| Spell F1 mean | 0.3524 |
| H F1 | 0.8337 |
| A F1 | 0.6632 |
| D F1 | 0.8436 |

The main diagnostic pattern was long dead-ball/live-boundary disagreement. That result is evidence for inspecting restart and settling logic, not proof that every hand-labelled frame is wrong or that the low-motion rule should be globally retuned.

The authoritative development benchmark remains the public five-minute reference in `evaluation/gt/BAR-ATH_possession_gt_public.csv`.
