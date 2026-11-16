# Possession Lab submission path

Read the notebooks in this order:

1. [01_eda.ipynb](./01_eda.ipynb) — data, public five-minute scope, Smart Tagging value and limits.
2. [02_feature_engineering.ipynb](./02_feature_engineering.ipynb) — geometry, speed, acceleration, pressure, reachability and missing-ball evidence.
3. [03_scenario_engineering.ipynb](./03_scenario_engineering.ipynb) — dead ball, control, flight, reception, long-ball and contested scenarios.
4. [04_anchors_rulebook.ipynb](./04_anchors_rulebook.ipynb) — observable evidence for easy H/A/D anchors.
5. [05_transition_markers.ipynb](./05_transition_markers.ipynb) — transition windows and unresolved contest evidence.
6. [06_viterbi_model.ipynb](./06_viterbi_model.ipynb) — sequence decoding after evidence engineering.
7. [07_run_pipeline.ipynb](./07_run_pipeline.ipynb) — development pipeline execution and diagnostics.
8. [08_run_pipeline.ipynb](./08_run_pipeline.ipynb) — organiser-facing run of the best visible-ball candidate and public evaluation.
9. [08_experimental_results.ipynb](./08_experimental_results.ipynb) — comparison of models, scenarios and rejected experiments.
10. [../notebooks/appendix/10_first_half_results.ipynb](../notebooks/appendix/10_first_half_results.ipynb) — archived noisy hand-labelled first-half diagnostic and error audit.

Core conclusion: the observations are useful; mapping them to confidence and
H/A/D labels is the bottleneck. No fixed D hold duration is used.

## Appendix

The [appendix/](./appendix/) folder contains supporting definitions, the
organiser-facing evidence report, the rule-book detail, retired experiments,
and the sample-data restart audit. It is not required for the main walkthrough.

## Implementation

The notebook does not duplicate the detailed theory. Reusable functions live in:

- [calculations/](./calculations/) — one canonical `build_engineered_features()` composition;
- [scenarios/](./scenarios/) — public scenario, anchor, reception, and boundary API;
- [models/](./models/) — public submission-model and decoder API;
- [baseline_pipeline.py](./baseline_pipeline.py) — loading, auditing, frame time, distances, and submission writing;
- [temporal_rules.py](./temporal_rules.py) — temporal features, pressure, reachability, anchors, receptions, and scenarios;
- [decoder_experiments.py](./decoder_experiments.py) — confidence evidence, bounded resolvers, and decoders;
- [scenario_decoder.py](./scenario_decoder.py) — scenario-level decoding helpers;
- [broadcast_alignment.py](./broadcast_alignment.py) — generic broadcast/tactical alignment audit;
- [../config/possession_rules.yaml](../config/possession_rules.yaml) — configuration-backed thresholds and windows.

The root modules remain as compatibility implementations for existing scripts.
New code should import through the three packages so every model receives the
same complete engineered feature table.
