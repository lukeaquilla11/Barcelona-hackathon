# Possession Lab architecture and notebook path

The active implementation is the shared evidence table and runner in this folder. Read the organiser-facing notebooks in this order:

0. [00_load_data.ipynb](../notebooks/00_load_data.ipynb) — load the shared match data.
1. [01_eda.ipynb](../notebooks/01_eda.ipynb) — define the problem and inspect the public scope.
2. [02_feature_engineering.ipynb](../notebooks/02_feature_engineering.ipynb) — construct reusable physical and missing-ball signals.
3. [03_scenario_engineering.ipynb](../notebooks/03_scenario_engineering.ipynb) — combine observations into scenario evidence and case studies.
4. [04_deadball_setpieces.ipynb](../notebooks/04_deadball_setpieces.ipynb) — study restart boundaries.
5. [04b_ball_missing_player_analysis.ipynb](../notebooks/04b_ball_missing_player_analysis.ipynb) — inspect missing-ball and missing-player cases.
6. [05_anchors_rulebook.ipynb](../notebooks/05_anchors_rulebook.ipynb) — define stable H/A/D evidence.
7. [06_transition_markers.ipynb](../notebooks/06_transition_markers.ipynb) — resolve transition windows conservatively.
8. [07_probability_transition_model.ipynb](../notebooks/07_probability_transition_model.ipynb) — test normalized evidence and decoder hypotheses.
9. [08_run_pipeline.ipynb](../notebooks/08_run_pipeline.ipynb) — run the organiser-facing candidate.
10. [08_model_evaluation.ipynb](../notebooks/08_model_evaluation.ipynb) — score and inspect disagreement intervals.
11. [09_experimental_results.ipynb](../notebooks/09_experimental_results.ipynb) — compare final representatives.

The path is deliberately **Problem → Idea → Construction → Pipeline → Evaluation**. The notebooks are clients of this package; they should not duplicate the implementation.

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
- [decoder_experiments.py](./decoder_experiments.py) — confidence evidence, bounded resolvers, no-switch interval retention, and decoders;
- [calibration.py](./calibration.py) — opt-in MLE temperature scaling and Dirichlet transition-prior experiments;
- [run_calibration_experiment.py](./run_calibration_experiment.py) — disjoint public-slice calibration/holdout diagnostics;
- [sequence_models.py](./sequence_models.py) — anchor-initialized HMM/HSMM, Haiqal-style Bayesian-anchor HMM, and constrained sequence decoders;
- [haiqal_baseline.py](./haiqal_baseline.py) — isolated raw distance/speed/dead-ball baseline adapted from `origin/haiqal`; it does not replace our scenario or anchor decisions;
- `diagnostic_haiqal_visible_ball_no_switch_interval` — evidence-only integration of Haiqal's closeness observation into the best visible-ball + NO_SWITCH resolver. It is retained as a reproducible ablation, not promoted: it currently produces exactly the same labels and public metrics as the best model.
- [pipeline.py](./pipeline.py) — stable orchestration wrappers for loading data, building observations, adding scenario/probability/transition evidence, adding Haiqal observations, and running a registered model;
- [scenario_decoder.py](./scenario_decoder.py) — scenario-level decoding helpers;
- [broadcast_alignment.py](./broadcast_alignment.py) — generic broadcast/tactical alignment audit;
- [../config/possession_rules.yaml](../config/possession_rules.yaml) — configuration-backed thresholds and windows.

The root modules remain as compatibility implementations for existing scripts.
New code should import through the three packages so every model receives the
same complete engineered feature table.

The latest `origin/haiqal` branch was checked at commit `5092f12`. Its new
`solution/` model is the same raw distance/speed/Smart-Tagging/dead-ball
three-state Viterbi approach represented by `haiqal_baseline.py`; it does not
add a scenario detector, calibrated probability model, or transition ledger.
The adapter is therefore the maintained comparison boundary, while
`pipeline.py` keeps its observations composable with our scenario-aware
models without copying Haiqal's final labels.
