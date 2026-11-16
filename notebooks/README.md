# Possession Lab notebook path

Read the notebooks in this order. The sequence follows **Problem → Idea → Construction → Pipeline → Evaluation**:

0. [00_load_data.ipynb](./00_load_data.ipynb) — locate the problem-repository data directory and load the shared data objects.
1. [01_eda.ipynb](./01_eda.ipynb) — data, public five-minute scope, Smart Tagging value and limits.
2. [02_feature_engineering.ipynb](./02_feature_engineering.ipynb) — modular geometry, speed, acceleration, distance-per-kick, and missing-ball signals.
3. [03_scenario_engineering.ipynb](./03_scenario_engineering.ipynb) — dead ball, control, flight, reception, long-ball and contested scenarios.
4. [04_deadball_setpieces.ipynb](./04_deadball_setpieces.ipynb) — focused restart boundary study for goal kicks, free kicks, corners, throw-ins, and kickoffs.
5. [04b_ball_missing_player_analysis.ipynb](./04b_ball_missing_player_analysis.ipynb) — missing-ball and missing-player observations, confidence signals, case diagrams, and the optional George fused-ball observation layer.
6. [05_anchors_rulebook.ipynb](./05_anchors_rulebook.ipynb) — observable evidence for easy H/A/D anchors.
7. [06_transition_markers.ipynb](./06_transition_markers.ipynb) — dead-ball/restart handling, transition windows, and unresolved contest evidence.
8. [07_probability_transition_model.ipynb](./07_probability_transition_model.ipynb) — normalized observation evidence, explicit no-switch/contested transition hypotheses, interval resolution, and optional Viterbi/HSMM roles.
9. [08_run_pipeline.ipynb](./08_run_pipeline.ipynb) — organiser-facing execution of the best visible-ball candidate.
10. [08_model_evaluation.ipynb](./08_model_evaluation.ipynb) — public five-minute scoring and every disagreement-window diagnostic.
11. [09_experimental_results.ipynb](./09_experimental_results.ipynb) — toggleable model reruns, final representatives, provided baselines, and ranked comparison.

Use that same template when presenting a result:

1. **Problem** — what is hard to observe or predict?
2. **Idea** — which evidence or hypothesis should help?
3. **Construction** — which reusable features, scenarios, anchors, or transitions encode it?
4. **Pipeline** — how does the active `possession_lab` runner consume the shared evidence?
5. **Evaluation** — what does the public sanity slice show, and which uncertainty remains?

The old registry-based runner was retired because it referenced obsolete model
names; it is not part of the presentation path. The current 08 notebook calls
`possession_lab/run_interpolated_flight_experiment.py` directly, so it runs the candidate that
actually produced the best public result rather than selecting the older `possession` registry
model.

The noisy hand-labelled first-half experiment is not kept as an executable
notebook: its conclusion is preserved in
[`possession_lab/appendix/8_FIRST_HALF_DIAGNOSTIC.md`](../possession_lab/appendix/8_FIRST_HALF_DIAGNOSTIC.md).
It is not a benchmark and is not used to tune the detector.

Core conclusion: the observations are useful; mapping them to confidence and
H/A/D labels is the bottleneck. No fixed D hold duration is used.

The probability notebook is deliberately diagnostic: its softmax vectors are
normalized rule evidence, not calibrated probabilities. Calibration requires a
separate reviewed-label experiment and must not be confused with the anchor
or transition resolver.

## Appendix

The [appendix/](./appendix/) folder contains the detailed visual findings
appendix, supporting definitions, the rule-book detail, retired experiments,
and the sample-data restart audit. The executable first-half and duplicate
load/Viterbi notebooks were removed because the main path now contains their
authoritative versions or their conclusions are already recorded as notes.

## Helper-first execution

The notebooks call the active `possession_lab` helpers rather than reimplementing calculations. The
main runner is:

```python
from possession_lab.pipeline import build_observations, detect_scenario_evidence, build_transition_evidence, run_model

observations = build_observations()
evidence = detect_scenario_evidence(observations)
transition_evidence = build_transition_evidence(evidence)
labels = run_model(transition_evidence, model_name="protected_aligned_reception")
```

For the organiser-facing full run, use `python -m possession_lab.run_interpolated_flight_experiment`;
it writes the current visible-ball + `NO_SWITCH` candidate and all declared diagnostics.

To compare the independently developed `stage/hq` decoder without changing the
shared evidence construction, select it by name:

```python
stage_hq = select_model("stage_hq_viterbi")
labels = run_model(observations, model_name="stage_hq_viterbi")
```

This is intentionally a decoder comparison: scenarios and anchors are still
detected once and passed to every model.

Plots are audit views over the same feature table. A visual observation is not a hand-written label.
The reusable evidence timeline is `possession.calculations.plotting.plot_timeline_window()`;
the composite pitch-plus-timeline view is `plot_evidence_window()`. They render match-clock time,
nearest-team observations, selected anchors, scenario markers, missing-ball intervals, and an
optional public ground-truth strip. Notebooks select the case window; they do not reimplement the
plotting logic.

The modelling distinction is deliberate: `H`, `A`, and `D` are the hidden possession states. Scenario names such as `CONTROLLED`, `CONTESTED`, `BALL_IN_FLIGHT`, `LONG_BALL`, and `RESTART` describe observable evidence. The decoder maps those observations to confidence over the hidden states; scenarios are not extra labels. George's ball-fill output, when available, is another source of observations with uncertainty—not a second set of possession labels.

## Implementation

The notebook does not duplicate the detailed theory. Reusable functions live in:

- [possession_lab/](../possession_lab/) — current scenario, transition, missing-ball, and candidate runner implementation used by notebook 08;

- [calculations/](../possession_lab/calculations/) — one canonical `build_engineered_features()` composition;
- [scenarios/](../possession_lab/scenarios/) — scenario, anchor, reception, and boundary API;
- [models/](../possession_lab/models/) — model specifications and decoder API;
- [baseline_pipeline.py](../possession_lab/baseline_pipeline.py) — loading, auditing, frame time, distances, and submission writing;
- [temporal_rules.py](../possession_lab/temporal_rules.py) — temporal features, pressure, reachability, anchors, receptions, and scenarios;
- [decoder_experiments.py](../possession_lab/decoder_experiments.py) — confidence evidence, bounded resolvers, and decoders;
- [scenario_decoder.py](../possession_lab/scenario_decoder.py) — scenario-level decoding helpers;
- [broadcast_alignment.py](../possession_lab/broadcast_alignment.py) — generic broadcast/tactical alignment audit;
- [../config/possession_rules.yaml](../config/possession_rules.yaml) — configuration-backed thresholds and windows.

The root modules remain as compatibility implementations for existing scripts.
New code should import through the three packages so every model receives the
same complete engineered feature table.
