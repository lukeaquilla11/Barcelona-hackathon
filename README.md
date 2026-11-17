# Barcelona hackathon: frame-level possession

This repository labels every tactical frame of Barcelona–Athletic as Home
(`H`), Away (`A`), dead ball (`D`), or not graded (`X`). The organiser-facing
story is in [ORGANISER_REPORT.md](./ORGANISER_REPORT.md). It explains the
football problem, the evidence we extract, the hypothesis-test resolver, and
why the final policy is designed for a full match rather than only the public
five-minute slice.

## Read the notebooks in order

The notebooks are explanations and demonstrations. They call reusable
functions from [possession_lab/](./possession_lab/) instead of maintaining a
second implementation in notebook cells.

1. [00_load_data.ipynb](./notebooks/00_load_data.ipynb) — load the supplied data.
2. [01_eda.ipynb](./notebooks/01_eda.ipynb) — establish the full-match problems: missing ball, missing players, restarts, and temporary track identities.
3. [02_feature_engineering.ipynb](./notebooks/02_feature_engineering.ipynb) — derive motion, direction, arrival, pressure, activity, and missing-ball evidence.
4. [03_scenario_engineering.ipynb](./notebooks/03_scenario_engineering.ipynb) — turn physical observations into football situations such as control, flight, reception, contest, and restart.
5. [04_deadball_setpieces.ipynb](./notebooks/04_deadball_setpieces.ipynb) — identify stoppage and restart boundaries without copying Smart Tags as labels.
6. [04b_ball_missing_player_analysis.ipynb](./notebooks/04b_ball_missing_player_analysis.ipynb) — inspect missing-ball and missing-player cases using backward and forward context.
7. [05_anchors_rulebook.ipynb](./notebooks/05_anchors_rulebook.ipynb) — define stable possession and restart anchors.
8. [06_transition_markers.ipynb](./notebooks/06_transition_markers.ipynb) — open local transition intervals when contact, arrival, restart, or dead-ball evidence changes.
9. [07_probability_transition_model.ipynb](./notebooks/07_probability_transition_model.ipynb) — compare evidence hypotheses and resolve ambiguous intervals conservatively.
10. [08_run_pipeline.ipynb](./notebooks/08_run_pipeline.ipynb) — rebuild the final submission from raw data.
11. [08_model_evaluation.ipynb](./notebooks/08_model_evaluation.ipynb) — score the public slice and inspect disagreements.
12. [09_experimental_results.ipynb](./notebooks/09_experimental_results.ipynb) — compact comparison of alternative policies; this is audit material, not the production runner.

Optional collaborator studies are kept in [notebooks/appendix/](./notebooks/appendix/), including the raw alternate-camera ball-observation walkthrough and the independent distance/speed observation baseline.

## Run the final pipeline

From the repository root, with the supplied data available under `data/`:

```bash
python -m possession_lab.reference_pipeline
```

This calls the modular loader, feature builder, scenario and anchor rules,
transition hypotheses, bounded three-family resolver, and submission writer.
It writes [final_submission.csv](./evaluation/results/final_submission.csv)
and runs the public evaluator. The active policy is:

```text
raw tracking
→ physical and missing-ball evidence
→ football scenarios and anchors
→ H↔A / H-A→D / D→H-A hypotheses
→ bounded interval resolver
→ H/A/D/X
```

The representative full-match candidate is the bounded three-family hybrid. It
ties the forward-settling reference on every public metric (`0.895` macro-F1,
`0.640` spell-F1, `0.486` transition F1@25), but is safer as a mechanism: it
opens `D` only when a dead-ball hypothesis is supported by forward settling,
releases it only with restart or persistent control evidence, and preserves
`CONTESTED` when the evidence is ambiguous.

## Robustness and scope

The public labels are used by the evaluator, not by the inference runner. The
active policy uses physical units, football context, local backtracking and
forward checking, and predeclared sensitivity checks. We do not select a model
because of one public-window threshold win: the public slice contains only a
small number of possession spells and is not representative of the missing-ball
rate in the full match.

George-style alternate-camera reconstruction and other learned or sequence
decoders remain explicit appendix experiments. They can provide future path,
uncertainty, receiver, and dead-ball evidence, but their final labels are never
copied into the promoted resolver.

## Dependencies

```bash
pip install -r requirements.txt
```

The optional raw video-observation appendix has additional dependencies listed
in [ballfill/requirements.txt](./ballfill/requirements.txt). They are not
needed for the canonical submission run.
