# Notebook guide

Read the notebooks as one concise story: **problem → evidence → football
scenarios → anchors → transition hypotheses → final submission → audit**.

1. [00_load_data.ipynb](./00_load_data.ipynb) — load the supplied tracking and Smart Tag data.
2. [01_eda.ipynb](./01_eda.ipynb) — show why full-match labelling is difficult, especially missing balls and missing players.
3. [02_feature_engineering.ipynb](./02_feature_engineering.ipynb) — derive speed, direction, arrival, pressure, activity and gap evidence.
4. [03_scenario_engineering.ipynb](./03_scenario_engineering.ipynb) — express those measurements as football situations.
5. [04_deadball_setpieces.ipynb](./04_deadball_setpieces.ipynb) — locate stoppage and restart boundaries.
6. [04b_ball_missing_player_analysis.ipynb](./04b_ball_missing_player_analysis.ipynb) — inspect two-sided missing-ball and missing-player cases.
7. [05_anchors_rulebook.ipynb](./05_anchors_rulebook.ipynb) — identify stable control evidence.
8. [06_transition_markers.ipynb](./06_transition_markers.ipynb) — build local transition intervals.
9. [07_probability_transition_model.ipynb](./07_probability_transition_model.ipynb) — compare KEEP, SWITCH, ENTER_D, RESTART and CONTESTED hypotheses.
10. [08_run_pipeline.ipynb](./08_run_pipeline.ipynb) — run the promoted bounded three-family hybrid and write `final_submission.csv`.
11. [08b_model_evaluation.ipynb](./08b_model_evaluation.ipynb) — show a small set of public disagreement diagrams and turn them into case ideas.
The optional source studies are in [appendix/](./appendix/):
[04c_alternate_ball_observations.ipynb](./appendix/04c_alternate_ball_observations.ipynb)
[05_independent_observation_baseline.ipynb](./appendix/05_independent_observation_baseline.ipynb)
and [09_experimental_results.ipynb](./appendix/09_experimental_results.ipynb).
They are not silently merged into the final labels.

Every notebook calls modular functions from [../possession_lab/](../possession_lab/).
The canonical command is:

```bash
python -m possession_lab.reference_pipeline
```

The public slice is an evaluation check only. The active runner does not read
its labels, and model selection favours stable football signals over a brittle
single-window threshold.
