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

The current organiser-facing winner is `diagnostic_visible_ball_no_switch_interval`
(Macro-F1 `0.8917`, Spell-F1 `0.6207`, Transition F1@25 `0.4571` on the public
five-minute slice). Its resolver is the balanced-current profile. George,
Haiqal, and Kinson remain optional observation adapters: they may add coverage
or auxiliary evidence on the private full match, but their final labels never
override our anchors or interval hypotheses. With the available
`data/ball/BAR-ATH_ball.csv`, George's observed/fused fill ablation scored
`0.8723 / 0.6000 / 0.3889`, below the winner, so it is not enabled by default.

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
- optional `supervised_training/`, `supervised_bridge.py`, and `fit_haiqal_params.py` from Kinson's latest branch commit — kept as a separately validated supervised/calibration experiment because they require external labelled matches and are not safe to merge into the label-free public-slice resolver;
- [haiqal_baseline.py](./haiqal_baseline.py) — isolated raw distance/speed/dead-ball baseline adapted from `origin/haiqal`; it does not replace our scenario or anchor decisions;
- `diagnostic_haiqal_visible_ball_no_switch_interval` — evidence-only integration of Haiqal's closeness observation into the best visible-ball + NO_SWITCH resolver. It is retained as a reproducible ablation, not promoted: it currently produces exactly the same labels and public metrics as the best model.
- [pipeline.py](./pipeline.py) — stable orchestration wrappers for loading data, building observations, adding scenario/probability/transition evidence, adding Haiqal observations, and running a registered model;
- [scenario_decoder.py](./scenario_decoder.py) — scenario-level decoding helpers;
- [broadcast_alignment.py](./broadcast_alignment.py) — generic broadcast/tactical alignment audit;
- [george_ballfill_adapter.py](./george_ballfill_adapter.py) — optional, source-aware adapter for
  George's fused broadcast/tactical ball observations. It preserves the tactical ball, records
  source and uncertainty, and fills only eligible tactical gaps before feature construction;
- [run_george_ballfill_experiment.py](./run_george_ballfill_experiment.py) — runs the current best
  visible-ball + NO_SWITCH resolver as a separate George-observation ablation and can compare the
  upstream motion/scenario signals against the no-George feature table. With the current local CSV
  it fills 16,140 tactical gaps, changes 7,751 final labels, and lowers the public score, so this
  remains an explicit ablation rather than the promoted model;
- [run_interpolated_flight_experiment.py](./run_interpolated_flight_experiment.py) — the canonical
  full runner; pass `--george-ballfill /path/to/BAR-ATH_ball.csv` to place George observations at
  the start of the same feature/scenario/anchor/transition pipeline;
- `diagnostic_channel_profile_anchor_dominant`, `diagnostic_channel_profile_balanced`, and
  `diagnostic_channel_profile_independent_support` — predeclared transition-channel ablations.
  They combine anchor agreement, ball motion/contact, player arrival, restart continuity, and
  bounded Haiqal support without importing either collaborator's labels. On the public slice all
  three are exact label no-ops relative to the current best; they remain useful as full-match
  audit outputs rather than promoted score improvements;
- `diagnostic_contact_arrival_local_boundary` — rejected local-boundary handoff guard. It tests
  contact plus team-arrival consensus without suppressing a whole restart interval; on the public
  slice it fell to Macro-F1 `0.8833`, so the consensus/continuity features remain evidence columns
  while this hard resolver stays in the appendix;
- `contact_arrival_coherence` — retained soft evidence channel. It links a plausible ball-contact
  event (deceleration or direction change) to a bounded future team-arrival/reception/post-control
  observation. It is exposed in the scenario-confidence and transition-ledger audits, but it did
  not change any full-length labels or public metrics in the current run, so it is not promoted as
  a hard handoff rule;
- `--kinson-model /path/to/model.joblib` — optional Kinson-compatible supervised evidence
  adapter. It attaches frame-aligned H/A/D probabilities after our observation and feature layers,
  then tests capped transition support (`0`, `0.02`, `0.04`, `0.08`) through our interval resolver.
  The Kinson Viterbi path is also emitted as a standalone diagnostic, never copied into the
  promoted labels. On the available public slice, the sample-only standalone path scored Macro-F1
  `0.6425`, Spell-F1 `0.3590`, and Transition F1@25 `0.2667`; every bounded integration was an
  exact label/score tie with the current best (`0.8917 / 0.6207 / 0.4571`) without George's fill
  being enabled. This is evidence that the resolver protects the pipeline from a weak auxiliary model, not
  evidence that the sample-only classifier is ready for the private match.
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

The current `origin/kinson` pointer is only the branch's initial commit; the later supervised
modules on this branch are the useful Kinson work copied into `e1e77b2`. They require
`scikit-learn`, `joblib`, and optional IDSSE access. The IDSSE provider was not reproducible in
this checkout because its metadata response was empty, so the local Kinson experiment used only
the supplied Metrica games' event-derived silver labels. Those labels are separate from the
hackathon public ground truth and are used only to create a diagnostic auxiliary model.
