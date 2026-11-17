# Barcelona hackathon: ball possession

Our entry for [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027): label every frame of Barcelona v Athletic as `H` / `A` / `D`.

## Public helper API

The active implementation is in `possession_lab/`. The older `possession/`
package is retained for compatibility and comparison baselines. The pipeline is
intentionally modular: notebooks and scripts call reusable functions rather
than reimplementing feature construction or decision logic:

- `build_evidence_table()` — load the supplied data and calculate the complete frame-level feature table;
- `detect_scenarios()` — attach scenario confidence and episode evidence;
- `select_model()` — choose one registered model specification, including the evaluated physics-veto candidate and the preserved `stage_hq_viterbi` diagnostic baseline;
- `run_decision_pipeline()` — pass the feature table and scenario evidence to that selected model function;
- `PipelineSpec` — declare the decision policy without duplicating implementation code.
- `plot_timeline_window()` and `plot_evidence_window()` — render reusable audit views with match-clock axes, scenario/anchor evidence, and an optional public ground-truth strip.

Lower-level calculations remain in `possession_lab/calculations/`, scenario rules in `possession_lab/scenarios/`, and model composition in `possession_lab/models/`. The notebooks are presentation and audit clients of these helpers; they should not become a second implementation of the pipeline.

## Layered architecture

The pipeline keeps evidence construction separate from decision policy:

1. **Measurement layer** — load the supplied tracking/Smart Tagging data and build one shared frame-level evidence table.
2. **Scenario layer** — combine the modular measurements into scenario context and confidence signals.
3. **Anchor/transition layer** — identify stable H/A/D evidence and resolve uncertain intervals conservatively.
4. **Model layer** — the model selector chooses how the shared evidence is handled. Direct policies use rules; decoder policies such as Viterbi/HMM add a transition model over the same evidence.
5. **Evaluation layer** — write the selected submission and score it only when the appropriate ground truth is available.

The runner is deliberately thin: it calls the reusable helpers, passes their shared evidence to one selected model policy, and returns frame-aligned labels. The notebooks are the organiser-facing explanation of the observations; helper modules contain the implementation.

The active implementation is organised under [`possession_lab/`](./possession_lab/):
raw tracking and optional source-aware ball observations become motion,
missing-ball, flight, and arrival features; then scenarios and anchors; then
state-specific transition hypotheses and a shared interval resolver. A live
H↔A candidate asks whether contact became controlled possession, a D→H/A
candidate asks whether a restart is actually active, and H/A→D is marked at a
local stop boundary. The final labels are `H/A/D/X`. Read the notebooks as one
story: **Problem → Idea → Construction → Pipeline → Evaluation**.

The current organiser-facing run is [notebooks/08_run_pipeline.ipynb](./notebooks/08_run_pipeline.ipynb).
It executes the newer `possession_lab/run_interpolated_flight_experiment.py` runner and writes
the `diagnostic_visible_ball_no_switch_interval` candidate. Public scoring and disagreement
plots are in [notebooks/08_model_evaluation.ipynb](./notebooks/08_model_evaluation.ipynb). The earlier registry-based notebook
was retired because it referenced obsolete model names; the current 08 notebook is the only presentation runner.

Visual checks are also helper-first. The evidence timeline is a plotting function in
`possession_lab/calculations/plotting.py`, not notebook-specific labelling logic. The notebooks
choose the frame window and provide optional public ground truth; the helper renders the same
observation lanes, scenario markers, missing-ball shading, anchor margin, and match-clock axis
for every case.

The `stage_hq_viterbi` option preserves the independently developed three-state
Viterbi decoder for a like-for-like experiment. It uses the same canonical
feature/scenario/anchor table as the other models, but retains its simpler
nearest-player, ball-speed, visibility, player-activity, out-of-play, and weak
Smart-Tagging evidence handling. It is a comparison baseline, not a replacement
for the richer physics-veto policy.

Haiqal's branch contributed reusable windowed evidence and a raw
distance/speed/dead-ball Viterbi baseline, which remain comparison material.
George's branch contributed the source-aware ball-observation layer. Its
flight rows now expose displacement, direction, cumulative travel, and a
landing-boundary candidate for long-ball analysis. When enabled, those
observations can affect continuity only inside a tactical missing-ball gap;
visible-ball geometry remains an auditable signal rather than an ownership
override. The
organiser-facing missing-ball walkthrough is
[04b_ball_missing_player_analysis.ipynb](./notebooks/04b_ball_missing_player_analysis.ipynb):
it explains George's alternate-camera sources, uncertainty, derived speed and
displacement, and why those observations must be checked before they influence
scenarios or transitions. This repository consumes George's precomputed CSV;
the alternate-video/alignment producer stays in George's branch.

Kinson's fitted classifier and Viterbi decoder are not part of the promoted
pipeline. They were trained/tuned against event-derived labels, so they remain
a separately documented supervised comparison rather than hidden evidence in a
label-free method. No collaborator's final labels are copied into the active
resolver. The local George coordinate-fill ablation changed upstream motion
enough to reduce the public result, so George is retained as an auditable
observation source and explicit ablation, not an automatic possession-label
override.

The canonical feature builder keeps one preliminary scenario lane,
`scenario_context_observation`, before anchor selection and one final
`scenario_context` lane after it. The first is an audit of what motion,
visibility, restart, and missing-ball evidence says before ownership anchors;
the second is the scenario context consumed downstream. Neither is a possession
label, so we do not maintain a redundant third “anchor-refined scenario” field.

`evidence_bounded_deadball` is an additional diagnostic policy: Smart Tagging
provides restart context, while D is released only where a post-restart H/A
control anchor is visible. It does not impose a fixed dead-ball duration.

## Loading the data

```bash
pip install -r requirements.txt
jupyter notebook notebooks/00_load_data.ipynb
```

- `possession_lab/baseline_pipeline.py` reads the match into numpy/pandas: ball and player positions in metres, team per track, Smart Tagging, homography and the public 5-minute ground truth. Row `i` is video frame `i`, the same as a submission row.
- [`notebooks/00_load_data.ipynb`](./notebooks/00_load_data.ipynb) walks through it with plots.


## Open Bundesliga data (for training)

7 Bundesliga matches with per-frame ball possession and in-play flags: the IDSSE dataset (Bassek et al., *Scientific Data* 2025, CC-BY 4.0).

- Download: https://huggingface.co/datasets/pysport/idsse-data
- Load with [kloppy](https://github.com/PySport/kloppy): `kloppy.sportec.load_open_tracking_data(match_id=...)`
- Paper: https://doi.org/10.1038/s41597-025-04505-y

## Credits

`games.py`, `starter/` and `evaluation/gt/` are from the challenge repository, MIT licence (see `LICENSE-more-than-a-hack`).
