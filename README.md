# Barcelona hackathon: ball possession

Our entry for [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027): label every frame of Barcelona v Athletic as `H` / `A` / `D`.

## Public helper API

The pipeline is intentionally thin. Use the public façade in `possession/pipeline.py` from notebooks and scripts; it only composes feature construction, scenario detection, and model application:

- `build_evidence_table()` — load the supplied data and calculate the complete frame-level feature table;
- `detect_scenarios()` — attach scenario confidence and episode evidence;
- `select_model()` — choose one registered model specification, including the evaluated physics-veto candidate and the preserved `stage_hq_viterbi` diagnostic baseline;
- `run_decision_pipeline()` — pass the feature table and scenario evidence to that selected model function;
- `PipelineSpec` — declare the decision policy without duplicating implementation code.

Lower-level calculations remain in `possession/calculations/`, scenario rules in `possession/scenarios/`, and model composition in `possession/models/`. The notebooks are presentation and audit clients of these helpers; they should not become a second implementation of the pipeline.

## Layered architecture

The pipeline keeps evidence construction separate from decision policy:

1. **Measurement layer** — load the supplied tracking/Smart Tagging data and build one shared frame-level evidence table.
2. **Scenario layer** — combine the modular measurements into scenario context and confidence signals.
3. **Anchor/transition layer** — identify stable H/A/D evidence and resolve uncertain intervals conservatively.
4. **Model layer** — the model selector chooses how the shared evidence is handled. Direct policies use rules; decoder policies such as Viterbi/HMM add a transition model over the same evidence.
5. **Evaluation layer** — write the selected submission and score it only when the appropriate ground truth is available.

The runner is deliberately thin: it calls the reusable helpers, passes their shared evidence to one selected model policy, and returns frame-aligned labels. The notebooks are the organiser-facing explanation of the observations; helper modules contain the implementation.

The `stage_hq_viterbi` option preserves the independently developed three-state
Viterbi decoder for a like-for-like experiment. It uses the same canonical
feature/scenario/anchor table as the other models, but retains its simpler
nearest-player, ball-speed, visibility, player-activity, out-of-play, and weak
Smart-Tagging evidence handling. It is a comparison baseline, not a replacement
for the richer physics-veto policy.

`evidence_bounded_deadball` is an additional diagnostic policy: Smart Tagging
provides restart context, while D is released only where a post-restart H/A
control anchor is visible. It does not impose a fixed dead-ball duration.

## Loading the data

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_load_data.ipynb
```

- `possession/load_data.py` reads the match into numpy/pandas: ball and player positions in metres, team per track, Smart Tagging, homography and the public 5-minute ground truth. Row `i` is video frame `i`, the same as a submission row.
- `notebooks/01_load_data.ipynb` walks through it with plots.


## Open Bundesliga data (for training)

7 Bundesliga matches with per-frame ball possession and in-play flags: the IDSSE dataset (Bassek et al., *Scientific Data* 2025, CC-BY 4.0).

- Download: https://huggingface.co/datasets/pysport/idsse-data
- Load with [kloppy](https://github.com/PySport/kloppy): `kloppy.sportec.load_open_tracking_data(match_id=...)`
- Paper: https://doi.org/10.1038/s41597-025-04505-y

## Credits

`games.py`, `starter/` and `evaluation/gt/` are from the challenge repository, MIT licence (see `LICENSE-more-than-a-hack`).
