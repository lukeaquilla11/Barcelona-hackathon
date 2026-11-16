# 5. Models and results

The implementation is rule-first because the challenge does not provide human-labelled training data. Models are consistency layers over physically meaningful evidence.

## Active pipeline

```text
load/audit → distances → temporal features → scenarios/anchors
→ bounded reception resolver → protect D/X → H/A/D/X submission
→ Macro-F1 + Spell-F1 evaluation
```

The canonical runner is [`run_interpolated_flight_experiment.py`](./run_interpolated_flight_experiment.py). Reusable functions live in [`baseline_pipeline.py`](./baseline_pipeline.py), [`temporal_rules.py`](./temporal_rules.py), and [`decoder_experiments.py`](./decoder_experiments.py).

## Benchmarks as modular pipeline specifications

Each benchmark is a `PipelineSpec` in [`models/pipelines.py`](./models/pipelines.py), not a separate copy of the whole algorithm. A specification chooses:

```text
anchor_source
use_smart_set_piece
use_dead_ball_boundary
transition_rule
decoder
protection rules
```

The shared feature table is built once. [`apply_pipeline_spec`](./models/pipelines.py) then applies the selected layers in order:

```text
common engineered features
  → selected anchor source
  → optional dead-ball boundary
  → selected transition/reception resolver
  → optional decoder
  → protected H/A/D/X output
```

This makes an ablation explicit: removing an anchor or scenario is a declared benchmark choice, not an accidental code edit. The canonical submission remains `protected_aligned_reception`; duration decoding is a diagnostic specification until it improves both Macro-F1 and Spell-F1.

### What “model” means here

In this project, a model is a callable decision pipeline, not necessarily a neural network:

```text
features → anchor/scenario evidence → transition logic → optional decoder → labels
```

`PipelineSpec.decoder` selects one alternative sequence algorithm while keeping the input feature table and evaluation fixed. This lets us ask a fair question: *does this decoder improve the same evidence?*

| Decoder choice | What it is | What it assumes | Current role |
|---|---|---|---|
| `none` | Rule/anchor output | Explicit evidence is enough | Main transparent reference |
| `viterbi` | Viterbi-style dynamic program | Per-frame scores plus transition costs | Diagnostic comparison |
| `selective_viterbi` | Viterbi only on ambiguous frames | Stable anchors are protected | Diagnostic comparison |
| `hmm` | Anchor-initialized HMM | Fixed Gaussian emissions and transition probabilities | Diagnostic, not label-trained |
| `hsmm` | HMM plus duration cleanup | States should persist for realistic durations | Diagnostic, not label-trained |
| `bayesian_hmm` | HMM with smoothed transition counts | Algorithmic anchors are noisy observations; Dirichlet prior avoids zeros | Diagnostic |
| `duration_viterbi` | Scenario-aware Viterbi-style decoder | Flight/reception episodes should not flicker | Diagnostic |

These are alternatives, not a recipe to run every method one after another. The active path remains rule-first because the challenge does not provide human-labelled training data. The HMM/HSMM variants are useful experiments only if their assumptions improve Macro-F1 *and* Spell-F1 without overriding hard `D`, anchor, or `X` evidence.

## The active algorithm: bounded reception-switch resolution

The objective is not to predict an event that has not happened. It is to choose a coherent H/A/D/X label path for an offline sequence while avoiding false switches and flickering.

For each candidate reception frame:

1. Find a reliable anchor before it and a reliable anchor after it within the configured maximum gap.
2. Require the two anchors to be different teams; otherwise this is a same-team trajectory, not a switch.
3. Require a controlled-reception signal: receiver distance, local minimum, approach/deceleration, and post-arrival support.
4. Require the candidate receiver team to agree with the later anchor.
5. Rewrite only the bounded interval from the evidence-backed reception toward the later anchor.
6. Never rewrite protected `D` or `X` frames.

In simplified pseudocode:

```text
for reception_candidate in frames:
    before = nearest reliable anchor looking backward
    after  = nearest reliable anchor looking forward

    if before.team != after.team \
       and reception_score >= threshold \
       and receiver_team == after.team:
        switch at the supported reception boundary
    else:
        keep the current team label
```

The resolver is implemented by [`resolve_aligned_reception_switch`](./decoder_experiments.py). It is deliberately conservative: nearest-player identity is used to construct evidence, not accepted as possession truth on every frame.

## Objective and dimensions

The model operates on a time-by-feature table:

```text
rows    = tactical-video frames
columns = supplied and engineered observations
output  = one label per frame: H, A, D, or X
```

The optimisation target is a balance of:

- per-frame class quality, measured by Macro-F1;
- temporal spell quality, measured by mean Spell-F1;
- sensible transition timing and low fragmentation as diagnostics.

We do not optimise a raw distance error. We are not estimating player coordinates; those are inputs. We are selecting the most defensible label path from evidence while preserving dead-ball boundaries and possession continuity.

## Model families

| Model | Role | Status |
|---|---|---|
| Nearest-player + set-piece | Transparent starting reference | Baseline only |
| Temporal anchors | Stable control from distance, margin, persistence, and pressure | Core evidence |
| Reception/switch resolver | Bounded bidirectional evidence around opposite anchors | Preferred upgrade |
| Scenario confidence layer | Scores control, flight, contest, reception, occlusion, and dead ball | Gating/diagnostic layer |
| Viterbi-style dynamic programming | Coherent path with evidence and switch penalties | Consistency layer |
| HMM / HSMM experiments | Probabilistic sequence formulations | Diagnostic; not automatically superior |

## Viterbi, HMM, and HSMM

The active dynamic program is Viterbi-style: it finds the best state path from per-frame evidence and transition penalties. It is not a fully learned HMM because transition and emission parameters are not estimated from human labels.

For states `S = {H, A, D}`, let `E_t(s)` be the evidence score for state `s` at frame `t`, and let `C(s_prev, s)` be the transition cost. The dynamic-programming recurrence is:

```text
V_0(s) = start_score(s) + E_0(s)

V_t(s) = E_t(s) + max over s_prev [ V_{t-1}(s_prev) + C(s_prev, s) ]

best_path = backtrack(argmax_s V_T(s))
```

In the duration-aware experiment, switching state subtracts a penalty:

```text
C(s_prev, s) = 0              if s_prev == s
               -switch_penalty if s_prev != s
```

The implementation stores the best score and the `back` pointer for every `(frame, state)`, then walks backward from the best final state. It decodes each graded half separately so the ungraded halftime gap cannot influence either half. After decoding, the duration pass removes only short H/A runs surrounded by the same state; it never removes `D`, and protected anchors are restored.

This is the exact reason it is “Viterbi-style”: it is a max-score path search with backtracking. It becomes an HMM only if the evidence scores are interpreted as emission log-probabilities and the transition costs as transition log-probabilities. Our active resolver is more conservative and rule-constrained than a learned HMM.

An HMM allows a state transition at each frame. An HSMM additionally models how long a state tends to last. That duration idea can reduce one-frame flicker, but must not erase real short events or strong `D` evidence.

The code-level mapping is therefore:

```python
spec = available_pipeline_specs()["anchor_hsmm_diagnostic"]
labels = apply_pipeline_spec(features, spec, rules)
```

The notebook can call the same function with `"protected_aligned_reception"`, `"rule_viterbi_diagnostic"`, `"anchor_hmm_diagnostic"`, or another registered specification. This is a benchmark switch, not hidden tuning.

EM/Baum–Welch or Bayesian parameter fitting would require reliable labels, defensible priors, or a separate labelled corpus. They remain future research rather than hidden tuning against the public score.

## Current measured reference

The latest verified aligned reception candidate is reproduced by the notebook and currently reports Macro-F1 `0.7736` and mean Spell-F1 `0.5677` on the available public sanity evaluation. The notebook displays the current scoreboard after running, so result tables are not maintained manually here.

## Missing-ball trajectory audit

The feature table now also audits internal ball-coordinate gaps using the two
observed endpoints and a small observed context on both sides. On the supplied
match this produced 228 bounded gap rows, of which 33 passed the broad geometric
coherence gate and 162 had positive candidate-receiver reachability support.
This did not change the canonical labels or public score because it is evidence
for future scenario resolution, not an automatic H/A rule. The next principled
use is to let a reception resolver require agreement between this audit, the
receiver persistence features, and the later team anchor.

### Strict same-team trajectory bridge (tested, not active)

The experiment `baseline_same_team_missing_ball_trajectory.csv` implements the
strong version of the proposed rule:

```text
same H/A anchor before and after
    + gap <= 5 seconds
    + coherent endpoint trajectory
    + same-team receiver run support
    -> fill the missing-ball rows with that same team
```

On the public sanity interval it produced Macro-F1 `0.6716` and Spell-F1
`0.5677`. Spell quality was unchanged, but frame quality fell substantially,
so this is retained as a benchmark rather than used in the canonical output.
The result suggests that endpoint agreement is not enough: some ATD ball gaps
occur inside periods where the anchor definition is still too sparse or where
the straight-line bridge is physically plausible but not the true ball path.

The coordinate-only refinement is also available as
`baseline_same_team_coordinate_path_bridge.csv`. It adds the requirement that
a same-team candidate remains within the configured possession-zone radius at
both observed endpoints and that its endpoint displacement aligns with the
ball displacement. This is the right geometric interpretation of “same player
with the ball” for temporary slots, but it remains a candidate until its
public score is checked.

### Path-agreement switch gate (tested, rejected)

Another label-free hypothesis was to require `receiver_arrival_path_agreement`
before allowing a reception-based H/A switch. It is physically meaningful, but
as a hard gate it removed too much valid reception evidence on the public sanity
frames:

| Candidate | Macro-F1 | Spell-F1 | Decision |
|---|---:|---:|---|
| Preferred aligned reception | 0.7736 | 0.5677 | active |
| Add path-agreement hard gate | 0.6732 | 0.5677 | rejected |

The feature stays in the engineered table for diagnostics and future soft
scoring. It should contribute confidence, not decide the switch alone.
