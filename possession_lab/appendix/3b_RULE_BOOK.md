# 3b. Rule book: anchors, scenarios, transitions

This is the operational policy behind the pipeline. It is deliberately
general: it reads signals from the match rather than naming particular frame
numbers or manually assigning labels.

## 1. The label hierarchy

Rules are applied in this order. A later rule cannot casually overwrite an
earlier, stronger decision.

```text
outside graded ranges ───────────────────────────────→ X
explicit dead-ball / restart evidence ──────────────→ D
confirmed restart control ───────────────────────────→ H or A
stable controlled possession anchor ────────────────→ H or A
bounded reception/switch evidence ──────────────────→ H or A
scenario evidence without a team ───────────────────→ keep current / X
```

| Priority | Evidence | Required signals | Allowed action |
|---|---|---|---|
| 0 | Grading mask | frame is outside either graded interval | force `X` |
| 1 | Dead ball | Smart Tagging context plus low ball/team activity and boundary evidence | protect `D`; do not let a decoder erase it |
| 2 | Restart control | restart team hint agrees with a sustained controlled receiver | create `H`/`A` anchor after control, not merely at the restart whistle |
| 3 | Stable control | ball distance, team margin, persistence, low pressure, stable speed | create a high-confidence team anchor |
| 4 | Genuine reception | local distance minimum, approach, deceleration, receiver/team agreement, post-arrival control | allow a bounded team switch |
| 5 | Scenario only | flight, contest, occlusion, pressure or pass evidence with no team agreement | explain uncertainty; never force a switch alone |

The configured source order is in
[`possession_rules.yaml`](../config/possession_rules.yaml):

```yaml
anchor_hierarchy:
  ordered_sources:
    - restart_control_anchor_team
    - pressure_anchor_team
    - anchor_team
```

The first available trustworthy source wins. `selected_anchor_team` and
`selected_anchor_source` are created by
[`apply_anchor_hierarchy`](./temporal_rules.py). The policy is therefore
inspectable and can be changed without rewriting the feature calculations.

## 1a. Evidence pipeline

- Ball evidence first
  - `v≈0`
  - restart / Smart Tag context
  - out-of-play coordinates
  - missing-coordinate episodes
- Restart interpretation
  - kickoff: first sustained movement + restart team context
  - set piece: last near-zero frame + post-restart movement
  - taker proxy: nearest player to the ball at `v≈0`
- Possession geometry second
  - nearest team
  - team distance margin
  - teammate support
  - opponent distance / density
  - pressure state
- Control evidence third
  - local speed minimum
  - speed below configured control threshold
  - player-ball proximity
  - persistence after the minimum
- Transition evidence fourth
  - approach
  - deceleration
  - receiver/path agreement
  - later controlled anchor
  - a transition anchor is a switch candidate, not immediate comfortable possession
  - inspect the next configured frames for follow-through by the proposed team
  - promote only after the minimum follow-through count; otherwise remain sticky/contested
- Long-ball handling
  - flight interval: preserve prior team
  - positive speed hike: departure evidence
  - long-ball score: reduce ordinary control confidence
  - later reception anchor: restore transition confidence
- Missing-ball handling
  - short internal gap: bridge candidate only
  - required: agreeing anchors on both sides
  - required: coherent endpoint trajectory
  - required: receiver-run support
  - otherwise: `OCCLUDED`; no invented speed or turnover

## 2. Scenario catalogue

Scenarios describe *what makes nearest-player possession unsafe*. They are not
automatically labels.

| Scenario | Evidence we look for | What it means for possession |
|---|---|---|
| Controlled possession | short ball distance, clear team margin, persistence, low pressure | nearest-player team is usable as an anchor |
| Short pass / reception | ball leaves one player, reaches a teammate, local distance minimum, post-arrival persistence | same team continues; fill the short interval as one spell |
| Long ball / through ball | sustained ball speed, projected receiver, arrival-time margin, later control | carry the prior team through flight unless the later reception supports a switch |
| Contested ball | both teams close, small team margin, competing arrival times | do not trust the nearest player; wait for control evidence |
| One-frame touch | distance minimum but no persistence and no stable receiver afterwards | do not create a new possession spell |
| Genuine reception | approach + local minimum + deceleration + team agreement + persistence | candidate boundary for a real switch |
| Ball out / restart | out-of-bounds or Smart Tagging restart context, low activity, restart hold | keep `D` until restart control is evidenced |
| Offside / whistle stoppage | no explicit offside tag is supplied; infer only from a supported stoppage/restart boundary | the post-whistle interval is `D`; do not label the preceding live attack as `D` |
| Occlusion / missing ball | missing coordinates followed by reappearance | carry the last supported team only when the bridge is bounded and consistent |
| Pressure / tackle | opponent close, possession player still supported or ball rapidly leaves | marks ambiguity; it does not prove a turnover |
| First-time pass | player receives briefly then ball accelerates away; teammate/receiver geometry agrees | preserve team possession if the next controlled state is the same team |

### Is a Smart Tagging “Long Ball” an H/A anchor?

Not by itself. Smart Tagging does contain a `LONG BALL` phase and a `Team`
field, so it is useful evidence about the team performing the long-ball action
and about which scenario we are in. But the interval is a phase, not an exact
kick or reception boundary. It may include preparation, flight, a deflection,
or the receiver's first touch.

The safe policy is:

```text
LONG BALL + valid Team tag
  → weak initiating-team prior and flight scenario

ATD control before the kick
  → stronger prior team anchor

receiver approach + local minimum + post-arrival control
  → genuine reception / possible switch
```

Therefore, do not assign the Smart Tagging team across the whole long-ball
interval. Keep the current supported H/A label during flight and allow a
switch only when later reception evidence supports the receiving team.

### When is Smart Tagging checked?

Smart Tagging is a *scenario gate*, not the first labeler we consult on every
frame:

1. Use the tag to open a context window such as `set piece`, `long ball`, or
   another phase where nearest-player geometry is known to be unreliable.
2. Within that window, inspect the ATD evidence: ball visibility, distance
   margin, pressure, movement, boundary evidence, and reception features.
3. Use the tag's `Team` only as a weak initiating-team clue. Do not copy it to
   every frame covered by the tag.
4. If the tag conflicts with sustained controlled-receiver evidence, the ATD
   evidence wins for the live H/A label; the tag remains useful for explaining
   why the frame was treated as a special scenario.

The diagnostic pitch plot in the full pipeline makes this visible: a tag can
cover a broad interval while the ball path, missing bridge, and receiver
evidence change inside it. This is why Smart Tagging is checked selectively,
rather than treated as frame-accurate ground truth.

### Literal out-of-pitch coordinates: useful, but not sufficient alone

The ATD contains genuine coordinates outside `[0,1]` in the graded data, so we
test them. They are strong evidence that the ball crossed a boundary, but they
are not reliable as a complete possession labeler: a tracker spike can be
short, and a boundary coordinate does not identify the restart or the team that
will control the ball next.

The diagnostic `literal_out_of_bounds_diagnostic` marks consecutive literal
out-of-bounds runs as `D`. As a standalone weak pipeline it scored poorly
(Macro-F1 `0.3307`, mean Spell-F1 `0.3294`), but that comparison also removed
the preferred reception/transition logic. The proper ablation adds only the
rule to the preferred output: it changed `2,671` labels while public Macro-F1
remained `0.7736` and Spell-F1 remained `0.5677`. It is therefore a plausible
boundary feature, but not yet a demonstrated score improvement. The active
policy uses out-of-bounds coordinates only to support/extend an already
evidenced dead-ball episode, with low-motion and Smart Tagging safeguards.

`D` is broader than “the ball crossed the line.” It includes any interval in
which the ball is not in play, including an offside or foul whistle, a goal,
and the waiting period before a restart. The current data has restart tags but
no explicit offside tag, so offside-specific detection remains an inference
problem rather than a supplied feature.

## 3. How transitions are found

For each candidate interval:

1. Find the nearest reliable anchor looking backward.
2. Find the nearest reliable anchor looking forward, within the configured
   maximum gap.
3. If both anchors are the same team, treat the interval as one trajectory;
   do not search for a turnover.
4. If anchors differ, search inside the interval for a genuine reception.
5. A switch is allowed only when reception evidence agrees with the later
   anchor and the receiving team remains controlled afterwards.
6. If no reception is supported, retain the earlier supported team through the
   ambiguous interval (or preserve `D`/`X` when protected).

This uses future evidence because the submission is offline. The look-ahead is
bounded by configuration (`short_pass_max_frames`, `long_ball_max_frames`, and
`interpolation_max_frames`); it is not a free search over the whole match.

## 4. Confidence policy

The scores are evidence scores, not calibrated probabilities. A signal gets
stronger when multiple independent observations agree:

```text
control confidence
  = closeness + team-distance margin + persistence + low pressure

reception confidence
  = approach + local minimum + deceleration
    + receiver/team agreement + post-arrival persistence

switch confidence
  = reception confidence
    × agreement with the future anchor
    × bounded-interval support
```

The implementation clips component scores to `[0, 1]`, combines them using the
configured weights, and records the component columns for audit. A small
scenario margin means “uncertain”; it does not mean “switch”. See
[`4_CONFIDENCE_AND_THRESHOLDS.md`](./4_CONFIDENCE_AND_THRESHOLDS.md) for the
formal score definitions and current thresholds.

## 5. Decoder policy

The rule book produces evidence. A decoder may make the final path less
flickery, but it must respect protected evidence:

```text
rules/anchors → optional transition resolver → optional decoder
             → restore D, confirmed anchors, and X outside grading
```

The available alternatives are registered in
[`models/pipelines.py`](./models/pipelines.py):

- `none`: transparent rule output;
- `viterbi`: max-score path with transition penalties;
- `selective_viterbi`: same, but only on ambiguous frames;
- `hmm`: anchor-initialized probabilistic sequence model;
- `hsmm`: HMM-style path with duration constraints;
- `bayesian_hmm`: transition counts smoothed with a Dirichlet prior;
- `duration_viterbi`: scenario/duration diagnostic.

These alternatives should be compared on the same feature table. We do not
stack all of them or tune them to a public score. The canonical submission is
the best rule-first configuration that preserves both Macro-F1 and Spell-F1.

## 6. When there is no usable anchor

Do not invent a team label. Keep the current supported team only if the gap is
bounded and the scenario is compatible with continuity. Otherwise leave the
frame unresolved for the final offline policy. Outside the graded intervals,
the answer is always `X`, regardless of the scenario.

### Missing-ball trajectory audit

For an internal ball-coordinate gap, the pipeline also records:

- `ball_missing_gap_frames` / `ball_missing_gap_seconds` — gap duration;
- `ball_gap_displacement_m` and `ball_gap_implied_speed_mps` — endpoint distance
  and implied straight-line speed;
- `ball_gap_direction_change_rad` — direction continuity in the observed context;
- `ball_gap_line_residual_m` — deviation of nearby observed points from the
  endpoint line;
- `ball_gap_receiver_team`, `ball_gap_receiver_closing_speed_mps`, and
  `ball_gap_receiver_run_supported` — whether a candidate receiver is moving
  toward the reappearing ball.
- `gap_player_path_team`, `gap_player_endpoint_distance_m`, and
  `gap_player_path_alignment` — whether a same-team player remains in the
  possession zone at both observed endpoints and moves in a direction aligned
  with the ball displacement.

These are plausibility signals, not hidden-path labels. A coherent bridge can
support a short-pass, flight, or reception scenario, while a candidate run can
support a receiver hypothesis. Neither signal alone is allowed to force an H/A
switch; the surrounding anchors and controlled-reception evidence still win.
