# 4. Confidence and thresholds

Confidence is an evidence score, not a learned probability. A score of `0.8` means the available signals support an interpretation strongly; it does not mean an 80% measured chance of being correct.

## Normalising evidence

Raw measurements are mapped to `[0, 1]`:

```text
clip((value - low) / (high - low), 0, 1)
```

Scores combine closeness, margin, low ball speed, pressure, persistence, receiver approach, arrival advantage, deceleration, path agreement, and post-arrival control.

## Current anchor requirements

The configuration in [`../config/possession_rules.yaml`](../config/possession_rules.yaml) currently requires approximately:

- ball distance ≤ `2 m`;
- team distance margin ≥ `1 m`;
- same-team persistence ≥ `80%`;
- at least `8` stable frames;
- low pressure when the nearest opponent is at least `5 m` away.

There is no single margin that proves possession. A one-metre margin for one frame is weak; the same margin sustained with low pressure is much stronger.

## Scenario confidence and margin

For each frame we score control, dead ball, flight, contest, occlusion, and reception. The strongest score becomes `scenario_confidence`:

```text
scenario_margin = best scenario score - second-best scenario score
```

Large margin means one interpretation dominates. Small margin means we should preserve the current label and use neighbouring evidence. High flight/contest/occlusion means nearest-player possession is unsafe.

## Hierarchical protection

```text
X outside grading
  ↓
D dead-ball protection
  ↓
confirmed reception / controlled switch
  ↓
stable control anchor
  ↓
scenario-aware filling and smoothing
```

The decoder cannot smooth away protected `D` or `X` evidence.

## Anchor hierarchy

The order is configuration-backed in [`../config/possession_rules.yaml`](../config/possession_rules.yaml):

1. `restart_control_anchor_team` — a restart is active and tracking agrees with the tagged team;
2. `pressure_anchor_team` — close control remains strong even under pressure;
3. `anchor_team` — ordinary stable low-pressure control;
4. no anchor — leave the frame for scenario/trajectory resolution.

The first available source wins. Flight, contest, occlusion, and reception evidence can explain why an anchor is unsafe, but they do not invent a team label when no team is supported.

## Preventing overfitting

- thresholds live in YAML, not hidden frame-specific code;
- no public labels are used for fitting;
- Macro-F1 and Spell-F1 are read together;
- a feature upgrade needs football meaning before testing;
- ablations and rejected models remain documented;
- no threshold is selected solely because it maximises the public score.

## What a score must not mean

- nearest opponent ≠ confirmed opponent possession;
- ball flight ≠ possession lost;
- pass attempt ≠ turnover;
- Smart Tagging context ≠ exact frame label;
- switch risk ≠ confirmed switch.
