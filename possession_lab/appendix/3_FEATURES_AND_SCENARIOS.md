# 3. Feature and scenario engineering

The pipeline progresses from reliable information to weaker inference. Each stage produces evidence for the next stage; it does not blindly overwrite the label.

## Stage 1 — supplied frame geometry

| Feature | Meaning |
|---|---|
| `frame` | Tactical frame index and time order |
| `ball_x`, `ball_y` | Ball position in pitch coordinates |
| `*_x`, `*_y` | Player position in pitch coordinates |
| `closest_team`, `closest_slot` | Team and slot of the nearest visible player |
| `closest_distance_m` | Distance from ball to that player |
| `team_distance_margin_m` | `away_distance - home_distance`; positive favours Home |
| `h_players_within_radius`, `a_players_within_radius` | Local support and pressure |
| `h_visible_players`, `a_visible_players` | Tracking completeness |

Player-ball distance is `sqrt((player_x - ball_x)^2 + (player_y - ball_y)^2)`.

## Stage 2 — temporal and spatial context

We derive lagged, rolling, and movement features: distance changes, closing rates, rolling mean/minimum/variability, team persistence, ball displacement/speed/acceleration/direction, opponent density, teammate support, pressure, pass options, projected arrival time, arrival-time margin, pitch zones, and tracking validity.

## Stage 3 — scenario identification

| Scenario | Evidence | Possession rule |
|---|---|---|
| Dead ball / restart | Smart Tagging plus low ball activity, out-of-play/missing ball, or whole-pitch low motion | Keep `D` while genuinely inactive |
| Comfortable control | Small ball distance, clear team margin, persistence, low pressure, low ball speed | Strong H/A anchor |
| Ball in flight | Fast displacement, acceleration, direction, or short-gap interpolation | Keep previous H/A; do not switch on nearest-player crossing |
| Contest / duel | Small team margin, both teams close, high opponent density | Wait for reception evidence |
| Reception candidate | Local distance minimum, closing player, deceleration, path/arrival agreement | Candidate switch only after control persists |
| Short pass | Short flight and same-team receiver evidence | Keep the team through the trajectory |
| Long ball / switch | Longer flight, projected receiver, arrival-time advantage | Switch only on confirmed control |
| First-time pass | Ball leaves quickly but teammate path and later control agree | Do not call it a failed reception |
| Occlusion | Ball coordinate missing | Use bounded surrounding evidence; not a turnover |

## Bidirectional offline scenario logic

```text
previous stable anchor  →  current trajectory  →  later controlled receiver

### Smart Tagging as a selective clue

The full-pipeline notebook includes an automatic pitch plot for an internal
missing-ball gap. It shows observed ball segments, a dotted missing bridge,
player positions, and nearest-player geometry. This is useful evidence because
one Smart Tagging interval can span preparation, flight, and reception while
the geometry changes inside it.

Operational rule:

```text
Smart Tagging event → open a scenario window
ATD geometry        → decide whether the window is dead ball, flight, contest,
                       or possible reception
controlled receiver → permit an H/A transition
```

So Smart Tagging is checked when it identifies a special context, but its
`Team` field is not copied across the entire interval as frame-level truth.

The feature table now exposes code-specific context flags such as
`smart_set_pieces`, `smart_long_ball`, `smart_attacking_transition`,
`smart_defensive_transition`, `smart_build_up`, `smart_progression`,
`smart_ball_in_final_third`, `smart_ball_in_the_box`, and `smart_goals`, plus
`smart_restart_type`. Their intended uses are:

| Context flag | Follow-up check | Direct H/A label? |
|---|---|---|
| `smart_set_pieces` / `smart_restart_type` | low activity, boundary, and later controlled receiver | no; candidate `D`/restart window |
| `smart_long_ball` | ball flight, projected receiver, arrival and post-arrival control | no; flight/reception scenario |
| attacking/defensive transition | bounded switch evidence and anchor agreement | no; switch-risk scenario |
| build-up/progression/final-third/box | pressure, support, and team control | no; spatial context |
| `smart_goals` | post-goal dead-ball boundary and restart control | only with boundary evidence |
```

The whole interval is one scenario episode. During it we keep the current H/A label because a ball travelling between teammates does not mean possession changed. A switch is allowed only when later receiver/control evidence agrees with the opposite anchor. `D` and `X` remain protected.

## Hierarchy

1. Outside graded intervals → `X`.
2. Strong dead-ball evidence → `D`.
3. Confirmed controlled reception → H/A transition if supported.
4. Stable comfortable control → H/A anchor.
5. Flight, contest, pressure, or occlusion → preserve the current label.
6. Unresolved graded evidence → `X` rather than an invented switch.

The selected source is exposed as `selected_anchor_source`, with the team in `selected_anchor_team` and strength in `selected_anchor_confidence`. This makes it possible to inspect whether a decision came from a restart, pressure-supported control, ordinary control, or no anchor.
