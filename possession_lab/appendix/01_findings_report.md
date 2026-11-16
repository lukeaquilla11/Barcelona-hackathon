# Possession inference: evidence, patterns, and conservative anchors

## Executive summary

This study does not hand-label frames. It builds observable evidence from the
provided tracking and Smart Tagging data, then asks when that evidence is strong
enough to support an operational anchor for possession or a restart.

The central finding is that possession should be treated as a sequence of
evidence-backed states, not as the identity of the nearest player on every
frame. The strongest anchors combine ball motion, player proximity, team margin,
pressure, trajectory continuity, and persistence.

The executable evidence archive is in [`01_findings_full_evidence.ipynb`](./01_findings_full_evidence.ipynb). The
public H/A/D labels are used only to compare the engineered evidence with the
known public interval; they are not used to create frame labels or fit the
feature rules.

## Scope and why the first analysis is narrow

The notebook is deliberately evaluated on the five-minute interval where public
ground truth is available. That makes it possible to compare an engineered
anchor with the first public H/A/D transition without pretending that an
unlabelled frame is ground truth.

This is a narrow *validation window*, not a narrow possession model. The same
features are computed for ball flight, missing-ball bridges, set pieces,
receptions, pressure, transitions, and player-density evidence. The notebook
also audits the full timeline for context, but conclusions about agreement with
ground truth are restricted to the public interval.

## Reusable evidence features

### Ball motion

- `ball_displacement_m`, derived speed, rolling speed, acceleration, and
  direction change are recomputed from ball coordinates.
- A near-zero speed is a restart/control candidate, not automatically H, A, or
  D.
- A large positive speed slope is departure evidence. A local speed minimum is
  the preceding reset/control point, not the kick itself.
- A sharp negative speed change is reception/control evidence.
- A sharp positive change after flight is a pass, deflection, miscontrol, or
  second-kick candidate and must be checked against direction and player
  geometry.

### Ball flight and long-ball evidence

- High speed combined with no player inside a plausible control zone marks a
  high-speed/no-control signal.
- Direction change distinguishes a continued flight path from a possible
  deflection or new pass.
- The closest player at the strongest positive acceleration is used as a
  kicker/player-of-interest proxy, with a backtracked coordinate path for
  visual evidence.
- Distance is integrated from each detected departure until the next supported
  touch, sharp change, or separated acceleration. This produces one row per
  kick-like segment rather than one cumulative distance for the whole window.

### Player geometry and pressure

- Nearest-player distance and team distance margin indicate which team is
  geometrically favoured.
- Opponent density near the candidate, teammate support, pass-option count, and
  isolation describe whether the ball is controllable or contested.
- Opponent closing speed and directional coverage add the missing temporal
  pressure factor: a nearby opponent moving into the ball or blocking short
  options is stronger evidence than static density alone.
- A gegenpress candidate requires pressure, closing movement, and restricted
  short options; it is evidence for uncertainty and turnover risk, not an
  automatic possession switch.

### Missing-ball bridges

Short internal gaps are bridged only when the before/after coordinates support a
coherent path and a plausible receiver run. The bridge records gap duration,
displacement, implied speed, direction change, line residual, receiver closing
speed, and ETA. Missing data is never silently treated as zero speed.

The missing-ball audit now separates continuity from an interaction. If the
ball's speed changes little across the gap, the same temporary player path stays
speed-compatible with the ball, and no opponent closes into the control zone,
the gap is a candidate for same-player dribble continuity. A sharp acceleration
is not treated as automatic continuation: a direction change supports a
pass/deflection/miscontrol interpretation, while a nearby closing opponent
raises tackle/pressure risk. These are evidence flags, not hand-written labels;
only the conservative same-player continuity case is eligible for the diagnostic
bridge candidate.

### Reception and anchor persistence

A reception candidate combines approach, a local ball-speed minimum, proximity,
and a stable nearest-player/team relationship. One-frame proximity is not enough
to switch the whole state.

- Same-team future touch + low pressure can refresh an anchor immediately.
- Opponent future touch requires persistence for several frames before becoming
  a confirmed transition anchor.
- During the ambiguous spell, the system can forecast the likely next receiver,
  but the operational anchor is emitted only when the touch and persistence
  requirements are met.

## Scenario patterns observed

### Goal kicks

The goal-kick signature belongs inside the goal-kick scenario. The shared
primitive is `near-zero → movement → positive acceleration`; the scenario adds
the spatial veto that the near-zero ball starts inside the goal area on either
side of the pitch. A nearby controlled teammate supports a short goal kick. A
large speed hike followed by flight and delayed reception is treated as a long
goal-kick launch, with reduced confidence until the receiver is observed.

The signature should not also appear as a separate top-level finding. Keeping it
in the goal-kick section avoids duplicate evidence and makes clear that speed
alone cannot identify a goal kick.

### Corners and free kicks

Smart Tagging supplies event context, but the public candidate may be outside
the labelled interval. In that case, the physical study still plots speed,
acceleration, trajectory, missing regions, and per-kick distance, while H/A/D
comparison is marked unavailable. Spatial proximity to the pitch corner or
restart location is an additional geometric check when coordinates support it.

### Long balls

The long-ball pattern is a low-control departure, sustained flight, and later
reception or sharp change. The departure is estimated at the strongest positive
speed slope after a preceding low-speed reset. The distance table currently
reports six separated strict candidates in the public scan; this is an evidence
inventory, not a claim that every candidate is a confirmed labelled long ball.

### Contested transitions

At an H↔A boundary, high ball speed without a player in a possession zone is a
strong ambiguity signal. Density, closing speed, directional coverage, pass
options, target geometry, direction changes, and future-touch forecasts are
examined together. The ground-truth boundary is retained as a comparison line;
the engineered anchor is allowed to lag it when the ball is still in flight or
both teams remain unstable.

### Dead ball and out of play

Dead-ball evidence is strongest when Smart Tagging context, near-zero motion,
out-of-bounds geometry, and restart movement agree. A tag alone does not
overwrite possession. The first moving-ball frame identifies departure evidence,
not necessarily the receiving team.

The public-window timing audit shows why D is currently the bottleneck. The
Goal Kick tag begins at frame 18320, while public D ends at 18388 and the first
public H/A frame is 2.76 seconds after the tag begins. The Free Kick tag begins
at frame 17533, public D ends at 17837, and the first public H/A frame is 12.20
seconds after the tag begins. The preferred decoder releases D into H 7.04
seconds before the Goal Kick D interval ends, and it misses one 20.44-second D
interval completely. This indicates that D is not simply a short pre-kick
placeholder: it can include preparation, flight, contest, and unresolved
reception. This is an observation for the decoder, not a fixed-duration rule:
release D only when kick/departure, flight or contest, and controlled-reception
evidence converge. The public timings guide inspection and evaluation; they are
not copied as hard-coded seconds.

## Ground truth versus engineered anchors

The public labels are coarse state labels: H, A, or D. An engineered anchor is a
more conservative control point. Therefore the expected relationship is:

```text
public H/A change → flight/contest/uncertainty → controlled reception anchor
```

For short restarts, the gap can be small. For long balls and contested plays,
the anchor should often lag the public transition because the ball has not yet
been controlled. This is not a failure of alignment; it is the intended
separation between an observable public state boundary and a defensible control
anchor.

## Confidence policy

Confidence increases when independent signals agree: plausible ball trajectory,
nearby player, team margin, low pressure, stable distance, and persistence.
Confidence is reduced by missing ball coordinates, high-speed/no-control flight,
sharp direction change, opponent closing pressure, and conflicting team
evidence. Long-ball confidence is explicitly penalised until reception is
confirmed.

For a long-ball landing or contested transition, the evidence is exposed as an
inspectable bundle rather than a hidden rule: recent ball distance travelled,
peak recent speed, speed-change rate, landing-area opponent density, opponent
approach speed, arrival-time margin, available teammate options, and direction
target. A long ball into a crowded area with opponents closing quickly is not
automatically a turnover; it is a low-certainty reception/contest observation
until a player touch and subsequent control evidence appear. A clean trajectory
into space with a reachable same-team player and later low-pressure control
supports a same-team anchor.

The pipeline is therefore conservative:

1. detect hard restart/dead-ball and missing-data evidence;
2. identify physical departure and flight;
3. forecast a likely receiver only as an anchor candidate;
4. confirm same-team control immediately only under low pressure;
5. require persistence after an opponent touch;
6. fill only short, locally justified gaps.

## What remains to validate

The evidence currently maps kick/departure, reception, contest, persistence,
boundary, dead-ball, and missing-ball anchors. The main remaining validation is
to measure these anchors across more public windows and, where available, compare
them with event-level touch or broadcast evidence. Smart Tagging should remain
context, not be promoted to frame-level truth.

