# Transition trust log

This document records the transition methodology as it evolves. It is
deliberately separate from the football labels: the system cannot observe
"true possession" directly, so it must preserve the evidence and the reason
for every inferred change.

## Current model boundary

- An **anchor** is a high-confidence, event-like observation: for example, a
  restart becomes live, a controlled reception is visible, or a ball exits and
  re-enters with a coherent trajectory.
- A **scenario** is structured context that changes how observations should be
  interpreted: `BALL_IN_FLIGHT`, `CONTESTED`, `OCCLUDED`, `RECEPTION_CANDIDATE`,
  `CONTROLLED`, and restart-related contexts are examples.
- A **transition-trust score** is the uncertain bridge between anchors. It is
  computed from several observations and includes a reason, candidate team,
  margin, and a three-way vector (`Home`, `Away`, `no switch`). It is currently
  an evidence score transformed by softmax, not a calibrated probability.
- A **state decoder** may smooth or fill a sequence, but it must not overwrite
  an anchor without an explicit conflict report.

The definition of scenario therefore does not need to become “every possible
combination.” Combinations are represented as features and reasons. A new
named scenario is justified only when it changes the transition rules or the
expected duration, not merely because two observations co-occurred.

## Baseline evidence

The existing public evaluation was run on the generated outputs before the
new trust-gated diagnostic was available. The strongest existing rule-first
variant was `baseline_aligned_physics_veto`:

- macro F1: **0.7736**
- spell F1: **0.5677**
- transition F1 at ±25 frames: **0.3333**
- predicted possessions: **14**
- mean transition latency: **-5.2 frames**

The diagnostic Bayesian HMM and HSMM variants were much weaker on this public
slice (macro F1 approximately **0.39** and **0.33**, with spell F1 near zero).
This is evidence against replacing the rule-first pipeline with an unconstrained
HMM/HSMM. It is not evidence that duration models are useless; it says they
need better emission features, trusted-anchor supervision, and duration priors.

## What helped

1. Bounded the switch search between reliable anchors instead of allowing a
   one-frame nearest-player change to become a turnover.
2. Used reception geometry, player proximity, closing effort, arrival advantage,
   ball-flight context, contact-direction change, and occlusion as separate
   pieces of evidence.
3. Kept the physics veto: a candidate interception should be consistent with a
   meaningful change in ball motion, not merely a player being near the ball.
4. Preserved possession through flight, poor touches, and missing-ball intervals
   when the surrounding evidence supports continuity.
5. Added kick episodes as motion-audit features. They describe acceleration,
   controlled deceleration, direction change, and distance per kick; they do
   not directly label possession.

## What did not help, or is not yet justified

- A universal HMM/HSMM over raw nearest-team observations produced excessive
  switching and poor spell quality.
- Treating a missing-ball bridge as automatically belonging to the same team
  is unsafe when the interval contains a plausible contest or restart.
- Treating a player entering a possession zone for one or two frames as a
  turnover is unsafe for missed interceptions and deflections.
- A softmax-normalised evidence vector is not automatically a probability.
  Calibration requires held-out event labels or a defensible anchor-based
  pseudo-labeling protocol.
- Naming every combination, such as “long ball + aerial cluster + missing
  receiver,” would create a brittle scenario taxonomy. Those observations
  should remain composable features unless they change the state machine.

## Current experiment

The code now computes transition-trust features in
`decoder_experiments.py` and adds a diagnostic pipeline variant named
`baseline_trusted_transition`. It uses the existing bounded reception switch,
the physics veto, and a provisional trust threshold of **0.55**. It is not the
canonical pipeline yet.

The vector is useful for ranking and auditing:

```text
P(Home switch), P(Away switch), P(no switch)
```

but the metadata explicitly marks it `transition_trust_calibrated=False`.
The next legitimate step toward probability is to compare these scores with
annotated transition outcomes, then fit calibration (for example isotonic or
logistic calibration) on held-out matches. Until then, call them scores or
posterior-like weights, not probabilities.

### First full-data result

The gated tactical bundle was found in the sibling staging data root and the
pipeline was rerun against the real match. The new `baseline_trusted_transition`
variant completed successfully.

- `baseline_aligned_physics_veto`: macro F1 **0.773604**, spell F1 **0.567742**
- `baseline_trusted_transition` at threshold 0.55: macro F1 **0.771958**, spell
  F1 **0.567742**
- Both variants produced 14 possessions, transition F1 at ±25 frames **0.333333**,
  and mean latency **-5.166667** frames.
- The exported trust report contained **672 candidate frames**, but none reached
  the provisional 0.55 gate. The resulting variant rejected 115 baseline switch
  frames and reverted to the prior anchor/state.
- Thresholds **0.35, 0.40, 0.45, and 0.50** were also tested. They all retained
  the same public metrics as the 0.55 variant and did not exceed the physics
  baseline.

Conclusion: the current trust score is useful as an audit signal, but this
particular weighted formula is not yet a better switch decision rule. The next
change should be evidence weighting/calibration or a transition-specific error
analysis—not simply lowering the threshold.

### True-switch and restart-activation ablations

The disagreement audit suggested two observations worth testing without using
the public labels during inference:

1. confirm a switch only when post-contact ball deceleration/direction change,
   team-level receiver arrival, and persistence of the new controlled state
   agree;
2. keep `D` until restart-compatible ball movement agrees with stable
   team-level control.

The runner evaluated each signal on top of the current `NO_SWITCH` candidate,
and also evaluated their composition. The direct label rewrites were not
promoted:

- true-switch confirmation: macro F1 **0.8866**, spell F1 **0.3184**,
  transition F1@25 **0.3273**;
- restart activation: macro F1 **0.8234**, spell F1 **0.5600**, transition
  F1@25 **0.3429**;
- both signals: macro F1 **0.8183**, spell F1 **0.2880**, transition F1@25
  **0.2545**.

The first signal appears too eager when used as a direct relabelling rule; the
second has useful boundary information but loses live-state continuity. The
correct next construction is therefore not another threshold sweep: retain
these as evidence channels that adjust interval trust, while keeping explicit
`NO_SWITCH` and `CONTESTED` hypotheses and requiring persistence before an
ownership change.

### Guarded handoff and restart-state follow-up

Two structural follow-ups were then run on top of the current `NO_SWITCH`
candidate. A hysteretic handoff required a later stable anchor run before
accepting a proposed switch. It produced exactly the same output as the
active candidate: macro F1 **0.8917**, spell F1 **0.6207**, and transition
F1@25 **0.4571**. This is a safe no-op, not an improvement.

A bounded restart state machine retained `D` until restart activity and a
sustained control run, then allowed a provisional team and later opposing
anchor handoff. It improved the direct restart-activation ablation to macro
F1 **0.8245**, spell F1 **0.5931**, and transition F1@25 **0.3529**, but it
remained below the active visible-ball `NO_SWITCH` resolver. Adding the
hysteretic handoff did not change that result. Both remain appendix
diagnostics; the active model is unchanged.

### Structural interval-hypothesis result

The next pipeline change now separates transition activity from ownership. It
adds interval-level `SWITCH_CANDIDATE`, `NO_SWITCH`, `CONTESTED`, and `OCCLUDED`
states, with separate team hypotheses and audit weights. The diagnostic output
is `baseline_transition_interval_hypothesis`.

On the real tactical bundle it produced **1,306** intervals covering **8,565**
frames. On the public graded slice, **203** active frames overlapped the
evaluation region; 88 were `SWITCH_CANDIDATE`, 115 were `NO_SWITCH`, and none
were silently forced into a switch merely because an interval was active.
Among the 88 candidate frames, the ownership hypothesis agreed with the public
H/A label on 41 live frames; 47 were dead-ball frames, which is precisely why
the explicit dead/contested alternatives matter.

The public metrics were unchanged from the trust-gated variant (macro F1
**0.771958**, spell F1 **0.567742**, transition F1 **0.333333**). This is not a
performance improvement yet, but it is a better representation: the pipeline
now records whether an event is active separately from who might own the next
state. The interval variant remains diagnostic until transition hypotheses are
aligned with more graded H/A examples.

## Observation ledger

The pipeline now writes `transition_hypothesis_ledger.csv`. It is an
observation-only handoff table for future decoding:

- **1,306** intervals were recorded;
- **1,098** were `NO_SWITCH`;
- **198** were `SWITCH_CANDIDATE`;
- **10** were `CONTESTED`;
- only **22** intervals were marked
  `ELIGIBLE_AFTER_ANCHOR_CHECK`; 10 were held/abstained and the remainder
  retained the prior state.

This is the right place to inspect why a transition was trusted or rejected.
The next decoder should consume this ledger as transition costs and hard
constraints, rather than treating every frame's nearest-team observation as an
emission label.

## How Viterbi and HSMM should fit

The recommended order is:

```text
signals → scenarios → deterministic anchors → transition trust
       → constrained Viterbi/HSMM (optional smoothing) → H/A/D/X output
```

Viterbi is appropriate for selecting the most coherent path through uncertain
frames, provided anchors are hard or near-hard constraints and transitions are
penalised according to scenario. HSMM is appropriate later when trusted data
supports duration distributions for `LIVE`, `DEAD`, `FLIGHT`, `CONTESTED`, and
`CONTROLLED`; it should not be used merely because it is available.

For missing ball, a Kalman or spline bridge can estimate a plausible path, but
it should emit uncertainty and geometry evidence only. It must not decide team
ownership by itself.

## Bottlenecks

- The active checkout's `data/tactical/` directory is empty, but the complete
  gated tactical bundle is available in the sibling staging data root. Runs
  therefore need `MTAH_DATA_DIR` pointed at that verified bundle.
- The public slice has useful evaluation labels but only a small number of
  possessions, so it is not enough to calibrate a rich probabilistic model.
- Tracking noise, camera cuts, ball occlusion, and missing player identities
  make “who touched it?” harder than “where did the ball go?”
- The active checkout does not contain the video asset. The verified staging
  bundle does contain `BAR-ATH_tactical.mp4`; the notebook should use the same
  `MTAH_DATA_DIR` when a synchronized video view is needed.

## Next test after the data bundle is restored

Run the full pipeline and compare the trusted variant against
`baseline_aligned_physics_veto` on the same frames. Inspect transition-level
false positives by reason (`RECEPTION_AND_ANCHOR_EVIDENCE`, `TRAJECTORY_CHANGE`,
`OCCLUSION_BRIDGE`, etc.), then decide whether the threshold should be learned,
calibrated, or replaced by a constrained decoder.

## Constrained Viterbi experiment

`TransitionLedgerViterbi` was added as a diagnostic branch. It consumes the
existing protected rule path as frame evidence and uses the interval ledger
only to change the cost of an H/A switch:

- outside `SWITCH_CANDIDATE`, `CONTESTED`, or `OCCLUDED` intervals, the base
  H/A state is preserved;
- inside an interval, the decoder can compare the current team against the
  explicit no-switch/contested alternatives;
- explicit upstream `D` labels are protected;
- conflicting team anchors do not create an impossible dynamic-programming
  column; dead-ball evidence takes precedence and the conflict remains an
  audit condition.

The first unconstrained version failed badly because it rebuilt emissions from
geometry and erased dead-ball evidence. The transition-specific audit exposed
an additional bug: conflicting dead-ball/team constraints produced all
`-inf` columns, after which backtracking defaulted to H. Fixing precedence and
using the base path as evidence reduced the diagnostic path to only 225 graded
frame changes relative to the protected baseline.

On the public evaluation slice, the final diagnostic result was:

- `baseline_aligned_physics_veto`: macro F1 **0.773604**, spell F1 **0.567742**;
- `baseline_transition_interval_hypothesis`: macro F1 **0.771958**, spell F1
  **0.567742**;
- `baseline_transition_ledger_viterbi`: macro F1 **0.772457**, spell F1
  **0.567742**, transition F1 at ±25 **0.333333**.

Therefore Viterbi is now structurally integrated but not promoted to the
canonical output. It changes the representation and gives us a safer place to
test transition hypotheses, but it has not produced a material public-score
gain. The bottleneck is evidence alignment/calibration, not the absence of a
stronger smoother.

## Current status of the three models

- **Kalman:** active as a conservative ball-path feature estimator, not as a
  possession decision-maker. It supplies path, velocity, displacement,
  direction, acceleration, uncertainty, and reliability for short bracketed
  gaps. The Kalman-only missing-ball bridge is currently identical to the
  ordinary bridge on public evaluation, so it is not promoted to the canonical
  possession output.
- **Viterbi:** active as `baseline_transition_ledger_viterbi`, diagnostic and
  anchor-protected.
- **HSMM:** existing duration-aware branches are explicit finite-state
  Viterbi-style experiments with hand-set minimum durations, not an EM-fitted
  HSMM. It remains diagnostic until duration distributions can be estimated
  from reliable event boundaries.

## Observation completeness and missing-ball causes

The canonical feature table now keeps all available observations together:
ball position/visibility, displacement, speed, acceleration, direction change,
kick episodes, player distance and closing speed, density/support, reception
and control evidence, deflection/tackle risk, restart/boundary context,
tracking validity, Kalman uncertainty, and the transition ledger. The rule is
not that every observation has equal authority. Each downstream layer must
preserve its provenance and missingness, and a kick or nearest-player event
must not directly overwrite an H/A/D state.

Missing-ball intervals are now audited with separate cause hypotheses:

- `DEAD_OR_OUT` — explicit boundary/out-of-play or tagged restart plus low
  player motion;
- `CROWD_OCCLUSION` — both teams/players are geometrically clustered near the
  last or predicted ball area;
- `VISIBILITY_UNCERTAINTY` — tracking validity or reprojection quality is poor;
- `FLIGHT_OCCLUSION` — the visible endpoints and speed support a moving ball;
- `CONTROL_CONTINUITY` — a same-player/path hypothesis is supported;
- `DEFLECTION_OR_CONTEST` — tackle/deflection risk or a sharp change is
  supported;
- `UNKNOWN` — the observations do not justify a stronger explanation.

These are cause scores, not calibrated probabilities and not possession
labels. On the current public slice, the audit found live H/A frames inside
`CROWD_OCCLUSION`, so crowd evidence must increase uncertainty rather than
force `D` or a team switch. The initial `DEAD_OR_OUT` rule also over-counted
active restart context; that was corrected so an active goal kick/corner is
not treated as dead merely because the ball is missing.

## Kalman design constraints

Kalman smoothing is appropriate for estimating a plausible ball path during a
short internal gap. It is not sufficient to infer the touch, team, or restart
state. The useful outputs are the path, velocity, displacement, direction,
acceleration, innovation/residual, covariance, gap length, and a reliability
flag. A future upgrade should use an adaptive or interacting motion model:

- constant velocity for ordinary flight/occlusion;
- higher process noise or a change-point at a detected kick;
- a deflection/measurement-break mode at a sharp innovation;
- no smoothing across an identified out-of-play/dead-ball boundary;
- a one-sided filter for causal use and a two-sided smoother only for offline
  audit/evaluation;
- an uncertainty/NIS gate that stops the estimate from bridging long gaps or
  camera cuts;
- multiple hypotheses when two players or a crowd make the continuation
  genuinely ambiguous.

The current implementation is deliberately conservative: only short,
bracketed gaps with low smoothed uncertainty are marked reliable, and Kalman
features may contribute to kick/trajectory evidence only on those frames.

Latest public rerun after separating kick/no-switch evidence and adding the
missing-ball cause audit:

- `baseline_aligned_physics_veto`: macro F1 **0.7736**, spell F1 **0.5677**,
  transition F1 at ±25 **0.3333**;
- `baseline_transition_ledger_viterbi`: macro F1 **0.7725**, spell F1
  **0.5677**, transition F1 at ±25 **0.3333**;
- `baseline_same_team_kalman_missing_ball_trajectory`: macro F1 **0.6716**,
  transition F1 at ±25 **0.1714**, identical to the non-Kalman bridge.

The current missing-ball audit contains live H/A frames under
`CROWD_OCCLUSION`, so crowd evidence must widen uncertainty rather than force
`D` or a team switch. This reinforces the authority order: deterministic
boundary/restart anchors first, interval-level scenario evidence second, and
Viterbi/HSMM only as constrained path selection after the evidence is formed.

## Haiqal models and probability experiment

The `haiqal` branch contains several model alternatives in its sequence-model
registry: rule-constrained Viterbi, selective Viterbi, anchor-initialised HMM,
an anchor-initialised HSMM-style duration cleanup, and a Bayesian-anchor HMM
with a Dirichlet-smoothed transition matrix. These are alternatives over the
same evidence table, not a chain that should all be run sequentially. The
branch does not contain Kalman filtering or a learned probability model.

The current workspace now adds an explicit observation-likelihood vector over
`H`, `A`, `D`, and `X` in `observation_probability_audit.csv`. It combines the
control, reception, distance-margin, dead-ball, flight, crowd, deflection,
visibility, and anchor evidence channels before a path decoder sees them. The
vector sums to one per frame, but is marked
`RULE_EVIDENCE_SOFTMAX_UNCALIBRATED`; normalization alone does not make it a
calibrated probability.

The optional probability-emission transition Viterbi scored macro F1
**0.7720**, compared with **0.7725** for the existing transition-ledger
Viterbi and **0.7736** for the canonical physics-veto path. It is therefore a
useful diagnostic observation layer, not the promoted label path. The next
probability step is calibration on a separate reviewed transition set, or an
explicitly permitted labelled corpus; the public evaluation labels must remain
evaluation-only.

The current combination is consequently:

1. deterministic anchors and rule-derived observations;
2. scenario/cause evidence vectors;
3. optional normalized observation likelihoods;
4. transition interval hypotheses with no-switch/contested alternatives;
5. Viterbi as the current constrained path smoother;
6. HSMM only after reliable state-duration distributions are available.

The pipeline does not currently combine HMM, HSMM, and Viterbi as three
independent smoothing passes. A learned HMM would provide emissions and a
transition matrix, and Viterbi would decode that HMM. An HSMM would replace
the frame-by-frame transition process with duration-aware segments. Running
all three blindly would double-count the same evidence and make errors harder
to audit.

## Haiqal branch audit

The requested branch is `origin/haiqal` at commit `5092f127`. It contains the
earlier modular scenario evidence and rule-book implementation, including
`possession/calculations/scenario_evidence.py` and the findings/rule-book
notebooks. The current workspace adapted that scenario evidence into
`possession_lab/calculations/scenario_evidence.py`, adding episode-level
significant-player counts, home/away near-ball counts, and a crowd candidate
feature. This is reusable model evidence, not a possession label.

The branch's written findings also confirm the intended policy: long-ball,
deflection, contest, and occlusion are scenario evidence; a switch requires a
later controlled reception and persistence. The branch has no Kalman or
missing-ball-cause implementation, so those remain newer workspace additions.

## Latest structural crowd and transition audit

The first crowd feature was intentionally conservative in authority but too
broad in scope: it marked an entire scenario episode as crowded when four
players had been near the ball at any point in that episode. The revised
feature keeps local same-frame home/away density separate from a weaker
episode-level prior, requires team balance for the crowd candidate, and emits
the continuous `scenario_crowd_score`. It reduced the broad crowd candidate
from 82,124 to 33,778 frames without changing the canonical public output.

The pipeline now also writes `transition_public_error_audit.csv` and
`transition_public_error_summary.csv`. These join the already-fixed interval
hypotheses to the public labels after inference, never during feature
construction. In the graded public windows, the near-switch intervals were
mostly `UNKNOWN` cause with reception/direction/contest evidence; no
missing-ball cause dominated them. This means the next bottleneck is not a
missing-ball threshold. It is timing and ownership resolution around
controlled/reception anchors: too many short `NO_SWITCH` hypotheses, and a
small number of `SWITCH_CANDIDATE` intervals that are not yet aligned to the
true post-switch anchor.

## Iteration toward the 0.80 / 0.60 target

The next experiment tested a structural visible-ball stoppage hypothesis. It
uses a trailing whole-pitch activity window and ball settling evidence; it does
not treat low activity alone as dead ball. A candidate must persist for an
episode, and the rule keeps the existing D boundary when a tagged restart
context supports it. The interval logic then does two bounded repairs:

- bridge an existing D anchor to a later visible-ball settling candidate inside
  the same restart context unless an explicit restart-end observation appears;
- bridge two settling candidates across a short gap only when scene activity
  remains low and there is no high-confidence H/A anchor in the gap.

This explicitly models “the middle observation is not trusted” as an interval
hypothesis rather than overwriting each frame from its nearest player. It is
implemented as `low_motion_dead_ball_evidence` in
`possession_lab/decoder_experiments.py`, with thresholds recorded in
`config/possession_rules.yaml`.

Public evaluation after the full pipeline rerun:

- canonical `baseline_aligned_physics_veto`: macro F1 **0.7736**, spell F1
  **0.5677**;
- diagnostic visible-ball settling candidate: macro F1 **0.8887**, spell F1
  **0.6000**, D F1 **0.9092**, D precision **0.9620**, D recall **0.8618**;
- probability-emission Viterbi remains unpromoted at macro F1 **0.7720** and
  spell F1 **0.5677**.

The candidate reaches the requested public target, but it remains diagnostic
until it is checked on held-out/private data. The public labels were read only
by the evaluator after the candidate was generated; they were not used in the
detector or its configuration. The main remaining bottleneck is now whether a
visible-ball settling episode is a real stoppage across camera views. The next
validation should therefore be a small manually reviewed holdout of live
walking phases, long balls, and restart contexts—not another public-threshold
sweep.

## Mislabel audit added to notebook 08

`08_experimental_results.ipynb` now calls the reusable
`possession_lab/mislabel_audit.py` report builder. It writes frame-level,
contiguous-run, error-summary, and scenario-context reports under
`evaluation/reports/`. The audit joins public labels only after prediction.

For the visible-ball candidate, the 7,500 graded frames contain 846 errors in
19 runs:

- H→A: 351 frames, dominated by long flight/unknown intervals;
- A→H: 242 frames, concentrated around flight, reception-candidate, and
  contested intervals;
- D→A: 181 frames, mainly the early portion of untagged dead-ball episodes;
- A→D: 50 frames, mostly short occlusion/restart-boundary false positives;
- D→H: 22 frames, mostly a late restart-boundary release.

The largest remaining problem is therefore not D recall. It is deciding when a
cross-team transition is trustworthy: the high-entropy H/A errors commonly have
`NO_SWITCH` hypotheses and `UNKNOWN` missing-ball cause. The next improvement
should use this audit to refine receiver/transition ownership evidence, not
expand the dead-ball rule again.

