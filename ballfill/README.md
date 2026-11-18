# ballfill: the ball on every frame

The tactical tracking has no ball on 40.7% of graded frames. This package recovers as much as can be
*observed* from both trackers, then looks for the ball in the tactical **video** inside each gap, then
fills what is still missing from the players, physics and the edges of each gap. Every number below
is checked against held-out truth.

```bash
python -m pip install -r requirements.txt -r ballfill/requirements.txt
export MTAH_DATA_DIR=/path/to/data      # the match bundle, incl. BAR-ATH_tactical.mp4
python -m ballfill.run        # -> cache/BAR-ATH_ball.csv: frame, x, y, source, sigma_m, gap_s,
                              #    p_video, p_carrier, p_dead
python -m ballfill.plot_window 0 5   # x and y over minutes 0-5: sources, gaps, fill (cache/*.png)
python -m ballfill.evaluate   # score every fill method on both test sets
python -m ballfill.tune       # two-fold (by half) parameter selection
```

`cache/` is gitignored. The output is derived from licensed data, so never commit it.

The video steps need the tactical video and a GPU (about 5 hours in all on an M-series Mac). Without
them `run.py` stops at the video step; the trained detectors and detections are in `cache/` on the
machine that made them (ask, rather than recompute). Uses the retained
`possession/{load_data,features,degrade,open_data}.py` compatibility helpers
and the open IDSSE matches (`OPEN_DATA_DIR`) for the pooled carrier classifier and test set C.

## Observed: 63.0% of graded frames

| step | finding |
|---|---|
| Align the broadcast | tactical = broadcast + 4014.2 (1st half) / 26920.1 (2nd), with −1 frame of clock drift per half. Interpolated at sub-frame offsets. |
| Live check | 8.5% of broadcast ball frames come from replays or other non-live shots, detected because their players don't match the tactical players. Those frames disagree with the tactical ball by a median 4.9 m (0.75 m when live), so they're dropped. |
| Camera geometry | Both cameras sit at about (53, 116), behind the y = 68 touchline at halfway. The tactical camera is 30 m high and the broadcast 21.5 m. Player positions agree to 0.28 m with no bias anywhere on the pitch. |
| Balls in the air | Disagreements over 3 m are smooth runs (median 1.2 s) that peak mid-run and point along the camera ray. Both trackers already correct for height, imperfectly. In 215 such flights the along-ray coordinate comes from the straight flight path between the agreed take-off and landing points. |
| Edge reliability | Each tracker is worse within about 1 s of its own losses, so `sigma_m` is widened there and near physically impossible steps. |

Sources: `agree` 36.2%, `tac` 15.2%, `flight` 5.3%, `bc` 3.7%, `disagree` 2.6%.

## Filled: 30.5% of graded frames (37.0% before the video)

A hidden Markov model over "where is the ball", run through each gap:

- **Player states.** Every tracked slot, weighted by a gradient-boosted **carrier classifier**: is this the player on the ball? It uses player-only features (opponent pressure and closing speed, convergence of other players, team shape, the Smart Tagging team, within-frame rankings) and the same features 0.4 s before and after. It's trained on our match *and* the 7 open Bundesliga matches (`bank.py`), always excluding the match or half being scored. Top-1 accuracy on held-out data: 50.6% on our match and 50.7% on an unseen match, up from 42.9% and 32.8% when trained on our match alone.
- **Ball states.** The ball resting where it was last or next seen, where play stopped (or the goal mouth, when the gap ends in a kick-off), and the ball continuing or arriving with its velocity at each edge.
- **Stoppage detector** (`dead.py`). The probability that play is stopped, frame by frame inside a gap, from player movement and gap context. It's trained on the open matches' dead/alive status: AUC 0.963 and 90% accuracy, holding out one match at a time. It raises the weight of the resting states during stoppages. It deliberately does **not** take weight off the players, because in a real stoppage someone picks the ball up and carries it. Forcing it onto fixed spots made our real gaps worse (see the ablation below).
- **Transitions.** Keep the carrier, follow the tracker's identity swaps, and pass to nearby players. Both sightings pin the ends. The estimate is the posterior-weighted position.

### Three test sets (truth never seen by the model)

| set | what | truth |
|---|---|---|
| **A** | 88 real tactical gaps in our match | the live-verified broadcast ball: 5,861 frames, mostly near gap edges |
| **B** | 357 stretches of our observed track, hidden on purpose | the hidden tactical ball: 71,922 frames |
| **C** | 528 gaps in the 7 open matches, degraded to look like our tracker (`opendata.py`) | the DFL ball on live frames: 180,699 frames, many deep inside long gaps. DFL freezes the ball while dead, so dead frames aren't scored. |

| median / mean error | A real | B synthetic | C open, live |
|---|---|---|---|
| linear interpolation | 4.83 / 9.76 m | 8.03 / 10.95 m | 18.90 / 22.86 m |
| follow the endpoint players | 3.56 / 9.06 m | 7.51 / 10.82 m | 16.05 / 19.18 m |
| HMM, classifier trained on our match only | 3.14 / 7.78 m | 4.15 / 7.17 m | 15.07 / 22.59 m |
| HMM, pooled classifier | 2.83 / 7.17 m | 3.69 / 6.29 m | 7.07 / 12.71 m |
| **+ stoppage detector (production)** | **2.87 / 7.05 m** | **3.72 / 5.94 m** | **5.03 / 7.79 m** |
| oracle: nearest player to the truth (ceiling) | 1.64 / 3.10 m | 1.85 / 2.99 m | 1.56 / 2.62 m |

## Seen in the video: 6.5% of graded frames, inside the gaps

The tactical video (frame i = tracking frame i, per-frame homography) shows the ball on most gap
frames. No hand labels are used: the detector learns from our own tracking.

```bash
python -m ballfill.video all      # extract, train (2 rounds), detect: ~3 h on an M-series GPU
python -m ballfill.video round3   # clean labels + hard negatives, one more round: ~2 h
python -m ballfill.verify         # tracklet verifier; scores A and B
```

1. **Detector** (`video.py`). Three consecutive frames → a heatmap U-Net (TrackNet-style), trained on
   frames where both trackers agree, on one half, and run on the other, so every detection is out
   of sample. Round 1 learns from the projected tracker ball (~10 px off the ball's centre); its
   peaks become tighter labels for round 2; round 3 adds hard negatives (linesman flags, gloves,
   pitch markings) mined from round 2. Top-5 peaks per frame on every graded frame.
2. **Tracklets** (`vidfuse.py`). Peaks are masked if static (penalty spots, spare balls), linked on
   the pitch with a constant-velocity model.
3. **Verifier** (`verify.py`). A gradient-boosted classifier on 25 tracklet features (score,
   speed, straightness, distance to players, agreement with the path either side of the gap, ...)
   says whether a tracklet is the ball. It is trained on the other half's tracklets with a random
   window hidden around each, so its inputs look like a real gap. Held-out AUC 0.92 / 0.96. Five
   seeded models are averaged (deterministic).
4. **Fill.** Verified tracklets (p >= 0.6, chosen on B) are the ball; the HMM fills between them.

In production 10,327 gap frames (6.5% of graded frames, 18% of the gap frames) become video
sightings; on A their median error is 0.55 m. Raw detector on A frames (truth = live broadcast): when the top peak scores >= 0.7, 91% are within
2 m of the truth.

| median / mean error (share within 2 m) | A real | B synthetic | D broadcast lost the ball |
|---|---|---|---|
| HMM alone (previous production) | 2.87 / 7.05 m (35%) | 3.72 / 5.94 m (32%) | 2.53 / 6.68 m (44%) |
| **video sightings + HMM (production)** | **1.42 / 6.11 m (56%)** | **0.96 / 2.96 m (68%)** | **0.95 / 2.47 m (69%)** |

D (`bothvid.py`): 71 real stretches where the broadcast tracker lost the ball but one tactical
tracker kept it; that tracker is hidden and is the truth. C has no video, so it can't score this.

### Tried for the video and not used

| what | result |
|---|---|
| WASB (BMVC 2023, pretrained football ball detector), full frame at 512x288 | 1% of frames within 2 m (ours: 63%) on held-out frames; our 6-8 px ball shrinks to ~2 px. Its tiled mode ran at ~150 s/frame on this machine and was not pursued. |
| A ball detector on the **broadcast** video (`bvideo.py`), same method, labels from the broadcast tracking | Accurate when confident (score >= 0.5: 81-100% within 2 m) but rarely confident: 46 s of the 565 s of live broadcast frames inside our gaps. Adding it to the tactical video did not improve D (0.96 / 2.54 m vs 0.95 / 2.47 m). |
| Score threshold only, no verifier | A median 2.38 m but mean 9.29 m: false peaks with high scores. |

### Tried for the interpolation and not used

Each scored on A / B / C (median / mean):

| what | A | B | C |
|---|---|---|---|
| auto-ARIMA, Kalman-smoothed through the gap (`tsfill.py`) | 13.33 / 16.53 m | 12.75 / 16.59 m | 21.84 / 26.00 m |
| Prophet (`tsfill.py`) | 6.05 / 10.52 m | 9.38 / 12.08 m | 19.24 / 23.12 m |
| learned imputer: set transformer over players + Bi-LSTM, trained on 7 open matches + ours (`imputer.py`) | 4.12 / 8.43 m | 4.02 / 5.90 m | not run |

The ball's own history (ARIMA, Prophet) does worse than a straight line: the ball's path in a gap is
decided by the players, not by its past. The learned imputer did not beat the HMM.

### Ablation (`ablate.py`): what was tried and rejected

Selection rule: best on C (independent matches), and no worse than the previous model on A.

- **Pass segments** (Viterbi carrier path, passes drawn as timed straight lines): worse on all three sets (C mean 15.8 m, A 7.5 m). Committing to one path costs more than averaging gains when the model is unsure.
- **Stoppage detector that also down-weights players**: C mean 9.7 m but A worsened to 8.4 m. The damage concentrated on frames called "stopped" (13.4 → 17.0 m), mostly 3 gaps including the 2nd-half goal. The ball wasn't at the exit or restart spot; a player had it.
- **Without the exit-point state**: within noise of the production model (C 7.81 m, A 7.16 m).

### Uncertainty (`sigma_m`)

Video frames: the median out-of-sample error of video sightings on A. Filled frames: for each bin of seconds-to-nearest-sighting, the larger of the out-of-sample median errors on A and C, made non-decreasing. C contributes the deep-gap bins that A can't measure. `python -m ballfill.run` prints the table.

## Known limits

- The truth in A is itself a tracker (roughly 0.5–1 m error), and there are only 12 real gaps over 30 s.
- Ball position **during stoppages** is only validated on A. C's dead-ball positions are placeholders.
- The carrier classifier is still the main headroom: the production median on C is 5.0 m against the oracle's 1.6 m.
- The video's gains are measured on A, B and D, all from our match; the open matches have no video.
- The verifier threshold is flat between 0.1 and 0.7 on A, so A's mean moves by about 0.5 m with it.
- Aerial balls: the video gives the ball's image position, projected onto the ground, so a ball in
  the air is placed along the camera ray, too far from the camera.

