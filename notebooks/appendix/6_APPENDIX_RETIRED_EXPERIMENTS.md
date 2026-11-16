# 6. Appendix — retired experiments

These methods remain documented for reproducibility and future live-system work, but are not active submission models.

| Experiment/scenario | Why it is not active now | Future use |
|---|---|---|
| Raw nearest-player | Flickers during flight, duels, and pressure | Sanity baseline only |
| Centred smoothing | Uses future frames without explaining the football event; can erase boundaries | Offline comparison, never live |
| Nearest player as automatic switch | A touch, deflection, or opponent pressure is not necessarily control | Candidate evidence only |
| Fixed HMM/HSMM probabilities | No labelled corpus supports trustworthy transition/emission estimates | Fit on a permitted labelled corpus |
| Public-label EM/Baum–Welch fitting | Would turn the sanity set into training/tuning data | Only with separate training matches |
| Broad Smart Tagging → `D` | Tags cover wide intervals and can include live play | Context for boundary detection |
| Long look-ahead for live use | Future evidence creates latency/look-ahead bias | Offline benchmark or delayed live system |
| Aggressive scenario decoder | Can replace good anchors and fragment spells | Research after better reception evidence |
| Broadcast substitution everywhere | Alignment and camera view are not guaranteed frame-perfect | Use only when a generic alignment audit passes |
| Literal out-of-bounds as standalone `D` | Weak standalone pipeline scored Macro-F1 `0.3307`, Spell-F1 `0.3294`; this was confounded by removing the preferred transition logic | Keep as supporting boundary evidence; isolated addition changed 2,671 labels but left preferred public scores unchanged at `0.7736` / `0.5677` |
| Haiqal raw distance/speed/dead-ball Viterbi | Useful independent baseline, but it does not consume our scenario or anchor evidence; public Macro-F1 `0.7677`, Spell-F1 `0.5143` | Keep as a comparison baseline in 09; use its modular raw clues only as auditable observations |
| Haiqal clues + our probability/transition ledger | Tests whether the independently derived raw clues add information to the layered resolver | Public scores matched the probability-ledger path (`0.7720` / `0.5677`), so the clues are currently redundant or underweighted |
| Haiqal Bayesian-anchor HMM and anchor HMM/HSMM | Tests whether a Dirichlet transition prior or fixed duration assumptions can regularise noisy anchors | Much lower public scores; retain as diagnostic code and explanation, not as an active decoder |
| Kalman-supported missing-ball bridge | Tests whether a geometrically plausible hidden ball path improves possession continuity | Only a small number of reliable bidirectional paths were available and the complete policy did not improve; keep Kalman upstream as trajectory uncertainty, never as ownership truth |
| MLE temperature calibration | Tests whether the normalized evidence vector is too sharp or too flat | Holdout log loss and Brier score worsened; calibration remains a separate experiment until a genuine reviewed calibration set exists |
| EM latent-state fitting | Tests whether states and emissions can be learned without labels | Not promoted: it needs label-switching, anchor-violation, and stability safeguards before it can be compared fairly |
| NO_SWITCH + true-switch confirmation | Tests post-contact motion change + team-level receiver arrival + persistence as a direct ownership rewrite | Public macro F1 **0.8866**, spell F1 **0.3184**, transition F1@25 **0.3273**; too aggressive and fragmented spells, so retain as evidence only |
| NO_SWITCH + restart activation | Tests holding `D` until restart-compatible movement and stable team-level control | Public macro F1 **0.8234**, spell F1 **0.5600**, transition F1@25 **0.3429**; useful timing signal but not a better complete decoder |
| NO_SWITCH + both transition signals | Tests composing the two direct rewrites | Public macro F1 **0.8183**, spell F1 **0.2880**, transition F1@25 **0.2545**; composition amplified the failure modes |

The active system keeps the useful parts of these ideas as evidence features, but does not let them override stronger anchors or protected `D`/`X` labels.

## Potential signals worth pursuing

The disagreement audit does not justify adding many named scenarios. The
remaining errors are long, ambiguous runs concentrated in `H↔A` live-ball
segments and `D→H/A` restart boundaries. The two highest-value, label-free
signals are therefore:

1. **True-switch confirmation:** after a flight/contest/reception candidate,
   require agreement between post-contact ball deceleration or direction change,
   team-level receiver arrival, and persistence of the new controlled state.
   A direction change or nearest-player change alone remains insufficient.
2. **Restart activation:** keep `D` until a restart-compatible ball movement
   and the first stable team-level control evidence agree. Player repositioning
   and Smart Tagging context open a restart hypothesis; they do not mark the
   exact in-play frame by themselves.

The first ablation has now been run. It shows that these signals should not
directly rewrite frame labels: they are promising observations for a guarded
transition-trust layer, but the resolver must preserve explicit `NO_SWITCH` and
`CONTESTED` alternatives. No threshold should be promoted merely because it
agrees with a public label on one ambiguous window.
