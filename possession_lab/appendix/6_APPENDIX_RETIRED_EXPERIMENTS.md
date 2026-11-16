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

The active system keeps the useful parts of these ideas as evidence features, but does not let them override stronger anchors or protected `D`/`X` labels.
