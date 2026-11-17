# `possession_lab` implementation guide

This package contains the reusable functions called by the notebooks and the
canonical runner. It keeps measurements, football scenarios, transition
hypotheses and final labels separate so that an observation cannot silently
become a possession decision.

## Canonical execution

```bash
python -m possession_lab.reference_pipeline
```

The runner starts from the supplied tactical tracking and Smart Tags, builds
one engineered evidence table, applies the scenario/anchor rules and writes
the bounded three-family final submission to
`evaluation/results/final_submission.csv`.

## Main modules

- [baseline_pipeline.py](./baseline_pipeline.py) — load data, frame timing,
  team/slot metadata and submission writing.
- [calculations/features.py](./calculations/features.py) — compose the shared
  temporal, reachability, missing-ball, path, reception, anchor and transition
  features.
- [temporal_rules.py](./temporal_rules.py) — physical measurements, pressure,
  arrival, anchors, receptions, restarts and football-context scenarios.
- [scenarios/](./scenarios/) — reusable scenario and boundary rules.
- [decoder_experiments.py](./decoder_experiments.py) — evidence aggregation,
  transition hypotheses, bounded interval resolvers and audit decoders.
- [scenario_decoder.py](./scenario_decoder.py) — scenario-level decoding
  helpers.
- [reference_pipeline.py](./reference_pipeline.py) — the small organiser-facing
  composition used for the final CSV.

The promoted resolver handles three families:

1. live `H ↔ A` contact and control handoffs;
2. `H/A → D` when stoppage evidence and forward settling agree;
3. `D → H/A` only after restart activation or persistent control evidence.

Its explicit hypotheses are `KEEP_CURRENT`, `SWITCH_TO_H`, `SWITCH_TO_A`,
`ENTER_D`, `RESTART_TO_H`, `RESTART_TO_A` and `CONTESTED`. `CONTESTED` is
preserved when the evidence is not strong enough to justify a label change.

## Appendix implementations

The following modules are retained for reproducibility and comparison but are
not imported by the canonical runner:

- [george_ballfill_stage.py](./george_ballfill_stage.py) and
  [george_ballfill_adapter.py](./george_ballfill_adapter.py) — source-aware
  alternate-camera/path observations with uncertainty and provenance.
- [haiqal_baseline.py](./haiqal_baseline.py) and
  [pipeline.py](./pipeline.py) — independent distance/speed/dead-ball
  observations and bounded adapter experiments.
- [sequence_models.py](./sequence_models.py), [calibration.py](./calibration.py)
  and [supervised_training/](./supervised_training/) — optional sequence,
  calibration and supervised comparisons.
- [run_interpolated_flight_experiment.py](./run_interpolated_flight_experiment.py)
  — full research/ablation runner used to produce the comparison evidence.

These alternatives remain separate because they either duplicate existing
signals, require external training labels, or change too many full-match
frames without a public improvement. Their code is not deleted: the appendix
notebooks and experiment outputs remain auditable, while the production path
stays small and label-free.
