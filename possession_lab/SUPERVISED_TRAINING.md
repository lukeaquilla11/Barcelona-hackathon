# Supervised Training Guide

This guide explains how to train the optional supervised possession model and, separately, how to fit the existing Haiqal baseline parameters. These additions do not replace the project's existing possession pipeline.

## 1. Check your data folders

Your project should contain these four Metrica CSVs:

```text
data/metrica/
├── Sample_Game_1_Combined_TrackingData.csv
├── Sample_Game_1_RawEventsData.csv
├── Sample_Game_2_Combined_TrackingData.csv
└── Sample_Game_2_RawEventsData.csv
```

The trainer also uses seven IDSSE match caches from `Working File Mini/param_fit/cache_idsse/`:

```text
idsse_J03WMX.npz
idsse_J03WN1.npz
idsse_J03WOH.npz
idsse_J03WOY.npz
idsse_J03WPY.npz
idsse_J03WQQ.npz
idsse_J03WR9.npz
```

You can leave these cache files in `Working File Mini`; you do not need to copy them into this project.

**Metrica player-slot mapping**

- Game 1: Home = slots 1–14; Away = slots 15–28.
- Game 2: Home = slots 1–14; Away = slots 15–26.

## 2. Install the required packages

Open a terminal in the project root (the folder containing `possession_lab`) and run:

```bat
python -m pip install numpy pandas scikit-learn joblib kloppy
```

## 3. Train the supervised model

Still from the project root, run the command below. Replace the IDSSE cache path with the actual location of your `Working File Mini` folder.

```bat
python -m possession_lab.train_supervised ^
  --tracking1 data/metrica/Sample_Game_1_Combined_TrackingData.csv ^
  --events1 data/metrica/Sample_Game_1_RawEventsData.csv ^
  --tracking2 data/metrica/Sample_Game_2_Combined_TrackingData.csv ^
  --events2 data/metrica/Sample_Game_2_RawEventsData.csv ^
  --idsse-cache-dir "C:\PATH\TO\Working File Mini\param_fit\cache_idsse"
```

The training process uses all nine matches. It evaluates by holding out one match at a time, tunes the Viterbi decoder using out-of-fold predictions, and then fits the final classifier on all nine matches. Tracking degradation is enabled by default to make training more robust to imperfect tracking.

**Useful options**

- Add `--clean` to run without tracking degradation.
- Add `--limit 5000` for a small IDSSE smoke test only; do not use this limit for final training.

## 4. Check the training outputs

By default, training saves:

```text
possession_lab/models/supervised_possession.joblib
```

It also writes a matching JSON training report. Check the terminal output and report for errors and evaluation results.

## 5. (Optional) Run supervised inference

The supervised model is not connected to the default possession resolver. You can call it separately to obtain frame-aligned probability evidence:

```python
from possession_lab.load_data import load_match
from possession_lab.supervised_bridge import predict_supervised_evidence

match = load_match()
probability_evidence = predict_supervised_evidence(
    "possession_lab/models/supervised_possession.joblib", match
)
```

Treat this as an independent signal for comparison before considering any integration into the main resolver. The model uses past/future frame windows and Viterbi decoding, so it is not a zero-latency live predictor.

The canonical organiser-facing runner does not accept this model. That is
intentional: the fitted classifier and decoder depend on event-derived labels,
so they are retained as a separately documented supervised comparison rather
than hidden inputs to the label-free resolver. Use `supervised_bridge.py` only
for an explicit offline comparison or audit, and report its training matches
and label provenance alongside any score.

---

## Optional: Fit the existing Haiqal baseline parameters

This is a separate workflow from supervised classifier training. It searches for improved values for the 11 numeric parameters in `possession_lab/haiqal_baseline.py`. It produces a report; it does **not** edit the baseline code automatically.

Run from the project root, using the same Metrica files and IDSSE cache directory:

```bat
python -m possession_lab.fit_haiqal_params ^
  --tracking1 data/metrica/Sample_Game_1_Combined_TrackingData.csv ^
  --events1 data/metrica/Sample_Game_1_RawEventsData.csv ^
  --tracking2 data/metrica/Sample_Game_2_Combined_TrackingData.csv ^
  --events2 data/metrica/Sample_Game_2_RawEventsData.csv ^
  --idsse-cache-dir "C:\PATH\TO\Working File Mini\param_fit\cache_idsse"
```

The report is saved to `possession_lab/reports/haiqal_param_fit.json`. Review it before manually transferring any recommended parameter changes into `haiqal_baseline.py`.

**Optional flags**

- `--quick` uses a smaller search grid.
- `--clean` disables tracking degradation.

The full parameter search can take a while. Start with `--quick` if you want to check that the setup runs before attempting the full search.

## Important notes

- Run commands from the project root, not from inside `possession_lab`.
- Keep the IDSSE cache files local. They are derived from licensed match data and should not be committed or shared.
- These training workflows are optional and do not alter the existing default possession pipeline.
- Full end-to-end training should be treated as unverified until it has completed successfully on your local CSVs and IDSSE caches.
