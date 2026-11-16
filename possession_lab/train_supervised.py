"""Train the optional supervised H/A/D model on 2 Metrica samples + 7 IDSSE matches.

Run from the consolidated project root:
  python -m possession_lab.train_supervised \\
    --tracking1 data/Sample_Game_1_Combined_TrackingData.csv \\
    --tracking2 data/Sample_Game_2_Combined_TrackingData.csv \\
    --events1 data/Sample_Game_1_RawEventsData.csv \\
    --events2 data/Sample_Game_2_RawEventsData.csv \\
    --idsse-cache-dir /path/to/Working-File-Mini/param_fit/cache_idsse

If an IDSSE cache is absent, it is downloaded via kloppy and cached locally. The two
Metrica tracking CSVs are not included in either supplied ZIP; provide their paths.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .supervised_training import model as M
from .supervised_training.degrade import degrade
from .supervised_training.loaders import (
    load_idsse, load_match_cache, load_metrica_sample, save_match,
)
from .supervised_training.labels_metrica import spells_summary

IDSSE_IDS = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
EVAL_SEED = 1000


def _idsse_match(match_id: str, cache_dir: Path, limit: int | None):
    suffix = f"_lim{limit}" if limit else ""
    path = cache_dir / f"idsse_{match_id}{suffix}.npz"
    if path.exists():
        return load_match_cache(str(path))
    match = load_idsse(match_id, limit=limit)
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_match(match, str(path))
    return match


def _load_training_matches(args):
    matches = {}
    # The slot assignments were explicitly supplied for these two sample games.
    matches["sample1"] = load_metrica_sample(
        args.tracking1, args.events1, name="sample1",
        home_slots="1-14", away_slots="15-28",
    )
    matches["sample2"] = load_metrica_sample(
        args.tracking2, args.events2, name="sample2",
        home_slots="1-14", away_slots="15-26",
    )
    for match_id in args.match_ids:
        match = _idsse_match(match_id, Path(args.idsse_cache_dir), args.limit)
        matches[match_id] = match
    return matches


def _make_training_xy(matches, args, degraded: bool):
    """Build each match's training matrix once; CV folds never share a held-out match."""
    xy = {}
    for name, match in matches.items():
        X, y = M.make_xy([match], seeds=args.seeds, stride=args.stride, degrade=degraded)
        xy[name] = (X, y)
        labels = match.labels
        print(
            f"{name}: frames={len(labels):,}; players={match.P.shape[1]}; "
            f"label_counts(H/A/D/unknown)={[int((labels == c).sum()) for c in (0, 1, 2, -1)]}",
            flush=True,
        )
    return xy


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracking1", required=True)
    ap.add_argument("--tracking2", required=True)
    ap.add_argument("--events1", required=True)
    ap.add_argument("--events2", required=True)
    ap.add_argument("--match-ids", nargs="+", default=IDSSE_IDS)
    ap.add_argument("--idsse-cache-dir", default="possession_lab/cache/training_idsse")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="quick test: first N frames per IDSSE match")
    ap.add_argument("--clean", action="store_true", help="disable tracking degradation")
    ap.add_argument("--no-cv", action="store_true", help="skip held-out folds and fit all matches")
    ap.add_argument("--out", default="possession_lab/models/supervised_possession.joblib")
    args = ap.parse_args(argv)

    ids = list(dict.fromkeys(args.match_ids))
    if len(ids) != 7:
        raise ValueError(f"Expected the seven configured IDSSE matches; received {len(ids)}: {ids}")
    missing = [p for p in (args.tracking1, args.tracking2, args.events1, args.events2) if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Missing Metrica input file(s): " + ", ".join(missing))

    matches = _load_training_matches(args)
    degraded = not args.clean
    xy = _make_training_xy(matches, args, degraded)

    def fit_on(names):
        X = pd.concat([xy[name][0] for name in names], ignore_index=True)
        y = np.concatenate([xy[name][1] for name in names])
        return M.fit_clf(X, y)

    report, decoder = {}, dict(M.DECODER_DEFAULT)
    if not args.no_cv:
        oof = []
        names = list(matches)
        for held_out in names:
            train_names = [name for name in names if name != held_out]
            clf = fit_on(train_names)
            match = matches[held_out]
            evaluation_match = degrade(match, EVAL_SEED) if degraded else match
            proba = M.predict_proba(clf, evaluation_match)
            oof.append((proba, evaluation_match))
            print(f"  held-out fold complete: {held_out}", flush=True)
        decoder, score = M.tune_decoder(oof)
        print("OOF-tuned decoder:", decoder, "score:", round(score, 4))
        for (proba, match), name in zip(oof, names):
            report[name] = {
                "raw": M.metrics.score_blocks(match.labels, proba.argmax(1), match.blocks()),
                "decoded": M.metrics.score_blocks(
                    match.labels, M.decode(proba, match.blocks(), decoder, match.fps), match.blocks()
                ),
            }

    final_model = fit_on(list(matches))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "training_matches": list(matches), "idsse_match_ids": ids,
        "degraded": degraded, "seeds": args.seeds, "stride": args.stride,
        "cross_validation": not args.no_cv, "report": report,
        "sample_slot_assignments": {"sample1": {"home": "1-14", "away": "15-28"},
                                     "sample2": {"home": "1-14", "away": "15-26"}},
    }
    M.save(str(out), final_model, decoder, info)
    report_path = out.with_name(out.stem + "_report.json")
    report_path.write_text(json.dumps({"decoder": decoder, "info": info}, indent=2, default=float))
    print("\nTraining complete")
    print("model:", out)
    print("report:", report_path)
    if report:
        for name, values in report.items():
            print(name, "raw:", values["raw"], "decoded:", values["decoded"])


if __name__ == "__main__":
    main()
