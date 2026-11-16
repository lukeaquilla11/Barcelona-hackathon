"""Tune the existing Haiqal-style rule/Viterbi baseline on two Metrica + seven IDSSE matches.

This ports Working File Mini's parameter-sweep logic, but targets
`possession_lab.haiqal_baseline.PARAMS`. It never edits that file automatically.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warnings
from .param_fit import sweep
from .supervised_training.degrade import degrade
from .supervised_training.loaders import load_idsse, load_match_cache, load_metrica_sample, save_match
from .param_fit.sweep import Game

IDSSE_IDS = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]


def _load_target_params():
    """Read the current baseline PARAMS without importing its gated match-loader dependencies."""
    source = Path(__file__).with_name("haiqal_baseline.py").read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "PARAMS":
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PARAMS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("Could not read PARAMS from possession_lab/haiqal_baseline.py")


def _rolling_mean(values, half):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    safe = np.where(valid, values, 0.0)
    cumulative = np.r_[0.0, np.cumsum(safe)]
    counts = np.r_[0, np.cumsum(valid.astype(np.int64))]
    index = np.arange(len(values))
    low, high = np.maximum(0, index - half), np.minimum(len(values), index + half + 1)
    count = counts[high] - counts[low]
    return np.where(count > 0, (cumulative[high] - cumulative[low]) / np.maximum(count, 1), np.nan)


class _HaiqalParameterTarget:
    """Pure scoring surface matching haiqal_baseline.frame_scores; no gated data access."""
    PARAMS = _load_target_params()

    @staticmethod
    def frame_scores(F, params=None):
        p = _HaiqalParameterTarget.PARAMS if params is None else params
        d_home, d_away = F["dH"], F["dA"]
        visible = F["vis"] & np.isfinite(d_home) & np.isfinite(d_away)
        difference = np.clip((d_away - d_home) / p["dist_scale"], -3, 3)
        near = np.exp(-np.fmin(d_home, d_away) / p["near"])
        speed = np.nan_to_num(F["bspeed"], nan=0.0)
        control = np.where(speed > p["fast_ball"], p["fast_weight"], 1.0)
        closeness = np.where(visible, difference * near * control, 0.0)
        player_speed = _rolling_mean(F["pspeed"], 25)
        speed = np.where(np.isfinite(player_speed), player_speed, p["speed_dead"])
        dead = np.clip((p["speed_dead"] - speed) / p["speed_scale"], -3, 3) * p["speed_weight"]
        dead += np.where(F["vis"], 0.0, p["missing_dead"])
        dead += np.where(F["out"], p["out_weight"], 0.0)
        scores = np.zeros((len(closeness), 3), dtype=float)
        scores[:, 0] += closeness / 2
        scores[:, 1] -= closeness / 2
        scores[:, 2] += dead
        return scores


def _get_idsse(match_id, cache_dir, limit):
    suffix = f"_lim{limit}" if limit else ""
    path = Path(cache_dir) / f"idsse_{match_id}{suffix}.npz"
    if path.exists():
        return load_match_cache(str(path))
    match = load_idsse(match_id, limit=limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_match(match, str(path))
    return match


def _haiqal_features(match):
    """Use the same raw clues as possession_lab.haiqal_baseline, not its real-match loader."""
    P, B, team, fps = match.P, match.B, match.team, match.fps
    bx, by = B[:, 0].astype(np.float64), B[:, 1].astype(np.float64)
    visible = np.isfinite(bx)
    distances = np.hypot(P[:, :, 0] - bx[:, None], P[:, :, 1] - by[:, None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d_home = np.nanmin(np.where(team[None, :] == 0, distances, np.nan), axis=1)
        d_away = np.nanmin(np.where(team[None, :] == 1, distances, np.nan), axis=1)
        velocity = np.full(P.shape[:2], np.nan, np.float32)
        velocity[1:] = np.sqrt(((P[1:] - P[:-1]) ** 2).sum(-1)) * fps
        velocity[velocity > 12] = np.nan
        player_speed = np.nanmedian(velocity, axis=1)
    ball_speed = np.full(len(bx), np.nan)
    ball_speed[1:] = np.hypot(np.diff(bx), np.diff(by)) * fps
    outside = (bx < -0.5) | (bx > match.L + 0.5) | (by < -0.5) | (by > match.W + 0.5)
    return dict(team=team, vis=visible, bx=bx, by=by, dH=d_home, dA=d_away,
                pspeed=player_speed, nplayers=np.isfinite(P[:, :, 0]).sum(1),
                bspeed=ball_speed, out=outside)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracking1", required=True); ap.add_argument("--tracking2", required=True)
    ap.add_argument("--events1", required=True); ap.add_argument("--events2", required=True)
    ap.add_argument("--match-ids", nargs="+", default=IDSSE_IDS)
    ap.add_argument("--idsse-cache-dir", default="possession_lab/cache/training_idsse")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--objective", choices=["both", "spell", "macro"], default="both")
    ap.add_argument("--out", default="possession_lab/reports/haiqal_param_fit.json")
    args = ap.parse_args(argv)

    for path in (args.tracking1, args.tracking2, args.events1, args.events2):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    ids = list(dict.fromkeys(args.match_ids))
    if len(ids) != 7:
        raise ValueError(f"Expected seven IDSSE match IDs; received {len(ids)}")

    matches = {
        "sample1": load_metrica_sample(args.tracking1, args.events1, name="sample1",
                                       home_slots="1-14", away_slots="15-28"),
        "sample2": load_metrica_sample(args.tracking2, args.events2, name="sample2",
                                       home_slots="1-14", away_slots="15-26"),
    }
    for match_id in ids:
        matches[match_id] = _get_idsse(match_id, args.idsse_cache_dir, args.limit)

    games = {}
    for name, match in matches.items():
        seeds = [0] if args.clean else args.seeds
        feature_sets = [_haiqal_features(match if args.clean else degrade(match, seed=seed)) for seed in seeds]
        games[name] = Game(name, feature_sets, match.labels,
                           [(a, b) for _, a, b in match.blocks()])
        print(name, "frames", len(match.labels), "labels H/A/D/unknown",
              [int((match.labels == c).sum()) for c in (0, 1, 2, -1)], flush=True)

    # One held-out fold per match. The rule-based model and its Viterbi decoder are tuned;
    # Smart Tagging weight is held at zero because these source datasets do not contain it.
    folds = {name: [(name, half) for half in range(len(game.blocks))]
             for name, game in games.items()}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    sweep.run(
        _HaiqalParameterTarget, games, list(games), [],
        SimpleNamespace(quick=args.quick, objective=args.objective, no_real=True, apply=False),
        repo=None, out_json=str(out), folds=folds,
    )
    print("parameter report ->", out)


if __name__ == "__main__":
    main()
