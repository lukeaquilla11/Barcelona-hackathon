"""Frame classifier (gradient boosting) + Viterbi decoder + decoder tuning + save/load."""
import itertools
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from .schema import Match, H, A, D, CLASSES
from .features import build_features
from .degrade import degrade as degrade_match
from . import metrics

DECODER_DEFAULT = dict(switch_cost=20.0, dead_cost=20.0, min_poss_s=1.0, min_dead_s=1.0)
DECODER_GRID = dict(switch_cost=[5, 10, 20, 40, 80], dead_cost=[5, 10, 20, 40],
                    min_poss_s=[0, 1, 2], min_dead_s=[0, 1, 2])
CLF_DEFAULT = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=100,
                   l2_regularization=1.0, early_stopping=False, random_state=0)


# ------------------------------------------------------------------------------------------- training data
def make_xy(matches, seeds=(0,), stride=3, degrade=True, mirror=True):
    """Stack features of every match x degradation seed (x home/away mirror). Only labelled frames, every `stride`-th."""
    Xs, ys = [], []
    for m in matches:
        for seed in seeds:
            dm = degrade_match(m, seed) if degrade else m
            for mm in ([dm, dm.swapped()] if mirror else [dm]):
                X, rows = build_features(mm)
                y = mm.labels[rows]
                keep = (y >= 0) & (np.arange(len(y)) % stride == 0)
                Xs.append(X[keep]); ys.append(y[keep])
    import pandas as pd
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def fit_clf(X, y, **kw):
    clf = HistGradientBoostingClassifier(**dict(CLF_DEFAULT, **kw), class_weight="balanced")
    clf.fit(X, y)
    clf.feature_names_ = list(X.columns)
    return clf


def fit(matches, seeds=(0,), stride=3, degrade=True, **kw):
    X, y = make_xy(matches, seeds, stride, degrade)
    return fit_clf(X, y, **kw)


# ------------------------------------------------------------------------------------------- inference
def _proba_rows(clf, X):
    P = np.full((len(X), 3), 1 / 3, np.float32)
    P[:, clf.classes_.astype(int)] = clf.predict_proba(X[clf.feature_names_])
    return P


def predict_proba(clf, m: Match):
    """(n,3) probabilities for H,A,D; rows outside any half = 1/3. Symmetrised over the home/away swap."""
    out = np.full((len(m.period), 3), 1 / 3, np.float32)
    X, rows = build_features(m)
    p = _proba_rows(clf, X)
    Xs, _ = build_features(m.swapped())
    ps = _proba_rows(clf, Xs)[:, [1, 0, 2]]
    out[rows] = (p + ps) / 2
    return out


def viterbi(logp, switch, dead):
    n = len(logp)
    T = -np.array([[0, switch, dead], [switch, 0, dead], [dead, dead, 0]], np.float64)
    score = logp[0].astype(np.float64)
    back = np.zeros((n, 3), np.int8)
    for t in range(1, n):
        cand = score[:, None] + T
        back[t] = cand.argmax(0)
        score = cand.max(0) + logp[t]
    path = np.empty(n, np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def merge_short(path, min_poss, min_dead):
    """Absorb spells shorter than the minimum into the longer neighbour (shortest first)."""
    path = path.copy()
    while True:
        cut = np.r_[0, np.flatnonzero(path[1:] != path[:-1]) + 1, len(path)]
        runs = list(zip(cut[:-1], cut[1:]))
        worst, wi = None, None
        for i, (a, b) in enumerate(runs):
            need = min_dead if path[a] == D else min_poss
            if (b - a) < need and len(runs) > 1 and (worst is None or (b - a) < worst):
                worst, wi = b - a, i
        if wi is None:
            return path
        a, b = runs[wi]
        left = runs[wi - 1][1] - runs[wi - 1][0] if wi > 0 else -1
        right = runs[wi + 1][1] - runs[wi + 1][0] if wi + 1 < len(runs) else -1
        path[a:b] = path[runs[wi - 1][0]] if left >= right else path[runs[wi + 1][0]]


def decode(proba, blocks, params=None, fps=25):
    q = dict(DECODER_DEFAULT, **(params or {}))
    out = np.full(len(proba), -1, np.int8)
    lp = np.log(np.clip(proba, 1e-4, 1))
    for _, a, b in blocks:
        path = viterbi(lp[a:b + 1], q["switch_cost"], q["dead_cost"])
        if q["min_poss_s"] or q["min_dead_s"]:
            path = merge_short(path, int(q["min_poss_s"] * fps), int(q["min_dead_s"] * fps))
        out[a:b + 1] = path
    return out


def tune_decoder(items, grid=None, objective="both"):
    """items: [(proba, Match)] with labels (preferably OUT-OF-FOLD probabilities). Grid search; returns (params, score)."""
    grid = grid or DECODER_GRID
    best, best_p = -1, dict(DECODER_DEFAULT)
    for sw, dd, mp, md in itertools.product(grid["switch_cost"], grid["dead_cost"], grid["min_poss_s"], grid["min_dead_s"]):
        q = dict(switch_cost=sw, dead_cost=dd, min_poss_s=mp, min_dead_s=md)
        sc = []
        for proba, m in items:
            r = metrics.score_blocks(m.labels, decode(proba, m.blocks(), q, m.fps), m.blocks())
            sc.append((r["macroF1"] + r["spellF1"]) / 2 if objective == "both" else r["macroF1"])
        k = float(np.mean(sc))
        if k > best + 1e-9:
            best, best_p = k, q
    return best_p, best


def evaluate(clf, m: Match, decoder=None):
    proba = predict_proba(clf, m)
    raw = np.where(np.arange(len(proba)) >= 0, proba.argmax(1), -1).astype(np.int8)
    dec = decode(proba, m.blocks(), decoder, m.fps)
    return dict(raw=metrics.score_blocks(m.labels, raw, m.blocks()),
                decoded=metrics.score_blocks(m.labels, dec, m.blocks()))


# ------------------------------------------------------------------------------------------- persistence
def save(path, clf, decoder, info=None):
    joblib.dump(dict(clf=clf, decoder=decoder, info=info or {}), path)


def load(path):
    d = joblib.load(path)
    return d["clf"], d["decoder"], d.get("info", {})


def labels_to_str(codes, outside="X"):
    s = np.full(len(codes), outside, dtype="<U1")
    ok = codes >= 0
    s[ok] = CLASSES[codes[ok]]
    return s
