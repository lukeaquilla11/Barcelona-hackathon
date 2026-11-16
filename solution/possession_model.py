"""Ball possession from automatic tracking: HMM integrating homography confidence,
Smart Tagging priors/transitions, sensor emission likelihoods, and Viterbi decoding.

    python3 solution/possession_model.py <team_name>.csv

Every frame is labelled H (home in possession), A (away in possession) or D (dead ball);
frames outside the two halves are X.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "starter"))
from starter.games import half_video_blocks, GAME
from features import features, rolling_mean

H, A, D = 0, 1, 2

PARAMS = dict(
    # --- sensor emissions: player proximity & ball kinematics ---
    dist_scale=2.0,         # metres of distance advantage worth one emission point
    near=4.0,               # metres: spatial proximity decay scale
    fast_ball=6.0,          # m/s: ball in flight (pass/shot) reduces single-player control certainty
    fast_weight=0.35,       # proximity weight when ball speed > fast_ball
    # --- smart tagging priors ---
    tag_weight=0.18,        # per-frame emission boost from Smart Tagging team phases
    dead_tag_weight=0.65,   # per-frame emission boost from expanded set-piece/stoppage tags
    dead_expand_frames=275, # frames (~11s) to expand set-piece tags forward
    use_empirical_trans=True,
    # --- dead-ball sensor emissions ---
    speed_dead=1.35,        # m/s: median player speed threshold during stoppages
    speed_scale=0.35,
    speed_weight=0.65,
    missing_dead=0.25,      # dead-ball emission when ball is unseen beyond 1s interpolation
    stationary_dead=0.30,   # dead-ball emission when ball displacement over 1s < 0.4m
    out_weight=2.0,         # dead-ball emission when ball is outside pitch boundaries
    # --- homography confidence weighting ---
    homo_dead_boost=0.80,   # dead-ball emission boost when camera homography is invalid (valid == 0)
    min_homo_conf=0.15,     # floor on spatial sensor trust when reprojection error is high
    # --- Viterbi transition costs & post-processing ---
    switch_cost=30.0,       # base log-cost of changing possession H <-> A
    dead_cost=15.0,          # base log-cost of entering or leaving a dead ball D
    min_spell_frames=125,    # minimum valid active spell length (0.76s at 25fps)
)


# ---------------------------------------------------------------- sensor emissions & priors

def closeness_emission(F, p):
    """Per-frame log-likelihood evidence for Home (+s) vs Away (-s), weighted by
    homography confidence and augmented with Smart Tagging team priors."""
    dH, dA = F["dH"], F["dA"]
    vis = F["vis"] & np.isfinite(dH) & np.isfinite(dA)
    diff = np.clip((dA - dH) / p["dist_scale"], -3.0, 3.0)
    near = np.exp(-np.fmin(dH, dA) / p["near"])
    bs = np.nan_to_num(F["bspeed"], nan=0.0)
    control = np.where(bs > p["fast_ball"], p["fast_weight"], 1.0)

    # Discount spatial tracking emissions when homography reprojection error is high
    homo_trust = np.clip(F["homo_conf"], p["min_homo_conf"], 1.0)
    s = np.where(vis, diff * near * control * homo_trust, 0.0)

    if p["tag_weight"]:
        try:
            from tag_check import tag_prior
            s = s + p["tag_weight"] * tag_prior(len(s))
        except Exception:
            pass
    return s


def dead_ball_emission(F, p):
    """Per-frame log-likelihood evidence for Deadball (D) combining player speed,
    ball visibility/stationarity, pitch bounds, homography validity, and stoppage tags."""
    ps = rolling_mean(F["pspeed"], 25)
    ps_f = np.where(np.isfinite(ps), ps, p["speed_dead"])
    d = np.clip((p["speed_dead"] - ps_f) / p["speed_scale"], -3.0, 3.0) * p["speed_weight"]
    d += np.where(F["vis"], 0.0, p["missing_dead"])
    d += np.where(F["out"], p["out_weight"], 0.0)

    # Stationary ball over 1-second window indicates stoppage / set-piece setup
    disp = F["disp_1s"]
    stationary = F["vis"] & np.isfinite(disp) & (disp < 0.4)
    d += np.where(stationary, p["stationary_dead"], 0.0)

    # Invalid homography frames strongly correlate with replays/cutaways during stoppages
    d += np.where(F["valid_homo"] == 0, p["homo_dead_boost"], 0.0)

    if p["dead_tag_weight"]:
        try:
            from tag_check import deadball_tag_prior
            d += p["dead_tag_weight"] * deadball_tag_prior(len(d), expand_frames=p["dead_expand_frames"])
        except Exception:
            pass
    return d


def frame_emissions(F, p):
    """Build (n, 3) emission log-likelihood matrix for [H, A, D] including empirical priors."""
    n = len(F["vis"])
    s = closeness_emission(F, p)
    e = np.zeros((n, 3), dtype=np.float64)
    e[:, H] += s / 2.0
    e[:, A] -= s / 2.0
    e[:, D] += dead_ball_emission(F, p)

    if p["use_empirical_trans"]:
        try:
            from tag_check import empirical_priors_and_transitions
            priors, _ = empirical_priors_and_transitions(n)
            e += 0.15 * np.log(priors)[None, :]
        except Exception:
            pass
    return e


# ---------------------------------------------------------------- decoding & smoothing

def build_transition_matrix(n, p):
    """Combine base switching penalties with empirical log-transition probabilities."""
    T_base = np.array([
        [0.0, -p["switch_cost"], -p["dead_cost"]],
        [-p["switch_cost"], 0.0, -p["dead_cost"]],
        [-p["dead_cost"], -p["dead_cost"], 0.0]
    ], dtype=np.float64)

    if p["use_empirical_trans"]:
        try:
            from tag_check import empirical_priors_and_transitions
            _, emp_trans = empirical_priors_and_transitions(n)
            log_emp = np.log(np.clip(emp_trans, 1e-4, 1.0))
            log_emp -= np.diag(log_emp)[:, None]
            T_base += 0.25 * log_emp
        except Exception:
            pass
    return T_base


def viterbi(e, T):
    """Globally optimal state sequence given per-frame emissions e (n x 3) and transition matrix T."""
    n = len(e)
    back = np.zeros((n, 3), np.int8)
    score = e[0] + T[D]  # each half starts from a dead ball (kickoff)
    for t in range(1, n):
        cand = score[:, None] + T
        back[t] = cand.argmax(0)
        score = cand.max(0) + e[t]
    path = np.empty(n, np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def smooth_micro_spells(labels, min_frames=19):
    """Merge isolated active possession spells shorter than min_frames into adjacent states."""
    if min_frames <= 1:
        return labels
    out = labels.copy()
    cut = np.r_[0, np.where(out[1:] != out[:-1])[0] + 1, len(out)]
    for a, b in zip(cut[:-1], cut[1:]):
        if (b - a) < min_frames and out[a] in ("H", "A"):
            left = out[a - 1] if a > 0 else "X"
            right = out[b] if b < len(out) else "X"
            repl = left if left in ("H", "A", "D") else right
            if repl in ("H", "A", "D"):
                out[a:b] = repl
    return out


def predict(F=None, **overrides):
    """Predict frame-level labels [H, A, D, X] for the full video."""
    p = dict(PARAMS, **overrides)
    F = F or features()
    e = frame_emissions(F, p)
    T = build_transition_matrix(len(e), p)
    labels = np.full(len(e), "X", dtype="<U1")
    for _, a, b in half_video_blocks(GAME):
        path = viterbi(e[a:b + 1], T)
        seg_labels = np.array(["H", "A", "D"])[path]
        labels[a:b + 1] = smooth_micro_spells(seg_labels, p["min_spell_frames"])
    return labels


def write(path, labels):
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{l}\n" for i, l in enumerate(labels))


def counts(labels):
    g = labels[labels != "X"]
    seg = g[np.r_[True, g[1:] != g[:-1]]]
    return dict(possessions=int(((seg == "H") | (seg == "A")).sum()), stoppages=int((seg == "D").sum()))


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "evaluation", "results", "possession.csv")
    labels = predict()
    print(counts(labels), {k: int((labels == k).sum()) for k in "HADX"})
    write(out, labels)
    print("->", out)