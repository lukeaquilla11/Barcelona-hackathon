"""Unsupervised check and prior/transition extraction from Metrica's automatic Smart Tagging.
Agreement = share of frames inside team-tagged attacking phases where we name the same team."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging as st

ATTACK = {"build up", "progression", "attacking transition", "creating chances", "long ball",
          "ball in final third", "players in the box"}
DEFEND = {"defending in defensive third", "defending in middle third", "defending in attacking third"}
DEADBALL_CODES = {"set pieces", "corner", "throw in", "goal", "foul", "card", "free kick", "goal kick", "penalty"}


def tag_prior(n):
    """Per-frame expected possessor from tags: +1 home, -1 away, 0 unknown."""
    v = np.zeros(n)
    for ev in st.load():
        code = ev["code"].lower(); team = str(ev["tags"].get("Team", "")).lower()
        sgn = 1 if team.startswith("barc") else -1 if team.startswith("atl") else 0
        if not sgn: continue
        if code in DEFEND: sgn = -sgn
        elif code not in ATTACK: continue
        v[ev["start_frame"]:ev["end_frame"]] += sgn
    return np.sign(v)


def deadball_tag_prior(n, expand_frames=300):
    """Per-frame dead-ball prior from Smart Tagging set pieces/stoppages, expanded forward
    to capture full stoppage durations (~12-14s at 25fps)."""
    d = np.zeros(n, dtype=np.float32)
    for ev in st.load():
        code = ev["code"].lower()
        if any(k in code for k in DEADBALL_CODES):
            s = max(0, int(ev["start_frame"]))
            e = min(n, int(ev["end_frame"]) + expand_frames)
            d[s:e] = 1.0
    return d


def empirical_priors_and_transitions(n=199319):
    """Compute empirical state priors and log-transition matrix across H (0), A (1), D (2)
    from Smart Tagging intervals."""
    tp = tag_prior(n)
    dp = deadball_tag_prior(n, expand_frames=250)
    states = np.where(dp > 0, 2, np.where(tp > 0, 0, np.where(tp < 0, 1, 2)))
    counts = np.bincount(states, minlength=3).astype(float)
    priors = np.maximum(counts / counts.sum(), 1e-4)

    trans = np.ones((3, 3), dtype=float)  # Laplace smoothing
    for t in range(len(states) - 1):
        trans[states[t], states[t + 1]] += 1.0
    trans /= trans.sum(axis=1, keepdims=True)
    return priors, trans


def agreement(labels, prior):
    m = (prior != 0) & np.isin(labels, ["H", "A"])
    if not m.any():
        return float("nan"), 0
    pred = np.where(labels[m] == "H", 1, -1)
    return (pred == prior[m]).mean(), m.sum()


if __name__ == "__main__":
    prior = tag_prior(199319)
    for f in sys.argv[1:]:
        lab = pd.read_csv(f).label.values.astype(str)
        a, k = agreement(lab, prior)
        print(f"{os.path.basename(f):28s} tag agreement {a:.3f} over {k} frames")