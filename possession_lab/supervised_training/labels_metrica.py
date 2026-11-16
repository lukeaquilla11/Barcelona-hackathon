"""Silver per-frame possession labels (H / A / D) from a Metrica *RawEventsData.csv.

Rules (events are 1-based frames, the same counter as tracking frame_id):
  * a PASS / SHOT / RECOVERY / SET PIECE by a team gives that team possession from its Start Frame;
  * BALL OUT -> dead from its End Frame (ball crosses the line); the next team event (normally a SET PIECE or a
    goal-kick PASS) ends the dead ball at its Start Frame;
  * SHOT whose subtype contains GOAL / OUT / OFF TARGET -> dead from its End Frame;
  * FAULT RECEIVED -> dead from its Start Frame; BALL LOST with OFFSIDE / HAND BALL / REFEREE / END HALF -> dead from
    its End Frame;
  * every period starts dead (until the kick-off) and the last state runs to the last tracked frame.
Plain BALL LOST / CHALLENGE / CARD events do not change the state: the ball stays with the team until the other team's
RECOVERY / PASS arrives.
These are *silver* labels (derived from events, not hand-labelled per frame), good to within a few frames.
"""
import numpy as np
import pandas as pd
from .schema import H, A, D

DEAD_SHOT = ("GOAL", "OUT", "OFF TARGET")
DEAD_LOST = ("OFFSIDE", "HAND BALL", "REFEREE", "END HALF")


def build_labels(events: pd.DataFrame, period_ranges: dict, n_total: int) -> np.ndarray:
    """period_ranges: {period: (first_frame_id, last_frame_id)}. Returns int8 array indexed by frame_id (-1 = unlabeled)."""
    lab = np.full(n_total, -1, np.int8)
    ev = events.copy()
    ev["Subtype"] = ev["Subtype"].fillna("").str.upper()
    for per, (f0, f1) in period_ranges.items():
        rows = ev[ev["Period"] == per].sort_values("Start Frame", kind="stable").to_dict("records")
        changes = [(int(f0), D)]
        dead = True

        def push(f, c):
            f = max(int(f), changes[-1][0])
            if changes[-1][1] == c:
                return
            if changes[-1][0] == f:
                changes[-1] = (f, c)
            else:
                changes.append((f, c))

        for r in rows:
            team = H if r["Team"] == "Home" else A
            typ, sub = r["Type"], r["Subtype"]
            s = int(r["Start Frame"])
            en = r["End Frame"]
            en = int(en) if pd.notna(en) and en >= s else s
            if typ in ("PASS", "RECOVERY", "SET PIECE", "SHOT"):
                push(s, team)
                dead = False
                if typ == "SHOT" and any(k in sub for k in DEAD_SHOT):
                    push(en, D)
                    dead = True
            elif typ == "BALL OUT" and not dead:
                push(en, D); dead = True
            elif typ == "FAULT RECEIVED" and not dead:
                push(s, D); dead = True
            elif typ == "BALL LOST" and not dead and any(k in sub for k in DEAD_LOST):
                push(en, D); dead = True
        changes.append((int(f1) + 1, -1))
        for (fa, c), (fb, _) in zip(changes[:-1], changes[1:]):
            lab[fa:min(fb, n_total)] = c
    return lab


def spells_summary(lab):
    """(n_possessions, n_dead, median_possession_s@25fps, shares) -- quick sanity check of the labels."""
    l = lab[lab >= 0]
    cut = np.r_[True, l[1:] != l[:-1]]
    starts = np.flatnonzero(cut)
    lens = np.diff(np.r_[starts, len(l)])
    labs = l[starts]
    poss = lens[labs != D] / 25.0
    return dict(possessions=int((labs != D).sum()), dead_spells=int((labs == D).sum()),
                median_poss_s=float(np.median(poss)) if len(poss) else float("nan"),
                share={k: round(float((l == i).mean()), 3) for k, i in (("H", H), ("A", A), ("D", D))})
