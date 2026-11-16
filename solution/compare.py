"""Side-by-side sanity check of submission files over the WHOLE match, without any ground truth.

    python3 solution/compare.py <team_name>.csv

What to look for (realistic ranges for a professional match, from football knowledge):
  possessions      roughly 250-400 for the whole match
  stoppages        roughly 100-160 (throw-ins, goal kicks, corners, fouls, goals, subs)
  median length    roughly 6-15 s
  dead share       roughly 35-45% of the scored frames (the ball is in play ~55-65% of the time)
  tag agreement    higher is better; Metrica's automatic tags are themselves only ~70% right
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)


def spells(labels):
    g = labels[labels != "X"]
    cut = np.r_[0, np.where(g[1:] != g[:-1])[0] + 1, len(g)]
    return [(g[a], (b - a) / 25) for a, b in zip(cut[:-1], cut[1:])]


def report(path, prior):
    lab = pd.read_csv(path).label.values.astype(str)
    sp = spells(lab)
    poss = [d for l, d in sp if l in "HA"]
    dead = [d for l, d in sp if l == "D"]
    g = lab[lab != "X"]
    row = dict(file=os.path.basename(path), possessions=len(poss), stoppages=len(dead),
               median_len_s=round(float(np.median(poss)), 1),
               short_under_2s=sum(d < 2 for d in poss),
               dead_share=f"{(g == 'D').mean():.0%}", home_share_in_play=f"{(g == 'H').sum() / np.isin(g, ['H', 'A']).sum():.0%}")
    if prior is not None:
        from tag_check import agreement
        row["tag_agreement"] = f"{agreement(lab, prior)[0]:.1%}"
    return row


if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        sys.exit(__doc__)
    try:
        from tag_check import tag_prior
        prior = tag_prior(len(pd.read_csv(files[0])))
    except Exception:
        prior = None
    print(pd.DataFrame([report(f, prior) for f in files]).to_string(index=False))
    print(__doc__.split("What to look for")[1])
