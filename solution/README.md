# Ball possession from automatic tracking — More than a Hack 2027

Labels every frame of the tactical video `H` (home in possession), `A` (away in possession), `D` (dead ball), or `X` (ungraded pre-match/halftime/post-match) using a Hidden Markov Model decoded via Viterbi.

This version integrates **homography confidence weighting**, **kinematic ball interpolation**, **Smart Tagging empirical state priors & transition matrices**, and **expanded stoppage emissions** into the modular `solution/` pipeline.

## Directory Structure & Files

| File | Description |
| :--- | :--- |
| `solution/load.py` | Parses tactical EPTS tracking data into cached NumPy arrays (`cache_tactical.npz`). |
| `solution/features.py` | Computes 25-frame kinematic ball interpolation, player-to-ball distances, median player speeds, 1s ball displacement, and Gaussian homography confidence weights (`BAR-ATH_tactical_homography.csv`). |
| `solution/tag_check.py` | Extracts per-frame team possession priors, expanded set-piece/stoppage dead-ball priors, and empirical state transition matrices from Metrica's Smart Tagging XML. |
| `solution/possession_model.py` | Combines sensor emission likelihoods, homography trust weights, and empirical priors into a 3-state Viterbi decoder with micro-spell smoothing (`<19` frames). |
| `solution/compare.py` | Whole-match sanity report (possession counts, stoppage counts, median spell duration, dead-ball share, and Smart Tagging agreement). |
| `solution/requirements.txt` | Python package dependencies (`numpy`, `pandas`). |

## How to Run

1. **Place the match data bundle** in `data/` at the repository root (ensuring `data/tactical/BAR-ATH_tactical_homography.csv` and the tactical EPTS files are present).
2. **Install dependencies** (Python 3.9+):
   ```bash
   python3 -m pip install -r solution/requirements.txt