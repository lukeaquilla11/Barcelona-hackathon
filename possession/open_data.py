"""Open matches with per-frame possession labels: our legal tuning and validation set.

CAVEAT -- the DFL possession flag is looser than the challenge ground truth. On J03WMX it gives
9.0 possessions per minute of live play (median 3.0 s, 17% under 1 s), against roughly 4.2 (median
7.6 s, 6% under 1 s) in the public 5 minutes of our match: DFL flips on any touch, including
deflections that go straight back. Use BallStatus (dead/alive) as is; derive "controlled"
possession from the DFL flag before tuning possession logic on it.

The IDSSE dataset (Bassek et al., Sci Data 2025, CC-BY 4.0) has 7 Bundesliga matches of DFL
tracking at 25 fps. Every ball frame carries the two flags the challenge ground truth is derived
from on Metrica's side:

    BallStatus     1 = ball in play, 0 = dead
    BallPossession 1 = home team owns it, 2 = away team

so `label = D if dead else owning team` gives exactly the H/A/D the challenge asks for. Rules
allow tuning on other games in any amount; it is the public 5 minutes we may not tune on.

    from possession.open_data import load_open_match, IDSSE_MATCHES
    o = load_open_match("J03WMX")      # downloads (~420 MB, once) and parses (~1 min, once)
    o.labels                           # (N,) 'H'/'A'/'D' per frame
    o.ball_xy, o.players_xy, o.track_team   # same layout and units as load_data.Match

Raw downloads go to OPEN_DATA_DIR (default: barcelona/open_data/idsse, outside the repo); the
parsed arrays are cached in cache/.
"""

import os
import sys
import urllib.request
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")
OPEN_DATA_DIR = os.environ.get("OPEN_DATA_DIR") or os.path.join(
    os.path.dirname(ROOT), "open_data", "idsse")
URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/{}"
FILES = {"tracking": "DFL_04_03_positions_raw_observed_DFL-COM-{comp}_DFL-MAT-{id}.xml",
         "meta": "DFL_02_01_matchinformation_DFL-COM-{comp}_DFL-MAT-{id}.xml",
         "event": "DFL_03_02_events_raw_DFL-COM-{comp}_DFL-MAT-{id}.xml"}
# Competition code in the file names: 000001 = Bundesliga, 000002 = 2. Bundesliga.
COMPETITION = {"J03WMX": "000001", "J03WN1": "000001"}

IDSSE_MATCHES = {
    "J03WMX": "1. FC Köln vs FC Bayern München",
    "J03WN1": "VfL Bochum vs Bayer 04 Leverkusen",
    "J03WPY": "Fortuna Düsseldorf vs 1. FC Nürnberg",
    "J03WOH": "Fortuna Düsseldorf vs SSV Jahn Regensburg",
    "J03WQQ": "Fortuna Düsseldorf vs FC St. Pauli",
    "J03WOY": "Fortuna Düsseldorf vs F.C. Hansa Rostock",
    "J03WR9": "Fortuna Düsseldorf vs 1. FC Kaiserslautern",
}
HOME, AWAY = 0, 1
PITCH = (105.0, 68.0)


@dataclass
class OpenMatch:
    match_id: str
    ball_xy: np.ndarray      # (N, 2) metres, corner origin, like load_data.Match
    ball_z: np.ndarray       # (N,) metres -- our match has no height, use only for analysis
    players_xy: np.ndarray   # (N, P, 2) metres, NaN when the player is off the pitch
    track_team: np.ndarray   # (P,) HOME / AWAY
    half: np.ndarray         # (N,) 1 or 2
    labels: np.ndarray       # (N,) 'H' / 'A' / 'D'
    pitch: tuple = PITCH

    @property
    def n_frames(self):
        return len(self.labels)

    def _with_halftime(self, labels):
        """Insert one ungraded 'X' frame between the halves, as our match has a halftime gap.

        Without it the scorer would see one continuous run and could merge the last possession
        of the first half with the first of the second.
        """
        labels = np.asarray(labels)
        cut = int(np.flatnonzero(self.half == 2)[0])
        return np.concatenate([labels[:cut], ["X"], labels[cut:]])

    @property
    def gt_codes(self):
        """Integer codes in the official scorer's convention, for pm.evaluate()."""
        return pm.label_codes(self._with_halftime(self.labels))

    def score(self, pred_labels, gt_labels=None):
        """Official metrics of a prediction against this match's labels (or `gt_labels`)."""
        gt = self.gt_codes if gt_labels is None else pm.label_codes(self._with_halftime(gt_labels))
        return pm.evaluate(gt, pm.label_codes(self._with_halftime(pred_labels)))


def download(match_id, kinds=("tracking", "meta")):
    """Fetch the raw XML for one match into OPEN_DATA_DIR, skipping files already there."""
    os.makedirs(OPEN_DATA_DIR, exist_ok=True)
    paths = {}
    for kind in kinds:
        name = FILES[kind].format(comp=COMPETITION.get(match_id, "000002"), id=match_id)
        path = os.path.join(OPEN_DATA_DIR, name)
        if not os.path.exists(path):
            print("downloading {} ...".format(name))
            urllib.request.urlretrieve(URL.format(name), path + ".part")
            os.replace(path + ".part", path)
        paths[kind] = path
    return paths


def _parse(match_id):
    from kloppy import sportec  # only needed on the first, uncached load

    paths = download(match_id)
    dataset = sportec.load_tracking(raw_data=paths["tracking"], meta_data=paths["meta"],
                                    coordinates="sportec")       # metres, centre origin
    df = dataset.to_df()
    home, away = dataset.metadata.teams                           # kloppy: home first

    player_ids, track_team = [], []
    for team_code, team in ((HOME, home), (AWAY, away)):
        for player in team.players:
            if "{}_x".format(player.player_id) in df.columns:
                player_ids.append(player.player_id)
                track_team.append(team_code)

    shift = np.array([PITCH[0] / 2, PITCH[1] / 2], dtype=np.float32)  # centre -> corner origin
    players_xy = np.stack([df[["{}_x".format(p), "{}_y".format(p)]].values for p in player_ids],
                          axis=1).astype(np.float32) + shift
    ball_xy = df[["ball_x", "ball_y"]].values.astype(np.float32) + shift

    owner = np.where(df["ball_owning_team_id"].values == home.team_id, "H", "A")
    labels = np.where(df["ball_state"].astype(str).values == "dead", "D", owner)
    return dict(ball_xy=ball_xy, ball_z=df["ball_z"].values.astype(np.float32),
                players_xy=players_xy, track_team=np.array(track_team, dtype=np.int8),
                half=df["period_id"].values.astype(np.int8), labels=labels.astype("<U1"))


def load_open_match(match_id, refresh=False):
    cache = os.path.join(CACHE_DIR, "idsse_{}.npz".format(match_id))
    if os.path.exists(cache) and not refresh:
        arrays = dict(np.load(cache))
    else:
        print("parsing {} (first run only, ~1 min) ...".format(match_id))
        arrays = _parse(match_id)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, **arrays)
    return OpenMatch(match_id=match_id, **arrays)


def possession_stats(labels):
    """Possession count and mean duration, computed the way the scorer counts them."""
    codes = pm.label_codes(np.asarray(labels))
    return pm.possession_stats(codes, pm.valid_runs(codes))


if __name__ == "__main__":
    for mid in sys.argv[1:] or ["J03WMX"]:
        o = load_open_match(mid)
        stats = possession_stats(o.labels)
        print("{} {}: {:,} frames, {} players, {} possessions of mean {:.1f} s, dead {:.0%}".format(
            mid, IDSSE_MATCHES[mid], o.n_frames, len(o.track_team), stats["n_spells"],
            stats["mean_duration_s"], (o.labels == "D").mean()))
