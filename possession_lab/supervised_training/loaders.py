"""Turn each data source into a schema.Match.

  load_metrica_sample(tracking_csv, events_csv)   your 'combined' Metrica sample tracking + RawEventsData (silver labels)
  match_from_kloppy_df(df, ...)                   any kloppy tracking dataframe (IDSSE / Metrica EPTS)
  load_idsse(match_id)                            IDSSE via kloppy.sportec.load_open_tracking_data (needs kloppy + internet)
  save_match / load_match_cache                   npz cache so IDSSE is downloaded once
"""
import os, re
import numpy as np
import pandas as pd
from .schema import Match, H, A, D
from .labels_metrica import build_labels

# --------------------------------------------------------------------------------------------- Metrica samples
SAMPLE_L, SAMPLE_W = 105.0, 68.0


def _slot_columns(df):
    return sorted(int(m.group(1)) for c in df.columns if (m := re.fullmatch(r"(\d+)_x", c)))


def _parse_slots(spec):
    """'1-14' or '1-13,27' -> set of ints."""
    out = set()
    for part in str(spec).split(","):
        a, _, b = part.strip().partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def load_metrica_sample(tracking_csv, events_csv, name=None, L=SAMPLE_L, W=SAMPLE_W, home_slots=None, away_slots=None):
    """Combined tracking (period_id,timestamp,frame_id,ball_*,{slot}_x/_y/_d/_s ; coords normalised 0-1) + events.
    Slot i == 'Player i' of the events file (home = Player1-14, away = Player15-28). The team of every slot is read
    from the events file (From/To columns); slots that never appear fall back to id<=14 -> home."""
    df = pd.read_csv(tracking_csv)
    ev = pd.read_csv(events_csv)
    slots = [s for s in _slot_columns(df) if df[f"{s}_x"].notna().any()]
    team_of = {}
    for col in ("From", "To"):
        for team, pl in zip(ev["Team"], ev[col]):
            if isinstance(pl, str) and pl.startswith("Player"):
                team_of.setdefault(int(pl[6:]), 0 if team == "Home" else 1)
    # 'To' of a pass belongs to the passer's team, 'From' of a challenge/loss is trustworthy too; From was set first.
    if home_slots or away_slots:        # manual override, e.g. home_slots="1-14", away_slots="15-28"
        hs, as_ = _parse_slots(home_slots or ""), _parse_slots(away_slots or "")
        team_of = {**team_of, **{s: 0 for s in hs}, **{s: 1 for s in as_}}
    team = np.array([team_of.get(s, 0 if s <= 14 else 1) for s in slots], np.int8)
    n = len(df)
    P = np.stack([np.column_stack([df[f"{s}_x"] * L, df[f"{s}_y"] * W]) for s in slots], axis=1).astype(np.float32)
    B = np.column_stack([df["ball_x"] * L, df["ball_y"] * W]).astype(np.float32)
    period = df["period_id"].to_numpy().astype(int)
    fid = df["frame_id"].to_numpy().astype(int)
    ranges = {int(p): (int(fid[period == p].min()), int(fid[period == p].max())) for p in np.unique(period)}
    lab_by_frame = build_labels(ev, ranges, int(fid.max()) + 2)
    labels = lab_by_frame[fid]
    m = Match(name or os.path.basename(tracking_csv), P, B, team, period, labels, 25.0, L, W)
    m.slots = slots
    print(f"[{m.name}] slots->team  home: {[s for s, t in zip(slots, team) if t == 0]}  away: {[s for s, t in zip(slots, team) if t == 1]}")
    print(f"[{m.name}] alignment check:", check_alignment(df, ev, slots, L, W))
    return m


def check_alignment(df, ev, slots, L=SAMPLE_L, W=SAMPLE_W):
    """Sanity check of slot numbering, 0-1 scaling and frame alignment: for every event with a 'From' player and a start
    position, how far (m) was that tracked slot from the event's Start X/Y at Start Frame? Median should be ~1-3 m;
    ~15-30 m means the slot numbering, coordinates or frame offset is wrong."""
    fid = df["frame_id"].to_numpy()
    row = {int(f): i for i, f in enumerate(fid)}
    d = []
    for r in ev.itertuples(index=False):
        frm, sf, sx, sy = r[8], r[4], r[10], r[11]
        if not (isinstance(frm, str) and frm.startswith("Player") and np.isfinite(sx)):
            continue
        s, i = int(frm[6:]), row.get(int(sf))
        if i is None or s not in slots:
            continue
        x, y = df[f"{s}_x"].iat[i], df[f"{s}_y"].iat[i]
        if np.isfinite(x):
            d.append(np.hypot((x - sx) * L, (y - sy) * W))
    if not d:
        return "no comparable events"
    d = np.array(d)
    return dict(n=len(d), median_m=round(float(np.median(d)), 2), p90_m=round(float(np.percentile(d, 90)), 2))


# --------------------------------------------------------------------------------------------- kloppy dataframes
def match_from_kloppy_df(df, home_player_ids, away_player_ids, home_team_id=None, L=105.0, W=68.0,
                         normalised=True, name="match", fps=25.0, label_from_df=True):
    """kloppy `dataset.to_df()` -> Match. Player columns are '{player_id}_x' / '{player_id}_y'.
    normalised=True : x,y in 0-1 (kloppy default coordinates) -> multiplied by pitch L,W.
    normalised=False: x,y already metres with origin at the pitch corner.
    Labels (IDSSE): ball_state 'dead' -> D, else ball_owning_team_id == home_team_id -> H, other -> A."""
    ids = [str(i) for i in list(home_player_ids) + list(away_player_ids)]
    ids = [i for i in ids if f"{i}_x" in df.columns and df[f"{i}_x"].notna().any()]
    hs = {str(i) for i in home_player_ids}
    team = np.array([0 if i in hs else 1 for i in ids], np.int8)
    sx, sy = (L, W) if normalised else (1.0, 1.0)
    P = np.stack([np.column_stack([df[f"{i}_x"] * sx, df[f"{i}_y"] * sy]) for i in ids], axis=1).astype(np.float32)
    B = np.column_stack([df["ball_x"] * sx, df["ball_y"] * sy]).astype(np.float32)
    period = df["period_id"].to_numpy().astype(int)
    labels = None
    if label_from_df and "ball_state" in df.columns and "ball_owning_team_id" in df.columns:
        dead = df["ball_state"].astype(str).str.lower().str.contains("dead").to_numpy()
        own = df["ball_owning_team_id"].astype(str).to_numpy()
        labels = np.full(len(df), -1, np.int8)
        labels[own == str(home_team_id)] = H
        labels[(own != str(home_team_id)) & (own != "None") & (own != "nan")] = A
        labels[dead] = D
        # alive frames with no owner stay -1 (excluded from training/scoring)
    return Match(name, P, B, team, period, labels, fps, L, W)


def load_idsse(match_id, limit=None):
    """IDSSE (Bassek et al. 2025) through kloppy. only_alive=False keeps the dead-ball frames (we need them for D)."""
    from kloppy import sportec
    kw = dict(match_id=match_id, only_alive=False)
    if limit:
        kw["limit"] = limit
    ds = sportec.load_open_tracking_data(**kw)
    md = ds.metadata
    home, away = md.teams[0], md.teams[1]
    L = float(getattr(md.pitch_dimensions, "pitch_length", None) or 105.0)
    W = float(getattr(md.pitch_dimensions, "pitch_width", None) or 68.0)
    df = ds.to_df(engine="pandas")
    return match_from_kloppy_df(df, [p.player_id for p in home.players], [p.player_id for p in away.players],
                                home_team_id=home.team_id, L=L, W=W, normalised=True,
                                name=f"idsse_{match_id}", fps=float(md.frame_rate or 25.0))


# --------------------------------------------------------------------------------------------- cache
def save_match(m: Match, path):
    np.savez_compressed(path, P=m.P, B=m.B, team=m.team, period=m.period,
                        labels=m.labels if m.labels is not None else np.array([]),
                        meta=np.array([m.fps, m.L, m.W]), name=np.array(m.name))


def load_match_cache(path):
    z = np.load(path, allow_pickle=False)
    lab = z["labels"] if z["labels"].size else None
    fps, L, W = z["meta"]
    return Match(str(z["name"]), z["P"], z["B"], z["team"], z["period"], lab, float(fps), float(L), float(W))
