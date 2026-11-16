import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "solution")
import numpy as np, possession_model as pm, external
from features import rolling_mean
from starter.games import half_video_blocks, GAME

F = pm.features(); p = pm.PARAMS; n = len(F["vis"])
m = np.zeros(n, bool)
for _, a, b in half_video_blocks(GAME):
    m[a:b + 1] = True                      # only frames inside the halves
ps = rolling_mean(F["pspeed"], 25)
print("ball unseen %      :", 100 * (~F["vis"])[m].mean())
print("ball outside %     :", 100 * F["out"][m].mean())
print("set-piece frames % :", 100 * external.set_piece_mask(n)[m].mean())
print("calm (ps<speed_dead) %:", 100 * np.nanmean((ps < p["speed_dead"])[m]))
print("player speed p10/25/50/75:", np.round(np.nanpercentile(ps[m], [10, 25, 50, 75]), 2))