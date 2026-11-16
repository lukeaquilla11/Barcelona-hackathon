"""Search / validate / filter engine (same logic as your auto_sweep.py) for the uploaded model's 11 tunable PARAMS.
Silver labels from sample events or IDSSE flags; tracking degraded to look like video tracking."""
import itertools, re, shutil, time
import numpy as np, pandas as pd
from .orig_bridge import viterbi_fast
from ..supervised_training import metrics

FPS = 25
NOCLUE = dict(tag_weight=0)             # no Smart Tagging exists for sample / IDSSE games
RANGES = dict(possessions=(250, 400), stoppages=(100, 160), median_len_s=(6, 15), dead_share=(0.35, 0.45))
MIN_GAIN = 0.002

DEC = dict(switch_cost=[5, 8, 12, 20, 30, 45], dead_cost=[4, 6, 10, 15, 25])
EMIT = dict(dist_scale=[1, 1.5, 2, 3, 4], near=[2, 3, 4, 6, 8], fast_ball=[4, 6, 8, 12],
            fast_weight=[0.1, 0.3, 0.6, 1.0], speed_dead=[1.0, 1.15, 1.3, 1.5, 1.7],
            speed_scale=[0.2, 0.35, 0.5, 0.8], speed_weight=[0.3, 0.6, 1.0, 1.5],
            missing_dead=[0.0, 0.05, 0.15, 0.3], out_weight=[0.0, 1.0, 2.0, 3.0])
KEYS = list(DEC) + list(EMIT)


class Game:
    def __init__(self, name, dss, labels, blocks):
        self.name, self.dss, self.labels, self.blocks = name, dss, labels, blocks    # dss: list of F dicts (one per degradation seed)


class Sweep:
    """Objective over 'units' = (game, half) pairs, each averaged over the degradation seeds."""
    def __init__(self, pm, games, objective="both"):
        self.pm, self.G, self.obj = pm, games, objective

    def f(self, m, s):
        return {"both": (m + s) / 2, "spell": s, "macro": m}[self.obj]

    def _scores(self, g, es, p):
        """array [seed, half, (macro, spell)] for game g given emission scores es (one per seed)."""
        G, out = self.G[g], []
        for e in es:
            row = []
            for a, b in G.blocks:
                path = viterbi_fast(e[a:b + 1], p["switch_cost"], p["dead_cost"])
                y = G.labels[a:b + 1]
                row.append((metrics.macro_f1(y, path), metrics.spell_f1(y, np.where(y >= 0, path, -1))[0]))
            out.append(row)
        return np.array(out)

    def _es(self, g, p):
        return [self.pm.frame_scores(F, p) for F in self.G[g].dss]

    def evaluate(self, p, units):
        vals = {}
        for g in sorted({g for g, _ in units}):
            sc = self._scores(g, self._es(g, p), p)
            vals[g] = np.array([[self.f(*x) for x in r] for r in sc]).mean(0)       # per half, mean over seeds
        return float(np.mean([vals[g][h] for g, h in units]))

    def grid(self, p, units, dec):
        gs = sorted({g for g, _ in units})
        es = {g: self._es(g, p) for g in gs}
        best = (-1, p)
        for sw, dd in itertools.product(dec["switch_cost"], dec["dead_cost"]):
            q = dict(p, switch_cost=sw, dead_cost=dd)
            vals = {}
            for g in gs:
                sc = self._scores(g, es[g], q)
                vals[g] = np.array([[self.f(*x) for x in r] for r in sc]).mean(0)
            k = float(np.mean([vals[g][h] for g, h in units]))
            if k > best[0] + 1e-12:
                best = (k, q)
        return best[1]

    def fit(self, start, units, dec, emit, tag):
        t0 = time.time()
        p = self.grid(start, units, dec)
        cur = self.evaluate(p, units)
        for _ in range(2):
            moved = False
            for name, vals in emit.items():
                bv, bk = p[name], cur
                for v in vals:
                    if v == p[name]:
                        continue
                    k = self.evaluate(dict(p, **{name: v}), units)
                    if k > bk + MIN_GAIN:
                        bv, bk = v, k
                if bv != p[name]:
                    p[name] = bv; cur = bk; moved = True
            if not moved:
                break
        p = self.grid(p, units, dec)
        print(f"  fit {tag:<24} done in {(time.time() - t0) / 60:.1f} min", flush=True)
        return p

    def report(self, g, p):
        sc = self._scores(g, self._es(g, p), p)                                      # seeds x halves x 2
        return dict(macro=float(sc[..., 0].mean()), spell=float(sc[..., 1].mean()),
                    key=float(np.mean([self.f(m, s) for m, s in sc.reshape(-1, 2)])))


def agreed_changes(cur, fits, extra=None):
    """Keep a change only if EVERY fold's fit moved it the same way; use the smallest agreeing move."""
    final = dict(cur)
    for k in KEYS:
        d = [np.sign(f[k] - cur[k]) for f in fits]
        if d[0] != 0 and all(x == d[0] for x in d):
            agree = [f[k] for f in fits]
            if extra is not None and np.sign(extra[k] - cur[k]) == d[0]:
                agree.append(extra[k])
            final[k] = min(agree, key=lambda v: abs(v - cur[k]))
    return final


def blend(cur, new, t):
    out = dict(cur)
    for k in KEYS:
        if new[k] != cur[k]:
            out[k] = round(float(cur[k] + t * (new[k] - cur[k])), 3)
    return out


def match_stats(lab):
    g = lab[lab != "X"]
    cut = np.r_[0, np.where(g[1:] != g[:-1])[0] + 1, len(g)]
    sp = [(g[a], (b - a) / FPS) for a, b in zip(cut[:-1], cut[1:])]
    poss = [d for l, d in sp if l in "HA"]; dead = [d for l, d in sp if l == "D"]
    return dict(possessions=len(poss), stoppages=len(dead), median_len_s=float(np.median(poss)),
                dead_share=float((g == "D").mean()))


def violation(s):
    return sum(max(lo - s[k], s[k] - hi, 0) / max(hi - lo, 1e-9) for k, (lo, hi) in RANGES.items())


def real_check(pm, cur, new, F_real):
    """Shrink the change toward `cur` until whole-match counts on the real match are inside RANGES (compare.py's ranges)."""
    rows, best = [], None
    for t in (1.0, 0.75, 0.5, 0.25, 0.0):
        p = blend(cur, new, t)
        s = match_stats(pm.predict(F=F_real, **{k: p[k] for k in KEYS}))
        v = violation(s)
        rows.append(dict(strength=f"{t:.0%}", **{k: round(x, 3) for k, x in s.items()}, violation=round(v, 3)))
        if best is None or v < best[0] - 1e-12:
            best = (v, p)
        if v == 0:
            break
    return best[1], pd.DataFrame(rows)


def apply_params(path, diff):
    shutil.copyfile(path, path + ".bak")
    src = open(path).read()
    for k, v in diff.items():
        src, n = re.subn(rf"(^\s+{k}=)[-0-9.eE]+(\s*,)", rf"\g<1>{v}\g<2>", src, count=1, flags=re.M)
        assert n == 1, f"could not find {k}= in PARAMS"
    open(path, "w").write(src)


def run(pm, games, fit_names, test_names, args, repo, out_json, folds=None):
    """The whole auto_sweep procedure. folds: {fold_name: [(game, half), ...]} (default: halves of the single fitting
    game, or one fold per fitting game when there are several)."""
    import json, os
    quick = args.quick
    dec, emit = DEC, EMIT
    if quick:
        dec = dict(switch_cost=[8, 20, 45], dead_cost=[6, 15]); emit = {k: v[::2] for k, v in EMIT.items()}
    sw = Sweep(pm, games, args.objective)
    if folds is None:
        if len(fit_names) >= 2:
            folds = {f"game {g}": [(g, h) for h in range(len(games[g].blocks))] for g in fit_names}
        else:
            g = fit_names[0]
            folds = {f"{g} half {h + 1}": [(g, h)] for h in range(len(games[g].blocks))}
    all_units = [u for us in folds.values() for u in us]
    cur = dict(pm.PARAMS, **NOCLUE)
    base_k = {f: sw.evaluate(cur, us) for f, us in folds.items()}
    print(f"\nFIT on {fit_names}, TEST on {test_names or 'nothing (cross-check only)'}, objective {args.objective}.")
    print("Current PARAMS per fold:", {f: round(v, 4) for f, v in base_k.items()}, flush=True)

    print("\n1. SEARCH")
    fits = {f: sw.fit(cur, us, dec, emit, f) for f, us in folds.items()}
    fa = sw.fit(cur, all_units, dec, emit, "all folds")

    print("\n2. CROSS-VALIDATION (the honest gain)")
    gains = []
    for f, p in fits.items():
        others = [u for g2, us in folds.items() if g2 != f for u in us]
        before, after = sw.evaluate(cur, others), sw.evaluate(p, others)
        gains.append(after - before)
        print(f"  fit on {f:<18} -> scored on the rest: {before:.4f} -> {after:.4f}  ({after - before:+.4f})")
    generalises = all(x > 0.005 for x in gains)
    print("  verdict:", "the search generalises across folds" if generalises
          else "NOT clearly better on held-out folds: keeping current PARAMS")

    print("\n3. FILTER (kept only where every fold's fit moved the same way)")
    tab = pd.DataFrame({"current": {k: cur[k] for k in KEYS}, **{f"fit {f}": {k: p[k] for k in KEYS} for f, p in fits.items()},
                        "fit all": {k: fa[k] for k in KEYS}})
    filt = agreed_changes(cur, list(fits.values()), fa) if generalises else dict(cur)
    tab["kept"] = pd.Series({k: filt[k] for k in KEYS})
    print(tab.to_string())
    k_new = {f: sw.evaluate(filt, us) for f, us in folds.items()}
    print("  key with kept changes (in-sample for the folds):", {f: f"{k_new[f]:.4f} (was {base_k[f]:.4f})" for f in folds})
    if np.mean(list(k_new.values())) < np.mean(list(base_k.values())) + MIN_GAIN:
        print("  kept changes do not beat the current PARAMS by the minimum gain: reverting to current"); filt = dict(cur)

    test_out = {}
    if test_names:
        print("\n4. HELD-OUT TEST (never used for fitting or filtering)")
        cands = {"current PARAMS": cur, "kept changes": filt, "fit all folds": fa, **{f"fit {f}": p for f, p in fits.items()}}
        for g in test_names:
            rows = {n: sw.report(g, dict(p)) for n, p in cands.items()}
            test_out[g] = {n: {k: round(v, 4) for k, v in r.items()} for n, r in rows.items()}
            df = pd.DataFrame(rows).T.round(4); df["vs current"] = (df["key"] - df.loc["current PARAMS", "key"]).round(4)
            print(f"  {g}:"); print(df.to_string())
            d = rows["kept changes"]["key"] - rows["current PARAMS"]["key"]
            print("  verdict:", "no change kept, nothing to test" if filt == cur else
                  f"kept changes {'help' if d > 0.005 else 'hurt' if d < -0.005 else 'make no clear difference on'} {g} ({d:+.4f})")

    final = filt
    if repo and not args.no_real and filt != cur:
        print("\n5. REAL-MATCH SANITY (all clues on; shrink the change until the counts are in range)")
        try:
            from features import features as real_features      # the repo's own (needs the gated data bundle)
            F_real = real_features()
            final, rows = real_check(pm, cur, filt, F_real)
            print(rows.to_string(index=False)); print("  ranges:", RANGES)
        except Exception as ex:
            final = filt
            print(f"  skipped ({type(ex).__name__}: {str(ex)[:120]}). Run solution/compare.py yourself after applying.")

    diff = {k: final[k] for k in KEYS if final[k] != pm.PARAMS[k]}
    print("\nFINAL: PARAMS that differ from possession_model.py:", diff if diff else "none (keep the file as it is)")
    if diff:
        print("Paste into PARAMS:"); [print(f"    {k}={v},") for k, v in diff.items()]
    json.dump({"fit": fit_names, "test": test_names, "final": {k: final[k] for k in KEYS}, "diff": diff,
               "cv_gains": [round(float(x), 4) for x in gains], "test_scores": test_out}, open(out_json, "w"), indent=1, default=float)
    if args.apply and diff:
        path = os.path.join(repo, "solution", "possession_model.py")
        apply_params(path, diff); print(f"applied to {path} (backup: .bak)")
    elif diff:
        print("not applied; re-run with --apply, or paste the values into PARAMS yourself")
    return final
