"""Frame macro-F1 and possession-spell F1 (own implementation, close in spirit to the organisers' scorer;
use evaluation/run_evaluation.py of the hackathon repo for the official numbers)."""
import numpy as np
from .schema import H, A, D


def f1_per_class(y, p, classes=(H, A, D)):
    out = []
    for c in classes:
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return np.array(out)


def macro_f1(y, p):
    m = y >= 0
    return float(f1_per_class(y[m], p[m]).mean())


def spells(codes):
    """Possession spells: maximal runs of H or A (a D or unknown frame ends a spell) -> [(label, start, end_excl)]."""
    out, n, i = [], len(codes), 0
    while i < n:
        j = i
        while j < n and codes[j] == codes[i]:
            j += 1
        if codes[i] in (H, A):
            out.append((int(codes[i]), i, j))
        i = j
    return out


def spell_f1(y, p, iou=0.5):
    g, q = spells(y), spells(p)
    used, tp = set(), 0
    for lab, a, b in g:
        best, bj = 0.0, None
        for j, (l2, a2, b2) in enumerate(q):
            if l2 != lab or j in used:
                continue
            inter = max(0, min(b, b2) - max(a, a2))
            u = (b - a) + (b2 - a2) - inter
            v = inter / u if u else 0
            if v > best:
                best, bj = v, j
        if best >= iou:
            tp += 1; used.add(bj)
    prec = tp / max(len(q), 1); rec = tp / max(len(g), 1)
    return (0.0 if tp == 0 else 2 * prec * rec / (prec + rec)), len(g), len(q)


def score_blocks(y, p, blocks):
    """Mean over halves of (macro F1, spell F1); unknown-label frames are excluded from F1 and treated as a break in spells."""
    ms, ss, ng, npred = [], [], 0, 0
    for _, a, b in blocks:
        yy, pp = y[a:b + 1], p[a:b + 1]
        ms.append(macro_f1(yy, pp))
        pp2 = np.where(yy >= 0, pp, -1)
        s, g, q = spell_f1(yy, pp2); ss.append(s); ng += g; npred += q
    return dict(macroF1=float(np.mean(ms)), spellF1=float(np.mean(ss)), gt_spells=ng, pred_spells=npred)
