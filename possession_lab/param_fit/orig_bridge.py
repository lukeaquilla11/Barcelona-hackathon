"""Fast Viterbi implementation used by the original Working File Mini parameter sweep."""
import numpy as np


def viterbi_fast(e, switch, dead):
    """Equivalent to the three-state H/A/D Viterbi decoder, with low Python overhead."""
    n = len(e)
    if n == 0:
        return np.empty(0, np.int8)
    el = e.tolist()
    s0, s1, s2 = el[0][0] - dead, el[0][1] - dead, el[0][2]
    back = bytearray(3 * n)
    for t in range(1, n):
        e0, e1, e2 = el[t]
        a, b, c = s0, s1 - switch, s2 - dead
        if b > a: m, i = b, 1
        else: m, i = a, 0
        if c > m: m, i = c, 2
        n0, k0 = m + e0, i
        a, b, c = s0 - switch, s1, s2 - dead
        if b > a: m, i = b, 1
        else: m, i = a, 0
        if c > m: m, i = c, 2
        n1, k1 = m + e1, i
        a, b, c = s0 - dead, s1 - dead, s2
        if b > a: m, i = b, 1
        else: m, i = a, 0
        if c > m: m, i = c, 2
        n2, k2 = m + e2, i
        s0, s1, s2 = n0, n1, n2
        o = 3 * t
        back[o], back[o + 1], back[o + 2] = k0, k1, k2
    path = np.empty(n, np.int8)
    st = 0 if (s0 >= s1 and s0 >= s2) else (1 if s1 >= s2 else 2)
    path[-1] = st
    for t in range(n - 1, 0, -1):
        st = back[3 * t + st]
        path[t - 1] = st
    return path
