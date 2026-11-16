"""Common in-memory representation shared by every data source (Metrica samples, IDSSE, hackathon)."""
from dataclasses import dataclass, replace
import numpy as np

H, A, D = 0, 1, 2                      # class codes
CLASSES = np.array(["H", "A", "D"])


@dataclass
class Match:
    name: str
    P: np.ndarray                      # (n, S, 2) player x,y in METRES, origin = pitch corner, NaN = not tracked
    B: np.ndarray                      # (n, 2)    ball x,y in metres, NaN = not tracked
    team: np.ndarray                   # (S,)      0 = home, 1 = away, -1 = ignore
    period: np.ndarray                 # (n,)      1,2,... ; <=0 means "not part of a half" (ignored)
    labels: np.ndarray = None          # (n,)      0=H 1=A 2=D, -1 = unknown
    fps: float = 25.0
    L: float = 105.0
    W: float = 68.0

    def blocks(self):
        """Contiguous runs of one period: [(period, first_idx, last_idx_inclusive)]."""
        out, p, n = [], self.period, len(self.period)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and p[j + 1] == p[i]:
                j += 1
            if p[i] > 0:
                out.append((int(p[i]), i, j))
            i = j + 1
        return out

    def slice(self, a, b):
        """Rows a..b inclusive as a new Match."""
        lab = None if self.labels is None else self.labels[a:b + 1]
        return replace(self, P=self.P[a:b + 1], B=self.B[a:b + 1], period=self.period[a:b + 1], labels=lab)

    def halves(self):
        return [self.slice(a, b) for _, a, b in self.blocks()]

    def swapped(self):
        """Same match with home and away exchanged (labels H<->A). Used for augmentation and test-time symmetrising."""
        t = np.where(self.team >= 0, 1 - self.team, -1)
        lab = None
        if self.labels is not None:
            lab = self.labels.copy()
            lab[self.labels == H] = A
            lab[self.labels == A] = H
        return replace(self, team=t, labels=lab)

    def with_tracking(self, P, B):
        return replace(self, P=P, B=B)
