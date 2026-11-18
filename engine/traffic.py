"""Congestion model: speed -> travel time -> energy, for any user-selected departure time."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def tod_to_idx(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":")[:2]); return (h * 12 + m // 5) % 288


def g_speed(v):
    return 1.0 + 0.6 * np.maximum(0.0, (50 - v) / 50) + 0.5 * np.maximum(0.0, (v - 80) / 80)


class Snapshot:
    """All-pairs routing data at one departure time. path_metric: 'distance' or 'time' (fastest)."""

    def __init__(self, net, consumption_kwh_km, hhmm, path_metric="distance"):
        self.hhmm, self.path_metric, self.net = hhmm, path_metric, net
        MASK, DIST = net.mask, net.dist
        v = net.profile_kmh[tod_to_idx(hhmm)]
        self.v_node = v
        self.cong = np.clip(1 - v / net.v_free, 0, 1)
        v_edge = 0.5 * (v[:, None] + v[None, :])
        self.t_edge = np.where(MASK, 60 * DIST / np.maximum(v_edge, 5.0), 0.0)
        self.e_edge = np.where(MASK, DIST * consumption_kwh_km * g_speed(v_edge), 0.0)
        self.v_edge = np.where(MASK, v_edge, np.nan)
        wmat = DIST if path_metric == "distance" else self.t_edge
        C, Pred = dijkstra(csr_matrix(wmat), directed=True, return_predecessors=True)
        self.Pred = Pred
        self.D, self.E, self.T = self._accumulate(Pred, C, [DIST, self.e_edge, self.t_edge])

    @staticmethod
    def _accumulate(Pred, C, mats):
        n = Pred.shape[0]; outs = [np.zeros((n, n)) for _ in mats]
        for s in range(n):
            for v in np.argsort(C[s]):
                p = Pred[s, v]
                if p < 0: continue
                for o, m in zip(outs, mats):
                    o[s, v] = o[s, p] + m[p, v]
        return outs

    def path(self, s, t):
        nodes = [t]
        while nodes[-1] != s:
            nodes.append(int(self.Pred[s, nodes[-1]]))
        return nodes[::-1]
