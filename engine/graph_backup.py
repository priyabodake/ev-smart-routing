"""Data loading (METR-LA or synthetic fallback) and road-graph model."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
import numpy as np, pandas as pd, networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class Network:
    N: int
    keep: np.ndarray            # indices of original sensors that remain in the graph
    mask: np.ndarray            # bool adjacency
    dist: np.ndarray            # edge length (km)
    graph: nx.Graph
    pos: np.ndarray             # [N,2] plot coordinates (schematic layout, or lon/lat if available)
    geo: bool                   # True if pos are real lon/lat
    deg: np.ndarray
    profile_kmh: np.ndarray     # [288, N] typical weekday speed profile
    v_free: np.ndarray          # [N] free-flow speed
    source: str
    k_used: int
    edges: np.ndarray = field(default=None)
    sensor_ids: list = field(default=None)


def _find(root, name):
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower() == name.lower():
                return os.path.join(dp, f)
    return None


def load_metr_la(data_dir):
    h5, adj = _find(data_dir, "metr-la.h5"), _find(data_dir, "adj.npz")
    if h5 is None or adj is None:
        return None
    speeds = pd.read_hdf(h5)
    z = np.load(adj)
    if all(k in z.files for k in ("data", "indices", "indptr", "shape")):
        W = csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])).toarray()
    elif "adjacency" in z.files:
        W = z["adjacency"]
    else:
        W = z[z.files[0]]
    if not isinstance(speeds.index, pd.DatetimeIndex):
        speeds.index = pd.date_range("2012-03-01", periods=len(speeds), freq="5min")
    return speeds, np.asarray(W, float)


def make_synthetic(n=150, days=14, seed=0):
    r = np.random.default_rng(seed)
    xy = r.uniform(0, 40, (n, 2))
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    W = np.exp(-(d / d.std()) ** 2); W[W < 0.1] = 0.0
    idx = pd.date_range("2012-03-05", periods=days * 288, freq="5min")
    h = idx.hour.values + idx.minute.values / 60.0
    wk = idx.dayofweek.values < 5
    core = np.exp(-np.linalg.norm(xy - xy.mean(0), axis=1) / 15.0)
    free = r.uniform(55, 68, n)
    dip = (0.55 * np.exp(-((h - 8.0) / 1.4) ** 2) + 0.45 * np.exp(-((h - 17.0) / 1.8) ** 2))[:, None] * wk[:, None]
    spd = free[None] * (1 - dip * (0.25 + 0.6 * core)[None]) + r.normal(0, 1.5, (len(idx), n))
    spd = np.clip(spd, 5, 75); spd[r.random(spd.shape) < 0.02] = 0.0
    return pd.DataFrame(spd, index=idx), W


def build_graph(W, knn, d_min, d_max):
    """kNN-sparsify the Gaussian-kernel similarity, keep the largest component, invert kernel -> km."""
    W = np.array(W, float); np.fill_diagonal(W, 0.0); W = np.maximum(W, W.T); n = W.shape[0]

    def knn_mask(k):
        idx = np.argsort(-W, axis=1)[:, :k]
        m = np.zeros((n, n), bool); m[np.repeat(np.arange(n), k), idx.ravel()] = True
        m &= (W > 0)
        return m | m.T

    k = knn
    while True:
        M = knn_mask(k)
        _, lab = connected_components(csr_matrix(M), directed=False)
        sizes = np.bincount(lab)
        if sizes.max() >= 0.95 * n or k >= 40:
            break
        k += 2
    keep = np.where(lab == sizes.argmax())[0]
    M = M[np.ix_(keep, keep)]; Wk = W[np.ix_(keep, keep)]
    s = np.sqrt(-np.log(np.clip(Wk[M], 1e-6, 1.0))); s = s / s.max()
    dist = np.zeros_like(Wk); dist[M] = d_min + (d_max - d_min) * s
    return keep, M, dist, k


def build_network(data_dir="data", knn=6, edge_km_min=6.0, edge_km_max=60.0) -> Network:
    loaded = load_metr_la(data_dir) if os.path.isdir(data_dir) else None
    if loaded is None:
        speeds, W = make_synthetic(); source = "synthetic"
    else:
        speeds, W = loaded; source = "METR-LA"
    keep, M, dist, k = build_graph(W, knn, edge_km_min, edge_km_max)
    N = len(keep)
    G = nx.from_numpy_array(np.where(M, dist, 0.0))
    nx.set_edge_attributes(G, {(u, v): d["weight"] for u, v, d in G.edges(data=True)}, "dist")
    spd = speeds.replace(0, np.nan)
    tod = (spd.index.hour * 12 + spd.index.minute // 5).values
    wd = spd.index.dayofweek < 5
    prof = spd[wd].groupby(tod[wd]).median().reindex(range(288))
    prof = prof.interpolate(limit_direction="both").fillna(spd.stack().median())
    profile = prof.values[:, keep] * 1.609344
    v_free = np.percentile(profile, 95, axis=0)
    geo, pos = False, None
    loc = _find(data_dir, "graph_sensor_locations.csv") if os.path.isdir(data_dir) else None
    if loc is not None and source == "METR-LA":
        df = pd.read_csv(loc)
        if len(df) == W.shape[0] and {"latitude", "longitude"} <= set(df.columns):
            pos = df[["longitude", "latitude"]].values[keep]; geo = True
    if pos is None:
        p = nx.kamada_kawai_layout(G, weight="dist"); pos = np.array([p[i] for i in range(N)])
    return Network(N=N, keep=keep, mask=M, dist=dist, graph=G, pos=np.asarray(pos), geo=geo,
                   deg=np.array([G.degree(i) for i in range(N)]), profile_kmh=profile, v_free=v_free,
                   source=source, k_used=k, edges=np.array(list(G.edges())),
                   sensor_ids=[str(c) for c in speeds.columns[keep]])
