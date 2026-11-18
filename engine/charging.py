"""Charging stations: configurable availability, Erlang-C waiting time, tapered charging time."""
import math
import numpy as np

W_CAP = 120.0


def erlang_c(c, a):
    if a >= c: return 1.0
    s = sum(a ** k / math.factorial(k) for k in range(c))
    top = a ** c / math.factorial(c) * c / (c - a)
    return top / (s + top)


def expected_wait_min(c, lam, mu):
    a = lam / mu
    if c < 1 or a >= c: return W_CAP
    return min(W_CAP, erlang_c(int(c), a) / (c * mu - lam) * 60.0)


def charge_minutes(e0, e1, p_kw, cfg):
    if e1 <= e0: return 0.0
    cap, knee = cfg.cap, cfg.taper_knee
    p = min(p_kw, cfg.vehicle_max_kw); t = 0.0
    a = min(e1, knee * cap)
    if e0 < a: t += (a - e0) / p
    b = max(e0, knee * cap)
    if e1 > b: t += (e1 - b) / (0.4 * p)
    return 60.0 * t


def make_stations(net, D_matrix, n_stations=24, seed=42):
    """Overlay of charging stations on graph nodes (farthest-point sampling among well-connected nodes)."""
    rng = np.random.default_rng(seed)
    cand = np.where(net.deg >= np.median(net.deg))[0]
    n_stations = min(n_stations, len(cand))
    chosen = [int(cand[np.argmax(net.deg[cand])])]; mind = D_matrix[chosen[0]].copy()
    while len(chosen) < n_stations:
        j = int(cand[np.argmax(mind[cand])]); chosen.append(j); mind = np.minimum(mind, D_matrix[j])
    K = len(chosen)
    return dict(node=np.array(chosen), ports=rng.choice([2, 3, 4, 6], K), power=rng.choice([50., 50., 100., 150.], K),
                rho0=rng.uniform(0.40, 0.80, K))


def resolve_ports_up(st, mode="random", p_down=0.10, seed=0, manual=None):
    """Station availability. mode: 'all_up' | 'random' | 'manual' (manual = list of ports in service)."""
    if mode == "all_up": return st["ports"].copy()
    if mode == "manual" and manual is not None:
        return np.clip(np.asarray(manual, int), 0, st["ports"])
    rng = np.random.default_rng(seed)
    return np.maximum(1, st["ports"] - rng.binomial(st["ports"], p_down))


def station_waits(st, ports_up, snap, session_min=30.0, lam_mult=None, ports_eff=None):
    mu = 60.0 / session_min
    rho = np.clip(st["rho0"] * (0.6 + 1.2 * snap.cong[st["node"]]), 0.05, 0.95)
    lam = rho * st["ports"] * mu * (1.0 if lam_mult is None else lam_mult)
    c = ports_up if ports_eff is None else ports_eff
    return np.array([expected_wait_min(int(ci), li, mu) for ci, li in zip(c, lam)]), lam
