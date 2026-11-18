"""Bi-objective EV routing problem (min distance, min charging delay) with SoC constraint."""
from __future__ import annotations
import itertools, time
from dataclasses import dataclass
import numpy as np
from .charging import charge_minutes, station_waits

BIG = 1e3  # penalty per kWh of SoC violation


@dataclass
class EVConfig:
    """Vehicle / optimisation settings (all taken from the UI, no globals)."""
    battery_kwh: float = 30.0
    consumption_kwh_km: float = 0.20
    reserve_frac: float = 0.10
    vehicle_max_kw: float = 100.0
    taper_knee: float = 0.80
    session_min: float = 30.0
    max_stops: int = 3
    levels: tuple = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    @property
    def cap(self): return self.battery_kwh
    @property
    def reserve(self): return self.reserve_frac * self.battery_kwh
    @property
    def NL(self): return len(self.levels)


class EVProblem:
    def __init__(self, snap, st, ports_up, src, dst, soc0, cfg: EVConfig):
        self.snap, self.st, self.cfg = snap, st, cfg
        self.D, self.E, self.T = snap.D.tolist(), snap.E.tolist(), snap.T.tolist()
        self.K = len(st["node"]); self.node = st["node"].tolist(); self.power = st["power"].tolist()
        self.src, self.dst, self.soc0 = int(src), int(dst), float(soc0)
        self.ports_up = np.asarray(ports_up)
        self.wait = station_waits(st, self.ports_up, snap, cfg.session_min)[0].tolist()
        D = self.D
        self.progress = [D[src][n] / max(1e-9, D[src][n] + D[n][dst]) for n in self.node]
        self.src_dist = [D[src][n] for n in self.node]
        self.ref1 = max(1.0, D[src][dst]); self.ref2 = 40.0
        self.CAP, self.RESERVE, self.LEVELS, self.NL = cfg.cap, cfg.reserve, cfg.levels, cfg.NL
        self.reset(1000)

    # ---- bookkeeping (every evaluated solution competes for the Pareto archive) ----
    def reset(self, budget, hist_points=25):
        self.budget = budget; self.n_evals = 0; self.archive = []; self.history = []
        self.hist_step = max(1, budget // hist_points)

    def _archive_add(self, sol, f1, f2):
        for a, b, _ in self.archive:
            if a <= f1 + 1e-9 and b <= f2 + 1e-9: return
        self.archive = [t for t in self.archive if not (f1 <= t[0] + 1e-9 and f2 <= t[1] + 1e-9)]
        self.archive.append((f1, f2, sol))

    def eval_raw(self, sol, wait=None):
        """(distance km, delay min, SoC violation kWh, total time min)"""
        D, E, T = self.D, self.E, self.T; wait = self.wait if wait is None else wait
        soc, pos = self.soc0, self.src; f1 = f2 = viol = tm = 0.0; n = len(sol)
        for idx in range(n + 1):
            if idx < n: k, l = sol[idx]; node = self.node[k]
            else: k, node = -1, self.dst
            f1 += D[pos][node]; tm += T[pos][node]; soc -= E[pos][node]
            if soc < self.RESERVE: viol += self.RESERVE - soc; soc = self.RESERVE
            if k >= 0:
                nxt = self.node[sol[idx + 1][0]] if idx + 1 < n else self.dst
                tgt = min(self.CAP, max(self.LEVELS[l] * self.CAP, E[node][nxt] + self.RESERVE))
                if tgt > soc + 1e-9:
                    dly = wait[k] + charge_minutes(soc, tgt, self.power[k], self.cfg); f2 += dly; tm += dly; soc = tgt
            pos = node
        return f1, f2, viol, tm

    def eval_sol(self, sol):
        self.n_evals += 1
        f1, f2, viol, _ = self.eval_raw(sol)
        if viol <= 1e-9: self._archive_add(sol, f1, f2)
        if self.n_evals % self.hist_step == 0:
            self.history.append((self.n_evals, [(a, b) for a, b, _ in self.archive]))
        pen = BIG * viol
        return f1 + pen, f2 + pen, viol

    def scal(self, f1, f2, w): return w * f1 / self.ref1 + (1 - w) * f2 / self.ref2

    # ---- random-key encoding / decoding / neighbourhood ----
    def normalize(self, sol): return tuple(sorted(sol, key=lambda kl: self.progress[kl[0]]))

    def decode(self, x):
        K, NL = self.K, self.NL
        sel = np.where(x[:K] > 0.5)[0]
        if len(sel) > self.cfg.max_stops: sel = sel[np.argsort(-x[sel])[:self.cfg.max_stops]]
        lv = np.clip((x[K:][sel] * NL).astype(int), 0, NL - 1)
        return self.normalize(tuple((int(k), int(l)) for k, l in zip(sel, lv)))

    def encode(self, sol, rng):
        K = self.K; x = np.concatenate([rng.uniform(0.0, 0.4, K), rng.random(K)])
        for k, l in sol: x[k] = rng.uniform(0.6, 1.0); x[K + k] = (l + 0.5) / self.NL
        return x

    def random_sol(self, rng):
        m = rng.integers(0, self.cfg.max_stops + 1); ks = rng.choice(self.K, m, replace=False)
        return self.normalize(tuple((int(k), int(rng.integers(self.NL))) for k in ks))

    def neighbors(self, sol):
        used = {k for k, _ in sol}; out = []; NL = self.NL
        if len(sol) < self.cfg.max_stops:
            for k in range(self.K):
                if k in used: continue
                for l in range(NL): out.append(((k,), self.normalize(sol + ((k, l),))))
        for i, (k, l) in enumerate(sol):
            out.append(((k,), sol[:i] + sol[i + 1:]))
            for dl in (-1, 1):
                if 0 <= l + dl < NL: out.append(((k,), sol[:i] + ((k, l + dl),) + sol[i + 1:]))
            for k2 in range(self.K):
                if k2 in used: continue
                out.append(((k, k2), self.normalize(sol[:i] + ((k2, l),) + sol[i + 1:])))
        return out

    def seed_vectors(self, pop, rng):
        X = rng.random((pop, 2 * self.K)); order = np.argsort(self.src_dist)
        X[0] = self.encode((), rng)
        for i in range(1, min(pop // 4, self.K) + 1): X[i] = self.encode(((int(order[i - 1]), self.NL - 1),), rng)
        return X[rng.permutation(pop)]

    def result(self, t_elapsed):
        arch = sorted(self.archive, key=lambda t: t[0])
        return dict(front=list(arch), history=list(self.history), evals=self.n_evals, time=t_elapsed)

    # ---- explanation of one solution (route, stops, SoC profile, energy) ----
    def describe(self, sol):
        wait_pos = [self.src] + [self.node[k] for k, _ in sol] + [self.dst]
        soc, dcum = self.soc0, 0.0; xs, ys = [0.0], [soc]; stops, legs = [], []
        tot_e = tot_t = tot_wait = tot_chg = 0.0
        for i in range(len(wait_pos) - 1):
            a, b = wait_pos[i], wait_pos[i + 1]
            d, e, t = self.D[a][b], self.E[a][b], self.T[a][b]
            soc -= e; dcum += d; tot_e += e; tot_t += t; xs.append(dcum); ys.append(soc)
            legs.append(dict(from_node=a, to_node=b, km=d, kwh=e, min=t, soc_arrival=soc))
            if i < len(sol):
                k, l = sol[i]; nxt = wait_pos[i + 2]
                tgt = min(self.CAP, max(self.LEVELS[l] * self.CAP, self.E[b][nxt] + self.RESERVE))
                w = self.wait[k]; c = charge_minutes(soc, tgt, self.power[k], self.cfg) if tgt > soc + 1e-9 else 0.0
                stops.append(dict(station=k, node=b, power_kw=self.power[k], ports_up=int(self.ports_up[k]),
                                  ports_total=int(self.st["ports"][k]), arrive_soc=soc, charge_to=max(soc, tgt),
                                  wait_min=w if c > 0 else 0.0, charge_min=c))
                if c > 0: tot_wait += w; tot_chg += c; tot_t += w + c; soc = tgt; xs.append(dcum); ys.append(soc)
        return dict(waypoints=wait_pos, legs=legs, stops=stops, soc_x=xs, soc_y=ys, total_km=dcum, energy_kwh=tot_e,
                    drive_min=tot_t - tot_wait - tot_chg, wait_min=tot_wait, charge_min=tot_chg, total_min=tot_t)


# ---------------- quality indicators ----------------
def hypervolume(front, ref=(1.1, 1.1)):
    pts = sorted(p for p in np.asarray(front).tolist() if p[0] < ref[0] and p[1] < ref[1])
    hv, prev2 = 0.0, ref[1]
    for a, b in pts:
        if b < prev2: hv += (ref[0] - a) * (prev2 - b); prev2 = b
    return hv


def front_array(front):
    return np.array([[a, b] for a, b, *_ in front]) if len(front) else np.zeros((0, 2))


def knee_point(front):
    F = front_array(front)
    if len(F) <= 1: return 0
    Z = (F - F.min(0)) / np.maximum(F.max(0) - F.min(0), 1e-9)
    return int(np.argmin(np.hypot(Z[:, 0], Z[:, 1])))


def nondominated(F):
    F = np.asarray(F); keep = []
    for i, p in enumerate(F):
        if not any((q <= p).all() and (q < p).any() for j, q in enumerate(F) if j != i): keep.append(i)
    return F[sorted(set(keep), key=lambda i: F[i][0])]


# ---------------- reactive baselines + exact reference ----------------
def greedy_baseline(P, mode):
    D, E = P.D, P.E; pos, soc, sol, used = P.src, P.soc0, [], set()
    for _ in range(P.cfg.max_stops + 1):
        if soc - E[pos][P.dst] >= P.RESERVE: return tuple(sol)
        reach = [k for k in range(P.K) if k not in used and soc - E[pos][P.node[k]] >= P.RESERVE]
        if not reach: return None
        if mode == "nearest": k = min(reach, key=lambda k: D[pos][P.node[k]])
        elif mode == "detour": k = min(reach, key=lambda k: D[pos][P.node[k]] + D[P.node[k]][P.dst])
        else: k = min(reach, key=lambda k: P.wait[k] + charge_minutes(soc - E[pos][P.node[k]], P.CAP, P.power[k], P.cfg))
        sol.append((k, P.NL - 1)); used.add(k); pos, soc = P.node[k], P.CAP
    return None


def run_greedy(P, mode):
    t0 = time.time(); P.reset(10); s = greedy_baseline(P, mode)
    if s is not None: P.eval_sol(s)
    return P.result(time.time() - t0)


def run_exact(P):
    """Exhaustive enumeration (feasible only for small K / max_stops)."""
    t0 = time.time(); arch = []

    def add(sol, f1, f2):
        for a, b, _ in arch:
            if a <= f1 + 1e-9 and b <= f2 + 1e-9: return
        arch[:] = [t for t in arch if not (f1 <= t[0] + 1e-9 and f2 <= t[1] + 1e-9)]; arch.append((f1, f2, sol))
    n_ev = 0
    for m in range(P.cfg.max_stops + 1):
        for ks in itertools.combinations(range(P.K), m):
            order = sorted(ks, key=lambda k: P.progress[k])
            for ls in itertools.product(range(P.NL), repeat=m):
                sol = tuple(zip(order, ls)); f1, f2, v, _ = P.eval_raw(sol); n_ev += 1
                if v <= 1e-9: add(sol, f1, f2)
    return dict(front=sorted(arch, key=lambda t: t[0]), history=[], evals=n_ev, time=time.time() - t0)
