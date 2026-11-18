"""Optimizer facade: run ONE selected problem with any method; benchmark; Plotly-ready outputs."""
import time, zlib
import numpy as np
from .ev_problem import (EVProblem, run_exact, run_greedy, hypervolume, front_array, knee_point, nondominated)
from .tabu_search import tabu_search
from .firefly import run_fa_ts


def run_random(P, budget, rng):
    t0 = time.time(); P.reset(budget)
    while P.n_evals < budget: P.eval_sol(P.random_sol(rng))
    return P.result(time.time() - t0)


def run_ts_only(P, budget, rng, run_evals=250):
    t0 = time.time(); P.reset(budget)
    n_runs = max(1, budget // run_evals); ws = np.linspace(0.05, 0.95, n_runs)
    X = P.seed_vectors(max(n_runs, 8), rng); r = 0
    while P.n_evals < budget:
        start = P.decode(X[r % len(X)]) if r < len(X) else P.random_sol(rng)
        tabu_search(P, start, ws[r % n_runs], min(budget, P.n_evals + run_evals), rng); r += 1
    return P.result(time.time() - t0)


def _nds(F):
    n = len(F); dom_by = [[] for _ in range(n)]; cnt = np.zeros(n, int); fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if F[i][0] <= F[j][0] and F[i][1] <= F[j][1] and (F[i][0] < F[j][0] or F[i][1] < F[j][1]): dom_by[i].append(j)
            elif F[j][0] <= F[i][0] and F[j][1] <= F[i][1] and (F[j][0] < F[i][0] or F[j][1] < F[i][1]): cnt[i] += 1
        if cnt[i] == 0: fronts[0].append(i)
    rank = np.zeros(n, int); f = 0
    while fronts[f]:
        nxt = []
        for i in fronts[f]:
            for j in dom_by[i]:
                cnt[j] -= 1
                if cnt[j] == 0: rank[j] = f + 1; nxt.append(j)
        f += 1; fronts.append(nxt)
    return rank, fronts[:-1]


def _crowd(F, idx):
    d = {i: 0.0 for i in idx}
    for m in (0, 1):
        s = sorted(idx, key=lambda i: F[i][m]); d[s[0]] = d[s[-1]] = float("inf")
        rg = max(F[s[-1]][m] - F[s[0]][m], 1e-12)
        for a in range(1, len(s) - 1): d[s[a]] += (F[s[a + 1]][m] - F[s[a - 1]][m]) / rg
    return d


def run_nsga2(P, budget, rng, pop=40):
    t0 = time.time(); P.reset(budget); K2 = 2 * P.K
    X = P.seed_vectors(pop, rng); F = [P.eval_sol(P.decode(x))[:2] for x in X]
    while P.n_evals < budget:
        rank, fronts = _nds(F); cd = {}
        for fr in fronts: cd.update(_crowd(F, fr))

        def tour():
            a, b = rng.integers(0, len(X), 2)
            return a if (rank[a], -cd[a]) < (rank[b], -cd[b]) else b
        kids = []
        while len(kids) < pop:
            p1, p2 = X[tour()], X[tour()]; c = p1.copy()
            if rng.random() < 0.9: c = p1 + rng.uniform(-0.25, 1.25, K2) * (p2 - p1)
            mut = rng.random(K2) < 1.5 / K2; c[mut] += rng.normal(0, 0.2, mut.sum()); kids.append(np.clip(c, 0, 1))
        Fk = []
        for c in kids:
            if P.n_evals >= budget: break
            Fk.append(P.eval_sol(P.decode(c))[:2])
        kids = kids[:len(Fk)]
        Z = np.vstack([X, np.array(kids)]) if kids else X; FZ = F + Fk
        rank, fronts = _nds(FZ); sel = []
        for fr in fronts:
            if len(sel) + len(fr) <= pop: sel += fr
            else:
                cdz = _crowd(FZ, fr); sel += sorted(fr, key=lambda i: -cdz[i])[:pop - len(sel)]; break
        X = Z[sel]; F = [FZ[i] for i in sel]
    return P.result(time.time() - t0)


METHODS = {
    "FA-TS (proposed)": lambda P, B, r: run_fa_ts(P, B, r),
    "FA only": lambda P, B, r: run_fa_ts(P, B, r, ts_frac=0.0),
    "TS only": run_ts_only,
    "NSGA-II": run_nsga2,
    "Random search": run_random,
}
GREEDY = {"SP + Nearest station": "nearest", "SP + Min-detour station": "detour", "SP + Least-delay station": "delay"}


def solve(P: EVProblem, method: str, budget: int, seed: int = 0):
    """Run exactly one method on the selected problem."""
    if method in METHODS:
        rng = np.random.default_rng(seed * 1000 + zlib.crc32(method.encode()) % 997)
        return METHODS[method](P, int(budget), rng)
    if method in GREEDY: return run_greedy(P, GREEDY[method])
    raise ValueError(method)


# ---------------- Plotly-compatible outputs (plain lists / dicts) ----------------
def front_trace(res, name):
    F = front_array(res["front"])
    return dict(name=name, x=F[:, 0].tolist(), y=F[:, 1].tolist(), mode="lines+markers", line=dict(shape="hv"))


def solution_payload(P, res):
    """Everything the UI needs for results: front, recommended (knee) index, per-solution explanations."""
    fr = res["front"]
    return dict(front=dict(distance_km=[f[0] for f in fr], delay_min=[f[1] for f in fr]),
                solutions=[f[2] for f in fr], knee=knee_point(fr) if fr else None,
                details=[P.describe(f[2]) for f in fr], evals=res["evals"], time_s=res["time"])


def hv_ratio_against(F, ref_F):
    ref_F = np.asarray(ref_F); ideal, nadir = ref_F.min(0), ref_F.max(0)
    nrm = lambda A: (np.asarray(A) - ideal) / np.maximum(nadir - ideal, 1e-9)
    hx = hypervolume(nrm(ref_F)); F = np.asarray(F)
    return hypervolume(nrm(F)) / hx if len(F) and hx > 0 else 0.0


def curve_against(history, ref_F):
    ref_F = np.asarray(ref_F); out = []
    for n_ev, pts in history:
        out.append((n_ev, hv_ratio_against(pts, ref_F) if len(pts) else 0.0))
    return out


def benchmark(P, methods, budget, n_seeds, seed=0, exact=False, P_fast=None, progress=None):
    """Compare methods on ONE selected problem (equal budget, several seeds)."""
    runs, rec = {}, []
    total = len(methods) * n_seeds; done = 0
    for m in methods:
        runs[m] = []
        for s in range(n_seeds if m in METHODS else 1):
            runs[m].append(solve(P, m, budget, seed + s)); done += 1
            if progress: progress(done / total, f"{m} – run {s + 1}")
    if P_fast is not None and "Fastest-path + Nearest" in methods:
        runs["Fastest-path + Nearest"] = [run_greedy(P_fast, "nearest")]
    exact_res = run_exact(P) if exact else None
    pool = [[f[0], f[1]] for rs in runs.values() for r in rs for f in r["front"]]
    ref = front_array(exact_res["front"]) if exact_res else (nondominated(pool) if pool else np.zeros((0, 2)))
    for m, rs in runs.items():
        for s, r in enumerate(rs):
            F = front_array(r["front"])
            rec.append(dict(method=m, seed=s, hv_ratio=hv_ratio_against(F, ref) if len(ref) else 0.0, front_size=len(F),
                            best_dist_km=F[:, 0].min() if len(F) else np.nan, best_delay_min=F[:, 1].min() if len(F) else np.nan,
                            time_s=r["time"], solved=len(F) > 0))
    curves = {m: [curve_against(r["history"], ref) for r in rs if r["history"]] for m, rs in runs.items() if m in METHODS}
    return dict(records=rec, runs=runs, ref_front=ref, exact=bool(exact_res), curves=curves)
