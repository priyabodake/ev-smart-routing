"""Multi-objective Firefly Algorithm (decomposition-based) and the hybrid FA-TS."""
import time
import numpy as np
from .tabu_search import tabu_search


class FireflyPop:
    def __init__(self, P, rng, pop=24, beta0=1.0, gamma=2.0, alpha0=0.30, decay=0.96):
        self.P, self.rng, self.beta0, self.gamma, self.alpha0, self.decay = P, rng, beta0, gamma, alpha0, decay
        self.X = P.seed_vectors(pop, rng); self.w = np.linspace(0.02, 0.98, pop); self.gen = 0
        self.cur = [self._ev(x) for x in self.X]

    def _ev(self, x):
        sol = self.P.decode(x); a, b, _ = self.P.eval_sol(sol); return sol, a, b

    def run_until(self, limit):
        P, rng = self.P, self.rng; pop, K2 = self.X.shape
        while P.n_evals < limit:
            alpha = self.alpha0 * self.decay ** self.gen; newX = self.X.copy()
            for i in range(pop):
                wi = self.w[i]; ci = P.scal(self.cur[i][1], self.cur[i][2], wi); xi = self.X[i].copy()
                for j in range(pop):
                    if P.scal(self.cur[j][1], self.cur[j][2], wi) < ci - 1e-12:
                        r2 = np.sum((xi - self.X[j]) ** 2) / K2
                        xi = xi + self.beta0 * np.exp(-self.gamma * r2) * (self.X[j] - xi)
                newX[i] = np.clip(xi + alpha * (rng.random(K2) - 0.5), 0, 1)
            for i in rng.permutation(pop):
                if P.n_evals >= limit: break
                new = self._ev(newX[i])
                if P.scal(new[1], new[2], self.w[i]) <= P.scal(self.cur[i][1], self.cur[i][2], self.w[i]):
                    self.X[i], self.cur[i] = newX[i], new
            self.gen += 1

    def inject(self, sol, f, w):
        i = int(round(w * (len(self.w) - 1)))
        self.X[i] = self.P.encode(sol, self.rng); self.cur[i] = (sol, f[0], f[1])


def archive_members(P, m):
    arch = sorted(P.archive, key=lambda t: t[0])
    if not arch: return []
    idx = np.unique(np.linspace(0, len(arch) - 1, min(m, len(arch))).round().astype(int))
    return [(arch[i][2], 0.9 - 0.8 * (i / max(1, len(arch) - 1))) for i in idx]


def run_fa_ts(P, budget, rng, pop=24, n_blocks=8, ts_frac=0.6, ts_members=5, **fa_kw):
    """ts_frac=0 -> FA only; ts_frac=0.6 -> proposed FA-TS (original algorithm, unchanged)."""
    t0 = time.time(); P.reset(budget)
    fa = FireflyPop(P, rng, pop, **fa_kw); block = budget / n_blocks
    for b in range(n_blocks):
        last = (b == n_blocks - 1)
        fa.run_until(budget if (ts_frac == 0 and last) else P.n_evals + (1 - ts_frac) * block)
        if ts_frac == 0: continue
        ts_lim = budget if last else min(budget, P.n_evals + ts_frac * block)
        mem = archive_members(P, ts_members) or [(P.random_sol(rng), 0.5)]
        per = max(1, (ts_lim - P.n_evals) / len(mem))
        for sol, w in mem:
            if P.n_evals >= ts_lim: break
            bs, bf = tabu_search(P, sol, w, min(ts_lim, P.n_evals + per), rng)
            fa.inject(bs, bf, w)
    return P.result(time.time() - t0)
