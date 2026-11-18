"""Tabu Search (discrete refinement) on the stop-set neighbourhood."""
import numpy as np


def tabu_search(P, start, w, eval_limit, rng, tenure=4, nbr_sample=30):
    """Weighted-sum TS: add / drop / swap / change-level moves; tabu attribute = station index;
    aspiration = new best scalar cost."""
    f1, f2, _ = P.eval_sol(start); cur = start
    best_sol, best_c = start, P.scal(f1, f2, w); best_f = (f1, f2)
    tabu, it = {}, 0
    while P.n_evals < eval_limit:
        it += 1; nb = P.neighbors(cur)
        if not nb: break
        if len(nb) > nbr_sample: nb = [nb[i] for i in rng.choice(len(nb), nbr_sample, replace=False)]
        cand, cc, ct = None, float("inf"), ()
        for touched, s in nb:
            if P.n_evals >= eval_limit: break
            a, b, _ = P.eval_sol(s); c = P.scal(a, b, w)
            if any(tabu.get(k, 0) > it for k in touched) and not (c < best_c - 1e-9): continue
            if c < cc: cand, cc, ct, cf = s, c, touched, (a, b)
        if cand is None: tabu.clear(); continue
        cur = cand
        for k in ct: tabu[k] = it + tenure
        if cc < best_c - 1e-12: best_sol, best_c, best_f = cand, cc, cf
    return best_sol, best_f
