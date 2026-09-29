# -*- coding: utf-8 -*-
"""
hsyk_constructability.py
========================
Large-N constructability study for "Deterministic Hypergraph SYK" (S. Prakash,
arXiv:2609.19255). No exact diagonalization: only the hypergraph is built and
Axioms 1-3 (and the finite-size symmetry filters) are tested.

What is computed, per (N, d) and instance
-----------------------------------------
Generation   q-uniform, d-regular, LINEAR hypergraph. Configuration model
             (random matching of N*d vertex stubs into N*d/q hyperedges of q
             slots), followed by a repair phase and a mixing phase built from
             one move: the incidence switch (v1 in e1) <-> (v2 in e2). Repair
             accepts a switch only if the two inserted incidences create no
             repeated vertex and no doubly covered pair, and it only moves
             incidences that currently take part in such a violation, so the
             number of violations strictly decreases. Mixing then attempts
             switches between uniformly random slots, accepted iff the result is
             still linear: the proposal is symmetric, so the uniform distribution
             on linear hypergraphs with these degrees is stationary.
Axiom 1      Verified independently (vectorized): no repeated vertex in a
             hyperedge, no vertex pair in two hyperedges, all degrees equal d.
Axiom 3      kappa_1 = largest non-uniform eigenvalue of K = A/(d(q-1)), A the
             site adjacency, by Lanczos on K restricted to the complement of the
             uniform vector; kappa_min = smallest eigenvalue. Reference values:
               kappa_Ram^+ = (q-2 + 2 sqrt((d-1)(q-1)))/(d(q-1))   (Alon-Boppana /
                                                                    Ramanujan edge)
               kappa_Ram^- = (q-2 - 2 sqrt((d-1)(q-1)))/(d(q-1)),  and the exact
               lower bound -1/(q-1) (the paper's lower spectral bound).
Axiom 2      Cycle rank (first Betti number) of radius-R balls of the Levi graph
             around randomly sampled hyperedges, R = 1, 2, ..., as long as the
             ball of a tree would stay below BALL_MAX_FRACTION of the Levi graph.
             Hyperedge-rooted balls suffice: a site ball B(v, R) is a connected
             subgraph of B(e, R+1) for any e containing v, and the cycle rank of
             a connected subgraph never exceeds that of the connected graph.
             The paper's birthday estimate says cycles appear when the tree-ball
             size n_R satisfies n_R^2 ~ N; the output reports x_R = n_R^2/(N+|E|).
Filters      (i) rank_F2(I) = N-1 and (ii) no parity-preserving monomial
             anticommuting with H, by bit-packed Gauss-Jordan elimination over
             F2 of [I^T | 1] (for N <= F2_MAX_N; cost ~ N^2 |E| / 64).
             (iii) Trivial automorphism group of the hypergraph, by a
             certificate: colour refinement (1-WL) on the Levi graph, seeded by
             the automorphism-invariant count of non-trivial site-graph
             triangles. If the refined partition is discrete, every automorphism
             fixes every vertex, so Aut(H) = {id}. If it is not discrete, the
             result is "not certified" (which does NOT imply a symmetry).
             Note: the ED script filters on Aut(line graph) = {id}, which
             implies Aut(H) = {id}; here Aut(H) is certified directly.

Outputs (in OUT_DIR)
--------------------
  constructability_instances.csv   one row per instance
  constructability_balls.csv       one row per (instance, R): cycle-rank statistics
  filter_pass_rates.csv            fraction of instances passing each filter, per (N, d)
  constructability_summary.json    all of the above plus the configuration
  fig_constructability.pdf         (a) kappa_1/Ramanujan edge vs N, (b) kappa_1, kappa_min vs d,
                                   (c) R = 3 cycle rank vs parameter-free prediction,
                                   (d) mean cycle rank vs x_R = n_R^2/(N+|E|)

Requirements: numpy, scipy, matplotlib. Runtime on one CPU core: QUICK_TEST a
few minutes; the default grid roughly 30-60 minutes (the N = 10^5, d = 20
instances take ~1-2 min each, dominated by the pure-Python switch loops).
"""

import os
import csv
import json
import math
import time
import random

import numpy as np
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt

# =====================================================================
# CONFIG
# =====================================================================

QUICK_TEST = False

Q = 4
N_VALUES = [1000, 3000, 10000, 30000, 100000]
D_VALUES = [4, 6, 8, 12, 16, 20]
N_INSTANCES = 3
SEED = 2026

MIX_SWEEPS = 1.0            # mixing attempts = MIX_SWEEPS * (number of slots)/2; 0 to skip
N_BALL_ROOTS = 300          # hyperedges sampled as ball centres (Axiom 2)
R_MAX = 10                  # largest Levi radius considered
BALL_MAX_FRACTION = 0.05    # skip radii whose tree ball exceeds this fraction of the Levi graph
F2_MAX_N = 3000             # F2 filters only up to this N
AUT_MAX_N = 100000          # automorphism certificate only up to this N
FILTER_RATE_N = [200, 500, 1000]   # filter pass-rate study: many small instances, filters only
FILTER_RATE_SAMPLES = 40           # instances per (N, d); 0 to skip
EIG_TOL = 1e-7              # Lanczos stopping: change of extreme Ritz values over 100 steps

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.path.join(_HERE, "..", "data", "constructability")
REPLOT_ONLY = False         # True: redraw the figure from an existing constructability_summary.json
LOAD_ONLY = False           # True: only define the functions (for checks / interactive use)

if QUICK_TEST:
    N_VALUES = [1000, 3000]
    D_VALUES = [4, 8]
    N_INSTANCES = 2
    N_BALL_ROOTS = 100
    FILTER_RATE_N = [200, 500]
    FILTER_RATE_SAMPLES = 8
    OUT_DIR = os.path.join(_HERE, "..", "data", "constructability_quicktest")


# =====================================================================
# 1. GENERATION: configuration model + incidence switches
# =====================================================================

def configuration_model(N, d, q, rng):
    """Random matching of N*d vertex stubs into N*d/q hyperedges of q slots."""
    stubs = np.repeat(np.arange(N, dtype=np.int64), d)
    rng.shuffle(stubs)
    return stubs.reshape(-1, q)


def violating_edges(E, N):
    """Indices of hyperedges with a repeated vertex or a vertex pair covered twice (vectorized)."""
    Es = np.sort(E, axis=1)
    repeated = (np.diff(Es, axis=1) == 0).any(axis=1)
    a, b = np.triu_indices(E.shape[1], 1)
    keys = Es[:, a] * N + Es[:, b]
    _, inv, cnt = np.unique(keys.ravel(), return_inverse=True, return_counts=True)
    duplicated = (cnt[inv.ravel()] > 1).reshape(keys.shape).any(axis=1)
    return np.flatnonzero(repeated | duplicated)


def is_linear_regular(E, N, d):
    """Axiom 1 (plus d-regularity), checked independently of the generator."""
    return (violating_edges(E, N).size == 0
            and np.array_equal(np.bincount(E.ravel(), minlength=N), np.full(N, d)))


class _Switcher:
    """Mutable hypergraph: edges[e] = list of q vertices (slots), inc[v] = list of incident edges
    (a multiset while the configuration still has repeated vertices)."""

    def __init__(self, E, N):
        self.q = E.shape[1]
        self.edges = E.tolist()
        self.inc = [[] for _ in range(N)]
        for e, row in enumerate(self.edges):
            for v in row:
                self.inc[v].append(e)

    def swap(self, e1, i1, e2, i2):
        """(v1 at slot i1 of e1) <-> (v2 at slot i2 of e2). Applying it twice is the identity."""
        r1, r2 = self.edges[e1], self.edges[e2]
        v1, v2 = r1[i1], r2[i2]
        r1[i1], r2[i2] = v2, v1
        l1, l2 = self.inc[v1], self.inc[v2]
        l1[l1.index(e1)] = e2
        l2[l2.index(e2)] = e1

    def vertex_clean(self, e, v):
        """No vertex of edge e other than v shares a further hyperedge with v."""
        nb = set()
        for f in set(self.inc[v]):
            if f != e:
                nb.update(self.edges[f])
        nb.discard(v)
        return nb.isdisjoint(self.edges[e])

    def slot_clean(self, e, i):
        row = self.edges[e]
        v = row[i]
        return row.count(v) == 1 and self.vertex_clean(e, v)

    def bad_slots(self, e):
        row = self.edges[e]
        return [i for i, v in enumerate(row) if row.count(v) > 1 or not self.vertex_clean(e, v)]


def _repair(sw, bad, rnd, max_steps):
    """Removes all violations by switches that strictly decrease their number."""
    q, n_slots = sw.q, len(sw.edges) * sw.q
    steps = 0
    while bad:
        if steps >= max_steps:
            return False, steps
        steps += 1
        e1 = rnd.choice(tuple(bad))
        slots = sw.bad_slots(e1)
        if not slots:
            bad.discard(e1)
            continue
        i1 = rnd.choice(slots)
        e2, i2 = divmod(rnd.randrange(n_slots), q)
        if e2 == e1 or sw.edges[e1][i1] == sw.edges[e2][i2]:
            continue
        v1, v2 = sw.edges[e1][i1], sw.edges[e2][i2]
        sw.swap(e1, i1, e2, i2)
        if sw.slot_clean(e1, i1) and sw.slot_clean(e2, i2):
            for f in {e1, e2, *sw.inc[v1], *sw.inc[v2]}:
                if sw.bad_slots(f):
                    bad.add(f)
                else:
                    bad.discard(f)
        else:
            sw.swap(e1, i1, e2, i2)
    return True, steps


def _mix(sw, rnd, n_attempts):
    """Symmetric switch chain restricted to linear hypergraphs; returns the acceptance count."""
    q, n_slots, edges = sw.q, len(sw.edges) * sw.q, sw.edges
    accepted = 0
    for _ in range(n_attempts):
        e1, i1 = divmod(rnd.randrange(n_slots), q)
        e2, i2 = divmod(rnd.randrange(n_slots), q)
        if e1 == e2:
            continue
        v1, v2 = edges[e1][i1], edges[e2][i2]
        if v1 in edges[e2] or v2 in edges[e1]:
            continue
        sw.swap(e1, i1, e2, i2)
        if sw.vertex_clean(e1, v2) and sw.vertex_clean(e2, v1):
            accepted += 1
        else:
            sw.swap(e1, i1, e2, i2)
    return accepted


def generate_linear_hypergraph(N, d, q, seed, mix_sweeps=MIX_SWEEPS, max_restarts=20):
    """Random linear q-uniform d-regular hypergraph on N sites; returns (E, stats), E of shape (N d/q, q)."""
    if (N * d) % q:
        raise ValueError("q must divide N d.")
    rng, rnd = np.random.default_rng(seed), random.Random(seed)
    t0 = time.time()
    for restart in range(max_restarts):
        E0 = configuration_model(N, d, q, rng)
        bad0 = violating_edges(E0, N)
        sw = _Switcher(E0, N)
        ok, steps = _repair(sw, set(bad0.tolist()), rnd, max_steps=200 * (bad0.size + 10) + 10000)
        if ok:
            break
    else:
        raise RuntimeError(f"Repair failed {max_restarts} times for N={N}, d={d}.")
    t_repair = time.time() - t0
    n_attempts = int(mix_sweeps * len(sw.edges) * q / 2)
    accepted = _mix(sw, rnd, n_attempts) if n_attempts else 0
    E = np.array(sw.edges, dtype=np.int64)
    if not is_linear_regular(E, N, d):
        raise AssertionError("Generator produced a non-linear or non-regular hypergraph.")
    return E, {"restarts": restart, "initial_violating_edges": int(bad0.size), "repair_steps": steps,
               "t_repair": t_repair, "mix_attempts": n_attempts,
               "mix_acceptance": accepted / n_attempts if n_attempts else None,
               "t_generate": time.time() - t0}


def incidence_array(E, N, d):
    """inc[v] = the d hyperedges containing v (shape (N, d))."""
    order = np.argsort(E.ravel(), kind="stable")
    return (order // E.shape[1]).reshape(N, d)


# =====================================================================
# 2. AXIOM 3: spectrum of the site-adjacency operator K
# =====================================================================

def site_adjacency(E, N):
    """0/1 site adjacency (for a linear hypergraph): A_ik = 1 iff i != k share a hyperedge."""
    i, j = np.nonzero(~np.eye(E.shape[1], dtype=bool))
    rows, cols = E[:, i].ravel(), E[:, j].ravel()
    return sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(N, N))


def ramanujan_edges(d, q):
    """Normalized Alon-Boppana / Ramanujan edges (q-2 +- 2 sqrt((d-1)(q-1)))/(d(q-1))."""
    c = 2 * math.sqrt((d - 1) * (q - 1))
    return (q - 2 + c) / (d * (q - 1)), (q - 2 - c) / (d * (q - 1))


def kappa_extremes(A, d, q, tol=EIG_TOL, seed=0, m_max=6000, check_every=100):
    """
    (kappa_1, kappa_min, lanczos_steps, last_change) for K = A/(d(q-1)).

    One plain Lanczos run on P K P, P the projection off the uniform vector (exact deflation,
    since K commutes with P). Its spectrum is {kappa_alpha, alpha != 0} together with a 0, so
    the largest and smallest Ritz values converge to kappa_1 and kappa_min. At large N both
    lie at the edges of a dense bulk (gaps ~ N^(-2/3)), where implicitly restarted methods
    with a small Krylov space stall. Plain Lanczos without reorthogonalization gives
    accurate extreme Ritz values (Paige) in O(N) memory. The Ritz values approach the
    extremes monotonically from inside; iteration stops when neither changes by more than
    tol over check_every further steps.
    """
    from scipy.linalg import eigvalsh_tridiagonal
    N = A.shape[0]
    c = 1.0 / (d * (q - 1))
    x = np.random.default_rng(seed).standard_normal(N)
    x -= x.mean()
    x /= np.linalg.norm(x)
    x_prev, beta = np.zeros(N), 0.0
    alphas, betas, last, change = [], [], None, np.inf
    for j in range(1, min(m_max, N - 1) + 1):
        w = c * (A @ (x - x.mean())) - beta * x_prev
        alpha = x @ w
        w -= alpha * x
        beta = np.linalg.norm(w)
        alphas.append(alpha); betas.append(beta)
        if beta < 1e-12:
            break
        x_prev, x = x, w / beta
        if j % check_every == 0:
            a, b = np.array(alphas), np.array(betas[:-1])
            lo = eigvalsh_tridiagonal(a, b, select="i", select_range=(0, 0))[0]
            hi = eigvalsh_tridiagonal(a, b, select="i", select_range=(j - 1, j - 1))[0]
            if last is not None:
                change = max(abs(hi - last[1]), abs(lo - last[0]))
                if change < tol:
                    break
            last = (lo, hi)
    a, b = np.array(alphas), np.array(betas[:-1])
    ev = eigvalsh_tridiagonal(a, b, select="i", select_range=(0, len(a) - 1)) if len(a) < 50 else \
        np.array([eigvalsh_tridiagonal(a, b, select="i", select_range=(0, 0))[0],
                  eigvalsh_tridiagonal(a, b, select="i", select_range=(len(a) - 1, len(a) - 1))[0]])
    return float(ev[-1]), float(ev[0]), len(alphas), float(change)


# =====================================================================
# 3. AXIOM 2: cycle rank of Levi-graph balls around hyperedges
# =====================================================================

def tree_ball_sizes(d, q, R_max):
    """Number of Levi-graph nodes within distance R of a hyperedge if the ball were a tree."""
    sizes, layer, total = [1], 1, 1
    for R in range(1, R_max + 1):
        layer *= q if R == 1 else ((q - 1) if R % 2 else (d - 1))
        total += layer
        sizes.append(total)
    return np.array(sizes)


def levi_ball_cycle_ranks(E, inc, roots, R_max):
    """
    ranks[k, R] = cycle rank of the radius-R Levi ball around hyperedge roots[k]
    (R = 0..R_max). The ball is connected, so rank = #incidences - #nodes + 1, with the
    incidences counted inside the ball. Visited marks are stamped with the root index,
    so no array is reset between roots.
    """
    N, M = inc.shape[0], E.shape[0]
    site_mark = np.full(N, -1, np.int64)
    edge_mark = np.full(M, -1, np.int64)
    ranks = np.zeros((len(roots), R_max + 1), np.int64)
    for k, e0 in enumerate(roots):
        edge_mark[e0] = k
        frontier, nodes, links = np.array([e0]), 1, 0
        for R in range(1, R_max + 1):
            if R % 2:      # hyperedges at distance R-1 -> new sites at distance R
                cand = np.unique(E[frontier].ravel())
                new = cand[site_mark[cand] != k]
                site_mark[new] = k
                links += int((edge_mark[inc[new]] == k).sum())
            else:          # sites at distance R-1 -> new hyperedges at distance R
                cand = np.unique(inc[frontier].ravel())
                new = cand[edge_mark[cand] != k]
                edge_mark[new] = k
                links += int((site_mark[E[new]] == k).sum())
            nodes += new.size
            frontier = new
            ranks[k, R] = links - nodes + 1
    return ranks


# =====================================================================
# 4. FILTERS: F2 rank and chiral monomials; automorphism certificate
# =====================================================================

def f2_rank_and_chiral(E, N):
    """
    Bit-packed Gauss-Jordan elimination over F2 of the augmented system [I^T | 1] (rows = hyperedges).
    Returns (rank_F2(I), chiral) with chiral in {None, 'odd', 'even'}: None if I^T x = 1 has no
    solution (no monomial anticommutes with H); otherwise the parity of |x|, which is well defined
    when rank = N - 1 (the kernel of I^T is then spanned by the all-ones vector, of even weight).
    """
    M, q = E.shape
    W = (N + 1 + 63) // 64
    A = np.zeros((M, W), np.uint64)
    rows, cols = np.repeat(np.arange(M), q), E.ravel()
    np.bitwise_or.at(A, (rows, cols // 64), np.left_shift(np.uint64(1), (cols % 64).astype(np.uint64)))
    A[:, N // 64] |= np.uint64(1) << np.uint64(N % 64)
    one = np.uint64(1)
    rank = 0
    for c in range(N):
        w, b = c // 64, np.uint64(c % 64)
        col = ((A[:, w] >> b) & one).astype(bool)
        piv = np.flatnonzero(col[rank:])
        if piv.size == 0:
            continue
        p = rank + piv[0]
        if p != rank:
            A[[rank, p]] = A[[p, rank]]
            col[[rank, p]] = col[[p, rank]]
        col[rank] = False
        A[col] ^= A[rank]
        rank += 1
        if rank == M:
            break
    rhs = ((A[:, N // 64] >> np.uint64(N % 64)) & one).astype(bool)
    if rhs[rank:].any():
        return rank, None
    return rank, ("odd" if int(rhs[:rank].sum()) % 2 else "even")


def nontrivial_site_triangles(A, d, q, chunk_nnz=2 * 10 ** 7):
    """Per site: site-graph triangles not contained in one hyperedge (6-cycles of the Levi graph).
    diag(A^3)/2 minus the d C(q-1, 2) triangles inside the hyperedge cliques; computed in row chunks."""
    N = A.shape[0]
    rows_per = max(1, int(chunk_nnz // (d * (q - 1)) ** 2))
    t = np.empty(N)
    for s in range(0, N, rows_per):
        Ar = A[s:s + rows_per]
        t[s:s + rows_per] = np.asarray((Ar @ A).multiply(Ar).sum(axis=1)).ravel() / 2
    return np.rint(t - d * math.comb(q - 1, 2)).astype(np.int64)


def _relabel(rows):
    return np.unique(rows, axis=0, return_inverse=True)[1].ravel()


def asymmetry_certificate(E, inc, site_seed, max_rounds=500):
    """
    Colour refinement on the Levi graph from an automorphism-invariant site colouring. The colours
    at every round are functions of isomorphism-invariant data, so automorphisms preserve them.
    Returns (certified, rounds): certified = True iff the final partition is discrete, which proves
    Aut(H) = {id}. False means only that refinement did not decide.
    """
    N, M = inc.shape[0], E.shape[0]
    sc, ec = _relabel(site_seed[:, None]), np.zeros(M, np.int64)
    prev = (-1, -1)
    for r in range(1, max_rounds + 1):
        ec = _relabel(np.column_stack([ec, np.sort(sc[E], axis=1)]))
        sc = _relabel(np.column_stack([sc, np.sort(ec[inc], axis=1)]))
        counts = (int(sc.max()) + 1, int(ec.max()) + 1)
        if counts == (N, M):
            return True, r
        if counts == prev:
            return False, r
        prev = counts
    return False, max_rounds


# =====================================================================
# 5. DRIVER
# =====================================================================

def instance_seed(N, d, m):
    return int(np.random.SeedSequence([SEED, N, d, m]).generate_state(1)[0])


def study_instance(N, d, q, m):
    seed = instance_seed(N, d, m)
    E, gen = generate_linear_hypergraph(N, d, q, seed)
    M = E.shape[0]
    inc = incidence_array(E, N, d)
    rec = {"N": N, "d": d, "q": q, "instance": m, "seed": seed, "num_hyperedges": M, **gen}

    t0 = time.time()
    A = site_adjacency(E, N)
    k1, kmin, steps, change = kappa_extremes(A, d, q)
    kr_plus, kr_minus = ramanujan_edges(d, q)
    rec.update({"kappa_1": k1, "kappa_min": kmin, "gap": 1 - k1,
                "lanczos_steps": steps, "lanczos_last_change": change,
                "kappa_ram_plus": kr_plus, "kappa_ram_minus": kr_minus,
                "kappa_lower_bound": -1 / (q - 1), "kappa1_below_otoc_threshold": bool(k1 < 1 / (q - 1)),
                "t_eig": time.time() - t0})

    t0 = time.time()
    sizes = tree_ball_sizes(d, q, R_MAX)
    R_use = int(np.max(np.flatnonzero(sizes <= BALL_MAX_FRACTION * (N + M)), initial=0))
    balls = []
    if R_use >= 1:
        roots = np.random.default_rng(seed + 1).choice(M, size=min(N_BALL_ROOTS, M), replace=False)
        ranks = levi_ball_cycle_ranks(E, inc, roots, R_use)
        for R in range(1, R_use + 1):
            x = ranks[:, R]
            balls.append({"N": N, "d": d, "instance": m, "R": R, "tree_ball_nodes": int(sizes[R]),
                          "x_R": float(sizes[R] ** 2 / (N + M)), "n_roots": int(x.size),
                          "rank_mean": float(x.mean()), "rank_p95": float(np.percentile(x, 95)),
                          "rank_max": int(x.max()), "tree_like_fraction": float((x == 0).mean())})
    rec["R_max_tested"] = R_use
    rec["t_balls"] = time.time() - t0

    rec["rank_F2"] = rec["full_rank"] = rec["chiral"] = rec["passes_F2_filters"] = None
    if N <= F2_MAX_N:
        t0 = time.time()
        rank, chiral = f2_rank_and_chiral(E, N)
        rec.update({"rank_F2": rank, "full_rank": rank == N - 1, "chiral": chiral or "none",
                    "passes_F2_filters": (rank == N - 1) and chiral != "even", "t_F2": time.time() - t0})

    rec["triangles_nontrivial_total"] = rec["aut_certified_trivial"] = rec["refinement_rounds"] = None
    if N <= AUT_MAX_N:
        t0 = time.time()
        tri = nontrivial_site_triangles(A, d, q)
        cert, rounds = asymmetry_certificate(E, inc, tri)
        rec.update({"triangles_nontrivial_total": int(tri.sum() // 3), "aut_certified_trivial": cert,
                    "refinement_rounds": rounds, "t_aut": time.time() - t0})
    return rec, balls


def filter_pass_rates(N_values, d_values, q, n_samples):
    """
    Fraction of random linear hypergraphs passing the finite-size symmetry filters:
    (i) rank_F2(I) = N-1, (ii) no parity-preserving anticommuting monomial, (iii) Aut(H) = {id}
    certified. Also the mean F2 corank N-1-rank (always >= 0: the all-ones vector lies in
    ker(I^T) since q is even). Instances are independent of those of the main grid.
    """
    rows = []
    for d in d_values:
        for N in N_values:
            if (N * d) % q or N - 1 < d * (q - 1):
                continue
            c = {"generated": 0, "generation_failed": 0, "full_rank": 0, "chiral_none": 0,
                 "chiral_odd": 0, "chiral_even": 0, "aut_certified": 0, "pass_all": 0}
            coranks, t0 = [], time.time()
            for m in range(n_samples):
                seed = int(np.random.SeedSequence([SEED, N, d, 10 ** 6 + m]).generate_state(1)[0])
                try:
                    E, _ = generate_linear_hypergraph(N, d, q, seed)
                except RuntimeError:
                    c["generation_failed"] += 1
                    continue
                c["generated"] += 1
                rank, chiral = f2_rank_and_chiral(E, N)
                full = rank == N - 1
                coranks.append(N - 1 - rank)
                c["full_rank"] += full
                if full:
                    c[f"chiral_{chiral or 'none'}"] += 1
                inc = incidence_array(E, N, d)
                cert, _ = asymmetry_certificate(E, inc, nontrivial_site_triangles(site_adjacency(E, N), d, q))
                c["aut_certified"] += cert
                c["pass_all"] += full and chiral != "even" and cert
            n = max(c["generated"], 1)
            row = {"N": N, "d": d, "samples": n_samples, **c,
                   "frac_full_rank": c["full_rank"] / n,
                   "frac_chiral_even_given_full_rank": c["chiral_even"] / max(c["full_rank"], 1),
                   "frac_aut_certified": c["aut_certified"] / n, "frac_pass_all": c["pass_all"] / n,
                   "mean_corank": float(np.mean(coranks)) if coranks else None, "t": time.time() - t0}
            rows.append(row)
            print(f"  filters N={N:5d} d={d:2d}: full rank {row['frac_full_rank']:.2f} "
                  f"(mean corank {row['mean_corank']:.2f}), chiral even | full rank "
                  f"{row['frac_chiral_even_given_full_rank']:.2f}, Aut=id certified "
                  f"{row['frac_aut_certified']:.2f}, pass all {row['frac_pass_all']:.2f} "
                  f"[{c['generated']} generated]")
    return rows


def plot_results(instances, balls, q, path):
    """Four-panel constructibility figure (fig_constructability.pdf)."""
    ram = lambda d, sgn=1: (q - 2 + sgn * 2 * math.sqrt((d - 1) * (q - 1))) / (d * (q - 1))
    ds = sorted({r["d"] for r in instances})
    cols = dict(zip(ds, plt.cm.viridis(np.linspace(0.02, 0.88, len(ds)))))
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4), dpi=200)
    ax = ax.ravel()

    for d in ds:                                               # (a) kappa_1 / Ramanujan edge
        rs = [r for r in instances if r["d"] == d]
        ax[0].plot([r["N"] for r in rs], [r["kappa_1"] / r["kappa_ram_plus"] for r in rs], "o", ms=4,
                   color=cols[d], label=f"$d={d}$")
    ax[0].axhline(1, color="k", lw=0.8)
    ax[0].set_xscale("log"); ax[0].set_xlabel("$N$"); ax[0].set_ylabel(r"$\kappa_1/\kappa_{\rm Ram}^{+}$")
    ax[0].set_title(r"(a) $\kappa_1$ relative to the Ramanujan edge"); ax[0].legend(fontsize=8.5, ncol=3, loc="lower right")

    Nmax = max(r["N"] for r in instances)                      # (b) extremes vs d at largest N
    rs = [r for r in instances if r["N"] == Nmax]
    dd = np.linspace(min(ds), max(ds), 200)
    ax[1].plot(dd, [ram(x) for x in dd], "C0--", lw=1, label=r"Ramanujan edges $\kappa^\pm_{\rm Ram}$")
    ax[1].plot(dd, [ram(x, -1) for x in dd], "C3--", lw=1)
    ax[1].plot([r["d"] for r in rs], [r["kappa_1"] for r in rs], "o", color="C0", label=r"$\kappa_1$")
    ax[1].plot([r["d"] for r in rs], [r["kappa_min"] for r in rs], "s", color="C3", label=r"$\kappa_{\min}$")
    ax[1].axhline(1 / (q - 1), color="k", ls=":", lw=1, label=r"$\pm 1/(q-1)$")
    ax[1].axhline(-1 / (q - 1), color="k", ls=":", lw=1)
    ax[1].set_xlabel("$d$"); ax[1].set_ylabel(r"eigenvalue of $K$"); ax[1].set_ylim(-0.42, 0.75)
    ax[1].set_title(rf"(b) extreme non-uniform eigenvalues, $N={Nmax}$"); ax[1].legend(fontsize=8.5, loc="upper right")

    pred3, obs3, cc = [], [], []                               # (c) R = 3 vs parameter-free prediction
    for d in ds:
        for N in sorted({r["N"] for r in instances}):
            v = [b["rank_mean"] for b in balls if b["R"] == 3 and b["d"] == d and b["N"] == N]
            if v:
                pred3.append(q * ((d - 1) * (q - 1)) ** 3 / (2 * d * N)); obs3.append(np.mean(v)); cc.append(cols[d])
    if pred3:
        ax[2].scatter(pred3, obs3, c=cc, s=22, zorder=3)
        xx = np.logspace(np.log10(min(pred3)) - 0.2, np.log10(max(pred3)) + 0.2, 50)
        ax[2].plot(xx, xx, "k--", lw=1, label="$y=x$ (no free parameter)")
        ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].legend(fontsize=8.5, loc="upper left")
    ax[2].set_xlabel(r"prediction $3C_6/|\mathcal{E}| = q\,[(d-1)(q-1)]^3/(2dN)$")
    ax[2].set_ylabel(r"mean cycle rank of $B(e,3)$"); ax[2].set_title("(c) radius-3 Levi balls: 6-cycles through $e$")

    Rs = sorted({b["R"] for b in balls if b["R"] >= 3})      # (d) all radii vs birthday parameter
    rc = dict(zip(Rs, plt.cm.plasma(np.linspace(0.05, 0.85, max(1, len(Rs))))))
    for R in Rs:
        bs = [b for b in balls if b["R"] == R and b["rank_mean"] > 0]
        ax[3].loglog([b["x_R"] for b in bs], [b["rank_mean"] for b in bs], "o", ms=3.5, color=rc[R], label=f"$R={R}$")
    xs = np.array([b["x_R"] for b in balls if b["R"] >= 3 and b["rank_mean"] > 0])
    if xs.size:
        xx = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 50)
        ax[3].loglog(xx, 0.4 * xx, "k--", lw=1, label=r"$0.4\,x_R$")
        ax[3].legend(fontsize=8.5, loc="upper left", ncol=2)
    ax[3].axvline(1, color="gray", lw=0.8, ls=":")
    ax[3].set_xlabel(r"birthday parameter $x_R = n_R^2/(N+|\mathcal{E}|)$"); ax[3].set_ylabel(r"mean cycle rank of $B(e,R)$")
    ax[3].set_title("(d) all radii")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close(fig)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    instances, balls = [], []
    for d in D_VALUES:
        for N in N_VALUES:
            if (N * d) % Q or N - 1 < d * (Q - 1):
                print(f"skipping N={N}, d={d} (q does not divide N d, or site degree too large)")
                continue
            for m in range(N_INSTANCES):
                t0 = time.time()
                rec, b = study_instance(N, d, Q, m)
                rec["t_total"] = time.time() - t0
                instances.append(rec)
                balls.extend(b)
                ranks = ", ".join(f"R{x['R']}:{x['rank_mean']:.2f}/{x['rank_max']}" for x in b if x["R"] >= 3)
                f2 = "" if rec["rank_F2"] is None else \
                    f", rank_F2={rec['rank_F2']}/{N - 1}, chiral={rec['chiral']}"
                aut = "" if rec["aut_certified_trivial"] is None else \
                    f", Aut=id certified: {rec['aut_certified_trivial']}"
                print(f"N={N:6d} d={d:2d} i{m}: kappa_1={rec['kappa_1']:.4f} "
                      f"(Ramanujan {rec['kappa_ram_plus']:.4f}), kappa_min={rec['kappa_min']:+.4f}; "
                      f"cycle rank mean/max [{ranks}]{f2}{aut}; {rec['t_total']:.0f} s")
                write_csv(os.path.join(OUT_DIR, "constructability_instances.csv"), instances)
                write_csv(os.path.join(OUT_DIR, "constructability_balls.csv"), balls)

    rates = []
    if FILTER_RATE_SAMPLES > 0:
        print("\nFilter pass rates")
        rates = filter_pass_rates(FILTER_RATE_N, D_VALUES, Q, FILTER_RATE_SAMPLES)
        write_csv(os.path.join(OUT_DIR, "filter_pass_rates.csv"), rates)

    keys = ["QUICK_TEST", "Q", "N_VALUES", "D_VALUES", "N_INSTANCES", "SEED", "MIX_SWEEPS", "N_BALL_ROOTS",
            "R_MAX", "BALL_MAX_FRACTION", "F2_MAX_N", "AUT_MAX_N", "FILTER_RATE_N", "FILTER_RATE_SAMPLES",
            "EIG_TOL", "OUT_DIR"]
    config = {k: globals()[k] for k in keys}
    with open(os.path.join(OUT_DIR, "constructability_summary.json"), "w") as f:
        json.dump({"config": config, "instances": instances, "balls": balls, "filter_pass_rates": rates},
                  f, indent=2, default=str)
    plot_results(instances, balls, Q, os.path.join(OUT_DIR, "fig_constructability.pdf"))
    print(f"\nOutputs written to '{OUT_DIR}/'.")


def replot():
    with open(os.path.join(OUT_DIR, "constructability_summary.json")) as f:
        data = json.load(f)
    plot_results(data["instances"], data["balls"], Q, os.path.join(OUT_DIR, "fig_constructability.pdf"))
    print(f"Figure redrawn in '{OUT_DIR}/'.")


if __name__ == "__main__":
    import sys
    # python hsyk_constructability.py [replot]. In a notebook cell sys.argv holds the kernel's
    # own flags and is ignored; any other unexpected argument is an error, never a full run.
    args = sys.argv[1:]
    if "ipykernel" in sys.modules and os.path.basename(sys.argv[0] if sys.argv else "") != "hsyk_constructability.py":
        args = []
    usage = "usage: python hsyk_constructability.py [replot]   (replot: redraw the figure from the saved JSON)"
    if args[:1] in (["-h"], ["--help"]):
        print(usage)
        sys.exit(0)
    if args not in ([], ["replot"]):
        sys.exit(f"error: unexpected argument(s) {' '.join(args)!r}\n{usage}")
    if args == ["replot"]:
        REPLOT_ONLY = True
    if not LOAD_ONLY:
        replot() if REPLOT_ONLY else main()
