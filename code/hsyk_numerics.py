# -*- coding: utf-8 -*-
"""
hsyk_numerics.py
================
Numerics for "Deterministic Hypergraph SYK: Melonic Dominance and Maximal
Chaos Without Disorder Averaging" (S. Prakash, arXiv:2609.19255, v2).

One file, no local dependencies. Set the CONFIG block below and run
`python hsyk_numerics.py`. Outputs go to OUT_DIR.

RUN_MODE = "full" produces:

  Instances    hsyk_N{N}_d{d}.json                (hyperedges + diagnostics)
  Tables       table_hypergraph_diagnostics.tex   (|E|, triangles, kappa_1, kappa_max)
               table_two_point_deviation.tex      (G(beta/2) - G_SD(beta/2), instance 0)
               table_level_statistics.tex         (single-instance <r>)
               table_search_statistics.tex        (attempts, valid rate, pool kappa_1)
               table_instances.tex, instances.csv (instance-to-instance ranges; per instance)
               table_controls.tex                 (dense SYK, random-sign and Gaussian-coupling
                                                   controls at N = FIG_N)
  Figures      fig_two_point_spectral_N{FIG_N}.pdf (G(tau) vs SD, P(s), SFF)
               fig_thermodynamics_N{FIG_N}.pdf     (energy and entropy vs SD)
               fig_otoc_modes_N{N}_d{d}.pdf        (mode-resolved OTOC)
  Data         data_N{N}_d{d}.npz, otoc_allpairs_N{N}_d{d}.npz, otoc_modes_N{N}_d{d}.npz,
               otoc_mode_rates_N{N}_d{d}.csv, controls_N{FIG_N}.npz (incl. E/N, S/N),
               summary.json

RUN_MODE = "postprocess" uses the saved data of a completed full run and produces
(without recomputing any hSYK spectra):

  postprocess_instances.csv   per instance: Delta G re-referenced to a converged SD solution,
                              parity-sector spectral symmetries, <r> on E > 0 where the sector
                              spectrum is E -> -E symmetric, z-scores vs regenerated RMT
                              calibration, E/N and S/N at beta (N = FIG_N)
  rmt_calibration.csv         finite-size RMT <r> (same protocol as hSYK)
  controls_thermo_N{FIG_N}.npz, dense_thermo_N{FIG_N}.tex
                              dense-SYK E/N, S/N (the dense realizations are regenerated from
                              the same RNG stream and checked against the saved G(tau))
  postprocess_summary.json

Requirements: numpy, scipy, networkx, matplotlib (see requirements.txt).

Runtime (single CPU core, rough):
  - instance search + G(tau) + level statistics + SFF: minutes per config
    at N <= 22, ~10-20 min at N = 24;
  - controls (dense SYK etc.): ~1 min per sample at N = 24;
  - mode-resolved OTOC: ~20-50 min per instance at N = 24;
  - postprocess: ~10 min (mostly RMT calibration; set RMT_SAMPLES lower to speed up).
Set QUICK_TEST = True first to check that everything runs (~ a few minutes).

Conventions
-----------
  {chi_i, chi_j} = delta_ij;  H = sum_e J_e chi_e,  chi_e = i^{q(q-1)/2} prod_{k in e} chi_k
  (sites ordered increasingly within e),  J_e = J/sqrt(d),  J = 1.
  K = (I I^T - d 1)/(d(q-1)) is the normalized site-adjacency operator;
  kappa_1 = largest non-uniform eigenvalue, kappa_max = max_{alpha != 0} |kappa_alpha|.

Instance selection
------------------
For each (N, d), candidate q-uniform, d-regular, linear (one-overlap)
hypergraphs are generated with a seeded RNG. A candidate is valid if
  (i)  rank_F2(I) = N - 1   (no monomial conserved quantity besides parity),
  (ii) the line graph has trivial automorphism group,
  (iii) (FILTER_CHIRAL = True) no parity-preserving monomial anticommutes with H.
Among the first POOL_SIZE valid candidates, the one with smallest kappa_1 is
selected. N_INSTANCES > 1 repeats the whole selection procedure with
independent random keys. Instance 0 is the representative instance. Nothing is
averaged.

Note: at d = q, a parity-ODD anticommuting monomial always exists (|E| = N
and ker_F2(I) = span(1)), so filter (iii) can only make it parity-odd.
"""

import os
import json
import time
import csv

import numpy as np
import scipy.sparse as sp
from scipy.special import gamma as gamma_fn
from scipy.optimize import brentq
from scipy.sparse.csgraph import shortest_path
import networkx as nx
from networkx.algorithms import isomorphism
import matplotlib
import matplotlib.pyplot as plt

# =====================================================================
# CONFIG
# =====================================================================

RUN_MODE = "full"           # (or pass the mode on the command line)
                            # "full": whole pipeline;  "postprocess": analysis of saved data only;
                            # "otoc_reanalysis": refit and redraw mode-resolved OTOCs from saved
                            # otoc_allpairs_*.npz (no diagonalization);
                            # "none": only define the functions (for checks / interactive use)
QUICK_TEST = False          # True: tiny sizes, checks the whole pipeline in minutes

RUN_CONFIGS = [(18, 4), (20, 4), (22, 4), (24, 4), (24, 5), (24, 6)]
FIG_N = 24                  # N used for the combined two-point/spectral figure (all d at this N)
OTOC_CONFIGS = [(24, 6)]    # mode-resolved OTOC; add (24, 4), (24, 5) for the d-dependence; [] to skip

Q = 4                       # interaction order (the generator below implements q = 4 only)
BETA = 5.0                  # inverse temperature (J = 1)
SEED = 42                   # base seed; config (N, d) uses SEED*100000 + N*100 + d
POOL_SIZE = 8               # number of valid candidates collected per (N, d)
FILTER_CHIRAL = True        # also reject candidates with a parity-preserving monomial that
                            # anticommutes with H (E -> -E within a parity sector).
                            # False reproduces the v1 instances; True changes the d = 4 ones.

N_INSTANCES = 5             # independent hSYK instances per (N, d); instance 0 is representative
MAX_CANDIDATE_CALLS = 20 * POOL_SIZE

N_TAU = 61                  # imaginary-time grid for G(tau); odd, so that tau = beta/2 is on the grid
SFF_T = np.logspace(-2, 5.0, 500)

OTOC_T = np.linspace(0.0, 5.0, 26)
OTOC_FIT_WINDOW = (1.5, 3.5)            # inspect the figure; change if needed
OTOC_ROBUSTNESS_WINDOWS = [(1.0, 3.0), (2.0, 4.0)]
OTOC_DTYPE = np.complex64
RUN_OTOC_VALIDATION = True              # compare fast OTOC vs direct formula at small N first
OTOC_INSTANCES = 1                      # instances per OTOC config

RUN_THERMO = True                       # energy/entropy vs temperature, ED vs SD (at N = FIG_N)
THERMO_BETAS = np.linspace(0.25, 10.0, 40)

# Controls at N = FIG_N (0 to skip). "dense" must remain the first key: postprocess
# regenerates the dense realizations by replaying the RNG stream from CONTROL_SEED.
CONTROL_SAMPLES = {"dense": 3, "signs": 3, "gaussian": 3}
CONTROL_SEED = 7

# Postprocess options
SD_CONVERGENCE_M = [2048, 8192, 32768]  # SD grid sizes for the convergence check (FFT solver)
RUN_RMT_CALIBRATION = True
RMT_SAMPLES = 200                       # random matrices per N for the finite-size <r> calibration
RMT_SEED = 12345
RUN_DENSE_THERMO_RECOVERY = True        # regenerate dense-SYK E/N, S/N (not saved by older runs)

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.path.join(_HERE, "..", "data", "ed")   # all inputs/outputs of this script
LOAD_SAVED_INSTANCES = True


if QUICK_TEST:
    RUN_CONFIGS = [(18, 4), (20, 4)]
    FIG_N = 20
    OTOC_CONFIGS = [(18, 4)]
    N_INSTANCES, OTOC_INSTANCES = 2, 2
    CONTROL_SAMPLES = {"dense": 1, "signs": 1, "gaussian": 1}
    THERMO_BETAS = np.linspace(0.5, 10.0, 20)     # step 0.5: contains BETA = 5 (see beta_index)
    RMT_SAMPLES = 20
    SD_CONVERGENCE_M = [2048, 8192]
    OUT_DIR = os.path.join(_HERE, "..", "data", "ed_quicktest")

# RMT large-N reference values of <r> (Atas, Bogomolny, Giraud, Roux, PRL 110, 084101 (2013))
R_TARGET = {"GOE": 0.5307, "GUE": 0.5996, "GSE": 0.6744}

plt.rcParams.update({'font.size': 11, 'figure.facecolor': 'white'})


# =====================================================================
# 1. HYPERGRAPH GENERATION
# =====================================================================

def generate_1overlap_hypergraph(N, q=4, d=4, max_attempts=1500):
    """
    Random q-uniform, d-regular, linear (one-overlap) hypergraph via a greedy
    triangle-clique search on the graph of still-available vertex pairs.
    Implemented for q = 4 only (q-1 = 3 new vertices per step).
    """
    if q != 4:
        raise NotImplementedError("generate_1overlap_hypergraph implements q = 4 only.")
    if (N * d) % q:
        # Otherwise K = N d / q is rounded down and a NON-regular hypergraph is returned
        # (some sites of degree < d), which breaks J_e = J/sqrt(d) and kappa_0 = 1.
        raise ValueError(f"A {q}-uniform {d}-regular hypergraph needs q | N d (N = {N}, d = {d}).")
    K = (N * d) // q
    for _ in range(max_attempts):
        deg = np.zeros(N, dtype=int)
        pair_used = np.zeros((N, N), dtype=bool)
        hyperedges, success = [], True
        for _k in range(K):
            avail = np.where(deg < d)[0]
            if len(avail) < q:
                success = False; break
            v1 = avail[np.argmax(deg[avail])]
            nbrs = [v for v in avail if v != v1 and not pair_used[v1, v]]
            n = len(nbrs)
            if n < q - 1:
                success = False; break
            triangles = []
            for i in range(n):
                u = nbrs[i]
                for j in range(i + 1, n):
                    v = nbrs[j]
                    if not pair_used[u, v]:
                        for l in range(j + 1, n):
                            w = nbrs[l]
                            if not pair_used[u, w] and not pair_used[v, w]:
                                triangles.append((u, v, w))
            if not triangles:
                success = False; break
            tri = triangles[np.random.randint(len(triangles))]
            quad = sorted([v1, tri[0], tri[1], tri[2]])
            for a in range(4):
                deg[quad[a]] += 1
                for b in range(a + 1, 4):
                    pair_used[quad[a], quad[b]] = pair_used[quad[b], quad[a]] = True
            hyperedges.append([int(x) for x in quad])
        if success and len(hyperedges) == K:
            return hyperedges
    raise ValueError(f"Failed to generate a one-overlap hypergraph for N={N}, d={d}.")


# =====================================================================
# 2. FILTERS AND GRAPH DIAGNOSTICS
# =====================================================================

def incidence_matrix(N, hyperedges):
    I = np.zeros((N, len(hyperedges)))
    for k, e in enumerate(hyperedges):
        for v in e:
            I[v, k] = 1.0
    return I


def f2_rank(A):
    """Rank over F2 of a 0/1 matrix."""
    M = (np.asarray(A) % 2).astype(np.uint8).copy()
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def f2_solve(A, b):
    """A solution x of A x = b over F2, or None if there is none."""
    A = (np.asarray(A) % 2).astype(np.uint8)
    b = (np.asarray(b) % 2).astype(np.uint8)
    M = np.concatenate([A, b[:, None]], axis=1)
    rows, cols = M.shape
    r, pivots = 0, []
    for c in range(cols - 1):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        pivots.append(c)
        r += 1
    if any(M[i, :-1].sum() == 0 and M[i, -1] for i in range(rows)):
        return None
    x = np.zeros(cols - 1, dtype=np.uint8)
    for i, c in enumerate(pivots):
        x[c] = M[i, -1]
    return x


def monomial_chiral_symmetry(N, hyperedges):
    """
    Returns None if no monomial P_S anticommutes with H; otherwise 'even' or
    'odd' for the parity of |S| (well defined when rank_F2(I) = N-1 and N is even).
    |e cap S| odd for all e  <=>  I^T x = 1 over F2.
      'even': P_S preserves fermion parity -> exact E -> -E symmetry within
              each parity sector.
      'odd' : P_S maps the two parity sectors onto each other with E -> -E.
    """
    I = incidence_matrix(N, hyperedges).astype(int)
    x = f2_solve(I.T, np.ones(len(hyperedges), dtype=int))
    if x is None:
        return None
    return "odd" if int(x.sum()) % 2 else "even"


def line_graph(hyperedges):
    L = nx.Graph()
    L.add_nodes_from(range(len(hyperedges)))
    for i in range(len(hyperedges)):
        si = set(hyperedges[i])
        for j in range(i + 1, len(hyperedges)):
            if si & set(hyperedges[j]):
                L.add_edge(i, j)
    return L


def has_trivial_automorphism(hyperedges):
    """|Aut(line graph)| == 1 (for a linear hypergraph, overlaps are 0 or 1)."""
    L = line_graph(hyperedges)
    count = 0
    for _ in isomorphism.GraphMatcher(L, L).isomorphisms_iter():
        count += 1
        if count > 1:
            return False
    return count == 1


def count_line_graph_triangles(hyperedges):
    """Hyperedge triples with pairwise overlap (crude proxy for local cycle density)."""
    return sum(nx.triangles(line_graph(hyperedges)).values()) // 3


def site_modes(N, hyperedges, q, d):
    """Eigenvalues (descending) and eigenvectors of K = (I I^T - d 1)/(d(q-1))."""
    I = incidence_matrix(N, hyperedges)
    K = (I @ I.T - d * np.eye(N)) / (d * (q - 1))
    kappa, V = np.linalg.eigh(K)
    order = np.argsort(kappa)[::-1]
    return kappa[order], V[:, order]


def site_graph_distances(N, hyperedges):
    A = np.zeros((N, N), dtype=int)
    for e in hyperedges:
        for a in e:
            for b in e:
                if a != b:
                    A[a, b] = 1
    dist = shortest_path(sp.csr_matrix(A), unweighted=True, directed=False)
    return dist


def hypergraph_diagnostics(N, hyperedges, q, d):
    kappa, _ = site_modes(N, hyperedges, q, d)
    dist = site_graph_distances(N, hyperedges)
    return {
        "num_hyperedges": len(hyperedges),
        "triangles": count_line_graph_triangles(hyperedges),
        "kappa_1": float(kappa[1]),
        "kappa_max": float(np.max(np.abs(kappa[1:]))),
        "kappa_min": float(kappa[-1]),
        "rank_F2": f2_rank(incidence_matrix(N, hyperedges)),
        "monomial_chiral_symmetry": monomial_chiral_symmetry(N, hyperedges),
        "site_graph_connected": bool(np.isfinite(dist).all()),
        "site_graph_diameter": int(dist.max()) if np.isfinite(dist).all() else None,
    }


# =====================================================================
# 3. INSTANCE SELECTION (with search statistics)
# =====================================================================

def select_instance(N, q, d, pool_size=POOL_SIZE, max_calls=MAX_CANDIDATE_CALLS, seed=None):
    """
    Collects up to pool_size valid candidates and returns the one with the
    smallest kappa_1, plus search statistics.
    """
    if (N * d) % q:   # checked here because the loop below treats generator ValueErrors as retries
        raise ValueError(f"A {q}-uniform {d}-regular hypergraph needs q | N d (N = {N}, d = {d}).")
    if seed is not None:
        np.random.seed(seed)
    pool, attempts = [], 0
    for _ in range(max_calls):
        attempts += 1
        try:
            edges = generate_1overlap_hypergraph(N, q, d)
        except ValueError:
            continue
        if f2_rank(incidence_matrix(N, edges)) != N - 1:
            continue
        if not has_trivial_automorphism(edges):
            continue
        if FILTER_CHIRAL and monomial_chiral_symmetry(N, edges) == "even":
            continue
        kappa, _ = site_modes(N, edges, q, d)
        pool.append((float(kappa[1]), edges))
        if len(pool) >= pool_size:
            break
    if not pool:
        raise RuntimeError(f"No valid candidate for N={N}, d={d} after {attempts} calls. "
                           f"Increase MAX_CANDIDATE_CALLS, or check whether the filters "
                           f"(e.g. FILTER_CHIRAL) can be satisfied at this size.")
    k1 = np.array([p[0] for p in pool])
    # Deterministic tie-break on kappa_1 rounded to 1e-10, then generation order.
    best = min(enumerate(pool), key=lambda ip: (round(ip[1][0], 10), ip[0]))[1]
    stats = {"attempts": attempts, "valid": len(pool), "valid_rate": len(pool) / attempts,
             "kappa1_pool_mean": float(k1.mean()), "kappa1_pool_std": float(k1.std()),
             "kappa1_best": best[0]}
    return best[1], stats


def selection_settings():
    return {"seed": SEED, "pool_size": POOL_SIZE, "max_candidate_calls": MAX_CANDIDATE_CALLS,
            "filter_chiral": FILTER_CHIRAL}


def save_instance(path, N, q, d, hyperedges, diagnostics, search_stats):
    with open(path, "w") as f:
        json.dump({"N": N, "q": q, "d": d, "hyperedges": hyperedges,
                   "couplings": "J_e = J/sqrt(d), J = 1, all positive",
                   "selection_settings": selection_settings(),
                   "diagnostics": diagnostics, "search_statistics": search_stats},
                  f, indent=2)


def load_instance(path):
    with open(path) as f:
        data = json.load(f)
    return ([list(map(int, e)) for e in data["hyperedges"]],
            data.get("search_statistics", {}), data.get("selection_settings"))


def instance_seed(N, d, m):
    """Seed of the selection procedure for instance m of (N, d). m = 0 reproduces the
    representative (v1) instance; m >= 1 uses independent keys derived from (SEED, N, d, m)."""
    if m == 0:
        return SEED * 100000 + N * 100 + d
    return int(np.random.SeedSequence([SEED, N, d, m]).generate_state(1)[0])


def instance_tag(m):
    return "" if m == 0 else f"_i{m}"


def levi_graph(N, hyperedges):
    B = nx.Graph()
    B.add_nodes_from(range(N), kind=0)
    B.add_nodes_from((N + k for k in range(len(hyperedges))), kind=1)
    for k, e in enumerate(hyperedges):
        for v in e:
            B.add_edge(v, N + k)
    return B


def isomorphic_hypergraphs(N, he1, he2, q, d):
    """True iff the two hypergraphs are isomorphic (same spectrum of K checked first)."""
    k1, _ = site_modes(N, he1, q, d)
    k2, _ = site_modes(N, he2, q, d)
    if not np.allclose(k1, k2, atol=1e-9):
        return False
    return nx.is_isomorphic(levi_graph(N, he1), levi_graph(N, he2),
                            node_match=lambda a, b: a["kind"] == b["kind"])


def get_instance(N, q, d, m=0, protect_saved=False):
    """
    (hyperedges, diagnostics, search statistics) of instance m of (N, d).

    A saved file is reused if its selection settings equal the current CONFIG; otherwise the
    instance is selected afresh and saved, overwriting the old file (the "full" run).
    protect_saved=True is for modes that only analyze existing data (otoc_reanalysis and the
    supplement): an existing file is then never written, and a settings mismatch raises
    instead of silently replacing the instance. A missing file is still generated and saved,
    since nothing is overwritten.
    """
    path = os.path.join(OUT_DIR, f"hsyk_N{N}_d{d}{instance_tag(m)}.json")
    exists = os.path.exists(path)
    if LOAD_SAVED_INSTANCES and exists:
        hyperedges, stats, settings = load_instance(path)
        # A file without recorded selection settings cannot be trusted to match the
        # current filters, so it is regenerated rather than reused.
        if settings is not None and settings == selection_settings():
            print(f"  loaded saved instance {path}")
            diag = hypergraph_diagnostics(N, hyperedges, q, d)
            if not protect_saved:
                save_instance(path, N, q, d, hyperedges, diag, stats)
            return hyperedges, diag, stats
        if protect_saved:
            raise RuntimeError(
                f"{path} was selected with settings {settings}, but the current CONFIG gives "
                f"{selection_settings()}. This mode does not replace saved instances: restore the "
                f"CONFIG values, or run 'python hsyk_numerics.py full' to reselect everything.")
        print(f"  saved instance {path} has missing or different selection settings; regenerating")
    elif exists and protect_saved:
        raise RuntimeError(f"LOAD_SAVED_INSTANCES = False would overwrite {path}; this mode does not "
                           f"replace saved instances.")
    hyperedges, stats = select_instance(N, q, d, seed=instance_seed(N, d, m))
    stats["seed"] = instance_seed(N, d, m)
    diag = hypergraph_diagnostics(N, hyperedges, q, d)
    save_instance(path, N, q, d, hyperedges, diag, stats)
    return hyperedges, diag, stats


# =====================================================================
# 4. EXACT DIAGONALIZATION: HAMILTONIANS, G(tau), LEVEL STATISTICS, SFF
# =====================================================================

def build_majoranas(N):
    """Jordan-Wigner Majoranas with {chi_i, chi_j} = delta_ij (sparse)."""
    I2 = sp.csr_matrix([[1, 0], [0, 1]])
    Z = sp.csr_matrix([[1, 0], [0, -1]])
    X = sp.csr_matrix([[0, 1], [1, 0]])
    Y = sp.csr_matrix([[0, -1j], [1j, 0]])

    def kron_list(ops):
        res = ops[0]
        for op in ops[1:]:
            res = sp.kron(res, op, format='csr')
        return res

    chis, L = [], N // 2
    for k in range(L):
        chis.append(kron_list([Z] * k + [X] + [I2] * (L - k - 1)))
        chis.append(kron_list([Z] * k + [Y] + [I2] * (L - k - 1)))
    return [c / np.sqrt(2.0) for c in chis]


def _monomial_full(chi):
    """(perm, coeff) with chi[a, perm[a]] = coeff[a] on the full space."""
    c = chi.tocsr()
    c.sort_indices()
    if not np.all(np.diff(c.indptr) == 1):
        raise ValueError("Majorana is not a monomial matrix.")
    return c.indices.copy(), c.data.copy()


def build_hamiltonian_terms(N, terms, couplings, q=4):
    """
    Dense H = sum_A c_A i^{q(q-1)/2} prod_{k in A} chi_k for index sets A (sorted).
    Each product of Majoranas is a signed permutation, so every term is added
    in O(dim). Returns (H, psi) with psi the sparse Majorana list.
    """
    psi = build_majoranas(N)
    mono = [_monomial_full(p) for p in psi]
    dim = 2 ** (N // 2)
    rows = np.arange(dim)
    H = np.zeros((dim, dim), dtype=complex)
    phase = (1j) ** (q * (q - 1) // 2)
    for A, cA in zip(terms, couplings):
        if len(A) != q:
            raise ValueError(f"Term {A} does not have q={q} sites.")
        perm, coeff = mono[A[0]]
        for k in A[1:]:
            pk, ck = mono[k]
            coeff = coeff * ck[perm]
            perm = pk[perm]
        H[rows, perm] += phase * cA * coeff
    return (H + H.conj().T) / 2.0, psi


def build_hamiltonian(N, hyperedges, d, q=4, J=1.0, couplings=None):
    """hSYK Hamiltonian; default couplings J_e = J/sqrt(d) for all e."""
    if couplings is None:
        couplings = np.full(len(hyperedges), J / np.sqrt(d))
    return build_hamiltonian_terms(N, hyperedges, couplings, q)


def dense_syk_terms(N, q, J, rng):
    """All q-subsets with Gaussian couplings of variance J^2 / C(N-1, q-1), so that
    Sigma = J^2 G^{q-1} exactly at the melonic level (same normalization as hSYK)."""
    from itertools import combinations
    from math import comb
    terms = [list(A) for A in combinations(range(N), q)]
    return terms, rng.normal(0.0, J / np.sqrt(comb(N - 1, q - 1)), size=len(terms))


def parity_indices(N):
    dim = 2 ** (N // 2)
    par = np.array([bin(i).count("1") % 2 for i in range(dim)])
    return np.where(par == 0)[0], np.where(par == 1)[0]


def diagonalize_and_G(psi, H, N, beta, tau):
    """
    Diagonalizes H in both parity sectors and computes
    G(tau) = (1/N) sum_i <T chi_i(tau) chi_i(0)>_beta.
    Returns (G, evals_for_statistics, full_spectrum), where evals_for_statistics
    is the even-sector spectrum with Kramers doublets removed when N = 4 mod 8,
    and full_spectrum = concatenate([even-sector, odd-sector]) spectra.
    """
    ie, io = parity_indices(N)
    Hs = sp.csr_matrix(H)
    Ee, Ve = np.linalg.eigh(Hs[ie, :][:, ie].toarray())
    Eo, Vo = np.linalg.eigh(Hs[io, :][:, io].toarray())
    M = np.zeros((len(Ee), len(Eo)))
    for p in psi:
        blk = p.tocsr()[ie, :][:, io]
        M += np.abs(Ve.conj().T @ (blk @ Vo)) ** 2
    E0 = min(Ee.min(), Eo.min())
    a, b = Ee - E0, Eo - E0
    Z = np.exp(-beta * a).sum() + np.exp(-beta * b).sum()
    G = np.array([(np.exp(-(beta - t) * a) @ M @ np.exp(-t * b) +
                   np.exp(-(beta - t) * b) @ M.T @ np.exp(-t * a)) / (N * Z) for t in tau])
    if N % 8 == 4:
        split = float(np.abs(Ee[0::2] - Ee[1::2]).max())
        if split > 1e-8 * max(1.0, float(np.abs(Ee).max())):
            raise RuntimeError(f"Kramers pairing failed (max splitting {split:.2e}).")
        evals = Ee[::2]
    else:
        evals = Ee
    return G, evals, np.concatenate([Ee, Eo])


def unfold_spectrum(evals, poly_deg=7):
    n = len(evals)
    return np.polyval(np.polyfit(evals, np.arange(n) / (n - 1), poly_deg), evals) * n


def level_statistics(evals, trim_frac=0.10):
    """Mean gap ratio <r> and unfolded spacings over the central (1 - 2 trim_frac) of the spectrum."""
    ev = np.sort(evals)
    k = int(trim_frac * len(ev))
    bulk = ev[k:len(ev) - k]
    g = np.diff(bulk)
    r = float(np.mean(np.minimum(g[:-1], g[1:]) / np.maximum(g[:-1], g[1:])))
    s = np.diff(unfold_spectrum(bulk))
    return r, s / s.mean()


def gap_ratio(evals, trim_frac=0.10):
    """<r> only (no unfolding), same trimming as level_statistics."""
    ev = np.sort(evals)
    k = int(trim_frac * len(ev))
    g = np.diff(ev[k:len(ev) - k])
    return float(np.mean(np.minimum(g[:-1], g[1:]) / np.maximum(g[:-1], g[1:])))


def log_smooth(t, y, sigma=0.15):
    lt = np.log10(t)
    w = np.exp(-0.5 * ((lt[:, None] - lt[None, :]) / sigma) ** 2)
    return (w / w.sum(axis=1, keepdims=True)) @ y


def spectral_form_factor(evals, beta, t):
    Zb = np.exp(-beta * evals).sum()
    raw = np.abs(np.exp(-(beta + 1j * t[:, None]) * evals).sum(axis=1)) ** 2 / Zb ** 2
    return raw, log_smooth(t, raw), np.exp(-2 * beta * evals).sum() / Zb ** 2


def symmetry_class(N):
    m = N % 8
    return "GOE" if m == 0 else ("GSE" if m == 4 else "GUE")


def beta_index(betas, beta):
    """
    Index k with betas[k] == beta. Raises if beta is not on the grid: the "at beta" thermodynamic
    values are read off the THERMO_BETAS grid, and picking the nearest point instead would
    silently report them at a different temperature.
    """
    betas = np.asarray(betas, dtype=float)
    k = int(np.argmin(np.abs(betas - beta)))
    if not np.isclose(betas[k], beta, rtol=0.0, atol=1e-9):
        raise ValueError(f"BETA = {beta:g} is not on the THERMO_BETAS grid (nearest point {betas[k]:g}). "
                         f"Choose THERMO_BETAS so that it contains BETA.")
    return k


def ed_thermodynamics(spectrum, N, betas):
    """Energy and entropy per Majorana from the full spectrum: E/N and S/N = (ln Z + beta E)/N."""
    E0 = spectrum.min()
    e = spectrum - E0
    out_E, out_S = [], []
    for b in betas:
        w = np.exp(-b * e)
        Z = w.sum()
        E = (spectrum * w).sum() / Z
        out_E.append(E / N)
        out_S.append((np.log(Z) - b * E0 + b * E) / N)
    return np.array(out_E), np.array(out_S)


# =====================================================================
# 5. SCHWINGER-DYSON SOLUTION (large-d reference) AND ITS THERMODYNAMICS
# =====================================================================

def solve_sd(beta, q=4, J=1.0, n_iter=400, M=2048):
    """
    Melonic SD equation G(iw)^{-1} = -iw - Sigma(iw), Sigma(tau) = J^2 G(tau)^{q-1},
    by damped iteration. Internally G(0+) = -1/2; the sign is flipped on
    return to match the ED convention G(0+) = +1/2. Grid tau_k = k beta / M.
    """
    tau = np.linspace(0, beta, M, endpoint=False)
    wn = (2 * np.arange(-M // 2, M // 2) + 1) * np.pi / beta
    F = np.exp(-1j * tau[:, None] * wn[None, :]) / beta
    Finv = np.exp(1j * wn[:, None] * tau[None, :]) * (beta / M)
    # Trapezoid rule for Sigma(iw) = int_0^beta dtau e^{iw tau} Sigma(tau): the endpoint terms
    # f(0)/2 + f(beta)/2 cancel for antiperiodic Sigma, so the tau = 0 sample gets weight 0.
    # (The plain left-point rule breaks tau -> beta - tau symmetry at O(1/M); G(beta/2) and the
    # energy are unaffected, but G(tau) elsewhere is off by ~2e-4 at M = 2048.)
    Finv[:, 0] = 0.0
    G0w = 1.0 / (1j * wn)
    G0t = -0.5 * np.ones(M)
    Gw = G0w.copy()
    for _ in range(n_iter):
        Gt = G0t + np.real(F @ (Gw - G0w))
        Sw = Finv @ (J ** 2 * Gt ** (q - 1))
        Gw = 0.7 * Gw + 0.3 / (1j * wn - Sw)
    return tau, -(G0t + np.real(F @ (Gw - G0w)))


def solve_sd_fft(beta, q=4, J=1.0, n_iter=400, M=2 ** 15):
    """
    Same equations, discretization and iteration as solve_sd, with the Fourier
    sums done by FFT (O(M log M) per step), so that large M is affordable.
    Agrees with solve_sd at equal M to rounding error (checked in postprocess).
    """
    k = np.arange(M)
    tau = k * beta / M
    n = np.arange(-M // 2, M // 2)
    wn = (2 * n + 1) * np.pi / beta
    ph = np.exp(-1j * np.pi * k / M)

    def to_tau(Gw):   # (1/beta) sum_n Gw_n exp(-i w_n tau_k)
        return ph * np.fft.fft(np.fft.ifftshift(Gw)) / beta

    def to_freq(Ft):  # (beta/M) sum_k Ft_k exp(+i w_n tau_k), trapezoid (tau = 0 weight 0, see solve_sd)
        Ft = Ft.copy()
        Ft[0] = 0.0
        return np.fft.fftshift(np.fft.ifft(np.conj(ph) * Ft) * M) * beta / M

    G0w = 1.0 / (1j * wn)
    G0t = -0.5 * np.ones(M)
    Gw = G0w.copy()
    for _ in range(n_iter):
        Gt = G0t + np.real(to_tau(Gw - G0w))
        Sw = to_freq(J ** 2 * Gt ** (q - 1))
        Gw = 0.7 * Gw + 0.3 / (1j * wn - Sw)
    return tau, -(G0t + np.real(to_tau(Gw - G0w)))


def sd_thermodynamics(betas, q=4, J=1.0):
    """
    E/N = -(J^2/q) int_0^beta dtau G(tau)^q at each beta (G > 0 convention), and
    S/N = (1/2) ln 2 + beta E/N - int_0^beta E/N dbeta' (trapezoid, with E(0) = 0).
    betas must be increasing.
    """
    E = []
    for b in betas:
        tau, G = solve_sd(b, q=q, J=J)
        E.append(-(J ** 2 / q) * np.sum(G ** q) * b / len(tau))
    E = np.array(E)
    bb = np.concatenate([[0.0], betas])
    EE = np.concatenate([[0.0], E])
    trap = getattr(np, 'trapezoid', None) or np.trapz
    integral = np.array([trap(EE[:k + 2], bb[:k + 2]) for k in range(len(betas))])
    S = 0.5 * np.log(2) + betas * E - integral
    return E, S


# =====================================================================
# 5b. CONTROLS: dense SYK, random signs, Gaussian couplings (same N)
# =====================================================================

def run_controls(N, q, hyperedges_by_d, beta, tau, tau_sd, G_sd, betas_thermo):
    """
    For each control kind and sample: G(tau), G(beta/2) - G_SD(beta/2), <r>,
    and (if RUN_THERMO) E/N, S/N. Returns a nested dict of results.
    """
    rng = np.random.default_rng(CONTROL_SEED)
    mid_ed, mid_sd = len(tau) // 2, len(tau_sd) // 2
    results = {}
    for kind, n_samples in CONTROL_SAMPLES.items():
        if n_samples <= 0:
            continue
        variants = [(None, None)] if kind == "dense" else sorted(hyperedges_by_d.items())
        for d, he in variants:
            key = kind if d is None else f"{kind}_d{d}"
            rows = []
            for smp in range(n_samples):
                if kind == "dense":
                    terms, cpl = dense_syk_terms(N, q, 1.0, rng)
                elif kind == "signs":
                    terms, cpl = he, rng.choice([-1.0, 1.0], size=len(he)) / np.sqrt(d)
                else:
                    terms, cpl = he, rng.normal(0.0, 1.0 / np.sqrt(d), size=len(he))
                H, psi = build_hamiltonian_terms(N, terms, cpl, q)
                G, evals, spec = diagonalize_and_G(psi, H, N, beta, tau)
                r, _ = level_statistics(evals)
                row = {"G": G, "dG_mid": float(G[mid_ed] - G_sd[mid_sd]), "r": r}
                if RUN_THERMO:
                    row["E"], row["S"] = ed_thermodynamics(spec, N, betas_thermo)
                rows.append(row)
                print(f"    control {key} sample {smp}: <r>={r:.4f}, G(beta/2)-G_SD(beta/2)={row['dG_mid']:+.5f}")
                del H
            results[key] = rows
    return results


def summarize_controls(results, betas=None, beta=None):
    out = {}
    k_beta = None if betas is None else beta_index(betas, beta)
    for key, rows in results.items():
        dG = np.array([r["dG_mid"] for r in rows])
        rr = np.array([r["r"] for r in rows])
        out[key] = {"n": len(rows), "dG_mid_mean": float(dG.mean()), "dG_mid_std": float(dG.std()),
                    "dG_mid_min": float(dG.min()), "dG_mid_max": float(dG.max()),
                    "r_mean": float(rr.mean()), "r_std": float(rr.std()),
                    "r_min": float(rr.min()), "r_max": float(rr.max())}
        if k_beta is not None and "E" in rows[0]:
            Eb = np.array([r["E"][k_beta] for r in rows])
            Sb = np.array([r["S"][k_beta] for r in rows])
            out[key].update({"beta": float(betas[k_beta]),
                             "E_beta_values": Eb.tolist(), "S_beta_values": Sb.tolist()})
    return out


def save_controls(path, controls, betas_thermo):
    """Saves G(tau), <r> and (if present) E/N, S/N of every control sample."""
    arrays = {"betas_thermo": np.asarray(betas_thermo)}
    for k, v in controls.items():
        arrays[f"{k}_G"] = np.array([row["G"] for row in v])
        arrays[f"{k}_r"] = np.array([row["r"] for row in v])
        arrays[f"{k}_dG_mid"] = np.array([row["dG_mid"] for row in v])
        if "E" in v[0]:
            arrays[f"{k}_E"] = np.array([row["E"] for row in v])
            arrays[f"{k}_S"] = np.array([row["S"] for row in v])
    np.savez_compressed(path, **arrays)


# =====================================================================
# 6. RETARDED LADDER KERNEL (Maldacena-Stanford eq. 3.156)
# =====================================================================

def k_retarded(hp, q):
    """
    k_R(hp) = Gamma(3-2D) Gamma(2D-hp) / [Gamma(1+2D) Gamma(2-2D-hp)],  D = 1/q,
    on the ansatz exp(-hp pi (t1+t2)/beta)/cosh(pi t12/beta)^{2D-hp}; growth
    rate -hp 2pi/beta. k_R(-1) = 1, k_R(0) = q-1. (Paper: k_SYK(h;q) = k_R(1-h).)
    """
    D = 1.0 / q
    return gamma_fn(3 - 2 * D) * gamma_fn(2 * D - hp) / (gamma_fn(1 + 2 * D) * gamma_fn(2 - 2 * D - hp))


def mode_growth_rate(kappa, q, beta):
    """Conformal growth rate lambda(kappa) from kappa k_R = 1; None for kappa <= 0."""
    if kappa <= 0:
        return None
    if kappa >= 1:
        return 2 * np.pi / beta
    D = 1.0 / q
    f = lambda hp: k_retarded(hp, q) - 1.0 / kappa
    hi = 2 * D - 1e-12
    if f(hi) <= 0:
        return -2 * D * 2 * np.pi / beta
    return -brentq(f, -1.0, hi) * 2 * np.pi / beta


def predicted_decay_rate(kappa, q, beta):
    """
    Conformal late-time decay rate of the mode-projected OTOC. The disconnected piece decays
    at the pole of k_R (hp = 2 Delta), so that rate cancels in (1 - kappa k_R)^(-1) F_0 for
    kappa != 0, and the pole condition kappa k_R(hp) = 1 applies on both sides of it:
    hp < 2 Delta for kappa > 0, hp > 2 Delta (where k_R < 0) for kappa < 0. For q = 4,
    k_R(hp) = (3/2)/(1/2 - hp) exactly (no other singularity), and the rate is the single line
    (2 pi/beta)(1 - 3 kappa)/2 for all kappa >= -1/(q-1) (the lower spectral bound of K);
    at the bound the decay rate equals 2 pi/beta. For q = 6, 8 a continued solution exists
    for all kappa >= -1/(q-1) (checked numerically).
    """
    if q == 4:
        return (2 * np.pi / beta) * 0.5 * (1 - 3 * kappa)
    D = 1.0 / q
    if kappa > 0:
        return -mode_growth_rate(kappa, q, beta)
    if kappa == 0:
        return 2 * D * 2 * np.pi / beta
    hp = np.linspace(2 * D + 1e-9, 2 * D + 1 - 1e-9, 20001)   # branch past the pole, k_R < 0 here
    f = k_retarded(hp, q) - 1.0 / kappa
    idx = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
    if len(idx) == 0:
        return 2 * D * 2 * np.pi / beta                       # no continued pole found
    return brentq(lambda x: k_retarded(x, q) - 1.0 / kappa, hp[idx[0]], hp[idx[0] + 1]) * 2 * np.pi / beta


# =====================================================================
# 7. ALL-PAIRS SITE-RESOLVED OTOC AND MODE-RESOLVED ANALYSIS
# =====================================================================

def _monomial_block(chi, rows, cols):
    """(perm, inv_perm, coeff) with chi[rows][:, cols][a, perm[a]] = coeff[a]."""
    blk = chi.tocsr()[rows, :][:, cols].tocsr()
    blk.sort_indices()
    if not np.all(np.diff(blk.indptr) == 1):
        raise ValueError("Majorana block is not a monomial matrix.")
    perm = blk.indices.copy()
    return perm, np.argsort(perm), blk.data.copy()


def _tr_square_pair(B, perm, inv, c1, c2):
    """
    Tr[(B X)^2] for the two Majoranas of one Jordan-Wigner qubit, which share
    the permutation part: X[b, perm[b]] = c[b] with c = c1 or c2.
        Tr = sum_{a,b} B[a,b] c[b] B[perm[b], inv[a]] c[inv[a]] = u^T W c,
    W = B * B[perm][:, inv].T,  u = c[inv].  W is shared by the pair.
    """
    W = B * B[perm][:, inv].T
    return c1[inv] @ W @ c1, c2[inv] @ W @ c2


def compute_all_site_otocs(H, psi, N, beta, t_array, sites=None, dtype=np.complex128, verbose=True):
    """
    Regularized OTOC F_ij(t) = Tr[y chi_i(t) y chi_j y chi_i(t) y chi_j], y = rho^{1/4},
    for all evolved sites i in `sites` (default all) and all j.
    Returns F of shape (len(t_array), len(sites), N).
    """
    sites = list(range(N)) if sites is None else list(sites)
    ie, io = parity_indices(N)
    Hs = sp.csr_matrix(H)
    Ee, Ve = np.linalg.eigh(Hs[ie, :][:, ie].toarray())
    Eo, Vo = np.linalg.eigh(Hs[io, :][:, io].toarray())
    E0 = min(Ee.min(), Eo.min())
    Ee, Eo = Ee - E0, Eo - E0
    Z = np.exp(-beta * Ee).sum() + np.exp(-beta * Eo).sum()
    ye, yo = np.exp(-beta * Ee / 4) / Z ** 0.25, np.exp(-beta * Eo / 4) / Z ** 0.25
    Ve, Vo = Ve.astype(dtype), Vo.astype(dtype)
    mono_eo = [_monomial_block(p, ie, io) for p in psi]
    mono_oe = [_monomial_block(p, io, ie) for p in psi]
    pairs = []
    for k in range(N // 2):
        pa, ia, ca = mono_oe[2 * k]
        pb, _, cb = mono_oe[2 * k + 1]
        if not np.array_equal(pa, pb):
            raise ValueError("Majoranas 2k, 2k+1 do not share a permutation.")
        pairs.append((pa, ia, ca.astype(dtype), cb.astype(dtype)))

    F = np.zeros((len(t_array), len(sites), N))
    t0 = time.time()
    for s, i in enumerate(sites):
        perm, _, c = mono_eo[i]
        X_eo = Ve.conj().T @ (c.astype(dtype)[:, None] * Vo[perm, :])
        for k, t in enumerate(t_array):
            Le = Ve * (ye * np.exp(1j * Ee * t)).astype(dtype)[None, :]
            Lo = Vo * (yo * np.exp(1j * Eo * t)).astype(dtype)[None, :]
            B_eo = (Le @ X_eo) @ Lo.conj().T          # even->odd block of y chi_i(t) y
            for q_idx, (pp, ii, c1, c2) in enumerate(pairs):
                t1, t2 = _tr_square_pair(B_eo, pp, ii, c1, c2)
                F[k, s, 2 * q_idx] = 2 * t1.real
                F[k, s, 2 * q_idx + 1] = 2 * t2.real
        if verbose and s == 0:
            dt = time.time() - t0
            print(f"    [OTOC] first site: {dt:.1f} s; estimated total {dt * len(sites) / 60:.1f} min")
    if verbose:
        print(f"    [OTOC] done in {(time.time() - t0) / 60:.1f} min")
    return F


def otoc_direct_reference(H, psi, beta, i, j, t_array):
    """Direct (slow) evaluation of F_ij(t) by full diagonalization, for validation."""
    E, U = np.linalg.eigh(H)
    E = E - E.min()
    Z = np.exp(-beta * E).sum()
    y = np.exp(-beta * E / 4) / Z ** 0.25
    ci = U.conj().T @ psi[i].toarray() @ U
    cj = U.conj().T @ psi[j].toarray() @ U
    out = []
    for t in t_array:
        ph = np.exp(1j * E * t)
        A = (y * ph)[:, None] * ci * (y * ph.conj())[None, :]
        Nm = A @ cj
        out.append(np.einsum('nm,mn->', Nm, Nm).real)
    return np.array(out)


def validate_otoc(N=16, d=4, q=4, beta=5.0, seed=1, dtype=np.complex128, tol=1e-8):
    """Checks compute_all_site_otocs (with the given dtype) against the direct formula."""
    np.random.seed(seed)
    he = generate_1overlap_hypergraph(N, q, d)
    H, psi = build_hamiltonian(N, he, d, q)
    t = np.linspace(0, 4, 5)
    F = compute_all_site_otocs(H, psi, N, beta, t, sites=[0, 3], dtype=dtype, verbose=False)
    dev = 0.0
    for s_idx, i in enumerate([0, 3]):
        for j in (1, 6, 9, i):
            dev = max(dev, np.abs(F[:, s_idx, j] - otoc_direct_reference(H, psi, beta, i, j, t)).max())
    print(f"  [OTOC validation] N={N}, dtype={np.dtype(dtype).name}: "
          f"max deviation from direct formula = {dev:.1e} (tolerance {tol:.0e})")
    if dev > tol:
        raise RuntimeError("OTOC validation failed.")
    return dev


def _eigen_groups(kappa, tol=1e-8):
    groups, cur = [], [0]
    for a in range(1, len(kappa)):
        if abs(kappa[a] - kappa[cur[-1]]) < tol:
            cur.append(a)
        else:
            groups.append(cur); cur = [a]
    groups.append(cur)
    return groups


def mode_projected_otocs(F, V, kappa, tol=1e-8):
    """
    Returns (kappa_groups, Ft, offdiag):
      Ft[t, g] = (1/m_g) Tr[P_g F_sym(t)] over (near-)degenerate eigenspaces g of K;
      offdiag[t] = ||off-diagonal part|| / ||all|| of V^T F_sym(t) V restricted to the
      non-uniform modes (Frobenius norms).
    A group with |kappa| < tol is set to kappa = 0 exactly: numerically it comes out as
    about +-1e-17, with a sign that depends on the numpy/BLAS build, and the sign decides
    whether the mode is counted in the kappa <= 0 fits of analyze_mode_otocs.
    """
    Fs = 0.5 * (F + np.transpose(F, (0, 2, 1)))
    Mt = np.einsum('ia,tij,jb->tab', V, Fs, V)
    groups = _eigen_groups(kappa, tol)
    label = np.empty(len(kappa), dtype=int)
    for g, idx in enumerate(groups):
        label[idx] = g
    diag_mask = label[:, None] == label[None, :]
    nu = label > 0
    blk = Mt[:, nu][:, :, nu]
    dm = diag_mask[nu][:, nu]
    offdiag = np.sqrt((np.abs(blk) ** 2 * ~dm).sum(axis=(1, 2)) / (np.abs(blk) ** 2).sum(axis=(1, 2)))
    per_mode = np.einsum('taa->ta', Mt)
    Ft = np.stack([per_mode[:, g].mean(axis=1) for g in groups], axis=1)
    kg = np.array([kappa[g[0]] for g in groups])
    kg[np.abs(kg) < tol] = 0.0
    return kg, Ft, offdiag


def fit_window_mask(t_array, window, min_points=3):
    """Boolean mask of t_array inside the closed fit window; raises if it has < min_points points."""
    t_array = np.asarray(t_array)
    m = (t_array >= window[0]) & (t_array <= window[1])
    if m.sum() < min_points:
        raise ValueError(f"OTOC fit window {tuple(window)} contains {int(m.sum())} point(s) of the "
                         f"time grid (need >= {min_points}); widen the window or refine OTOC_T.")
    return m


def check_otoc_windows():
    """Validates OTOC_FIT_WINDOW and OTOC_ROBUSTNESS_WINDOWS before any OTOC is computed."""
    for w in [OTOC_FIT_WINDOW, *OTOC_ROBUSTNESS_WINDOWS]:
        fit_window_mask(OTOC_T, w)


def analyze_mode_otocs(F, t_array, kappa, V, q, beta, fit_window, verbose=True):
    """Fits late-time decay rates of each non-uniform mode and compares with the conformal prediction."""
    t_array = np.asarray(t_array)
    kg, Ft, offdiag = mode_projected_otocs(F, V, kappa)
    m = fit_window_mask(t_array, fit_window)      # raises before y[0] below could fail
    rates = np.full(len(kg), np.nan)
    sign_changes = 0
    for g in range(1, len(kg)):
        y = Ft[m, g]
        if np.any(np.sign(y) != np.sign(y[0])):
            sign_changes += 1
            continue
        if np.all(np.abs(y) > 1e-14):
            rates[g] = -np.polyfit(t_array[m], np.log(np.abs(y)), 1)[0]
    pred = np.array([np.nan] + [predicted_decay_rate(k, q, beta) for k in kg[1:]])

    res = {"fit_window": list(fit_window), "modes_with_sign_change": sign_changes,
           "offdiag_fraction_in_window": float(offdiag[m].mean())}
    pos = ~np.isnan(rates) & (kg > 0) & (np.arange(len(kg)) > 0)
    neg = ~np.isnan(rates) & (kg <= 0)
    allm = ~np.isnan(rates) & (np.arange(len(kg)) > 0)
    if allm.sum() >= 3:
        res["slope_all_modes"] = float(np.polyfit(kg[allm], rates[allm], 1)[0])
        res["offset_all_modes_mean"] = float(np.mean(rates[allm] - pred[allm]))
    if neg.sum() >= 2:
        res["slope_kappa_nonpos"] = float(np.polyfit(kg[neg], rates[neg], 1)[0])
    if pos.sum() >= 3:
        res["slope_observed"] = float(np.polyfit(kg[pos], rates[pos], 1)[0])
        res["slope_conformal"] = float(np.polyfit(kg[pos], pred[pos], 1)[0])
        res["offset_mean"] = float(np.mean(rates[pos] - pred[pos]))
    if neg.sum() >= 2:
        res["rate_kappa_nonpos_mean"] = float(rates[neg].mean())
        res["rate_kappa_nonpos_std"] = float(rates[neg].std())
        res["rate_kappa_nonpos_disconnected"] = (2.0 / q) * 2 * np.pi / beta
        res["rate_kappa_nonpos_conformal_mean"] = float(np.mean(pred[neg]))
    if verbose:
        print(f"    window {fit_window}: " + ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in res.items() if k != "fit_window"))
        if sign_changes:
            print(f"    [warning] {sign_changes} mode(s) change sign in the fit window; not fitted")
    return kg, Ft, offdiag, rates, pred, res


def plot_mode_otoc_figure(t, kg, Ft, rates, q, beta, fit_window, N, d, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=220)
    idx = sorted({1, len(kg) // 2, len(kg) - 1})
    for c, g in zip(plt.cm.viridis(np.linspace(0.1, 0.85, len(idx))), idx):
        axes[0].semilogy(t, np.abs(Ft[:, g]), lw=2, color=c, label=rf'$\kappa_\alpha={kg[g]:+.3f}$')
    axes[0].axvspan(*fit_window, color='gray', alpha=0.12, label='fit window')
    axes[0].set_xlabel(r'$t$'); axes[0].set_ylabel(r'$|\widetilde F_\alpha(t)|$')
    axes[0].set_title(rf'(a) Mode-projected OTOC ($N={N},\,d={d},\,\beta={beta:g}$)')
    axes[0].legend(frameon=True, fontsize=9)

    kk = np.linspace(min(kg[1:].min(), -0.05), max(kg[1:].max(), 0.35), 200)
    axes[1].plot(kk, [predicted_decay_rate(k, q, beta) for k in kk], 'k--', lw=1.8,
                 label='strong-coupling prediction')
    axes[1].plot(kg[1:], rates[1:], 'o', ms=6, color='#1f77b4', label='observed (fit)')
    axes[1].axvline(0, color='gray', lw=0.8, alpha=0.6)
    axes[1].axhline((2.0 / q) * 2 * np.pi / beta, color='gray', ls=':', lw=1.2,
                    label=r'disconnected rate $2\Delta\cdot 2\pi/\beta$')
    axes[1].set_xlabel(r'site-space eigenvalue $\kappa_\alpha$')
    axes[1].set_ylabel(r'late-time decay rate of $\widetilde F_\alpha$')
    axes[1].set_title(r'(b) Decay rate vs $\kappa_\alpha$')
    axes[1].legend(frameon=True, fontsize=9)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.close(fig)


def run_otoc_for_instance(N, d, q, hyperedges, beta, tag="", reuse_only=False):
    """Mode-resolved OTOC of one instance. reuse_only=True (otoc_reanalysis) never computes
    the OTOC: it raises if no saved all-pairs data match the instance, beta and time grid."""
    kappa, V = site_modes(N, hyperedges, q, d)
    path = os.path.join(OUT_DIR, f"otoc_allpairs_N{N}_d{d}{tag}.npz")
    F = None
    if os.path.exists(path):
        data = np.load(path)
        if (data["t"].shape == OTOC_T.shape and np.allclose(data["t"], OTOC_T)
                and float(data["beta"]) == beta
                and np.array_equal(data["hyperedges"], np.array(hyperedges))):
            print(f"    [OTOC] reusing {path}")
            F = data["F"]
        else:
            print(f"    [OTOC] {path} does not match this instance/time grid"
                  + ("" if reuse_only else "; recomputing"))
    if F is None and reuse_only:
        raise RuntimeError(f"No saved all-pairs OTOC in {path} matches this instance, beta and OTOC_T; "
                           f"otoc_reanalysis only refits saved data. Run 'python hsyk_numerics.py full' "
                           f"to compute it.")
    if F is None:
        H, psi = build_hamiltonian(N, hyperedges, d, q)
        F = compute_all_site_otocs(H, psi, N, beta, OTOC_T, dtype=OTOC_DTYPE)
        np.savez_compressed(path, t=OTOC_T, beta=beta, F=F, hyperedges=np.array(hyperedges))

    kg, Ft, offdiag, rates, pred, res = analyze_mode_otocs(F, OTOC_T, kappa, V, q, beta, OTOC_FIT_WINDOW)
    robustness = [analyze_mode_otocs(F, OTOC_T, kappa, V, q, beta, w)[5] for w in OTOC_ROBUSTNESS_WINDOWS]
    plot_mode_otoc_figure(OTOC_T, kg, Ft, rates, q, beta, OTOC_FIT_WINDOW, N, d,
                          os.path.join(OUT_DIR, f"fig_otoc_modes_N{N}_d{d}{tag}.pdf"))
    np.savez_compressed(os.path.join(OUT_DIR, f"otoc_modes_N{N}_d{d}{tag}.npz"),
                        t=OTOC_T, kappa_groups=kg, Ft=Ft, offdiag=offdiag, rates=rates, pred=pred)
    with open(os.path.join(OUT_DIR, f"otoc_mode_rates_N{N}_d{d}{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kappa", "rate_observed", "rate_strong_coupling"])
        for k, r, p in zip(kg[1:], rates[1:], pred[1:]):
            w.writerow([f"{k:.6f}", f"{r:.6f}", f"{p:.6f}"])
    return {"main": res, "robustness": robustness}, (kg, rates)


def plot_otoc_rate_overlay(per_instance, q, beta, N, d, save_path):
    """Decay rate vs kappa for several instances on one plot (no averaging)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=220)
    allk = np.concatenate([kg[1:] for kg, _ in per_instance.values()])
    kk = np.linspace(min(allk.min(), -0.05), max(allk.max(), 0.35), 200)
    ax.plot(kk, [predicted_decay_rate(k, q, beta) for k in kk], 'k--', lw=1.8,
            label='strong-coupling prediction')
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    for (m, (kg, rates)), mk in zip(sorted(per_instance.items()), markers * 4):
        ax.plot(kg[1:], rates[1:], mk, ms=5, alpha=0.85, label=f'instance {m}')
    ax.axvline(0, color='gray', lw=0.8, alpha=0.6)
    ax.set_xlabel(r'site-space eigenvalue $\kappa_\alpha$')
    ax.set_ylabel(r'late-time decay rate of $\widetilde F_\alpha$')
    ax.set_title(rf'Decay rate vs $\kappa_\alpha$, {len(per_instance)} instances ($N={N},\,d={d}$)')
    ax.legend(frameon=True, fontsize=8)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.close(fig)


# =====================================================================
# 8. FIGURES AND TABLES
# =====================================================================

def plot_two_point_spectral_figure(res_by_d, tau_ed, tau_sd, G_sd, N, save_path, controls=None,
                                   other_dG=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)
    colors = plt.cm.tab10(np.arange(len(res_by_d)))
    markers = ['o', 's', '^', 'D', 'v']
    mid_ed, mid_sd = len(tau_ed) // 2, len(tau_sd) // 2
    inv_d, dG = [], []
    for (d, r), c, mk in zip(sorted(res_by_d.items()), colors, markers):
        axes[0].plot(tau_ed, r['G'], ls='', marker=mk, ms=4.5, color=c, alpha=0.85, label=f'$d={d}$')
        inv_d.append(1.0 / d); dG.append(r['G'][mid_ed] - G_sd[mid_sd])
    axes[0].plot(tau_sd, G_sd, 'k--', lw=2, label=r'SD ($d\to\infty$)')
    axes[0].set_title(rf'(a) $G(\tau)$ ($N={N}$)'); axes[0].set_xlabel(r'$\tau$'); axes[0].set_ylabel(r'$G(\tau)$')
    axes[0].legend(frameon=True, loc='upper center')
    if len(inv_d) >= 2:
        ins = axes[0].inset_axes([0.34, 0.28, 0.32, 0.32])
        ins.plot(inv_d, dG, 'o-', color='crimson', ms=4, lw=1, label='hSYK')
        if other_dG:
            first = True
            for d_o, vals in sorted(other_dG.items()):
                if vals:
                    ins.plot([1.0 / d_o] * len(vals), vals, 'o', mfc='none', mec='crimson', ms=3,
                             alpha=0.7, label='other instances' if first else None)
                    first = False
        ins.axhline(0, color='k', lw=0.6)
        if controls and "dense" in controls:
            dd = np.array([row["dG_mid"] for row in controls["dense"]])
            ins.axhspan(dd.mean() - dd.std(), dd.mean() + dd.std(), color='gray', alpha=0.3)
            ins.axhline(dd.mean(), color='gray', lw=1, label=f'dense SYK, $N={N}$')
        ins.legend(fontsize=6, frameon=False, loc='lower right')
        ins.set_xlabel(r'$1/d$', fontsize=7.5); ins.set_ylabel(r'$\Delta G(\beta/2)$', fontsize=7.5)
        ins.tick_params(labelsize=6.5)

    cls = symmetry_class(N)
    sx = np.linspace(0, 3.5, 300)
    surmise = {"GOE": (np.pi / 2) * sx * np.exp(-(np.pi / 4) * sx ** 2),
               "GUE": (32 / np.pi ** 2) * sx ** 2 * np.exp(-(4 / np.pi) * sx ** 2),
               "GSE": (262144 / (729 * np.pi ** 3)) * sx ** 4 * np.exp(-(64 / (9 * np.pi)) * sx ** 2)}[cls]
    axes[1].plot(sx, surmise, 'r-', lw=2, label=f'{cls} surmise', zorder=10)
    axes[1].plot(sx, np.exp(-sx), 'k--', lw=1.5, label='Poisson', zorder=9)
    for (d, r), c in zip(sorted(res_by_d.items()), colors):
        axes[1].hist(r['s'], bins=30, density=True, histtype='step', lw=1.8, color=c,
                     label=rf"$d={d}$ ($\langle r\rangle={r['r']:.4f}$)")
    axes[1].set_xlim(0, 3.5); axes[1].set_title(rf'(b) Level spacing ($N={N}$)')
    axes[1].set_xlabel(r'$s$'); axes[1].set_ylabel(r'$P(s)$'); axes[1].legend(frameon=True, loc='upper right')

    for (d, r), c in zip(sorted(res_by_d.items()), colors):
        axes[2].loglog(SFF_T, r['sff_raw'], color=c, alpha=0.12, lw=0.6)
        axes[2].loglog(SFF_T, r['sff_smooth'], color=c, lw=2, label=f'$d={d}$')
        axes[2].axhline(r['plateau'], color=c, ls=':', alpha=0.6)
    axes[2].set_title(rf'(c) Spectral form factor ($N={N}$)'); axes[2].set_xlabel(r'$t$'); axes[2].set_ylabel(r'SFF$(t)$')
    axes[2].legend(frameon=True, loc='upper left')
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.close(fig)


def plot_thermodynamics_figure(betas, thermo_by_d, E_sd, S_sd, N, save_path, controls=None,
                               other_thermo=None):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=220)
    T = 1.0 / betas
    colors = plt.cm.tab10(np.arange(len(thermo_by_d)))
    for (d, (E, S)), c in zip(sorted(thermo_by_d.items()), colors):
        axes[0].plot(T, E, 'o', ms=3.5, color=c, label=f'hSYK $d={d}$')
        axes[1].plot(T, S, 'o', ms=3.5, color=c, label=f'hSYK $d={d}$')
        for Eo, So in (other_thermo or {}).get(d, []):
            axes[0].plot(T, Eo, '-', color=c, lw=0.7, alpha=0.5)
            axes[1].plot(T, So, '-', color=c, lw=0.7, alpha=0.5)
    if controls and "dense" in controls:
        Ed = np.mean([row["E"] for row in controls["dense"]], axis=0)
        Sd = np.mean([row["S"] for row in controls["dense"]], axis=0)
        axes[0].plot(T, Ed, '-', color='gray', lw=1.5, label=f'dense SYK, $N={N}$')
        axes[1].plot(T, Sd, '-', color='gray', lw=1.5, label=f'dense SYK, $N={N}$')
    axes[0].plot(T, E_sd, 'k--', lw=2, label=r'SD ($N,d\to\infty$)')
    axes[1].plot(T, S_sd, 'k--', lw=2, label=r'SD ($N,d\to\infty$)')
    axes[0].set_xscale('log'); axes[1].set_xscale('log')
    axes[0].set_xlabel(r'$T/J$'); axes[0].set_ylabel(r'$E/N$'); axes[0].set_title('(a) Energy per Majorana')
    axes[1].set_xlabel(r'$T/J$'); axes[1].set_ylabel(r'$S/N$'); axes[1].set_title('(b) Entropy per Majorana')
    axes[0].legend(frameon=True, fontsize=8); axes[1].legend(frameon=True, fontsize=8)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.close(fig)


def write_instance_tables(inst_rows):
    """Per-instance CSV and a LaTeX table of instance-to-instance ranges (no averaging)."""
    keys = ["N", "d", "instance", "seed", "kappa_1", "kappa_max", "chiral", "isomorphic_to",
            "dG_mid", "r", "E_beta", "S_beta"]
    with open(os.path.join(OUT_DIR, "instances.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in inst_rows:
            w.writerow([r.get(k, "") for k in keys])

    def rng(vals, fmt, math=False):
        # The range dash stays in text mode: inside $...$ "--" is two minus signs, not a dash.
        vals = [v for v in vals if v is not None]
        if not vals:
            return "--"
        wrap = (lambda x: f"${x}$") if math else (lambda x: x)
        return wrap(fmt % vals[0]) if len(vals) == 1 else f"{wrap(fmt % min(vals))} -- {wrap(fmt % max(vals))}"

    with open(os.path.join(OUT_DIR, "table_instances.tex"), "w") as f:
        f.write("% N & d & instances (distinct up to isomorphism) & kappa_1 range & "
                "G(beta/2)-G_SD(beta/2) range & <r> range & E/N at beta range (N = FIG_N only)\n")
        for (N, d) in sorted({(r["N"], r["d"]) for r in inst_rows}):
            rs = [r for r in inst_rows if r["N"] == N and r["d"] == d]
            distinct = sum(1 for r in rs if r["isomorphic_to"] in (None, ""))
            f.write(f"{N} & {d} & {len(rs)} ({distinct}) & {rng([r['kappa_1'] for r in rs], '%.4f')} & "
                    f"{rng([r['dG_mid'] for r in rs], '%+.5f', math=True)} & {rng([r['r'] for r in rs], '%.4f')} & "
                    f"{rng([r.get('E_beta') for r in rs], '%.4f', math=True)} \\\\\n")


def write_tables(rows, controls_summary=None):
    with open(os.path.join(OUT_DIR, "table_hypergraph_diagnostics.tex"), "w") as f:
        f.write("% N & d & |E| & Triangles & kappa_1 & kappa_max\n")
        for r in rows:
            g = r["diag"]
            f.write(f"{r['N']} & {r['d']} & {g['num_hyperedges']} & {g['triangles']} & "
                    f"{g['kappa_1']:.4f} & {g['kappa_max']:.4f} \\\\\n")
    with open(os.path.join(OUT_DIR, "table_level_statistics.tex"), "w") as f:
        f.write("% N & M (even-parity sector dim.) & class (N mod 8) & <r>_infty (large-N RMT) & "
                "<r> hSYK & d & monomial chiral symmetry\n")
        for r in rows:
            chi = r["diag"]["monomial_chiral_symmetry"] or "none"
            f.write(f"{r['N']} & {2 ** (r['N'] // 2 - 1)} & {r['class']} & {R_TARGET[r['class']]:.4f} & "
                    f"{r['r']:.4f} & {r['d']} & {chi} \\\\\n")
    with open(os.path.join(OUT_DIR, "table_two_point_deviation.tex"), "w") as f:
        f.write("% N & d & G_ED(beta/2) - G_SD(beta/2)\n")
        for r in rows:
            f.write(f"{r['N']} & {r['d']} & ${r['dG_mid']:+.5f}$ \\\\\n")
    with open(os.path.join(OUT_DIR, "table_search_statistics.tex"), "w") as f:
        f.write("% N & d & Attempts & Valid rate & kappa_1 pool (mean +- std) & kappa_1 best\n")
        for r in rows:
            s = r["search"]
            if not s:
                continue
            f.write(f"{r['N']} & {r['d']} & {s['attempts']} & {s['valid_rate']:.3f} & "
                    f"${s['kappa1_pool_mean']:.4f} \\pm {s['kappa1_pool_std']:.4f}$ & "
                    f"{s['kappa1_best']:.4f} \\\\\n")
    if controls_summary:
        with open(os.path.join(OUT_DIR, "table_controls.tex"), "w") as f:
            f.write(f"% model (N = {FIG_N}) & samples & G(beta/2) - G_SD(beta/2) [min, max] & <r> [min, max]"
                    f" & E/N at beta [min, max]\n")
            for r in rows:
                if r["N"] == FIG_N:
                    f.write(f"hSYK $d={r['d']}$ & 1 & ${r['dG_mid']:+.5f}$ & {r['r']:.4f} & -- \\\\\n")
            for key, c in controls_summary.items():
                Eb = c.get("E_beta_values")
                Estr = f"$[{min(Eb):.4f},\\,{max(Eb):.4f}]$" if Eb else "--"
                f.write(f"{key.replace('_', ' ')} & {c['n']} & "
                        f"$[{c['dG_mid_min']:+.5f},\\,{c['dG_mid_max']:+.5f}]$ & "
                        f"$[{c['r_min']:.4f},\\,{c['r_max']:.4f}]$ & {Estr} \\\\\n")


# =====================================================================
# 9. MAIN (full pipeline)
# =====================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if OTOC_CONFIGS:
        check_otoc_windows()                      # fail now, not after the ED stage
    tau_ed = np.linspace(0, BETA, N_TAU)
    tau_sd, G_sd = solve_sd(BETA, q=Q)
    mid_ed, mid_sd = len(tau_ed) // 2, len(tau_sd) // 2
    assert abs(tau_ed[mid_ed] - BETA / 2) < 1e-12 and abs(tau_sd[mid_sd] - BETA / 2) < 1e-12
    print(f"SD check: G_SD(0+) = {G_sd[0]:.4f} (expect +0.5000)")

    rows, fig_res, thermo_by_d, he_fig = [], {}, {}, {}
    inst_rows, other_dG, other_thermo = [], {}, {}
    k_beta = beta_index(THERMO_BETAS, BETA) if RUN_THERMO else None   # fails now, not after hours
    summary = {"config": {"q": Q, "beta": BETA, "n_instances": N_INSTANCES, **selection_settings()},
               "instances": {}}
    for N, d in RUN_CONFIGS:
        previous = []
        for m in range(N_INSTANCES):
            tag = instance_tag(m)
            print("\n" + "=" * 64 + f"\nN={N}, d={d}, instance {m}" +
                  (" (representative)" if m == 0 else "") + "\n" + "=" * 64)
            he, diag, search = get_instance(N, Q, d, m)
            iso = next((mp for mp, hp in previous if isomorphic_hypergraphs(N, he, hp, Q, d)), None)
            previous.append((m, he))
            print("  " + ", ".join(f"{k}={v}" for k, v in diag.items()))
            if iso is not None:
                print(f"  note: isomorphic to instance {iso} (a different labeling, hence possibly "
                      f"a different sign class; still a distinct Hamiltonian)")

            H, psi = build_hamiltonian(N, he, d, Q)
            G, evals, spec = diagonalize_and_G(psi, H, N, BETA, tau_ed)
            del H
            r, s = level_statistics(evals)
            cls = symmetry_class(N)
            dG_mid = float(G[mid_ed] - G_sd[mid_sd])
            print(f"  G_ED(0) = {G[0]:.4f};  G(beta/2) - G_SD(beta/2) = {dG_mid:+.5f};  "
                  f"<r> = {r:.4f}  ({cls}, large-N target {R_TARGET[cls]:.4f})")

            E_b = S_b = None
            if RUN_THERMO and N == FIG_N:
                E_th, S_th = ed_thermodynamics(spec, N, THERMO_BETAS)
                E_b, S_b = float(E_th[k_beta]), float(S_th[k_beta])
            inst_rows.append({"N": N, "d": d, "instance": m, "seed": search.get("seed", instance_seed(N, d, m)),
                              "kappa_1": diag["kappa_1"], "kappa_max": diag["kappa_max"],
                              "chiral": diag["monomial_chiral_symmetry"] or "none",
                              "isomorphic_to": iso, "dG_mid": dG_mid, "r": r, "E_beta": E_b, "S_beta": S_b})
            summary["instances"][f"N{N}_d{d}{tag}"] = {"diagnostics": diag, "search": search, "r_mean": r,
                                                         "class": cls, "dG_mid": dG_mid,
                                                         "isomorphic_to": iso}

            if m == 0:
                raw, smooth, plateau = spectral_form_factor(evals, BETA, SFF_T)
                np.savez_compressed(os.path.join(OUT_DIR, f"data_N{N}_d{d}.npz"),
                                    tau=tau_ed, G=G, tau_sd=tau_sd, G_sd=G_sd, evals=evals, spectrum=spec,
                                    s=s, r_mean=r, sff_t=SFF_T, sff_raw=raw, sff_smooth=smooth,
                                    plateau=plateau, hyperedges=np.array(he))
                rows.append({"N": N, "d": d, "diag": diag, "search": search, "r": r, "class": cls,
                             "dG_mid": dG_mid})
                if N == FIG_N:
                    fig_res[d] = {"G": G, "s": s, "r": r, "sff_raw": raw, "sff_smooth": smooth,
                                  "plateau": plateau}
                    he_fig[d] = he
                    if RUN_THERMO:
                        thermo_by_d[d] = (E_th, S_th)
            else:
                np.savez_compressed(os.path.join(OUT_DIR, f"data_N{N}_d{d}{tag}.npz"),
                                    tau=tau_ed, G=G, evals=evals, spectrum=spec, r_mean=r,
                                    hyperedges=np.array(he))
                if N == FIG_N:
                    other_dG.setdefault(d, []).append(dG_mid)
                    if RUN_THERMO:
                        other_thermo.setdefault(d, []).append((E_th, S_th))

    controls, controls_summary = None, None
    if any(v > 0 for v in CONTROL_SAMPLES.values()) and he_fig:
        print("\n" + "=" * 64 + f"\nControls at N={FIG_N}\n" + "=" * 64)
        controls = run_controls(FIG_N, Q, he_fig, BETA, tau_ed, tau_sd, G_sd, THERMO_BETAS)
        controls_summary = summarize_controls(controls, THERMO_BETAS if RUN_THERMO else None, BETA)
        summary["controls"] = controls_summary
        save_controls(os.path.join(OUT_DIR, f"controls_N{FIG_N}.npz"), controls, THERMO_BETAS)
        if RUN_THERMO and "dense" in controls_summary:
            c = controls_summary["dense"]
            print(f"  dense SYK at beta={c['beta']:g}: E/N = {c['E_beta_values']}, S/N = {c['S_beta_values']}")

    write_tables(rows, controls_summary)
    write_instance_tables(inst_rows)
    if fig_res:
        plot_two_point_spectral_figure(fig_res, tau_ed, tau_sd, G_sd, FIG_N,
                                       os.path.join(OUT_DIR, f"fig_two_point_spectral_N{FIG_N}.pdf"),
                                       controls, other_dG)
    if RUN_THERMO and thermo_by_d:
        print("\nSD thermodynamics ...")
        E_sd, S_sd = sd_thermodynamics(THERMO_BETAS, q=Q)
        plot_thermodynamics_figure(THERMO_BETAS, thermo_by_d, E_sd, S_sd, FIG_N,
                                   os.path.join(OUT_DIR, f"fig_thermodynamics_N{FIG_N}.pdf"), controls,
                                   other_thermo)
        np.savez_compressed(os.path.join(OUT_DIR, f"thermodynamics_N{FIG_N}.npz"), betas=THERMO_BETAS,
                            E_sd=E_sd, S_sd=S_sd,
                            **{f"E_d{d}": v[0] for d, v in thermo_by_d.items()},
                            **{f"S_d{d}": v[1] for d, v in thermo_by_d.items()})
        k = k_beta
        summary["thermodynamics_at_beta"] = {
            "beta": float(THERMO_BETAS[k]), "E_sd": float(E_sd[k]), "S_sd": float(S_sd[k]),
            **{f"E_d{d}": float(v[0][k]) for d, v in thermo_by_d.items()},
            **{f"S_d{d}": float(v[1][k]) for d, v in thermo_by_d.items()}}
        print("  " + ", ".join(f"{a}={b:.4f}" for a, b in summary["thermodynamics_at_beta"].items()))

    if OTOC_CONFIGS:
        print("\n" + "=" * 64 + "\nMode-resolved OTOC\n" + "=" * 64)
        if RUN_OTOC_VALIDATION:
            validate_otoc(q=Q, beta=BETA)
            if np.dtype(OTOC_DTYPE) != np.dtype(np.complex128):
                validate_otoc(q=Q, beta=BETA, dtype=OTOC_DTYPE, tol=1e-5)
        for N, d in OTOC_CONFIGS:
            per_instance = {}
            for m in range(max(1, OTOC_INSTANCES)):
                print(f"  N={N}, d={d}, instance {m}")
                he, _, _ = get_instance(N, Q, d, m)
                res, per_instance[m] = run_otoc_for_instance(N, d, Q, he, BETA, instance_tag(m))
                summary["instances"].setdefault(f"N{N}_d{d}{instance_tag(m)}", {})["otoc"] = res
            if len(per_instance) > 1:
                plot_otoc_rate_overlay(per_instance, Q, BETA, N, d,
                                       os.path.join(OUT_DIR, f"fig_otoc_rates_instances_N{N}_d{d}.pdf"))

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll outputs written to '{OUT_DIR}/'.")



# =====================================================================
# 10. POSTPROCESS (uses saved data of a completed full run)
# =====================================================================

def sample_rmt_spectrum(cls, n_levels, rng):
    """One spectrum with n_levels distinct levels from GOE, GUE or GSE (Kramers doublets removed)."""
    n = n_levels
    if cls == "GOE":
        X = rng.normal(size=(n, n))
        return np.linalg.eigvalsh((X + X.T) / 2)
    if cls == "GUE":
        X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return np.linalg.eigvalsh((X + X.conj().T) / 2)
    # GSE: H = [[A, B], [-B*, A*]] with A Hermitian, B complex antisymmetric (doubly degenerate)
    X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    A = (X + X.conj().T) / 2
    Y = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    B = (Y - Y.T) / 2
    H = np.block([[A, B], [-B.conj(), A.conj()]])
    ev = np.linalg.eigvalsh(H)
    return ev[::2]


def rmt_calibration(Ns, n_samples, seed):
    """Finite-size <r> for the parity-sector dimension of each N, same protocol as hSYK
    (one sector, Kramers doublets removed, central 80%)."""
    rng = np.random.default_rng(seed)
    out = {}
    for N in sorted(set(Ns)):
        cls = symmetry_class(N)
        M = 2 ** (N // 2 - 1)
        n_lev = M // 2 if cls == "GSE" else M
        t0 = time.time()
        rs = np.array([gap_ratio(sample_rmt_spectrum(cls, n_lev, rng)) for _ in range(n_samples)])
        out[N] = {"class": cls, "sector_dim": M, "levels": n_lev, "n_samples": n_samples,
                  "r_mean": float(rs.mean()), "r_std": float(rs.std(ddof=1)),
                  "r_mean_stderr": float(rs.std(ddof=1) / np.sqrt(n_samples))}
        print(f"  RMT {cls}, N={N} ({n_lev} levels): <r> = {rs.mean():.4f} +- {rs.std(ddof=1):.4f} "
              f"({n_samples} samples, {time.time() - t0:.0f} s)")
    return out


def recover_dense_thermo(N, q, beta, tau, betas):
    """
    Regenerates the dense-SYK control realizations by replaying the RNG stream of
    run_controls (dense is drawn first from default_rng(CONTROL_SEED)), checks G(tau)
    against the saved controls, and returns E/N, S/N on betas for each realization.
    """
    n = CONTROL_SAMPLES.get("dense", 0)
    if n <= 0 or next(iter(CONTROL_SAMPLES)) != "dense":
        raise RuntimeError("Dense recovery requires 'dense' to be the first CONTROL_SAMPLES key.")
    saved_G = None
    p = os.path.join(OUT_DIR, f"controls_N{N}.npz")
    if os.path.exists(p):
        with np.load(p) as z:
            if "dense_G" in z.files:
                saved_G = z["dense_G"]
    rng = np.random.default_rng(CONTROL_SEED)
    Es, Ss, Gs, devs = [], [], [], []
    for smp in range(n):
        t0 = time.time()
        terms, cpl = dense_syk_terms(N, q, 1.0, rng)
        H, psi = build_hamiltonian_terms(N, terms, cpl, q)
        G, _, spec = diagonalize_and_G(psi, H, N, beta, tau)
        del H
        E, S = ed_thermodynamics(spec, N, betas)
        dev = None
        if saved_G is not None and smp < len(saved_G):
            dev = float(np.abs(G - saved_G[smp]).max())
            if dev > 1e-10:
                raise RuntimeError(f"Dense realization {smp} does not reproduce the saved G(tau) "
                                   f"(max dev {dev:.1e}); the RNG stream or settings differ.")
        Es.append(E); Ss.append(S); Gs.append(G); devs.append(dev)
        print(f"    dense sample {smp}: reproduces saved G(tau) to {dev if dev is None else f'{dev:.1e}'}; "
              f"{time.time() - t0:.0f} s")
    return np.array(Es), np.array(Ss), np.array(Gs), devs


def postprocess():
    os.makedirs(OUT_DIR, exist_ok=True)
    tau_ed = np.linspace(0, BETA, N_TAU)
    mid_ed = len(tau_ed) // 2
    k_beta = (beta_index(THERMO_BETAS, BETA)
              if RUN_DENSE_THERMO_RECOVERY and CONTROL_SAMPLES.get("dense", 0) > 0 else None)
    report = {"config": {"q": Q, "beta": BETA, "n_instances": N_INSTANCES, **selection_settings()}}

    # ---- 1. SD convergence ------------------------------------------------------------
    print("\n" + "=" * 64 + "\nSchwinger-Dyson convergence\n" + "=" * 64)
    tau_run, G_run = solve_sd(BETA, q=Q)                       # reference used by the full run
    G_mid_run = float(G_run[len(tau_run) // 2])
    _, G_fft = solve_sd_fft(BETA, q=Q, M=len(tau_run))
    fft_dev = float(np.abs(G_fft - G_run).max())
    print(f"  FFT solver vs dense solver at M={len(tau_run)}: max |dG| = {fft_dev:.1e}")
    conv = {}
    for M in SD_CONVERGENCE_M:
        t_, G_ = solve_sd_fft(BETA, q=Q, M=M)
        conv[M] = {"G_mid": float(G_[M // 2]), "G_0plus": float(G_[0])}
        print(f"  M={M:6d}: G_SD(beta/2) = {G_[M // 2]:.7f}, G_SD(0+) = {G_[0]:.5f}")
    M_fine = SD_CONVERGENCE_M[-1]
    _, G_long = solve_sd_fft(BETA, q=Q, M=M_fine, n_iter=1000)
    G_mid_fine = float(G_long[M_fine // 2])
    shift = G_mid_fine - G_mid_run
    print(f"  M={M_fine}, 1000 iterations: G_SD(beta/2) = {G_mid_fine:.7f}")
    print(f"  => Delta G values of the run shift by {-shift:+.2e} when re-referenced")
    report["sd_convergence"] = {"fft_vs_dense_max_dev": fft_dev, "G_mid_run": G_mid_run,
                                "by_M": {str(k): v for k, v in conv.items()},
                                "G_mid_fine_1000iter": G_mid_fine, "shift": shift}

    # ---- 2. RMT calibration -------------------------------------------------------------
    rmt = {}
    if RUN_RMT_CALIBRATION:
        print("\n" + "=" * 64 + "\nFinite-size RMT calibration\n" + "=" * 64)
        rmt = rmt_calibration([N for N, _ in RUN_CONFIGS], RMT_SAMPLES, RMT_SEED)
        with open(os.path.join(OUT_DIR, "rmt_calibration.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["N", "class", "sector_dim", "levels", "n_samples", "r_mean", "r_std", "r_mean_stderr"])
            for N, c in sorted(rmt.items()):
                w.writerow([N, c["class"], c["sector_dim"], c["levels"], c["n_samples"],
                            f"{c['r_mean']:.5f}", f"{c['r_std']:.5f}", f"{c['r_mean_stderr']:.5f}"])
        report["rmt_calibration"] = {str(k): v for k, v in rmt.items()}

    # ---- 3. Per-instance analysis from saved data ------------------------------------
    print("\n" + "=" * 64 + "\nPer-instance analysis of saved spectra\n" + "=" * 64)
    rows = []
    for N, d in RUN_CONFIGS:
        for m in range(N_INSTANCES):
            path = os.path.join(OUT_DIR, f"data_N{N}_d{d}{instance_tag(m)}.npz")
            if not os.path.exists(path):
                print(f"  missing {path}; skipped")
                continue
            with np.load(path) as D:
                G, spec, evals = D["G"], D["spectrum"], D["evals"]
            half = len(spec) // 2
            Ee, Eo = np.sort(spec[:half]), np.sort(spec[half:])
            scale = float(np.abs(spec).max())
            mirror_between = float(np.abs(Ee + Eo[::-1]).max()) / scale   # Ee = -Eo ?
            mirror_within = float(np.abs(Ee + Ee[::-1]).max()) / scale    # Ee = -Ee ?
            degenerate = float(np.abs(Ee - Eo).max()) / scale             # Ee = Eo ?
            r_run = gap_ratio(evals)
            r_pos = None
            if mirror_within < 1e-8 and N % 8 != 4:
                r_pos = gap_ratio(Ee[Ee > 0])
            row = {"N": N, "d": d, "instance": m,
                   "dG_mid_run": float(G[mid_ed] - G_mid_run),
                   "dG_mid_fineSD": float(G[mid_ed] - G_mid_fine),
                   "mirror_even_odd": mirror_between, "mirror_within_sector": mirror_within,
                   "degenerate_even_odd": degenerate,
                   "r": r_run, "r_positive_half": r_pos}
            if N in rmt:
                row["z"] = (r_run - rmt[N]["r_mean"]) / rmt[N]["r_std"]
                if r_pos is not None:
                    row["z_positive_half"] = (r_pos - rmt[N]["r_mean"]) / rmt[N]["r_std"]
            if N == FIG_N:
                E, S = ed_thermodynamics(spec, N, [BETA])
                row["E_beta"], row["S_beta"] = float(E[0]), float(S[0])
            rows.append(row)
            flags = []
            if mirror_between < 1e-8: flags.append("even = -odd")
            if mirror_within < 1e-8: flags.append("sector symmetric under E->-E")
            if degenerate < 1e-8: flags.append("even = odd")
            z_str = f", z={row['z']:+.2f}" if "z" in row else ""
            rp = f", <r>(E>0)={r_pos:.4f}" if r_pos is not None else ""
            print(f"  N={N} d={d} i{m}: dG(fine SD)={row['dG_mid_fineSD']:+.5f}, <r>={r_run:.4f}{z_str}{rp}"
                  f"  [{'; '.join(flags) or 'no exact sector relations'}]")

    keys = ["N", "d", "instance", "dG_mid_run", "dG_mid_fineSD", "r", "z", "r_positive_half",
            "z_positive_half", "mirror_even_odd", "mirror_within_sector", "degenerate_even_odd",
            "E_beta", "S_beta"]
    with open(os.path.join(OUT_DIR, "postprocess_instances.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow(["" if r.get(k) is None else r.get(k) for k in keys])
    report["instances"] = rows

    # instance ranges of thermodynamics at beta (N = FIG_N)
    thermo_ranges = {}
    for d in sorted({r["d"] for r in rows if r["N"] == FIG_N}):
        Eb = [r["E_beta"] for r in rows if r["N"] == FIG_N and r["d"] == d]
        Sb = [r["S_beta"] for r in rows if r["N"] == FIG_N and r["d"] == d]
        thermo_ranges[d] = {"E_min": min(Eb), "E_max": max(Eb), "S_min": min(Sb), "S_max": max(Sb)}
        print(f"  N={FIG_N} d={d}: E/N in [{min(Eb):.4f}, {max(Eb):.4f}], S/N in [{min(Sb):.4f}, {max(Sb):.4f}]"
              f"  (beta={BETA:g}, {len(Eb)} instances)")
    report["thermo_instance_ranges"] = {str(k): v for k, v in thermo_ranges.items()}

    # ---- 4. Dense-SYK thermodynamics --------------------------------------------------
    if RUN_DENSE_THERMO_RECOVERY and CONTROL_SAMPLES.get("dense", 0) > 0:
        print("\n" + "=" * 64 + f"\nDense-SYK thermodynamics at N={FIG_N}\n" + "=" * 64)
        Es, Ss, Gs, devs = recover_dense_thermo(FIG_N, Q, BETA, tau_ed, THERMO_BETAS)
        np.savez_compressed(os.path.join(OUT_DIR, f"controls_thermo_N{FIG_N}.npz"),
                            betas=THERMO_BETAS, dense_E=Es, dense_S=Ss, dense_G=Gs)
        Eb, Sb = Es[:, k_beta], Ss[:, k_beta]
        dGd = Gs[:, mid_ed] - G_mid_fine
        print(f"  beta={THERMO_BETAS[k_beta]:g}: E/N = {np.round(Eb, 5).tolist()}, "
              f"S/N = {np.round(Sb, 5).tolist()}")
        with open(os.path.join(OUT_DIR, f"dense_thermo_N{FIG_N}.tex"), "w") as f:
            f.write(f"% dense SYK, N = {FIG_N}, beta = {THERMO_BETAS[k_beta]:g}, "
                    f"{len(Eb)} realizations: [min, max]\n")
            f.write(f"E/N & $[{Eb.min():.4f},\\,{Eb.max():.4f}]$ \\\\\n")
            f.write(f"S/N & $[{Sb.min():.4f},\\,{Sb.max():.4f}]$ \\\\\n")
            f.write(f"G(beta/2)-G_SD(beta/2) (fine SD) & $[{dGd.min():+.5f},\\,{dGd.max():+.5f}]$ \\\\\n")
        report["dense_thermo"] = {"beta": float(THERMO_BETAS[k_beta]), "E_beta": Eb.tolist(),
                                  "S_beta": Sb.tolist(), "dG_mid_fineSD": dGd.tolist(),
                                  "G_reproduction_max_dev": devs}

    with open(os.path.join(OUT_DIR, "postprocess_summary.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nPostprocess outputs written to '{OUT_DIR}/'.")



def otoc_reanalysis():
    """Refits and redraws every mode-resolved OTOC from saved all-pairs data (minutes)."""
    if OTOC_CONFIGS:
        check_otoc_windows()
    out = {}
    for N, d in OTOC_CONFIGS:
        for m in range(max(1, OTOC_INSTANCES)):
            he, _, _ = get_instance(N, Q, d, m, protect_saved=True)
            print(f"  N={N}, d={d}, instance {m}")
            res, _ = run_otoc_for_instance(N, d, Q, he, BETA, instance_tag(m), reuse_only=True)
            out[f"N{N}_d{d}{instance_tag(m)}"] = res
    with open(os.path.join(OUT_DIR, "otoc_reanalysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_DIR}/otoc_reanalysis.json and redrew the OTOC figures.")


RUN_MODES = ("full", "postprocess", "otoc_reanalysis", "none")


def mode_from_argv(default, modes, script, argv=None, in_kernel=None):
    """
    Run mode from the command line:  python <script> [mode].
    No argument: the CONFIG value `default`. One known mode name: that mode. -h / --help:
    print the modes and exit. Anything else (unknown mode, a flag, extra arguments): exit
    with an error, so that a typo can never start a different computation.
    In a notebook cell, sys.argv holds the kernel's own flags ('-f <kernel.json>'), so it is
    ignored and the CONFIG value is used; with  %run <script> <mode>  sys.argv[0] is the
    script itself, and the mode is honoured.
    """
    import sys
    if in_kernel is None:
        argv0 = os.path.basename(sys.argv[0]) if sys.argv else ""
        in_kernel = argv is None and "ipykernel" in sys.modules and argv0 != script
    args = [] if in_kernel else list(sys.argv[1:] if argv is None else argv)
    usage = f"usage: python {script} [{' | '.join(modes)}]   (no argument: RUN_MODE = {default!r} from CONFIG)"
    if not args:
        return default
    if args[0] in ("-h", "--help"):
        print(usage)
        raise SystemExit(0)
    if len(args) > 1 or args[0] not in modes:
        raise SystemExit(f"error: unexpected argument(s) {' '.join(args)!r}\n{usage}")
    return args[0]


if __name__ == "__main__":
    RUN_MODE = mode_from_argv(RUN_MODE, RUN_MODES, "hsyk_numerics.py")
    if RUN_MODE == "postprocess":
        postprocess()
    elif RUN_MODE == "otoc_reanalysis":
        otoc_reanalysis()
    elif RUN_MODE == "full":
        main()
    elif RUN_MODE != "none":
        raise ValueError(f"Unknown RUN_MODE {RUN_MODE!r}")
