# -*- coding: utf-8 -*-
"""
hsyk_supplement.py
==================
Supplementary computations for "Deterministic Hypergraph SYK" (S. Prakash,
arXiv:2609.19255) that are not part of the main pipeline hsyk_numerics.py.
All hSYK instances are taken from hsyk_numerics (saved instance files are
reused; missing ones are regenerated deterministically with the same seeds).

Modes (python hsyk_supplement.py <mode>):

  check_data        Lists which files of data/ed/ and data/constructability/ are
                    present or missing. Nothing is computed.
  wmin              Minimum weight w_min of ker_F2(I) for every saved instance
                    (sign classes). Output: wmin.csv.
                    Seconds.
  resources         Quantum-simulation resources: Jordan-Wigner Pauli weights, CNOTs per Trotter
                    step for the original and an optimized site labeling,
                    anticommuting pairs, exact minimum number of commuting
                    (site-disjoint) layers, and the dense-SYK comparison.
                    Output: resources.json, table_simulation.tex. ~10-20 min.
  sparse_controls   Controls: random d-regular hypergraphs WITHOUT linearity,
                    with the same symmetry filters and kappa_1 selection as hSYK,
                    each evaluated with uniform, binary (+-J/sqrt d) and Gaussian
                    couplings. Output: controls_sparse_N{N}.json/.npz,
                    table_controls_sparse.tex. ~30-60 min at N = 24.
  all               wmin, resources, sparse_controls.

Seeds: labeling optimization 0, sparse controls 8.
"""

import os
import csv
import glob
import json
import time
import itertools
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from math import comb, pi

import hsyk_numerics as hn

# =====================================================================
# CONFIG
# =====================================================================

RUN_MODE = "all"
OUT_DIR = hn.OUT_DIR
Q = hn.Q
BETA = hn.BETA
RESOURCE_CONFIGS = hn.RUN_CONFIGS          # representative instances for the resource estimates
SPARSE_N = hn.FIG_N                        # N for the sparse controls
SPARSE_DS = sorted({d for (N, d) in hn.RUN_CONFIGS if N == hn.FIG_N})
SPARSE_SAMPLES = 5                         # random hypergraphs per d
SPARSE_SEED = 8
SPARSE_SELECT = True                       # smallest kappa_1 among POOL_SIZE filtered candidates
ANNEAL_ITERS = 40000
ANNEAL_SEED = 0
CHROMATIC_TIME_LIMIT = 300                 # seconds per k in the exact coloring search
WMIN_MAX_KERNEL_DIM = 22                   # enumerate 2^dim codewords up to this dimension


# =====================================================================
# 1. Sign classes: minimum weight of ker_F2(I)
# =====================================================================

def f2_kernel(A):
    """Basis (rows) of ker(A) over F2, A an (m x n) 0/1 matrix."""
    A = (np.asarray(A) % 2).astype(np.uint8).copy()
    m, n = A.shape
    pivots, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, m) if A[i, c]), None)
        if p is None:
            continue
        A[[r, p]] = A[[p, r]]
        for i in range(m):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        pivots.append(c)
        r += 1
        if r == m:
            break
    basis = []
    for f in (c for c in range(n) if c not in pivots):
        x = np.zeros(n, dtype=np.uint8)
        x[f] = 1
        for i, c in enumerate(pivots):
            x[c] = A[i, f]
        basis.append(x)
    return np.array(basis, dtype=np.uint8).reshape(len(basis), n)


def min_codeword_weight(N, hyperedges, max_dim=WMIN_MAX_KERNEL_DIM, chunk=2 ** 16):
    """(dim ker_F2(I), w_min, number of minimum-weight codewords); w_min = None if the
    kernel is too large to enumerate. Codewords are sets of hyperedges covering every
    site an even number of times. The 2^k - 1 nonzero codewords are enumerated in blocks
    of `chunk`, so memory is O(chunk |E|) instead of O(2^k |E|) (~GBs at k = 22)."""
    B = f2_kernel(hn.incidence_matrix(N, hyperedges).astype(int)).astype(np.int64)
    k = len(B)
    if k == 0:
        return 0, None, 0
    if k > max_dim:
        return k, None, None
    bits = np.arange(k)
    wmin, count = None, 0
    for start in range(1, 2 ** k, chunk):
        idx = np.arange(start, min(start + chunk, 2 ** k))
        w = (((idx[:, None] >> bits) & 1) @ B % 2).sum(axis=1)
        m = int(w.min())
        if wmin is None or m < wmin:
            wmin, count = m, 0
        if m == wmin:
            count += int((w == m).sum())
    return k, wmin, count


def run_wmin():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "hsyk_N*_d*.json")))
    if not files:
        print("No saved instances in", OUT_DIR, "- run hsyk_numerics.py full first (or copy its outputs).")
        return []
    rows = []
    for path in files:
        data = json.load(open(path))
        N, d, he = data["N"], data["d"], data["hyperedges"]
        k, wmin, nmin = min_codeword_weight(N, he)
        name = os.path.basename(path)
        rows.append({"file": name, "N": N, "d": d, "num_hyperedges": len(he), "dim_ker_F2": k,
                     "sign_classes_log2": k, "w_min": wmin, "n_min_weight_codewords": nmin})
        print(f"  {name}: dim ker = {k}, w_min = {wmin} ({nmin} codewords)")
    with open(os.path.join(OUT_DIR, "wmin.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_DIR}/wmin.csv")
    return rows


# =====================================================================
# 2. Circuit resources for quantum simulation
# =====================================================================

def majorana_pauli(N):
    """Symplectic (x, z) bits of the Jordan-Wigner Majoranas of hsyk_numerics.build_majoranas:
    chi_{2k} ~ Z^{(x)k} X_k, chi_{2k+1} ~ Z^{(x)k} Y_k."""
    L = N // 2
    out = []
    for k in range(L):
        for typ in ("X", "Y"):
            x = np.zeros(L, dtype=np.uint8)
            z = np.zeros(L, dtype=np.uint8)
            z[:k] = 1
            x[k] = 1
            if typ == "Y":
                z[k] = 1
            out.append((x, z))
    return out


def jw_weight(sites, P):
    """Pauli weight (number of qubits acted on nontrivially) of prod_{k in sites} chi_k."""
    x = np.zeros_like(P[0][0])
    z = np.zeros_like(P[0][1])
    for s in sites:
        x ^= P[s][0]
        z ^= P[s][1]
    return int(np.sum(x | z))


def term_weights(hyperedges, perm, P):
    """Weights after relabeling site v -> perm[v]."""
    return [jw_weight([perm[v] for v in e], P) for e in hyperedges]


def optimize_labeling(hyperedges, N, P, iters=ANNEAL_ITERS, seed=ANNEAL_SEED):
    """Simulated annealing over site permutations minimizing sum_e w_e. Justified by the paper's
    sign-class lemma: a relabeling changes at most the sign class."""
    rng = np.random.default_rng(seed)
    perm = np.arange(N)
    cur = sum(term_weights(hyperedges, perm, P))
    best, bestp = cur, perm.copy()
    T0 = 2.0
    for it in range(iters):
        T = T0 * (1 - it / iters) + 1e-3
        a, b = rng.choice(N, 2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
        new = sum(term_weights(hyperedges, perm, P))
        if new <= cur or rng.random() < np.exp((cur - new) / T):
            cur = new
            if cur < best:
                best, bestp = cur, perm.copy()
        else:
            perm[a], perm[b] = perm[b], perm[a]
    return bestp


def _heuristic_coloring(G, trials=3000, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for strat in ["largest_first", "DSATUR", "smallest_last", "independent_set", "connected_sequential_bfs"]:
        c = nx.greedy_color(G, strategy=strat)
        k = max(c.values()) + 1
        if best is None or k < best:
            best = k
    nodes = list(G.nodes())
    for _ in range(trials):
        col = {}
        for v in rng.permutation(nodes):
            used = {col[u] for u in G[v] if u in col}
            c = 0
            while c in used:
                c += 1
            col[v] = c
        best = min(best, max(col.values()) + 1)
    return best


def _k_colorable(G, k, time_limit=CHROMATIC_TIME_LIMIT):
    """Exact backtracking (DSATUR order, symmetry breaking). True / False / None (timeout)."""
    nodes = list(G.nodes())
    adj = {v: set(G[v]) for v in nodes}
    col, t0 = {}, time.time()

    def select():
        best, key = None, (-1, -1)
        for v in nodes:
            if v in col:
                continue
            kv = (len({col[u] for u in adj[v] if u in col}), len(adj[v]))
            if kv > key:
                key, best = kv, v
        return best

    def bt():
        if time.time() - t0 > time_limit:
            raise TimeoutError
        v = select()
        if v is None:
            return True
        used = {col[u] for u in adj[v] if u in col}
        top = max(col.values()) + 1 if col else 0
        for c in range(min(k, top + 1)):
            if c in used:
                continue
            col[v] = c
            if bt():
                return True
            del col[v]
        return False

    try:
        return bt()
    except TimeoutError:
        return None


def chromatic_number(G, lower=1):
    """Exact chromatic number if the search finishes, else (lower bound, upper bound)."""
    upper = _heuristic_coloring(G)
    for k in range(max(lower, 1), upper):
        r = _k_colorable(G, k)
        if r:
            return k
        if r is None:
            return (k, upper)
    return upper


def anticommuting_pairs(hyperedges):
    n = 0
    for a in range(len(hyperedges)):
        sa = set(hyperedges[a])
        for b in range(a + 1, len(hyperedges)):
            n += len(sa & set(hyperedges[b])) % 2
    return n


def run_resources():
    rows = []
    for N, d in RESOURCE_CONFIGS:
        t0 = time.time()
        he, _, _ = hn.get_instance(N, Q, d, 0, protect_saved=True)
        P = majorana_pauli(N)
        w0 = term_weights(he, np.arange(N), P)
        popt = optimize_labeling(he, N, P)
        w1 = term_weights(he, popt, P)
        layers = chromatic_number(hn.line_graph(he), lower=d)
        n_anti = anticommuting_pairs(he)
        assert n_anti == N * d * (d - 1) // 2
        row = {"N": N, "d": d, "num_terms": len(he), "anticommuting_pairs": n_anti,
               "w_mean": float(np.mean(w0)), "w_max": int(max(w0)), "cnot": int(sum(2 * (w - 1) for w in w0)),
               "w_mean_opt": float(np.mean(w1)), "w_max_opt": int(max(w1)),
               "cnot_opt": int(sum(2 * (w - 1) for w in w1)),
               "layers": layers if isinstance(layers, int) else list(layers),
               "layers_lower_bound": d, "layers_greedy_bound": Q * (d - 1) + 1,
               "trotter_commutator_sum": N * (d - 1) / 2 ** Q,
               "optimized_labeling": [int(x) for x in popt]}
        rows.append(row)
        print(f"  N={N} d={d}: CNOT {row['cnot']} -> {row['cnot_opt']}, layers {row['layers']}, "
              f"anti {n_anti} ({time.time() - t0:.0f} s)")

    # dense SYK at the largest N
    Nd = max(N for N, _ in RESOURCE_CONFIGS)
    P = majorana_pauli(Nd)
    ws = [jw_weight(A, P) for A in itertools.combinations(range(Nd), Q)]
    per_term = sum(comb(Q, o) * comb(Nd - Q, Q - o) for o in range(1, Q, 2))
    pairs = comb(Nd, Q) * per_term // 2
    sigma2 = 1.0 / comb(Nd - 1, Q - 1)
    dense_csum = pairs * 2 * (2 / pi) * sigma2 / 2 ** Q        # typical |J_A||J_B| = (2/pi) sigma^2
    dense = {"N": Nd, "num_terms": len(ws), "w_mean": float(np.mean(ws)), "w_max": int(max(ws)),
             "cnot": int(sum(2 * (w - 1) for w in ws)), "anticommuting_pairs": pairs,
             "trotter_commutator_sum_typical": dense_csum}
    print(f"  dense N={Nd}: {dense['num_terms']} terms, CNOT {dense['cnot']}, anti {pairs}, "
          f"commutator sum {dense_csum:.1f}")

    json.dump({"hsyk": rows, "dense": dense,
               "notes": "CNOT = sum_e 2(w_e - 1), all-to-all connectivity; commutator sums with J = 1"},
              open(os.path.join(OUT_DIR, "resources.json"), "w"), indent=2)
    with open(os.path.join(OUT_DIR, "table_simulation.tex"), "w") as f:
        f.write("% N & d & |E| & Anti. & w_mean & CNOT & w_mean(opt) & CNOT(opt) & Layers\n")
        for r in rows:
            lay = r["layers"] if isinstance(r["layers"], int) else f"{r['layers'][0]}--{r['layers'][1]}"
            f.write(f"{r['N']} & {r['d']} & {r['num_terms']} & {r['anticommuting_pairs']} & {r['w_mean']:.2f} & "
                    f"{r['cnot']} & {r['w_mean_opt']:.2f} & {r['cnot_opt']} & {lay} \\\\\n")
        f.write(f"% dense N={Nd}: {dense['num_terms']} terms, mean weight {dense['w_mean']:.2f}, "
                f"{dense['cnot']} CNOT per step, {pairs} anticommuting pairs\n")
    print(f"Wrote {OUT_DIR}/resources.json and table_simulation.tex")
    return rows, dense


# =====================================================================
# 3. Random non-linear sparse controls
# =====================================================================

def random_regular_hypergraph(N, q, d, rng, max_tries=500000):
    """q-uniform, d-regular hypergraph from the configuration model, rejecting hyperedges
    with a repeated site and repeated hyperedges. Linearity is NOT imposed."""
    if (N * d) % q:   # otherwise the last "hyperedge" is short and every try is rejected
        raise ValueError(f"A {q}-uniform {d}-regular hypergraph needs q | N d (N = {N}, d = {d}).")
    for _ in range(max_tries):
        stubs = np.repeat(np.arange(N), d)
        rng.shuffle(stubs)
        E = [tuple(sorted(int(x) for x in stubs[i:i + q])) for i in range(0, len(stubs), q)]
        if all(len(set(e)) == q for e in E) and len(set(E)) == len(E):
            return [list(e) for e in E]
    raise RuntimeError(f"Could not sample a simple {q}-uniform {d}-regular hypergraph on {N} sites.")


def has_trivial_levi_automorphism(N, hyperedges):
    """|Aut| == 1 for the Levi graph with its bipartition fixed (valid without linearity)."""
    B = hn.levi_graph(N, hyperedges)
    count = 0
    for _ in isomorphism.GraphMatcher(B, B, node_match=lambda a, b: a["kind"] == b["kind"]).isomorphisms_iter():
        count += 1
        if count > 1:
            return False
    return count == 1


def overlap_counts(hyperedges):
    c = {1: 0, 2: 0, 3: 0}
    for a in range(len(hyperedges)):
        sa = set(hyperedges[a])
        for b in range(a + 1, len(hyperedges)):
            o = len(sa & set(hyperedges[b]))
            if o in c:
                c[o] += 1
    return c


def sparse_control_hypergraph(N, q, d, rng, max_candidates=5000):
    """Random non-linear hypergraph passing the hSYK symmetry filters; with SPARSE_SELECT,
    the one with smallest kappa_1 among POOL_SIZE such candidates."""
    pool = []
    for _ in range(max_candidates):
        he = random_regular_hypergraph(N, q, d, rng)
        if hn.f2_rank(hn.incidence_matrix(N, he)) != N - 1:
            continue
        if hn.FILTER_CHIRAL and hn.monomial_chiral_symmetry(N, he) == "even":
            continue
        if not has_trivial_levi_automorphism(N, he):
            continue
        if not SPARSE_SELECT:
            return he
        pool.append((float(hn.site_modes(N, he, q, d)[0][1]), he))
        if len(pool) >= hn.POOL_SIZE:
            break
    if not pool:
        raise RuntimeError(f"No filtered random sparse hypergraph for N={N}, d={d}.")
    return min(pool, key=lambda p: p[0])[1]


def run_sparse_controls():
    N = SPARSE_N
    tau = np.linspace(0, BETA, hn.N_TAU)
    tau_sd, G_sd = hn.solve_sd(BETA, q=Q)
    mid_ed, mid_sd = len(tau) // 2, len(tau_sd) // 2
    k_beta = hn.beta_index(hn.THERMO_BETAS, BETA)
    rng = np.random.default_rng(SPARSE_SEED)

    def evaluate(terms, cpl):
        H, psi = hn.build_hamiltonian_terms(N, terms, cpl, Q)
        G, evals, spec = hn.diagonalize_and_G(psi, H, N, BETA, tau)
        del H
        r, _ = hn.level_statistics(evals)
        E, S = hn.ed_thermodynamics(spec, N, hn.THERMO_BETAS)
        return G, {"dG_mid": float(G[mid_ed] - G_sd[mid_sd]), "r": float(r),
                   "E_beta": float(E[k_beta]), "S_beta": float(S[k_beta])}

    out, arrays = {"N": N, "beta": BETA, "seed": SPARSE_SEED, "select": SPARSE_SELECT,
                   "samples": SPARSE_SAMPLES, "hsyk_reference": {}, "controls": {}}, {}
    for d in SPARSE_DS:
        he0, _, _ = hn.get_instance(N, Q, d, 0, protect_saved=True)
        G0, ref = evaluate(he0, np.full(len(he0), 1.0 / np.sqrt(d)))
        ref["kappa_1"] = float(hn.site_modes(N, he0, Q, d)[0][1])
        out["hsyk_reference"][f"d{d}"] = ref
        print(f"  hSYK N={N} d={d}: dG = {ref['dG_mid']:+.5f}, <r> = {ref['r']:.4f}, kappa_1 = {ref['kappa_1']:.4f}")
        for coupling in ("uniform", "binary", "gaussian"):
            out["controls"][f"{coupling}_d{d}"] = []
        for smp in range(SPARSE_SAMPLES):
            t0 = time.time()
            he = sparse_control_hypergraph(N, Q, d, rng)
            kappa1 = float(hn.site_modes(N, he, Q, d)[0][1])
            ov = overlap_counts(he)
            signs = rng.choice([-1.0, 1.0], size=len(he))
            gauss = rng.normal(0.0, 1.0 / np.sqrt(d), size=len(he))
            for coupling, cpl in (("uniform", np.full(len(he), 1.0 / np.sqrt(d))),
                                  ("binary", signs / np.sqrt(d)),
                                  ("gaussian", gauss)):
                G, row = evaluate(he, cpl)
                row.update({"sample": smp, "kappa_1": kappa1, "overlap2": ov[2], "overlap3": ov[3],
                            "hyperedges": he})
                out["controls"][f"{coupling}_d{d}"].append(row)
                arrays[f"{coupling}_d{d}_s{smp}_G"] = G
                print(f"    d={d} sample {smp} {coupling:8s}: dG = {row['dG_mid']:+.5f}, <r> = {row['r']:.4f}, "
                      f"kappa_1 = {kappa1:.4f}, overlaps 2/3 = {ov[2]}/{ov[3]}")
            print(f"    ({time.time() - t0:.0f} s)")

    # summary and table
    summary = {}
    for key, rows in out["controls"].items():
        dG = np.array([r["dG_mid"] for r in rows])
        rr = np.array([r["r"] for r in rows])
        summary[key] = {"n": len(rows), "dG_mean": float(dG.mean()), "dG_std": float(dG.std()),
                        "dG_min": float(dG.min()), "dG_max": float(dG.max()),
                        "r_mean": float(rr.mean()), "r_std": float(rr.std()),
                        "kappa_1_mean": float(np.mean([r["kappa_1"] for r in rows])),
                        "overlap2_mean": float(np.mean([r["overlap2"] for r in rows]))}
    out["summary"] = summary
    json.dump(out, open(os.path.join(OUT_DIR, f"controls_sparse_N{N}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT_DIR, f"controls_sparse_N{N}.npz"), tau=tau, **arrays)
    with open(os.path.join(OUT_DIR, "table_controls_sparse.tex"), "w") as f:
        f.write(f"% N = {N}, beta = {BETA}: model & samples & dG mean +- std [min, max] & <r> mean +- std & "
                f"mean kappa_1 & mean pairs sharing 2 sites\n")
        for d in SPARSE_DS:
            ref = out["hsyk_reference"][f"d{d}"]
            f.write(f"hSYK $d={d}$ & 1 & ${ref['dG_mid']:+.5f}$ & {ref['r']:.4f} & {ref['kappa_1']:.3f} & 0 \\\\\n")
            for coupling in ("uniform", "binary", "gaussian"):
                s = summary[f"{coupling}_d{d}"]
                f.write(f"sparse {coupling} $d={d}$ & {s['n']} & ${s['dG_mean']:+.5f} \\pm {s['dG_std']:.5f}$ "
                        f"$[{s['dG_min']:+.4f}, {s['dG_max']:+.4f}]$ & ${s['r_mean']:.4f} \\pm {s['r_std']:.4f}$ & "
                        f"{s['kappa_1_mean']:.3f} & {s['overlap2_mean']:.1f} \\\\\n")
    print(f"Wrote {OUT_DIR}/controls_sparse_N{N}.json/.npz and table_controls_sparse.tex")
    return out


# =====================================================================
# 4. Data check
# =====================================================================

def expected_ed_files():
    files = []
    for N, d in hn.RUN_CONFIGS:
        for m in range(hn.N_INSTANCES):
            t = hn.instance_tag(m)
            files += [f"hsyk_N{N}_d{d}{t}.json", f"data_N{N}_d{d}{t}.npz"]
    files += ["instances.csv", "summary.json", "table_hypergraph_diagnostics.tex", "table_level_statistics.tex",
              "table_two_point_deviation.tex", "table_search_statistics.tex", "table_controls.tex",
              "table_instances.tex", f"controls_N{hn.FIG_N}.npz", f"thermodynamics_N{hn.FIG_N}.npz",
              f"fig_two_point_spectral_N{hn.FIG_N}.pdf", f"fig_thermodynamics_N{hn.FIG_N}.pdf"]
    for N, d in hn.OTOC_CONFIGS:
        files += [f"otoc_allpairs_N{N}_d{d}.npz", f"otoc_modes_N{N}_d{d}.npz",
                  f"otoc_mode_rates_N{N}_d{d}.csv", f"fig_otoc_modes_N{N}_d{d}.pdf"]
    files += ["otoc_reanalysis.json", "postprocess_instances.csv", "postprocess_summary.json",
              "rmt_calibration.csv", f"controls_thermo_N{hn.FIG_N}.npz", f"dense_thermo_N{hn.FIG_N}.tex",
              "dense_sweep.json", "fig_deltaG_vs_N.pdf",
              "wmin.csv", "resources.json", "table_simulation.tex",
              f"controls_sparse_N{SPARSE_N}.json", f"controls_sparse_N{SPARSE_N}.npz", "table_controls_sparse.tex"]
    return files


def run_check_data():
    missing = [f for f in expected_ed_files() if not os.path.exists(os.path.join(OUT_DIR, f))]
    present = len(expected_ed_files()) - len(missing)
    print(f"data/ed: {present} of {len(expected_ed_files())} expected files present.")
    for f in missing:
        print("  missing:", f)
    cdir = os.path.join(os.path.dirname(OUT_DIR), "constructability")
    cfiles = ["constructability_instances.csv", "constructability_balls.csv", "filter_pass_rates.csv",
              "constructability_summary.json", "fig_constructability.pdf"]
    cmiss = [f for f in cfiles if not os.path.exists(os.path.join(cdir, f))]
    print(f"data/constructability: {len(cfiles) - len(cmiss)} of {len(cfiles)} expected files present.")
    for f in cmiss:
        print("  missing:", f)
    return missing + cmiss


if __name__ == "__main__":
    RUN_MODE = hn.mode_from_argv(RUN_MODE, ("check_data", "wmin", "resources", "sparse_controls", "all", "none"),
                                 "hsyk_supplement.py")
    os.makedirs(OUT_DIR, exist_ok=True)
    if RUN_MODE == "check_data":
        run_check_data()
    elif RUN_MODE == "wmin":
        run_wmin()
    elif RUN_MODE == "resources":
        run_resources()
    elif RUN_MODE == "sparse_controls":
        run_sparse_controls()
    elif RUN_MODE == "all":
        run_wmin()
        run_resources()
        run_sparse_controls()
    elif RUN_MODE != "none":
        raise ValueError(f"Unknown RUN_MODE {RUN_MODE!r}")
