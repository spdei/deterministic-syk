# -*- coding: utf-8 -*-
"""
hsyk_selfcheck.py
=================
Independent checks of the hSYK numerics. Each test compares a function of
hsyk_numerics.py or hsyk_constructability.py with a brute-force calculation, an
exact identity, or a known analytic limit, and prints PASS or FAIL with the
size of the discrepancy.

Run from the code/ directory:  python3 hsyk_selfcheck.py
Runtime is about 2-3 minutes; nothing is written outside a temporary directory.
"""

import os
import re
import math
import itertools
import numpy as np
import scipy.sparse as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."


def load_script(name, as_main=False, **overrides):
    """Executes a script's definitions without running its pipeline; CONFIG values can be
    overridden. The script runs under its module name, not "__main__", so its command-line
    block (which reads sys.argv) is skipped; as_main=True is only for testing that block."""
    src = open(os.path.join(SCRIPT_DIR, name)).read()
    src = re.sub(r'^RUN_MODE = "[a-z_]+"', 'RUN_MODE = "none"', src, flags=re.M)
    src = re.sub(r'^LOAD_ONLY = False', 'LOAD_ONLY = True', src, flags=re.M)
    for k, v in overrides.items():
        src = re.sub(rf'^{k} = .*$', f'{k} = {v!r}', src, count=1, flags=re.M)
    ns = {"__name__": "__main__" if as_main else os.path.splitext(name)[0],
          "__file__": os.path.join(SCRIPT_DIR, name)}
    exec(compile(src, name, "exec"), ns)
    return ns


def raises(exc, fn, *args, **kwargs):
    """True iff fn(*args, **kwargs) raises exc."""
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    return False


RESULTS = []


def check(tag, desc, ok, detail=""):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: {desc}" + (f"  ({detail})" if detail else ""))


def popcount_parity(dim):
    return np.array([(-1) ** bin(i).count("1") for i in range(dim)])


def main():
    H_ = load_script("hsyk_numerics.py")
    q, J, beta = 4, 1.0, 5.0

    # ------------------------------------------------------------------ algebra
    print("\n--- Majoranas and Hamiltonian ---")
    psi = H_["build_majoranas"](10)
    err = max(abs((psi[a] @ psi[b] + psi[b] @ psi[a]).toarray() - (a == b) * np.eye(32)).max()
              for a in range(10) for b in range(10))
    check("C1", "Majoranas satisfy {chi_i, chi_j} = delta_ij", err < 1e-12, f"max error {err:.1e}")

    np.random.seed(3)
    N, d = 16, 4
    he = H_["generate_1overlap_hypergraph"](N, q, d)
    Hm, psi = H_["build_hamiltonian"](N, he, d, q)
    dim = 2 ** (N // 2)
    Hb = np.zeros((dim, dim), complex)                      # brute force: explicit matrix products
    for e in he:
        P = psi[e[0]]
        for k in e[1:]:
            P = P @ psi[k]
        Hb += (1j) ** (q * (q - 1) // 2) * (J / math.sqrt(d)) * P.toarray()
    check("C2", "signed-permutation Hamiltonian equals explicit product of Majorana matrices",
          np.abs(Hm - Hb).max() < 1e-12, f"max diff {np.abs(Hm - Hb).max():.1e}")
    check("C3", "H is Hermitian before symmetrization (the phase i^{q(q-1)/2} is right)",
          np.abs(Hb - Hb.conj().T).max() < 1e-12, f"max |H - H^dag| {np.abs(Hb - Hb.conj().T).max():.1e}")
    Pm = np.diag(popcount_parity(dim))
    check("C4", "H commutes with fermion parity (block-diagonal in popcount parity)",
          np.abs(Hm @ Pm - Pm @ Hm).max() < 1e-12)

    # ------------------------------------------------------------------ G(tau)
    print("\n--- Two-point function ---")
    tau = np.linspace(0, beta, 21)
    G, evals, spec = H_["diagonalize_and_G"](psi, Hm, N, beta, tau)
    check("C5", "G(0) = 1/2 exactly (since chi_i^2 = 1/2)", abs(G[0] - 0.5) < 1e-12, f"G(0) = {G[0]:.12f}")
    check("C6", "G(beta - tau) = G(tau) (Majorana KMS symmetry)",
          np.abs(G - G[::-1]).max() < 1e-10, f"max asymmetry {np.abs(G - G[::-1]).max():.1e}")
    E, U = np.linalg.eigh(Hm)                                 # brute force at one tau
    rho_w = np.exp(-beta * (E - E.min())); Z = rho_w.sum()
    t0 = tau[7]
    Gdir = 0.0
    for p in psi:
        c = U.conj().T @ p.toarray() @ U
        Gdir += np.real(np.sum(np.exp(-(beta - t0) * (E[:, None] - E.min()) - t0 * (E[None, :] - E.min()))
                               * np.abs(c) ** 2)) / Z
    Gdir /= N
    check("C7", "sector-resolved G(tau) equals the full-space spectral sum", abs(G[7] - Gdir) < 1e-12,
          f"{G[7]:.10f} vs {Gdir:.10f}")
    check("C8", "spectrum from parity sectors equals full diagonalization",
          np.abs(np.sort(spec) - E).max() < 1e-10)

    # ------------------------------------------------------------------ thermodynamics
    print("\n--- Thermodynamics ---")
    b_small = 1e-3
    Eed, Sed = H_["ed_thermodynamics"](spec, N, [b_small])
    Ehigh = -b_small * J ** 2 / (q * 2 ** q)
    check("C9", "high-T limit E/N = -beta J^2/(q 2^q) (exact second moment of H)",
          abs(Eed[0] - Ehigh) / abs(Ehigh) < 1e-2, f"ED {Eed[0]:.3e} vs {Ehigh:.3e}")
    check("C10", "S/N -> (1/2) ln 2 at infinite temperature", abs(Sed[0] - 0.5 * math.log(2)) < 1e-5,
          f"{Sed[0]:.6f} vs {0.5 * math.log(2):.6f}")
    Esd, Ssd = H_["sd_thermodynamics"](np.array([0.01, 0.02]), q=q)
    check("C11", "SD energy has the same high-T limit", abs(Esd[0] / (-0.01 / (q * 2 ** q)) - 1) < 2e-2,
          f"ratio {Esd[0] / (-0.01 / (q * 2 ** q)):.4f}")

    # ------------------------------------------------------------------ Schwinger-Dyson
    print("\n--- Schwinger-Dyson solver ---")
    t_, G0 = H_["solve_sd"](beta, q=q, J=1e-6)
    check("C12", "free limit J -> 0 gives G = 1/2", np.abs(G0[1:] - 0.5).max() < 1e-3,
          f"max |G - 1/2| {np.abs(G0[1:] - 0.5).max():.1e}")
    t_, Gd = H_["solve_sd"](beta, q=q)
    t_, Gf = H_["solve_sd_fft"](beta, q=q, M=2048)
    check("C13", "FFT solver equals dense solver", np.abs(Gd - Gf).max() < 1e-12, f"{np.abs(Gd - Gf).max():.1e}")
    t_, Gfine = H_["solve_sd_fft"](beta, q=q, M=32768)
    check("C14", "G_SD(beta/2) converged in the number of Matsubara frequencies",
          abs(Gd[1024] - Gfine[16384]) < 1e-7, f"shift {abs(Gd[1024] - Gfine[16384]):.1e}")
    M = len(Gd)
    inner = slice(M // 10, M - M // 10 + 1)          # away from the jump at tau = 0, beta
    asym_in = np.abs(Gd[1:] - Gd[1:][::-1])[inner].max()
    asym_all = np.abs(Gd[1:] - Gd[1:][::-1]).max()
    check("C15", "G_SD(beta - tau) = G_SD(tau) (requires the trapezoid rule in the Fourier transform)",
          asym_all < 1e-10, f"max asymmetry {asym_all:.1e}")
    # the endpoint artifact must not affect the thermodynamic integral
    b = beta
    tt, G1 = H_["solve_sd"](b, q=q)
    tt2, G2 = H_["solve_sd_fft"](b, q=q, M=32768)
    E1 = -(J ** 2 / q) * np.sum(G1 ** q) * b / len(G1)
    E2 = -(J ** 2 / q) * np.sum(G2 ** q) * b / len(G2)
    check("C15b", "SD energy at beta = 5 converged in M (2048 vs 32768)", abs(E1 - E2) < 1e-6,
          f"{E1:.7f} vs {E2:.7f}")

    # ------------------------------------------------------------------ F2 filters
    print("\n--- F2 filters (brute force over all subsets) ---")
    done, seed = 0, 20
    while done < 3 and seed < 200:
        seed += 1
        np.random.seed(seed)
        Nn = 16
        try:
            h = H_["generate_1overlap_hypergraph"](Nn, q, 4, max_attempts=200)
        except ValueError:
            continue
        trial = done
        done += 1
        I = H_["incidence_matrix"](Nn, h).astype(int)
        X = ((np.arange(1, 2 ** Nn)[:, None] >> np.arange(Nn)[None, :]) & 1)   # all non-empty subsets
        ov = (X @ I) % 2                                    # |e cap S| mod 2 for every hyperedge
        wt = X.sum(axis=1) % 2
        commuting = int((ov.sum(axis=1) == 0).sum())
        anti = ov.all(axis=1)
        anti_even, anti_odd = int((anti & (wt == 0)).sum()), int((anti & (wt == 1)).sum())
        rank = H_["f2_rank"](I)
        chi = H_["monomial_chiral_symmetry"](Nn, h)
        ok_rank = (commuting + 1) == 2 ** (Nn - rank)       # kernel size incl. the empty set
        brute_chi = None if anti_even + anti_odd == 0 else ("odd" if anti_odd and not anti_even else
                                                           "even" if anti_even and not anti_odd else "mixed")
        ok_chi = (chi == brute_chi) or (brute_chi == "mixed" and rank < Nn - 1)
        check("C16", f"F2 rank and chiral monomial vs enumeration of all 2^{Nn} monomials (instance {trial})",
              ok_rank and ok_chi, f"rank {rank}, chiral {chi}, brute {brute_chi}")

    # ------------------------------------------------------------------ site operator K
    print("\n--- Site-adjacency operator K ---")
    np.random.seed(5)
    h = H_["generate_1overlap_hypergraph"](20, q, 4)
    kappa, V = H_["site_modes"](20, h, q, 4)
    A = np.zeros((20, 20))
    for e in h:
        for a in e:
            for b in e:
                if a != b:
                    A[a, b] += 1
    kd = np.sort(np.linalg.eigvalsh(A / (4 * 3)))[::-1]
    check("C17", "site_modes eigenvalues equal those of the explicit normalized adjacency",
          np.abs(kappa - kd).max() < 1e-12)
    check("C18", "K has kappa_0 = 1 with uniform eigenvector, and kappa >= -1/(q-1) (Lemma)",
          abs(kappa[0] - 1) < 1e-12 and np.allclose(np.abs(V[:, 0]), 1 / np.sqrt(20)) and kappa[-1] > -1 / 3 - 1e-12,
          f"kappa_min = {kappa[-1]:.4f}")

    # ------------------------------------------------------------------ OTOC
    print("\n--- OTOC ---")
    dev = H_["validate_otoc"](q=q, beta=beta)
    check("C19", "fast all-pairs OTOC equals direct formula (complex128)", dev < 1e-8, f"{dev:.1e}")
    dev32 = H_["validate_otoc"](q=q, beta=beta, dtype=np.complex64, tol=1e-5)
    check("C20", "fast all-pairs OTOC in complex64 (used in production)", dev32 < 1e-5, f"{dev32:.1e}")
    ks = [0.3, 0.1, 0.0, -0.1, -0.243, -0.33]
    line = [(2 * np.pi / beta) * 0.5 * (1 - 3 * k) for k in ks]
    gen = []
    for k in ks:
        D = 1 / q
        if k > 0:
            gen.append(-H_["mode_growth_rate"](k, q, beta))
        elif k == 0:
            gen.append(2 * D * 2 * np.pi / beta)
        else:
            from scipy.optimize import brentq
            f = lambda x: H_["k_retarded"](x, q) - 1 / k
            gen.append(brentq(f, 2 * D + 1e-9, 2 * D + 1 - 1e-9) * 2 * np.pi / beta)
    check("C21", "retarded kernel reproduces the exact q=4 decay line on both sides of kappa = 0",
          np.allclose(gen, line, atol=1e-9), f"max diff {np.max(np.abs(np.array(gen) - line)):.1e}")
    kR = [H_["k_retarded"](hp, q) for hp in (-1.0, 0.0)]
    check("C22", "k_R(-1) = 1 (maximal chaos) and k_R(0) = q-1 (growth threshold)",
          abs(kR[0] - 1) < 1e-12 and abs(kR[1] - (q - 1)) < 1e-12, f"{kR}")
    # mode projection on synthetic data with known rates
    Nm = 8
    kap = np.array([1, 0.25, 0.1, 0.05, -0.05, -0.1, -0.2, -0.3])
    Vm = np.linalg.qr(np.column_stack([np.ones(Nm), np.random.default_rng(1).standard_normal((Nm, Nm - 1))]))[0]
    tt = np.linspace(0, 5, 26)
    Ft = np.array([np.exp(-H_["predicted_decay_rate"](k, q, beta) * tt) for k in kap]).T
    F = np.einsum("ia,ta,ja->tij", Vm, Ft, Vm)
    kg, _, off, rates, pred, res = H_["analyze_mode_otocs"](F, tt, kap, Vm, q, beta, (1.5, 3.5), verbose=False)
    check("C23", "mode projection + fit recovers planted decay rates exactly",
          np.nanmax(np.abs(rates[1:] - pred[1:])) < 1e-8 and res["offdiag_fraction_in_window"] < 1e-10,
          f"slope {res['slope_all_modes']:.4f} (expect {-3 * np.pi / beta:.4f})")

    # ------------------------------------------------------------------ level statistics
    print("\n--- Level statistics ---")
    rng = np.random.default_rng(0)
    rs = [H_["gap_ratio"](H_["sample_rmt_spectrum"]("GOE", 1000, rng)) for _ in range(20)]
    check("C24", "GOE sample gives <r> near the Atas et al. value 0.5307", abs(np.mean(rs) - 0.5307) < 0.01,
          f"{np.mean(rs):.4f}")
    rs = [H_["gap_ratio"](H_["sample_rmt_spectrum"]("GSE", 300, rng)) for _ in range(20)]
    check("C25", "GSE construction (Kramers-reduced) gives <r> near 0.6744", abs(np.mean(rs) - 0.6744) < 0.015,
          f"{np.mean(rs):.4f}")
    Hp = np.random.default_rng(2).poisson  # Poisson reference
    pois = np.cumsum(np.random.default_rng(3).exponential(size=5000))
    check("C26", "Poisson spectrum gives <r> near 2 ln 2 - 1 = 0.386", abs(H_["gap_ratio"](pois) - 0.3863) < 0.01,
          f"{H_['gap_ratio'](pois):.4f}")

    # ------------------------------------------------------------------ constructability
    print("\n--- Constructability script ---")
    C_ = load_script("hsyk_constructability.py")
    N2, d2 = 600, 6
    E2, _ = C_["generate_linear_hypergraph"](N2, d2, q, seed=11)
    pairs, okl = set(), True
    for row in E2.tolist():
        if len(set(row)) < q:
            okl = False
        for p in itertools.combinations(sorted(row), 2):
            okl &= p not in pairs
            pairs.add(p)
    check("C27", "generated hypergraph is linear and d-regular (brute force)",
          okl and np.all(np.bincount(E2.ravel(), minlength=N2) == d2))
    A2 = C_["site_adjacency"](E2, N2)
    ev = np.linalg.eigvalsh(A2.toarray() / (d2 * (q - 1)))
    k1, kmin, steps, _ = C_["kappa_extremes"](A2, d2, q)
    check("C28", "Lanczos kappa_1, kappa_min equal dense diagonalization",
          abs(k1 - ev[-2]) < 1e-6 and abs(kmin - ev[0]) < 1e-6, f"{k1:.6f}/{ev[-2]:.6f}, {kmin:.6f}/{ev[0]:.6f}")
    inc = C_["incidence_array"](E2, N2, d2)
    import networkx as nx
    B = nx.Graph()
    B.add_edges_from((v, N2 + e) for e, row in enumerate(E2.tolist()) for v in row)
    roots = np.arange(0, E2.shape[0], 37)
    ranks = C_["levi_ball_cycle_ranks"](E2, inc, roots, 4)
    bad = sum((nx.ego_graph(B, N2 + e0, radius=R).number_of_edges() - nx.ego_graph(B, N2 + e0, radius=R).number_of_nodes() + 1)
              != ranks[kk, R] for kk, e0 in enumerate(roots) for R in range(5))
    check("C29", "Levi-ball cycle ranks equal networkx subgraph counts", bad == 0, f"{bad} mismatches")
    tri = C_["nontrivial_site_triangles"](A2, d2, q)
    Gx = nx.Graph(A2)
    tri_bf = np.array([nx.triangles(Gx, v) for v in range(N2)]) - d2 * math.comb(q - 1, 2)
    check("C30", "non-trivial triangle counts equal networkx", np.array_equal(tri, tri_bf))
    r, c = C_["f2_rank_and_chiral"](E2, N2)
    r0 = H_["f2_rank"](H_["incidence_matrix"](N2, E2.tolist()))
    check("C31", "bit-packed F2 rank equals the reference implementation", r == r0, f"{r} vs {r0}")
    Ed = np.vstack([E2[:, :], E2 + N2])                      # two disjoint copies: has a symmetry
    cert, _ = C_["asymmetry_certificate"](Ed, C_["incidence_array"](Ed, 2 * N2, d2),
                                          C_["nontrivial_site_triangles"](C_["site_adjacency"](Ed, 2 * N2), d2, q))
    check("C32", "automorphism certificate refuses a hypergraph with a symmetry", cert is False)

    # ------------------------------------------------------------------ supplement
    print("\n--- Supplement: sign classes, circuit resources, sparse controls ---")
    import sys
    sys.path.insert(0, SCRIPT_DIR)
    import hsyk_supplement as S_
    rng = np.random.default_rng(5)

    # C33: sign-class lemma -- Tr H_s^n is independent of the signs for n < w_min
    np.random.seed(3)
    E3 = H_["generate_1overlap_hypergraph"](16, q, 4)
    k3, w3, _ = S_.min_codeword_weight(16, E3)
    nmax = min(w3 - 1, 8)
    H0, _ = H_["build_hamiltonian_terms"](16, E3, np.full(len(E3), 0.5), q)
    Hs, _ = H_["build_hamiltonian_terms"](16, E3, rng.choice([-0.5, 0.5], size=len(E3)), q)
    mom0 = [np.trace(np.linalg.matrix_power(H0, n)).real for n in range(1, nmax + 1)]
    moms = [np.trace(np.linalg.matrix_power(Hs, n)).real for n in range(1, nmax + 1)]
    dev = max(abs(a - b) / max(abs(a), 1e-30) for a, b in zip(mom0, moms) if abs(a) > 1e-12)
    check("C33", f"Tr H^n independent of the signs for n <= {nmax} < w_min = {w3} (sign-class lemma)", dev < 1e-10,
          f"max relative deviation {dev:.1e}")

    # C34: w_min by kernel enumeration equals brute force over all subsets of hyperedges
    ok34 = True
    for (N4, d4, seed4) in [(12, 3, 1), (16, 4, 2)]:
        E4 = S_.random_regular_hypergraph(N4, q, d4, np.random.default_rng(seed4))
        I4 = H_["incidence_matrix"](N4, E4).astype(np.int64)
        subsets = ((np.arange(1, 2 ** len(E4))[:, None] >> np.arange(len(E4))) & 1).astype(np.int64)
        even = np.all((subsets @ I4.T) % 2 == 0, axis=1)
        bf = int(subsets[even].sum(axis=1).min()) if even.any() else None
        ok34 &= S_.min_codeword_weight(N4, E4)[1] == bf
    check("C34", "w_min by kernel enumeration equals brute force over all hyperedge subsets", ok34)

    # C35: random non-linear hypergraphs are simple, q-uniform, d-regular; Levi automorphism test works
    E5 = S_.random_regular_hypergraph(24, q, 6, np.random.default_rng(7))
    ok5 = (all(len(set(e)) == q for e in E5) and len({tuple(e) for e in E5}) == len(E5)
           and np.all(np.bincount(np.ravel(E5), minlength=24) == 6))
    sym = [[0, 1, 2, 3], [4, 5, 6, 7]]                       # swapping sites 0 and 1 is a symmetry
    check("C35", "random sparse hypergraph is simple and regular; Levi automorphism test detects a symmetry",
          ok5 and S_.has_trivial_levi_automorphism(8, sym) is False)

    # C36: Jordan-Wigner Pauli weight equals the number of qubits acted on, from explicit matrices
    N6 = 8
    psi6 = [p.toarray() for p in H_["build_majoranas"](N6)]
    P6 = S_.majorana_pauli(N6)
    X = np.array([[0, 1], [1, 0]]); Z = np.array([[1, 0], [0, -1]]); I2 = np.eye(2)

    def on_qubit(op, k):
        mats = [I2] * (N6 // 2)
        mats[k] = op
        out = mats[0]
        for m_ in mats[1:]:
            out = np.kron(out, m_)
        return out

    bad6 = 0
    for A in itertools.combinations(range(N6), q):
        M = psi6[A[0]] @ psi6[A[1]] @ psi6[A[2]] @ psi6[A[3]]
        w_explicit = sum(1 for k in range(N6 // 2)
                         if not (np.allclose(M @ on_qubit(X, k), on_qubit(X, k) @ M)
                                 and np.allclose(M @ on_qubit(Z, k), on_qubit(Z, k) @ M)))
        bad6 += w_explicit != S_.jw_weight(A, P6)
    check("C36", "Jordan-Wigner Pauli weights equal explicit operator support (all 70 terms, N = 8)", bad6 == 0,
          f"{bad6} mismatches")

    # C37: exact chromatic number equals brute force on small random graphs
    import networkx as nx
    ok7, detail7 = True, []
    for seed7 in (1, 2, 3):
        G7 = nx.gnp_random_graph(9, 0.4, seed=seed7)
        nodes, edges = list(G7.nodes()), np.array(list(G7.edges()))
        bf7 = None
        for k in range(1, 10):
            cols = ((np.arange(k ** 9)[:, None] // k ** np.arange(9)) % k)
            if edges.size == 0 or np.any(np.all(cols[:, edges[:, 0]] != cols[:, edges[:, 1]], axis=1)):
                bf7 = k
                break
        ex7 = S_.chromatic_number(G7)
        ok7 &= ex7 == bf7
        detail7.append(f"{ex7}/{bf7}")
    check("C37", "exact chromatic number equals brute force on random graphs", ok7, ", ".join(detail7))

    # ------------------------------------------------------------------ guards
    print("\n--- Guards: regularity, beta grid, saved-data protection, command line, output ---")
    import sys
    import json
    import tempfile

    # C38: q must divide N d (otherwise a non-regular hypergraph used to be returned silently)
    ok38 = (raises(ValueError, H_["generate_1overlap_hypergraph"], 26, q, 3)
            and raises(ValueError, H_["select_instance"], 26, q, 3, max_calls=2)
            and raises(ValueError, S_.random_regular_hypergraph, 26, q, 3, np.random.default_rng(0)))
    check("C38", "generators refuse q not dividing N d (N = 26, d = 3) instead of returning a non-regular hypergraph",
          ok38)

    # C39: thermodynamics 'at beta' must be read at exactly beta, in the default and QUICK_TEST grids
    HQ = load_script("hsyk_numerics.py", QUICK_TEST=True)
    on_grid = []
    for ns_ in (H_, HQ):
        try:
            on_grid.append(ns_["THERMO_BETAS"][ns_["beta_index"](ns_["THERMO_BETAS"], ns_["BETA"])] == ns_["BETA"])
        except ValueError as exc:
            on_grid.append(False)
            print(f"    C39: {exc}")
    ok39 = all(on_grid) and raises(ValueError, H_["beta_index"], np.linspace(0.5, 10.0, 12), 5.0)
    check("C39", "BETA lies exactly on THERMO_BETAS (default and QUICK_TEST); an off-grid BETA raises", ok39,
          f"default grid {'ok' if on_grid[0] else 'BAD'}, QUICK_TEST grid {'ok' if on_grid[1] else 'BAD'}")

    # C40: analysis-only modes never rewrite or replace a saved instance
    out_dir = H_["OUT_DIR"]
    with tempfile.TemporaryDirectory() as tmp:
        H_["OUT_DIR"] = tmp
        try:
            path = os.path.join(tmp, "hsyk_N16_d4.json")
            np.random.seed(3)
            he16 = H_["generate_1overlap_hypergraph"](16, q, 4)
            H_["save_instance"](path, 16, q, 4, he16, {}, {"seed": 0})
            before = open(path).read()
            he_back, _, _ = H_["get_instance"](16, q, 4, 0, protect_saved=True)
            same = he_back == he16 and open(path).read() == before
            data = json.loads(before)
            data["selection_settings"]["pool_size"] += 1          # a saved file from other settings
            json.dump(data, open(path, "w"))
            changed = open(path).read()
            refused = raises(RuntimeError, H_["get_instance"], 16, q, 4, 0, protect_saved=True)
            ok40 = same and refused and open(path).read() == changed
        finally:
            H_["OUT_DIR"] = out_dir
    check("C40", "protected get_instance reuses a saved instance without rewriting it, and refuses to replace "
          "one saved with different settings", ok40)

    # C41: command-line handling. A typo, a flag or --help must never start a computation;
    # a notebook kernel's own argv ('-f <kernel.json>') is ignored; '%run script mode' works.
    import types
    argv, added = sys.argv, "ipykernel" not in sys.modules
    try:
        import contextlib
        import io
        mf, modes = H_["mode_from_argv"], H_["RUN_MODES"]
        with contextlib.redirect_stdout(io.StringIO()):
          cli = (mf("full", modes, "hsyk_numerics.py", argv=[]) == "full"
                 and mf("full", modes, "hsyk_numerics.py", argv=["postprocess"]) == "postprocess"
                 and all(raises(SystemExit, mf, "full", modes, "hsyk_numerics.py", argv=a)
                         for a in (["--help"], ["-h"], ["-f", "k.json"], ["postproces"], ["full", "extra"])))
        sys.argv = ["hsyk_selfcheck.py", "no_such_mode"]
        load_script("hsyk_numerics.py")                  # loading must not run the __main__ block
        if added:
            sys.modules["ipykernel"] = types.ModuleType("ipykernel")     # simulate a notebook kernel
        sys.argv = ["ipykernel_launcher.py", "-f", "/tmp/kernel.json"]   # code pasted into a cell
        cell = load_script("hsyk_numerics.py", as_main=True)["RUN_MODE"] == "none"
        cell &= mf("full", modes, "hsyk_numerics.py") == "full"
        load_script("hsyk_constructability.py", as_main=True)            # must not exit
        sys.argv = ["hsyk_numerics.py", "postprocess"]                    # %run hsyk_numerics.py postprocess
        run_magic = mf("full", modes, "hsyk_numerics.py") == "postprocess"
        if added:
            del sys.modules["ipykernel"]
            added = False
        sys.argv = ["hsyk_constructability.py", "--replot"]
        constr = raises(SystemExit, load_script, "hsyk_constructability.py", as_main=True)
        ok41 = cli and cell and run_magic and constr
    except BaseException as exc:
        ok41 = False
        print(f"    C41 error: {exc!r}")
    finally:
        sys.argv = argv
        if added:
            sys.modules.pop("ipykernel", None)
    check("C41", "command line: unknown modes, flags, extra arguments and --help never start a run; "
          "notebook-kernel argv ignored; %run arguments honoured", ok41)

    # C42: no LaTeX range dash inside math mode (there "--" renders as two minus signs)
    with tempfile.TemporaryDirectory() as tmp:
        H_["OUT_DIR"] = tmp
        try:
            rows = [{"N": 24, "d": 4, "instance": m, "seed": m, "kappa_1": 0.3 + 0.01 * m, "kappa_max": 0.33,
                     "chiral": "odd", "isomorphic_to": None, "dG_mid": -0.007 - 0.001 * m, "r": 0.52,
                     "E_beta": -0.037 + 0.0001 * m, "S_beta": 0.29} for m in range(2)]
            rows.append(dict(rows[0], N=22, E_beta=None, S_beta=None))
            H_["write_instance_tables"](rows)
            tex = open(os.path.join(tmp, "table_instances.tex")).read()
        finally:
            H_["OUT_DIR"] = out_dir
    bad42 = [seg for line in tex.splitlines() if not line.startswith("%")
             for seg in line.split("$")[1::2] if "--" in seg]           # odd segments = inside $...$
    check("C42", "table_instances.tex keeps range dashes out of math mode", not bad42, f"{len(bad42)} bad ranges")

    # C43: chunked w_min enumeration equals one-shot enumeration, across many chunks
    E43 = S_.random_regular_hypergraph(24, q, 6, np.random.default_rng(6))   # min weight found late
    B43 = S_.f2_kernel(H_["incidence_matrix"](24, E43).astype(int)).astype(np.int64)
    k43 = len(B43)
    coeffs = ((np.arange(1, 2 ** k43)[:, None] >> np.arange(k43)) & 1).astype(np.int64)
    w43 = ((coeffs @ B43) % 2).sum(axis=1)
    ref43 = (k43, int(w43.min()), int((w43 == w43.min()).sum()))
    drops43 = len(np.unique(np.minimum.accumulate(w43))) - 1    # > 0: the running-minimum reset is exercised
    got43 = [S_.min_codeword_weight(24, E43, chunk=c) for c in (1, 7, 777, 2 ** 16)]   # chunk = 1 updates
    check("C43", f"chunked w_min enumeration equals one-shot enumeration (k = {k43}; chunks of 1, 7, 777, 2^16)",
          all(g == ref43 for g in got43) and drops43 > 0 and ref43[2] > 1,
          f"{got43[0]} vs {ref43}; running minimum drops {drops43} times")          # the minimum repeatedly

    # C44: an OTOC fit window without enough time points raises instead of crashing on y[0]
    t44 = np.linspace(0.0, 5.0, 26)
    ok44 = (int(H_["fit_window_mask"](t44, (1.5, 3.5)).sum()) == 10
            and raises(ValueError, H_["fit_window_mask"], t44, (3.51, 3.59)))
    check("C44", "OTOC fit windows are validated (>= 3 grid points)", ok44)

    # C45: a kappa = 0 mode comes out as about +-1e-17; the sign (set by the numpy/BLAS build) must
    # not decide whether it enters the kappa <= 0 fits. Its planted rate is off the conformal line,
    # so including or excluding it would change those fits.
    kap45 = np.array([1, 0.25, 0.1, 0.0, -0.05, -0.1, -0.2, -0.3])
    V45 = np.linalg.qr(np.column_stack([np.ones(8), np.random.default_rng(1).standard_normal((8, 7))]))[0]
    rate45 = [H_["predicted_decay_rate"](k, q, H_["BETA"]) + (0.3 if k == 0 else 0.0) for k in kap45]
    F45 = np.einsum("ia,ta,ja->tij", V45, np.exp(-np.outer(t44, rate45)), V45)
    out45 = []
    for eps in (1e-17, -1e-17):
        k45 = kap45.copy()
        k45[3] = eps
        out45.append(H_["analyze_mode_otocs"](F45, t44, k45, V45, q, H_["BETA"], (1.5, 3.5), verbose=False)[5])
    check("C45", "a kappa = 0 mode is classified the same for rounding errors of either sign",
          out45[0] == out45[1] and "slope_kappa_nonpos" in out45[0])

    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed.")


if __name__ == "__main__":
    main()
