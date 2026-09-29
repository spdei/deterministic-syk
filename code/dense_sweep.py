# -*- coding: utf-8 -*-
"""
dense_sweep.py
==============
Dense-SYK benchmark at N = 18, 20, 22 (Table "Midpoint deviation", Fig. "Deviation
from the melonic solution versus N"). The N = 24 dense realizations are part of the
controls of hsyk_numerics.py (controls_N24.npz).

For each N, three realizations of dense SYK with couplings of variance
J^2 / C(N-1, 3) (so that Sigma = J^2 G^3 exactly at the melonic level) are
diagonalized, and G(beta/2) - G_SD(beta/2), <r>, E/N and S/N at beta = 5 are recorded.
The SD reference uses 32768 Matsubara frequencies.

Usage:  python dense_sweep.py            (about 1 minute)
Output: <OUT_DIR>/dense_sweep.json with, for each N, a list of
        [G(beta/2) - G_SD(beta/2), <r>, E/N, S/N] per realization.
"""

import os
import json
import time
import numpy as np

import hsyk_numerics as hn

N_VALUES = [18, 20, 22]
SAMPLES = 3
SEED = 11


def main():
    os.makedirs(hn.OUT_DIR, exist_ok=True)
    tau = np.linspace(0, hn.BETA, hn.N_TAU)
    mid = len(tau) // 2
    _, Gsd = hn.solve_sd_fft(hn.BETA, q=hn.Q, M=32768)
    Gmid = Gsd[len(Gsd) // 2]
    rng = np.random.default_rng(SEED)
    out = {}
    for N in N_VALUES:
        rows = []
        for s in range(SAMPLES):
            t0 = time.time()
            terms, cpl = hn.dense_syk_terms(N, hn.Q, 1.0, rng)
            H, psi = hn.build_hamiltonian_terms(N, terms, cpl, hn.Q)
            G, ev, spec = hn.diagonalize_and_G(psi, H, N, hn.BETA, tau)
            del H
            E, S = hn.ed_thermodynamics(spec, N, [hn.BETA])
            rows.append([float(G[mid] - Gmid), hn.gap_ratio(ev), float(E[0]), float(S[0])])
            print(f"N={N} sample {s}: dG={rows[-1][0]:+.5f}  <r>={rows[-1][1]:.4f}  "
                  f"E/N={rows[-1][2]:.5f}  S/N={rows[-1][3]:.5f}  ({time.time() - t0:.0f} s)")
        out[str(N)] = rows
    path = os.path.join(hn.OUT_DIR, "dense_sweep.json")
    with open(path, "w") as f:
        json.dump({"seed": SEED, "beta": hn.BETA, "columns": ["dG_mid", "r", "E_over_N", "S_over_N"],
                   "results": out}, f, indent=2)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
