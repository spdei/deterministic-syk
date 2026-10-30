"""
Finite-size <r> reference for the N = 18 and N = 22 instances (d = q = 4), and the
combined level-statistics test of the paper (Table 2 and the "Combined test" paragraph
of the ED appendix).

At d = q and N = 2, 6 (mod 8) each parity sector of hSYK is symmetric under E -> -E
(ED appendix, filter 2), so every gap ratio appears twice and <r> scatters more than for
a GUE matrix of the same size. The reference is therefore computed from random matrices
with the same symmetry, classes D (H = iX, X real antisymmetric) and C
(H = [[A, B], [B*, -A*]], A Hermitian, B complex symmetric). The two classes agree; their
draws are pooled. Processing is identical to rmt_calibration in hsyk_numerics.py (one
sector, central 80%, no unfolding).

Usage (from the repository root):
    python3 tools/rmt_mirror_reference.py          # both N (about 4 minutes)
    python3 tools/rmt_mirror_reference.py 18       # one N; results are merged into the CSV

Outputs (data/ed/):
    rmt_calibration_mirror.csv   N, n_matrices, r_mean, r_std
    rmt_combined_test.json       z-values and sum z^2, written once both N are present
"""
import csv
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ED = os.path.join(ROOT, 'data', 'ed')
DRAWS = 200                                    # per class
RUNS = {18: [('D', 1), ('C', 1)], 22: [('D', 2), ('C', 3)]}   # (class, seed)


def gap_ratio(evals, trim_frac=0.10):
    ev = np.sort(evals)
    k = int(trim_frac * len(ev))
    g = np.diff(ev[k:len(ev) - k])
    return float(np.mean(np.minimum(g[:-1], g[1:]) / np.maximum(g[:-1], g[1:])))


def class_d(n, rng):
    X = rng.normal(size=(n, n))
    return np.linalg.eigvalsh(1j * (X - X.T) / 2)


def class_c(n, rng):
    m = n // 2
    X = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    A = (X + X.conj().T) / 2
    Y = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    B = (Y + Y.T) / 2
    return np.linalg.eigvalsh(np.block([[A, B], [B.conj(), -A.conj()]]))


def reference(N):
    dim = 2 ** (N // 2 - 1)
    rs = []
    for cls, seed in RUNS[N]:
        rng = np.random.default_rng(seed)
        f = class_d if cls == 'D' else class_c
        rs += [gap_ratio(f(dim, rng)) for _ in range(DRAWS)]
    rs = np.array(rs)
    return {'N': N, 'n_matrices': len(rs), 'r_mean': float(rs.mean()), 'r_std': float(rs.std(ddof=1))}


def main():
    Ns = [int(a) for a in sys.argv[1:]] or sorted(RUNS)
    path = os.path.join(ED, 'rmt_calibration_mirror.csv')
    rows = {}
    if os.path.exists(path):
        rows = {int(r['N']): r for r in csv.DictReader(open(path))}
    for N in Ns:
        r = reference(N)
        rows[N] = r
        print(f"N={N}: mirror-symmetric reference <r> = {r['r_mean']:.4f} +- {r['r_std']:.4f} ({r['n_matrices']} matrices)")
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['N', 'n_matrices', 'r_mean', 'r_std'])
        w.writeheader()
        w.writerows(rows[N] for N in sorted(rows))
    if not all(N in rows for N in RUNS):
        return
    # combined test: mirror-symmetric reference for N = 18, 22; GOE/GSE references from the main run
    ref = {N: (float(v['r_mean']), float(v['r_std'])) for N, v in rows.items()}
    for N, v in json.load(open(os.path.join(ED, 'postprocess_summary.json')))['rmt_calibration'].items():
        ref.setdefault(int(N), (v['r_mean'], v['r_std']))
    inst = list(csv.DictReader(open(os.path.join(ED, 'instances.csv'))))
    z = {f"N{r['N']}_d{r['d']}_i{r['instance']}": (float(r['r']) - ref[int(r['N'])][0]) / ref[int(r['N'])][1] for r in inst}
    s_all = sum(v * v for v in z.values())
    s4 = sum(v * v for k, v in z.items() if '_d4_' in k)
    s56 = s_all - s4
    out = {'z': z, 'n_beyond_2sigma': sum(abs(v) > 2 for v in z.values()),
           'sum_z2_all': s_all, 'p_all': float(stats.chi2.sf(s_all, len(z))),
           'sum_z2_d4': s4, 'p_d4': float(stats.chi2.sf(s4, 20)),
           'sum_z2_d56': s56, 'p_d56': float(stats.chi2.sf(s56, 10))}
    json.dump(out, open(os.path.join(ED, 'rmt_combined_test.json'), 'w'), indent=1)
    print(f"combined test: {out['n_beyond_2sigma']} of {len(z)} beyond 2 sigma; sum z^2 = {s_all:.1f} (p = {out['p_all']:.2f}); "
          f"d = 4: {s4:.1f} (p = {out['p_d4']:.2f}); d = 5, 6: {s56:.1f}")


if __name__ == '__main__':
    main()
