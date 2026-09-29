# data/constructability: files produced by hsyk_constructability.py

Produced by a single run of `hsyk_constructability.py` (default settings: three
instances per (N, d), N in {1e3, 3e3, 1e4, 3e4, 1e5}, d in {4, 6, 8, 12, 16, 20};
plus 40-instance filter pass-rate samples at N in {200, 500, 1000}). No
Hamiltonian is diagonalized anywhere in this folder; every file here concerns
the hypergraph only.

| File | Content |
|---|---|
| `constructability_instances.csv` | One row per (N, d, instance): κ₁, κ_min, Ramanujan edges, generation and Lanczos diagnostics, F₂ rank, chiral type, automorphism certificate. The κ columns of the paper's large-N diagnostics table are the N = 1e5 rows of this file; its C₆ column is the mean of `triangles_nontrivial_total` over all 15 instances at each d. |
| `constructability_balls.csv` | One row per (N, d, instance, R) with R ≥ 1 and the tree ball at or below the 5% cutoff: mean/max cycle rank of 300 sampled Levi balls, tree-ball size, birthday parameter x_R. Feeds panels (c) and (d) of the constructibility figure and the R_max column of the large-N diagnostics table. |
| `filter_pass_rates.csv` | One row per (N, d) with N in {200, 500, 1000}, over 40 independently generated instances each: fraction with full F₂ rank, count with a parity-preserving chiral monomial, automorphism certificate rate, fraction passing all three filters. This is the paper's symmetry-filter table. |
| `constructability_summary.json` | Everything above in one file, plus the run configuration (Q, N_VALUES, D_VALUES, N_INSTANCES, SEED, MIX_SWEEPS, N_BALL_ROOTS, R_MAX, BALL_MAX_FRACTION, F2_MAX_N, AUT_MAX_N, FILTER_RATE_N, FILTER_RATE_SAMPLES, EIG_TOL). This is what `hsyk_constructability.py replot` reads to redraw the figure. |
| `fig_constructability.pdf` | The four-panel constructibility figure: (a) κ₁ relative to the Ramanujan edge vs N; (b) extreme non-uniform eigenvalues vs d at N = 1e5; (c) radius-3 cycle rank vs the parameter-free prediction 3C₆/\|E\|; (d) cycle rank vs the birthday parameter x_R, all radii. |

## Reproducing this folder

```
cd code
python3 hsyk_constructability.py          # full run, ~30-45 min
python3 hsyk_constructability.py replot   # redraw fig_constructability.pdf only, from the saved JSON, seconds
```

Seeds are fixed (`SEED = 2026` for the main grid; the filter pass-rate study
uses `SeedSequence([2026, N, d, 10**6 + m])`), so a fresh run reproduces every
number in this folder, up to Lanczos convergence noise at the ~1e-7 level in κ₁
and κ_min.
