# data/ed: files produced by `run_all.sh`

Produced by `hsyk_numerics.py` (full, postprocess, otoc_reanalysis), `dense_sweep.py` and `hsyk_supplement.py`.

## Files

| File(s) | Produced by | Content |
|---|---|---|
| `hsyk_N{N}_d{d}.json`, `hsyk_N{N}_d{d}_i{1..4}.json` (30) | full | instance hyperedges, diagnostics, selection settings, search statistics |
| `data_N{N}_d{d}.npz`, `data_N{N}_d{d}_i{1..4}.npz` (30) | full | G(τ), spectra (both parity sectors), level-statistics eigenvalues; instance 0 also P(s) and SFF |
| `instances.csv` | full | one row per instance: κ₁, chiral type, ΔG, ⟨r⟩, E/N, S/N |
| `summary.json` | full | configuration and all per-instance results |
| `table_*.tex` (6) | full | LaTeX table rows used in the paper |
| `controls_N24.npz` | full | dense, random-sign and Gaussian-coupling controls: G(τ), ⟨r⟩, ΔG(β/2), E/N and S/N vs β |
| `thermodynamics_N24.npz` | full | E/N and S/N vs β for hSYK and SD |
| `otoc_allpairs_N24_d6.npz` | full | all-pairs regularized OTOC F_ij(t) |
| `otoc_modes_N24_d6.npz`, `otoc_mode_rates_N24_d6.csv` | full / otoc_reanalysis | mode-projected OTOCs, fitted and predicted rates |
| `otoc_reanalysis.json` | otoc_reanalysis | fit results for the three fit windows |
| `postprocess_instances.csv`, `postprocess_summary.json` | postprocess | SD convergence, sector symmetries, z-scores, ⟨r⟩ on E > 0 |
| `rmt_calibration.csv` | postprocess | finite-size RMT ⟨r⟩ (200 matrices per N) |
| `controls_thermo_N24.npz`, `dense_thermo_N24.tex` | postprocess | dense-SYK E/N, S/N |
| `dense_sweep.json` | dense_sweep.py | dense SYK at N = 18, 20, 22 |
| `fig_two_point_spectral_N24.pdf`, `fig_thermodynamics_N24.pdf`, `fig_otoc_modes_N24_d6.pdf`, `fig_deltaG_vs_N.pdf` | full / otoc_reanalysis / make_fig | paper figures |
| `wmin.csv` | hsyk_supplement.py wmin | dim ker_F₂(I) and w_min for every instance |
| `controls_sparse_N24.json`, `controls_sparse_N24.npz`, `table_controls_sparse.tex` | hsyk_supplement.py sparse_controls | non-linear sparse controls: hypergraphs, ΔG, ⟨r⟩, E/N, S/N, κ₁, overlap counts; G(τ) |
| `resources.json`, `table_simulation.tex` | hsyk_supplement.py resources | quantum-simulation resources: Pauli weights, CNOT counts, optimized labelings, layers, dense comparison |

## Where each paper table and figure comes from

Tables are named by the bold title of their caption.

| Paper table or figure | Files |
|---|---|
| Hypergraph diagnostics | `table_hypergraph_diagnostics.tex`; κ₁ ranges from `instances.csv` |
| Midpoint deviation from the melonic solution | `table_two_point_deviation.tex` and `instances.csv` (hSYK); `dense_sweep.json` and `controls_N24.npz` (dense SYK) |
| Single-instance level-spacing ratios | `table_level_statistics.tex`, `postprocess_instances.csv` (z-scores), `rmt_calibration.csv` |
| Search statistics for instance 0 | `table_search_statistics.tex` |
| hSYK and controls at N = 24 | `table_controls.tex`, `controls_N24.npz` (hSYK hypergraphs); `table_controls_sparse.tex`, `controls_sparse_N24.json` (non-linear hypergraphs); `wmin.csv` |
| Energy and entropy per Majorana at β = 5 | `instances.csv`, `summary.json` (hSYK and SD); `dense_thermo_N24.tex` (dense SYK) |
| Circuit resources for the representative hSYK instances | `table_simulation.tex`, `resources.json` |
| Two-point and spectral diagnostics figure | `fig_two_point_spectral_N24.pdf` |
| Thermodynamics figure | `fig_thermodynamics_N24.pdf` |
| Mode-resolved OTOC figure | `fig_otoc_modes_N24_d6.pdf`; fits in `otoc_reanalysis.json` |
| Deviation from the melonic solution versus N | `fig_deltaG_vs_N.pdf` |

The large-N tables and figure are in `../constructability/`.
