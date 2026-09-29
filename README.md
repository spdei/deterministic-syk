# Deterministic hypergraph SYK

Code and data for

> S. Prakash, *Deterministic Hypergraph SYK: Melonic Dominance and Maximal Chaos
> Without Disorder Averaging*, arXiv:2609.19255 (2026).

The scripts reproduce the numerical tables and figures of the paper: exact
diagonalization of deterministic hSYK instances up to N = 24, the dense-SYK and
coupling controls, thermodynamics, mode-resolved out-of-time-order correlators,
the large-N constructability study and the quantum-simulation resource
estimates.

## Requirements

Python 3.9 or later with numpy, scipy, networkx and matplotlib:

```
pip install -r requirements.txt
```

The code has been tested with Python 3.9 (numpy 1.24, scipy 1.10, networkx 3.0,
matplotlib 3.7) and Python 3.12 (numpy 2.4, scipy 1.17, networkx 3.6,
matplotlib 3.10). Exact diagonalization at N = 24 needs about 2 GB of memory.

## Repository layout

```
code/
  hsyk_numerics.py            instance selection, exact diagonalization, controls,
                              thermodynamics, OTOCs, postprocessing
  hsyk_constructability.py    large-N constructability study
  hsyk_supplement.py          minimum codeword weights, non-linear sparse controls,
                              quantum-simulation resources, data check
  dense_sweep.py              dense SYK at N = 18, 20, 22
  make_fig_deltaG_vs_N.py     figure of G(β/2) − G_SD(β/2) against N
  hsyk_selfcheck.py           self-tests
data/
  ed/                         outputs of the exact-diagonalization scripts
  constructability/           outputs of hsyk_constructability.py
tools/
  compare_outputs.py          compares a new run with the data in data/
run_all.sh                    runs the full chain
```

Each folder in `data/` has a `MANIFEST.md` describing its files and the paper
table or figure each one feeds.

## Tests

```
cd code
python3 hsyk_selfcheck.py
```

Each check compares a component of the code with an independent calculation:
explicit matrix products, brute-force enumeration, exact identities, known
limits or random-matrix values. The script runs in a few minutes and should
report 48/48 checks passed.

## Reproducing the results

From the repository root,

```
bash run_all.sh
```

runs the full chain in about 2 hours on a 2-core Github codespace and writes a log for each
step to `logs/`. If the run is interrupted, the same command resumes it,
skipping the steps that have finished; delete `logs/` to start from scratch.
`bash run_all.sh supplement` runs only the self-tests and the
`hsyk_supplement.py` computations, which use existing ED outputs.

The steps can also be run individually from `code/`:

| Command | Output | Time |
|---|---|---|
| `python3 hsyk_numerics.py full` | instance selection and search statistics; hypergraph diagnostics; hSYK Green's functions, level statistics and thermodynamics; controls on the hSYK hypergraphs; mode-resolved OTOCs; two-point/spectral, thermodynamics and OTOC figures | ~40 min |
| `python3 hsyk_numerics.py postprocess` | random-matrix calibration and z-scores; dense-SYK thermodynamics at N = 24; parity-sector symmetry checks at d = q | ~8 min |
| `python3 hsyk_numerics.py otoc_reanalysis` | OTOC fits for all three fit windows and the OTOC figure, from saved OTOC data | ~ seconds |
| `python3 dense_sweep.py` | dense SYK at N = 18, 20, 22 | ~30 s |
| `python3 make_fig_deltaG_vs_N.py` | figure of ΔG against N | seconds |
| `python3 hsyk_constructability.py` | large-N constructability study: spectral gap, local cycles, symmetry filters, figure | 15 min |
| `python3 hsyk_supplement.py wmin` | minimum codeword weights and sign classes of every instance | seconds |
| `python3 hsyk_supplement.py sparse_controls` | controls on non-linear hypergraphs | 30–40 min |
| `python3 hsyk_supplement.py resources` | quantum-simulation resource estimates | 10–20 min |
| `python3 hsyk_supplement.py check_data` | list of missing output files | seconds |

Times are for a 2-core GitHub codespace. Settings are in the `CONFIG` block at the top of each
script. With `QUICK_TEST = True` a reduced version runs in a few minutes and
writes to separate `*_quicktest` folders. `hsyk_numerics.py`,
`hsyk_supplement.py` and `hsyk_constructability.py` list their run modes with
`--help`. In a Jupyter notebook, use for example
`%run hsyk_numerics.py postprocess`, or set `RUN_MODE` in `CONFIG`.

To compare a new run with the data in this repository:

```
cp -r data data_before
bash run_all.sh
python3 tools/compare_outputs.py data_before data
```

Each file is reported as identical, as differing only by floating-point
rounding, or as changed. Timings are ignored.

## Reproducibility

Random seeds are fixed in the `CONFIG` block of each script: 42 for the
instance search, 7 for the controls, 12345 for the RMT calibration, 11 for the
dense sweep, 2026 for the constructability study, 8 for the sparse controls and
0 for the labeling optimization. Instance 0 of each (N, d) uses the seed
42·10⁵ + 100N + d; instances 1–4 use `SeedSequence([42, N, d, m])`.

The selected hypergraphs are stored in `data/ed/hsyk_N{N}_d{d}[_i{m}].json`
together with the settings used to select them, so the instance search does not
have to be repeated. `hsyk_numerics.py full` selects an instance again only if
the settings in `CONFIG` differ from the stored ones; the analysis-only modes
stop with an error instead.

The Schwinger–Dyson solver evaluates the imaginary-time Fourier integrals with
the trapezoid rule. The runs reported in the paper used the left-point rule;
the two differ by 1.2 × 10⁻⁸ in G_SD(β/2), below the precision of every quoted
number.

## Citation

If you use this code or data, please cite the paper (metadata in
`CITATION.cff`):

```
@article{Prakash2026,
  author        = {Prakash, Shiroman},
  title         = {Deterministic Hypergraph {SYK}: Melonic Dominance and Maximal
                   Chaos Without Disorder Averaging},
  year          = {2026},
  eprint        = {2609.19255},
  archivePrefix = {arXiv}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The data in `data/`
are released under CC BY 4.0.
