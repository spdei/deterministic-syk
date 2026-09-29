# -*- coding: utf-8 -*-
"""
make_fig_deltaG_vs_N.py
=======================
Figure "Deviation from the melonic solution versus N" (fig_deltaG_vs_N.pdf).

Inputs (all in OUT_DIR of hsyk_numerics.py):
  instances.csv       hSYK: G(beta/2) - G_SD(beta/2) for every instance   (hsyk_numerics.py, full run)
  controls_N24.npz    dense SYK at N = 24: G(tau) of each realization     (hsyk_numerics.py, full run)
  dense_sweep.json    dense SYK at N = 18, 20, 22                         (dense_sweep.py)

Usage:  python make_fig_deltaG_vs_N.py
"""

import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hsyk_numerics as hn


def load_data(out_dir):
    hsyk = {}
    with open(os.path.join(out_dir, "instances.csv")) as f:
        for row in csv.DictReader(f):
            hsyk.setdefault(int(row["d"]), {}).setdefault(int(row["N"]), []).append(float(row["dG_mid"]))
    with open(os.path.join(out_dir, "dense_sweep.json")) as f:
        dense = {int(N): [r[0] for r in rows] for N, rows in json.load(f)["results"].items()}
    ctrl = os.path.join(out_dir, f"controls_N{hn.FIG_N}.npz")
    if os.path.exists(ctrl):
        with np.load(ctrl) as z:
            Gd = z["dense_G"]
        _, Gsd = hn.solve_sd_fft(hn.BETA, q=hn.Q, M=32768)
        # extend, not overwrite: if FIG_N is also in dense_sweep.json (e.g. QUICK_TEST, FIG_N = 20),
        # both sets are valid dense realizations at that N
        dense.setdefault(hn.FIG_N, []).extend(float(g[len(g) // 2] - Gsd[len(Gsd) // 2]) for g in Gd)
    return hsyk, dense


def plot(hsyk, dense, path):
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=200)
    NN = np.array(sorted(dense))
    ax.plot(NN, 0.5 / NN * 1e3, color="gray", ls=":", lw=1.2, label=r"$0.5/N$")
    for N in NN:
        ax.plot([N] * len(dense[N]), -np.array(dense[N]) * 1e3, "s", mfc="none", mec="k", ms=6,
                label="dense SYK" if N == NN[0] else None)
    ax.plot(NN, [np.mean(-np.array(dense[N])) * 1e3 for N in NN], "k-", lw=0.8, alpha=0.6)
    cols = {4: "C0", 5: "C1", 6: "C2"}
    off = {4: 0.0, 5: -0.25, 6: 0.25}
    for d in sorted(hsyk):
        Ns = sorted(hsyk[d])
        for N in Ns:
            v = -np.array(hsyk[d][N]) * 1e3
            ax.plot([N + off.get(d, 0)] * len(v), v, "o", color=cols.get(d), ms=4.5, alpha=0.85,
                    label=f"hSYK, $d={d}$" if N == Ns[0] else None)
        if len(Ns) > 1:
            ax.plot(Ns, [np.mean(-np.array(hsyk[d][N])) * 1e3 for N in Ns], "-", color=cols.get(d), lw=0.8, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$-10^3\,\Delta G$,  $\Delta G = G(\beta/2)-G_{\rm SD}(\beta/2)$")
    ax.set_xticks(sorted(set(NN) | {N for d in hsyk for N in hsyk[d]}))
    # The paper's range is 1.5-45; widen it if any point would fall outside (none do for the paper data).
    vals = np.array([-v * 1e3 for vs in dense.values() for v in vs] +
                    [-v * 1e3 for d in hsyk for vs in hsyk[d].values() for v in vs])
    if (vals <= 0).any():
        print(f"warning: {(vals <= 0).sum()} point(s) have Delta G >= 0 and cannot be shown on the log axis")
    pos = vals[vals > 0]
    lo = min(1.5, pos.min() / 1.3) if pos.size else 1.5
    hi = max(45.0, pos.max() * 1.3) if pos.size else 45.0
    ticks = [t for t in (1, 2, 3, 5, 10, 20, 30, 50, 100) if lo <= t <= hi]
    ax.set_yticks(ticks); ax.set_yticklabels([str(t) for t in ticks]); ax.set_ylim(lo, hi)
    ax.set_title(rf"Deviation from the melonic solution ($\beta J={hn.BETA:g}$)")
    ax.legend(frameon=True, fontsize=9, loc="lower left", ncol=2)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    hsyk, dense = load_data(hn.OUT_DIR)
    path = os.path.join(hn.OUT_DIR, "fig_deltaG_vs_N.pdf")
    plot(hsyk, dense, path)
    print(f"Wrote {path}")
