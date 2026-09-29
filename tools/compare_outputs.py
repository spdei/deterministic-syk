# -*- coding: utf-8 -*-
"""
compare_outputs.py: compare a fresh rerun with the data shipped before it.

Usage (from the repository root):
    python3 tools/compare_outputs.py data_before data

For every file present in both folders it compares all numbers (npz arrays, JSON and CSV
values, numbers in .tex tables) and prints one line:
    identical     max |difference| <= 1e-12
    tiny          rounding level: every difference <= 1e-6, or, for a number printed in a
                  CSV or TeX file, at most one unit in its last printed digit
    CHANGED       larger; the worst entry is shown
plus keys or files that were added or removed. PDFs are only checked for existence.
Timing fields (t, t_*, runtimes), Lanczos iteration counts and file paths are ignored.
Needs only numpy (no pandas).
"""

import csv
import json
import math
import os
import re
import sys

import numpy as np

TINY, IDENT = 1e-6, 1e-12
# ignored: timings, and iteration counts that depend on floating-point rounding
TIMING = re.compile(r"^(t|t_.*|.*_time|.*runtime.*|elapsed|lanczos_steps)$")
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def flatten(obj, prefix=""):
    """{path: value} for every leaf of a JSON object, skipping timing keys and paths."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if TIMING.match(str(k)) or k == "OUT_DIR":
                continue
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def resolution(text):
    """One unit in the last printed digit of a decimal number ("0.906171" -> 1e-6,
    "1.25e-05" -> 1e-7); 0 for integers, which must match exactly."""
    m = re.fullmatch(r"[-+]?\d*\.(\d+)(?:[eE]([-+]?\d+))?", text.strip())
    if not m:
        return 0.0
    return 10.0 ** (int(m.group(2) or 0) - len(m.group(1)))


def compare_values(a, b, res_a=None, res_b=None):
    """(max abs diff, where, beyond_rounding, notes) over two {key: value} maps. res_a/res_b
    give the printed resolution of text-file numbers (see resolution())."""
    worst, where, beyond, notes = 0.0, None, False, []
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    if added:
        notes.append(f"{len(added)} new entries (e.g. {added[0]})")
    if removed:
        notes.append(f"{len(removed)} entries gone (e.g. {removed[0]})")
    for k in set(a) & set(b):
        x, y = a[k], b[k]
        if is_num(x) and is_num(y):
            if math.isnan(x) and math.isnan(y):
                continue
            dx = abs(x - y) if not (math.isnan(x) or math.isnan(y)) else math.inf
            tol = max(TINY, 1.01 * max((res_a or {}).get(k, 0.0), (res_b or {}).get(k, 0.0)))
            beyond |= dx > tol
            if dx > worst:
                worst, where = dx, k
        elif isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x, y = np.asarray(x), np.asarray(y)
            if x.shape != y.shape:
                notes.append(f"{k}: shape {x.shape} -> {y.shape}")
                continue
            if x.dtype.kind in "fciu" and y.dtype.kind in "fciu":
                d = np.abs(x.astype(complex) - y.astype(complex))
                d = d[~(np.isnan(x.astype(complex)) & np.isnan(y.astype(complex)))] if d.size else d
                if d.size:
                    beyond |= bool(np.nanmax(d) > TINY)
                if d.size and np.nanmax(d) > worst:
                    worst, where = float(np.nanmax(d)), k
            elif not np.array_equal(x, y):
                notes.append(f"{k}: non-numeric content differs")
        elif x != y and not (isinstance(x, str) and ("/" in x or "\\" in x)):
            notes.append(f"{k}: {x!r} -> {y!r}")
    beyond |= bool(notes)          # added/removed entries, shapes or non-numeric content
    return worst, where, beyond, notes


def load(path):
    """({key: value}, {key: printed resolution} or None), or (None, None) if not comparable."""
    ext = os.path.splitext(path)[1]
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}, None
    if ext == ".json":
        with open(path) as f:
            return flatten(json.load(f)), None
    if ext == ".csv":
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        out, res = {}, {}
        for i, r in enumerate(rows):
            for k, v in r.items():
                if k is None or TIMING.match(k):
                    continue
                fv = to_float(v)
                out[f"row{i}.{k}"] = fv if fv is not None else v
                if fv is not None:
                    res[f"row{i}.{k}"] = resolution(v)
        return out, res
    if ext == ".tex":
        out, res = {}, {}
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.replace("--", " ")          # range dashes are not minus signs
                for j, m in enumerate(NUMBER.findall(line)):
                    out[f"line{i + 1}.num{j}"] = float(m)
                    res[f"line{i + 1}.num{j}"] = resolution(m)
        return out, res
    return None, None


def main(before, after):
    names = sorted(set(os.listdir(before)) | set(os.listdir(after)))
    counts = {"identical": 0, "tiny": 0, "CHANGED": 0}
    for name in names:
        pa, pb = os.path.join(before, name), os.path.join(after, name)
        if os.path.isdir(pa) or os.path.isdir(pb):
            continue
        if not os.path.exists(pb):
            print(f"  MISSING in rerun   {name}")
            continue
        if not os.path.exists(pa):
            print(f"  new file           {name}")
            continue
        (a, res_a), (b, res_b) = load(pa), load(pb)
        if a is None:
            print(f"  present            {name}")
            continue
        worst, where, beyond, notes = compare_values(a, b, res_a, res_b)
        status = "CHANGED" if beyond else ("identical" if worst <= IDENT else "tiny")
        counts[status] += 1
        detail = "" if worst <= IDENT else f"  max |diff| {worst:.2e} at {where}"
        print(f"  {status:<9}          {name}{detail}")
        for n in notes[:4]:
            print(f"                       - {n}")
        if len(notes) > 4:
            print(f"                       - ... {len(notes) - 4} more")
    print(f"\n{counts['identical']} identical, {counts['tiny']} tiny differences, {counts['CHANGED']} changed.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python3 tools/compare_outputs.py <data_before> <data_after>")
    before, after = sys.argv[1], sys.argv[2]
    for sub in ("ed", "constructability"):
        if os.path.isdir(os.path.join(before, sub)):
            print(f"\n=== {sub}/")
            main(os.path.join(before, sub), os.path.join(after, sub))
