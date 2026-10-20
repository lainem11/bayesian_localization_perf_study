"""Compare two benchmark result files for numerical equivalence.

Usage: python compare.py REF.json NEW.json

Checks "didt", "responses" and "adam steps" of every run. Exits 0 if all are
identical, 1 otherwise.
"""
import json
import sys

import numpy as np

KEYS = ["didt", "responses", "adam steps"]


def compare_key(ref, new):
    """Return None if identical, else (first differing trial, max abs diff, note)."""
    if len(ref) != len(new):
        return (min(len(ref), len(new)), None, f"length {len(ref)} vs {len(new)}")
    a = np.asarray(ref, dtype=float)
    b = np.asarray(new, dtype=float)
    # Treat NaN at the same position as equal.
    diff = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    if not diff.any():
        return None
    trial_diff = diff.reshape(len(a), -1).any(axis=1)
    first = int(np.argmax(trial_diff))
    with np.errstate(invalid="ignore"):
        absdiff = np.abs(a - b)
    absdiff = np.where(diff, np.nan_to_num(absdiff, nan=np.inf), 0.0)
    return first, float(absdiff.max()), ""


def main(ref_path, new_path):
    with open(ref_path) as f:
        ref = json.load(f)
    with open(new_path) as f:
        new = json.load(f)

    ref_runs, new_runs = ref["runs"], new["runs"]
    if len(ref_runs) != len(new_runs):
        print(f"Number of runs differs: {len(ref_runs)} vs {len(new_runs)}")
    all_same = len(ref_runs) == len(new_runs)

    for i, (r, n) in enumerate(zip(ref_runs, new_runs)):
        for key in KEYS:
            res = compare_key(r[key], n[key])
            if res is None:
                print(f"run {i} {key}: identical")
                continue
            all_same = False
            first, maxd, note = res
            msg = f"run {i} {key}: DIFFERENT, first at trial {first}"
            if maxd is not None:
                msg += f", max abs diff {maxd:.6g}"
            if note:
                msg += f" ({note})"
            print(msg)

    print("\nRESULT:", "all identical" if all_same else "differences found")
    return 0 if all_same else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
