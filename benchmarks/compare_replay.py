"""Compare two replay_localize.py outputs trial by trial.

Usage: python compare_replay.py REF.npz NEW.npz
"""
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
la, lb = a["log_post"].astype(np.float64), b["log_post"].astype(np.float64)
pa, pb = np.exp(la), np.exp(lb)

for i in range(19, 50):                            # trials 20..50
    order = np.argsort(pa[i])[::-1]                # vertices from most to least probable
    k = np.searchsorted(np.cumsum(pa[i][order]), 0.95) + 1   # how many hold 95% of the mass
    top = order[:k]
    rel = np.abs(pb[i][top] - pa[i][top]) / pa[i][top]
    print(f"trial {i+1}: {k} vertices, max relative difference {rel.max():.2%}")

tv = 0.5 * np.abs(pa - pb).sum(axis=1)       # total variation distance per trial, 0..1
maxlog = np.abs(la - lb).max(axis=1)          # largest log-probability difference per trial
same_argmax = la.argmax(1) == lb.argmax(1)
dsteps = b["steps"] - a["steps"]

print(f"{'trial':>5}{'TV dist':>12}{'max |dlog p|':>14}{'argmax ref/new':>18}{'steps ref/new':>16}")
for i in range(len(tv)):
    am = f"{la[i].argmax()}/{lb[i].argmax()}"
    st = f"{a['steps'][i]}/{b['steps'][i]}"
    print(f"{i + 1:>5}{tv[i]:>12.2e}{maxlog[i]:>14.2e}{am:>18}{st:>16}")
print(f"\nmax TV distance {tv.max():.2e} (trial {tv.argmax() + 1}); "
      f"argmax equal in {same_argmax.sum()}/{len(tv)} trials; "
      f"Adam step counts differ in {(dsteps != 0).sum()} trials")