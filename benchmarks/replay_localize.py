"""Level-2 equivalence for localize(): replay recorded trials, save posteriors.

Feeds the didt values and responses of run 0 of a reference results file into a
fresh session, one trial at a time, and calls only localize() (no DE). Every
variant therefore fits exactly the same data, so their posteriors can be compared
trial by trial.

Usage: python replay_localize.py REF.json OUT.npz [--merged_adam] [--compile_loss]
"""
import argparse
import json
import time

import numpy as np
import torch

import compiled_loss
import merged_adam
from env_info import collect_env
from shapes import make_session

p = argparse.ArgumentParser()
p.add_argument("ref")
p.add_argument("out")
p.add_argument("--merged_adam", action="store_true")
p.add_argument("--compile_loss", action="store_true")
args = p.parse_args()
if args.merged_adam:
    merged_adam.install()
if args.compile_loss:
    compiled_loss.install()

run = json.load(open(args.ref))["runs"][0]
S = make_session()
posteriors, steps, times = [], [], []
for didt, resp in zip(run["didt"], run["responses"]):
    S.start_trial()
    S.record_efield_trial(didt)               # same E-field as in the reference run
    S.record_response_trial(bool(resp))       # same response
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    S.localize()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
    posteriors.append(S.response_model.log_source_prob.detach().cpu().numpy())
    steps.append(len(S.response_model.losses))
    S.complete_trial()
S.end_session()

np.savez(args.out, log_post=np.stack(posteriors), steps=np.array(steps), times=np.array(times),
         env=json.dumps(collect_env(), default=str), args=json.dumps(vars(args)))
print(f"saved {args.out}: {len(steps)} trials, final argmax {int(np.argmax(posteriors[-1]))}")