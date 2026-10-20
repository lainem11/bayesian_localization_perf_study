import jax
jax.config.update("jax_log_compiles", True)
from shapes import make_session
import torch
import numpy as np
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict
from env_info import collect_env

def timed(name, fn, *args):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return result, name, t1-t0

def to_jsonable(obj):
    """Fallback serializer for json.dump (tensors, arrays, paths)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)

def parse_args():
    p = argparse.ArgumentParser(description="Run a benchmark session.")
    p.add_argument("--trials", type=int, default=50, help="number of trials")
    p.add_argument("--warmup", type=int, default=1, help="warmup iterations (K)")
    p.add_argument("--repeats", type=int, default=1, help="session repeats (R)")
    p.add_argument("--seed", type=int, default=0, help="random seed (S).")
    p.add_argument("--out", type=Path, default=Path("results.json"), help="output path")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    all_runs = []

    for r in range(args.repeats):
        results = defaultdict(list)

        # Initialize a LocalizationSession
        S = make_session(args.seed)

        # Run warmup
        ses_t0 = time.perf_counter()

        # Run trials
        for i in range(S.trial_number, args.trials):

            results["trial number"].append(S.trial_number)

            # Prepare for a new trial
            S.start_trial()  

            # Optimize the E-field pattern based on the current model
            didt, name, dur = timed("choose_next_stimulus() duration", S.choose_next_stimulus)
            results["didt"].append(didt)
            results[name].append(dur)
            
            _, name, dur = timed("record_efield_trial() duration", S.record_efield_trial, didt)
            results[name].append(dur)

            # Recorded the stimulation-induced response
            _, name, dur = timed("simulate_response() duration", S.simulate_response)
            results[name].append(dur)

            # Use the measured data to localize the responses
            _, name, dur = timed("localize() duration", S.localize)
            results[name].append(dur)

            # Get maximum probability index
            max_prob_ind = torch.argmax(S.response_model.log_source_prob).item()
            results["max prob. index"].append(max_prob_ind)

            # Get response function fitting steps
            results["adam steps"].append(len(S.response_model.losses))

            # Complete the trial
            S.complete_trial()

        ses_t1 = time.perf_counter()
        results["total session duration"].append(ses_t1-ses_t0)

        responses = S.get_active_responses()
        results["responses"] = responses

        all_runs.append(dict(results))

    # Produce an output JSON
    output = {
        "env": collect_env(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "runs": all_runs,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, default=to_jsonable)
    print(f"Results written to {args.out}")

    # Summary of per-stage timings (only post-warmup trials are timed)
    stages = [k for k in all_runs[0] if k.endswith("duration")]
    n = len(all_runs[0][stages[0]]) * len(all_runs) if stages else 0
    print(f"\nTiming summary over {n} trials (after {args.warmup} warm-up), seconds:")
    print(f"{'stage':<40}{'median':>10}{'IQR':>10}{'Q1':>10}{'Q3':>10}")
    for stage in stages:
        vals = np.array([v for run in all_runs for v in run[stage][args.warmup:]])
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        print(f"{stage:<40}{med:>10.4f}{q3 - q1:>10.4f}{q1:>10.4f}{q3:>10.4f}")