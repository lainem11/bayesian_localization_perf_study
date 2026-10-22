import jax
from shapes import make_session
import torch
import numpy as np
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict
from env_info import collect_env
import fixed_steps
import merged_adam

def timed(name, fn, *args):
    torch.cuda.synchronize()
    with torch.cuda.nvtx.range(name):
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
    p.add_argument("--out", type=Path, default=Path("results.json"), help="output path")
    p.add_argument("--fixed_steps", type=Path, default=None)
    p.add_argument("--log_compiles", action="store_true", help="print JAX compilation log")
    p.add_argument("--merged_adam", action="store_true", help="merge parameter operations")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    jax.config.update("jax_log_compiles", args.log_compiles)

    # Check for looping options from arguments
    if args.fixed_steps is not None:
        fixed_steps.install(args.fixed_steps)
    if args.merged_adam:
        merged_adam.install()
    all_runs = []

    for r in range(args.repeats):
        results = defaultdict(list)

        # Initialize a LocalizationSession
        S, name, dur = timed("make_session() duration", make_session)
        results[name].append(dur)
        
        # Run trials
        ses_t0 = time.perf_counter()
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

        S.end_session()

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
    # Per-trial stages only; "total session duration" has one entry per run
    stages = [k for k in all_runs[0]
              if k.endswith("() duration") and k != "make_session() duration"]
    n = sum(len(run[stages[0]][args.warmup:]) for run in all_runs) if stages else 0
    print(f"\nTiming summary over {n} trials (excluding first {args.warmup} per run), seconds:")
    print(f"{'stage':<40}{'median':>10}{'IQR':>10}{'Q1':>10}{'Q3':>10}")
    for stage in stages:
        vals = np.array([v for run in all_runs for v in run[stage][args.warmup:]])
        if vals.size == 0:
            print(f"{stage:<40}{'n/a (no trials after warm-up)':>40}")
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        print(f"{stage:<40}{med:>10.4f}{q3 - q1:>10.4f}{q1:>10.4f}{q3:>10.4f}")
    totals = [run["total session duration"][0] for run in all_runs]
    print(f"{'total session duration (per run)':<40}{np.median(totals):>10.4f}")