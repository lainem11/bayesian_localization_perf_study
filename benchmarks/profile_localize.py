import torch
from torch.profiler import profile, ProfilerActivity, record_function
from shapes import make_session
import argparse
import merged_adam
from env_info import collect_env

p = argparse.ArgumentParser()
p.add_argument("--merged_adam", action="store_true")
args = p.parse_args()
if args.merged_adam:
    merged_adam.install()                  # before make_session(), as in run_session.py
for k, v in collect_env().items():
    print(f"{k}: {v}")

S = make_session()
for _ in range(10):                      # warm up and build up some trials
    S.start_trial(); d = S.choose_next_stimulus(); S.record_efield_trial(d)
    S.simulate_response(); S.localize(); S.complete_trial()

torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        with record_function("localize"):
            S.localize()
    torch.cuda.synchronize()

S.end_session()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=60))
prof.export_chrome_trace("results/trace_localize.json")

for e in prof.key_averages():
    if e.key in ("cudaLaunchKernel", "aten::item", "aten::_local_scalar_dense", "cudaStreamSynchronize"):
        print(f"{e.key:<30} calls={e.count:<8} cpu_total={e.cpu_time_total/1e3:.1f} ms")