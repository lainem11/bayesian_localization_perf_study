Testing done on WSL2. This might add some overhead to every launch and sync, compared to native Linux. 


# Baseline test

The script "run_session.py" runs simulations of the ABL localization session, including session setup and iterations of electric field optimization, response simulation, and localization analysis. The execution times of the key functions are recorded for each trial, among other information, stored in /results.

Repeating sessions are verified to be bitwise identical across 3 repeats (didt, responses, Adam steps) due to the default seeds that are created during the LocalizationSession instantiation.

Here's the command I ran:
```
python run_session.py --repeats 3 --warmup 1 --out results/2026-10-20_baseline_r3.json
```
The command defaults to 50 trials. In the end, it prints a table of the per-function execution times, which are wall-clock with a sync around each stage:

Timing summary over 147 trials (excluding first 1 per run), seconds:

| Stage | Median (s) | IQR (s) | Q1 (s) | Q3 (s) |
|---|---:|---:|---:|---:|
| `choose_next_stimulus()` duration | 0.3396 | 0.0008 | 0.3391 | 0.3400 |
| `record_efield_trial()` duration | 0.0002 | 0.0000 | 0.0002 | 0.0002 |
| `simulate_response()` duration | 0.0011 | 0.0001 | 0.0010 | 0.0012 |
| `localize()` duration | 0.5863 | 0.0196 | 0.5766 | 0.5962 |

Total session duration (per run): 49.6045 s

It shows that choose_next_stimulus() and localize() are the duration bottlenecks, with localize() taking about two times longer in comparison.

The result file "2026-10-20_baseline_r3.json" also shows that the duration is significantly longer for the first trial. With choose_next_stimulus, this is expected due to the JIT precompilation which takes around 4 seconds. The other functions are also slower on the first trial but only for the very first repeat. This is probably because PyTorch loads GPU kernels and reserves memory. The number of compilations done by JAX drops after the first session, so repeated sessions re-use some operations.

The result file also confirms the determinism, as the didt values which the electric field optimizer returns are equal across repeats.

### Hotfix

I removed implementation of a random seed in run_session.py because it had not effect, since seeds are set inside the LocalizationSession class, and it's verified to work.

## Baseline test summary

I now have a reproducible benchmark harness. The results file records the environment and git state, and it times every stage of every trial, with syncs around each stage. The session runs are bitwise deterministic across repeats and processes.

On the benchmark system, `localize()` has a 0.586 s median per trial (IQR 0.020) and `choose_next_stimulus` a 0.340 s (IQR 0.001). Each session is around 50 s.

First trial costs split into JIT recompilation in every session (~3.7 s) and PyTorch initializations.

# Profiling

## localize()

Running the script `profile_localize.py` run PyTorch Profiler to calculate CPU and CUDA durations for various functions calls inside the `localize()` function. The profiling records three calls of the `localize()` function, after a warm-up, and saves the results as a table. I used the command:
```
python profile_localize.py > results/profile_localize_2026-10-20.txt
```

code = 4fed753 + the script as committed in 7fb35e9 / 63622a0

The profiling slows down the execution somewhat, with 2.735/3 = 0.91 s duration per call compared to 0.59 recorded at the baseline. Most of the time goes on the CPU: Python code and launch overhead. The whole-run GPU busy fraction was 205 ms / 2.735 s = 7.5%. A total of 85809 CUDA kernels were launched, with about 1006 ms / 85809 = 12 us of CPU time and 205 ms / 85809 = 2.4 us of GPU time per kernel. Therefore, CPU dominates the per-kernel times. One Adam step launches around 190 kernels. The Adam step uses on average 1.271 ms of CPU time versus 140 us of GPU time. So GPU is busy roughly 10% of the time. The computation is launch-bound and the fix is to send fewer launches and less Python per step. We also see that cudaStreamSynchronize has a total CPU time of 10.7 ms. This around 0.4% of the wall time, so the sync is not a bottleneck to the performance here.

## choose_next_stimulus()

Running `profile_de.py` profiles a single run of the 200-step differential evolution optimization inside the `choose_next_stimulus()` method, after a 3-trial warm-up. I ran it with:
```
python profile_de.py > results/profile_de_2026-10-21.txt
```
The results show a median of 2.5 ms per objective call when each step is synced. Without sync, an iteration averages just 1.5 ms. We can see the kernel stats from the XLA HLO dump, which can be formatted nicely with command:
```
python hlo_traffic.py results/de_objective_hlo.txt > results/de_objective_traffic.txt
```
The number of kernels launched was 22. A cuBLAS gemm kernel multiplies a [22332 x 5] matrix by [5 x 512] and gets [22332 x 512]: that is the E-field of every candidate at every vertex. This operation writes roughly 46 MB. The data sizes are estimated from the HLO by `hlo_traffic.py`, assumign each inputs and outputs are read/written once. A large chain of operations of E-field magnitude, angle, sigmoid, and logs are fused in to a single kernel: `loop_add_log_fusion`.

The total read/write is 430 MB per call, which with the 1.541 ms duration 280 GB/s. This is close to the 4060 Ti limit of 288 GB/s, so the differential evolution may be memory-bound. The choose_next_stimulus() was timed to 0.3396 s, which is on average 1.7 ms per step. The objective evaluation thus takes around 90% of the time during the optimization.