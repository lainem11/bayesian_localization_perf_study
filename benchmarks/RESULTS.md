# ABL GPU performance study: results log

Working log for a hands-on GPU performance study of [ABL (Adaptive Bayesian Localization)](https://github.com/lainem11/bayesian_localization). Each entry records the change or experiment, the hypothesis, the command, the numbers and the conclusion. Every number below comes from a file in `benchmarks/results/`. Where a number was derived by hand from such a file, the derivation is stated.

**Status:** Phase 0 (baseline), Phase 1 (profiling), and Phase 2 (framework-level optimization) are complete. Phase 3 (Triton) is next.

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti, 16 GB |
| Driver | 610.88 (Windows driver, used through WSL2) |
| OS | Windows 11 + WSL2 (Linux 6.18.54.1-microsoft-standard-WSL2, glibc 2.43) |
| Python | 3.13.16 |
| PyTorch | 2.12.0+cu126 (built with CUDA 12.6) |
| JAX / jaxlib / jax-cuda12-plugin | 0.7.2 / 0.7.2 / 0.7.2 |
| evosax / numpyro | 0.2.0 / 0.20.1 |
| Profilers | PyTorch Profiler; Nsight Systems CLI 2026.5.1 and Nsight Compute, both installed from NVIDIA's packages inside WSL |

**Limitations.** WSL2 adds overhead to every kernel launch and synchronization compared with native Linux, and the GPU also drives the display. Small-kernel timings are therefore somewhat pessimistic.

**Branch and code state.** Work is on branch `perf-study`. The JAX/PyTorch E-field projection inconsistency (mean subtraction applied only in JAX) was fixed upstream in `61a9084` *before* any measurement; its numerical effect was on the order of 1e-6. A shutdown fix for `torch.cuda.empty_cache()` followed in `ffd7b30`. No other `src/` changes have been made.

**Workflow.** Commit code → run on a clean tree → commit results and the log entry. Every results file records the git commit and a dirty flag. The dirty flag ignores `benchmarks/results/`, so redirecting output into that folder doesn't mark the run dirty.

### Benchmark scripts (`benchmarks/`)

| Script | Purpose |
|---|---|
| `env_info.py` | `collect_env()`: git state, GPU, driver, library versions, XLA preallocation setting |
| `shapes.py` | `make_session()` builds a standard session; prints tensor shapes |
| `run_session.py` | Benchmark harness (see Phase 0) |
| `compare.py REF NEW` | Checks two results files for bitwise-identical `didt`, responses and Adam step counts |
| `profile_localize.py` | PyTorch Profiler on 3 `localize()` calls |
| `profile_de.py` | Times the DE objective (synced and back-to-back) and dumps its compiled HLO |
| `hlo_traffic.py` | Estimates per-kernel memory traffic from an HLO dump |
| `simple_sync_test.py` | Demonstrates asynchronous GPU execution |

---

# Phase 0: Baseline

## 0.1 Problem size

From `shapes.py`:

| Tensor | Shape | dtype / device |
|---|---|---|
| `efield_set` | [5, 7444, 3] (0.43 MB) | float32, cuda |
| `efield_basis` | [7444, 3, 2] | float32, cuda |
| `null_dir_basis` | [1, 7444, 2] | float32, cuda |
| `efield_trials` (trial buffer) | [150, 7444, 3] | float32, cuda |

5 coils, 7444 ROI vertices. The trial buffer is preallocated for 150 trials, so a 50-trial session never reallocates.

## 0.2 Harness

`run_session.py` runs simulated sessions (source vertex 2679, `silent_mode=True`).

- **Timing:** every stage of every trial (`choose_next_stimulus`, `record_efield_trial`, `simulate_response`, `localize`) is timed with `time.perf_counter()`, with `torch.cuda.synchronize()` before and after. Without the syncs, a timer measures only how long the work took to *queue*, because GPU calls return before the GPU has finished. `make_session()` is timed the same way, and each stage is wrapped in an NVTX range for Nsight Systems.
- **Recorded per trial:** chosen `didt`, response, number of Adam steps, posterior argmax (read outside the timed sections, because it forces a sync).
- **Warm-up:** all trials are stored; the summary excludes the first `--warmup` trials of each run.
- **Seeding:** `LocalizationSession.__init__` and `initialize_trials` call `_utils.set_random_seed()` (seed 10), and the optimizer seeds its Sobol sampler and JAX key with 0. Seeds set outside the session are overwritten, so the harness has no `--seed` option.
- **Caveat:** the syncs around each stage stop the CPU from queuing the next stage while the GPU is busy, so stage times can add up to slightly more than an unsynced trial.

## 0.3 Baseline timing

Command (commit `45cd423`, clean tree):
```
python run_session.py --repeats 3 --warmup 1 --out results/2026-10-20_baseline_r3.json
```
50 trials × 3 repeats; statistics over 147 trials (first trial of each run excluded), wall-clock with a sync around each stage:

| Stage | Median (s) | IQR (s) | Q1 (s) | Q3 (s) | Share of trial |
|---|---:|---:|---:|---:|---:|
| `choose_next_stimulus` | 0.3396 | 0.0008 | 0.3391 | 0.3400 | 36.6% |
| `record_efield_trial` | 0.0002 | 0.0000 | 0.0002 | 0.0002 | <0.1% |
| `simulate_response` | 0.0011 | 0.0001 | 0.0010 | 0.0012 | 0.1% |
| `localize` | 0.5863 | 0.0196 | 0.5766 | 0.5962 | 63.2% |

Total session time: 51.8 s, 49.6 s and 49.5 s for the three runs. The first run is longer because it includes one-time process initialization.

**`localize` doesn't get slower as trials accumulate.** It fits on 2 trials at the start and 50 at the end, yet stays at ~0.55–0.62 s. Its time follows the number of Adam steps instead: trials that converged early (60, 103 and 121 steps instead of 150) were proportionally faster. Dividing `localize` time by Adam steps in the baseline file gives **~3.9 ms per step**, regardless of step count or trial count. This was the first hint that `localize` is limited by fixed per-step overhead, not by arithmetic (confirmed in Phase 1).

## 0.4 First-trial cost

| Trial 1 | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| `choose_next_stimulus` | 4.24 s | 3.70 s | 3.75 s |
| `localize` | 2.31 s | 0.59 s | 0.59 s |
| `simulate_response` | 0.15 s | 0.001 s | 0.001 s |

- **Once per session: JAX compilation (~3.7 s).** Each new `LocalizationSession` creates new jitted functions, so the DE scan (`jit(scan)`, ~1.8 s to compile in one logged run) and the objective (~0.8 s) recompile in every session. With `jax.config.update("jax_log_compiles", True)`, there were no compilations after trial 1 within a session. One session logged 27 compilations and three sessions logged 68, so later sessions reuse only a few small one-off operations. (Compile counts are from console output that wasn't saved.)
- **Once per process: PyTorch initialization.** `localize` and `simulate_response` are slow only in the first session of the process: PyTorch loads GPU kernels and libraries and reserves memory on first use.
- One warm-up trial per session is enough.

## 0.5 Session setup

Command (commit `51d02dd`, clean tree):
```
python run_session.py --repeats 1 --trials 3 --out results/2026-10-21_baseline_r1.json
```
`make_session()` took **4.26 s** (single run, first session in the process, so it includes CUDA initialization). The breakdown is not measured yet; see Phase 1 for the QR kernels it launches.

## 0.6 Determinism and equivalence checking

- Within `2026-10-20_baseline_r3.json`, all three runs have bitwise-identical `didt`, responses and Adam step counts, and all end at the true source vertex 2679.
- `compare.py results/2026-10-20_baseline_r3.json results/2026-10-20_baseline_r3_rerun.json` reports **all identical**. The rerun was a separate process (commit `631beb9`, dirty tree with benchmark-only changes), so the result holds across processes too.

Two levels of equivalence will be used in Phase 2:
1. **Bitwise (`compare.py`)** for changes that shouldn't touch the maths (removing a sync, caching a value). The whole trajectory must stay identical.
2. **Tolerance-based, per stage on fixed inputs** for changes that alter floating-point order (`torch.compile`, custom kernels, dtypes). The DE feedback loop amplifies tiny differences, so whole trajectories can't be compared. `localize` will be replayed on recorded trials (`import_trial_data`) and the DE objective evaluated on a fixed population. Not built yet.

## Phase 0 summary

**Can claim:**
- Built a reproducible benchmark harness that records the environment and git state and times every stage of every trial with explicit synchronization.
- Baseline on an RTX 4060 Ti (WSL2), 50-trial session, 3 repeats: `localize` 0.586 s (IQR 0.020) and `choose_next_stimulus` 0.340 s (IQR 0.001) per trial; ~50 s per session; `make_session` 4.26 s.
- First-trial cost splits into JAX recompilation per session (~3.7 s) and PyTorch initialization per process.
- Sessions are bitwise deterministic across repeats and processes, with a script to verify it.

**Cannot claim yet (at the end of Phase 0):** why either stage is slow; any speedup.

---

# Phase 1: Profiling

## 1.1 Asynchronous execution check

`simple_sync_test.py` times an 8192 × 8192 matrix multiply. The call *returned* after 0.191 ms, but the work *finished* after 84.7 ms (console output, not saved). GPU calls return as soon as the work is queued, so timing requires an explicit sync (`torch.cuda.synchronize()`, `.item()`, or `block_until_ready()` in JAX). That's why the harness syncs around every stage.

As a side calculation: 2 × 8192³ ≈ 1.1 × 10¹² operations in 84.7 ms is ~13 TFLOP/s for this matmul.

## 1.2 `localize()` with PyTorch Profiler

Command (code = `4fed753` + the script as committed in `7fb35e9`; this output has no environment header):
```
python profile_localize.py > results/profile_localize_2026-10-20.txt
```
The script warms up with 10 trials, then profiles 3 `localize()` calls (each labelled with `record_function("localize")`).

**Reading the table.** With GPU recording on, each labelled region appears twice:
- A **CPU-side row** (CPU columns filled). Its *CUDA total* is the summed duration of the kernels launched inside it: GPU *busy* time.
- A **GPU-side marker row** (CPU columns zero, odd percentages above 100%). Its *Self CUDA* is the window on the GPU timeline from first to last kernel, *including idle gaps*.

**Results:**

| Quantity | Value | Source (row, column) |
|---|---:|---|
| Elapsed time, 3 calls | 2.735 s (0.91 s per call; profiler overhead vs 0.59 s in the harness) | `localize` CPU row, CPU total |
| GPU kernel time, whole run | 204.9 ms | last line, Self CUDA time total |
| GPU busy fraction | 204.9 / 2735 = **7.5%** | derived |
| Kernel launches | **85,809** (≈ **190 per Adam step**: 3 calls × 150 steps = 450 steps) | `cudaLaunchKernel` calls |
| CPU time per launch | 1006.1 ms / 85,809 = **11.7 µs** | `cudaLaunchKernel` cpu_total |
| GPU time per kernel (average) | 204.9 ms / 85,809 = **2.4 µs** | derived |
| Adam step | 1.271 ms CPU vs 139.6 µs GPU per step (~11% busy) | `Optimizer.step#Adam.step` CPU row, CPU time avg and CUDA time avg |
| Example op `aten::mul` | 20.5 µs CPU vs 1.9 µs GPU per call; 24,345 calls (54 per step) | `aten::mul` row |
| Python time | 1.474 s Self CPU inside `localize` (time not inside any PyTorch op) | `localize` CPU row, Self CPU |
| GPU syncs | 465 `cudaStreamSynchronize` = 10.7 ms total | sync rows |

**Interpretation.**
- A **kernel** is one function the CPU asks the GPU to run over a whole tensor. In eager PyTorch that's roughly one kernel per tensor operation.
- Each kernel finishes in ~2 µs, but the CPU needs ~12 µs to launch the next one, plus Python time. The GPU is idle ~92% of the time waiting for work. `localize` is **CPU-overhead-bound** (launch-bound plus Python), which explains the constant ~3.9 ms per Adam step from Phase 0.
- **Syncs are cheap here.** The 465 syncs are 450 `loss.item()` calls (one per step) plus 15 others (5 per `localize`: the `vector_angle` check, `isnan().any()`, etc.). Together they cost 10.7 ms, **~0.4% of elapsed time**. A sync waits for the GPU to finish its queue, and this GPU is almost always already idle.
- **Non-sync `.item()` calls.** `aten::item` ran 6,312 times (14 per step), but most read CPU tensors and don't wait for the GPU. Likely breakdown: 1 is `loss.item()`, 1 is the convergence `if` on the CPU `losses` tensor, and 12 are Adam reading its CPU step counters (6 groups × 2).
- **Lead for Phase 2.** The Adam `_foreach_*` kernels each run 2,700 times = 6 per step. `_setup_optimizer` puts each of the 6 parameters in its own group with its own learning rate, so the "foreach" batching launches once per group instead of once in total.

## 1.3 DE objective in JAX

Command (commit `7a0fb51`, clean tree):
```
python profile_de.py > results/profile_de_2026-10-21.txt
python hlo_traffic.py results/de_objective_hlo.txt > results/de_objective_traffic.txt
```
`profile_de.py` runs 3 trials, then calls the jitted objective on a random population of 512 candidates.

**Timing:**

| Measurement | Value |
|---|---:|
| Synced (wait after every call), median | 2.481 ms (IQR 0.033) |
| Back-to-back (200 calls queued, one wait at the end) | **1.544 ms per call** |
| Whole DE per generation (0.3396 s / 200, baseline) | 1.70 ms |

The synced time includes ~0.94 ms per call of dispatch and waiting, which disappears inside the scan. **The objective is ~91% of each DE generation** (1.544 / 1.70). evosax's ask/tell make up the rest (~0.15 ms).

**What XLA compiled** (from `de_objective_hlo.txt`; `hlo_traffic.py` lists each kernel with its read and write sizes):
- **22 kernels** per objective call.
- The vmapped einsum over coils became **one cuBLAS matrix multiply**, [22332 × 5] × [5 × 512] → [22332 × 512] (22332 = 7444 vertices × 3 components). That is the E-field of all 512 candidates at all vertices: 46 MB written.
- `efield_set * max_current_slope` was folded into a compile-time constant (`constant_13_0`), so it isn't recomputed per candidate.
- The whole elementwise chain (magnitude, angle, sigmoid, logs, both posterior numerators) is **fused into one kernel**, `loop_add_log_fusion`.
- The remaining kernels are reductions: two passes per logsumexp (max, then sum), the expected change, and the E-field max penalty, which re-reads the 46 MB E-field array.

**Memory traffic model.** Assuming each kernel reads its inputs once and writes its outputs once: 291.6 MB read + 138.6 MB written = **~430 MB per call**. 430 MB / 1.544 ms ≈ **278 GB/s**, against NVIDIA's spec of ~288 GB/s for the 4060 Ti. Reaching ~97% of spec is unusually high, which suggests part of the traffic is served from the 32 MB L2 cache. With 512 candidates the main intermediate arrays are 15–46 MB each, mostly larger than L2.

*Note:* `de_objective_hlo.txt` was regenerated when `profile_de.py` ran under Nsight Compute (commit `51d02dd`). Running `hlo_traffic.py` on the new file gives the same 22 kernels and 430 MB.

**Hypothesis after this step:** the DE objective is memory-bound because it writes and re-reads ~430 MB of large intermediate arrays per generation (tested in 1.5).

## 1.4 Whole session with Nsight Systems

Two traces of a 10-trial session, both with commit `f840466` code (NVTX labels added):
```
# Trace A: default CUDA-graph tracing; includes API and memory-copy summaries
nsys profile -o results/nsys_session --trace=cuda,nvtx,osrt --force-overwrite=true \
    python run_session.py --trials 10 --warmup 1 --out results/nsys_run.json
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,cuda_gpu_mem_time_sum \
    results/nsys_session.nsys-rep > results/nsys_stats_2026-10-21.txt

# Trace B: kernels inside CUDA graphs traced individually
nsys profile -o results/nsys_session_nodes --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
    --force-overwrite=true python run_session.py --trials 10 --warmup 1 --out results/nsys_run_nodes.json
nsys stats --report nvtx_gpu_proj_sum,cuda_gpu_kern_sum \
    results/nsys_session_nodes.nsys-rep > results/nsys_stats_nodes_2026-10-21.txt
```
(Trace B also includes the `make_session()` timer added in `0fea58b`. Ubuntu's own `nsight-systems` package couldn't convert traces because its importer was missing; NVIDIA's package works.)

**Stage times under nsys (NVTX ranges = CPU wall-clock, medians):**

| Stage | nsys median | Harness median | Change |
|---|---:|---:|---:|
| `choose_next_stimulus` | 330.8 ms | 339.6 ms | ≈ same |
| `localize` | 935.4 ms | 586.3 ms | **+60%** |

nsys records every CUDA API call, and this session made 621,335 `cudaLaunchKernel` calls (Trace A, `cuda_api_sum`). Launch-heavy code suffers most from that recording overhead, so `localize`'s slowdown is further evidence that it is launch-bound. Means are misleading here because trial 1 takes 2.6 s and 6.9 s, so medians are used.

**CUDA graphs in JAX.** In Trace A the DE kernels are missing from the kernel summary. Instead there are 4,040 `cuGraphLaunch` calls and NVTX ranges named `command_buffer`. XLA records the scan body's kernels once as a **CUDA graph** and replays the whole sequence with a single launch call, which removes per-kernel launch overhead. This is why the DE is not launch-bound. Trace B (`--cuda-graph-trace=node`) lists the kernels inside the graphs.

**GPU time by stage** (derived by hand from Trace B `cuda_gpu_kern_sum`, 10 trials, grouping kernels by name and instance count):

| Group | How identified | GPU time | Share (excl. setup) |
|---|---|---:|---:|
| DE loop (JAX) | ~2,000–2,016 instances each (`loop_add_log_fusion`, `gemm_fusion_dot_general_9`, `triton_softmax_1`, reductions, `ampere_sgemm`, …) | 3,596 ms | **~88%** |
| PyTorch | `at::native::…` kernels | 464 ms | **~12%** |
| Session setup | 37,220 / 74,440 instances (QR, see below) | 383 ms | – |

**The DE uses ~7× more GPU time than `localize`, while `localize` takes more wall time.** On average the DE kernels add up to ~360 ms per trial (including trial 1), about the same as the DE's ~331 ms wall time. So the GPU is busy nearly the whole time during the DE ("~100%" is approximate, because node-level tracing inflates kernel durations). During `localize` the GPU is mostly idle.

**DE kernel durations** (Trace B, average) with read+write sizes from the HLO model:

| Kernel | Avg time | MB moved (model) | Implied bandwidth |
|---|---:|---:|---:|
| `loop_add_log_fusion` | 496 µs | 107 | ~216 GB/s |
| `gemm_fusion_dot_general_9` (2D projection) | 381 µs | 76 | ~200 GB/s |
| `triton_softmax_1` (normalization) | 256 µs | 61 | ~238 GB/s |
| `input_reduce_fusion` (E-field max penalty) | 212 µs | 46 | ~216 GB/s |
| `ampere_sgemm_128x64_tn` (matmul) | 118 µs | 46 | ~390 GB/s |

The matmul's apparent bandwidth exceeds the spec, which is only possible if part of its 46 MB output stays in L2 and is read from there by the next kernel. The per-kernel model doesn't capture caching. These durations add up to ~1.75 ms per generation, above the 1.544 ms back-to-back time, which is consistent with node-level tracing adding a small cost per kernel.

**Note on `nvtx_gpu_proj_sum`.** Its "projected time" is the span on the GPU timeline from the start of a range's first kernel to the end of its last, gaps included, *not* the summed busy time. For `localize`, projected time (11,049 ms) ≈ range time (11,051 ms). Busy time therefore has to come from the kernel summary, as above.

**Session setup (QR decomposition).** The kernels `larft`, `orgqr`, `lacpy`, `batch_eye` and `copy_info_kernel` each ran exactly 37,220 times, and `gemmSN_NN` ran 74,440 times. These are the building blocks of a QR decomposition, called from `torch.pca_lowrank` in `EFieldModel.apply_pca` during `make_session()`. The template arguments `<float, 5, 3>` show each vertex's matrix is 5 × 3 (coils × xyz). 37,220 = 7444 × 5. *Hypothesis:* `pca_lowrank` calls QR 5 times (its default `niter=2`), and the batched QR is solved one tiny matrix at a time. Under nsys `make_session()` took 7.27 s, vs 4.26 s without nsys (0.5), consistent with ~300k traced launches. A breakdown of setup time is still to do.

**Compile-time autotuning.** NVTX shows XLA's `gemm-fusion-autotuner` pass taking 659 ms during the first compilation of the objective. The kernels `xla_fp_comparison`, `Redzone…` and two ~28 ms `cutlass` kernels are the autotuner trying candidate GEMM implementations. This is part of the trial-1 cost.

**Host↔device transfers** (Trace A, `cuda_gpu_mem_time_sum`): host→device 1.4 ms, device→host 2.9 ms of GPU time for the whole session. **Negligible.**

## 1.5 One kernel with Nsight Compute

Command (commit `df9cc66`, clean tree):
```
ncu --kernel-name regex:loop_add_log_fusion --launch-skip 20 --launch-count 1 \
    --section SpeedOfLight \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
    python profile_de.py > results/ncu_loop_add_log_2026-10-21.txt
```
`loop_add_log_fusion` is the largest single kernel (~21% of all GPU time in Trace B). Nsight Compute replays it while reading hardware counters. Grid: 7444 blocks × 128 threads.

| Metric | Measured | Model (`hlo_traffic.py`) |
|---|---:|---:|
| DRAM bytes read | 77.2 MB | 76.5 MB |
| DRAM bytes written | 20.6 MB | 30.5 MB |
| Duration | 377 µs | – |
| Memory (DRAM) throughput | **92.7%** of peak | – |
| Compute (SM) throughput | **28.1%** of peak | – |

- **Memory-bound.** Memory throughput is near its limit while the arithmetic units are mostly idle.
- **Bandwidth cross-check:** 97.8 MB / 377 µs ≈ 259 GB/s. ncu reports a DRAM frequency of 8.74 GHz, which on the 128-bit bus gives a ~280 GB/s peak; 259 / 280 ≈ 93%, consistent with the reported throughput.
- **Fewer writes than the model.** About 10 MB of the output is still in L2 when the kernel ends, and the next kernels read it from there. The model is not a lower bound: caching can make real traffic smaller.
- **Caveats.**
  - The duration here (377 µs) is shorter than the 496 µs in the node-level nsys trace. Node tracing inflates kernel times, so per-kernel figures should come from ncu.
  - The timing lines at the top of this output file were measured while ncu was replaying kernels and are not valid timings.

## 1.6 Bottleneck table

Wall-time shares: harness baseline (`2026-10-20_baseline_r3.json`). GPU-time shares: Trace B (`nsys_stats_nodes_2026-10-21.txt`), session setup excluded.

| # | Hypothesis | Stage | Share of wall time | Share of GPU time | Evidence (source) | Verdict |
|---|---|---|---:|---:|---|---|
| 1 | The DE objective dominates GPU time per trial | `choose_next_stimulus` | 36.6% | ~88% | DE kernels 3,596 ms vs PyTorch 464 ms over 10 trials; GPU busy ~100% during the DE; objective is 91% of each generation (`nsys_stats_nodes`, `profile_de`) | **Confirmed** for GPU time; not the largest share of wall time |
| 2 | The DE is memory-bound because of large intermediate arrays | `choose_next_stimulus` | (as #1) | (as #1) | Elementwise chain fused into 1 kernel; 22 kernels; ~430 MB per call (model). `loop_add_log_fusion`: DRAM 92.7%, SM 28.1%, 77.2 MB read / 20.6 MB written (`de_objective_traffic`, `ncu_loop_add_log`) | **Confirmed** for the largest kernel; others consistent by duration |
| 3 | The Adam loop is launch/overhead-bound | `localize` | 63.2% | ~12% | ≈190 launches per step; 11.7 µs CPU per launch vs 2.4 µs GPU per kernel; GPU busy 7.5%; 1.47 s Python self-time; ~3.9 ms per step regardless of data size; +60% under nsys (`profile_localize`, `baseline_r3`, `nsys_stats`) | **Confirmed**: CPU-side overhead (Python + launches) |
| 4 | `loss.item()` every step is a costly sync | `localize` | ~0.4% | – | 465 syncs (450 in the loop) = 10.7 ms; GPU never behind the CPU (`profile_localize`) | **Rejected**. Phase 2 prediction: removing it changes `localize` by <2% |
| 5 | `vector_angle`'s data-dependent `if` (host sync) matters | `localize` | negligible | – | Among the 15 non-loop syncs per 3 calls, inside the 10.7 ms of #4 (`profile_localize`) | **Rejected** |
| 6 | E-field magnitude and angle recomputed twice per `localize` matter | `localize` | negligible | negligible | A few dozen kernels out of ~28,600 per call (kernel-count estimate, not timed separately) | **Negligible** (estimate) |
| 7 | Per-trial host↔device transfers matter | all | – | <0.5% | H→D 1.4 ms, D→H 2.9 ms for a 10-trial session (`nsys_stats`) | **Rejected** |
| 8 | JAX recompiles | `choose_next_stimulus` | trial 1 only | – | No compiles after trial 1 within a session; ~3.7 s recompile per new session; separately, PyTorch init once per process (`baseline_r3`, compile log) | **Confirmed per session**; none within a session |
| 9 | NVML can't separate PyTorch and JAX memory; use per-framework stats | – | – | – | Not measured | **Not tested** |
| 10 | (new) Session setup is expensive | `make_session` | once per experiment | – | 4.26 s (clean single run, includes process init); QR kernels 37,220 calls each, 383 ms GPU, from `torch.pca_lowrank` (`baseline_r1`, `nsys_stats_nodes`) | **Open**: breakdown needed |
| 11 | (new) XLA autotuning adds to the first compile | `choose_next_stimulus` | trial 1 only | – | `gemm-fusion-autotuner` 659 ms; autotuner kernels in the trace (`nsys_stats`) | **Confirmed** as part of the trial-1 cost |

**In short:** the DE uses most of the GPU and is memory-bound. `localize` uses most of the wall time and is CPU-overhead-bound with the GPU mostly idle. They need different fixes: fewer bytes moved for the DE, fewer launches and less Python per step for `localize`.

## Phase 1 summary

**Can claim:**
- Profiled a mixed PyTorch + JAX application with PyTorch Profiler, Nsight Systems and Nsight Compute, and read XLA's compiled HLO to see what the compiler actually generated.
- Showed the PyTorch fitting loop is CPU-overhead-bound: ~190 kernel launches per Adam step, ~12 µs of CPU per launch vs ~2.4 µs of GPU per kernel, GPU busy ~7.5%. Showed its per-step syncs cost <1%, rejecting the original sync hypothesis.
- Showed the JAX DE objective is memory-bound: XLA fuses it into 22 kernels moving ~430 MB per call (estimated from HLO). The largest kernel reaches 92.7% of DRAM bandwidth at 28% compute throughput, with measured bytes close to the model.
- Found that JAX runs the DE loop as CUDA graphs, which is why it isn't launch-bound, and that the DE takes ~88% of GPU time but only ~37% of wall time.
- Found additional costs: session setup (4.26 s, with a QR decomposition launched per vertex) and XLA autotuning in the first compile.

**Cannot claim yet:** any speedup; per-framework memory use; a breakdown of session setup; `localize(full=True)` (offline mode, not profiled).

---

## Appendix: pitfalls found

- Ubuntu's `nsight-systems` package lacks the `.qdstrm` importer. Use NVIDIA's packages.
- PyTorch Profiler marker rows (zero CPU time) show GPU-timeline windows, not kernel time.
- `nvtx_gpu_proj_sum` reports the span on the GPU timeline, not busy time.
- `--cuda-graph-trace=node` is needed to see JAX's kernels, but it inflates kernel durations.
- Profilers distort timing: use them for counts and proportions, and the harness for times.
- `Self CPU time total` adds up time across threads (backward runs on its own thread), so it can exceed elapsed time.
- Use medians: trial 1 skews means.
- Redirecting output into tracked files marked the tree dirty; `collect_env` now ignores `benchmarks/results/`.

# Phase 2: Framework-level optimization

## 2.0 Reproducibility: XLA autotuning

The 10-22 baseline at `1ba1067` (`2026-10-22_baseline_r3_unpinned.json`) differed from `2026-10-20_baseline_r3.json` from trial 1, with no code changes. The committed HLO dumps show why: the E-field matmul is cuBLAS in `de_objective_hlo.txt` at `63622a0` and a Triton GEMM at `51d02dd`. XLA's autotuner picks the faster implementation at compile time, and the choice changed between processes. The two kernels round differently, and the DE amplifies the difference. (A run with `--xla_gpu_enable_triton_gemm=false` restored the old trial-1 `didt`; that output was not saved.)

Fix: autotuning results are saved once and loaded in every run ([persisted autotuning](https://openxla.org/xla/persisted_autotuning)). `autotune/rtx4060ti.textproto` (`a50aa8b`) pins cuBLAS for the E-field matmul, i.e. the original state. With it loaded, `compare.py results/2026-10-20_baseline_r3.json results/2026-10-22_baseline_pinned_r3.json` reports all identical. `collect_env()` records `XLA_FLAGS`. All Phase 2 runs load this file.

Day-to-day drift with identical trajectories: `localize` median 0.5863 s (10-20) vs 0.5730 s (10-22 pinned), `choose_next_stimulus` 0.3396 s vs 0.3436 s. Effects below ~2% therefore need before and after runs interleaved in one sitting.

## 2.1 Step (a): per-step `loss.item()`

**Hypothesis.** The per-step sync is cheap (Phase 1: 465 syncs = 10.7 ms over 3 calls, ~0.6% of `localize`).

**Method.** `fixed_steps.py` replaces the fitting loop at run time with one that runs the step counts recorded in the reference file, with no `.item()` and no convergence check. Same arithmetic, so the trajectory must stay identical. Runs alternated A B A B, commit `4720545`, autotuning file loaded:
```
python run_session.py --repeats 3 --out results/2026-10-22_a_base1.json
python run_session.py --repeats 3 --fixed_steps results/2026-10-22_baseline_pinned_r3.json --out results/2026-10-22_a_fixed1.json
(same again as a_base2, a_fixed2)
```

**Results.** `compare.py` against the pinned baseline: all identical for all four files. `localize` median (IQR) over 147 trials per file:

| base1 | fixed1 | base2 | fixed2 |
|---:|---:|---:|---:|
| 0.5912 (0.0153) | 0.5845 (0.0137) | 0.5972 (0.0299) | 0.5831 (0.0183) |

**Conclusion.** Both fixed runs are faster than both baseline runs, by about 7–14 ms (1–2%). The two baseline runs differ from each other by 6 ms, so the effect is only slightly larger than the run-to-run variation. Removing the sync gains at most ~1–2%. A deployable version would change step counts (level-2 equivalence), so it is not pursued now.

## 2.2 Step (b1): merged Adam parameter groups

**Hypothesis.** In `profile_localize_2026-10-20.txt`, each `aten::_foreach_*` op has 2700 calls = 6 per Adam step (2700 / 450 steps). `_setup_optimizer` puts each of the 6 parameters in its own parameter group, and `foreach=True` batches tensors only within a group, so every op launches 6 times. One update over all 6 tensors should cut launches and speed up the launch-bound `localize`.

**Method.** `merged_adam.py` replaces `_setup_optimizer` at run time with an Adam that makes each of the 7 foreach calls of PyTorch's `_multi_tensor_adam` once over all 6 tensors, with a per-tensor list of learning rates. Same operations in the same order, so the trajectory must stay identical. Runs at commit `6d66aeb`, clean tree, autotuning file loaded, alternating A B A B:
```
export XLA_FLAGS=--xla_gpu_load_autotune_results_from=$PWD/autotune/rtx4060ti.textproto
python run_session.py --repeats 3 --out results/2026-10-22_b1_base1.json
python run_session.py --repeats 3 --merged_adam --out results/2026-10-22_b1_merged1.json
(same again as b1_base2, b1_merged2)
python compare.py results/2026-10-22_baseline_pinned_r3.json results/2026-10-22_b1_merged1.json
```

**Results.** `compare.py`: all identical. `localize` median (IQR) over 147 trials per file:

| base1 | merged1 | base2 | merged2 |
|---:|---:|---:|---:|
| 0.5813 (0.0188) | 0.4749 (0.0185) | 0.5853 (0.0168) | 0.4778 (0.0190) |

`localize` is ~18% faster (≈ −107 ms per call); the two baseline runs differ by 4 ms.

**Profile** (commit `82fcac5`, 3 `localize` calls = 450 Adam steps):
```
python profile_localize.py > results/profile_localize_b1_base.txt
python profile_localize.py --merged_adam > results/profile_localize_b1_merged.txt
```

| | base | merged |
|---|---:|---:|
| `cudaLaunchKernel` calls | 85,809 | 70,059 |
| `cudaLaunchKernel` CPU total | 1051.4 ms | 878.5 ms |
| `localize` CPU total | 2.897 s | 2.428 s |
| Self CUDA time total | 189.7 ms | 161.4 ms |

**Conclusion.** Launches dropped by exactly 7 ops × 5 × 450 = 15,750: each foreach op now launches once per step instead of 6 times. Launch CPU time explains only 0.173 s (37%) of the 0.469 s drop in `localize` CPU time; the rest is other CPU time, not yet broken down. The biggest remaining op is `aten::mul` (24,345 calls ≈ 54 per step, forward and backward of the loss), a candidate for fusion with `torch.compile`.

## 2.3 Step (b2): `torch.compile` of the loss

**Hypothesis.** Eager PyTorch runs each operation as soon as Python reaches it: one kernel and one intermediate tensor per operation (≈54 `aten::mul` per Adam step, 2.2). Compiling the loss (forward and backward) with `torch.compile` (Inductor backend) should fuse these into a few kernels and cut launches.

**Method.** `compiled_loss.py` keeps the original fitting loop (including `loss.item()` and the convergence check) and runs only the loss through `torch.compile`. It is stacked on merged Adam, so the comparison is merged vs merged + compile. Fusion changes the order of floating-point operations, so the trajectory is no longer bitwise identical and `compare.py` no longer applies.

**Equivalence (level 2).** `replay_localize.py` feeds the `didt` values and responses of run 0 of a reference file into a fresh session and calls only `localize()`, so every variant fits the same 50 trials; it saves the log-posterior after each trial. `compare_replay.py` compares two such files. Criterion, set before the comparison: from trial 20 on, the maximum relative difference in posterior probability over the vertices holding 95% of the reference mass must be below 5% (earlier trials have too little data and the fit varies a lot).
```
export XLA_FLAGS=--xla_gpu_load_autotune_results_from=$PWD/autotune/rtx4060ti.textproto
REF=results/2026-10-22_baseline_pinned_r3.json
python replay_localize.py $REF results/2026-10-22_replay_base.npz
python replay_localize.py $REF results/2026-10-22_replay_merged.npz --merged_adam
python replay_localize.py $REF results/2026-10-22_replay_compiled.npz --merged_adam --compile_loss
python compare_replay.py results/2026-10-22_replay_base.npz results/2026-10-22_replay_merged.npz
python compare_replay.py results/2026-10-22_replay_merged.npz results/2026-10-22_replay_compiled.npz
```
- Base vs merged: zero difference in every trial (checks the tool).
- Merged vs compiled: max relative difference 4.5% at trial 20 and below 1% from trial 21 on. **Passes.** Total variation distance ≤ 8.65e-4 in every trial (at most 0.09% of the posterior mass moved); argmax equal in 50/50 trials; Adam step counts differ in 3 trials (the convergence threshold reacts to rounding).

**Timing** (commit `2801e83`, clean tree, alternating A B A B):
```
python run_session.py --repeats 3 --merged_adam --out results/2026-10-23_b2_merged1.json
python run_session.py --repeats 3 --merged_adam --compile_loss --out results/2026-10-23_b2_compiled1.json
(same again as b2_merged2, b2_compiled2)
```
`localize` median (IQR) over 147 trials per file:

| merged1 | compiled1 | merged2 | compiled2 |
|---:|---:|---:|---:|
| 0.4480 (0.0098) | 0.1354 (0.0038) | 0.4482 (0.0112) | 0.1353 (0.0047) |

`localize` is 0.313 s (3.3×) faster per trial.

**Compile cost.** A smoke test (`2026-10-22_b2_smoke.json`, `TORCH_LOGS=recompiles`) showed one recompile at trial 2 (`tensor 'e_mag' size mismatch at index 0`); after that the trial dimension is treated as dynamic and no further compiles occur. In `compiled1` run 0, trials 1–2 took 3.79 s and 6.19 s, ≈8.6 s more than in `merged1`, earned back after ~27 trials. Later sessions in the same process didn't compile again. In a new process (`compiled2` run 0), trials 1–2 took 1.02 s and 0.37 s, probably thanks to Inductor's on-disk cache (not verified).

**Closed loop.** The compiled sessions follow a different trajectory and end at vertex 2512 instead of 2679 at trial 50. The two are neighbouring vertices on the mesh (checked by hand), so the difference is acceptable in practice. This is one trajectory per variant, not a statistical comparison.

**Conclusion.** Compiling the loss gives the largest gain so far (3.3× per `localize`) at an acceptable numerical difference, but costs ~10 s of compile time in trials 1–2 of a fresh process.

## Phase 2 summary

**Practical relevance.** In a real experiment, the trial rate is limited by the few seconds required between stimuli, and the per-trial computation already fits within that. A faster online loop therefore brings little practical benefit, while a ~10 s compile stall in trial 2 is a real cost. The practical fix would be to compile during session setup (dynamic trial dimension from the start, one call on dummy data). The offline `localize(full=True)` (~5 min, NumPyro NUTS) would be the target with practical value.

**Not done:** reducing the DE's memory traffic, compiling at setup, per-framework memory statistics, profiling `localize(full=True)`.

**Can claim:**
- Found that XLA's GEMM autotuner chose different kernels in different processes, which changed the trajectory, and pinned its choices with persisted autotuning, restoring a bitwise-reproducible baseline.
- Showed the per-step `loss.item()` sync costs only 1–2% of `localize`, using a bitwise-identical fixed-step replay.
- Merged Adam's 6 parameter groups into one foreach update: `localize` −18%, bitwise-identical trajectory, kernel launches reduced by 15,750 per 450 steps, exactly 7 foreach ops × 5 launches removed per step.
- Compiled the loss with `torch.compile`: `localize` 0.448 → 0.135 s per trial, equivalent under a criterion fixed in advance on replayed data, with ~10 s first-time compile cost.
- Built a two-level equivalence check (bitwise trajectory; per-trial posterior comparison on replayed data).

**Cannot claim:** a practical speedup of the experiment; any change to the DE or to `localize(full=True)`; statistical evidence about closed-loop localization accuracy (one trajectory per variant).