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

The result file "2026-10-20_baseline_r3.json" also shows that the duration is significantly longer for the first trial. With choose_next_stimulus, this is expected due to the JIT precompilation which takes around 4 seconds. The other functions are also slower on the first trial but only for the very first repeat. This is probably because PyTorch loads GPU kernels and reserves memory.

The result file also confirms the determinism, as the didt values which the electric field optimizer returns are equal across repeats.

### Hotfix

I removed implementation of a random seed in run_session.py because it had not effect, since seeds are set inside the LocalizationSession class, and it's verified to work.