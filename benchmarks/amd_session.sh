#!/usr/bin/env bash
# benchmarks/amd_session.sh
# One AMD cloud session, in order, run from benchmarks/. Each step writes results/amd_<TAG>/<step>.txt;
# a failing step is logged (exit code, seconds) and the session continues, so nothing is lost to one error.
#
# Usage:  bash amd_session.sh mi300x_2026-10-29              # on the AMD instance, inside the ROCm container
#         SKIP_PROFILERS=1 bash amd_session.sh test_4060ti   # dry run on NVIDIA: environment + Python steps only
# Afterwards: everything is in results/amd_<TAG>/ and results/amd_<TAG>.tar.gz
set -u
TAG=${1:?usage: bash amd_session.sh TAG}
PY=${PY:-python}
STATE=results/de_state_2026-10-26.npz
OUT=results/amd_$TAG
STEP_TIMEOUT=${STEP_TIMEOUT:-1800}          # seconds; a hung step is killed, the session goes on
mkdir -p "$OUT"
LOG=$OUT/session_log.txt

step() {                                    # step NAME CMD...: run CMD, output to $OUT/NAME.txt
    local name=$1; shift
    echo "=== $(date -Is) $name: $*" | tee -a "$LOG"
    local t0=$(date +%s)
    timeout "$STEP_TIMEOUT" "$@" > "$OUT/$name.txt" 2>&1
    local rc=$?
    echo "    exit $rc, $(( $(date +%s) - t0 )) s" | tee -a "$LOG"
}

# Inside a container the mounted repository has another owner and git refuses to run ("dubious
# ownership"), so env_light would record the commit as "unavailable". Only then mark it safe.
if ! git rev-parse HEAD > /dev/null 2>&1; then
    git config --global --add safe.directory "$(cd .. && pwd)"
fi
echo "commit $(git rev-parse HEAD), status (excluding results/): '$(git status --porcelain -- . ':!results')'" | tee -a "$LOG"

# 0. Environment and tools
step 00_env_light   $PY env_light.py
step 01_versions    bash -c "$PY --version; $PY -m pip list 2>/dev/null | grep -i -E 'torch|triton|numpy'; \
                             ls -d /opt/rocm*; cat /opt/rocm/.info/version; uname -a; nproc; free -g; df -h ."
step 02_gpu_info    bash -c "amd-smi static || rocm-smi --showall || nvidia-smi"
step 03_tools       bash -c "which rocprofv3 rocprof-compute rocprofv3-avail; rocprofv3 --help; \
                             rocprof-compute --version; rocprof-compute profile --help"
step 04_counters    bash -c "rocprofv3-avail list || rocprofv3 -L"     # one of these lists the counters (to check)

# 1. Correctness and timing (the main results)
step 10_smoke       $PY amd_run.py --state "$STATE" --quick
step 11_full        $PY amd_run.py --state "$STATE" --save-outputs "$OUT/de_outputs.npz"

# Fastest DE configuration of the sweep, e.g. "1 512 4" (falls back to the NVIDIA best)
BEST=$(sed -nE 's/^fastest 5 \(median ms\): P([0-9]+) V([0-9]+) w([0-9]+) .*/\1 \2 \3/p' "$OUT/11_full.txt")
BEST=${BEST:-"8 128 4"}
BEST_NAME=$(echo "$BEST" | awk '{print "P"$1"V"$2"w"$3}')
echo "fastest DE config on this GPU: $BEST_NAME" | tee -a "$LOG"

if [ "${SKIP_PROFILERS:-0}" = "1" ]; then
    echo "SKIP_PROFILERS=1: profiler steps skipped" | tee -a "$LOG"
else
    # 2. rocprofv3: kernel trace (duration, VGPR/SGPR/LDS, grid and workgroup sizes per dispatch) and stats
    for cfg in "8 128 4" "$BEST"; do
        name=$(echo "$cfg" | awk '{print "P"$1"V"$2"w"$3}')
        step 20_rocprofv3_trace_de_$name rocprofv3 --kernel-trace --stats --output-format csv \
            --output-directory "$OUT/rocprofv3_de_$name" -- $PY amd_run.py --state "$STATE" --profile $cfg
    done
    step 21_rocprofv3_trace_add rocprofv3 --kernel-trace --stats --output-format csv \
        --output-directory "$OUT/rocprofv3_add" -- $PY amd_run.py --state "$STATE" --profile-add 28 4096
    # A few counters; the names are a guess until 04_counters is read, so this step may fail.
    step 22_rocprofv3_pmc_de rocprofv3 --pmc SQ_WAVES GRBM_GUI_ACTIVE --output-format csv \
        --output-directory "$OUT/rocprofv3_pmc_de" -- $PY amd_run.py --state "$STATE" --profile 8 128 4

    # 3. rocprof-compute: full counter set (several replays of the program), then the text report
    run_rc() {                              # run_rc NAME KERNEL_FILTER EXTRA_FLAGS -- PROGRAM...
        local name=$1 kfilter=$2 extra=$3; shift 4
        step 30_rocprof_compute_profile_$name rocprof-compute profile --name "${name}_$TAG" -k "$kfilter" $extra -- "$@"
        local wl=$(ls -d workloads/"${name}_$TAG"/*/ 2>/dev/null | head -1)
        if [ -n "$wl" ]; then
            step 31_rocprof_compute_analyze_$name rocprof-compute analyze -p "$wl"
        else
            echo "    no workload directory for $name, analyze skipped" | tee -a "$LOG"
        fi
    }
    run_rc de_P8V128w4 de_objective_kernel ""        -- $PY amd_run.py --state "$STATE" --profile 8 128 4
    [ "$BEST_NAME" != "P8V128w4" ] && \
    run_rc de_$BEST_NAME de_objective_kernel ""      -- $PY amd_run.py --state "$STATE" --profile $BEST
    run_rc add add_kernel "--no-roof"                -- $PY amd_run.py --state "$STATE" --profile-add 28 4096
fi

# 4. Repeat of the full timing run, after the profilers: drift and run-to-run variation on a shared host
step 40_full_repeat $PY amd_run.py --state "$STATE"

# 5. Pack everything for the copy back
[ -d workloads ] && cp -r workloads "$OUT/"
tar czf "results/amd_$TAG.tar.gz" -C results "amd_$TAG"
echo "done: results/amd_$TAG.tar.gz ($(du -h "results/amd_$TAG.tar.gz" | cut -f1))" | tee -a "$LOG"