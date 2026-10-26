# AMD cloud session checklist (Phase 4)

Goal: run `amd_session.sh` once on a single AMD Instinct MI300X, copy the results back, destroy the instance.
The billed time should be well under 2 hours: setup ~20 min, session ~30–60 min, and these are estimates, not measurements.

Facts checked on 2026-10-26 (re-check anything that looks different on the day):
- AMD Developer Cloud runs on DigitalOcean GPU Droplets; the developer-program credit covers MI300X only and expires 30 days after deposit. A powered-off droplet is still billed: **destroy** it. [AMD](https://amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html)
- MI300X (data sheet): 304 CUs, 192 GB HBM3, **5.3 TB/s peak memory bandwidth**, 256 MB Infinity Cache, 4 MB L2 per XCD (8 XCDs), 163.4 TFLOPS peak FP32 vector. [AMD data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
- Latest ROCm: 10.1.0 (2026-10-05). AMD's PyTorch install page lists ROCm Docker images `rocm/pytorch:<rocm>_<ubuntu>_<python>_pytorch_release_<version>`, including PyTorch 2.12.0 (the NVIDIA side's version), e.g. `rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0`. [ROCm releases](https://rocm.docs.amd.com/en/latest/release/versions.html), [Install PyTorch for ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
- rocprofv3: `--kernel-trace`, `--stats`, `--output-format csv`, `--output-directory`, `--pmc`; the kernel trace has VGPR/SGPR/LDS counts and grid/workgroup sizes. rocprof-compute: `profile --name N -k KERNEL [--no-roof] -- cmd`, then `analyze -p workloads/N/<gpu>`. [rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/advanced-rocprofv3-options.html), [rocprof-compute profile mode](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html)

Not verified yet (the session records them): the droplet's image, host ROCm and driver; whether the image above exists and runs on that driver; whether `rocprof-compute` is installed in the container and needs extra Python packages; whether counter collection needs more container privileges; the counter names for `--pmc`; what Triton's `num_warps` and `n_regs` mean on this backend.

## Before (free)

1. [ ] Credit approved (email). Sign in at devcloud.amd.com; Billing → **Credits** tab shows the amount.
2. [ ] SSH key: `ssh-keygen -t ed25519 -f ~/.ssh/amd_devcloud` (add `~/.ssh/amd_devcloud.pub` when creating the droplet).
3. [ ] Dry run on the 4060 Ti, clean tree, from `benchmarks/`:
   `SKIP_PROFILERS=1 bash amd_session.sh test_4060ti` → every step `exit 0` in `results/amd_test_4060ti/session_log.txt` (03/04 fail on NVIDIA; that's expected).
4. [ ] Everything committed and pushed (the instance clones the public fork; no credentials needed).

## On the instance (billed from here)

5. [ ] Create a droplet: **1× MI300X**, the default ROCm image. Note the image name and the time.
6. [ ] `ssh -i ~/.ssh/amd_devcloud root@<IP>`, then record the host:
   ```bash
   cat /etc/os-release | head -2; cat /opt/rocm/.info/version; amd-smi version; docker images
   ```
7. [ ] `git clone -b perf-study https://github.com/lainem11/bayesian_localization_perf_study.git ~/repo`
8. [ ] Container. First choice, the PyTorch version of the NVIDIA runs:
   ```bash
   IMG=rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0
   docker pull $IMG
   ```
   If the pull fails, or the host ROCm is much older than 7.14 (container user space and host driver must be compatible), use a preloaded PyTorch image from `docker images`. Write down the choice and why.
9. [ ] Start it (standard ROCm device flags; profilers may need `--privileged` instead, see step 12):
   ```bash
   docker run -it --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16G \
     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v ~/repo:/repo -w /repo/benchmarks $IMG bash
   ```
10. [ ] 30-second check: `python -c "import torch, triton; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0), triton.__version__)"`
11. [ ] Run: `bash amd_session.sh mi300x_$(date +%F)`. Watch from a second ssh window:
    `docker exec -it $(docker ps -q) tail -f /repo/benchmarks/results/amd_mi300x_*/session_log.txt`
12. [ ] If a profiler step fails, read its file in `results/amd_mi300x_*/` first:
    - `rocprof-compute` import errors → install its requirements (find them with `find /opt/rocm -name 'requirements*.txt' | grep -i compute`) and rerun only that step's command from the script.
    - permission or counter errors → restart the container with `--privileged` and rerun only the profiler commands.
    - `--pmc` names unknown → pick 2–3 names from `04_counters.txt` (waves, busy cycles, HBM read/write bytes) and rerun step 22.
    Don't debug for long: the timing and correctness results (step 11) are the main deliverable.
13. [ ] Copy back, on your machine (the repo is mounted, so the files are on the host):
    `scp -i ~/.ssh/amd_devcloud root@<IP>:~/repo/benchmarks/results/amd_mi300x_*.tar.gz benchmarks/results/`
14. [ ] **Destroy** the droplet (not power off). Check Billing a few minutes later. Note the end time.

## After (free)

15. [ ] Unpack into `benchmarks/results/`, commit, push. Then: log entry 4.2 (environment, correctness, timings vs the 5.3 TB/s peak) and 4.3 (profilers).