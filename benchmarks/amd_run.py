# benchmarks/amd_run.py
# One script for the AMD instance (and tested on the RTX 4060 Ti): the three Phase 3a exercise kernels
# and the fused DE kernel from a saved state. Needs only torch, triton and numpy (no JAX, no ABL).
# Every check prints PASS/FAIL and the run continues, so one failure doesn't waste the cloud session.
#
# Usage (from benchmarks/):
#   python amd_run.py --state results/de_state_2026-10-26.npz > results/amd_run_<gpu>_<date>.txt
#   python amd_run.py --state ... --quick                      # small sizes, short sweep (smoke test)
#   python amd_run.py --state ... --only de                    # sections: vector_add,elementwise,lse,de
#   python amd_run.py --state ... --profile 8 128 4            # 3 DE kernel calls only, for a profiler
#   python amd_run.py --state ... --only de --save-outputs results/de_outputs_<gpu>.npz
import argparse
import itertools
import traceback
import numpy as np
import torch
import triton
import triton.testing
from env_light import collect_env_light
import triton_vector_add as va
import triton_fused_elementwise as fe
import triton_logsumexp as lse
from de_torch import objective_torch, strain_torch, load_state, back_to_back_ms
from de_kernel import prepare, de_objective_triton, kernel_info

DEV = "cuda"            # PyTorch's ROCm build exposes HIP GPUs through the torch.cuda API and device "cuda"
QUANT = [0.25, 0.5, 0.75]
torch_sync = lambda out=None: torch.cuda.synchronize()


def check(name, fn):
    """Runs a correctness check; prints PASS, or FAIL with the error, and never raises."""
    try:
        fn()
        print(f"  check {name}: PASS")
        return True
    except Exception as e:
        print(f"  check {name}: FAIL ({type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''})")
        return False


def bench(fn):
    q25, q50, q75 = triton.testing.do_bench(fn, quantiles=QUANT)
    return q25, q50, q75


def section(title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}", flush=True)


# ------------------------------------------------------------------------------------------------
def run_vector_add(quick):
    section("1. Vector add (triton_vector_add.py kernel); GB/s = 3 x n x 4 bytes / median time")
    exps = [20, 24] if quick else [20, 22, 24, 26, 28]
    print(f"{'n':>10} {'MB moved':>9} {'triton1024 ms':>14} {'GB/s':>7} {'triton4096 ms':>14} {'GB/s':>7} {'torch ms':>9} {'GB/s':>7}")
    for p in exps:
        n = 2 ** p
        x = torch.rand(n, device=DEV)
        y = torch.rand(n, device=DEV)
        check(f"n=2^{p} bitwise vs torch", lambda: torch.testing.assert_close(va.add(x, y), x + y, rtol=0, atol=0))
        t1 = bench(lambda: va.add(x, y, 1024))[1]
        t4 = bench(lambda: va.add(x, y, 4096))[1]
        tt = bench(lambda: x + y)[1]
        print(f"{n:>10} {3*n*4/1e6:>9.2f} {t1:>14.4f} {va.gbps(n, t1):>7.1f} {t4:>14.4f} {va.gbps(n, t4):>7.1f} "
              f"{tt:>9.4f} {va.gbps(n, tt):>7.1f}", flush=True)
        del x, y

    n = 2 ** exps[-1]
    x = torch.rand(n, device=DEV)
    y = torch.rand(n, device=DEV)
    print(f"\nBlock-size sweep, n = 2^{exps[-1]} ({3*n*4/1e6:.1f} MB moved)")
    print(f"{'BLOCK':>6} {'programs':>9} {'num_warps':>9} {'q25 ms':>8} {'median ms':>10} {'q75 ms':>8} {'GB/s (median)':>14}")
    for bs in [256, 1024, 4096, 16384] if quick else [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        out = torch.empty_like(x)
        grid = (triton.cdiv(n, bs),)
        ck = va.add_kernel[grid](x, y, out, n, BLOCK_SIZE=bs)
        check(f"BLOCK {bs} bitwise", lambda: torch.testing.assert_close(out, x + y, rtol=0, atol=0))
        q25, q50, q75 = bench(lambda: va.add(x, y, bs))
        print(f"{bs:>6} {grid[0]:>9} {ck.metadata.num_warps:>9} {q25:>8.4f} {q50:>10.4f} {q75:>8.4f} {va.gbps(n, q50):>14.1f}", flush=True)
    print(f"warp size reported by the compiled kernel: {getattr(ck.metadata, 'warp_size', 'not reported')}")


def run_elementwise(quick):
    section("2. Fused elementwise chain (triton_fused_elementwise.py); eff. GB/s counts 16 bytes per element")
    n = 2 ** 22 if quick else 2 ** 27
    e = torch.rand(n, device=DEV) * 150
    s = (torch.rand(n, device=DEV) + 0.5) / 8
    t = (torch.rand(n, device=DEV) + 0.5) * 100
    compiled = torch.compile(fe.log_resp_torch)
    eager_out = fe.log_resp_torch(e, s, t)
    for name, fn in {"triton": lambda: fe.log_resp_triton(e, s, t), "compiled": lambda: compiled(e, s, t)}.items():
        out = fn()
        check(f"{name} vs eager (assert_close defaults)", lambda: torch.testing.assert_close(out, eager_out))
        print(f"  {name} vs eager: bitwise-equal fraction {(out == eager_out).float().mean().item():.6f}, "
              f"max abs diff {(out - eager_out).abs().max().item():.3e}")
    print(f"\nn = 2^{int(np.log2(n))}, minimum traffic {16 * n / 1e6:.1f} MB")
    print(f"{'variant':>9} {'q25 ms':>8} {'median ms':>10} {'q75 ms':>8} {'eff. GB/s':>10}")
    for name, fn in {"triton": lambda: fe.log_resp_triton(e, s, t), "eager": lambda: fe.log_resp_torch(e, s, t),
                     "compiled": lambda: compiled(e, s, t)}.items():
        q25, q50, q75 = bench(fn)
        print(f"{name:>9} {q25:>8.4f} {q50:>10.4f} {q75:>8.4f} {fe.gbps(n, q50):>10.1f}", flush=True)


def run_lse(quick):
    section("3. Row-wise online logsumexp (triton_logsumexp.py); one program per row")
    shapes = [(512, 7444), (512, 8 * 7444)] if quick else [(512, 7444), (512, 8 * 7444), (512, 64 * 7444)]
    for rows, cols in shapes:
        x = torch.randn(rows, cols, device=DEV) * 5 - 20
        ref = torch.logsumexp(x, dim=1)
        check(f"[{rows} x {cols}] triton vs torch", lambda: torch.testing.assert_close(lse.lse_triton(x), ref))
        if cols == 7444:
            xs = x + 100
            print(f"  stability (x + 100): naive finite = {torch.isfinite(lse.lse_naive(xs)).all().item()}, "
                  f"triton finite = {torch.isfinite(lse.lse_triton(xs)).all().item()}, max abs diff vs torch = "
                  f"{(lse.lse_triton(xs) - torch.logsumexp(xs, 1)).abs().max().item():.3e}")
        compiled = torch.compile(lambda x: torch.logsumexp(x, dim=1))
        check(f"[{rows} x {cols}] compiled vs torch", lambda: torch.testing.assert_close(compiled(x), ref))
        variants = {"torch eager": lambda: torch.logsumexp(x, dim=1), "compiled": lambda: compiled(x)}
        for b in [512, 1024, 4096] if quick else [256, 512, 1024, 2048, 4096]:
            variants[f"triton {b}"] = (lambda b=b: lse.lse_triton(x, b))
        print(f"\nrows {rows}, cols {cols}, minimum traffic {(rows * cols + rows) * 4 / 1e6:.2f} MB")
        print(f"{'variant':>12} {'q25 ms':>8} {'median ms':>10} {'q75 ms':>8} {'eff. GB/s':>10}")
        for name, fn in variants.items():
            q25, q50, q75 = bench(fn)
            print(f"{name:>12} {q25:>8.4f} {q50:>10.4f} {q75:>8.4f} {lse.gbps(rows, cols, q50):>10.1f}", flush=True)
        del x


# ------------------------------------------------------------------------------------------------
def rel(ref, new):
    ref, new = ref.double().cpu(), new.double().cpu()
    return ((new - ref).abs() / ref.abs()).max().item()


def de_checks(name, fit, obj, ref):
    """Criterion fixed in 3b.1: expected change max rel diff < 1e-3 vs the reference, same argmin of the fitness."""
    fit, obj = fit.cpu(), obj.cpu()
    r_t, r_j = rel(ref["obj_torch"], obj), rel(ref["obj_jax"], obj)
    am = {k: ref[f"fit_{k}"].argmin().item() for k in ["torch", "jax", "triton"]}
    ok = r_t < 1e-3 and r_j < 1e-3 and fit.argmin().item() == am["torch"] == am["jax"]
    print(f"  {name}: expected change max rel diff vs saved torch {r_t:.3e}, vs saved JAX {r_j:.3e}; "
          f"fitness max rel diff vs saved JAX {rel(ref['fit_jax'], fit):.3e}; argmin {fit.argmin().item()} "
          f"(saved torch {am['torch']}, JAX {am['jax']}) -> {'PASS' if ok else 'FAIL'}")
    print(f"    bitwise equal to the saved Triton output (expected only on the export GPU): "
          f"fitness {torch.equal(fit, ref['fit_triton'])}, expected change {torch.equal(obj, ref['obj_triton'])}")
    return ok


def run_de(state_path, quick, save_outputs):
    section("4. Fused DE objective kernel (de_kernel.py) on the saved fixed state")
    pop, lsp, p, c, ref, meta = load_state(state_path, DEV)
    print(f"state {state_path}: population {meta['population']}, vertices {meta['vertices']}, "
          f"exported at commit {meta['env'].get('GIT commit')} on {meta['env'].get('GPU model and driver version')}")
    print(f"saved Triton reference config: {meta['triton_config']}")
    k = prepare(c, p, lsp)

    print("\nCorrectness")
    fit_t, obj_t = objective_torch(pop, lsp, p, c)
    print(f"  PyTorch reference on this GPU vs saved torch: expected change max rel diff {rel(ref['obj_torch'], obj_t):.3e}, "
          f"fitness {rel(ref['fit_torch'], fit_t):.3e}, argmin {fit_t.argmin().item()}")
    fit_d, obj_d = de_objective_triton(pop, k, 8, 128, 4, fused_strain=True)
    de_checks("Triton P8 V128 w4", fit_d, obj_d, ref)
    if save_outputs:
        np.savez(save_outputs, fit_triton=fit_d.cpu().numpy(), obj_triton=obj_d.cpu().numpy(),
                 fit_torch=fit_t.cpu().numpy(), obj_torch=obj_t.cpu().numpy())
        print(f"  saved outputs to {save_outputs}")

    if quick:
        configs = [(1, 512, 4), (4, 128, 4), (8, 128, 4), (8, 128, 8)]
    else:
        configs = list(itertools.product([1, 2, 4, 8, 16], [64, 128, 256, 512], [1, 2, 4, 8]))
    print(f"\nSweep: ms per call, 200 calls back to back, 7 repeats; max rel diff of the expected change vs saved torch")
    hdr = f"{'config':>16} {'programs':>9} {'regs':>5} {'spills':>6} {'q25':>8} {'median':>8} {'q75':>8} {'max rel diff':>13} {'ok':>4}"
    print(hdr)
    rows = []
    for bp, bv, nw in configs:
        name = f"P{bp} V{bv} w{nw}"
        try:
            info = kernel_info(pop, k, bp, bv, nw)
            f, o = de_objective_triton(pop, k, bp, bv, nw)
            r = rel(ref["obj_torch"], o)
            ok = r < 1e-3 and f.argmin().item() == ref["fit_torch"].argmin().item()
            med, q25, q75 = back_to_back_ms(lambda: de_objective_triton(pop, k, bp, bv, nw)[0], torch_sync)
            rows.append((med, name))
            print(f"{name:>16} {triton.cdiv(pop.shape[0], bp):>9} {str(info['n_regs']):>5} {str(info['n_spills']):>6} "
                  f"{q25:>8.3f} {med:>8.3f} {q75:>8.3f} {r:>13.3e} {'yes' if ok else 'NO':>4}", flush=True)
        except Exception as e:
            print(f"{name:>16} FAILED: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}", flush=True)
    print(f"\nkernel_info for P8 V128 w4: {kernel_info(pop, k, 8, 128, 4)}")
    print("fastest 5 (median ms):", ", ".join(f"{n} {m:.3f}" for m, n in sorted(rows)[:5]))

    print("\nBaselines on this GPU (same timing method)")
    print(f"{'variant':>30} {'q25':>8} {'median':>8} {'q75':>8}")
    compiled = torch.compile(objective_torch)
    check("torch.compile vs eager reference (assert_close defaults)",
          lambda: torch.testing.assert_close(compiled(pop, lsp, p, c)[1], obj_t))
    variants = {
        "Triton P8 V128 w4, start": lambda: de_objective_triton(pop, k, 8, 128, 4)[0],
        "Triton P4 V128 w4, eager strain": lambda: de_objective_triton(pop, k, 4, 128, 4, fused_strain=False)[0],
        "torch eager": lambda: objective_torch(pop, lsp, p, c)[0],
        "torch.compile": lambda: compiled(pop, lsp, p, c)[0],
        "Triton P8 V128 w4, end": lambda: de_objective_triton(pop, k, 8, 128, 4)[0],
    }
    for name, fn in variants.items():
        med, q25, q75 = back_to_back_ms(fn, torch_sync)
        print(f"{name:>30} {q25:>8.3f} {med:>8.3f} {q75:>8.3f}", flush=True)


def run_profile(state_path, bp, bv, nw):
    """Three calls of one configuration, for a profiler (call 1 compiles; profile call 3)."""
    pop, lsp, p, c, ref, meta = load_state(state_path, DEV)
    k = prepare(c, p, lsp)
    for _ in range(3):
        fit, obj = de_objective_triton(pop, k, bp, bv, nw, fused_strain=True)
    torch.cuda.synchronize()
    print(f"config: BLOCK_P {bp}, BLOCK_V {bv}, num_warps {nw}, strain in kernel; kernel_info {kernel_info(pop, k, bp, bv, nw)}")
    de_checks(f"P{bp} V{bv} w{nw}", fit, obj, ref)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="npz written by export_de_state.py")
    ap.add_argument("--only", default="vector_add,elementwise,lse,de")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--profile", nargs=3, type=int, metavar=("BLOCK_P", "BLOCK_V", "NUM_WARPS"))
    ap.add_argument("--save-outputs", default=None)
    args = ap.parse_args()

    for key, value in collect_env_light().items():
        print(f"{key}: {value}")
    print(f"command args: {vars(args)}")
    torch.manual_seed(0)

    if args.profile:
        run_profile(args.state, *args.profile)
    else:
        sections = {"vector_add": lambda: run_vector_add(args.quick), "elementwise": lambda: run_elementwise(args.quick),
                    "lse": lambda: run_lse(args.quick), "de": lambda: run_de(args.state, args.quick, args.save_outputs)}
        for name in args.only.split(","):
            try:
                sections[name]()
            except Exception:
                print(f"\nSECTION {name} FAILED:\n{traceback.format_exc()}", flush=True)
            torch.cuda.empty_cache()
