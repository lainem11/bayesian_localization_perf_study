# benchmarks/triton_fused_elementwise.py
import torch
import triton
import triton.language as tl

@triton.jit
def log_resp_kernel(e_ptr, s_ptr, t_ptr, out_ptr, n, lo, hi, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    e = tl.load(e_ptr + offs, mask=mask, other=1.0)        # (1)
    s = tl.load(s_ptr + offs, mask=mask, other=0.0)
    t = tl.load(t_ptr + offs, mask=mask, other=0.0)
    inner = tl.clamp(-s * (e - t), -16.0, 16.0)            # (2)
    p = e / (e + tl.exp(inner))
    p = tl.clamp(p, lo, hi)
    tl.store(out_ptr + offs, tl.log(p), mask=mask)         # (3)

def log_resp_triton(e, s, t, lo=1e-6, hi=1 - 1e-6, block_size=4096):
    out = torch.empty_like(e)
    n = out.numel()
    grid = (triton.cdiv(n, block_size),)
    log_resp_kernel[grid](e, s, t, out, n, lo, hi, BLOCK_SIZE=block_size)
    return out

def log_resp_torch(e, s, t, lo=1e-6, hi=1 - 1e-6):
    inner = torch.clamp(-s * (e - t), -16, 16)
    p = e / (e + torch.exp(inner))
    return torch.log(torch.clamp(p, lo, hi))

def gbps(n, ms):
    return 4 * n * 4 / (ms * 1e-3) / 1e9        # float32: 3 reads + 1 write

if __name__ == "__main__":
    from env_info import collect_env      # imported here so the kernels can be imported without JAX
    for key, value in collect_env().items():
        print(f"{key}: {value}")
    print(f"Triton version: {triton.__version__}\n")

    n = 2 ** 26
    e = torch.rand(n, device="cuda") * 150
    s = (torch.rand(n, device="cuda") + 0.5) / 8
    t = (torch.rand(n, device="cuda") + 0.5) * 100

    compiled_torch = torch.compile(log_resp_torch)

    # Correctness (the first compiled call also does the compilation, outside timing)
    eager_out = log_resp_torch(e, s, t)
    outputs = {"triton": log_resp_triton(e, s, t), "compiled": compiled_torch(e, s, t)}
    for name, out in outputs.items():
        torch.testing.assert_close(out, eager_out)
        frac = (out == eager_out).float().mean().item()
        maxdiff = (out - eager_out).abs().max().item()
        print(f"{name} vs eager: bitwise-equal fraction {frac:.6f}, max abs diff {maxdiff:.3e}")

    # Timing
    variants = {
        "triton":   lambda: log_resp_triton(e, s, t),
        "eager":    lambda: log_resp_torch(e, s, t),
        "compiled": lambda: compiled_torch(e, s, t),
    }
    print(f"\nn = {n}, minimum traffic {16 * n / 1e6:.1f} MB (3 reads + 1 write, float32)")
    print(f"{'variant':>9} {'q25 ms':>8} {'median ms':>10} {'q75 ms':>8} {'eff. GB/s':>10}")
    for name, fn in variants.items():
        q25, q50, q75 = triton.testing.do_bench(fn, quantiles=[0.25, 0.5, 0.75])
        print(f"{name:>9} {q25:>8.4f} {q50:>10.4f} {q75:>8.4f} {gbps(n, q50):>10.1f}")