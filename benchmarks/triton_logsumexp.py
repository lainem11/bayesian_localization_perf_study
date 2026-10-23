# benchmarks/triton_logsumexp.py
import torch
import triton
import triton.language as tl
from env_info import collect_env

@triton.jit
def lse_kernel(x_ptr, out_ptr, n_cols, stride_row, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)                    # one program per row
    row_ptr = x_ptr + row * stride_row             # stride_row = elements between rows
    m = -float("inf")                              # running max
    s = 0.0                                        # running sum of exp(x - m)
    for start in range(0, n_cols, BLOCK):          # loop over chunks of the row
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(row_ptr + offs, mask=offs < n_cols, other=-float("inf"))  # exp(-inf) = 0
        m_new = tl.maximum(m, tl.max(x, axis=0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    tl.store(out_ptr + row, m + tl.log(s))

def lse_triton(x, block=1024):
    out = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
    lse_kernel[(x.shape[0],)](x, out, x.shape[1], x.stride(0), BLOCK=block)
    return out

def lse_naive(x):
    return torch.log(torch.exp(x).sum(dim=1))      # no max subtraction

def gbps(rows, cols, ms):
    return (rows * cols + rows) * 4 / (ms * 1e-3) / 1e9

if __name__ == "__main__":
    for key, value in collect_env().items():
        print(f"{key}: {value}")
    print(f"Triton version: {triton.__version__}\n")

    rows, cols = 512, 7444                         # DE shape: candidates x vertices
    for rows, cols in [(512, 7444), (512, 8 * 7444)]:      # 15 MB (fits in L2), 122 MB (doesn't)
        x = torch.randn(rows, cols, device="cuda") * 5 - 20
        ref = torch.logsumexp(x, dim=1)
        torch.testing.assert_close(lse_triton(x), ref)
        if cols == 7444:
            print(f"Stability (x + 100): naive finite = {torch.isfinite(lse_naive(x + 100)).all().item()}, "
                    f"triton finite = {torch.isfinite(lse_triton(x + 100)).all().item()}, "
                    f"max abs diff vs torch = {(lse_triton(x + 100) - torch.logsumexp(x + 100, 1)).abs().max().item():.3e}")

        compiled = torch.compile(lambda x: torch.logsumexp(x, dim=1))
        torch.testing.assert_close(compiled(x), ref)

        variants = {"torch eager": lambda: torch.logsumexp(x, dim=1), "compiled": lambda: compiled(x)}
        for b in [256, 512, 1024, 2048, 4096]:
            variants[f"triton {b}"] = (lambda b=b: lse_triton(x, b))
        print(f"\nrows {rows}, cols {cols}, minimum traffic {(rows * cols + rows) * 4 / 1e6:.2f} MB")
        print(f"{'variant':>12} {'q25 ms':>8} {'median ms':>10} {'q75 ms':>8} {'eff. GB/s':>10}")
        for name, fn in variants.items():
            q25, q50, q75 = triton.testing.do_bench(fn, quantiles=[0.25, 0.5, 0.75])
            print(f"{name:>12} {q25:>8.4f} {q50:>10.4f} {q75:>8.4f} {gbps(rows, cols, q50):>10.1f}")