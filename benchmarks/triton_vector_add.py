# benchmarks/triton_vector_add.py
import torch
import triton
import triton.language as tl
from env_info import collect_env

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def add(x, y, block_size=1024):
    out = torch.empty_like(x)
    n = out.numel()
    grid = (triton.cdiv(n, block_size),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
    return out

def gbps(n, ms):
    return 3 * n * 4 / (ms * 1e-3) / 1e9        # float32: 4 bytes, 2 reads + 1 write

if __name__ == "__main__":
    for key, value in collect_env().items():
        print(f"{key}: {value}")
    print(f"Triton version: {triton.__version__}\n")

    print(f"{'n':>10} {'MB moved':>9} {'triton ms':>10} {'triton GB/s':>12} {'torch ms':>9} {'torch GB/s':>11}")
    for p in range(12, 27, 2):                   # n = 2^12, 2^14, ..., 2^26
        n = 2 ** p
        x = torch.rand(n, device="cuda")
        y = torch.rand(n, device="cuda")
        torch.testing.assert_close(add(x, y), x + y, rtol=0, atol=0)

        ms_triton = triton.testing.do_bench(lambda: add(x, y))
        ms_torch = triton.testing.do_bench(lambda: x + y)
        print(f"{n:>10} {3*n*4/1e6:>9.2f} {ms_triton:>10.4f} {gbps(n, ms_triton):>12.1f} "
              f"{ms_torch:>9.4f} {gbps(n, ms_torch):>11.1f}")