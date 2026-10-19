import time, torch
x = torch.randn(8192, 8192, device="cuda")
for _ in range(3): x @ x            # warm-up
torch.cuda.synchronize()
t0 = time.perf_counter(); y = x @ x; t1 = time.perf_counter()
torch.cuda.synchronize(); t2 = time.perf_counter()
print(f"call returned: {1e3*(t1-t0):.3f} ms   work finished: {1e3*(t2-t0):.3f} ms")