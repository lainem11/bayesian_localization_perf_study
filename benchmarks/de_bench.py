# benchmarks/de_bench.py
# Times the DE objective on the fixed state of de_reference.py: XLA (the session's jitted objective),
# eager PyTorch and torch.compile of the reference. Same method for all: 200 calls queued
# back to back, one sync at the end (as inside the DE's scan), repeated 7 times.
import torch
from env_info import collect_env
from de_reference import fixed_state
from de_torch import objective_torch, back_to_back_ms   # back_to_back_ms moved to de_torch.py


if __name__ == "__main__":
    for key, value in collect_env().items():
        print(f"{key}: {value}")

    st = fixed_state()
    opt = st["opt"]
    pop_j, lsp_j, params_j = st["pop_j"], st["lsp_j"], st["params_j"]
    pop_t, lsp_t, p_t, c = st["pop_t"], st["lsp_t"], st["p_t"], st["c"]
    compiled = torch.compile(objective_torch)

    jax_sync = lambda out=None: out.block_until_ready() if out is not None else None
    torch_sync = lambda out=None: torch.cuda.synchronize()
    variants = {
        "XLA (JAX)":     (lambda: opt.jitted_objective(pop_j, lsp_j, params_j), jax_sync),
        "torch eager":   (lambda: objective_torch(pop_t, lsp_t, p_t, c)[0], torch_sync),
        "torch.compile": (lambda: compiled(pop_t, lsp_t, p_t, c)[0], torch_sync),
    }
    print(f"\npopulation {pop_t.shape[0]}, vertices {lsp_t.shape[0]}; ms per call, 200 calls back to back, 7 repeats")
    print(f"{'variant':>14} {'q25':>8} {'median':>8} {'q75':>8}")
    for name, (fn, sync) in variants.items():
        med, q25, q75 = back_to_back_ms(fn, sync)
        print(f"{name:>14} {q25:>8.3f} {med:>8.3f} {q75:>8.3f}")
    st["S"].end_session()