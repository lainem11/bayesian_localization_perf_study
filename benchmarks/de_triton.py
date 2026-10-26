# benchmarks/de_triton.py
# Fused Triton kernel for the DE objective: correctness against the PyTorch reference and JAX,
# and a block-size sweep timed back to back against XLA. The kernel itself is in de_kernel.py.
import torch
import triton
from env_info import collect_env
from de_reference import fixed_state
from de_torch import objective_torch, strain_torch, compare, back_to_back_ms
from de_kernel import prepare, de_objective_kernel, de_objective_triton


if __name__ == "__main__":
    for key, value in collect_env().items():
        print(f"{key}: {value}")
    print(f"Triton version: {triton.__version__}")

    st = fixed_state()
    opt, to_t = st["opt"], st["to_t"]
    pop_j, lsp_j, params_j = st["pop_j"], st["lsp_j"], st["params_j"]
    pop_t, lsp_t, p_t, c = st["pop_t"], st["lsp_t"], st["p_t"], st["c"]
    k = prepare(c, p_t, lsp_t)

    # Correctness (criterion from 3b.1: expected change max rel diff < 1e-3, same argmin)
    fit_ref, obj_ref = objective_torch(pop_t, lsp_t, p_t, c)
    fit_tr, obj_tr = de_objective_triton(pop_t, k)
    print("\nDefault config (BLOCK_P 4, BLOCK_V 128, 4 warps)")
    compare("expected change, Triton vs torch reference", obj_ref, obj_tr)
    compare("full fitness, Triton vs torch reference", fit_ref, fit_tr)
    compare("full fitness, Triton vs JAX", to_t(opt.jitted_objective(pop_j, lsp_j, params_j)), fit_tr)

    jax_sync = lambda out=None: out.block_until_ready() if out is not None else None
    torch_sync = lambda out=None: torch.cuda.synchronize()
    xla = lambda: opt.jitted_objective(pop_j, lsp_j, params_j)

    configs = [(1, 512, 4), (1, 1024, 4), (2, 256, 4), (2, 512, 4), (4, 128, 4),
               (4, 256, 4), (8, 64, 4), (8, 128, 4), (16, 64, 4)]
    print(f"\nms per call, 200 calls back to back, 7 repeats")
    print(f"{'variant':>30} {'programs':>9} {'q25':>8} {'median':>8} {'q75':>8} {'max rel diff':>13}")
    med, q25, q75 = back_to_back_ms(xla, jax_sync)
    print(f"{'XLA (JAX), start':>30} {'':>9} {q25:>8.3f} {med:>8.3f} {q75:>8.3f}")
    for bp, bv, nw in configs:
        fn = lambda bp=bp, bv=bv, nw=nw: de_objective_triton(pop_t, k, bp, bv, nw)[0]
        _, o = de_objective_triton(pop_t, k, bp, bv, nw)
        rel = ((o.double() - obj_ref.double()).abs() / obj_ref.double().abs()).max().item()
        med, q25, q75 = back_to_back_ms(fn, torch_sync)
        print(f"{f'Triton P{bp} V{bv} w{nw}':>30} {512 // bp:>9} {q25:>8.3f} {med:>8.3f} {q75:>8.3f} {rel:>13.3e}")
    fn = lambda: de_objective_triton(pop_t, k, 4, 128, 4, fused_strain=False)[0]
    med, q25, q75 = back_to_back_ms(fn, torch_sync)
    print(f"{'Triton P4 V128, eager strain':>30} {128:>9} {q25:>8.3f} {med:>8.3f} {q75:>8.3f}")
    med, q25, q75 = back_to_back_ms(xla, jax_sync)
    print(f"{'XLA (JAX), end':>30} {'':>9} {q25:>8.3f} {med:>8.3f} {q75:>8.3f}")
    st["S"].end_session()