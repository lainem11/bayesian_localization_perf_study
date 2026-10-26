# benchmarks/export_de_state.py
# Saves the fixed DE state of de_reference.py (session after 3 trials, constrained population of 512)
# to one .npz file, with reference outputs, so the DE kernel can be checked on a machine without
# JAX or ABL (the AMD instance). Runs on the NVIDIA machine; needs the XLA autotuning file.
#
# Saved: the inputs of objective_torch in their original layout (prepare() runs again on the target),
# and four reference outputs, each (fitness, -expected change) per candidate:
#   jax     the session's jitted objective (fitness) and the penalty-free JAX objective (expected change)
#   torch   objective_torch on this GPU
#   triton  de_objective_triton, P8 V128, 4 warps, strain in the kernel, on this GPU
# Then reloads the file with de_torch.load_state() and checks the reload reproduces torch and triton bitwise.
#
# Usage: python export_de_state.py results/de_state_YYYY-MM-DD.npz
import json
import os
import sys
import numpy as np
import torch
import triton
from env_info import collect_env
from de_reference import fixed_state, change_only_objective
from de_torch import objective_torch, compare, load_state
from de_kernel import prepare, de_objective_triton

TRITON_CFG = dict(BLOCK_P=8, BLOCK_V=128, num_warps=4, fused_strain=True)


def _scalar(x):
    """Plain Python number, keeping int vs float (the kernel specializes on the argument type)."""
    x = x.item() if hasattr(x, "item") else x
    assert isinstance(x, (int, float)), type(x)
    return x


if __name__ == "__main__":
    out_path = sys.argv[1]
    assert "xla_gpu_load_autotune_results_from" in (os.getenv("XLA_FLAGS") or ""), \
        "set XLA_FLAGS to load the autotuning file first (the fixed state depends on it)"
    env = {k: str(v) for k, v in collect_env().items()}
    env["Triton version"] = triton.__version__
    for key, value in env.items():
        print(f"{key}: {value}")

    st = fixed_state()
    opt = st["opt"]
    pop_t, lsp_t, p_t, c = st["pop_t"], st["lsp_t"], st["p_t"], st["c"]
    pop_j, lsp_j, params_j = st["pop_j"], st["lsp_j"], st["params_j"]

    to_t = st["to_t"]
    fit_jax = to_t(opt.jitted_objective(pop_j, lsp_j, params_j))
    obj_jax = to_t(change_only_objective(st)(pop_j, lsp_j, params_j))
    fit_torch, obj_torch = objective_torch(pop_t, lsp_t, p_t, c)
    k = prepare(c, p_t, lsp_t)
    fit_tri, obj_tri = de_objective_triton(pop_t, k, **TRITON_CFG)

    np_ = lambda x: x.detach().cpu().numpy()
    scalars = {name: _scalar(c[name]) for name in ["lo", "hi", "penalty", "strain_limit", "efield_limit"]}
    meta = {"env": env, "param_keys": list(opt.param_keys), "scalars": scalars,
            "scalar_types": {n: type(v).__name__ for n, v in scalars.items()},
            "triton_config": TRITON_CFG, "population": list(pop_t.shape), "vertices": int(lsp_t.shape[0])}
    arrays = {"pop": np_(pop_t), "lsp": np_(lsp_t),
              "efield_norm": np_(c["efield_norm"]), "basis": np_(c["basis"]), "null_dir": np_(c["null_dir"]),
              **{f"p_{name}": np_(v) for name, v in p_t.items()},
              "out_fit_jax": np_(fit_jax), "out_obj_jax": np_(obj_jax),
              "out_fit_torch": np_(fit_torch), "out_obj_torch": np_(obj_torch),
              "out_fit_triton": np_(fit_tri), "out_obj_triton": np_(obj_tri)}
    for name, a in arrays.items():
        assert a.dtype == np.float32, (name, a.dtype)
    np.savez(out_path, meta_json=np.array(json.dumps(meta)), **arrays)

    print(f"\nwrote {out_path}: {os.path.getsize(out_path) / 1e6:.2f} MB")
    print(f"scalars: {scalars} (types {meta['scalar_types']})")
    for name, a in arrays.items():
        print(f"  {name:16s} {str(a.shape):14s} {a.dtype}")

    # Sanity: the saved outputs agree with each other as in 3b.1 / 3b.3
    print()
    compare("expected change, torch vs JAX", obj_jax, obj_torch)
    compare("full fitness, torch vs JAX", fit_jax, fit_torch)
    compare("expected change, Triton vs torch", obj_torch, obj_tri)

    # Round trip: reload and recompute on this GPU; must be bitwise identical to the live state
    pop2, lsp2, p2, c2, ref, _ = load_state(out_path)
    f2, o2 = objective_torch(pop2, lsp2, p2, c2)
    ft2, ot2 = de_objective_triton(pop2, prepare(c2, p2, lsp2), **TRITON_CFG)
    print("\nround trip (reloaded state, recomputed here), bitwise equal to live state:")
    print(f"  torch  fitness {torch.equal(f2, fit_torch)}, expected change {torch.equal(o2, obj_torch)}")
    print(f"  triton fitness {torch.equal(ft2, fit_tri)}, expected change {torch.equal(ot2, obj_tri)}")
    print(f"  saved jax outputs equal live: {torch.equal(ref['fit_jax'], fit_jax.cpu()) and torch.equal(ref['obj_jax'], obj_jax.cpu())}")
    st["S"].end_session()
