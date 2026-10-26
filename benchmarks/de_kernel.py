# benchmarks/de_kernel.py
# Fused Triton kernel for the DE objective (single_objective_fun, src/ABL/optimizer.py:130-158).
# Imports only torch and triton (no JAX, no ABL), so it also runs on ROCm. Moved unchanged from
# de_triton.py; kernel_info() is new.
# No [P x V] intermediate is written to memory: each program handles BLOCK_P candidates and loops
# over the vertices twice (pass 1: online logsumexp for y=1 and y=0 and the max E-field;
# pass 2: expected change, recomputing the numerators).
import torch
import triton
import triton.language as tl
from de_torch import strain_torch


def prepare(c, p_t, lsp_t):
    """Per-session constants, reordered so every row is contiguous along vertices (coalesced loads)."""
    V = lsp_t.shape[0]
    nd = c["null_dir"]                                                   # [V, 2]
    a0 = torch.atan2(nd[:, 1], nd[:, 0])                                 # reference angle per vertex
    return {
        "W": c["efield_norm"].permute(0, 2, 1).reshape(15, V).contiguous(),   # row c*3+d: coil c, component d
        "B": c["basis"].permute(1, 2, 0).reshape(6, V).contiguous(),          # row d*2+t: component d, subspace axis t
        "CS": torch.stack([torch.cos(a0), torch.sin(a0)]).contiguous(),       # [2, V]
        "PRM": torch.stack([p_t[k] for k in ["rf_slope", "rf_translate", "rf_fourier1",
                                             "rf_fourier2", "rf_fourier3", "rf_fourier4"]]).contiguous(),  # [6, V]
        "LSP": lsp_t.contiguous(), "V": V,
        "lo": c["lo"], "hi": c["hi"], "efield_limit": c["efield_limit"],
        "penalty": c["penalty"], "strain_limit": c["strain_limit"],
    }


@triton.jit
def _row(ptr, r, V, v_offs, v_mask):
    """Row r of a [R, V] array for this vertex chunk, shaped [1, BLOCK_V] for broadcasting."""
    return tl.load(ptr + r * V + v_offs, mask=v_mask, other=0.0)[None, :]


@triton.jit
def _tile(pc0, pc1, pc2, pc3, pc4, W, B, CS, PRM, LSP, V, v_offs, v_mask, lo, hi):
    """Numerators for y=1 and y=0, E-field magnitude and log-prior on a [BLOCK_P, BLOCK_V] tile."""
    # E-field of every candidate at every vertex of the chunk: sum over the 5 coils (pc*: [BLOCK_P, 1])
    E0 = (pc0 * _row(W, 0, V, v_offs, v_mask) + pc1 * _row(W, 3, V, v_offs, v_mask) + pc2 * _row(W, 6, V, v_offs, v_mask)
          + pc3 * _row(W, 9, V, v_offs, v_mask) + pc4 * _row(W, 12, V, v_offs, v_mask))
    E1 = (pc0 * _row(W, 1, V, v_offs, v_mask) + pc1 * _row(W, 4, V, v_offs, v_mask) + pc2 * _row(W, 7, V, v_offs, v_mask)
          + pc3 * _row(W, 10, V, v_offs, v_mask) + pc4 * _row(W, 13, V, v_offs, v_mask))
    E2 = (pc0 * _row(W, 2, V, v_offs, v_mask) + pc1 * _row(W, 5, V, v_offs, v_mask) + pc2 * _row(W, 8, V, v_offs, v_mask)
          + pc3 * _row(W, 11, V, v_offs, v_mask) + pc4 * _row(W, 14, V, v_offs, v_mask))
    e_mag = tl.sqrt(E0 * E0 + E1 * E1 + E2 * E2)

    # Projection to the 2D subspace and its direction (cos phi, sin phi)
    t0 = E0 * _row(B, 0, V, v_offs, v_mask) + E1 * _row(B, 2, V, v_offs, v_mask) + E2 * _row(B, 4, V, v_offs, v_mask)
    t1 = E0 * _row(B, 1, V, v_offs, v_mask) + E1 * _row(B, 3, V, v_offs, v_mask) + E2 * _row(B, 5, V, v_offs, v_mask)
    inv = 1.0 / tl.sqrt(t0 * t0 + t1 * t1)
    cphi = t0 * inv
    sphi = t1 * inv

    # theta = phi - phi0 without atan2: angle-difference and double-angle identities
    ca = _row(CS, 0, V, v_offs, v_mask)
    sa = _row(CS, 1, V, v_offs, v_mask)
    cos1 = cphi * ca + sphi * sa
    sin1 = sphi * ca - cphi * sa
    cos2 = cos1 * cos1 - sin1 * sin1
    sin2 = 2.0 * sin1 * cos1

    # Response model (directional sigmoid)
    trans = (_row(PRM, 1, V, v_offs, v_mask) + _row(PRM, 2, V, v_offs, v_mask) * cos1 + _row(PRM, 3, V, v_offs, v_mask) * sin1
             + _row(PRM, 4, V, v_offs, v_mask) * cos2 + _row(PRM, 5, V, v_offs, v_mask) * sin2)
    inner = tl.clamp(-_row(PRM, 0, V, v_offs, v_mask) * (e_mag - trans), -16.0, 16.0)
    r = tl.clamp(e_mag / (e_mag + tl.exp(inner)), lo, hi)

    lsp = _row(LSP, 0, V, v_offs, v_mask)
    num1 = lsp + tl.log(r)            # log prior + log P(y=1)
    num0 = lsp + tl.log(1.0 - r)      # log prior + log P(y=0); 1 - r is exact for r in [0.5, 1]
    return num1, num0, e_mag, lsp


@triton.jit
def de_objective_kernel(pop_ptr, W, B, CS, PRM, LSP, fit_ptr, obj_ptr, P, V, lo, hi, efield_limit, penalty,
                        strain_limit, BLOCK_P: tl.constexpr, BLOCK_V: tl.constexpr, FUSED_STRAIN: tl.constexpr):
    pid = tl.program_id(axis=0)
    p_offs = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < P
    # This program's candidates: 5 coil weights each, [BLOCK_P]
    w0 = tl.load(pop_ptr + p_offs * 5 + 0, mask=p_mask, other=0.0)
    w1 = tl.load(pop_ptr + p_offs * 5 + 1, mask=p_mask, other=0.0)
    w2 = tl.load(pop_ptr + p_offs * 5 + 2, mask=p_mask, other=0.0)
    w3 = tl.load(pop_ptr + p_offs * 5 + 3, mask=p_mask, other=0.0)
    w4 = tl.load(pop_ptr + p_offs * 5 + 4, mask=p_mask, other=0.0)
    pc0 = w0[:, None]                 # as [BLOCK_P, 1] columns for broadcasting
    pc1 = w1[:, None]
    pc2 = w2[:, None]
    pc3 = w3[:, None]
    pc4 = w4[:, None]

    # Pass 1: online logsumexp over vertices for both outcomes, and the max E-field
    m1 = tl.full([BLOCK_P], float("-inf"), tl.float32)
    s1 = tl.zeros([BLOCK_P], dtype=tl.float32)
    m0 = tl.full([BLOCK_P], float("-inf"), tl.float32)
    s0 = tl.zeros([BLOCK_P], dtype=tl.float32)
    emax = tl.zeros([BLOCK_P], dtype=tl.float32)
    for start in range(0, V, BLOCK_V):
        v_offs = start + tl.arange(0, BLOCK_V)
        v_mask = v_offs < V
        num1, num0, e_mag, lsp = _tile(pc0, pc1, pc2, pc3, pc4, W, B, CS, PRM, LSP, V, v_offs, v_mask, lo, hi)
        vm = v_mask[None, :]
        num1 = tl.where(vm, num1, float("-inf"))
        num0 = tl.where(vm, num0, float("-inf"))
        emax = tl.maximum(emax, tl.max(tl.where(vm, e_mag, 0.0), axis=1))
        m1_new = tl.maximum(m1, tl.max(num1, axis=1))
        s1 = s1 * tl.exp(m1 - m1_new) + tl.sum(tl.exp(num1 - m1_new[:, None]), axis=1)
        m1 = m1_new
        m0_new = tl.maximum(m0, tl.max(num0, axis=1))
        s0 = s0 * tl.exp(m0 - m0_new) + tl.sum(tl.exp(num0 - m0_new[:, None]), axis=1)
        m0 = m0_new
    lse1 = m1 + tl.log(s1)            # log P(y=1) per candidate
    lse0 = m0 + tl.log(s0)
    py1 = tl.exp(lse1)

    # Pass 2: expected absolute change in source probability, numerators recomputed
    acc = tl.zeros([BLOCK_P], dtype=tl.float32)
    for start in range(0, V, BLOCK_V):
        v_offs = start + tl.arange(0, BLOCK_V)
        v_mask = v_offs < V
        num1, num0, e_mag, lsp = _tile(pc0, pc1, pc2, pc3, pc4, W, B, CS, PRM, LSP, V, v_offs, v_mask, lo, hi)
        src = tl.exp(lsp)
        term = ((1.0 - py1)[:, None] * tl.abs(src - tl.exp(num0 - lse0[:, None]))
                + py1[:, None] * tl.abs(src - tl.exp(num1 - lse1[:, None])))
        acc += tl.sum(tl.where(v_mask[None, :], term, 0.0), axis=1)

    obj = -acc / V
    viol = tl.maximum(emax - efield_limit, 0.0)
    fit = obj + penalty * viol * viol
    if FUSED_STRAIN:
        # 5-coil strain penalty (port of optimizer_constraints.py:19-37), [BLOCK_P]
        a0 = tl.abs(w0)
        a1 = tl.abs(w1)
        a2 = tl.abs(w2)
        a3 = tl.abs(w3)
        a4 = tl.abs(w4)
        coil5 = (tl.sqrt(0.9 * a0 * a0 + a1 * a1) * 0.2 + tl.sqrt(0.9 * a2 * a2 + a3 * a3)) * a4
        coil1 = (0.9 * a1 + 0.7 * tl.sqrt(a2 * a2 + 0.9 * a3 * a3) + 0.2 * a4) * a0
        sv = tl.maximum(tl.maximum(coil5, coil1) - strain_limit, 0.0)
        fit += penalty * sv * sv
    tl.store(obj_ptr + p_offs, obj, mask=p_mask)
    tl.store(fit_ptr + p_offs, fit, mask=p_mask)


def de_objective_triton(pop, k, BLOCK_P=4, BLOCK_V=128, num_warps=4, fused_strain=True):
    """Returns (fitness, -expected_change), both [P], like objective_torch."""
    assert pop.shape[1] == 5 and pop.is_contiguous() and pop.dtype == torch.float32
    P = pop.shape[0]
    fit = torch.empty(P, device=pop.device, dtype=torch.float32)
    obj = torch.empty_like(fit)
    grid = (triton.cdiv(P, BLOCK_P),)
    de_objective_kernel[grid](pop, k["W"], k["B"], k["CS"], k["PRM"], k["LSP"], fit, obj, P, k["V"],
                              k["lo"], k["hi"], k["efield_limit"], k["penalty"], k["strain_limit"],
                              BLOCK_P=BLOCK_P, BLOCK_V=BLOCK_V, FUSED_STRAIN=fused_strain, num_warps=num_warps)
    if not fused_strain:      # strain penalty in eager PyTorch: ~20 small kernels, each launched from the CPU
        fit = fit + k["penalty"] * torch.relu(strain_torch(pop) - k["strain_limit"]) ** 2
    return fit, obj


def kernel_info(pop, k, BLOCK_P, BLOCK_V, num_warps, fused_strain=True):
    """Launches the kernel once and returns what the compiled kernel reports about itself.
    Attribute names differ between Triton versions and backends, so missing ones give None."""
    P = pop.shape[0]
    fit = torch.empty(P, device=pop.device, dtype=torch.float32)
    obj = torch.empty_like(fit)
    ck = de_objective_kernel[(triton.cdiv(P, BLOCK_P),)](
        pop, k["W"], k["B"], k["CS"], k["PRM"], k["LSP"], fit, obj, P, k["V"],
        k["lo"], k["hi"], k["efield_limit"], k["penalty"], k["strain_limit"],
        BLOCK_P=BLOCK_P, BLOCK_V=BLOCK_V, FUSED_STRAIN=fused_strain, num_warps=num_warps)
    md = ck.metadata
    return {"n_regs": getattr(ck, "n_regs", None), "n_spills": getattr(ck, "n_spills", None),
            "num_warps": getattr(md, "num_warps", None), "warp_size": getattr(md, "warp_size", None),
            "shared_bytes": getattr(md, "shared", None), "target": str(getattr(md, "target", None))}
