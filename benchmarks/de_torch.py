# benchmarks/de_torch.py
# PyTorch-only pieces of the DE study (no JAX, no ABL), so they also run on ROCm:
# the PyTorch reference of the objective, the 5-coil strain, a comparison printer and
# the back-to-back timer (moved unchanged from de_reference.py and de_bench.py), and
# load_state() for the file written by export_de_state.py (new).
import json
import time
import numpy as np
import torch


def strain_torch(w):
    """Port of _calculate_strain (optimizer_constraints.py:19-37); w is [P, 5]."""
    a = w.abs()
    coil5 = (torch.sqrt(0.9 * a[:, 0]**2 + a[:, 1]**2) * 0.2
             + torch.sqrt(0.9 * a[:, 2]**2 + a[:, 3]**2)) * a[:, 4]
    coil1 = (0.9 * a[:, 1] + 0.7 * torch.sqrt(a[:, 2]**2 + 0.9 * a[:, 3]**2) + 0.2 * a[:, 4]) * a[:, 0]
    return torch.maximum(coil5, coil1)


def objective_torch(pop, lsp, p, c):
    """pop [P, C], lsp [V], p: dict of [V] parameters, c: constants. Returns (fitness, -expected_change), both [P]."""
    E = torch.einsum('cvd,pc->pvd', c["efield_norm"], pop)               # [P, V, 3] E-field per candidate
    e_mag = torch.linalg.norm(E, dim=-1)                                 # [P, V]
    e_t = torch.einsum('pvd,vdt->pvt', E, c["basis"])                    # [P, V, 2] projection to 2D subspace
    e_t = e_t / torch.linalg.norm(e_t, dim=-1, keepdim=True)
    nd = c["null_dir"]                                                   # [V, 2]
    ang = (torch.atan2(e_t[..., 1], e_t[..., 0]) - torch.atan2(nd[:, 1], nd[:, 0])) % (2 * torch.pi)

    trans = (p["rf_translate"] + p["rf_fourier1"] * torch.cos(ang) + p["rf_fourier2"] * torch.sin(ang)
             + p["rf_fourier3"] * torch.cos(2 * ang) + p["rf_fourier4"] * torch.sin(2 * ang))
    inner = torch.clamp(-p["rf_slope"] * (e_mag - trans), -16, 16)
    resp = torch.clamp(e_mag / (e_mag + torch.exp(inner)), c["lo"], c["hi"])

    log_r = torch.log(torch.clamp(resp, c["lo"], c["hi"]))               # safe_log
    log_1m = torch.log(-torch.expm1(log_r))                              # log(1 - r)
    num1 = lsp + log_r                                                   # [P, V] log-posterior numerators
    num0 = lsp + log_1m
    lse1 = torch.logsumexp(num1, dim=-1, keepdim=True)                   # [P, 1]
    lse0 = torch.logsumexp(num0, dim=-1, keepdim=True)
    post1, post0 = num1 - lse1, num0 - lse0

    py1 = torch.exp(lse1)                                                # P(y = 1) per candidate
    src = torch.exp(lsp)
    change = ((1 - py1) * (src - torch.exp(post0)).abs() + py1 * (src - torch.exp(post1)).abs()).mean(dim=-1)
    obj = -change

    pen = c["penalty"] * torch.relu(strain_torch(pop) - c["strain_limit"])**2
    pen = pen + c["penalty"] * torch.relu(e_mag.max(dim=-1).values - c["efield_limit"])**2
    return obj + pen, obj


def compare(name, ref, new):
    ref, new = ref.double(), new.double()
    d = (new - ref).abs()
    print(f"{name}: ref range [{ref.min().item():.6e}, {ref.max().item():.6e}], "
          f"max abs diff {d.max().item():.3e}, max rel diff {(d / ref.abs()).max().item():.3e}, "
          f"argmin equal: {ref.argmin().item() == new.argmin().item()} ({ref.argmin().item()} vs {new.argmin().item()})")


def load_state(path, device="cuda"):
    """Loads a fixed state written by export_de_state.py. Returns (pop, lsp, p, c, ref, meta):
    the inputs of objective_torch on `device`, the saved reference outputs (CPU tensors) and metadata."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    t = lambda name: torch.from_numpy(z[name]).to(device)
    c = {"efield_norm": t("efield_norm"), "basis": t("basis"), "null_dir": t("null_dir"), **meta["scalars"]}
    p = {k: t(f"p_{k}") for k in meta["param_keys"]}
    ref = {k[4:]: torch.from_numpy(z[k]) for k in z.files if k.startswith("out_")}
    return t("pop"), t("lsp"), p, c, ref, meta


def back_to_back_ms(fn, sync, calls=200, repeats=7):
    sync(fn())                                        # warm-up (and compilation)
    per_call = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(calls):
            out = fn()                                # queue calls without waiting
        sync(out)                                     # wait once, at the end
        per_call.append(1e3 * (time.perf_counter() - t0) / calls)
    return np.median(per_call), np.percentile(per_call, 25), np.percentile(per_call, 75)
