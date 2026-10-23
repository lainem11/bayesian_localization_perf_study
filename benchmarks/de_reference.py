# benchmarks/de_reference.py
# PyTorch reference of the DE objective (src/ABL/optimizer.py, single_objective_fun),
# vectorized over the population, checked against the JAX version on a fixed state.
import numpy as np
import jax
import jax.numpy as jnp
import torch
from shapes import make_session
from env_info import collect_env
from ABL.optimizer import _create_objective_function


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


def fixed_state():
    """Session after 3 trials (as in profile_de.py), a fixed constrained population, and the
    objective inputs as JAX arrays (pop_j, lsp_j, params_j) and PyTorch tensors (pop_t, lsp_t, p_t, c)."""
    S = make_session()
    for _ in range(3):                          # same state as profile_de.py: 3 trials
        S.start_trial(); d = S.choose_next_stimulus(); S.record_efield_trial(d)
        S.simulate_response(); S.localize(); S.complete_trial()
    opt, rm, em = S.optimizer, S.response_model, S.efield_model

    # Fixed inputs, as JAX arrays
    lsp_j = jnp.asarray(rm.log_source_prob.cpu().numpy().astype(np.float32))
    params_j = tuple(jnp.asarray(rm.param_dict[k]['values'].detach().cpu().numpy().astype(np.float32)) for k in opt.param_keys)
    pop_j = jax.random.uniform(jax.random.key(0), (512, opt.n_coils), minval=-1.0, maxval=1.0)
    pop_j = opt.jitted_constraint_func(pop_j)   # scale to the strain limit, like the DE's initial population

    # PyTorch: same inputs
    dev = "cuda"
    to_t = lambda a: torch.from_numpy(np.asarray(a)).to(dev)
    c = {
        "efield_norm": em.efield_set.float().to(dev) * opt.max_current_slope.flatten().float().to(dev)[:, None, None],
        "basis": em.efield_basis.float().to(dev),
        "null_dir": em.null_dir_basis.squeeze().float().to(dev),
        "lo": float(np.float32(rm.prob_lim[0])), "hi": float(np.float32(rm.prob_lim[1])),
        "penalty": opt.optimizer_settings['constraint_penalty'],
        "strain_limit": opt.strain_limit, "efield_limit": float(opt.efield_limit),
    }
    p_t = {k: to_t(v) for k, v in zip(opt.param_keys, params_j)}
    return dict(S=S, opt=opt, rm=rm, em=em, pop_j=pop_j, lsp_j=lsp_j, params_j=params_j,
                pop_t=to_t(pop_j), lsp_t=to_t(lsp_j), p_t=p_t, c=c, to_t=to_t)


if __name__ == "__main__":
    for key, value in collect_env().items():
        print(f"{key}: {value}")

    st = fixed_state()
    S, opt, rm, em, to_t = st["S"], st["opt"], st["rm"], st["em"], st["to_t"]
    pop_j, lsp_j, params_j = st["pop_j"], st["lsp_j"], st["params_j"]

    # JAX: the session's objective, and one without penalties (expected change only)
    fit_j = opt.jitted_objective(pop_j, lsp_j, params_j)
    to_j = lambda t: jnp.asarray(t.detach().cpu().numpy().astype(np.float32))
    change_only_j = _create_objective_function(
        efield_basis_jax=to_j(em.efield_basis), null_dir_basis_jax=to_j(em.null_dir_basis.squeeze()),
        prob_lim_jax=jnp.asarray(rm.prob_lim, dtype=jnp.float32), efield_set_jax=to_j(em.efield_set),
        max_current_slope_jax=to_j(opt.max_current_slope.flatten()), efield_limit=1e30,
        strain_func_jax=None, strain_limit=None, constraint_penalty=opt.optimizer_settings['constraint_penalty'],
        param_keys=opt.param_keys)
    obj_j = change_only_j(pop_j, lsp_j, params_j)

    fit_t, obj_t = objective_torch(st["pop_t"], st["lsp_t"], st["p_t"], st["c"])

    print(f"\npopulation 512 x {opt.n_coils}, vertices {lsp_j.shape[0]}")
    print(f"candidates with strain penalty > 0 (torch): {(strain_torch(st['pop_t']) > st['c']['strain_limit'] + 1e-6).sum().item()}")
    compare("expected-change objective (no penalties)", to_t(obj_j), obj_t)
    compare("full fitness (with penalties)", to_t(fit_j), fit_t)
    S.end_session()