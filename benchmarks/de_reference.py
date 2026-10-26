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
from de_torch import strain_torch, objective_torch, compare   # moved to de_torch.py (no JAX)


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


def change_only_objective(st):
    """The JAX objective built from src's _create_objective_function with both penalties off
    (strain function None, E-field limit 1e30): the expected change alone."""
    opt, rm, em = st["opt"], st["rm"], st["em"]
    to_j = lambda t: jnp.asarray(t.detach().cpu().numpy().astype(np.float32))
    return _create_objective_function(
        efield_basis_jax=to_j(em.efield_basis), null_dir_basis_jax=to_j(em.null_dir_basis.squeeze()),
        prob_lim_jax=jnp.asarray(rm.prob_lim, dtype=jnp.float32), efield_set_jax=to_j(em.efield_set),
        max_current_slope_jax=to_j(opt.max_current_slope.flatten()), efield_limit=1e30,
        strain_func_jax=None, strain_limit=None, constraint_penalty=opt.optimizer_settings['constraint_penalty'],
        param_keys=opt.param_keys)


if __name__ == "__main__":
    for key, value in collect_env().items():
        print(f"{key}: {value}")

    st = fixed_state()
    S, opt, rm, em, to_t = st["S"], st["opt"], st["rm"], st["em"], st["to_t"]
    pop_j, lsp_j, params_j = st["pop_j"], st["lsp_j"], st["params_j"]

    # JAX: the session's objective, and one without penalties (expected change only)
    fit_j = opt.jitted_objective(pop_j, lsp_j, params_j)
    obj_j = change_only_objective(st)(pop_j, lsp_j, params_j)

    fit_t, obj_t = objective_torch(st["pop_t"], st["lsp_t"], st["p_t"], st["c"])

    print(f"\npopulation 512 x {opt.n_coils}, vertices {lsp_j.shape[0]}")
    print(f"candidates with strain penalty > 0 (torch): {(strain_torch(st['pop_t']) > st['c']['strain_limit'] + 1e-6).sum().item()}")
    compare("expected-change objective (no penalties)", to_t(obj_j), obj_t)
    compare("full fitness (with penalties)", to_t(fit_j), fit_t)
    S.end_session()