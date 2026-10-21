import time, numpy as np, jax, jax.numpy as jnp
from shapes import make_session
from env_info import collect_env

for key, value in collect_env().items():
    print(f"{key}: {value}")

S = make_session()
for _ in range(3):  # a few trials so the model has realistic parameters
    S.start_trial(); d = S.choose_next_stimulus(); S.record_efield_trial(d)
    S.simulate_response(); S.localize(); S.complete_trial()

opt, rm = S.optimizer, S.response_model
lsp = jnp.asarray(rm.log_source_prob.cpu().numpy())
params = tuple(jnp.asarray(rm.param_dict[k]['values'].cpu().numpy()) for k in opt.param_keys)
pop = jax.random.uniform(jax.random.key(0), (512, opt.n_coils), minval=-1.0, maxval=1.0)
f = opt.jitted_objective

f(pop, lsp, params).block_until_ready()          # compile + warm-up
ts = []
for _ in range(200):
    t0 = time.perf_counter()
    f(pop, lsp, params).block_until_ready()      # JAX is async too: wait for the result
    ts.append(time.perf_counter() - t0)
print(f"objective: median {1e3*np.median(ts):.3f} ms, IQR {1e3*np.subtract(*np.percentile(ts, [75, 25])):.3f} ms")

hlo = f.lower(pop, lsp, params).compile().as_text()   # the program XLA actually runs
open("results/de_objective_hlo.txt", "w").write(hlo)
entry = hlo[hlo.index("ENTRY"):]
entry = entry[:entry.index("\n}")]
print("top-level kernels (fusion / custom-call):",
      sum((" fusion(" in l) or ("custom-call(" in l) for l in entry.splitlines()))

# Re-run the optimizer without syncs after each iteration
lsp = jnp.asarray(rm.log_source_prob.cpu().numpy())
params = tuple(jnp.asarray(rm.param_dict[k]['values'].cpu().numpy()) for k in opt.param_keys)
pop = jax.random.uniform(jax.random.key(0), (512, opt.n_coils), minval=-1.0, maxval=1.0)
f = opt.jitted_objective

f(pop, lsp, params).block_until_ready()
t0 = time.perf_counter()
for _ in range(200):
    out = f(pop, lsp, params)        # queue 200 calls without waiting
out.block_until_ready()              # wait once, at the end
print(f"objective, back-to-back: {1e3*(time.perf_counter()-t0)/200:.3f} ms per call")

S.end_session()