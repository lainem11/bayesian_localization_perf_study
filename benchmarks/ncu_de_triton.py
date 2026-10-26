# benchmarks/ncu_de_triton.py
# Runs the fused DE kernel a few times in one configuration, for Nsight Compute.
# Usage: python ncu_de_triton.py BLOCK_P BLOCK_V
import sys
import torch
import triton
from env_info import collect_env
from de_reference import fixed_state
from de_triton import prepare, de_objective_triton

bp, bv = int(sys.argv[1]), int(sys.argv[2])
for key, value in collect_env().items():
    print(f"{key}: {value}")
print(f"Triton version: {triton.__version__}")

st = fixed_state()
k = prepare(st["c"], st["p_t"], st["lsp_t"])
for _ in range(3):                                    # call 1 compiles; ncu profiles call 3
    de_objective_triton(st["pop_t"], k, bp, bv, 4, fused_strain=True)
torch.cuda.synchronize()
print(f"config: BLOCK_P {bp}, BLOCK_V {bv}, 4 warps, strain in kernel")
st["S"].end_session()