
import os
import subprocess
import torch
import jax
import jaxlib
import evosax
import numpyro
import importlib.metadata
import platform

def collect_env():
    my_dict = dict()
    my_dict["GIT branch"] = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    my_dict["GIT commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    cmd = ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"]
    output = subprocess.check_output(cmd, text=True).strip()
    model, driver = output.split(',')
    my_dict["GPU model and driver version"] = f"{model} | {driver}"
    
    my_dict["GPU memory:"] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.0f} GB"
    my_dict["PyTorch version"] = torch.__version__
    my_dict["CUDA version for PyTorch"] = torch.version.cuda
    my_dict["Is CUDA available"] = torch.cuda.is_available()
    my_dict["JAX version"] = jax.__version__
    my_dict["jaxlib version"] = jaxlib.__version__
    my_dict["JAX CUDA libs version"] = importlib.metadata.version("jax-cuda12-plugin")
    my_dict["JAX devices"] = jax.devices()
    my_dict["XLA_PYTHON_CLIENT_PREALLOCATE"] = os.getenv('XLA_PYTHON_CLIENT_PREALLOCATE')
    my_dict["EvoSax version"] = importlib.metadata.version("evosax")
    my_dict["NumPyro version"] = numpyro.__version__
    my_dict["Python version"] = platform.python_version()
    my_dict["OS:"] = platform.platform()

    return my_dict

if __name__ == "__main__":
    for key, value in collect_env().items():
        print(f"{key}: {value}<br>")
