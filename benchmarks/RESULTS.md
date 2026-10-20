Testing done on WSL2. This might add some overhead to every launch and sync, compared to native Linux. 

## Environment info

GIT branch: perf-study<br>
GIT commit: f9289aa5b1cc61376b8ea92fe22d03287a3ba94e<br>
GPU model and driver version: NVIDIA GeForce RTX 4060 Ti |  610.88<br>
GPU memory:: 16 GB<br>
PyTorch version: 2.12.0+cu126<br>
CUDA version for PyTorch: 12.6<br>
Is CUDA available: True<br>
JAX version: 0.7.2<br>
jaxlib version: 0.7.2<br>
JAX CUDA libs version: 0.7.2<br>
JAX devices: [CudaDevice(id=0)]<br>
XLA_PYTHON_CLIENT_PREALLOCATE: None<br>
EvoSax version: 0.2.0<br>
NumPyro version: 0.20.1<br>
Python version: 3.13.16<br>
OS:: Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.43<br>

## E-field tensor info

efield_set; Shape: torch.Size([5, 7444, 3]), dtype: torch.float32, device: cuda:0, size: 0.43 MB.<br>
efield_basis; Shape: torch.Size([7444, 3, 2]), dtype: torch.float32, device: cuda:0.<br>
efield_null_dir_basis; Shape: torch.Size([1, 7444, 2]), dtype: torch.float32, device: cuda:0.<br>
efield_trials; Shape: torch.Size([150, 7444, 3]), dtype: torch.float32, device: cuda:0.<br>
efield_basis; Shape: torch.Size([7444, 3, 2]), dtype: torch.float32, device: cuda:0.<br>
ROI length: 7444.<br>
Number of coils: 5.<br>