import torch
import jax

# PyTorch device
DEVICE_TORCH = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else torch.device("cpu")
)

# JAX device 
try:
    DEVICE_JAX = jax.devices("gpu")[0]
except RuntimeError:
    DEVICE_JAX = jax.devices("cpu")[0]