from jax import config, default_device

from . import splitm
from . import constructions
from . import utils
from .globvar import DEVICE_TORCH, DEVICE_JAX


# enable float64 in JAX
config.update("jax_enable_x64", True)
default_device(DEVICE_JAX)

__all__ = ["splitm", "constructions", "utils", "DEVICE_TORCH", "DEVICE_JAX"]