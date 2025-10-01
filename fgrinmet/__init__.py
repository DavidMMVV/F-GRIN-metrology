from jax import config

from . import geopt
from . import splitm
from . import constructions
from . import utils
from .globvar import DEVICE_TORCH


# enable float64 in JAX
config.update("jax_enable_x64", True)

__all__ = ["geopt", "splitm", "constructions", "utils"]