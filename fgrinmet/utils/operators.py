import numpy as np
import torch
import jax.numpy as jnp

from typing import overload

@overload
def FT2(x: jnp.ndarray) -> jnp.ndarray: ...
@overload
def FT2(x: np.ndarray) -> np.ndarray: ...
@overload
def FT2(x: torch.Tensor) -> torch.Tensor: ...
def FT2(x):
    if isinstance(x, jnp.ndarray):
        return jnp.fft.fftshift(jnp.fft.fft2(x))
    elif isinstance(x, torch.Tensor):
        return torch.fft.fftshift(torch.fft.fft2(x))
    elif isinstance(x, np.ndarray):
        return np.fft.fftshift(np.fft.fft2(x))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
    
@overload
def iFT2(x: jnp.ndarray) -> jnp.ndarray: ...
@overload
def iFT2(x: np.ndarray) -> np.ndarray: ...
@overload
def iFT2(x: torch.Tensor) -> torch.Tensor: ...
def iFT2(x):
    if isinstance(x, jnp.ndarray):
        return jnp.fft.ifft2(jnp.fft.ifftshift(x))
    elif isinstance(x, torch.Tensor):
        return torch.fft.ifft2(torch.fft.ifftshift(x))
    elif isinstance(x, np.ndarray):
        return np.fft.ifft2(np.fft.ifftshift(x))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
    