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
        return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(x), norm="ortho"))
    elif isinstance(x, torch.Tensor):
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x), norm="ortho"))
    elif isinstance(x, np.ndarray):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))
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
        return jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(x), norm="ortho"))
    elif isinstance(x, torch.Tensor):
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x), norm="ortho"))
    elif isinstance(x, np.ndarray):
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x), norm="ortho"))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")