import torch
import jax.numpy as jnp
import jax

from typing import List, Tuple, Union

from fgrinmet.globvar import DEVICE_TORCH

def coord_jax(
        shape: List[int] | Tuple[int,...],
        pix_size: float | list[float] | Tuple[float,...],
) -> tuple[jnp.ndarray, ...]:
    """Generates the grid with coordinates for real space as a jax array.

    Args:
        shape (List[int] | Tuple[int,...]): Shape of the grid.
        pix_size (float | list[float] | Tuple[float,...]): Pixel sizes over each dimension. If 
        it is a float, takes square pixels

    Returns:
        Xi (tuple[jnp.ndarray, ...]): Tuple of each coordinate of the grid
    """

    xi = [(jnp.arange(int(shape[i]), dtype=jnp.float64) - shape[i] / 2) *
                                 (pix_size[i] if isinstance(pix_size, list | tuple) else pix_size)
                                 for i in range(len(shape))]
    Xi = jnp.meshgrid(*xi, indexing='ij')
    return tuple(Xi)

def fft_coord_jax(
        shape: List[int] | Tuple[int,...],
        pix_size: float | list[float] | Tuple[float,...],
) -> tuple[jnp.ndarray, ...]:
    """Generates the grid with coordinates for fourier space as a jax array.

    Args:
        shape (List[int] | Tuple[int,...]): Shape of the grid.
        pix_size (float | list[float] | Tuple[float,...]): Pixel sizes over each dimension. If 
        it is a float, takes square pixels

    Returns:
        Fi (tuple[jnp.ndarray, ...]): Tuple of each coordinate of the grid.
    """
    fi = [jnp.fft.fftshift(jnp.fft.fftfreq(shape[i], 
                                               d=(pix_size[i] if isinstance(pix_size, list | tuple) else pix_size), 
                                               dtype=jnp.float64)) 
                                               for i in range(len(shape))]
    Fi = jnp.meshgrid(*fi, indexing='ij')
    return tuple(Fi)

def coord_pytorch(
        shape: List[int] | Tuple[int,...],
        pix_size: float | list[float] | Tuple[float,...],
        dtype: torch.dtype = torch.float64,
        device: torch.device = DEVICE_TORCH,
) -> tuple[torch.Tensor, ...]:
    """Generates the grid with coordinates for real space as a pytorch tensor.

    Args:
        shape (List[int] | Tuple[int,...]): Shape of the grid.
        pix_size (float | list[float] | Tuple[float,...]): Pixel sizes over each dimension. If 
        it is a float, takes square pixels
        dtype (torch.dtype, optional): Data type of the output tensors. Defaults to torch.float64.
        device (torch.device, optional): Device in which the tensors are stored. Defaults to DEVICE.

    Returns:
        Xi (tuple[torch.Tensor, ...]): Tuple of each coordinate of the grid
    """

    xi = [(torch.arange(int(shape[i]), dtype=dtype, device=device) - shape[i] / 2) *
                                 (pix_size[i] if isinstance(pix_size, list | tuple) else pix_size)
                                 for i in range(len(shape))]
    Xi = torch.meshgrid(*xi, indexing='ij')
    return Xi


def fft_coord_pytorch(
        shape: List[int] | Tuple[int,...],
        pix_size: float | list[float] | Tuple[float,...],
        dtype: torch.dtype = torch.float64,
        device: torch.device = DEVICE_TORCH,
) -> tuple[torch.Tensor, ...]:
    """Generates the grid with coordinates for fourier space as a pytorch tensor.

    Args:
        shape (List[int] | Tuple[int,...]): Shape of the grid.
        pix_size (float | list[float] | Tuple[float,...]): Pixel sizes over each dimension. If 
        it is a float, takes square pixels
        dtype (torch.dtype, optional): Data type of the output tensors. Defaults to torch.float64.
        device (torch.device, optional): Device in which the tensors are stored. Defaults to DEVICE.

    Returns:
        Fi (tuple[torch.Tensor, ...]): Tuple of each coordinate of the grid.
    """
    fi = [torch.fft.fftshift(torch.fft.fftfreq(shape[i], 
                                               d=(pix_size[i] if isinstance(pix_size, list | tuple) else pix_size), 
                                               dtype=dtype).to(device)) 
                                               for i in range(len(shape))]
    Fi = torch.meshgrid(*fi, indexing='ij')
    return Fi

if __name__ == "__main__":
    shape = (5,5,5)
    pix_size = (1.0, 1.0, 1.0)
    Z,Y,X = coord_jax(shape, pix_size)
    print("Z:", Z)
    print("Y:", Y)
    print("X:", X)

    Fz,Fy,Fx = fft_coord_jax(shape, pix_size)
    print("Fz:", Fz)
    print("Fy:", Fy)
    print("Fx:", Fx)